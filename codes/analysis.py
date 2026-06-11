"""
analyze.py — TUNES Stage 2: Preprocessed .fif → Features + Visualisations

What this script does
---------------------
Reads the preprocessed .fif files written by preprocess.py and runs the
full analysis pipeline

  1.  Sleep staging (YASA, 100 Hz 3-channel copy; cached to CSV)
  2.  Individual spindle frequency (adaptation session)
  3.  Spindle detection (YASA)
  4.  Slow-wave detection (YASA)
  5.  Spectral band power
  6.  Burst-level (pulse-level) analysis + MNE Epochs .fif
  7.  Visualisations: raw-vs-preprocessed, spectrogram, topoplots,
      boxplots, violins, ERPs, TFRs
  8.  Response neural profile (build_response_profile / run_response_statistics
      / plot_response_profile) — runs automatically once all participants
      have been processed.

"""

import argparse
import gc
import re
import time
import traceback
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import mne
import numpy as np
import pandas as pd
import yasa
from scipy.signal import welch
from scipy.signal import spectrogram as scipy_spectrogram
from scipy import stats as sp_stats
from scipy.signal import butter, filtfilt

mne.set_log_level('WARNING')


# =============================================================================
# Settings — edit these paths
# =============================================================================
PREPROCESSED_DIR = '/Users/folasewaabdulsalam/Downloads/TUNES/preprocessed'
DATA_ROOT      = '/Users/folasewaabdulsalam/Downloads/TUNES/subjects'
OUTPUT_DIR     = '/Users/folasewaabdulsalam/Downloads/TUNES/results'
PARTICIPANTS   = ['03', '06', '08']

RESAMPLE_FREQ  = 500
STAGING_FREQ   = 100
RUN_SLEEP_STAGING = True

BANDPASS_LOW   = 0.1    
BANDPASS_HIGH  = 40.0

SPINDLE_CHANNELS = ['C3', 'C4']
SW_CHANNELS      = ['F3', 'F4']
POWER_CHANNELS   = ['C3', 'C4', 'F3', 'F4', 'Fz', 'Cz', 'Pz']
STAGING_EEG_CH   = 'C4'
STAGING_EOG_CH   = 'HEOG'
STAGING_EMG_CH   = 'EMG'
VIZ_CHANNELS     = ['Fp1', 'Fp2', 'F3', 'F4', 'Fz',
                     'C3',  'C4',  'Cz',
                     'P3',  'P4',  'Pz',
                     'O1',  'O2']

NREM_STAGES      = [2, 3]
SPINDLE_FREQ_DEFAULT = (12.0, 15.0)
SPINDLE_FREQ_MARGIN  = 1.5
SW_FREQ          = (0.5, 4.0)
FREQ_BANDS       = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'sigma': (12.0, 15.0),
}

TUS_MARKER_CODE    = 'A'
INTENSITY_COMMENTS = {
    '60w':             'active_60w',
    '1w':              'sham_1isppa',
    '30w':             'ignore',
    '10w':             'ignore',
    'no stim':         'ignore',
    'stim':            'ignore',
    'transducer away': 'ignore',
    'reprep':          'ignore',
}
ACTIVE_CONDITIONS = {'active_60w'}
SHAM_CONDITIONS   = {'sham_1isppa'}
TUS_EPOCH_PRE_SEC  = 3.0
TUS_EPOCH_POST_SEC = 5.0

KNOWN_TARGETS = {'adapt', 'thalamus', 'ventricle', 'ventricles'}


# =============================================================================
# Preprocessed file helpers
# =============================================================================

def fif_path(participant_id, target):
    """Return path to preprocessed .fif for a given target."""
    return Path(PREPROCESSED_DIR) / participant_id / f'{target}_raw.fif'


def snapshot_path(participant_id, target):
    return Path(PREPROCESSED_DIR) / participant_id / f'{target}_raw_snapshot.npy'


def snapshot_channels_path(participant_id, target):
    return Path(PREPROCESSED_DIR) / participant_id / f'{target}_snapshot_channels.npy'


def load_preprocessed(participant_id, target):
    """
    Load a preprocessed .fif as a lazy Raw object (preload=False).
    Data is only pulled into RAM when you call get_data() or load_data().
    """
    p = fif_path(participant_id, target)
    if not p.exists():
        raise FileNotFoundError(
            f'Preprocessed file not found: {p}\n'
            f'Run preprocess.py first.'
        )
    raw = mne.io.read_raw_fif(str(p), preload=False, verbose=False)
    print(f'    Loaded (lazy) {p.name}  '
          f'[{len(raw.ch_names)} ch | {raw.info["sfreq"]:.0f} Hz | '
          f'{raw.times[-1]/60:.1f} min]')
    return raw


def set_channel_types(raw):
    mapping = {}
    for ch in raw.ch_names:
        if ch in ['HEOG', 'VEOG']:
            mapping[ch] = 'eog'
        elif ch in ['EMG', 'APBr', 'FDIr', 'ADMr']:
            mapping[ch] = 'emg'
    if mapping:
        raw.set_channel_types(mapping, on_unit_change='ignore')
    return raw


def find_vmrk(participant_id, target):
    """Locate the original .vmrk file (stays in raw subjects folder)."""
    subject_folder = Path(DATA_ROOT) / participant_id
    # Walk all subdirectories, match on target keyword
    for folder in sorted(subject_folder.iterdir()):
        if not folder.is_dir():
            continue
        text = folder.name.lower()
        if target.lower().replace('ventricle', 'vent') not in text and \
           target.lower() not in text:
            continue
        vmrk_files = sorted(folder.glob('*.vmrk'))
        if vmrk_files:
            return str(vmrk_files[0])
    return None


def find_vhdr_for_staging(participant_id, target):
    """Return list of .vhdr paths for the given session (needed for staging)."""
    subject_folder = Path(DATA_ROOT) / participant_id
    for folder in sorted(subject_folder.iterdir()):
        if not folder.is_dir():
            continue
        text = folder.name.lower()
        match = (target == 'adapt' and 'adapt' in text) or \
                (target == 'thalamus' and 'thalamus' in text) or \
                (target == 'ventricle' and ('ventricle' in text or 'ventricles' in text))
        if not match:
            continue
        files = sorted(folder.glob('*.vhdr'))
        if files:
            return [str(f) for f in files], folder
    return [], None


# =============================================================================
# Sleep staging 
# =============================================================================

def load_staging_only(vhdr_files):
    """Load a minimal 3-channel, 100 Hz copy from the raw .vhdr files."""
    staging_channels = [STAGING_EEG_CH, STAGING_EOG_CH, STAGING_EMG_CH]
    raws = []
    for vhdr in vhdr_files:
        r = mne.io.read_raw_brainvision(vhdr, preload=False, verbose=False)
        available = [ch for ch in staging_channels if ch in r.ch_names]
        r.pick_channels(available)
        r.resample(STAGING_FREQ)
        r.load_data()
        raws.append(r)
    raw_staging = raws[0] if len(raws) == 1 else mne.concatenate_raws(raws)
    del raws
    set_channel_types(raw_staging)
    mb = raw_staging.get_data().nbytes / 1e6
    print(f'    Staging raw: {raw_staging.ch_names} | '
          f'{raw_staging.info["sfreq"]:.0f} Hz | '
          f'{raw_staging.times[-1]/60:.1f} min | {mb:.1f} MB')
    return raw_staging


def run_sleep_staging(vhdr_files, session_name, participant_id, output_dir):
    print(f'\n[2] Sleep staging: {participant_id} / {session_name}')
    if not RUN_SLEEP_STAGING:
        print('    Skipped (RUN_SLEEP_STAGING=False)')
        return None, None, False

    hypno_path = Path(output_dir) / f'{participant_id}_{session_name}_hypnogram.csv'
    if hypno_path.exists():
        print('    Loading cached hypnogram')
        hypno_str = pd.read_csv(hypno_path)['stage'].tolist()
        hypno_int = np.asarray(yasa.Hypnogram(hypno_str).as_int(), dtype=int)
        return hypno_int, hypno_str, True

    if not vhdr_files:
        print('    No .vhdr files available for staging')
        return None, None, False

    peek = mne.io.read_raw_brainvision(vhdr_files[0], preload=False, verbose=False)
    if STAGING_EEG_CH not in peek.ch_names:
        print(f'    Missing staging channel: {STAGING_EEG_CH}')
        return None, None, False
    del peek

    raw_staging = load_staging_only(vhdr_files)
    kwargs = {'eeg_name': STAGING_EEG_CH}
    if STAGING_EOG_CH in raw_staging.ch_names:
        kwargs['eog_name'] = STAGING_EOG_CH
    if STAGING_EMG_CH in raw_staging.ch_names:
        kwargs['emg_name'] = STAGING_EMG_CH

    try:
        gc.collect()
        print('    Running YASA …')
        t0  = time.time()
        sls = yasa.SleepStaging(raw_staging, **kwargs)
        hypno_pred = sls.predict()
        del sls, raw_staging
        gc.collect()
        print(f'    Done in {time.time()-t0:.1f} s')

        hypno_str = list(hypno_pred.hypno)
        hypno_int = np.asarray(list(hypno_pred.as_int()), dtype=int)
    except Exception as exc:
        print(f'    Staging failed: {exc}')
        return None, None, False

    print(f'    Staged {len(hypno_int)} epochs | {dict(Counter(hypno_str))}')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'epoch': range(len(hypno_str)), 'stage': hypno_str}).to_csv(
        hypno_path, index=False
    )
    fig, ax = plt.subplots(figsize=(12, 3))
    try:
        yasa.plot_hypnogram(hypno_pred, ax=ax)
    except Exception:
        yasa.plot_hypnogram(hypno_int, ax=ax)
    ax.set_title(f'{participant_id} - {session_name} - Hypnogram')
    fig.savefig(
        Path(output_dir) / f'{participant_id}_{session_name}_hypnogram.png',
        dpi=100, bbox_inches='tight'
    )
    plt.close(fig)
    return hypno_int, hypno_str, True


# =============================================================================
# Analysis helpers
# =============================================================================

def nrem_mask_from_hypno(hypno_int, raw):
    if hypno_int is None:
        return np.ones(raw.n_times, dtype=bool)
    mask = np.zeros(raw.n_times, dtype=bool)
    spe  = int(raw.info['sfreq'] * 30)
    for epoch, stage in enumerate(hypno_int):
        if stage in NREM_STAGES:
            s = epoch * spe
            mask[s:min(s + spe, raw.n_times)] = True
    return mask


def upsample_hypno(hypno_int, raw):
    spe     = int(raw.info['sfreq'] * 30)
    hyp_up  = np.repeat(hypno_int, spe)
    if len(hyp_up) > raw.n_times:
        hyp_up = hyp_up[:raw.n_times]
    elif len(hyp_up) < raw.n_times:
        hyp_up = np.pad(hyp_up, (0, raw.n_times - len(hyp_up)), mode='edge')
    return hyp_up


def _nrem_minutes(hypno_int, raw):
    if hypno_int is not None:
        return max(float(np.sum(np.isin(hypno_int, NREM_STAGES)) * 0.5), 1.0)
    return max(raw.times[-1] / 60, 1.0)


def get_individual_spindle_frequency(raw, hypno_int, session_name,
                                     participant_id, output_dir):
    print(f'\n[4] Individual spindle freq: {participant_id} / {session_name}')
    if STAGING_EEG_CH not in raw.ch_names:
        return np.mean(SPINDLE_FREQ_DEFAULT), SPINDLE_FREQ_DEFAULT

    sfreq = raw.info['sfreq']
    data  = raw.get_data(picks=[STAGING_EEG_CH])[0][nrem_mask_from_hypno(hypno_int, raw)]
    if len(data) < sfreq * 60:
        print('    Not enough NREM; using default spindle band')
        return np.mean(SPINDLE_FREQ_DEFAULT), SPINDLE_FREQ_DEFAULT

    freqs, psd = welch(data, fs=sfreq, nperseg=int(sfreq * 4), noverlap=int(sfreq * 2))
    del data
    band = (freqs >= SPINDLE_FREQ_DEFAULT[0]) & (freqs <= SPINDLE_FREQ_DEFAULT[1])
    if not band.any():
        return np.mean(SPINDLE_FREQ_DEFAULT), SPINDLE_FREQ_DEFAULT

    peak      = round(float(freqs[band][np.argmax(psd[band])]), 2)
    freq_band = (
        round(max(peak - SPINDLE_FREQ_MARGIN, 10.0), 1),
        round(min(peak + SPINDLE_FREQ_MARGIN, 16.0), 1),
    )
    print(f'    Peak {peak} Hz | band {freq_band}')
    pd.DataFrame([{
        'participant_id': participant_id, 'session': session_name,
        'peak_freq_hz': peak, 'band_low_hz': freq_band[0], 'band_high_hz': freq_band[1],
    }]).to_csv(Path(output_dir) / f'{participant_id}_individual_spindle_freq.csv', index=False)
    return peak, freq_band


def load_individual_spindle_frequency(participant_id, output_dir):
    path = Path(output_dir) / f'{participant_id}_individual_spindle_freq.csv'
    if not path.exists():
        return np.mean(SPINDLE_FREQ_DEFAULT), SPINDLE_FREQ_DEFAULT
    row = pd.read_csv(path).iloc[0]
    return float(row['peak_freq_hz']), (float(row['band_low_hz']), float(row['band_high_hz']))


def detect_spindles(raw, hypno_int, hypno_up, freq_band,
                    session_name, participant_id, output_dir):
    print(f'\n[5] Spindles: {participant_id} / {session_name}')
    channels = [ch for ch in SPINDLE_CHANNELS if ch in raw.ch_names]
    if not channels:
        return {}
    obj = yasa.spindles_detect(raw, ch_names=channels, freq_sp=freq_band,
                               hypno=hypno_up, include=NREM_STAGES)
    if obj is None:
        print('    No spindles detected')
        return {}
    summary = obj.summary()
    summary.to_csv(
        Path(output_dir) / f'{participant_id}_{session_name}_spindles.csv', index=False
    )
    nrem_min = _nrem_minutes(hypno_int, raw)
    features = {}
    for ch in channels:
        d = summary[summary['Channel'] == ch]
        features[ch] = {
            'spindle_density_per_min': round(len(d) / nrem_min, 3),
            'spindle_amplitude_uv':    round(d['Amplitude'].mean(), 3) if len(d) else np.nan,
            'spindle_frequency_hz':    round(d['Frequency'].mean(), 3) if len(d) else np.nan,
            'spindle_duration_sec':    round(d['Duration'].mean(), 3)  if len(d) else np.nan,
            'spindle_rms_uv':          round(d['RMS'].mean(), 3)       if len(d) else np.nan,
            'n_spindles':              int(len(d)),
        }
    print(f'    Detected {len(summary)} spindles')
    return features


def detect_slow_waves(raw, hypno_int, hypno_up, session_name, participant_id, output_dir):
    print(f'\n[6] Slow waves: {participant_id} / {session_name}')
    channels = [ch for ch in SW_CHANNELS if ch in raw.ch_names]
    if not channels:
        return {}
    obj = yasa.sw_detect(raw, ch_names=channels, freq_sw=SW_FREQ,
                         hypno=hypno_up, include=NREM_STAGES)
    if obj is None:
        print('    No slow waves detected')
        return {}
    summary = obj.summary()
    summary.to_csv(
        Path(output_dir) / f'{participant_id}_{session_name}_slowwaves.csv', index=False
    )
    nrem_min = _nrem_minutes(hypno_int, raw)
    features = {}
    for ch in channels:
        d = summary[summary['Channel'] == ch]
        features[ch] = {
            'sw_density_per_min':  round(len(d) / nrem_min, 3),
            'sw_amplitude_uv':     round(d['PTP'].mean(), 3)       if len(d) else np.nan,
            'sw_negative_peak_uv': round(d['NegPeak'].mean(), 3)   if len(d) else np.nan,
            'sw_positive_peak_uv': round(d['PosPeak'].mean(), 3)   if len(d) else np.nan,
            'sw_slope_uvs':        round(d['Slope'].mean(), 3)     if len(d) and 'Slope' in d else np.nan,
            'sw_duration_sec':     round(d['Duration'].mean(), 3)  if len(d) and 'Duration' in d else np.nan,
            'n_slow_waves':        int(len(d)),
        }
    print(f'    Detected {len(summary)} slow waves')
    return features


def compute_spectral_power(raw, hypno_int, session_name, participant_id, output_dir):
    print(f'\n[7] Spectral power: {participant_id} / {session_name}')
    mask  = nrem_mask_from_hypno(hypno_int, raw)
    sfreq = raw.info['sfreq']
    features, rows = {}, []
    channels = [ch for ch in (POWER_CHANNELS or raw.ch_names) if ch in raw.ch_names]
    for ch in channels:
        data = raw.get_data(picks=[ch])[0][mask]
        if len(data) < sfreq * 30:
            del data
            continue
        freqs, psd = welch(data, fs=sfreq, nperseg=int(sfreq*4), noverlap=int(sfreq*2))
        del data
        row = {'participant_id': participant_id, 'session': session_name, 'channel': ch}
        for band_name, (low, high) in FREQ_BANDS.items():
            band  = (freqs >= low) & (freqs <= high)
            value = float(psd[band].mean()) if band.any() else np.nan
            row[f'{band_name}_power_v2hz']  = value
            row[f'{band_name}_power_uv2hz'] = value * 1e12 if not np.isnan(value) else np.nan
            features[f'{ch}_{band_name}_power_uv2hz'] = row[f'{band_name}_power_uv2hz']
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(
            Path(output_dir) / f'{participant_id}_{session_name}_spectral_power.csv',
            index=False
        )
        _plot_spectral_power(df, session_name, participant_id, output_dir)
    return features


def _plot_spectral_power(power_df, session_name, participant_id, output_dir):
    plot_df = power_df[
        ['channel'] + [f'{b}_power_uv2hz' for b in FREQ_BANDS]
    ].set_index('channel')
    fig, ax = plt.subplots(figsize=(12, max(4, len(plot_df) * 0.18)))
    im = ax.imshow(np.log10(plot_df.replace(0, np.nan).values),
                   aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels([c.replace('_power_uv2hz', '') for c in plot_df.columns])
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)
    ax.set_title(f'{participant_id} - {session_name} spectral power')
    fig.colorbar(im, ax=ax, label='log10 power (uV²/Hz)')
    fig.tight_layout()
    fig.savefig(
        Path(output_dir) / f'{participant_id}_{session_name}_spectral_power.png', dpi=150
    )
    plt.close(fig)


# =============================================================================
# Marker / burst parsing  
# =============================================================================

def read_original_sfreq(vhdr_path):
    with open(vhdr_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('SamplingInterval='):
                return 1_000_000.0 / float(line.split('=')[1])
    return 5000.0


def parse_tus_markers_bursts(vmrk_path, original_sfreq, burst_gap_threshold=0.5):
    pulses            = []
    current_condition = 'unknown'
    with open(vmrk_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.startswith('Mk'):
                continue
            _, fields_str = line.strip().split('=', 1)
            fields = fields_str.split(',')
            if len(fields) < 3:
                continue
            marker_type = fields[0]
            label       = fields[1].strip()
            sample      = int(fields[2])
            if marker_type == 'Comment':
                text = label.lower()
                for keyword, condition in INTENSITY_COMMENTS.items():
                    if keyword in text:
                        current_condition = condition
                        break
            elif marker_type == 'Stimulus' and label == TUS_MARKER_CODE:
                pulses.append({
                    'sample_original': sample,
                    'time_sec':        sample / original_sfreq,
                    'condition':       current_condition,
                })
    if not pulses:
        return pd.DataFrame()
    df = pd.DataFrame(pulses)
    df = df[df['condition'] != 'unknown'].reset_index(drop=True)
    df = df.sort_values('sample_original').reset_index(drop=True)
    df['trigger_seq_all']       = np.arange(1, len(df) + 1)
    df['trigger_seq_condition'] = df.groupby('condition').cumcount() + 1
    df['gap_sec']  = df['time_sec'].diff()
    df['burst_id'] = (df['gap_sec'].isna() | (df['gap_sec'] > burst_gap_threshold)).cumsum()
    burst_rows = []
    for burst_id, group in df.groupby('burst_id'):
        group = group.sort_values('sample_original')
        burst_rows.append({
            'burst_id':                    int(burst_id),
            'sample_original':             int(group['sample_original'].iloc[0]),
            'time_sec':                    float(group['time_sec'].iloc[0]),
            'condition':                   group['condition'].iloc[0],
            'n_pulses':                    len(group),
            'duration_sec':                float(group['time_sec'].iloc[-1] - group['time_sec'].iloc[0]),
            'first_trigger_seq_all':       int(group['trigger_seq_all'].iloc[0]),
            'last_trigger_seq_all':        int(group['trigger_seq_all'].iloc[-1]),
            'first_trigger_seq_condition': int(group['trigger_seq_condition'].iloc[0]),
            'last_trigger_seq_condition':  int(group['trigger_seq_condition'].iloc[-1]),
        })
    bursts = pd.DataFrame(burst_rows)
    bursts.insert(0, 'burst_seq_all', np.arange(1, len(bursts) + 1))
    print(f'    Bursts: {bursts["condition"].value_counts().to_dict()}')
    return bursts


def band_power(data, sfreq, low, high):
    if len(data) < sfreq:
        return np.nan
    freqs, psd = welch(data, fs=sfreq, nperseg=min(int(sfreq * 2), len(data)))
    band = (freqs >= low) & (freqs <= high)
    return float(psd[band].mean()) if band.any() else np.nan


def compute_window_features(window_data, ch_names, sfreq, freq_band):
    pre_samples = int(TUS_EPOCH_PRE_SEC * sfreq)
    features    = {}
    relevant_chs = [ch for ch in SPINDLE_CHANNELS + SW_CHANNELS if ch in ch_names]
    for ch in relevant_chs:
        data = window_data[ch_names.index(ch)] * 1e6
        pre  = data[:pre_samples]
        post = data[pre_samples:]
        pre_sigma  = band_power(pre,  sfreq, *freq_band)
        post_sigma = band_power(post, sfreq, *freq_band)
        features[f'{ch}_pre_sigma_power']    = round(pre_sigma,  6)
        features[f'{ch}_post_sigma_power']   = round(post_sigma, 6)
        features[f'{ch}_pre_delta_power']    = round(band_power(pre,  sfreq, *SW_FREQ), 6)
        features[f'{ch}_post_delta_power']   = round(band_power(post, sfreq, *SW_FREQ), 6)
        features[f'{ch}_sigma_power_change'] = (
            round((post_sigma - pre_sigma) / pre_sigma, 4)
            if pre_sigma and pre_sigma > 0 else np.nan
        )
        features[f'{ch}_post_ptp_uv'] = round(float(np.ptp(post)), 3) if len(post) else np.nan
    return features


def compute_event_locked_spindle_features(burst_times_by_group, spindle_summary,
                                           post_window_sec=5.0):
    if spindle_summary is None or spindle_summary.empty:
        return {}
    out = {}
    spindle_starts = spindle_summary['Start'].values
    for group, burst_times in burst_times_by_group.items():
        if not burst_times:
            continue
        burst_times = np.asarray(burst_times)
        n_bursts    = len(burst_times)
        sp = spindle_starts[np.newaxis, :]
        bt = burst_times[:, np.newaxis]
        in_window        = (sp > bt) & (sp <= bt + post_window_sec)
        spindle_counts   = in_window.sum(axis=1)
        bursts_with_sp   = int((spindle_counts > 0).sum())
        total_sp         = int(spindle_counts.sum())
        sp_rate_per_burst = float(spindle_counts.mean())
        matched_idx = np.where(in_window.any(axis=0))[0]
        matched     = spindle_summary.iloc[matched_idx]
        amp  = float(matched['Amplitude'].mean()) if len(matched) else np.nan
        freq = float(matched['Frequency'].mean()) if len(matched) else np.nan
        dur  = float(matched['Duration'].mean())  if len(matched) else np.nan
        rms  = float(matched['RMS'].mean())        if len(matched) else np.nan
        prefix = f'event_locked_{group}'
        out[f'{prefix}_n_bursts']                  = n_bursts
        out[f'{prefix}_n_bursts_with_spindle']     = bursts_with_sp
        out[f'{prefix}_spindle_rate_per_burst']    = round(sp_rate_per_burst, 4)
        out[f'{prefix}_spindle_density_per_s']     = round(sp_rate_per_burst / post_window_sec, 4)
        out[f'{prefix}_spindle_amplitude_uv']      = round(amp,  3) if not np.isnan(amp)  else np.nan
        out[f'{prefix}_spindle_frequency_hz']      = round(freq, 3) if not np.isnan(freq) else np.nan
        out[f'{prefix}_spindle_duration_sec']      = round(dur,  3) if not np.isnan(dur)  else np.nan
        out[f'{prefix}_spindle_rms_uv']            = round(rms,  3) if not np.isnan(rms)  else np.nan
        out[f'{prefix}_total_spindles_in_windows'] = total_sp
        print(f'    Event-locked [{group}]: {n_bursts} bursts, {total_sp} post-burst spindles')
    for metric in ('spindle_rate_per_burst', 'spindle_density_per_s',
                   'spindle_amplitude_uv', 'spindle_frequency_hz', 'spindle_duration_sec'):
        a = out.get(f'event_locked_active_{metric}', np.nan)
        s = out.get(f'event_locked_sham_{metric}',   np.nan)
        out[f'event_locked_active_minus_sham_{metric}'] = (
            round(a - s, 4) if not (np.isnan(a) or np.isnan(s)) else np.nan
        )
    return out


# =============================================================================
# Burst-level analysis
# =============================================================================

def run_pulse_level_analysis(raw, vmrk_path, hypno_int, hypno_up,
                              freq_band, session_name, participant_id,
                              is_adaptation, output_dir):
    if is_adaptation or not vmrk_path or not Path(vmrk_path).exists():
        return {}
    print(f'\n[8] Burst-level analysis: {participant_id} / {session_name}')
    original_sfreq = read_original_sfreq(vmrk_path.replace('.vmrk', '.vhdr'))
    bursts = parse_tus_markers_bursts(vmrk_path, original_sfreq)
    if bursts.empty:
        print('    No bursts found')
        return {}

    analysis_is_nrem = hypno_int is not None
    sfreq        = raw.info['sfreq']
    nrem_mask    = nrem_mask_from_hypno(hypno_int, raw)
    pre_samples  = int(TUS_EPOCH_PRE_SEC * sfreq)
    post_samples = int(TUS_EPOCH_POST_SEC * sfreq)

    channels    = [ch for ch in SPINDLE_CHANNELS if ch in raw.ch_names]
    sw_channels = [ch for ch in SW_CHANNELS       if ch in raw.ch_names]
    spindle_starts_sec, slowwave_starts_sec = np.array([]), np.array([])
    sp_summary = None
    if channels:
        sp_obj = yasa.spindles_detect(raw, ch_names=channels, freq_sp=freq_band,
                                      hypno=hypno_up, include=NREM_STAGES)
        if sp_obj is not None:
            sp_summary = sp_obj.summary()
            spindle_starts_sec = sp_summary['Start'].values
    if sw_channels:
        sw_obj = yasa.sw_detect(raw, ch_names=sw_channels, freq_sw=SW_FREQ,
                                hypno=hypno_up, include=NREM_STAGES)
        if sw_obj is not None:
            slowwave_starts_sec = sw_obj.summary()['Start'].values

    counts = {'total': len(bursts), 'skipped_condition': 0, 'skipped_bounds': 0,
              'skipped_nrem': 0, 'skipped_spindle': 0, 'kept_active': 0, 'kept_sham': 0}
    suffix      = 'nrem' if analysis_is_nrem else 'full_recording'
    out_csv     = Path(output_dir) / f'{participant_id}_{session_name}_{suffix}_per_pulse_features.csv'
    BATCH_SIZE  = 100
    rows, first_write = [], True
    burst_times_by_group = {'active': [], 'sham': []}

    for _, burst in bursts.iterrows():
        condition = burst['condition']
        group = ('active' if condition in ACTIVE_CONDITIONS else
                 'sham'   if condition in SHAM_CONDITIONS   else None)
        if group is None:
            counts['skipped_condition'] += 1
            continue
        burst_time_sec = burst['time_sec']
        center = int(burst_time_sec * sfreq)
        start, stop = center - pre_samples, center + post_samples
        if start < 0 or stop > raw.n_times:
            counts['skipped_bounds'] += 1
            continue
        if analysis_is_nrem and not nrem_mask[center]:
            counts['skipped_nrem'] += 1
            continue
        if len(spindle_starts_sec) > 0:
            prev = spindle_starts_sec[spindle_starts_sec < burst_time_sec]
            if len(prev) > 0 and (burst_time_sec - prev[-1]) < 3.5:
                counts['skipped_spindle'] += 1
                continue

        brain_state = 'none'
        if len(spindle_starts_sec) > 0 and np.any(
            (spindle_starts_sec >= burst_time_sec - 0.5) &
            (spindle_starts_sec <= burst_time_sec + 0.5)
        ):
            brain_state = 'spindle'
        if brain_state == 'none' and len(slowwave_starts_sec) > 0 and np.any(
            (slowwave_starts_sec >= burst_time_sec - 0.5) &
            (slowwave_starts_sec <= burst_time_sec + 0.5)
        ):
            brain_state = 'slow_wave'
        sleep_stage = (int(hypno_int[min(int(burst_time_sec/30), len(hypno_int)-1)])
                       if hypno_int is not None else -1)
        window = raw.get_data(start=start, stop=stop)
        row = {
            'participant_id': participant_id, 'session': session_name,
            'analysis_scope': 'NREM_N2_N3' if analysis_is_nrem else 'FULL_RECORDING_NO_STAGING',
            'condition': condition, 'group': group,
            'burst_time_s': round(burst_time_sec, 3),
            'n_pulses': int(burst['n_pulses']),
            'burst_duration_sec': round(float(burst['duration_sec']), 4),
            'sleep_stage': sleep_stage, 'brain_state': brain_state,
            'burst_seq_all': int(burst.get('burst_seq_all', np.nan)) if 'burst_seq_all' in burst.index else np.nan,
            'first_trigger_seq_all': int(burst.get('first_trigger_seq_all', np.nan)) if 'first_trigger_seq_all' in burst.index else np.nan,
        }
        row.update(compute_window_features(window, raw.ch_names, sfreq, freq_band))
        del window
        rows.append(row)
        counts[f'kept_{group}'] += 1
        burst_times_by_group[group].append(burst_time_sec)
        if len(rows) >= BATCH_SIZE:
            pd.DataFrame(rows).to_csv(out_csv, mode='a', header=first_write, index=False)
            first_write = False
            rows.clear()
            gc.collect()
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, mode='a', header=first_write, index=False)
        rows.clear()
        gc.collect()

    print(f'    Kept active={counts["kept_active"]} sham={counts["kept_sham"]}')
    if not out_csv.exists():
        return {}

    pulse_df = pd.read_csv(out_csv)
    excluded = {'participant_id', 'session', 'analysis_scope', 'condition', 'group',
                'burst_time_s', 'n_pulses', 'burst_duration_sec', 'sleep_stage', 'brain_state'}
    feature_cols = [c for c in pulse_df.columns if c not in excluded]
    summary = (pulse_df.groupby('group')[feature_cols].mean(numeric_only=True).reset_index())
    summary.insert(0, 'analysis_scope', 'NREM_N2_N3' if analysis_is_nrem else 'FULL_RECORDING_NO_STAGING')
    summary.insert(0, 'session', session_name)
    summary.insert(0, 'participant_id', participant_id)
    summary.to_csv(
        Path(output_dir) / f'{participant_id}_{session_name}_{suffix}_pulse_summary.csv',
        index=False
    )
    del pulse_df, summary
    gc.collect()

    el_spindle_features = compute_event_locked_spindle_features(
        burst_times_by_group, sp_summary, post_window_sec=TUS_EPOCH_POST_SEC
    )
    if el_spindle_features:
        el_df = pd.DataFrame([{
            'participant_id': participant_id, 'session': session_name,
            'analysis_scope': suffix, **el_spindle_features
        }])
        el_df.to_csv(
            Path(output_dir) / f'{participant_id}_{session_name}_{suffix}_event_locked_spindles.csv',
            index=False
        )
        del el_df
        gc.collect()

    # Save MNE Epochs
    try:
        vhdr_path_epo = vmrk_path.replace('.vmrk', '.vhdr')
        raw_epo = mne.io.read_raw_brainvision(vhdr_path_epo, preload=False, verbose=False)
        viz_present = [ch for ch in VIZ_CHANNELS if ch in raw_epo.ch_names]
        if viz_present:
            raw_epo.pick_channels(viz_present)
        if raw_epo.info['sfreq'] > RESAMPLE_FREQ:
            raw_epo.resample(RESAMPLE_FREQ, verbose=False)
        raw_epo.load_data()
        raw_epo.filter(BANDPASS_LOW, BANDPASS_HIGH, verbose=False)
        burst_df_epo = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
        if not burst_df_epo.empty:
            cond_to_code = {c: i for i, c in enumerate(burst_df_epo['condition'].unique(), 1)}
            epo_sfreq    = raw_epo.info['sfreq']
            events_epo   = np.array([
                [int(row['burst_time_s'] * epo_sfreq), 0, cond_to_code.get(row['condition'], 1)]
                for _, row in burst_df_epo.iterrows()
                if row['burst_time_s'] * epo_sfreq < raw_epo.n_times
            ], dtype=int)
            if len(events_epo):
                valid_mask = [row['burst_time_s'] * epo_sfreq < raw_epo.n_times
                              for _, row in burst_df_epo.iterrows()]
                metadata_epo = burst_df_epo[valid_mask].reset_index(drop=True)
                epochs_obj   = mne.Epochs(
                    raw_epo, events_epo, event_id=cond_to_code,
                    tmin=-TUS_EPOCH_PRE_SEC, tmax=TUS_EPOCH_POST_SEC,
                    baseline=None, preload=True, reject_by_annotation=False, verbose=False,
                )
                if len(epochs_obj) <= len(metadata_epo):
                    epochs_obj.metadata = metadata_epo.iloc[epochs_obj.selection].reset_index(drop=True)
                epo_path = Path(output_dir) / f'{participant_id}_{session_name}_{suffix}_epochs-epo.fif'
                epochs_obj.save(str(epo_path), overwrite=True, verbose=False)
                print(f'    Saved epochs: {epo_path.name}')
                del epochs_obj
        del burst_df_epo, raw_epo
        gc.collect()
    except Exception as exc:
        print(f'    Epochs save skipped: {exc}')

    return {
        'n_active': counts['kept_active'], 'n_sham': counts['kept_sham'],
        'analysis_scope': suffix, 'event_locked_spindles': el_spindle_features,
    }


# =============================================================================
# Visualisations
# =============================================================================

def safe_plot(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as e:
        print(f'    Plot failed ({fn.__name__}): {e}')
    finally:
        plt.close('all')
        gc.collect()


def plot_raw_vs_preprocessed(raw_snapshot_uv, raw_post, channels,
                              snapshot_secs, session_name, participant_id, output_dir):
    """QC plot: pre-ICA snapshot (loaded from .npy) vs. preprocessed .fif."""
    sfreq     = raw_post.info['sfreq']
    n_samp    = raw_snapshot_uv.shape[1]
    times_min = np.linspace(0, snapshot_secs / 60, n_samp)
    n_ch      = len(channels)
    fig, axes = plt.subplots(n_ch, 2, figsize=(20, max(2.2 * n_ch, 8)), sharex=True)
    if n_ch == 1:
        axes = axes[np.newaxis, :]
    for row, ch in enumerate(channels):
        ax_raw  = axes[row, 0]
        ax_post = axes[row, 1]
        raw_trace  = raw_snapshot_uv[row]
        post_samp  = min(n_samp, raw_post.n_times)
        post_trace = raw_post.get_data(picks=[ch], start=0, stop=post_samp)[0] * 1e6
        ylim = max(np.percentile(np.abs(raw_trace), 99) * 1.15, 10.0)
        for ax, trace, color, label in [
            (ax_raw,  raw_trace,  '#d62728', 'Before ICA'),
            (ax_post, post_trace, '#1f77b4', 'After ICA'),
        ]:
            ax.plot(times_min[:len(trace)], trace, lw=0.5, color=color, alpha=0.85, rasterized=True)
            ax.set_ylim(-ylim, ylim)
            ax.axhline(0,     color='grey',   lw=0.5, ls='--', alpha=0.4)
            ax.axhline( 150,  color='orange', lw=0.7, ls=':',  alpha=0.6)
            ax.axhline(-150,  color='orange', lw=0.7, ls=':',  alpha=0.6)
            ax.set_ylabel(f'{ch}\n(µV)', fontsize=7, labelpad=2)
            if row == 0:
                ax.set_title(label, fontsize=11, fontweight='bold', color=color)
            ax.tick_params(labelsize=6)
    for ax in axes[-1, :]:
        ax.set_xlabel('Time (min)', fontsize=9)
    fig.suptitle(
        f'{participant_id} – {session_name}: Raw vs Preprocessed  '
        f'(first {snapshot_secs/60:.1f} min | {n_ch} channels)',
        fontsize=12, fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(
        Path(output_dir) / f'{participant_id}_{session_name}_raw_vs_preprocessed.png',
        dpi=120, bbox_inches='tight'
    )
    plt.close(fig)
    print(f'    Saved raw vs preprocessed')


def plot_spectrogram(raw, hypno_int, session_name, participant_id, output_dir):
    channels = [ch for ch in VIZ_CHANNELS if ch in raw.ch_names]
    if not channels:
        return
    sfreq = raw.info['sfreq']
    n_fft = int(sfreq * 4)
    hop   = int(sfreq * 2)
    fig, axes = plt.subplots(len(channels), 1, figsize=(16, 3.5 * len(channels)), sharex=True)
    if len(channels) == 1:
        axes = [axes]
    for ax, ch in zip(axes, channels):
        data = raw.get_data(picks=[ch])[0]
        freqs, times, Sxx = scipy_spectrogram(
            data, fs=sfreq, nperseg=n_fft, noverlap=n_fft - hop, scaling='density'
        )
        fmask  = freqs <= 30.0
        Sxx_db = 10 * np.log10(Sxx[fmask] + 1e-30)
        pcm    = ax.pcolormesh(times / 60, freqs[fmask], Sxx_db,
                               cmap='inferno', shading='gouraud',
                               vmin=np.percentile(Sxx_db, 5),
                               vmax=np.percentile(Sxx_db, 98))
        del data, Sxx, Sxx_db
        gc.collect()
        fig.colorbar(pcm, ax=ax, label='dB/Hz', pad=0.01)
        if hypno_int is not None:
            for ei, stage in enumerate(hypno_int):
                if stage in NREM_STAGES:
                    ax.axvspan(ei * 30 / 60, (ei + 1) * 30 / 60, color='cyan', alpha=0.12)
        ax.axhline(SPINDLE_FREQ_DEFAULT[0], color='white', lw=0.8, ls='--', alpha=0.6)
        ax.axhline(SPINDLE_FREQ_DEFAULT[1], color='white', lw=0.8, ls='--', alpha=0.6, label='spindle band')
        ax.set_ylabel('Hz')
        ax.set_title(ch, fontsize=10)
    axes[-1].set_xlabel('Time (min)')
    fig.suptitle(f'{participant_id} – {session_name}: spectrogram  [cyan = NREM]',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(
        Path(output_dir) / f'{participant_id}_{session_name}_spectrogram.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close(fig)
    print('    Saved spectrogram')


# =============================================================================
# ERP and TFR visualisations
# =============================================================================
#
# -----------------------------------------------------
# plot_erps
# Multiple baseline corrections are tried (none, pre-mean,pre-zscore) and each is saved as a separate figure.  
# After averaging across trials, channels are ranked by peak absolute ERP amplitude.  
# Individual trials are then only plotted for the top-N_BEST_CHANNELS channels.
#              
# Noisy trials are flagged and excluded before averaging and  before individual-trial plots.  A trial is considered noisy
# if its peak-to-peak amplitude (in the analysis window) exceeds TRIAL_NOISE_THRESHOLD_UV. 
#             
#  Per-trial mean amplitude during 0–1 s post-onset is extracted,plotted against trial number, and a linear regression (slope + R²) is fitted and displayed
# — this is the habituation/drift analysis. 
# A scalp topography of the peak ERP amplitude (max abs across  the 0–1 s window) is plotted using mne.viz.plot_topomap.
#             
#            
# plot_tfrs
#   Baseline window is now −0.3 to −0.05 s.  
#   Multiple baseline windows are tried and each is saved separately.  
#   Average power per frequency band, per trial, per channel,and per post-onset time window (0–1 s) is extracted and saved
#   to a CSV for downstream statistics.  
#   The "focus channel" — the channel with the biggest ERP response — is passed in and used to produce a dedicated
#   per-trial TFR panel.   
#   A scalp topography of mean TFR band power (averaged over  0–1 s post-onset) is plotted for each frequency band.
#   Trial-by-trial habituation/drift for TFR band power is computed, plotted, and a linear regression is fitted — one
#  panel per band on the focus channel. 
#         
# =============================================================================


# baseline options that will be tried for ERPs and TFRs
ERP_BASELINES = {
    'none':    None,                  # raw signal, no baseline
    'pre_mean': 'pre_mean',           # subtract mean of pre-stimulus window
    'pre_zscore': 'pre_zscore',       # z-score relative to pre-stimulus window
}
TFR_BASELINES = {
    'tight_300_50ms':  (-0.30, -0.05),   # lead's recommended baseline
    'tight_500_100ms': (-0.50, -0.10),   # slightly wider alternative
    'full_pre':        (-TUS_EPOCH_PRE_SEC, -0.5),  # original (kept for comparison)
}

# number of "biggest response" channels to show individual trials for
N_BEST_CHANNELS = 3

# peak-to-peak threshold for trial exclusion (µV, broadband)
TRIAL_NOISE_THRESHOLD_UV = 500.0

# time window for amplitude extraction and habituation analysis
HABITUATION_WINDOW_SEC = (0.0, 1.0)

# Frequency bands for TFR band-power extraction (CHANGED 8)
TFR_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'sigma': (12.0, 15.0),
    'beta':  (15.0, 30.0),
}


def _apply_erp_baseline(epochs_2d, pre_samples, mode):
    """
    Apply one of three baseline corrections to a (n_trials, n_times) array.
    """
    out = epochs_2d.copy()
    pre = epochs_2d[:, :pre_samples]
    if mode == 'pre_mean':
        out = out - pre.mean(axis=1, keepdims=True)
    elif mode == 'pre_zscore':
        mu  = pre.mean(axis=1, keepdims=True)
        sd  = pre.std(axis=1, keepdims=True) + 1e-12
        out = (out - mu) / sd
    # mode == 'none': return as-is
    return out


def _exclude_noisy_trials(epochs_2d, threshold_uv):
    """
    Return a boolean mask (True = keep) based on peak-to-peak amplitude.

    """
    ptp = np.ptp(epochs_2d, axis=1)          # peak-to-peak per trial
    mask = ptp <= threshold_uv
    n_excluded = int((~mask).sum())
    if n_excluded:
        print(f'      Noise exclusion: removed {n_excluded} / {len(mask)} trials '
              f'(threshold {threshold_uv} µV p-p)')
    return mask


def _rank_channels_by_erp(mean_erps, ch_names, post_start_idx):
    """
    Rank channels by peak absolute amplitude in the post-onset window.
    Returns list of channel names sorted largest → smallest.

    CHANGED 2: the original plotted all VIZ_CHANNELS with no ranking.
    """
    scores = {}
    for ch, erp in zip(ch_names, mean_erps):
        scores[ch] = np.max(np.abs(erp[post_start_idx:]))
    return sorted(scores, key=scores.get, reverse=True), scores


def _habituation_plot(trial_amplitudes, trial_numbers, ch_name, condition,
                      session_name, participant_id, output_dir, suffix, kind):
    """
    Plot amplitude (or band power) vs trial number with a linear regression.
    """
    from scipy.stats import linregress

    valid = ~np.isnan(trial_amplitudes)
    x = trial_numbers[valid]
    y = trial_amplitudes[valid]
    if len(x) < 3:
        return

    slope, intercept, r, p, _ = linregress(x, y)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x, y, color='steelblue', s=30, alpha=0.7, zorder=3)
    ax.plot(x, slope * x + intercept, color='crimson', lw=1.8,
            label=f'slope={slope:.4f}  R²={r**2:.3f}  p={p:.3f}')
    ax.axhline(0, color='grey', lw=0.7, ls='--', alpha=0.5)
    ax.set_xlabel('Trial number')
    ylabel = 'Mean amplitude (µV)' if kind == 'ERP' else f'Mean {kind} power (dB)'
    ax.set_ylabel(ylabel)
    ax.set_title(
        f'{participant_id} – {session_name}\n'
        f'{kind} habituation/drift  |  {ch_name}  [{condition}]'
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fname = (f'{participant_id}_{session_name}_{suffix}_'
             f'habituation_{kind}_{ch_name}_{condition}.png')
    fig.savefig(Path(output_dir) / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'      Saved habituation plot: {fname}')


def _erp_topomap(mean_amp_by_channel, ch_names_topo, info_topo,
                 session_name, participant_id, output_dir, suffix, condition, baseline_name):
    """
    Plot scalp topography of peak ERP amplitude (max |ERP| in 0–1 s window).
    """
    vals = np.array([mean_amp_by_channel.get(ch, np.nan) for ch in ch_names_topo])
    if np.all(np.isnan(vals)):
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    mne.viz.plot_topomap(
        np.nan_to_num(vals), info_topo,
        axes=ax, show=False, cmap='RdBu_r',
        vlim=(-np.nanpercentile(np.abs(vals), 95),
               np.nanpercentile(np.abs(vals), 95)),
    )
    ax.set_title(
        f'{condition}  |  ERP peak |amplitude| (0–1 s)\n'
        f'baseline: {baseline_name}',
        fontsize=9
    )
    fig.tight_layout()
    fname = (f'{participant_id}_{session_name}_{suffix}_'
             f'ERP_topomap_{condition}_{baseline_name}.png')
    fig.savefig(Path(output_dir) / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'      Saved ERP topomap: {fname}')


def plot_erps(raw, bursts_df, freq_band, session_name, participant_id,
              output_dir, suffix=''):
    """
    ERP analysis 
    -------------------
   Three baseline modes are tried and each produces its own figure.
   Channels are ranked by peak |ERP|; individual trials are only plotted for the top N_BEST_CHANNELS.
   Noisy trials are excluded before averaging.
   Habituation/drift figure: mean amplitude (0–1 s) vs trial number with linear regression, one figure per channel × condition.
   Scalp topomap of peak ERP amplitude for each condition/baseline.
    """
   

    channels = [ch for ch in VIZ_CHANNELS if ch in raw.ch_names]
    if not channels or bursts_df.empty:
        return

    sfreq        = raw.info['sfreq']
    pre_samples  = int(TUS_EPOCH_PRE_SEC * sfreq)
    post_samples = int(TUS_EPOCH_POST_SEC * sfreq)
    n_samples    = pre_samples + post_samples
    times        = np.linspace(-TUS_EPOCH_PRE_SEC, TUS_EPOCH_POST_SEC, n_samples)

    # Index into the post-onset window for habituation / ranking
    post_start_idx   = pre_samples
    hab_start        = pre_samples + int(HABITUATION_WINDOW_SEC[0] * sfreq)
    hab_end          = pre_samples + int(HABITUATION_WINDOW_SEC[1] * sfreq)

    def bandpass(data, low, high, fs):
        b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype='band')
        return filtfilt(b, a, data)

    # Build a topomap info object once
    montage   = mne.channels.make_standard_montage('standard_1020')
    known_chs = set(montage.ch_names)
    topo_chs  = [ch for ch in channels if ch in known_chs]
    info_topo = None
    if len(topo_chs) >= 3:
        info_topo = mne.create_info(topo_chs, sfreq=sfreq, ch_types='eeg')
        info_topo.set_montage(montage, on_missing='ignore')

    # Collect all epochs per channel across both conditions first,
    # then loop over baselines so we only read raw data once.
    all_epochs = {ch: {} for ch in channels}   # ch -> {condition: (n_trials, n_times)}
    all_trial_nums = {}                         # condition -> array of 1-based trial indices

    for group_label, condition_set in [('sham', SHAM_CONDITIONS), ('active', ACTIVE_CONDITIONS)]:
        mask     = bursts_df['condition'].isin(condition_set)
        group_df = bursts_df[mask].reset_index(drop=True)
        trial_nums = np.arange(1, len(group_df) + 1)
        all_trial_nums[group_label] = trial_nums

        for ch in channels:
            ch_idx = raw.ch_names.index(ch)
            trials = []
            for _, burst in group_df.iterrows():
                center = int(burst['time_sec'] * sfreq)
                start, stop = center - pre_samples, center + post_samples
                if start < 0 or stop > raw.n_times:
                    trials.append(np.full(n_samples, np.nan))
                    continue
                trial = raw.get_data(start=start, stop=stop)[ch_idx] * 1e6
                try:
                    trial = bandpass(trial, freq_band[0], freq_band[1], sfreq)
                except Exception:
                    pass
                trials.append(trial)
            all_epochs[ch][group_label] = np.array(trials)   # (n_trials, n_times)

    # loop over baseline modes
    for baseline_name, baseline_mode in ERP_BASELINES.items():
        print(f'\n    ERP baseline: {baseline_name}')

        # Per-condition, per-channel mean ERPs (after baseline + noise exclusion)
        mean_erps_by_condition = {}
        peak_amp_by_condition  = {}   # for topomap (CHANGED 5)
        focus_channel_by_condition = {}   # CHANGED 2

        for group_label in ('sham', 'active'):
            mean_erps   = []
            clean_masks = []   #keep track of which trials survived
            for ch in channels:
                raw_trials = all_epochs[ch][group_label].copy()

                # apply the current baseline correction
                corrected = _apply_erp_baseline(raw_trials, pre_samples, baseline_mode)

                # exclude noisy trials
                finite_mask = np.all(np.isfinite(corrected), axis=1)
                noise_mask  = _exclude_noisy_trials(
                    corrected[finite_mask], TRIAL_NOISE_THRESHOLD_UV
                )
                keep = np.where(finite_mask)[0][noise_mask]
                clean = corrected[keep]
                clean_masks.append(keep)
                mean_erps.append(clean.mean(axis=0) if len(clean) else np.full(n_samples, np.nan))

            mean_erps_by_condition[group_label] = mean_erps

            # rank channels by post-onset peak amplitude
            ranked_chs, scores = _rank_channels_by_erp(mean_erps, channels, post_start_idx)
            focus_channel_by_condition[group_label] = ranked_chs[0]
            peak_amp_by_condition[group_label]      = {ch: scores[ch] for ch in channels}

            print(f'      [{group_label}] Channel ranking (peak |ERP|, 0–end):')
            for rank_i, rc in enumerate(ranked_chs[:5], 1):
                print(f'        {rank_i}. {rc}  {scores[rc]:.2f} µV')

            # Per-condition ERP figure (mean + best channels individual trials)
            best_chs = ranked_chs[:N_BEST_CHANNELS]   
            fig, axes = plt.subplots(
                len(best_chs), 1,
                figsize=(14, 4 * len(best_chs)),
                sharex=True,
            )
            if len(best_chs) == 1:
                axes = [axes]

            color = '#4B7BE0' if group_label == 'sham' else '#E04B4B'
            for ax, ch in zip(axes, best_chs):
                ch_idx_in_list = channels.index(ch)
                keep           = clean_masks[ch_idx_in_list]
                clean_trials   = _apply_erp_baseline(
                    all_epochs[ch][group_label], pre_samples, baseline_mode
                )[keep]

                # individual trials for top channels only
                for trial in clean_trials:
                    ax.plot(times, trial, color=color, alpha=0.12, lw=0.6)

                mean_erp = mean_erps[ch_idx_in_list]
                sem_erp  = (clean_trials.std(axis=0) / np.sqrt(len(clean_trials))
                            if len(clean_trials) > 1 else np.zeros(n_samples))
                ax.fill_between(times, mean_erp - sem_erp, mean_erp + sem_erp,
                                color=color, alpha=0.3)
                ax.plot(times, mean_erp, color=color, lw=2.0,
                        label=f'Mean (n={len(clean_trials)})')
                ax.axvline(0, color='black', lw=1.0, ls='--', alpha=0.7, label='TUS onset')
                ax.axvspan(-TUS_EPOCH_PRE_SEC, 0, color='grey', alpha=0.07, label='Baseline')
                ax.axhline(0, color='grey', lw=0.6, ls=':')
                ax.set_ylabel('µV' if baseline_name != 'pre_zscore' else 'z-score')
                rank_pos = ranked_chs.index(ch) + 1
                ax.set_title(
                    f'{ch}  [rank #{rank_pos} by peak |ERP|  |  n={len(clean_trials)} trials]',
                    fontsize=10, fontweight='bold'
                )
                ax.legend(fontsize=8, loc='upper right')

            axes[-1].set_xlabel('Time (s)')
            fig.suptitle(
                f'{participant_id} – {session_name}  |  {group_label.upper()}  ERP  '
                f'[{freq_band[0]}–{freq_band[1]} Hz]  |  baseline: {baseline_name}\n'
                f'Individual trials shown for top {N_BEST_CHANNELS} channels by response amplitude',
                fontsize=11, fontweight='bold'
            )
            fig.tight_layout()
            fname = (f'{participant_id}_{session_name}_{suffix}_'
                     f'ERP_{group_label}_{baseline_name}.png')
            fig.savefig(Path(output_dir) / fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'      Saved ERP figure: {fname}')

            # habituation/drift — mean amplitude (0–1 s) vs trial number
            trial_nums = all_trial_nums[group_label]
            focus_ch   = focus_channel_by_condition[group_label]
            focus_idx  = channels.index(focus_ch)
            keep_focus = clean_masks[focus_idx]

            # Align trial numbers to surviving trials
            hab_amps = []
            for t_idx in keep_focus:
                corrected_trial = _apply_erp_baseline(
                    all_epochs[focus_ch][group_label][[t_idx]], pre_samples, baseline_mode
                )[0]
                hab_amps.append(corrected_trial[hab_start:hab_end].mean())

            if len(hab_amps) >= 3:
                _habituation_plot(
                    trial_amplitudes=np.array(hab_amps),
                    trial_numbers=np.arange(1, len(hab_amps) + 1),
                    ch_name=focus_ch, condition=group_label,
                    session_name=session_name, participant_id=participant_id,
                    output_dir=output_dir, suffix=f'{suffix}_{baseline_name}', kind='ERP'
                )

            # topomap of peak ERP amplitude
            if info_topo is not None:
                _erp_topomap(
                    peak_amp_by_condition[group_label],
                    topo_chs, info_topo,
                    session_name, participant_id, output_dir,
                    suffix, group_label, baseline_name
                )

    # Return the focus channel from the tight-baseline active condition so
    # plot_tfrs can use it (CHANGED 9 in plot_tfrs).
    return focus_channel_by_condition.get('active', channels[0]) if channels else None


def plot_tfrs(raw, bursts_df, freq_band, session_name, participant_id,
              output_dir, suffix='', focus_channel=None):
    """
    TFR analysis 

   
    -------------------
    Default baseline is −0.3 to −0.05 s
    Multiple baseline windows are tried; each produces its own figure.
    Average power per band, per trial, per channel, and per post-onset time window is extracted and saved to a CSV.
    The focus_channel (biggest ERP response, from plot_erps) gets a dedicated per-trial TFR panel.
    Scalp topomaps of mean TFR band power (0–1 s) for each band.
    Habituation/drift for TFR band power on the focus channel.
    """
    channels = [ch for ch in VIZ_CHANNELS if ch in raw.ch_names]
    if not channels or bursts_df.empty:
        return

    sfreq        = raw.info['sfreq']
    pre_samples  = int(TUS_EPOCH_PRE_SEC  * sfreq)
    post_samples = int(TUS_EPOCH_POST_SEC * sfreq)
    n_samples    = pre_samples + post_samples
    times        = np.linspace(-TUS_EPOCH_PRE_SEC, TUS_EPOCH_POST_SEC, n_samples)

    freqs    = np.arange(1.0, 31.0, 1.0)
    n_cycles = freqs / 2.0

    hab_start_idx = pre_samples + int(HABITUATION_WINDOW_SEC[0] * sfreq)
    hab_end_idx   = pre_samples + int(HABITUATION_WINDOW_SEC[1] * sfreq)

    # Topomap setup 
    montage   = mne.channels.make_standard_montage('standard_1020')
    known_chs = set(montage.ch_names)
    topo_chs  = [ch for ch in channels if ch in known_chs]
    info_topo = None
    if len(topo_chs) >= 3:
        info_topo = mne.create_info(topo_chs, sfreq=sfreq, ch_types='eeg')
        info_topo.set_montage(montage, on_missing='ignore')

    # fall back if no focus channel was returned
    if focus_channel is None or focus_channel not in channels:
        focus_channel = channels[0]
    print(f'    TFR focus channel (from ERP ranking): {focus_channel}')

    def morlet_tfr(epochs_2d):
        """(n_trials, n_times) → (n_trials, n_freqs, n_times) power."""
        data_3d = epochs_2d[:, np.newaxis, :]
        power_4d = mne.time_frequency.tfr_array_morlet(
            data_3d, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles,
            output='power', verbose=False,
        )
        return power_4d[:, 0, :, :]   # (n_trials, n_freqs, n_times)

    def apply_tfr_baseline(power_3d, bl_start_sec, bl_end_sec):
        """
        dB baseline correction on (n_trials, n_freqs, n_times).

        CHANGED 6/7: baseline window is now a parameter rather than hardcoded.
        """
        bl_s = pre_samples + int(bl_start_sec * sfreq)
        bl_e = pre_samples + int(bl_end_sec   * sfreq)
        bl_s = max(bl_s, 0)
        bl_e = min(bl_e, n_samples)
        bl_power = power_3d[:, :, bl_s:bl_e].mean(axis=2, keepdims=True)
        return 10 * np.log10(power_3d / (bl_power + 1e-30))

    # Collect raw (unbaselined) epochs per channel per condition once
    raw_power = {}   # (ch, group_label) -> (n_trials, n_freqs, n_times)
    for group_label, condition_set in [('sham', SHAM_CONDITIONS), ('active', ACTIVE_CONDITIONS)]:
        mask     = bursts_df['condition'].isin(condition_set)
        group_df = bursts_df[mask].reset_index(drop=True)
        if len(group_df) < 2:
            continue
        for ch in channels:
            ch_idx = raw.ch_names.index(ch)
            epochs = []
            for _, burst in group_df.iterrows():
                center = int(burst['time_sec'] * sfreq)
                start, stop = center - pre_samples, center + post_samples
                if start < 0 or stop > raw.n_times:
                    continue
                epochs.append(raw.get_data(start=start, stop=stop)[ch_idx] * 1e6)
            if len(epochs) >= 2:
                raw_power[(ch, group_label)] = morlet_tfr(np.array(epochs))

    # loop over baseline windows
    for bl_name, (bl_start, bl_end) in TFR_BASELINES.items():
        print(f'\n    TFR baseline: {bl_name}  ({bl_start:.2f} to {bl_end:.2f} s)')

        #  container for band × trial × channel × window CSV
        band_power_rows = []

        for group_label, condition_set in [('sham', SHAM_CONDITIONS), ('active', ACTIVE_CONDITIONS)]:
            mask     = bursts_df['condition'].isin(condition_set)
            group_df = bursts_df[mask].reset_index(drop=True)
            n_trials = sum(
                1 for _, burst in group_df.iterrows()
                if 0 <= int(burst['time_sec'] * sfreq) - pre_samples
                and int(burst['time_sec'] * sfreq) + post_samples <= raw.n_times
            )
            if n_trials < 2:
                continue

            color = '#4B7BE0' if group_label == 'sham' else '#E04B4B'

            # --- Mean TFR figure per channel ---
            for ch in channels:
                key = (ch, group_label)
                if key not in raw_power:
                    continue
                power_3d = apply_tfr_baseline(raw_power[key], bl_start, bl_end)
                mean_tfr = power_3d.mean(axis=0)   # (n_freqs, n_times)
                n_used   = power_3d.shape[0]

                vmax = np.nanpercentile(np.abs(mean_tfr), 97)
                fig, ax = plt.subplots(figsize=(12, 5))
                pcm = ax.pcolormesh(times, freqs, mean_tfr,
                                    cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                                    shading='gouraud')
                fig.colorbar(pcm, ax=ax, label='dB (re: baseline)')
                ax.axvline(0, color='black', lw=1.2, ls='--', alpha=0.8, label='TUS onset')
                ax.axhspan(freq_band[0], freq_band[1], color='yellow',
                           alpha=0.15, label='Spindle band')
                # Mark baseline window 
                ax.axvspan(bl_start, bl_end, color='lime', alpha=0.12, label='Baseline window')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Frequency (Hz)')
                ax.set_title(
                    f'{group_label.upper()}  |  {ch}  (n={n_used} trials)',
                    fontsize=11, fontweight='bold'
                )
                ax.legend(fontsize=8, loc='upper right')
                fig.suptitle(
                    f'{participant_id} – {session_name}  |  {ch}  TFR  '
                    f'[Morlet]  |  baseline: {bl_name}',
                    fontsize=11, fontweight='bold'
                )
                fig.tight_layout()
                fname = (f'{participant_id}_{session_name}_{suffix}_'
                         f'TFR_{ch}_{group_label}_{bl_name}.png')
                fig.savefig(Path(output_dir) / fname, dpi=150, bbox_inches='tight')
                plt.close(fig)

                # extract per-band per-trial power in the 0–1 s window
                for band_name, (b_low, b_high) in TFR_BANDS.items():
                    freq_mask = (freqs >= b_low) & (freqs <= b_high)
                    if not freq_mask.any():
                        continue
                    # power per trial in the hab window
                    trial_band_power = power_3d[:, freq_mask, :][:, :, hab_start_idx:hab_end_idx].mean(axis=(1, 2))
                    for t_idx, bp in enumerate(trial_band_power):
                        band_power_rows.append({
                            'participant_id': participant_id,
                            'session':        session_name,
                            'baseline':       bl_name,
                            'condition':      group_label,
                            'channel':        ch,
                            'band':           band_name,
                            'trial':          t_idx + 1,
                            'mean_power_db':  round(float(bp), 6),
                        })

            # per-trial TFR on the focus channel
            key_focus = (focus_channel, group_label)
            if key_focus in raw_power:
                power_3d_focus  = apply_tfr_baseline(raw_power[key_focus], bl_start, bl_end)
                n_focus_trials  = power_3d_focus.shape[0]
                n_show          = min(n_focus_trials, 16)
                ncols           = 4
                nrows           = int(np.ceil(n_show / ncols))
                fig, axes = plt.subplots(nrows, ncols,
                                         figsize=(ncols * 4, nrows * 3),
                                         sharex=True, sharey=True)
                axes = np.array(axes).ravel()
                vmax_focus = np.nanpercentile(np.abs(power_3d_focus), 97)
                for ti in range(n_show):
                    ax = axes[ti]
                    pcm = ax.pcolormesh(times, freqs, power_3d_focus[ti],
                                        cmap='RdBu_r', vmin=-vmax_focus, vmax=vmax_focus,
                                        shading='gouraud')
                    ax.axvline(0, color='white', lw=0.8, ls='--', alpha=0.7)
                    ax.set_title(f'Trial {ti+1}', fontsize=8)
                for j in range(n_show, len(axes)):
                    axes[j].set_visible(False)
                fig.suptitle(
                    f'{participant_id} – {session_name}  |  {focus_channel}  '
                    f'per-trial TFR  [{group_label.upper()}]  |  baseline: {bl_name}',
                    fontsize=11, fontweight='bold'
                )
                fig.tight_layout()
                fname = (f'{participant_id}_{session_name}_{suffix}_'
                         f'TFR_per_trial_{focus_channel}_{group_label}_{bl_name}.png')
                fig.savefig(Path(output_dir) / fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f'      Saved per-trial TFR: {fname}')

                # habituation/drift for each band on focus channel
                for band_name, (b_low, b_high) in TFR_BANDS.items():
                    freq_mask = (freqs >= b_low) & (freqs <= b_high)
                    if not freq_mask.any():
                        continue
                    trial_bp = power_3d_focus[:, freq_mask, :][:, :, hab_start_idx:hab_end_idx].mean(axis=(1, 2))
                    _habituation_plot(
                        trial_amplitudes=trial_bp,
                        trial_numbers=np.arange(1, len(trial_bp) + 1),
                        ch_name=focus_channel, condition=group_label,
                        session_name=session_name, participant_id=participant_id,
                        output_dir=output_dir,
                        suffix=f'{suffix}_{bl_name}', kind=band_name
                    )

            # topomap of mean band power (0–1 s) per band
            if info_topo is not None:
                for band_name, (b_low, b_high) in TFR_BANDS.items():
                    freq_mask = (freqs >= b_low) & (freqs <= b_high)
                    if not freq_mask.any():
                        continue
                    topo_vals = {}
                    for ch in topo_chs:
                        key = (ch, group_label)
                        if key not in raw_power:
                            continue
                        p3d = apply_tfr_baseline(raw_power[key], bl_start, bl_end)
                        topo_vals[ch] = float(
                            p3d[:, freq_mask, :][:, :, hab_start_idx:hab_end_idx].mean()
                        )
                    vals_arr = np.array([topo_vals.get(ch, np.nan) for ch in topo_chs])
                    if np.all(np.isnan(vals_arr)):
                        continue
                    fig, ax = plt.subplots(figsize=(5, 4))
                    mne.viz.plot_topomap(
                        np.nan_to_num(vals_arr), info_topo,
                        axes=ax, show=False, cmap='RdBu_r',
                        vlim=(-np.nanpercentile(np.abs(vals_arr), 95),
                               np.nanpercentile(np.abs(vals_arr), 95)),
                    )
                    ax.set_title(
                        f'{group_label}  |  {band_name}  power (0–1 s)\n'
                        f'baseline: {bl_name}', fontsize=9
                    )
                    fig.tight_layout()
                    fname = (f'{participant_id}_{session_name}_{suffix}_'
                             f'TFR_topomap_{group_label}_{band_name}_{bl_name}.png')
                    fig.savefig(Path(output_dir) / fname, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f'      Saved TFR topomap: {fname}')

        # save the band-power CSV for this baseline
        if band_power_rows:
            bp_df = pd.DataFrame(band_power_rows)
            bp_csv = (Path(output_dir) /
                      f'{participant_id}_{session_name}_{suffix}_TFR_band_power_{bl_name}.csv')
            bp_df.to_csv(bp_csv, index=False)
            print(f'      Saved TFR band-power CSV: {bp_csv.name}')


# =============================================================================
# Feature assembly + session runner
# =============================================================================

def flatten_features(peak_freq, spindle_features, sw_features, power_features, pulse_results):
    out = {'individual_spindle_peak_hz': peak_freq}
    for ch_features in [spindle_features, sw_features]:
        for ch, feats in ch_features.items():
            out.update({f'{ch}_{name}': value for name, value in feats.items()})
    out.update(power_features)
    out['pulse_n_active'] = pulse_results.get('n_active', 0) if pulse_results else 0
    out['pulse_n_sham']   = pulse_results.get('n_sham',   0) if pulse_results else 0
    el = (pulse_results or {}).get('event_locked_spindles', {})
    out.update(el)
    return out


def process_one_session(participant_id, target, session_name, is_adaptation,
                         participant_output_dir):
    """
    Run full analysis for one session using the preprocessed .fif.

    Parameters
    ----------
    participant_id       : e.g. 'tunes-08'
    target               : 'adapt' | 'thalamus' | 'ventricle'
    session_name         : e.g. 'tunes-08_thalamus'  (used for file naming)
    is_adaptation        : bool
    participant_output_dir : results output folder
    """
    try:
        # Load preprocessed .fif 
        print(f'\n[1] Load preprocessed: {participant_id} / {target}')
        raw = load_preprocessed(participant_id, target)
        raw.load_data()   # pull into RAM once; from here it's at RESAMPLE_FREQ

        # Sleep staging 
        vhdr_files, _ = find_vhdr_for_staging(participant_id, target)
        hypno_int, hypno_str, staging_ok = run_sleep_staging(
            vhdr_files, session_name, participant_id, participant_output_dir
        )

        # QC plot 
        snap_p = snapshot_path(participant_id, target)
        snap_ch_p = snapshot_channels_path(participant_id, target)
        if snap_p.exists() and snap_ch_p.exists():
            snap_uv       = np.load(str(snap_p))
            snap_channels = list(np.load(str(snap_ch_p), allow_pickle=True))
            snap_secs     = snap_uv.shape[1] / raw.info['sfreq']
            safe_plot(plot_raw_vs_preprocessed,
                      snap_uv, raw, snap_channels, snap_secs,
                      session_name, participant_id, participant_output_dir)
            del snap_uv
            gc.collect()

        # Upsample hypno for YASA
        hypno_up = upsample_hypno(hypno_int, raw) if hypno_int is not None else None

        if is_adaptation:
            print('    Session role: ADAPTATION')
            peak_freq, freq_band = get_individual_spindle_frequency(
                raw, hypno_int, session_name, participant_id, participant_output_dir
            )
            del raw
            gc.collect()
            return {
                'participant_id': participant_id, 'session': session_name,
                'target': target,
                'session_role': 'adaptation_spindle_frequency_only',
                'analysis_scope': 'NREM_N2_N3' if hypno_int is not None else 'FULL_RECORDING_NO_STAGING',
                'is_adaptation': True,
                'individual_spindle_peak_hz': peak_freq,
                'spindle_band_low_hz': freq_band[0], 'spindle_band_high_hz': freq_band[1],
            }

        print(f'    Session role: EXPERIMENTAL TARGET ({target})')
        peak_freq, freq_band = load_individual_spindle_frequency(
            participant_id, participant_output_dir
        )

        # Analysis 
        spindle_features = detect_spindles(
            raw, hypno_int, hypno_up, freq_band, session_name, participant_id, participant_output_dir
        )
        sw_features = detect_slow_waves(
            raw, hypno_int, hypno_up, session_name, participant_id, participant_output_dir
        )
        power_features = compute_spectral_power(
            raw, hypno_int, session_name, participant_id, participant_output_dir
        )

        vmrk = find_vmrk(participant_id, target)
        pulse_results = run_pulse_level_analysis(
            raw, vmrk, hypno_int, hypno_up, freq_band,
            session_name, participant_id, is_adaptation, participant_output_dir
        )

        # Spectrogram while raw is still loaded
        safe_plot(plot_spectrogram, raw, hypno_int,
                  session_name, participant_id, participant_output_dir)

        del raw, hypno_up
        gc.collect()
        print('    Raw released — running CSV-based visualisations')

        # CSV-based visualisations 
        # plot_topoplots, plot_spindle_boxplots, plot_spindle_violins,
        # plot_erps, plot_tfrs etc. .

        row = flatten_features(peak_freq, spindle_features, sw_features,
                               power_features, pulse_results)
        row.update({
            'participant_id': participant_id, 'session': session_name, 'target': target,
            'session_role': 'experimental_target_analysis',
            'analysis_scope': 'NREM_N2_N3' if hypno_int is not None else 'FULL_RECORDING_NO_STAGING',
            'is_adaptation': False,
            'spindle_band_low_hz': freq_band[0], 'spindle_band_high_hz': freq_band[1],
        })
        return row

    except Exception as exc:
        print(f'\nSession failed: {exc}')
        traceback.print_exc()
        return None


def process_participant(participant_id):
    print('\n' + '=' * 70)
    print(f'PARTICIPANT: {participant_id}')
    print('=' * 70)

    out_dir = Path(OUTPUT_DIR) / participant_id
    out_dir.mkdir(parents=True, exist_ok=True)

    target_map = {
        'adapt':     ('adaptation', True),
        'thalamus':  ('thalamus',   False),
        'ventricle': ('ventricle',  False),
    }
    rows = []
    for target, (label, is_adaptation) in target_map.items():
        if not fif_path(participant_id, target).exists():
            print(f'  Skipping {target}: {fif_path(participant_id, target)} not found')
            print(f'  (Run preprocess.py first)')
            continue
        session_name = f'{participant_id}_{target}'
        row = process_one_session(
            participant_id, target, session_name, is_adaptation, str(out_dir)
        )
        if row:
            rows.append(row)

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f'{participant_id}_session_features.csv', index=False)
    print(f'\n  Feature table saved: {out_dir}')
    return df


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TUNES analysis pipeline')
    parser.add_argument('--participants', nargs='+', default=None)
    args = parser.parse_args()

    participants = args.participants if args.participants else PARTICIPANTS
    print(f'Analysing {len(participants)} participant(s): {participants}')

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    all_tables = []

    for pid in participants:
        df = process_participant(pid)
        if df is not None:
            all_tables.append(df)

    if all_tables:
        group_df   = pd.concat(all_tables, ignore_index=True)
        group_path = Path(OUTPUT_DIR) / 'all_session_features.csv'
        group_df.to_csv(group_path, index=False)
        print(f'\nGroup feature table → {group_path}')
        # build_response_profile / run_response_statistics / plot_response_profile
        # are unchanged — paste them here or import from a shared module.
    else:
        print('\nNo participant tables produced.')