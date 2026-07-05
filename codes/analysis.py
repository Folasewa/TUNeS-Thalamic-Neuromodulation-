"""
analysis.py — TUNES Stage 2: Preprocessed .fif → Features + Visualisations

What this script does
---------------------
Reads the preprocessed .fif files written by preprocess.py and runs the
full analysis pipeline

  1.  Sleep staging (YASA, 100 Hz 3-channel copy extracted from .fif; cached to CSV)
  2.  Individual spindle frequency (adaptation session)
  3.  Spindle detection (YASA)
  4.  Slow-wave detection (YASA)
  5.  Spectral band power
  6.  Burst-level (pulse-level) analysis + MNE Epochs .fif
  7.  Visualisations: raw-vs-preprocessed, spectrogram, topoplots,
      boxplots, violins, ERPs, TFRs

Note: sleep staging and all analyses are performed directly from the
preprocessed .fif files. No raw .vhdr files are ever loaded here.
The .vmrk file is still needed only for TUS burst/pulse marker parsing
(burst-level analysis), as those markers live in the original recording
file.  The original hardware sampling frequency — required to convert
.vmrk sample numbers to seconds — is read from the small JSON written by
preprocess.py ({target}_info.json), so no .vhdr access is needed either.
"""

import argparse
import gc
import time
import traceback
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import mne
import numpy as np
import pandas as pd
import yasa
from scipy.signal import welch
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.stats import wilcoxon, mannwhitneyu
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
from scipy.signal import hilbert
mne.set_log_level('WARNING')


# =============================================================================
# Settings — edit these paths
# =============================================================================
PREPROCESSED_DIR = '/home/e_fabdulsa/Desktop/TUNeS_sharbie/preprocessed'
# MARKERS_ROOT is the ONLY path to raw data still needed in analysis.py.
# It is used solely to locate the .vmrk marker files for TUS burst/pulse
# timing.  No .vhdr or EEG signal data are ever loaded from here.
MARKERS_ROOT     = '/home/e_fabdulsa/Desktop/TUNeS_sharbie/subjects'
OUTPUT_DIR       = '/home/e_fabdulsa/Desktop/TUNeS_sharbie/results'
PARTICIPANTS     = ['02', '03', '06', '08', '10']

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
CSV_ISPPA_MAP = {
    60.0: 'active_60w',
     1.0: 'sham_1isppa',
}
ACTIVE_CONDITIONS = {'active_60w'}
SHAM_CONDITIONS   = {'sham_1isppa'}
SHOW_BURST_OVERLAY = False
TUS_EPOCH_PRE_SEC  = 3.0
TUS_EPOCH_POST_SEC = 5.0
EPOCH_REJECT_UV = 250.0  # fixed absolute-amplitude rejection threshold (paper-style)

KNOWN_TARGETS = {'adapt', 'thalamus', 'ventricle', 'ventricles'}
EXCLUDE_CHANNELS = {'TP9', 'TP10', 'FT9', 'FT10'}

# =============================================================================
# Resume helper
# =============================================================================

def _already_done(output_dir, fname):
    """
    Return True (and print a skip message) if `fname` already exists in
    output_dir.  Use this at the top of every plot/CSV-writing function so
    that re-runs skip work that has already been completed.
    """
    p = Path(output_dir) / fname
    if p.exists():
        print(f'    [skip] already exists: {fname}')
        return True
    return False


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


def original_sfreq_path(participant_id, target):
    """Path to the JSON saved by preprocess.py that records the hardware sfreq."""
    return Path(PREPROCESSED_DIR) / participant_id / f'{target}_info.json'


def load_original_sfreq(participant_id, target, fallback=5000.0):
    """
    Read the original hardware sampling frequency saved by preprocess.py.
    Falls back to `fallback` (5000 Hz) if the file is absent so that old
    preprocessed datasets still work.
    """
    p = original_sfreq_path(participant_id, target)
    if p.exists():
        import json
        with open(p) as f:
            return float(json.load(f).get('original_sfreq', fallback))
    print(f'    Warning: {p.name} not found — assuming original sfreq = {fallback} Hz')
    return fallback


def find_vmrk(participant_id, target):
    """
    Locate ALL .vmrk marker files for a session.
    Returns a list of paths (sorted), or empty list if none found.
    This is the ONLY access to the raw subjects folder in analysis.py.
    """
    subject_folder = Path(MARKERS_ROOT) / participant_id
    for folder in sorted(subject_folder.iterdir()):
        if not folder.is_dir():
            continue
        text = folder.name.lower()
        if target.lower().replace('ventricle', 'vent') not in text and \
           target.lower() not in text:
            continue
        vmrk_files = sorted(folder.glob('*.vmrk'))
        if vmrk_files:
            return [str(p) for p in vmrk_files]
    return []


# statistics helpers
def _bh_fdr_correct(pvals):
    """
    Benjamini-Hochberg FDR correction.
    pvals : array-like of raw p-values (may contain np.nan)
    Returns corrected p-values in the same order, nan preserved.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    out = np.full(n, np.nan)
 
    valid_mask = ~np.isnan(pvals)
    valid_p = pvals[valid_mask]
    n_valid = len(valid_p)
    if n_valid == 0:
        return out
 
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    corrected = ranked * n_valid / (np.arange(n_valid) + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0, 1)
 
    corrected_full = np.empty(n_valid)
    corrected_full[order] = corrected
    out[valid_mask] = corrected_full
    return out


def _p_to_stars(p):
    if p is None or np.isnan(p):
        return ''
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


def _paired_wilcoxon(pre_vals, post_vals):
    """
    Paired Wilcoxon signed-rank test between pre and post values
    for the same trials/participant. Returns p-value or np.nan if
    the test cannot be run (too few pairs, all-zero differences, etc).
    """
    pre_vals = np.asarray(pre_vals, dtype=float)
    post_vals = np.asarray(post_vals, dtype=float)
    n = min(len(pre_vals), len(post_vals))
    if n < 3:
        return np.nan
    pre_vals = pre_vals[:n]
    post_vals = post_vals[:n]
    diffs = post_vals - pre_vals
    if np.all(diffs == 0):
        return np.nan
    try:
        _, p = wilcoxon(pre_vals, post_vals, zero_method='wilcox')
        return p
    except ValueError:
        return np.nan


def _unpaired_mannwhitney(vals_a, vals_b):
    """
    Mann-Whitney U test between two independent groups
    (e.g. sham vs active). Returns p-value or np.nan.
    """
    vals_a = np.asarray(vals_a, dtype=float)
    vals_b = np.asarray(vals_b, dtype=float)
    if len(vals_a) < 3 or len(vals_b) < 3:
        return np.nan
    try:
        _, p = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
        return p
    except ValueError:
        return np.nan


def _add_sig_bracket(ax, x1, x2, y, p_corrected, color='#222', fontsize=8,
                      bracket_frac=0.035):
    """
    Draw a significance bracket with stars (or 'ns') between x1 and x2
    at height y. Skips drawing entirely if p_corrected is nan (test
    could not be run).
    """
    if p_corrected is None or np.isnan(p_corrected):
        return
    stars = _p_to_stars(p_corrected)
    ylo, yhi = ax.get_ylim()
    h = (yhi - ylo) * bracket_frac
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=1.0, c=color, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, stars, ha='center', va='bottom',
            fontsize=fontsize, color=color, clip_on=False)


def _bracket_y(*value_arrays, pad_frac=0.10):
    """
    Compute a sensible y position for a significance bracket, sitting
    just above the highest data point (incl. whiskers) across the
    given arrays. Falls back to 0 if everything is empty.
    """
    all_vals = np.concatenate([np.asarray(v, dtype=float).ravel()
                                for v in value_arrays if len(v)])
    if len(all_vals) == 0:
        return 0.0, 1.0
    lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    return hi + span * pad_frac, span


# =============================================================================
# Sleep staging
# =============================================================================

def run_sleep_staging_from_fif(raw, session_name, participant_id, output_dir):
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

    if STAGING_EEG_CH not in raw.ch_names:
        print(f'    Missing required staging channel: {STAGING_EEG_CH} — skipping staging')
        return None, None, False

    staging_channels = [STAGING_EEG_CH, STAGING_EOG_CH, STAGING_EMG_CH]
    available = [ch for ch in staging_channels if ch in raw.ch_names]
    missing   = [ch for ch in staging_channels if ch not in raw.ch_names]
    if missing:
        print(f'    Note: optional staging channels not found (will proceed without): {missing}')

    raw_staging = raw.copy().pick_channels(available)
    raw_staging.resample(STAGING_FREQ)
    set_channel_types(raw_staging)

    mb = raw_staging.get_data().nbytes / 1e6
    print(f'    Staging raw: {raw_staging.ch_names} | '
          f'{raw_staging.info["sfreq"]:.0f} Hz | '
          f'{raw_staging.times[-1]/60:.1f} min | {mb:.1f} MB')

    kwargs = {'eeg_name': STAGING_EEG_CH}
    if STAGING_EOG_CH in raw_staging.ch_names:
        kwargs['eog_name'] = STAGING_EOG_CH
    if STAGING_EMG_CH in raw_staging.ch_names:
        kwargs['emg_name'] = STAGING_EMG_CH

    try:
        gc.collect()
        print('    Running YASA ...')
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
    eeg_all  = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude='bads')]
    channels = [ch for ch in (POWER_CHANNELS if POWER_CHANNELS else eeg_all)
                if ch in raw.ch_names]
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
def load_csv_condition_sequence(session_folder: str) -> list:
    folder = Path(session_folder)
    # Condition_matrix_*.csv files may live directly inside the session
    # folder, or nested inside a subfolder (e.g. "condition matrix",
    # "Condition_Matrix", etc). Search recursively and match by filename,
    # not by folder name, so we don't need to guess the subfolder's exact
    # name or casing.
    csv_files = sorted(
        p for p in folder.rglob('*.csv')
        if p.name.lower().startswith('condition_matrix')
    )
    if not csv_files:
        print(f'    [CSV] No Condition_matrix_*.csv found under {folder} (searched recursively)')
        return []
    print(f'    [CSV] Found {len(csv_files)} condition matrix file(s) under {folder}:')
    for p in csv_files:
        print(f'      {p.relative_to(folder)}')

    conditions = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, sep=',', encoding='latin-1', engine='python')
            isppa_cols = [c for c in df.columns if 'ISPPA' in c.upper()]
            if not isppa_cols:
                print(f'    [CSV] No ISPPA column in {csv_path.name} — skipping')
                continue
            isppa_col = isppa_cols[0]
            delivered = df[df[isppa_col].notna()]
            for val in delivered[isppa_col]:
                cond = CSV_ISPPA_MAP.get(float(val), 'unknown')
                conditions.append(cond)
        except Exception as exc:
            print(f'    [CSV] Could not read {csv_path.name}: {exc}')
    print(f'    [CSV path] {len(csv_files)} CSV files → {len(conditions)} delivered trials')
    return conditions


def _parse_vmrk_pulses(vmrk_path: str, original_sfreq: float):
    pulses            = []
    b_markers         = []
    current_condition = 'unknown'
    has_comments      = False
    with open(vmrk_path, 'r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if not line.startswith('Mk'):
                continue
            _, rest = line.strip().split('=', 1)
            parts = rest.split(',')
            if len(parts) < 3:
                continue
            marker_type = parts[0]
            label       = parts[1].strip()
            sample      = int(parts[2])
            if marker_type == 'Comment':
                text = label.lower()
                for keyword, condition in INTENSITY_COMMENTS.items():
                    if keyword in text:
                        current_condition = condition
                        # Only real intensity markers (60w/1w) indicate the
                        # old Comment-driven protocol. Generic setup comments
                        # ('stim', 'no stim', 'reprep', etc.) can appear in
                        # BOTH protocols and must not force PATH 1.
                        if condition not in ('ignore',):
                            has_comments = True
                        break
            elif marker_type == 'Stimulus' and label == 'B':
                b_markers.append({
                    'sample_original': sample,
                    'time_sec':        sample / original_sfreq,
                })
            elif marker_type == 'Stimulus' and label == TUS_MARKER_CODE:
                pulses.append({
                    'sample_original': sample,
                    'time_sec':        sample / original_sfreq,
                    'condition':       current_condition,
                })
    return pulses, b_markers, has_comments

def _group_into_bursts(df: pd.DataFrame, burst_gap_threshold: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values('sample_original').reset_index(drop=True)
    df['trigger_seq_all']       = np.arange(1, len(df) + 1)
    df['trigger_seq_condition'] = df.groupby('condition').cumcount() + 1
    df['gap_sec']               = df['time_sec'].diff()
    df['burst_id']              = (
        df['gap_sec'].isna() | (df['gap_sec'] > burst_gap_threshold)
    ).cumsum()
    burst_rows = []
    for burst_id, group in df.groupby('burst_id'):
        group = group.sort_values('sample_original')
        burst_rows.append({
            'burst_id':                    int(burst_id),
            'sample_original':             int(group['sample_original'].iloc[0]),
            'time_sec':                    float(group['time_sec'].iloc[0]),
            'condition':                   group['condition'].iloc[0],
            'n_pulses':                    len(group),
            'duration_sec':                float(
                group['time_sec'].iloc[-1] - group['time_sec'].iloc[0]
            ),
            'first_trigger_seq_all':       int(group['trigger_seq_all'].iloc[0]),
            'last_trigger_seq_all':        int(group['trigger_seq_all'].iloc[-1]),
            'first_trigger_seq_condition': int(group['trigger_seq_condition'].iloc[0]),
            'last_trigger_seq_condition':  int(group['trigger_seq_condition'].iloc[-1]),
        })
    bursts = pd.DataFrame(burst_rows)
    bursts.insert(0, 'burst_seq_all', np.arange(1, len(bursts) + 1))
    return bursts

def parse_tus_markers_bursts(vmrk_path: str,
                              original_sfreq: float,
                              burst_gap_threshold: float = 0.5,
                              session_folder: str = None) -> pd.DataFrame:
    if session_folder is None:
        session_folder = str(Path(vmrk_path).parent)

    pulses, b_markers, has_comments = _parse_vmrk_pulses(vmrk_path, original_sfreq)
    print(f'    [routing] has_comments={has_comments} | '
          f'{len(pulses)} A-pulses, {len(b_markers)} B-markers found')
    if not pulses:
        return pd.DataFrame()

    if has_comments:
        # ── PATH 1: Comment-marker protocol (old sessions) ────────────────
        print(f'    [parse_tus_markers_bursts] PATH 1: Comment markers found')
        df = pd.DataFrame(pulses)
        df = df[df['condition'] != 'unknown'].reset_index(drop=True)
        df = df[df['condition'] != 'ignore'].reset_index(drop=True)
        bursts = _group_into_bursts(df, burst_gap_threshold)

    else:
        # ── PATH 2: Raspberry Pi CSV fallback (new sessions) ──────────────
        print(f'    [parse_tus_markers_bursts] PATH 2: No Comment markers — '
              f'using Raspberry Pi CSV condition sequence')
        csv_conditions = load_csv_condition_sequence(session_folder)
        if not csv_conditions:
            print(f'    [CSV path] WARNING: no Condition_matrix CSV files found '
                  f'in {session_folder} — cannot assign conditions')
            return pd.DataFrame()

        b_markers_sorted = sorted(b_markers, key=lambda x: x['sample_original'])
        n_b   = len(b_markers_sorted)
        n_csv = len(csv_conditions)
        n_use = min(n_b, n_csv)
        if n_b != n_csv:
            print(f'    [CSV path] WARNING: {n_b} vmrk B-markers vs '
                  f'{n_csv} CSV delivered trials — using first {n_use} of each')

        df = pd.DataFrame(pulses)
        df = df.sort_values('sample_original').reset_index(drop=True)
        df['gap_sec']  = df['time_sec'].diff()
        df['burst_id'] = (
            df['gap_sec'].isna() | (df['gap_sec'] > burst_gap_threshold)
        ).cumsum()

        burst_rows = []
        cond_seq   = {}
        for burst_idx, (burst_id, group) in enumerate(df.groupby('burst_id')):
            if burst_idx >= n_use:
                break
            condition = csv_conditions[burst_idx]
            if condition == 'unknown':
                continue
            group = group.sort_values('sample_original')
            cond_seq[condition] = cond_seq.get(condition, 0) + 1
            burst_rows.append({
                'burst_id':                    int(burst_id),
                'sample_original':             int(group['sample_original'].iloc[0]),
                'time_sec':                    float(group['time_sec'].iloc[0]),
                'condition':                   condition,
                'n_pulses':                    len(group),
                'duration_sec':                float(
                    group['time_sec'].iloc[-1] - group['time_sec'].iloc[0]
                ),
                'first_trigger_seq_all':       int(group.index[0]  + 1),
                'last_trigger_seq_all':        int(group.index[-1] + 1),
                'first_trigger_seq_condition': cond_seq[condition],
                'last_trigger_seq_condition':  cond_seq[condition],
            })

        if not burst_rows:
            print('    [CSV path] No bursts matched — check CSV files and vmrk')
            return pd.DataFrame()

        bursts = pd.DataFrame(burst_rows)
        bursts.insert(0, 'burst_seq_all', np.arange(1, len(bursts) + 1))

    if bursts.empty or 'condition' not in bursts.columns:
        print('    No valid bursts after condition filtering — check .vmrk comments/CSV mapping')
        return pd.DataFrame()

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
        features[f'{ch}_pre_theta_power']    = round(band_power(pre,  sfreq, *FREQ_BANDS['theta']), 6)
        features[f'{ch}_post_theta_power']   = round(band_power(post, sfreq, *FREQ_BANDS['theta']), 6)
        features[f'{ch}_pre_alpha_power']    = round(band_power(pre,  sfreq, 8.0, 12.0), 6)
        features[f'{ch}_post_alpha_power']   = round(band_power(post, sfreq, 8.0, 12.0), 6)
    return features


def compute_event_locked_spindle_features(burst_times_by_group, spindle_summary,
                                           post_window_sec=5.0):
    if spindle_summary is None or spindle_summary.empty:
        return {}

    sp_deduped = spindle_summary.sort_values('Start').copy()
    sp_deduped['_event_bin'] = (sp_deduped['Start'] / 0.1).round().astype(int)
    sp_deduped = (
        sp_deduped
        .groupby('_event_bin', sort=False)
        .first()
        .reset_index(drop=True)
    )
    spindle_starts = sp_deduped['Start'].values

    out = {}
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
        matched     = sp_deduped.iloc[matched_idx]
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
# Burst-locked spindle characterisation CSV  ← NEW
# =============================================================================

def save_burst_locked_spindle_csv(burst_times_by_group, spindle_summary,
                                   session_name, participant_id, output_dir,
                                   suffix, post_window_sec=5.0):
    """
    Write a tidy CSV where every row is one burst, with columns describing
    the spindles that occurred in the post-stimulus window.

    Columns
    -------
    participant_id, session, condition (active/sham), burst_index,
    burst_time_s, post_window_sec,
    n_spindles_in_window,
    mean_spindle_amplitude_uv, mean_spindle_frequency_hz,
    mean_spindle_duration_sec, mean_spindle_rms_uv,
    spindle_density_per_s   (= n_spindles / post_window_sec)

    This is the file that lets you answer per-condition questions such as:
      - Did spindle density change in active vs sham?
      - Is spindle amplitude different post-burst in active vs sham?
      - How does duration / frequency / RMS compare?
    """
    fname = f'{participant_id}_{session_name}_{suffix}_burst_locked_spindles.csv'
    if _already_done(output_dir, fname):
        return

    if spindle_summary is None or spindle_summary.empty:
        print('    Burst-locked spindle CSV skipped: no spindle data')
        return

    # De-duplicate across channels (same logic as compute_event_locked_spindle_features)
    sp = spindle_summary.sort_values('Start').copy()
    sp['_event_bin'] = (sp['Start'] / 0.1).round().astype(int)
    sp = sp.groupby('_event_bin', sort=False).first().reset_index(drop=True)

    rows = []
    for condition_label, burst_times in burst_times_by_group.items():
        for burst_idx, bt in enumerate(burst_times):
            window_mask = (sp['Start'] > bt) & (sp['Start'] <= bt + post_window_sec)
            matched = sp[window_mask]
            n_sp    = len(matched)
            rows.append({
                'participant_id':           participant_id,
                'session':                  session_name,
                'condition':                condition_label,
                'burst_index':              burst_idx + 1,
                'burst_time_s':             round(bt, 4),
                'post_window_sec':          post_window_sec,
                'n_spindles_in_window':     n_sp,
                'mean_spindle_amplitude_uv':  round(matched['Amplitude'].mean(), 4) if n_sp else np.nan,
                'mean_spindle_frequency_hz':  round(matched['Frequency'].mean(), 4) if n_sp else np.nan,
                'mean_spindle_duration_sec':  round(matched['Duration'].mean(),  4) if n_sp else np.nan,
                'mean_spindle_rms_uv':        round(matched['RMS'].mean(),       4) if n_sp else np.nan,
                'spindle_density_per_s':      round(n_sp / post_window_sec,      4),
            })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(Path(output_dir) / fname, index=False)
        print(f'    Saved burst-locked spindle CSV: {fname}')
        # The summary is always (re)computed from the per-burst CSV so the
        # terminal output and the saved file stay in sync.
        _save_burst_locked_spindle_summary(df, session_name, participant_id,
                                           output_dir, suffix)


def _save_burst_locked_spindle_summary(burst_locked_df, session_name,
                                        participant_id, output_dir, suffix):
    """
    Collapse the per-burst spindle CSV into a tidy summary table:

        participant_id | session | target | condition | n_bursts
        | n_bursts_with_spindle | pct_bursts_with_spindle
        | total_spindles | mean_spindles_per_burst | mean_spindle_density_per_s
        | mean_amplitude_uv | sd_amplitude_uv
        | mean_frequency_hz | sd_frequency_hz
        | mean_duration_sec | sd_duration_sec
        | mean_rms_uv       | sd_rms_uv

    One row per condition (active / sham) — at a glance you can see whether
    any characteristic changed between conditions.
    """
    fname_summary = (f'{participant_id}_{session_name}_{suffix}'
                     f'_burst_locked_spindles_summary.csv')
    if _already_done(output_dir, fname_summary):
        return

    stat_cols = ['n_spindles_in_window', 'mean_spindle_amplitude_uv',
                 'mean_spindle_frequency_hz', 'mean_spindle_duration_sec',
                 'mean_spindle_rms_uv', 'spindle_density_per_s']

    rows = []
    for condition, grp in burst_locked_df.groupby('condition'):
        n_bursts          = len(grp)
        n_with_sp         = int((grp['n_spindles_in_window'] > 0).sum())
        pct_with_sp       = round(100 * n_with_sp / n_bursts, 1) if n_bursts else np.nan
        total_sp          = int(grp['n_spindles_in_window'].sum())
        mean_sp_per_burst = round(grp['n_spindles_in_window'].mean(), 4)

        # Amplitude / frequency / duration / RMS: mean only over bursts that
        # actually had spindles, otherwise the NaN rows skew the average.
        has_sp = grp[grp['n_spindles_in_window'] > 0]

        def _m(col): return round(has_sp[col].mean(), 4) if len(has_sp) else np.nan
        def _s(col): return round(has_sp[col].std(),  4) if len(has_sp) > 1 else np.nan

        rows.append({
            'participant_id':             participant_id,
            'session':                    session_name,
            'condition':                  condition,
            'n_bursts':                   n_bursts,
            'n_bursts_with_spindle':      n_with_sp,
            'pct_bursts_with_spindle':    pct_with_sp,
            'total_spindles_in_windows':  total_sp,
            'mean_spindles_per_burst':    mean_sp_per_burst,
            'mean_spindle_density_per_s': round(grp['spindle_density_per_s'].mean(), 4),
            'sd_spindle_density_per_s':   round(grp['spindle_density_per_s'].std(),  4),
            'mean_amplitude_uv':          _m('mean_spindle_amplitude_uv'),
            'sd_amplitude_uv':            _s('mean_spindle_amplitude_uv'),
            'mean_frequency_hz':          _m('mean_spindle_frequency_hz'),
            'sd_frequency_hz':            _s('mean_spindle_frequency_hz'),
            'mean_duration_sec':          _m('mean_spindle_duration_sec'),
            'sd_duration_sec':            _s('mean_spindle_duration_sec'),
            'mean_rms_uv':                _m('mean_spindle_rms_uv'),
            'sd_rms_uv':                  _s('mean_spindle_rms_uv'),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(Path(output_dir) / fname_summary, index=False)

    print(f'\n    ── Burst-locked spindle summary ({session_name}) ──')
    print(summary_df.to_string(index=False))
    print(f'    Saved: {fname_summary}\n')


# =============================================================================
# Slow-wave-locked sigma power time course (Thalamus vs Ventricle, active)
# =============================================================================

SW_LOCK_PRE_SEC, SW_LOCK_POST_SEC = 2.0, 2.0
SW_LOCK_BASELINE_SEC = (-2.0, -1.0)   # relative to SW trough, used for z-scoring

def _butter_filter(data, sfreq, low, high, order=4):
    nyq = sfreq / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)

def compute_evoked_band_responses(raw, bursts_df, session_name, participant_id,
                                   output_dir, suffix, hypno_int):
    """
    Paper-style evoked response analysis (per-participant, NREM only):
      - filters the CONTINUOUS recording to the SW band (0.1-4 Hz) and the
        spindle band (11-16 Hz, then Hilbert envelope) BEFORE epoching
      - epochs -3 to +5 s around each burst, N2/N3 trials only
      - baseline-corrects with PAPER_BASELINE_SEC
      - averages across clean trials per condition
      - extracts evoked SW peak-to-peak and evoked spindle magnitude
    """
    if hypno_int is None:
        print('    Evoked band responses skipped: NREM staging required')
        return

    fname = f'{participant_id}_{session_name}_{suffix}_evoked_band_metrics.csv'
    if _already_done(output_dir, fname):
        return

    if 'burst_time_s' not in bursts_df.columns:
        return
    bursts_df = bursts_df.copy()
    bursts_df['burst_time_s'] = pd.to_numeric(bursts_df['burst_time_s'], errors='coerce')
    bursts_df = bursts_df.dropna(subset=['burst_time_s'])
    if bursts_df.empty:
        return

    channels = [ch for ch in SPINDLE_CHANNELS + SW_CHANNELS if ch in raw.ch_names and ch not in raw.info['bads']]
    if not channels:
        return

    sfreq        = raw.info['sfreq']
    pre_samples  = int(TUS_EPOCH_PRE_SEC * sfreq)
    post_samples = int(TUS_EPOCH_POST_SEC * sfreq)
    times        = np.linspace(-TUS_EPOCH_PRE_SEC, TUS_EPOCH_POST_SEC,
                                pre_samples + post_samples)

    raw_data         = raw.get_data(picks=channels) * 1e6
    sw_filtered      = _butter_filter(raw_data, sfreq, *SW_EVOKED_BAND)
    spindle_filtered = _butter_filter(raw_data, sfreq, *SPINDLE_EVOKED_BAND)
    spindle_envelope = np.abs(hilbert(spindle_filtered, axis=-1))
    del raw_data, spindle_filtered
    gc.collect()

    rows = []
    for group_label, condition_set in [('sham', SHAM_CONDITIONS), ('active', ACTIVE_CONDITIONS)]:
        group_df = bursts_df[bursts_df['condition'].isin(condition_set)].reset_index(drop=True)

        for ch_idx, ch in enumerate(channels):
            sw_trials, sp_trials = [], []
            for _, burst in group_df.iterrows():
                t_sec = burst['burst_time_s']
                epoch_idx = min(int(t_sec / 30), len(hypno_int) - 1)
                if hypno_int[epoch_idx] not in NREM_STAGES:
                    continue
                center = int(t_sec * sfreq)
                start, stop = center - pre_samples, center + post_samples
                if start < 0 or stop > raw.n_times:
                    continue
                sw_trials.append(sw_filtered[ch_idx, start:stop])
                sp_trials.append(spindle_envelope[ch_idx, start:stop])

            if len(sw_trials) < 3:
                continue

            sw_trials = np.array(sw_trials)
            sp_trials = np.array(sp_trials)

            mask_clean, _ = _exclude_noisy_trials(sw_trials)
            sw_clean, sp_clean = sw_trials[mask_clean], sp_trials[mask_clean]
            if len(sw_clean) < 3:
                continue

            sw_evoked = _apply_erp_baseline(sw_clean, pre_samples, 'pre_mean', sfreq).mean(axis=0)
            sp_evoked = _apply_erp_baseline(sp_clean, pre_samples, 'pre_mean', sfreq).mean(axis=0)

            (w1_lo, w1_hi), (w2_lo, w2_hi) = SW_PTP_WINDOWS
            sw_ptp = float(
                sw_evoked[(times >= w1_lo) & (times <= w1_hi)].mean()
                - sw_evoked[(times >= w2_lo) & (times <= w2_hi)].mean()
            )
            sp_lo, sp_hi = SPINDLE_EVOKED_WINDOW
            sp_mag = float(sp_evoked[(times >= sp_lo) & (times <= sp_hi)].mean())

            rows.append({
                'participant_id': participant_id, 'session': session_name,
                'channel': ch, 'group': group_label, 'n_trials': len(sw_clean),
                'evoked_sw_ptp_uv': round(sw_ptp, 4),
                'evoked_spindle_magnitude_uv': round(sp_mag, 4),
            })

    if rows:
        pd.DataFrame(rows).to_csv(Path(output_dir) / fname, index=False)
        print(f'    Saved evoked band metrics: {fname}')

def compute_sw_locked_sigma_timecourse(raw, sw_starts_sec, burst_times_sec,
                                        freq_band, channels,
                                        pre_sec=SW_LOCK_PRE_SEC, post_sec=SW_LOCK_POST_SEC,
                                        near_burst_window=5.0):
    """
    Build a sigma-power (z-scored) time course locked to slow-wave troughs,
    restricted to slow waves occurring near a TUS burst (within
    `near_burst_window` seconds), so the timecourse reflects stimulation-
    proximal slow-wave activity rather than the whole recording.
    """
    if len(sw_starts_sec) == 0 or len(burst_times_sec) == 0:
        return None
    sfreq = raw.info['sfreq']
    pre_samples, post_samples = int(pre_sec * sfreq), int(post_sec * sfreq)
    n_samples = pre_samples + post_samples
    times_sec = np.linspace(-pre_sec, post_sec, n_samples)

    burst_times_sec = np.asarray(burst_times_sec)
    near_mask = np.array([
        np.any(np.abs(burst_times_sec - t) <= near_burst_window)
        for t in sw_starts_sec
    ])
    sw_times = np.asarray(sw_starts_sec)[near_mask]
    if len(sw_times) < 3:
        return None

    picks = [ch for ch in channels if ch in raw.ch_names]
    if not picks:
        return None

    data = raw.get_data(picks=picks) * 1e6
    nyq  = sfreq / 2.0
    b, a = butter(4, [freq_band[0] / nyq, freq_band[1] / nyq], btype='band')
    envelope = np.abs(hilbert(filtfilt(b, a, data, axis=1), axis=1))
    del data
    gc.collect()

    bl_s = pre_samples + int(SW_LOCK_BASELINE_SEC[0] * sfreq)
    bl_e = pre_samples + int(SW_LOCK_BASELINE_SEC[1] * sfreq)

    trials = []
    for t in sw_times:
        center = int(t * sfreq)
        start, stop = center - pre_samples, center + post_samples
        if start < 0 or stop > raw.n_times:
            continue
        trial_mean = envelope[:, start:stop].mean(axis=0)
        bl = trial_mean[max(bl_s, 0):max(bl_e, 0)]
        if len(bl) < 3 or bl.std() == 0:
            continue
        trials.append((trial_mean - bl.mean()) / bl.std())

    del envelope
    gc.collect()

    if len(trials) < 3:
        return None
    trials = np.array(trials)
    return times_sec, trials.mean(axis=0), trials.std(axis=0) / np.sqrt(len(trials)), len(trials)


def save_sw_locked_sigma_timecourse(raw, sw_summary, active_burst_times, freq_band,
                                     session_name, participant_id, target, output_dir):
    """
    Compute and cache the SW-locked sigma-power timecourse for the ACTIVE
    condition only, for a single target (thalamus/ventricle). Saved as .npz
    so the participant-level comparison plot can load both targets later.
    """
       
    fname = f'{participant_id}_{target}_active_sw_locked_sigma.npz'
    out_path = Path(output_dir) / fname
    if out_path.exists():
        print(f'    [skip] already exists: {fname}')
        return
    if sw_summary is None or sw_summary.empty or not active_burst_times:
        print('    SW-locked sigma timecourse skipped: missing SW or burst data')
        return

    result = compute_sw_locked_sigma_timecourse(
        raw, sw_summary['Start'].values, active_burst_times, freq_band, SPINDLE_CHANNELS
    )
    if result is None:
        print('    SW-locked sigma timecourse skipped: too few trials')
        return
    times_sec, mean_z, sem_z, n_trials = result
    np.savez(out_path, times_sec=times_sec, mean_z=mean_z, sem_z=sem_z, n_trials=n_trials)
    print(f'    Saved SW-locked sigma timecourse: {fname}  (n={n_trials} SWs)')

def plot_sw_locked_sigma_single_region(participant_id, target, output_dir):
    """
    Plots the SW-locked sigma-power timecourse for a SINGLE region
    (thalamus OR ventricle), independent of whether the other region's
    session exists. Useful when only one target has been recorded/processed
    for a given participant.
    """
    fname = f'{participant_id}_{target}_sw_locked_sigma_single.png'
    if _already_done(output_dir, fname):
        return

    npz_path = Path(output_dir) / f'{participant_id}_{target}_active_sw_locked_sigma.npz'
    if not npz_path.exists():
        print(f'    SW-locked sigma single-region plot skipped: no data for {target}')
        return

    d = np.load(npz_path)
    t, m, sem, n_trials = d['times_sec'], d['mean_z'], d['sem_z'], int(d['n_trials'])

    color = '#8E44AD' if target == 'thalamus' else '#16A085'
    label = f'{target.title()} (active)'

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(t, m - sem, m + sem, color=color, alpha=0.25)
    ax.plot(t, m, color=color, lw=2.2, label=f'{label} (n={n_trials})')
    ax.axvline(0, color='black', lw=1.0, ls='--', alpha=0.7, label='Slow wave trough')
    ax.axhline(0, color='grey', lw=0.6, ls=':')
    ax.set_xlabel('Time relative to slow wave (s)')
    ax.set_ylabel('Sigma power (z-score)')
    ax.set_title(
        f'{participant_id}: sigma power time course locked to slow waves\n'
        f'{target.title()} (active TUS)',
        fontsize=11, fontweight='bold'
    )
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / fname, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved SW-locked sigma single-region plot ({target}): {fname}')

def plot_sw_locked_sigma_thalamus_vs_ventricle(participant_id, output_dir):
    """
    Participant-level comparison: overlays the SW-locked sigma-power
    timecourse for Thalamus (active) vs Ventricle (active), loaded from the
    cached .npz files written by save_sw_locked_sigma_timecourse.
    """
    fname = f'{participant_id}_sw_locked_sigma_thalamus_vs_ventricle.png'
    if _already_done(output_dir, fname):
        return
    thal_path = Path(output_dir) / f'{participant_id}_thalamus_active_sw_locked_sigma.npz'
    vent_path = Path(output_dir) / f'{participant_id}_ventricle_active_sw_locked_sigma.npz'
    if not (thal_path.exists() and vent_path.exists()):
        print('    SW-locked sigma comparison skipped: need both targets computed first')
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for path, color, label in [
        (thal_path, '#8E44AD', 'Thalamus (active)'),
        (vent_path, '#16A085', 'Ventricle (active)'),
    ]:
        d = np.load(path)
        t, m, sem = d['times_sec'], d['mean_z'], d['sem_z']
        ax.fill_between(t, m - sem, m + sem, color=color, alpha=0.25)
        ax.plot(t, m, color=color, lw=2.0, label=f'{label} (n={int(d["n_trials"])})')
    ax.axvline(0, color='black', lw=1.0, ls='--', alpha=0.7, label='Slow wave trough')
    ax.axhline(0, color='grey', lw=0.6, ls=':')
    ax.set_xlabel('Time relative to slow wave (s)')
    ax.set_ylabel('Sigma power (z-score)')
    ax.set_title(
        f'{participant_id}: sigma power time course locked to slow waves\n'
        f'Thalamus vs Ventricle (active TUS)',
        fontsize=11, fontweight='bold'
    )
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / fname, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved SW-locked sigma comparison: {fname}')


# =============================================================================
# Burst-locked slow-wave characterisation CSV + region comparison
# =============================================================================

def save_burst_locked_slowwave_csv(burst_times_by_group, sw_summary,
                                    session_name, participant_id, output_dir,
                                    suffix, post_window_sec=5.0):
    """
    Mirrors save_burst_locked_spindle_csv, but for slow waves: one row per
    burst, describing slow waves occurring in the post-stimulus window.
    """
    fname = f'{participant_id}_{session_name}_{suffix}_burst_locked_slowwaves.csv'
    if _already_done(output_dir, fname):
        return
    if sw_summary is None or sw_summary.empty:
        print('    Burst-locked slow-wave CSV skipped: no SW data')
        return

    sw = sw_summary.sort_values('Start').copy()
    rows = []
    for condition_label, burst_times in burst_times_by_group.items():
        for burst_idx, bt in enumerate(burst_times):
            window_mask = (sw['Start'] > bt) & (sw['Start'] <= bt + post_window_sec)
            matched = sw[window_mask]
            n_sw = len(matched)
            rows.append({
                'participant_id':        participant_id,
                'session':               session_name,
                'condition':             condition_label,
                'burst_index':           burst_idx + 1,
                'burst_time_s':          round(bt, 4),
                'post_window_sec':       post_window_sec,
                'n_slowwaves_in_window': n_sw,
                'mean_sw_amplitude_uv':  round(matched['PTP'].mean(), 4) if n_sw else np.nan,
                'mean_sw_duration_sec':  round(matched['Duration'].mean(), 4)
                                          if n_sw and 'Duration' in matched else np.nan,
                'mean_sw_slope_uvs':     round(matched['Slope'].mean(), 4)
                                          if n_sw and 'Slope' in matched else np.nan,
                'sw_density_per_s':      round(n_sw / post_window_sec, 4),
            })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(Path(output_dir) / fname, index=False)
        print(f'    Saved burst-locked slow-wave CSV: {fname}')


def plot_slowwave_region_comparison(participant_id, output_dir):
    """
    Participant-level comparison of slow-wave properties: Thalamus vs
    Ventricle, Sham vs Active — with significance brackets for both
    Sham-vs-Active (within region) and Thalamus-vs-Ventricle (within
    condition), Mann-Whitney U, BH-FDR corrected across metrics.
    """
    fname = f'{participant_id}_slowwave_region_comparison.png'
    if _already_done(output_dir, fname):
        return

    dfs = []
    for target in ('thalamus', 'ventricle'):
        found = False
        for suffix in ('nrem', 'full_recording'):
            csv_path = (Path(output_dir) /
                        f'{participant_id}_{participant_id}_{target}_{suffix}_burst_locked_slowwaves.csv')
            if csv_path.exists():
                d = pd.read_csv(csv_path)
                d['target'] = target
                dfs.append(d)
                found = True
                break
        if not found:
            print(f'    Slow-wave region comparison: no burst-locked SW CSV for {target}')

    if len(dfs) < 2:
        print('    Slow-wave region comparison skipped: need both targets')
        return
    df = pd.concat(dfs, ignore_index=True)

    metrics = [
        ('n_slowwaves_in_window', 'N slow waves / window'),
        ('sw_density_per_s',      'SW density (per s)'),
        ('mean_sw_amplitude_uv',  'Mean amplitude (µV)'),
        ('mean_sw_duration_sec',  'Mean duration (s)'),
    ]
    groups        = [('thalamus', 'sham'), ('thalamus', 'active'),
                      ('ventricle', 'sham'), ('ventricle', 'active')]
    group_labels  = ['Thal\nSham', 'Thal\nActive', 'Vent\nSham', 'Vent\nActive']
    colors        = ['#A8C8E8', '#922B21', '#A9DFBF', '#7D3C98']

    available_metrics = [(c, l) for c, l in metrics if c in df.columns]
    if not available_metrics:
        print('    Slow-wave region comparison skipped: no metric columns found')
        return

    # --- Pass 1: raw p-values for all comparisons, all metrics, so FDR
    #     correction can run across metrics within each comparison family ---
    comparisons = [
        ('thal_sham_vs_active', ('thalamus', 'sham'),  ('thalamus', 'active')),
        ('vent_sham_vs_active', ('ventricle', 'sham'), ('ventricle', 'active')),
        ('sham_thal_vs_vent',   ('thalamus', 'sham'),  ('ventricle', 'sham')),
        ('active_thal_vs_vent', ('thalamus', 'active'),('ventricle', 'active')),
    ]
    raw_p = {name: [] for name, _, _ in comparisons}
    for col, _ in available_metrics:
        for name, (t1, c1), (t2, c2) in comparisons:
            v1 = df.loc[(df['target'] == t1) & (df['condition'] == c1), col].dropna().values
            v2 = df.loc[(df['target'] == t2) & (df['condition'] == c2), col].dropna().values
            raw_p[name].append(_unpaired_mannwhitney(v1, v2))
    corrected_p = {name: _bh_fdr_correct(vals) for name, vals in raw_p.items()}

    fig, axes = plt.subplots(1, len(available_metrics), figsize=(5.5 * len(available_metrics), 5.5))
    if len(available_metrics) == 1:
        axes = [axes]

    for m_idx, (ax, (col, label)) in enumerate(zip(axes, available_metrics)):
        data = [
            df.loc[(df['target'] == t) & (df['condition'] == c), col].dropna().values
            for t, c in groups
        ]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        medianprops=dict(color='white', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(group_labels, fontsize=8)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)

        # --- significance brackets ---
        y_base, _ = _bracket_y(*data)
        ylo, yhi = ax.get_ylim()
        step = (yhi - ylo) * 0.14

        # tier 1: sham vs active within each region
        _add_sig_bracket(ax, 1, 2, y_base, corrected_p['thal_sham_vs_active'][m_idx])
        _add_sig_bracket(ax, 3, 4, y_base, corrected_p['vent_sham_vs_active'][m_idx])

        # tier 2: thalamus vs ventricle within each condition (raised above tier 1)
        _add_sig_bracket(ax, 1, 3, y_base + step,       corrected_p['sham_thal_vs_vent'][m_idx])
        _add_sig_bracket(ax, 2, 4, y_base + step * 1.8, corrected_p['active_thal_vs_vent'][m_idx])

        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.15)

    fig.suptitle(
        f'{participant_id}: slow-wave properties — Thalamus vs Ventricle, Sham vs Active\n'
        f'Mann–Whitney U, BH–FDR corrected across metrics '
        f'(* p<0.05, ** p<0.01, *** p<0.001)',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(Path(output_dir) / fname, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved slow-wave region comparison: {fname}')

# =============================================================================
# Region comparison boxplots (Thalamus vs Ventricle, pre/post, sham/active)
# =============================================================================

def plot_region_comparison_boxplots(participant_id, output_dir):
    """
    Participant-level comparison across BOTH targets in one figure per band:
    for each channel, shows pre/post pairs for Thalamus-Sham, Thalamus-Active,
    Ventricle-Sham, Ventricle-Active side by side, with significance testing
    for: (1) pre vs post within each group [paired Wilcoxon], (2) sham vs
    active within each region [Mann-Whitney, post values], (3) thalamus vs
    ventricle within each condition [Mann-Whitney, post values].
    """
    dfs = []
    for target in ('thalamus', 'ventricle'):
        found = False
        for suffix in ('nrem', 'full_recording'):
            csv_path = (Path(output_dir) /
                        f'{participant_id}_{participant_id}_{target}_{suffix}_per_pulse_features.csv')
            if csv_path.exists():
                d = pd.read_csv(csv_path)
                d['target'] = target
                dfs.append(d)
                found = True
                break
        if not found:
            print(f'    Region comparison boxplots: no per-pulse CSV for {target}')

    if len(dfs) < 2:
        print('    Region comparison boxplots skipped: need both targets')
        return
    df = pd.concat(dfs, ignore_index=True)

    channels = sorted({
        c.split('_')[0] for c in df.columns if c.endswith('_pre_sigma_power')
    })
    if not channels:
        print('    Region comparison boxplots skipped: no channel columns found')
        return

    bands = [
        ('sigma_power', 'Sigma power'),
        ('delta_power', 'Delta power'),
        ('theta_power', 'Theta power'),
        ('alpha_power', 'Alpha power'),
    ]
    groups = [('thalamus', 'sham'), ('thalamus', 'active'),
              ('ventricle', 'sham'), ('ventricle', 'active')]

    for band_key, band_label in bands:
        band_fname = f'{participant_id}_region_comparison_{band_key}_prepost.png'
        if _already_done(output_dir, band_fname):
            continue

        valid_channels = [
            ch for ch in channels
            if f'{ch}_pre_{band_key}' in df.columns and f'{ch}_post_{band_key}' in df.columns
        ]
        if not valid_channels:
            continue

        # --- Pass 1: raw p-values for all comparison families, across all
        #     channels, for FDR correction within each family ---
        prepost_raw      = {g: [] for g in groups}
        sham_v_act_raw   = {'thalamus': [], 'ventricle': []}
        thal_v_vent_raw  = {'sham': [], 'active': []}

        for ch in valid_channels:
            pre_col, post_col = f'{ch}_pre_{band_key}', f'{ch}_post_{band_key}'
            post_by_group = {}
            for target, condition in groups:
                mask = (df['target'] == target) & (df['group'] == condition)
                pair_df = df.loc[mask, [pre_col, post_col]].dropna()
                prepost_raw[(target, condition)].append(
                    _paired_wilcoxon(pair_df[pre_col].values, pair_df[post_col].values)
                )
                post_by_group[(target, condition)] = df.loc[mask, post_col].dropna().values

            for target in ('thalamus', 'ventricle'):
                sham_v_act_raw[target].append(_unpaired_mannwhitney(
                    post_by_group[(target, 'sham')], post_by_group[(target, 'active')]
                ))
            for condition in ('sham', 'active'):
                thal_v_vent_raw[condition].append(_unpaired_mannwhitney(
                    post_by_group[('thalamus', condition)], post_by_group[('ventricle', condition)]
                ))

        prepost_corr     = {g: _bh_fdr_correct(v) for g, v in prepost_raw.items()}
        sham_v_act_corr  = {t: _bh_fdr_correct(v) for t, v in sham_v_act_raw.items()}
        thal_v_vent_corr = {c: _bh_fdr_correct(v) for c, v in thal_v_vent_raw.items()}

        fig, axes = plt.subplots(len(valid_channels), 1,
                                 figsize=(13, 3.8 * len(valid_channels)), squeeze=False)

        for row_idx, ch in enumerate(valid_channels):
            ax = axes[row_idx][0]
            pre_col, post_col = f'{ch}_pre_{band_key}', f'{ch}_post_{band_key}'

            positions, data, colors, labels, pos = [], [], [], [], 1
            group_positions = {}
            for target, condition in groups:
                mask = (df['target'] == target) & (df['group'] == condition)
                data += [df.loc[mask, pre_col].dropna().values,
                         df.loc[mask, post_col].dropna().values]
                positions += [pos, pos + 1]
                group_positions[(target, condition)] = (pos, pos + 1)
                colors += (['#AAB7C4', '#2C3E50'] if condition == 'sham'
                           else ['#F1948A', '#922B21'])
                labels += [f'{target[:4].title()}\n{condition.title()}\nPre',
                           f'{target[:4].title()}\n{condition.title()}\nPost']
                pos += 3

            bp = ax.boxplot(data, positions=positions, widths=0.8, patch_artist=True,
                            medianprops=dict(color='white', linewidth=1.8))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.88)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=6)
            ax.axhline(0, color='grey', lw=0.6, ls='--', alpha=0.5)
            ax.set_ylabel(ch, fontsize=9, fontweight='bold')
            ax.spines[['top', 'right']].set_visible(False)

            # --- significance brackets ---
            y_base, _ = _bracket_y(*data)
            ylo, yhi = ax.get_ylim()
            step = (yhi - ylo) * 0.16

            # tier 0: pre vs post within each group
            for target, condition in groups:
                x1, x2 = group_positions[(target, condition)]
                _add_sig_bracket(ax, x1, x2, y_base, prepost_corr[(target, condition)][row_idx])

            thal_sham_post_x = group_positions[('thalamus', 'sham')][1]
            thal_act_post_x  = group_positions[('thalamus', 'active')][1]
            vent_sham_post_x = group_positions[('ventricle', 'sham')][1]
            vent_act_post_x  = group_positions[('ventricle', 'active')][1]

            # tier 1: sham vs active within region (post vs post)
            _add_sig_bracket(ax, thal_sham_post_x, thal_act_post_x, y_base + step,
                             sham_v_act_corr['thalamus'][row_idx])
            _add_sig_bracket(ax, vent_sham_post_x, vent_act_post_x, y_base + step,
                             sham_v_act_corr['ventricle'][row_idx])

            # tier 2: thalamus vs ventricle within condition (post vs post)
            _add_sig_bracket(ax, thal_sham_post_x, vent_sham_post_x, y_base + step * 2.2,
                             thal_v_vent_corr['sham'][row_idx])
            _add_sig_bracket(ax, thal_act_post_x, vent_act_post_x, y_base + step * 3.0,
                             thal_v_vent_corr['active'][row_idx])

            ylo, yhi = ax.get_ylim()
            ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.20)

        fig.suptitle(
            f'{participant_id}: {band_label} pre/post — Thalamus vs Ventricle, Sham vs Active\n'
            f'Wilcoxon (pre/post) & Mann–Whitney U (sham/active, thal/vent), '
            f'BH–FDR corrected across channels\n'
            f'(* p<0.05, ** p<0.01, *** p<0.001)',
            fontsize=11, fontweight='bold'
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(Path(output_dir) / band_fname, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved region comparison ({band_label}): {band_fname}')


# =============================================================================
# Burst-level analysis
# =============================================================================

def run_pulse_level_analysis(raw, vmrk_path, hypno_int, hypno_up,
                              freq_band, session_name, participant_id,
                              target, is_adaptation, output_dir):
    if is_adaptation or not vmrk_path:
        return {}
    print(f'\n[8] Burst-level analysis: {participant_id} / {session_name}')
    original_sfreq = load_original_sfreq(participant_id, target)
    # NEW
    all_bursts = []
    for vp in vmrk_path:   # vmrk_path is now a list
        b = parse_tus_markers_bursts(vp, original_sfreq,
                                    session_folder=str(Path(vp).parent))
        if not b.empty:
            all_bursts.append(b)
    if not all_bursts:
        print('    No bursts found')
        return {}
    bursts = pd.concat(all_bursts, ignore_index=True)
    bursts['burst_seq_all'] = np.arange(1, len(bursts) + 1)
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
    sw_summary_full = None
    if sw_channels:
        sw_obj = yasa.sw_detect(raw, ch_names=sw_channels, freq_sw=SW_FREQ,
                                hypno=hypno_up, include=NREM_STAGES)
        if sw_obj is not None:
            sw_summary_full = sw_obj.summary()
            slowwave_starts_sec = sw_summary_full['Start'].values

    counts = {'total': len(bursts), 'skipped_condition': 0, 'skipped_bounds': 0,
              'skipped_nrem': 0, 'skipped_spindle': 0, 'kept_active': 0, 'kept_sham': 0, 'skipped_bad_segment': 0}
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
        if window_overlaps_bad_annotation(raw, start / sfreq, stop / sfreq):
            counts['skipped_bad_segment'] = counts.get('skipped_bad_segment', 0) + 1
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

    # NEW: write per-burst spindle characterisation CSV
    save_burst_locked_spindle_csv(burst_times_by_group, sp_summary,session_name, participant_id, output_dir, suffix,post_window_sec=TUS_EPOCH_POST_SEC,)
    save_sw_locked_sigma_timecourse(raw, sw_summary_full, burst_times_by_group.get('active', []), freq_band,session_name, participant_id, target, output_dir)
    save_burst_locked_slowwave_csv(burst_times_by_group, sw_summary_full,session_name, participant_id, output_dir, suffix, post_window_sec=TUS_EPOCH_POST_SEC)
    # Save MNE Epochs
    try:
        eeg_all     = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude='bads')]
        raw_epo     = raw.copy().pick_channels(eeg_all) if eeg_all else raw.copy()
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
                valid_mask   = [row['burst_time_s'] * epo_sfreq < raw_epo.n_times
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


def safe_plot_returning(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f'    Plot failed ({fn.__name__}): {e}')
        return None
    finally:
        plt.close('all')
        gc.collect()


def plot_raw_vs_preprocessed(raw_snapshot_uv, raw_post, channels,
                              snapshot_secs, session_name, participant_id, output_dir):
    fname = f'{participant_id}_{session_name}_raw_vs_preprocessed.png'
    if _already_done(output_dir, fname):
        return

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
    fig.savefig(Path(output_dir) / fname, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved raw vs preprocessed')


def plot_spectrogram(raw, hypno_int, session_name, participant_id, output_dir,
                     burst_times_sec=None):
    fname = f'{participant_id}_{session_name}_spectrogram.png'
    if _already_done(output_dir, fname):
        return
 
    channels = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude='bads')]
    if not channels:
        return
    
    sfreq = raw.info['sfreq']
    n_fft = int(sfreq * 4)
    hop   = int(sfreq * 2)
    n_ch  = len(channels)

    # grid dimensions
    ncols = 6
    nrows = int(np.ceil(n_ch / ncols))

    # --- compute a single global colour scale across all channels ---
    print('    Computing global spectrogram scale ...')
    _all_db = []
    for ch in channels:
        data = raw.get_data(picks=[ch])[0]

        freqs_tmp, _, Sxx_tmp = scipy_spectrogram(
            data,
            fs=sfreq,
            nperseg=n_fft,
            noverlap=n_fft - hop,
            scaling='density'
        )

        fmask = freqs_tmp <= 30.0
        Sxx_db = 10 * np.log10(Sxx_tmp[fmask] + 1e-30)
        _all_db.append(Sxx_db.ravel())
        del data, Sxx_tmp, Sxx_db
    _all_db_flat = np.concatenate(_all_db)
    global_vmin  = float(np.percentile(_all_db_flat, 5))
    global_vmax  = float(np.percentile(_all_db_flat, 98))
    del _all_db, _all_db_flat
    gc.collect()
    print(f'    Global scale: {global_vmin:.1f} – {global_vmax:.1f} dB/Hz')

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.5, nrows * 2.8),
        sharex=True, sharey=True,
    )
    axes_flat = np.array(axes).ravel()

    last_pcm = None
    for idx, ch in enumerate(channels):
        ax   = axes_flat[idx]
        data = raw.get_data(picks=[ch])[0]
        freqs, times, Sxx = scipy_spectrogram(
            data, fs=sfreq, nperseg=n_fft, noverlap=n_fft - hop, scaling='density'
        )
        fmask  = freqs <= 30.0
        Sxx_db = 10 * np.log10(Sxx[fmask] + 1e-30)
        pcm = ax.pcolormesh(
            times / 60, freqs[fmask], Sxx_db,
            cmap='inferno', shading='gouraud',
            vmin=global_vmin, vmax=global_vmax,
        )
        last_pcm = pcm
        del data, Sxx, Sxx_db
        gc.collect()

        if hypno_int is not None:
            for ei, stage in enumerate(hypno_int):
                if stage in NREM_STAGES:
                    ax.axvspan(ei * 30 / 60, (ei + 1) * 30 / 60,
                               color='cyan', alpha=0.10)

        # NEW: mark TUS burst onsets to check whether warm stripes are burst-locked
        if SHOW_BURST_OVERLAY and burst_times_sec is not None and len(burst_times_sec):
             for bt in burst_times_sec:    
                ax.axvline(bt / 60, color='lime', lw=0.4, alpha=0.5, zorder=10)
 
        ax.axhline(SPINDLE_FREQ_DEFAULT[0], color='white', lw=0.7, ls='--', alpha=0.55)
        ax.axhline(SPINDLE_FREQ_DEFAULT[1], color='white', lw=0.7, ls='--', alpha=0.55)
        ax.set_title(ch, fontsize=8)
        ax.tick_params(labelsize=6)
 
        # y-label only on leftmost column
        if idx % ncols == 0:
            ax.set_ylabel('Hz', fontsize=7)
        # x-label only on bottom row
        if idx >= ncols * (nrows - 1):
            ax.set_xlabel('Time (min)', fontsize=7)
 
    # Hide unused axes
    for idx in range(n_ch, nrows * ncols):
        axes_flat[idx].set_visible(False)

    # Legend note for the burst overlay
    title_suffix = '  [cyan = NREM'
    if SHOW_BURST_OVERLAY and burst_times_sec is not None and len(burst_times_sec):
        title_suffix += ' | lime = TUS burst onset'
    title_suffix += ']'

    # One shared colorbar for the whole figure
    fig.suptitle(f'{participant_id} – {session_name}: spectrogram{title_suffix}',
                 fontsize=12, fontweight='bold',)
    fig.subplots_adjust(top=0.93, right=0.88)
    if last_pcm is not None:
        cax = fig.add_axes([0.90, 0.15, 0.015, 0.70])
        fig.colorbar(last_pcm, cax=cax, label='dB/Hz')
    fig.savefig(Path(output_dir) / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('    Saved spectrogram')


# =============================================================================
# ERP and TFR visualisations
# =============================================================================

ERP_BASELINES = {
    'none':       None,
    'pre_mean':   'pre_mean',
    'pre_zscore': 'pre_zscore',
}
PAPER_BASELINE_SEC = (-2.5, -0.5)   # baseline window relative to onset; # excludes the final 500 ms to avoid absorbing pre-stimulus endogenous coupling
TFR_BASELINES = {
    'tight_300_50ms':  (-0.30, -0.05),
    'tight_500_100ms': (-0.50, -0.10),
    'full_pre':        (-TUS_EPOCH_PRE_SEC, -0.5),
}
ERP_TOPO_TIMECOURSE_PRE_MS     = 300   # start of window (ms before TUS onset)
ERP_TOPO_TIMECOURSE_POST_MS    = 500   # end of window (ms after TUS onset)
ERP_TOPO_TIMECOURSE_STEP_MS    = 100   # spacing between topomap frames
ERP_TOPO_TIMECOURSE_HALFWIN_MS = 25    # ± window averaged into each frame's value

N_BEST_CHANNELS          = 3
FOCUS_CHANNEL_PRIORITY = ['C4', 'C3', 'Cz']

HABITUATION_WINDOW_SEC   = (0.0, 1.0)
TFR_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'sigma': (12.0, 15.0),
    'beta':  (15.0, 30.0),
}
SW_EVOKED_BAND        = (0.1, 4.0)     # evoked SW analysis band
SPINDLE_EVOKED_BAND   = (11.0, 16.0)   # evoked spindle analysis band
SW_PTP_WINDOWS        = ((0.5, 0.6), (0.8, 1.0))   # (early, late) windows, sec post-onset
SPINDLE_EVOKED_WINDOW = (0.75, 1.5)                # sec post-onset

def _apply_erp_baseline(epochs_2d, pre_samples, mode, sfreq):
    """
    Baseline-correct using PAPER_BASELINE_SEC (-2.5 to -0.5 s relative to
    onset), deliberately excluding the final 500 ms before onset.
    """
    out = epochs_2d.copy()
    bl_start = max(pre_samples + int(PAPER_BASELINE_SEC[0] * sfreq), 0)
    bl_end   = max(pre_samples + int(PAPER_BASELINE_SEC[1] * sfreq), bl_start + 1)
    pre = epochs_2d[:, bl_start:bl_end]
    if mode == 'pre_mean':
        out = out - pre.mean(axis=1, keepdims=True)
    elif mode == 'pre_zscore':
        mu  = pre.mean(axis=1, keepdims=True)
        sd  = pre.std(axis=1, keepdims=True) + 1e-12
        out = (out - mu) / sd
    return out

def window_overlaps_bad_annotation(raw, start_sec, stop_sec):
    """True if [start_sec, stop_sec] overlaps any BAD_* annotation."""
    for ann in raw.annotations:
        if not ann['description'].startswith('BAD'):
            continue
        ann_start = ann['onset']
        ann_end   = ann['onset'] + ann['duration']
        if start_sec < ann_end and stop_sec > ann_start:
            return True
    return False


def _exclude_noisy_trials(epochs_2d, sd_multiplier=None):
    """
    Paper-style rejection: demean each epoch, then reject trials whose
    absolute amplitude exceeds EPOCH_REJECT_UV anywhere in the epoch.
    """
    demeaned  = epochs_2d - epochs_2d.mean(axis=1, keepdims=True)
    max_abs   = np.max(np.abs(demeaned), axis=1)
    mask      = max_abs <= EPOCH_REJECT_UV
    n_excluded = int((~mask).sum())
    if n_excluded:
        print(f'      Noise exclusion: removed {n_excluded} / {len(mask)} trials '
              f'[fixed threshold: ±{EPOCH_REJECT_UV} µV]')
    return mask, float(EPOCH_REJECT_UV)

def _rank_channels_by_erp(mean_erps, ch_names, post_start_idx):
    scores = {}
    for ch, erp in zip(ch_names, mean_erps):
        scores[ch] = np.sqrt(np.nanmean(erp[post_start_idx:] ** 2))
    return sorted(scores, key=scores.get, reverse=True), scores

def _plot_erp_topo_overlay(
    mean_erps_active, mean_erps_sham,
    channels, times,
    session_name, participant_id,
    output_dir, suffix, baseline_name,
    t_min=-0.5, t_max=1.0,
):
    """
    EEGLAB-style topo layout using MNE's native layout engine.
    Overlays active and sham ERP traces per channel at their correct scalp locations.
    """
    fname = (f'{participant_id}_{session_name}_{suffix}_'
             f'ERP_topo_overlay_{baseline_name}.png')
    if _already_done(output_dir, fname):
        return

    # 1. Create standard montage and filter valid channels
    montage = mne.channels.make_standard_montage('standard_1020')
    pos_dict = montage._get_ch_pos()
    valid_chs = [ch for ch in channels if ch in pos_dict]
    
    if len(valid_chs) < 3:
        print(f'      Topo overlay skipped: too few channels with known positions')
        return

    # 2. Generate MNE native layout
    info = mne.create_info(ch_names=valid_chs, sfreq=1000, ch_types='eeg')
    info.set_montage(montage)
    layout = mne.channels.find_layout(info, ch_type='eeg')

    # 3. Time masking and color setups
    time_mask = (times >= t_min) & (times <= t_max)
    times_zoom = times[time_mask]

    color_active = '#E04B4B'
    color_sham   = '#4B7BE0'

    # 4. Compute a shared y-scale across all channels
    all_vals = []
    for ch in valid_chs:
        idx = channels.index(ch)
        a = mean_erps_active[idx][time_mask]
        s = mean_erps_sham[idx][time_mask]
        if not np.all(np.isnan(a)): 
            all_vals.extend(a[np.isfinite(a)])
        if not np.all(np.isnan(s)): 
            all_vals.extend(s[np.isfinite(s)])
    if not all_vals:
        return
    y_abs = np.percentile(np.abs(all_vals), 95)
    ylim  = (-y_abs, y_abs)

    # 5. Initialize figure
    fig = plt.figure(figsize=(26, 20))

    # Separate x/y scale factors: shrink height MORE than width so each box
    # has internal headroom for the channel label without needing space
    # above the axis (which is what caused labels to be overwritten by
    # neighboring subplots).
    box_scale_x = 0.86
    box_scale_y = 0.78

    # 6. Plot each channel using layout geometries
    for ch, pos in zip(layout.names, layout.pos):
        x_pos, y_pos, width, height = pos

        new_w = width * box_scale_x
        new_h = height * box_scale_y
        new_x = x_pos + (width - new_w) / 2
        new_y = y_pos + (height - new_h) / 2

        ax = fig.add_axes([new_x * 0.84 + 0.08, new_y * 0.84 + 0.08, new_w * 0.84, new_h * 0.84])

        idx = channels.index(ch)
        erp_a = mean_erps_active[idx][time_mask]
        erp_s = mean_erps_sham[idx][time_mask]

        if not np.all(np.isnan(erp_s)):
            ax.plot(times_zoom, erp_s, color=color_sham,   lw=0.8, alpha=0.9)
        if not np.all(np.isnan(erp_a)):
            ax.plot(times_zoom, erp_a, color=color_active, lw=0.8, alpha=0.9)

        # Inner axes markings
        ax.axvline(0,  color='black', lw=0.5, ls='--', alpha=0.4)
        ax.axhline(0,  color='grey',  lw=0.4, ls=':',  alpha=0.4)
        ax.set_xlim(t_min, t_max)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        # ── FIX: channel label placed INSIDE the axis (not as a title
        # above it), with clip_on=True so it can never overlap or be
        # overwritten by a neighboring subplot, and a light background
        # box so it stays legible over the traces.
        ax.text(
            0.5, 0.98, ch,
            transform=ax.transAxes, ha='center', va='top',
            fontsize=7.5, fontweight='bold', color='#222222',
            clip_on=True,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.72, pad=1.0),
        )

    # 7. Shared legend
    legend_elements = [
        Line2D([0], [0], color=color_active, lw=1.5, label='Active'),
        Line2D([0], [0], color=color_sham,   lw=1.5, label='Sham'),
        Line2D([0], [0], color='black', lw=0.8, ls='--', label='TUS onset'),
    ]
    fig.legend(handles=legend_elements, loc='lower right',
               fontsize=10, framealpha=0.9)

    # 8. Shared scale bar (bottom left)
    scale_ax = fig.add_axes([0.05, 0.05, 0.07, 0.05])
    scale_ax.set_xlim(t_min, t_max)
    scale_ax.set_ylim(ylim)
    scale_ax.axvline(0, color='black', lw=0.6, ls='--', alpha=0.6)
    scale_ax.set_xlabel(f'{t_min}–{t_max} s', fontsize=8)
    scale_ax.set_ylabel(f'±{y_abs:.1f} µV', fontsize=8)
    scale_ax.set_xticks([t_min, 0, t_max])
    scale_ax.set_xticklabels([str(t_min), '0', str(t_max)], fontsize=7)
    scale_ax.set_yticks([])
    for spine in scale_ax.spines.values():
        spine.set_linewidth(0.5)

    # Title styling
    fig.suptitle(
        f'{participant_id} – {session_name}  |  ERP topo layout  '
        f'[baseline: {baseline_name}]\n'
        f'Active (red) vs Sham (blue)  |  window: {t_min} to {t_max} s',
        fontsize=6.5, fontweight='bold', y=0.96
    )
    
    fig.savefig(Path(output_dir) / fname, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'      Saved ERP topo overlay: {fname}')


def _habituation_plot(trial_amplitudes, trial_numbers, ch_name, condition,
                      session_name, participant_id, output_dir, suffix, kind):
    """Single-channel habituation plot used by TFR band-power analysis."""
    from scipy.stats import linregress
    fname = (f'{participant_id}_{session_name}_{suffix}_'
             f'habituation_{kind}_{ch_name}_{condition}.png')
    if _already_done(output_dir, fname):
        return
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
    fig.savefig(Path(output_dir) / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'      Saved habituation plot: {fname}')


def _habituation_plot_all_channels(all_epochs, channels, pre_samples, baseline_mode, hab_start, hab_end,
                                    clean_trials_by_channel, condition, session_name, participant_id,output_dir, suffix, kind,best_channel=None):
 
    fname = (f'{participant_id}_{session_name}_{suffix}_'
             f'habituation_{kind}_{condition}_all_channels.png')
    if _already_done(output_dir, fname):
        return
 
    n_ch  = len(channels)
    ncols = 4                                     
    nrows = int(np.ceil(n_ch / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.2, nrows * 2.8),
        squeeze=False,
    )
 
    ylabel = 'Mean amplitude (µV)' if kind == 'ERP' else f'Mean {kind} power (dB)'
 
    for idx, ch in enumerate(channels):
        row_i, col_i = divmod(idx, ncols)
        ax = axes[row_i][col_i]
 
        clean = clean_trials_by_channel[idx]
        hab_amps   = np.array([t[hab_start:hab_end].mean() for t in clean])
        trial_nums = np.arange(1, len(hab_amps) + 1)
 
        valid = ~np.isnan(hab_amps)
        x, y  = trial_nums[valid], hab_amps[valid]
 
        is_best = (ch == best_channel)
        scatter_color = '#C0392B' if is_best else '#2C7BB6'
        line_color    = '#922B21' if is_best else '#1A5276'
 
        if len(x) >= 3:
            slope, intercept, r, p, _ = linregress(x, y)
            ax.scatter(x, y, color=scatter_color, s=12, alpha=0.6, zorder=3,
                       linewidths=0)
            ax.plot(x, slope * x + intercept, color=line_color, lw=1.4,
                    label=f'R²={r**2:.2f}  p={p:.3f}')
            ax.legend(fontsize=6, loc='upper right', framealpha=0.7,
                      borderpad=0.3, handlelength=1.2)
        else:
            ax.scatter(x, y, color=scatter_color, s=12, alpha=0.6, zorder=3,
                       linewidths=0)
            ax.text(0.5, 0.5, 'too few trials', transform=ax.transAxes,
                    ha='center', va='center', color='grey', fontsize=7)
 
        ax.axhline(0, color='grey', lw=0.6, ls='--', alpha=0.5)
 
        # Axis labels only on edges
        if col_i == 0:
            ax.set_ylabel(ylabel, fontsize=7, labelpad=2)
        if row_i == nrows - 1:
            ax.set_xlabel('Trial number', fontsize=7, labelpad=2)
 
        ax.tick_params(labelsize=6, length=3, pad=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
 
        title_str = f'★ {ch}' if is_best else ch
        ax.set_title(title_str, fontsize=8,
                     fontweight='bold' if is_best else 'normal',
                     color='#C0392B' if is_best else 'black',
                     pad=3)
 
    # Hide unused axes
    for idx in range(n_ch, nrows * ncols):
        row_i, col_i = divmod(idx, ncols)
        axes[row_i][col_i].set_visible(False)
 
    best_note = f'  |  ★ = {best_channel}' if best_channel else ''
    condition_label = condition.upper()
    fig.suptitle(
        f'{participant_id} – {session_name}  |  {kind} habituation  '
        f'[{condition_label}]{best_note}',
        fontsize=10, fontweight='bold', y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(Path(output_dir) / fname, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'      Saved habituation plot (all channels): {fname}')


def _erp_topomap(mean_amp_by_channel, ch_names_topo, info_topo,
                 session_name, participant_id, output_dir, suffix, condition, baseline_name):
    fname = (f'{participant_id}_{session_name}_{suffix}_'
             f'ERP_topomap_{condition}_{baseline_name}.png')
    if _already_done(output_dir, fname):
        return

    vals = np.array([mean_amp_by_channel.get(ch, np.nan) for ch in ch_names_topo])
    valid_mask = ~np.isnan(vals)
    if not valid_mask.any():
        return

    vals_valid   = vals[valid_mask]
    chs_valid    = [ch for ch, ok in zip(ch_names_topo, valid_mask) if ok]
    info_valid   = mne.create_info(chs_valid, sfreq=info_topo['sfreq'], ch_types='eeg')
    montage      = mne.channels.make_standard_montage('standard_1020')
    info_valid.set_montage(montage, on_missing='ignore')

    vlim_val = np.nanpercentile(np.abs(vals_valid), 95)
    vlim_val = vlim_val if vlim_val > 0 else 1.0

    fig, ax = plt.subplots(figsize=(5, 4))
    im, _ = mne.viz.plot_topomap(
        vals_valid, info_valid,
        axes=ax, show=False, cmap='RdBu_r',
        vlim=(-vlim_val, vlim_val),
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='RMS µV (post-stimulus)')
    ax.set_title(
        f'{condition}  |  ERP post-stimulus RMS (0–{TUS_EPOCH_POST_SEC:.0f} s)\n'
        f'baseline: {baseline_name}  |  n={valid_mask.sum()} channels',
        fontsize=9
    )
    fig.tight_layout()
    fig.savefig(Path(output_dir) / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'      Saved ERP topomap: {fname}')

def plot_erp_topomap_evolution(raw, bursts_df, freq_band, session_name, participant_id,
                                output_dir, suffix=''):
    """
    Shows how the ERP scalp topography evolves over time, from
    -ERP_TOPO_TIMECOURSE_PRE_MS to +ERP_TOPO_TIMECOURSE_POST_MS relative to
    TUS onset, for Active and Sham separately, across all baseline
    corrections. One figure per (condition, baseline) — a grid of topomap
    snapshots at ERP_TOPO_TIMECOURSE_STEP_MS intervals.
    """
    if 'burst_time_s' not in bursts_df.columns:
        print('    plot_erp_topomap_evolution: burst_time_s column missing — skipping')
        return
    bursts_df = bursts_df.copy()
    bursts_df['burst_time_s'] = pd.to_numeric(bursts_df['burst_time_s'], errors='coerce')
    bursts_df = bursts_df.dropna(subset=['burst_time_s'])

    channels = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude='bads')
                if raw.ch_names[i] not in EXCLUDE_CHANNELS]
    if not channels or bursts_df.empty:
        return

    montage   = mne.channels.make_standard_montage('standard_1020')
    known_chs = set(montage.ch_names)
    topo_chs  = [ch for ch in channels if ch in known_chs]
    if len(topo_chs) < 3:
        print('    plot_erp_topomap_evolution: too few channels with known positions — skipping')
        return

    sfreq        = raw.info['sfreq']
    pre_samples  = int(TUS_EPOCH_PRE_SEC * sfreq)
    post_samples = int(TUS_EPOCH_POST_SEC * sfreq)
    n_samples    = pre_samples + post_samples
    times_ms     = np.linspace(-TUS_EPOCH_PRE_SEC, TUS_EPOCH_POST_SEC, n_samples) * 1000

    time_points = np.arange(-ERP_TOPO_TIMECOURSE_PRE_MS,
                            ERP_TOPO_TIMECOURSE_POST_MS + 1,
                            ERP_TOPO_TIMECOURSE_STEP_MS)

    # Extract epochs once (reused across all baselines/conditions below)
    all_epochs = {ch: {} for ch in topo_chs}
    for group_label, condition_set in [('sham', SHAM_CONDITIONS), ('active', ACTIVE_CONDITIONS)]:
        mask = bursts_df['condition'].isin(condition_set)
        group_df = bursts_df[mask].reset_index(drop=True)
        for ch in topo_chs:
            ch_idx = raw.ch_names.index(ch)
            trials = []
            for _, burst in group_df.iterrows():
                center = int(burst['burst_time_s'] * sfreq)
                start, stop = center - pre_samples, center + post_samples
                if start < 0 or stop > raw.n_times:
                    trials.append(np.full(n_samples, np.nan))
                    continue
                trial = raw.get_data(picks=[ch_idx], start=start, stop=stop)[0] * 1e6
                trials.append(trial)
            all_epochs[ch][group_label] = np.array(trials)

    halfwin_samples = int(ERP_TOPO_TIMECOURSE_HALFWIN_MS / 1000 * sfreq)

    for baseline_name, baseline_mode in ERP_BASELINES.items():
        for group_label in ('sham', 'active'):
            fname = (f'{participant_id}_{session_name}_{suffix}_'
                     f'ERP_topomap_evolution_{group_label}_{baseline_name}.png')
            if _already_done(output_dir, fname):
                continue

            n_trials = all_epochs[topo_chs[0]][group_label].shape[0]
            if n_trials == 0:
                continue

            # Same noisy-channel / noisy-trial exclusion logic as plot_erps,
            # so this figure stays consistent with the main ERP figures.
            BAD_CHANNEL_REJECTION_RATE = 0.30
            good_channels = []
            for ch in topo_chs:
                raw_trials  = all_epochs[ch][group_label].copy()
                corrected   = _apply_erp_baseline(raw_trials, pre_samples, baseline_mode, sfreq)
                finite_mask = np.all(np.isfinite(corrected), axis=1)
                if finite_mask.sum() == 0:
                    continue
                noise_mask, _ = _exclude_noisy_trials(corrected[finite_mask])
                n_rejected = int((~noise_mask).sum())
                if (n_rejected / n_trials) <= BAD_CHANNEL_REJECTION_RATE:
                    good_channels.append(ch)
            if not good_channels:
                continue

            global_keep = np.ones(n_trials, dtype=bool)
            for ch in good_channels:
                raw_trials  = all_epochs[ch][group_label].copy()
                corrected   = _apply_erp_baseline(raw_trials, pre_samples, baseline_mode, sfreq)
                finite_mask = np.all(np.isfinite(corrected), axis=1)
                noise_mask, _ = _exclude_noisy_trials(corrected[finite_mask])
                trial_keep    = np.zeros(n_trials, dtype=bool)
                trial_keep[np.where(finite_mask)[0][noise_mask]] = True
                global_keep  &= trial_keep

            mean_erp_by_ch = {}
            for ch in good_channels:
                raw_trials = all_epochs[ch][group_label].copy()
                corrected  = _apply_erp_baseline(raw_trials, pre_samples, baseline_mode, sfreq)
                clean      = corrected[global_keep]
                mean_erp_by_ch[ch] = (clean.mean(axis=0)
                                      if len(clean) and not np.all(np.isnan(clean))
                                      else np.full(n_samples, np.nan))
            if not mean_erp_by_ch:
                continue

            # --- shared color scale across all frames ---
            frame_vals = []
            for t_ms in time_points:
                t_idx = int(np.argmin(np.abs(times_ms - t_ms)))
                s0, s1 = max(t_idx - halfwin_samples, 0), min(t_idx + halfwin_samples, n_samples)
                vals = np.array([
                    np.nanmean(mean_erp_by_ch[ch][s0:s1]) if ch in mean_erp_by_ch else np.nan
                    for ch in topo_chs
                ])
                frame_vals.append(vals)
            all_frame_vals = np.array(frame_vals)
            finite_vals = all_frame_vals[np.isfinite(all_frame_vals)]
            if len(finite_vals) == 0:
                continue
            vlim = np.nanpercentile(np.abs(finite_vals), 95)
            vlim = vlim if vlim > 0 else 1.0

            ncols = 5
            nrows = int(np.ceil(len(time_points) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
            axes_flat = np.array(axes).ravel()

            last_im = None
            for i, t_ms in enumerate(time_points):
                ax = axes_flat[i]
                vals = all_frame_vals[i]
                valid_mask = np.isfinite(vals)
                if valid_mask.sum() < 3:
                    ax.text(0.5, 0.5, 'n/a', transform=ax.transAxes,
                            ha='center', va='center', fontsize=8, color='grey')
                    ax.set_axis_off()
                    ax.set_title(f'{int(t_ms)} ms', fontsize=9)
                    continue
                vals_valid = vals[valid_mask]
                chs_valid  = [ch for ch, ok in zip(topo_chs, valid_mask) if ok]
                info_valid = mne.create_info(chs_valid, sfreq=sfreq, ch_types='eeg')
                info_valid.set_montage(montage, on_missing='ignore')
                im, _ = mne.viz.plot_topomap(
                    vals_valid, info_valid, axes=ax, show=False,
                    cmap='RdBu_r', vlim=(-vlim, vlim), contours=4,
                )
                last_im = im
                title_color = 'black' if t_ms < 0 else '#922B21'
                ax.set_title(f'{int(t_ms)} ms', fontsize=9, fontweight='bold', color=title_color)

            for j in range(len(time_points), nrows * ncols):
                axes_flat[j].set_visible(False)

            if last_im is not None:
                fig.subplots_adjust(right=0.90, top=0.88)
                cax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
                fig.colorbar(last_im, cax=cax, label='µV (re: baseline)')

            fig.suptitle(
                f'{participant_id} – {session_name}  |  {group_label.upper()}  ERP topomap evolution\n'
                f'baseline: {baseline_name}  |  −{ERP_TOPO_TIMECOURSE_PRE_MS} to '
                f'+{ERP_TOPO_TIMECOURSE_POST_MS} ms  (step: {ERP_TOPO_TIMECOURSE_STEP_MS} ms)',
                fontsize=12, fontweight='bold'
            )
            fig.savefig(Path(output_dir) / fname, dpi=180, bbox_inches='tight')
            plt.close(fig)
            print(f'      Saved ERP topomap evolution: {fname}')

def _plot_erp_difference(mean_erps_active, mean_erps_sham, channels, times,
                          pre_samples, session_name, participant_id,
                          output_dir, suffix, baseline_name):
    """
    Plot the difference wave (active minus sham) for every channel.
    One PNG per baseline type, all channels in a 3-column grid.
    """
    fname = (f'{participant_id}_{session_name}_{suffix}_'
             f'ERP_difference_{baseline_name}_all_channels.png')
    if _already_done(output_dir, fname):
        return

    ncols = 3
    nrows = int(np.ceil(len(channels) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 4 * nrows),
        sharex=True, squeeze=False,
    )

    for idx, ch in enumerate(channels):
        r_i, c_i = divmod(idx, ncols)
        ax = axes[r_i][c_i]

        active_erp = mean_erps_active[idx]
        sham_erp   = mean_erps_sham[idx]

        if np.all(np.isnan(active_erp)) or np.all(np.isnan(sham_erp)):
            ax.text(0.5, 0.5, 'insufficient data',
                    transform=ax.transAxes, ha='center', va='center',
                    color='grey', fontsize=8)
            ax.set_title(ch, fontsize=8)
            continue

        diff = active_erp - sham_erp
        t_min, t_max = 0.0, 0.3  # 0–300 ms

        mask = (times >= t_min) & (times <= t_max)

        ax_in = ax.inset_axes([0.55, 0.55, 0.4, 0.4])

        ax_in.fill_between(times[mask], diff[mask], 0,
                        where=(diff[mask] >= 0), color='#E04B4B', alpha=0.35)
        ax_in.fill_between(times[mask], diff[mask], 0,
                        where=(diff[mask] < 0), color='#4B7BE0', alpha=0.35)

        ax_in.plot(times[mask], diff[mask], color='black', lw=1.2)
        ax_in.axvline(0, color='black', lw=0.8, ls='--', alpha=0.7)
        ax_in.axhline(0, color='grey', lw=0.5, ls=':')

        ax_in.set_xlim(t_min, t_max)
        ax_in.set_xticks([])
        ax_in.set_yticks([])
        # shade positive (active > sham) and negative (sham > active) regions
        ax.fill_between(times, diff, 0,
                        where=(diff >= 0), color='#E04B4B', alpha=0.35,
                        label='active > sham')
        ax.fill_between(times, diff, 0,
                        where=(diff < 0),  color='#4B7BE0', alpha=0.35,
                        label='sham > active')
        ax.plot(times, diff, color='black', lw=1.5)
        ax.axvline(0,  color='black', lw=0.9, ls='--', alpha=0.7)
        ax.axhline(0,  color='grey',  lw=0.6, ls=':')
        ax.axvspan(-TUS_EPOCH_PRE_SEC, 0, color='grey', alpha=0.06)
        ax.set_xlim(-1.0, TUS_EPOCH_POST_SEC)
        ax.set_ylabel('µV (active − sham)', fontsize=7)
        ax.set_title(ch, fontsize=8)
        ax.tick_params(labelsize=6)
        if idx == 0:
            ax.legend(fontsize=6, loc='upper right')

    # hide unused panels
    for idx in range(len(channels), nrows * ncols):
        r_i, c_i = divmod(idx, ncols)
        axes[r_i][c_i].set_visible(False)

    # x-axis label on every visible bottom-row panel
    for ax_row in axes:
        for ax in ax_row:
            if ax.get_visible():
                ax.set_xlabel('Time (s)', fontsize=7)

    fig.suptitle(
        f'{participant_id} – {session_name}  |  ERP difference  (active − sham)\n'
        f'baseline: {baseline_name}',
        fontsize=11, fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(Path(output_dir) / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'      Saved ERP difference figure: {fname}')

def plot_erps(raw, bursts_df, freq_band, session_name, participant_id, output_dir, suffix=''):

    if 'burst_time_s' not in bursts_df.columns:
        print('    plot_erps: burst_time_s column missing — skipping')
        return None
    bursts_df['burst_time_s'] = pd.to_numeric(bursts_df['burst_time_s'], errors='coerce')
    bursts_df = bursts_df.dropna(subset=['burst_time_s'])

    channels = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude='bads')
            if raw.ch_names[i] not in EXCLUDE_CHANNELS]
    if not channels or bursts_df.empty:
        return None

    sfreq        = raw.info['sfreq']
    pre_samples  = int(TUS_EPOCH_PRE_SEC * sfreq)
    post_samples = int(TUS_EPOCH_POST_SEC * sfreq)
    n_samples    = pre_samples + post_samples
    times        = np.linspace(-TUS_EPOCH_PRE_SEC, TUS_EPOCH_POST_SEC, n_samples)

    post_start_idx = pre_samples
    hab_start      = pre_samples + int(HABITUATION_WINDOW_SEC[0] * sfreq)
    hab_end        = pre_samples + int(HABITUATION_WINDOW_SEC[1] * sfreq)

    montage   = mne.channels.make_standard_montage('standard_1020')
    known_chs = set(montage.ch_names)
    topo_chs  = [ch for ch in channels if ch in known_chs]
    info_topo = None
    if len(topo_chs) >= 3:
        info_topo = mne.create_info(topo_chs, sfreq=sfreq, ch_types='eeg')
        info_topo.set_montage(montage, on_missing='ignore')

    all_epochs     = {ch: {} for ch in channels}
    all_trial_nums = {}

    for group_label, condition_set in [('sham', SHAM_CONDITIONS), ('active', ACTIVE_CONDITIONS)]:
        mask       = bursts_df['condition'].isin(condition_set)
        group_df   = bursts_df[mask].reset_index(drop=True)
        trial_nums = np.arange(1, len(group_df) + 1)
        all_trial_nums[group_label] = trial_nums

        for ch in channels:
            ch_idx = raw.ch_names.index(ch)
            trials = []
            for _, burst in group_df.iterrows():
                center = int(burst['burst_time_s'] * sfreq)
                start, stop = center - pre_samples, center + post_samples
                if start < 0 or stop > raw.n_times:
                    trials.append(np.full(n_samples, np.nan))
                    continue
                if window_overlaps_bad_annotation(raw, start / sfreq, stop / sfreq):
                    trials.append(np.full(n_samples, np.nan))
                    continue
                trial = raw.get_data(picks=[ch_idx], start=start, stop=stop)[0] * 1e6
                trials.append(trial)
            all_epochs[ch][group_label] = np.array(trials)

    focus_channel_by_condition = {}

    for baseline_name, baseline_mode in ERP_BASELINES.items():
        print(f'\n    ERP baseline: {baseline_name}')
        mean_erps_by_condition = {}
        peak_amp_by_condition  = {}

        for group_label in ('sham', 'active'):
            mean_erps            = []
            clean_trials_by_channel = []

            # ── Step 1: identify bad channels (>30% rejection) ────────────
            BAD_CHANNEL_REJECTION_RATE = 0.30
            n_trials      = all_epochs[channels[0]][group_label].shape[0]
            good_channels = []
            bad_channels  = []

            for ch in channels:
                raw_trials  = all_epochs[ch][group_label].copy()
                corrected   = _apply_erp_baseline(raw_trials, pre_samples, baseline_mode, sfreq)
                finite_mask = np.all(np.isfinite(corrected), axis=1)
                noise_mask, _ = _exclude_noisy_trials(corrected[finite_mask])
                n_rejected     = int((~noise_mask).sum())
                rejection_rate = n_rejected / n_trials
                if rejection_rate > BAD_CHANNEL_REJECTION_RATE:
                    bad_channels.append((ch, round(rejection_rate * 100, 1)))
                else:
                    good_channels.append(ch)

            if bad_channels:
                print(f'      [{group_label}] Excluded noisy channels '
                      f'(>{BAD_CHANNEL_REJECTION_RATE*100:.0f}% trials rejected):')
                for ch, pct in bad_channels:
                    print(f'        {ch}: {pct}% rejected')
            print(f'      [{group_label}] Clean channels for ERP: '
                  f'{len(good_channels)} / {len(channels)}')

            if not good_channels:
                print(f'      [{group_label}] No clean channels — skipping')
                for ch in channels:
                    clean_trials_by_channel.append(np.empty((0, n_samples)))
                    mean_erps.append(np.full(n_samples, np.nan))
                mean_erps_by_condition[group_label] = mean_erps
                continue

            # ── Step 2: build global keep-mask from good channels only ────
            global_keep = np.ones(n_trials, dtype=bool)

            for ch in good_channels:
                raw_trials  = all_epochs[ch][group_label].copy()
                corrected   = _apply_erp_baseline(raw_trials, pre_samples, baseline_mode, sfreq)
                finite_mask = np.all(np.isfinite(corrected), axis=1)
                noise_mask, _ = _exclude_noisy_trials(corrected[finite_mask])
                trial_keep    = np.zeros(n_trials, dtype=bool)
                trial_keep[np.where(finite_mask)[0][noise_mask]] = True
                global_keep  &= trial_keep

            n_kept = int(global_keep.sum())
            print(f'      [{group_label}] Global trial mask: {n_kept} / {n_trials} trials kept '
                  f'(across {len(good_channels)} clean channels)')

            # ── Step 3: apply uniform mask across all channels ────────────
            for ch in channels:
                raw_trials = all_epochs[ch][group_label].copy()
                corrected  = _apply_erp_baseline(raw_trials, pre_samples, baseline_mode, sfreq)
                if ch in good_channels:
                    clean = corrected[global_keep]
                else:
                    clean = np.full((n_kept, n_samples), np.nan)
                clean_trials_by_channel.append(clean)
                mean_erps.append(
                    clean.mean(axis=0) if (len(clean) and not np.all(np.isnan(clean)))
                    else np.full(n_samples, np.nan)
                )

            mean_erps_by_condition[group_label] = mean_erps
            ranked_chs, scores = _rank_channels_by_erp(mean_erps, channels, post_start_idx)
            if group_label == 'active':
                focus_channel_by_condition['active'] = ranked_chs[0]
            peak_amp_by_condition[group_label] = {ch: scores[ch] for ch in channels}

            print(f'      [{group_label}] Channel ranking (peak |ERP|, 0–end):')
            for rank_i, rc in enumerate(ranked_chs[:5], 1):
                print(f'        {rank_i}. {rc}  {scores[rc]:.2f} µV')

            best_ch    = ranked_chs[0]
            color      = '#4B7BE0' if group_label == 'sham' else '#E04B4B'
            ylabel_erp = 'µV' if baseline_name != 'pre_zscore' else 'z-score'

            # Figure 1: all channels (3-col grid)
            fname_all = (f'{participant_id}_{session_name}_{suffix}_'
                         f'ERP_{group_label}_{baseline_name}_all_channels.png')
            if not _already_done(output_dir, fname_all):
                ncols_erp = 3
                nrows_erp = int(np.ceil(len(channels) / ncols_erp))
                fig_all, axes_all = plt.subplots(
                    nrows_erp, ncols_erp,
                    figsize=(7 * ncols_erp, 4 * nrows_erp),
                    sharex=True, squeeze=False,
                )
                for idx_ch, ch in enumerate(channels):
                    r_i, c_i    = divmod(idx_ch, ncols_erp)
                    ax          = axes_all[r_i][c_i]
                    ch_idx_list = channels.index(ch)
                    clean_trials = clean_trials_by_channel[ch_idx_list]
                    mean_erp = mean_erps[ch_idx_list]
                    sem_erp  = (clean_trials.std(axis=0) / np.sqrt(len(clean_trials))
                                if len(clean_trials) > 1 else np.zeros(n_samples))
                    ax.fill_between(times, mean_erp - sem_erp, mean_erp + sem_erp,color=color, alpha=0.3)
                    ax.plot(times, mean_erp, color=color, lw=1.8,label=f'n={len(clean_trials)}')
                    ax.axvline(0, color='black', lw=0.9, ls='--', alpha=0.7)
                    ax.axhline(0, color='grey',  lw=0.5, ls=':')
                    ax.axvspan(PAPER_BASELINE_SEC[0], PAPER_BASELINE_SEC[1], color='grey', alpha=0.07, label='Baseline' if idx_ch == 0 else '_')
                    ax.set_xlim(-1.0, TUS_EPOCH_POST_SEC)
                    ax.set_ylabel(ylabel_erp, fontsize=7)
                    ax.tick_params(labelsize=6)
                    rank_pos   = ranked_chs.index(ch) + 1
                    is_best_ch = (ch == best_ch)
                    ch_title   = (f'★ {ch}  [#1 highest ERP]' if is_best_ch
                                  else f'{ch}  [rank #{rank_pos}]')
                    ax.set_title(ch_title, fontsize=8,
                                 fontweight='bold' if is_best_ch else 'normal',
                                 color='#C0392B' if is_best_ch else 'black')
                    ax.legend(fontsize=6, loc='upper right')
                for idx_ch in range(len(channels), nrows_erp * ncols_erp):
                    r_i, c_i = divmod(idx_ch, ncols_erp)
                    axes_all[r_i][c_i].set_visible(False)
                for ax_row in axes_all:
                    for ax in ax_row:
                        if ax.get_visible():
                            ax.set_xlabel('Time (s)', fontsize=7)
                fig_all.suptitle(
                    f'{participant_id} – {session_name}  |  {group_label.upper()}  ERP  '
                    f'[baseline: {baseline_name}]\n'
                    f'All channels  |  ★ = highest ERP response ({best_ch})',
                    fontsize=11, fontweight='bold'
                )
                fig_all.tight_layout(rect=[0, 0, 1, 0.96])
                fig_all.savefig(Path(output_dir) / fname_all, dpi=150, bbox_inches='tight')
                plt.close(fig_all)
                print(f'      Saved ERP (all channels): {fname_all}')

            # Figure 2: top-10
            top10_chs   = ranked_chs[:10]
            fname_top10 = (f'{participant_id}_{session_name}_{suffix}_'
                           f'ERP_{group_label}_{baseline_name}_top10.png')
            if not _already_done(output_dir, fname_top10):
                ncols_top = 2
                nrows_top = 5
                fig10, axes10 = plt.subplots(
                    nrows_top, ncols_top,
                    figsize=(14, 4 * nrows_top),
                    sharex=True, sharey=False,
                    squeeze=False,
                )
                axes10_flat = axes10.ravel()
                for j in range(len(top10_chs), len(axes10_flat)):
                    axes10_flat[j].set_visible(False)

                for panel_idx, ch in enumerate(top10_chs):
                    ax          = axes10_flat[panel_idx]
                    ch_idx_list = channels.index(ch)
                    clean_trials = clean_trials_by_channel[ch_idx_list]
                    mean_erp = mean_erps[ch_idx_list]
                    sem_erp  = (clean_trials.std(axis=0) / np.sqrt(len(clean_trials))
                                if len(clean_trials) > 1 else np.zeros(n_samples))
                    ax.fill_between(times, mean_erp - sem_erp, mean_erp + sem_erp,
                                    color=color, alpha=0.3)
                    ax.plot(times, mean_erp, color=color, lw=2.0,
                            label=f'Mean (n={len(clean_trials)})')
                    ax.axvline(0, color='black', lw=1.0, ls='--', alpha=0.7,
                               label='TUS onset')
                    ax.axvspan(PAPER_BASELINE_SEC[0], PAPER_BASELINE_SEC[1], color='grey', alpha=0.07, label='Baseline')
                    ax.axhline(0, color='grey', lw=0.6, ls=':')
                    ax.set_xlim(-1.0, TUS_EPOCH_POST_SEC)
                    ax.set_ylabel(ylabel_erp, fontsize=8)
                    rank_pos   = ranked_chs.index(ch) + 1
                    is_best_ch = (ch == best_ch)
                    ch_title   = (
                        f'★ {ch}  [rank #1 – highest ERP  |  n={len(clean_trials)} trials]'
                        if is_best_ch else
                        f'{ch}  [rank #{rank_pos}  |  n={len(clean_trials)} trials]'
                    )
                    ax.set_title(ch_title, fontsize=9,
                                 fontweight='bold' if is_best_ch else 'normal',
                                 color='#C0392B' if is_best_ch else 'black')
                    ax.legend(fontsize=7, loc='upper right')
                    ax.tick_params(labelsize=7)
                    if panel_idx >= ncols_top * (nrows_top - 1):
                        ax.set_xlabel('Time (s)', fontsize=8)

                fig10.suptitle(
                    f'{participant_id} – {session_name}  |  {group_label.upper()}  ERP  '
                    f'[baseline: {baseline_name}]\n'
                    f'Top 10 channels by peak |ERP| response  |  ★ = {best_ch} (highest)',
                    fontsize=11, fontweight='bold'
                )
                fig10.tight_layout(rect=[0, 0, 1, 0.96])
                fig10.savefig(Path(output_dir) / fname_top10, dpi=150, bbox_inches='tight')
                plt.close(fig10)
                print(f'      Saved ERP top-10 figure: {fname_top10}')

            # Figure 2b: top-3
            top3_chs   = ranked_chs[:3]
            fname_top3 = (f'{participant_id}_{session_name}_{suffix}_'
                          f'ERP_{group_label}_{baseline_name}_top3.png')
            if not _already_done(output_dir, fname_top3):
                fig3, axes3 = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

                for panel_idx, ch in enumerate(top3_chs):
                    ax          = axes3[panel_idx]
                    ch_idx_list = channels.index(ch)
                    clean_trials = clean_trials_by_channel[ch_idx_list]
                    mean_erp    = mean_erps[ch_idx_list]
                    n_tr        = len(clean_trials)
                    sem_erp     = (clean_trials.std(axis=0) / np.sqrt(n_tr)
                                   if n_tr > 1 else np.zeros(n_samples))

                    ax.axvspan(PAPER_BASELINE_SEC[0], PAPER_BASELINE_SEC[1], color='grey', alpha=0.08,label='Baseline' if panel_idx == 0 else '_')
                    ax.fill_between(times, mean_erp - sem_erp, mean_erp + sem_erp,color=color, alpha=0.3)
                    ax.plot(times, mean_erp, color=color, lw=2,label=f'Mean ± SEM  (n={n_tr})' if panel_idx == 0 else f'n={n_tr}')
                    ax.axhline(0, color='grey', lw=0.6, ls=':')
                    ax.set_xlim(-1.0, TUS_EPOCH_POST_SEC)
                    ax.set_ylabel(ylabel_erp, fontsize=9)
                    ax.set_title(f'{ch}  [rank #{panel_idx + 1}  |  n={n_tr} trials]',fontsize=10,fontweight='bold' if panel_idx == 0 else 'normal',color='#C0392B' if panel_idx == 0 else 'black',)
                    ax.tick_params(labelsize=8)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    if panel_idx == 0:
                        ax.legend(fontsize=8, loc='upper right')

                axes3[-1].set_xlabel('Time (s)', fontsize=9)
                fig3.suptitle(
                    f'{participant_id} – {session_name}  |  {group_label.upper()}  ERP  '
                    f'[baseline: {baseline_name}]\n'
                    f'Top 3 channels by post-stimulus RMS',
                    fontsize=11, fontweight='bold'
                )
                fig3.tight_layout(rect=[0, 0, 1, 0.95])
                fig3.savefig(Path(output_dir) / fname_top3, dpi=150, bbox_inches='tight')
                plt.close(fig3)
                print(f'      Saved ERP top-3 figure: {fname_top3}')

            # Figure 3: habituation for ALL channels
            _habituation_plot_all_channels(
                all_epochs=all_epochs,
                channels=channels,
                pre_samples=pre_samples,
                baseline_mode=baseline_mode,
                hab_start=hab_start,
                hab_end=hab_end,
                clean_trials_by_channel=clean_trials_by_channel,
                condition=group_label,
                session_name=session_name,
                participant_id=participant_id,
                output_dir=output_dir,
                suffix=f'{suffix}_{baseline_name}',
                kind='ERP',
                best_channel=best_ch,
            )

            if info_topo is not None:
                _erp_topomap(
                    peak_amp_by_condition[group_label], topo_chs, info_topo,
                    session_name, participant_id, output_dir,
                    suffix, group_label, baseline_name,
                )

        # ── outside group_label loop, inside baseline loop ────────────────
        if 'active' in mean_erps_by_condition and 'sham' in mean_erps_by_condition:
            _plot_erp_difference(
                mean_erps_by_condition['active'], mean_erps_by_condition['sham'],
                channels, times, pre_samples,
                session_name, participant_id, output_dir, suffix, baseline_name,
            )
            _plot_erp_topo_overlay(
                mean_erps_by_condition['active'], mean_erps_by_condition['sham'],
                channels, times,
                session_name, participant_id, output_dir, suffix, baseline_name,
                t_min=-0.5, t_max=1.5,
            )

    # ── outside baseline loop ─────────────────────────────────────────────
    for ch in FOCUS_CHANNEL_PRIORITY:
        if ch in channels:
            return ch
    return channels[0] if channels else None

def plot_erps_500ms(raw, bursts_df, freq_band, session_name, participant_id, output_dir, suffix=''):
    """
    Plots ERPs strictly using a 500ms post-stimulus window across all baselines.
    Outputs figures for: All Channels, Top 10 Channels, and Top 3 Channels.
    X-axis is scaled to milliseconds (ms) to clearly evaluate early peaks.
    """
    if 'burst_time_s' not in bursts_df.columns:
        print('    plot_erps_500ms: burst_time_s column missing — skipping')
        return None

    bursts_df['burst_time_s'] = pd.to_numeric(bursts_df['burst_time_s'], errors='coerce')
    bursts_df = bursts_df.dropna(subset=['burst_time_s'])

    channels = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude='bads')
                if raw.ch_names[i] not in EXCLUDE_CHANNELS]
    if not channels or bursts_df.empty:
        return None

    # ── FIXED 500MS POST-STIMULUS CONFIGURATION ──────────────────────────────
    POST_STIMULUS_SEC = 0.5   
    sfreq             = raw.info['sfreq']
    pre_samples       = int(TUS_EPOCH_PRE_SEC * sfreq)
    post_samples      = int(POST_STIMULUS_SEC * sfreq)
    n_samples         = pre_samples + post_samples

    # Generate time base and instantly scale to milliseconds
    times_ms          = np.linspace(-TUS_EPOCH_PRE_SEC, POST_STIMULUS_SEC, n_samples) * 1000
    post_start_idx    = pre_samples
    # ─────────────────────────────────────────────────────────────────────────

    # Extract raw data epochs for Sham and Active blocks
    all_epochs = {ch: {} for ch in channels}
    for group_label, condition_set in [('sham', SHAM_CONDITIONS), ('active', ACTIVE_CONDITIONS)]:
        mask = bursts_df['condition'].isin(condition_set)
        group_df = bursts_df[mask].reset_index(drop=True)

        for ch in channels:
            ch_idx = raw.ch_names.index(ch)
            trials = []
            for _, burst in group_df.iterrows():
                center = int(burst['burst_time_s'] * sfreq)
                start, stop = center - pre_samples, center + post_samples
                if start < 0 or stop > raw.n_times:
                    trials.append(np.full(n_samples, np.nan))
                    continue
                trial = raw.get_data(picks=[ch_idx], start=start, stop=stop)[0] * 1e6
                trials.append(trial)
            all_epochs[ch][group_label] = np.array(trials)

    # Loop through your 3 baseline styles (none, pre_mean, pre_zscore)
    for baseline_name, baseline_mode in ERP_BASELINES.items():
        print(f'\n    ERP 500ms baseline: {baseline_name}')

        # We will collect active vs sham means to help track rankings and plotting limits
        mean_erps_active = []
        mean_erps_sham   = []
        clean_trials_active = {}
        clean_trials_sham   = {}

        # ── DATA CLEANING & BASELINING FOR BOTH CONDITIONS ───────────────────
        for group_label in ('sham', 'active'):
            BAD_CHANNEL_REJECTION_RATE = 0.30
            n_trials = all_epochs[channels[0]][group_label].shape[0]
            good_channels = []

            for ch in channels:
                raw_trials = all_epochs[ch][group_label].copy()
                corrected = _apply_erp_baseline(raw_trials, pre_samples, baseline_mode, sfreq)
                finite_mask = np.all(np.isfinite(corrected), axis=1)
                noise_mask, _ = _exclude_noisy_trials(corrected[finite_mask])
                if (int((~noise_mask).sum()) / n_trials) <= BAD_CHANNEL_REJECTION_RATE:
                    good_channels.append(ch)

            global_keep = np.ones(n_trials, dtype=bool)
            for ch in good_channels:
                raw_trials = all_epochs[ch][group_label].copy()
                corrected = _apply_erp_baseline(raw_trials, pre_samples, baseline_mode, sfreq)
                finite_mask = np.all(np.isfinite(corrected), axis=1)
                noise_mask, _ = _exclude_noisy_trials(corrected[finite_mask])
                trial_keep = np.zeros(n_trials, dtype=bool)
                trial_keep[np.where(finite_mask)[0][noise_mask]] = True
                global_keep &= trial_keep

            for ch in channels:
                raw_trials = all_epochs[ch][group_label].copy()
                corrected = _apply_erp_baseline(raw_trials, pre_samples, baseline_mode, sfreq)
                clean = corrected[global_keep] if ch in good_channels else np.full((int(global_keep.sum()), n_samples), np.nan)
                m_erp = clean.mean(axis=0) if (len(clean) and not np.all(np.isnan(clean))) else np.full(n_samples, np.nan)

                if group_label == 'active':
                    mean_erps_active.append(m_erp)
                    clean_trials_active[ch] = clean
                else:
                    mean_erps_sham.append(m_erp)
                    clean_trials_sham[ch] = clean

        # Calculate Channel Rankings based on peak |ERP| amplitude in the active group
        ranked_chs, scores = _rank_channels_by_erp(mean_erps_active, channels, post_start_idx)
        ylabel_erp = 'µV' if baseline_name != 'pre_zscore' else 'z-score'

        # ── DEFINE THE 3 CHANNEL SUBSETS TO PLOT ─────────────────────────────
        plot_configurations = [
            ('all', channels),
            ('top10', ranked_chs[:10]),
            ('top3', ranked_chs[:3])
        ]

        # ── PLOTTING LOOP FOR THE 3 CONFIGURATIONS ───────────────────────────
        for subset_name, ch_list in plot_configurations:
            if not ch_list:
                continue

            fname = f'{participant_id}_{session_name}_{suffix}_ERP_{baseline_name}_{subset_name}_500ms.png'
            if _already_done(output_dir, fname):
                continue

            ncols = 3 if subset_name != 'top3' else 3
            nrows = int(np.ceil(len(ch_list) / ncols))

            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharex=True, squeeze=False)

            for idx_ch, ch in enumerate(ch_list):
                r_i, c_i = divmod(idx_ch, ncols)
                ax = axes[r_i][c_i]

                ch_idx = channels.index(ch)

                # Plot Sham (Blue)
                sham_clean = clean_trials_sham[ch]
                sham_mean  = mean_erps_sham[ch_idx]
                sham_sem   = (sham_clean.std(axis=0) / np.sqrt(len(sham_clean)) if len(sham_clean) > 1 else np.zeros(n_samples))
                ax.fill_between(times_ms, sham_mean - sham_sem, sham_mean + sham_sem, color='#4B7BE0', alpha=0.2)
                ax.plot(times_ms, sham_mean, color='#4B7BE0', lw=1.2, label='Sham')

                # Plot Active (Red)
                act_clean = clean_trials_active[ch]
                act_mean  = mean_erps_active[ch_idx]
                act_sem   = (act_clean.std(axis=0) / np.sqrt(len(act_clean)) if len(act_clean) > 1 else np.zeros(n_samples))
                ax.fill_between(times_ms, act_mean - act_sem, act_mean + act_sem, color='#E04B4B', alpha=0.2)
                ax.plot(times_ms, act_mean, color='#E04B4B', lw=1.2, label='Active')

                # Visual Anchors for Early Peak Tracking
                ax.axvline(0, color='black', lw=1.0, ls='--', alpha=0.7)  # TUS Onset Line
                ax.axhline(0, color='grey', lw=0.5, ls=':', alpha=0.5)
                ax.set_xlim(-300, 500)
                ax.set_xticks(np.arange(-300, 501, 100))
                ax.set_title(f'Channel: {ch}', fontsize=12, fontweight='bold')
                ax.set_ylabel(ylabel_erp, fontsize=10)
                ax.grid(True, ls='--', alpha=0.4) # Dashed background grid to visually map latency

                # Label the X-axis on the bottom row charts
                if r_i == nrows - 1:
                    ax.set_xlabel('Time (ms)', fontsize=11)

                if idx_ch == 0:
                    ax.legend(loc='upper right', fontsize=9)

            # Hide empty subplots if the grid has leftover spaces
            for remaining in range(idx_ch + 1, nrows * ncols):
                r_i, c_i = divmod(remaining, ncols)
                fig.delaxes(axes[r_i][c_i])

            fig.suptitle(
                f'{participant_id} ({session_name}) | 500ms Window ({subset_name.upper()})\n'
                f'Baseline Treatment: {baseline_name}', 
                fontsize=14, fontweight='bold', y=0.98
            )
            fig.tight_layout()
            fig.savefig(Path(output_dir) / fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'      Saved 500ms ERP subset figure: {fname}')


def plot_tfrs(
    raw,
    bursts_df,
    freq_band,
    session_name,
    participant_id,
    output_dir,
    suffix="",
    focus_channel=None,
):
    # ── FIX: the per-pulse CSV stores burst time as 'burst_time_s', not 'time_sec'
    if "burst_time_s" not in bursts_df.columns:
        print("    plot_tfrs: burst_time_s column missing — skipping")
        return
    bursts_df["burst_time_s"] = pd.to_numeric(
        bursts_df["burst_time_s"], errors="coerce"
    )

    bursts_df = bursts_df.dropna(subset=["burst_time_s"])

    channels = [
        raw.ch_names[i]
        for i in mne.pick_types(raw.info, eeg=True, exclude='bads')
        if raw.ch_names[i] not in EXCLUDE_CHANNELS
    ]
    if not channels or bursts_df.empty:
        return

    sfreq = raw.info["sfreq"]
    pre_samples = int(TUS_EPOCH_PRE_SEC * sfreq)
    post_samples = int(TUS_EPOCH_POST_SEC * sfreq)
    n_samples = pre_samples + post_samples
    times = np.linspace(-TUS_EPOCH_PRE_SEC, TUS_EPOCH_POST_SEC, n_samples)

    freqs = np.arange(1.0, 31.0, 1.0)
    n_cycles = freqs / 2.0

    hab_start_idx = pre_samples + int(HABITUATION_WINDOW_SEC[0] * sfreq)
    hab_end_idx = pre_samples + int(HABITUATION_WINDOW_SEC[1] * sfreq)

    montage = mne.channels.make_standard_montage("standard_1020")
    known_chs = set(montage.ch_names)
    topo_chs = [ch for ch in channels if ch in known_chs]
    info_topo = None
    if len(topo_chs) >= 3:
        info_topo = mne.create_info(topo_chs, sfreq=sfreq, ch_types="eeg")
        info_topo.set_montage(montage, on_missing="ignore")

    if focus_channel is None or focus_channel not in channels:
        focus_channel = next((ch for ch in FOCUS_CHANNEL_PRIORITY if ch in channels),
                              channels[0])
    print(f"    TFR focus channel (fixed ROI): {focus_channel}")

    def morlet_tfr(epochs_2d):
        data_3d = epochs_2d[:, np.newaxis, :]
        power_4d = mne.time_frequency.tfr_array_morlet(
            data_3d,
            sfreq=sfreq,
            freqs=freqs,
            n_cycles=n_cycles,
            output="power",
            verbose=False,
        )
        return power_4d[:, 0, :, :]

    def apply_tfr_baseline(power_3d, bl_start_sec, bl_end_sec):
        bl_s = pre_samples + int(bl_start_sec * sfreq)
        bl_e = pre_samples + int(bl_end_sec * sfreq)
        bl_s = max(bl_s, 0)
        bl_e = min(bl_e, n_samples)
        bl_power = power_3d[:, :, bl_s:bl_e].mean(axis=2, keepdims=True)
        return 10 * np.log10(power_3d / (bl_power + 1e-30))

    raw_power = {}
    for group_label, condition_set in [
        ("sham", SHAM_CONDITIONS),
        ("active", ACTIVE_CONDITIONS),
    ]:
        mask = bursts_df["condition"].isin(condition_set)
        group_df = bursts_df[mask].reset_index(drop=True)
        if len(group_df) < 2:
            continue
        for ch in channels:
            ch_idx = raw.ch_names.index(ch)
            all_trials_uv = []
            for _, burst in group_df.iterrows():
                center = int(burst["burst_time_s"] * sfreq)
                start, stop = center - pre_samples, center + post_samples
                if start < 0 or stop > raw.n_times:
                    continue
                if window_overlaps_bad_annotation(raw, start / sfreq, stop / sfreq):
                    continue
                trial = raw.get_data(start=start, stop=stop)[ch_idx] * 1e6
                demeaned = trial - trial.mean()
                if np.max(np.abs(demeaned)) <= EPOCH_REJECT_UV:
                    all_trials_uv.append(trial)
            if len(all_trials_uv) < 2:
                continue
            epochs = all_trials_uv
            if len(epochs) >= 2:
                raw_power[(ch, group_label)] = morlet_tfr(np.array(epochs))
    for bl_name, (bl_start, bl_end) in TFR_BASELINES.items():
        print(f"\n    TFR baseline: {bl_name}  ({bl_start:.2f} to {bl_end:.2f} s)")
        band_power_rows = []

        for group_label, condition_set in [
            ("sham", SHAM_CONDITIONS),
            ("active", ACTIVE_CONDITIONS),
        ]:
            mask = bursts_df["condition"].isin(condition_set)
            group_df = bursts_df[mask].reset_index(drop=True)
            n_trials = sum(
                1
                for _, burst in group_df.iterrows()
                if 0 <= int(burst["burst_time_s"] * sfreq) - pre_samples
                and int(burst["burst_time_s"] * sfreq) + post_samples <= raw.n_times
            )
            if n_trials < 2:
                continue

            for ch in channels:
                key = (ch, group_label)
                if key not in raw_power:
                    continue
                power_3d = apply_tfr_baseline(raw_power[key], bl_start, bl_end)
                mean_tfr = power_3d.mean(axis=0)
                n_used = power_3d.shape[0]

                fname = (
                    f"{participant_id}_{session_name}_{suffix}_"
                    f"TFR_{ch}_{group_label}_{bl_name}.png"
                )
                if not _already_done(output_dir, fname):
                    vmax = np.nanpercentile(np.abs(mean_tfr), 97)
                    fig, ax = plt.subplots(figsize=(12, 5))
                    pcm = ax.pcolormesh(
                        times,
                        freqs,
                        mean_tfr,
                        cmap="RdBu_r",
                        vmin=-vmax,
                        vmax=vmax,
                        shading="gouraud",
                    )
                    fig.colorbar(pcm, ax=ax, label="dB (re: baseline)")
                    ax.axvline(
                        0, color="black", lw=1.2, ls="--", alpha=0.8, label="TUS onset"
                    )
                    ax.axhspan(
                        freq_band[0],
                        freq_band[1],
                        color="yellow",
                        alpha=0.15,
                        label="Spindle band",
                    )
                    ax.axvspan(
                        bl_start,
                        bl_end,
                        color="lime",
                        alpha=0.12,
                        label="Baseline window",
                    )
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Frequency (Hz)")
                    ax.set_title(
                        f"{group_label.upper()}  |  {ch}  (n={n_used} trials)",
                        fontsize=11,
                        fontweight="bold",
                    )
                    ax.legend(fontsize=8, loc="upper right")
                    fig.suptitle(
                        f"{participant_id} – {session_name}  |  {ch}  TFR  "
                        f"[Morlet]  |  baseline: {bl_name}",
                        fontsize=11,
                        fontweight="bold",
                    )
                    fig.tight_layout()
                    fig.savefig(Path(output_dir) / fname, dpi=150, bbox_inches="tight")
                    plt.close(fig)

                for band_name, (b_low, b_high) in TFR_BANDS.items():
                    freq_mask = (freqs >= b_low) & (freqs <= b_high)
                    if not freq_mask.any():
                        continue
                    trial_band_power = power_3d[:, freq_mask, :][
                        :, :, hab_start_idx:hab_end_idx
                    ].mean(axis=(1, 2))
                    for t_idx, bp in enumerate(trial_band_power):
                        band_power_rows.append(
                            {
                                "participant_id": participant_id,
                                "session": session_name,
                                "baseline": bl_name,
                                "condition": group_label,
                                "channel": ch,
                                "band": band_name,
                                "trial": t_idx + 1,
                                "mean_power_db": round(float(bp), 6),
                            }
                        )

            key_focus = (focus_channel, group_label)
            if key_focus in raw_power:
                power_3d_focus = apply_tfr_baseline(
                    raw_power[key_focus], bl_start, bl_end
                )
                n_focus_trials = power_3d_focus.shape[0]
                n_show = min(n_focus_trials, 16)

                fname_pertrial = (
                    f"{participant_id}_{session_name}_{suffix}_"
                    f"TFR_per_trial_{focus_channel}_{group_label}_{bl_name}.png"
                )
                if not _already_done(output_dir, fname_pertrial):
                    ncols = 4
                    nrows = int(np.ceil(n_show / ncols))
                    fig, axes = plt.subplots(
                        nrows,
                        ncols,
                        figsize=(ncols * 4, nrows * 3),
                        sharex=True,
                        sharey=True,
                    )
                    axes = np.array(axes).ravel()
                    vmax_focus = np.nanpercentile(np.abs(power_3d_focus), 97)
                    for ti in range(n_show):
                        ax = axes[ti]
                        pcm = ax.pcolormesh(
                            times,
                            freqs,
                            power_3d_focus[ti],
                            cmap="RdBu_r",
                            vmin=-vmax_focus,
                            vmax=vmax_focus,
                            shading="gouraud",
                        )
                        ax.axvline(0, color="white", lw=0.8, ls="--", alpha=0.7)
                        ax.set_title(f"Trial {ti+1}", fontsize=8)
                    for j in range(n_show, len(axes)):
                        axes[j].set_visible(False)
                    fig.suptitle(
                        f"{participant_id} – {session_name}  |  {focus_channel}  "
                        f"per-trial TFR  [{group_label.upper()}]  |  baseline: {bl_name}",
                        fontsize=11,
                        fontweight="bold",
                    )
                    fig.tight_layout()
                    fig.savefig(
                        Path(output_dir) / fname_pertrial, dpi=150, bbox_inches="tight"
                    )
                    plt.close(fig)
                    print(f"      Saved per-trial TFR: {fname_pertrial}")

                for band_name, (b_low, b_high) in TFR_BANDS.items():
                    freq_mask = (freqs >= b_low) & (freqs <= b_high)
                    if not freq_mask.any():
                        continue
                    trial_bp = power_3d_focus[:, freq_mask, :][
                        :, :, hab_start_idx:hab_end_idx
                    ].mean(axis=(1, 2))
                    _habituation_plot(
                        trial_amplitudes=trial_bp,
                        trial_numbers=np.arange(1, len(trial_bp) + 1),
                        ch_name=focus_channel,
                        condition=group_label,
                        session_name=session_name,
                        participant_id=participant_id,
                        output_dir=output_dir,
                        suffix=f"{suffix}_{bl_name}",
                        kind=band_name,
                    )

            if info_topo is not None:
                for band_name, (b_low, b_high) in TFR_BANDS.items():
                    freq_mask = (freqs >= b_low) & (freqs <= b_high)
                    if not freq_mask.any():
                        continue
                    fname_topo = (
                        f"{participant_id}_{session_name}_{suffix}_"
                        f"TFR_topomap_{group_label}_{band_name}_{bl_name}.png"
                    )
                    if _already_done(output_dir, fname_topo):
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
                    valid_mask = ~np.isnan(vals_arr)
                    if not valid_mask.any():
                        continue

                    vals_valid = vals_arr[valid_mask]
                    chs_valid = [ch for ch, ok in zip(topo_chs, valid_mask) if ok]
                    info_valid = mne.create_info(
                        chs_valid, sfreq=info_topo["sfreq"], ch_types="eeg"
                    )
                    montage_t = mne.channels.make_standard_montage("standard_1020")
                    info_valid.set_montage(montage_t, on_missing="ignore")

                    vlim_tfr = np.nanpercentile(np.abs(vals_valid), 95)
                    vlim_tfr = vlim_tfr if vlim_tfr > 0 else 1.0

                    fig, ax = plt.subplots(figsize=(5, 4))
                    im, _ = mne.viz.plot_topomap(
                        vals_valid,
                        info_valid,
                        axes=ax,
                        show=False,
                        cmap="RdBu_r",
                        vlim=(-vlim_tfr, vlim_tfr),
                    )
                    fig.colorbar(
                        im, ax=ax, fraction=0.046, pad=0.04, label="dB (re: baseline)"
                    )
                    ax.set_title(
                        f"{group_label}  |  {band_name}  power (0–{HABITUATION_WINDOW_SEC[1]:.0f} s)\n"
                        f"baseline: {bl_name}  |  n={valid_mask.sum()} channels",
                        fontsize=9,
                    )
                    fig.tight_layout()
                    fig.savefig(
                        Path(output_dir) / fname_topo, dpi=150, bbox_inches="tight"
                    )
                    plt.close(fig)
                    print(f"      Saved TFR topomap: {fname_topo}")
        if band_power_rows:
            bp_df = pd.DataFrame(band_power_rows)
            bp_csv = (
                Path(output_dir)
                / f"{participant_id}_{session_name}_{suffix}_TFR_band_power_{bl_name}.csv"
            )
            bp_df.to_csv(bp_csv, index=False)
            print(f"      Saved TFR band-power CSV: {bp_csv.name}")


# =============================================================================
# Boxplot and violin plots
# =============================================================================

def plot_spindle_boxplots(pulse_df, session_info, output_dir, suffix=''):
    pid     = session_info['participant_id']
    session = session_info['session_name']
 
    fname  = f'{pid}_{session}_{suffix}_spindle_prepost_boxplots.png'
    fname2 = f'{pid}_{session}_{suffix}_spindle_change_boxplots.png'
    both_done = _already_done(output_dir, fname) and _already_done(output_dir, fname2)
    if both_done:
        return
 
    if pulse_df is None or pulse_df.empty:
        return
 
    channels = sorted({
        col.split('_')[0] for col in pulse_df.columns
        if col.endswith('_pre_sigma_power') or col.endswith('_post_sigma_power')
    })
    if not channels:
        return
 
    pre_post_pairs = [
    ('pre_sigma_power', 'post_sigma_power', 'Sigma power'),
    ('pre_delta_power', 'post_delta_power', 'Delta power'),
    ('pre_theta_power', 'post_theta_power', 'Theta power'),
    ('pre_alpha_power', 'post_alpha_power', 'Alpha power'),
]
 
    # Palette — sham: blue family, active: red family
    colors = {
        'sham_pre':    '#A8C8E8',
        'sham_post':   '#1A5276',
        'active_pre':  '#F5B8B8',
        'active_post': '#922B21',
    }
    labels_order = ['sham_pre', 'sham_post', 'active_pre', 'active_post']
    tick_labels  = ['Sham\nPre', 'Sham\nPost', 'Active\nPre', 'Active\nPost']
 
    # ── Figure 1: pre vs post ──────────────────────────────────────────────
    if not _already_done(output_dir, fname):
        nrow = len(channels)
        ncol = len(pre_post_pairs)
 
        # --- Pass 1: compute raw p-values for every channel × feature ×
        #     condition, so FDR correction can run across channels within
        #     each (feature, condition) combination. ---
        raw_p = {}  # (feat_idx, condition) -> list of p-values, indexed by channel row
        for col_idx, (pre_feat, post_feat, feat_label) in enumerate(pre_post_pairs):
            for condition in ['sham', 'active']:
                pvals = []
                for ch in channels:
                    pre_col  = f'{ch}_{pre_feat}'
                    post_col = f'{ch}_{post_feat}'
                    if pre_col in pulse_df.columns and post_col in pulse_df.columns:
                        mask = pulse_df['group'] == condition
                        pre_vals  = pulse_df.loc[mask, pre_col].dropna().values
                        post_vals = pulse_df.loc[mask, post_col].dropna().values
                        pvals.append(_paired_wilcoxon(pre_vals, post_vals))
                    else:
                        pvals.append(np.nan)
                raw_p[(col_idx, condition)] = pvals
 
        corrected_p = {
            key: _bh_fdr_correct(vals) for key, vals in raw_p.items()
        }
 
        fig, axes = plt.subplots(
            nrow, ncol,
            figsize=(ncol * 3.6, nrow * 3.2),
            squeeze=False,
        )
 
        for row_idx, ch in enumerate(channels):
            for col_idx, (pre_feat, post_feat, feat_label) in enumerate(pre_post_pairs):
                ax = axes[row_idx][col_idx]
                groups_data = []
                for key in labels_order:
                    condition, timing = key.split('_', 1)
                    col_name = (f'{ch}_{pre_feat}' if timing == 'pre'
                                else f'{ch}_{post_feat}')
                    mask = pulse_df['group'] == condition
                    groups_data.append(
                        pulse_df.loc[mask, col_name].dropna().values
                        if col_name in pulse_df.columns else np.array([])
                    )
 
                bp = ax.boxplot(
                    groups_data,
                    patch_artist=True,
                    widths=0.50,
                    medianprops=dict(color='white', linewidth=2.0),
                    whiskerprops=dict(linewidth=1.0, color='#444'),
                    capprops=dict(linewidth=1.0, color='#444'),
                    flierprops=dict(marker='o', markersize=2.5, alpha=0.35,
                                    linestyle='none', markerfacecolor='#666'),
                    boxprops=dict(linewidth=0.8),
                )
                for patch, key in zip(bp['boxes'], labels_order):
                    patch.set_facecolor(colors[key])
                    patch.set_alpha(0.88)
 
                # Median connector lines
                for x_pre, x_post, condition in [(1, 2, 'sham'), (3, 4, 'active')]:
                    pre_col  = f'{ch}_{pre_feat}'
                    post_col = f'{ch}_{post_feat}'
                    if pre_col in pulse_df.columns and post_col in pulse_df.columns:
                        mask     = pulse_df['group'] == condition
                        pre_med  = pulse_df.loc[mask, pre_col].median()
                        post_med = pulse_df.loc[mask, post_col].median()
                        ax.plot([x_pre, x_post], [pre_med, post_med],
                                color='#333', lw=1.2, ls='--',
                                alpha=0.65, zorder=5)
 
                ax.axvline(2.5, color='#AAAAAA', lw=0.8, ls=':', alpha=0.8)
                ax.set_xticks([1, 2, 3, 4])
                ax.set_xticklabels(tick_labels, fontsize=8)
                ax.axhline(0, color='grey', lw=0.6, ls='--', alpha=0.5)
                ax.tick_params(labelsize=7, length=3, pad=2)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
 
                # y-label only on left column
                if col_idx == 0:
                    ax.set_ylabel(f'{ch}', fontsize=8, fontweight='bold', labelpad=4)
 
                # Panel title only on top row
                if row_idx == 0:
                    ax.set_title(feat_label, fontsize=9, fontweight='bold', pad=5)
 
                # Sham / Active group labels below x-axis on bottom row
                if row_idx == nrow - 1:
                    for x_center, lbl in [(1.5, 'Sham'), (3.5, 'Active')]:
                        ax.text(x_center, -0.22, lbl,
                                transform=ax.get_xaxis_transform(),
                                ha='center', fontsize=8, color='#555')
 
                # --- Significance brackets: pre vs post within each condition ---
                sham_p   = corrected_p[(col_idx, 'sham')][row_idx]
                active_p = corrected_p[(col_idx, 'active')][row_idx]
 
                y_sham, _   = _bracket_y(groups_data[0], groups_data[1])
                y_active, _ = _bracket_y(groups_data[2], groups_data[3])
                y_top = max(y_sham, y_active)
 
                _add_sig_bracket(ax, 1, 2, y_top, sham_p)
                _add_sig_bracket(ax, 3, 4, y_top, active_p)
 
                # give brackets room to breathe
                ylo, yhi = ax.get_ylim()
                ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.12)
 
        fig.suptitle(
            f'{pid} – {session}: pre vs post power  [Sham | Active]\n'
            f'Baseline: −3 to 0 s  |  Response: 0 to +5 s  (re: TUS onset)\n'
            f'Wilcoxon signed-rank, BH–FDR corrected across channels '
            f'(* p<0.05, ** p<0.01, *** p<0.001)',
            fontsize=10, fontweight='bold', y=1.03,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(Path(output_dir) / fname, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'    Saved pre/post boxplots: {fname}')
 
    # ── Figure 2: change metrics ───────────────────────────────────────────
    ratio_features = [
        ('sigma_power_change', 'Sigma power change\n(post−pre) / pre'),
        ('post_ptp_uv',        'Post-pulse peak-to-peak (µV)'),
    ]
 
    if not _already_done(output_dir, fname2):
        nrow2 = len(channels)
        ncol2 = len(ratio_features)
 
        # --- Pass 1: raw p-values per channel for each feature column,
        #     corrected across channels within that feature. ---
        raw_p2 = {}
        for col_idx, (feat_suffix, feat_label) in enumerate(ratio_features):
            pvals = []
            for ch in channels:
                col_name = f'{ch}_{feat_suffix}'
                if col_name in pulse_df.columns:
                    sham_vals   = pulse_df.loc[pulse_df['group'] == 'sham',   col_name].dropna().values
                    active_vals = pulse_df.loc[pulse_df['group'] == 'active', col_name].dropna().values
                    pvals.append(_unpaired_mannwhitney(sham_vals, active_vals))
                else:
                    pvals.append(np.nan)
            raw_p2[col_idx] = pvals
 
        corrected_p2 = {key: _bh_fdr_correct(vals) for key, vals in raw_p2.items()}
 
        fig2, axes2 = plt.subplots(
            nrow2, ncol2,
            figsize=(ncol2 * 3.6, nrow2 * 3.2),
            squeeze=False,
        )
 
        condition_colors = {'sham': '#2C7BB6', 'active': '#C0392B'}
 
        for row_idx, ch in enumerate(channels):
            for col_idx, (feat_suffix, feat_label) in enumerate(ratio_features):
                ax       = axes2[row_idx][col_idx]
                col_name = f'{ch}_{feat_suffix}'
                sham_vals   = (pulse_df.loc[pulse_df['group'] == 'sham',   col_name]
                               .dropna().values
                               if col_name in pulse_df.columns else np.array([]))
                active_vals = (pulse_df.loc[pulse_df['group'] == 'active', col_name]
                               .dropna().values
                               if col_name in pulse_df.columns else np.array([]))
 
                bp = ax.boxplot(
                    [sham_vals, active_vals],
                    patch_artist=True,
                    widths=0.45,
                    medianprops=dict(color='white', linewidth=2.0),
                    whiskerprops=dict(linewidth=1.0, color='#444'),
                    capprops=dict(linewidth=1.0, color='#444'),
                    flierprops=dict(marker='o', markersize=2.5, alpha=0.35,
                                    linestyle='none', markerfacecolor='#666'),
                    boxprops=dict(linewidth=0.8),
                )
                bp['boxes'][0].set_facecolor(condition_colors['sham'])
                bp['boxes'][0].set_alpha(0.85)
                bp['boxes'][1].set_facecolor(condition_colors['active'])
                bp['boxes'][1].set_alpha(0.85)
 
                ax.set_xticks([1, 2])
                ax.set_xticklabels(['Sham', 'Active'], fontsize=9)
                ax.axhline(0, color='grey', lw=0.7, ls='--', alpha=0.6)
                ax.tick_params(labelsize=7, length=3, pad=2)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
 
                if col_idx == 0:
                    ax.set_ylabel(f'{ch}', fontsize=8, fontweight='bold', labelpad=4)
                if row_idx == 0:
                    ax.set_title(feat_label, fontsize=9, fontweight='bold', pad=5)
 
                # --- Significance bracket: sham vs active ---
                p_corr = corrected_p2[col_idx][row_idx]
                y_top, _ = _bracket_y(sham_vals, active_vals)
                _add_sig_bracket(ax, 1, 2, y_top, p_corr)
 
                ylo, yhi = ax.get_ylim()
                ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.12)
 
        fig2.suptitle(
            f'{pid} – {session}: sigma change & amplitude  [Sham vs Active]\n'
            f'Baseline: −3 to 0 s  |  Response: 0 to +5 s  (re: TUS onset)\n'
            f'Mann–Whitney U, BH–FDR corrected across channels '
            f'(* p<0.05, ** p<0.01, *** p<0.001)',
            fontsize=10, fontweight='bold', y=1.03,
        )
        fig2.tight_layout(rect=[0, 0, 1, 0.97])
        fig2.savefig(Path(output_dir) / fname2, dpi=200, bbox_inches='tight')
        plt.close(fig2)
        print(f'    Saved change boxplots: {fname2}')


def plot_spindle_violins(pulse_df, session_info, output_dir, suffix=''):
    pid     = session_info['participant_id']
    session = session_info['session_name']
 
    fname = f'{pid}_{session}_{suffix}_spindle_change_violins.png'
    if _already_done(output_dir, fname):
        return
 
    if pulse_df is None or pulse_df.empty:
        return
 
    channels = sorted({
        col.split('_')[0] for col in pulse_df.columns
        if col.endswith('_sigma_power_change') or col.endswith('_post_ptp_uv')
    })
    if not channels:
        return
 
    ratio_features = [
        ('sigma_power_change', 'Sigma power change\n(post−pre) / pre'),
        ('post_ptp_uv',        'Post-pulse peak-to-peak (µV)'),
    ]
    palette = {'sham': '#2C7BB6', 'active': '#C0392B'}
 
    nrow = len(channels)
    ncol = len(ratio_features)
 
    # --- Pass 1: raw p-values per channel for each feature column,
    #     corrected across channels within that feature (same test
    #     family as Figure 2 above — Mann-Whitney, sham vs active). ---
    raw_p = {}
    for col_idx, (feat_suffix, feat_label) in enumerate(ratio_features):
        pvals = []
        for ch in channels:
            col_name = f'{ch}_{feat_suffix}'
            if col_name in pulse_df.columns:
                sham_vals   = pulse_df.loc[pulse_df['group'] == 'sham',   col_name].dropna().values
                active_vals = pulse_df.loc[pulse_df['group'] == 'active', col_name].dropna().values
                pvals.append(_unpaired_mannwhitney(sham_vals, active_vals))
            else:
                pvals.append(np.nan)
        raw_p[col_idx] = pvals
 
    corrected_p = {key: _bh_fdr_correct(vals) for key, vals in raw_p.items()}
 
    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(ncol * 3.6, nrow * 3.4),
        squeeze=False,
    )
 
    rng = np.random.default_rng(42)
 
    for row_idx, ch in enumerate(channels):
        for col_idx, (feat_suffix, feat_label) in enumerate(ratio_features):
            ax       = axes[row_idx][col_idx]
            col_name = f'{ch}_{feat_suffix}'
            groups_data, positions, colors_list = [], [], []
 
            for pos, group in enumerate(['sham', 'active'], start=1):
                vals = (pulse_df.loc[pulse_df['group'] == group, col_name]
                        .dropna().astype(float).values
                        if col_name in pulse_df.columns else np.array([]))
                groups_data.append(vals)
                positions.append(pos)
                colors_list.append(palette[group])
 
            # Violin body
            has_data = [len(d) >= 3 for d in groups_data]
            if any(has_data):
                parts = ax.violinplot(
                    [d if ok else [np.nan]
                     for d, ok in zip(groups_data, has_data)],
                    positions=positions,
                    showmedians=False,
                    showextrema=False,
                    widths=0.65,
                )
                for body, color in zip(parts['bodies'], colors_list):
                    body.set_facecolor(color)
                    body.set_alpha(0.40)
                    body.set_edgecolor('none')
 
            # Overlaid boxplot
            bp = ax.boxplot(
                groups_data,
                positions=positions,
                widths=0.14,
                patch_artist=True,
                medianprops=dict(color='white', linewidth=2.0),
                whiskerprops=dict(linewidth=0.9, color='#333'),
                capprops=dict(linewidth=0.9, color='#333'),
                flierprops=dict(marker='o', markersize=2.0, alpha=0.30,
                                linestyle='none', markerfacecolor='#555'),
                boxprops=dict(linewidth=0.7),
            )
            for patch, color in zip(bp['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.92)
 
            # Jittered individual points
            for pos, vals, color in zip(positions, groups_data, colors_list):
                if len(vals):
                    jitter = rng.uniform(-0.055, 0.055, len(vals))
                    ax.scatter(pos + jitter, vals,
                               s=6, color=color, alpha=0.35,
                               zorder=3, linewidths=0)
 
            ax.axhline(0, color='grey', lw=0.7, ls='--', alpha=0.6)
            ax.set_xticks(positions)
            ax.set_xticklabels(['Sham', 'Active'], fontsize=9)
            ax.tick_params(labelsize=7, length=3, pad=2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
 
            # Labels only on edges
            if col_idx == 0:
                ax.set_ylabel(f'{ch}', fontsize=8, fontweight='bold', labelpad=4)
            if row_idx == 0:
                ax.set_title(feat_label, fontsize=9, fontweight='bold', pad=5)
 
            # --- Significance bracket: sham vs active ---
            p_corr = corrected_p[col_idx][row_idx]
            y_top, _ = _bracket_y(groups_data[0], groups_data[1])
            _add_sig_bracket(ax, positions[0], positions[1], y_top, p_corr)
 
            ylo, yhi = ax.get_ylim()
            ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.12)
 
    fig.suptitle(
        f'{pid} – {session}: sigma change & amplitude  [violin + box]\n'
        f'Baseline: −3 to 0 s  |  Response: 0 to +5 s  (re: TUS onset)\n'
        f'Mann–Whitney U, BH–FDR corrected across channels '
        f'(* p<0.05, ** p<0.01, *** p<0.001)',
        fontsize=10, fontweight='bold', y=1.03,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(Path(output_dir) / fname, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved violin plots: {fname}')

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
    try:
        print(f'\n[1] Load preprocessed: {participant_id} / {target}')
        raw = load_preprocessed(participant_id, target)
        raw.load_data()

        hypno_int, hypno_str, staging_ok = run_sleep_staging_from_fif(
            raw, session_name, participant_id, participant_output_dir
        )

        snap_p    = snapshot_path(participant_id, target)
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
        pulse_results = run_pulse_level_analysis(raw, vmrk, hypno_int, hypno_up, freq_band,session_name, participant_id, target, is_adaptation, participant_output_dir)

        # Load burst times (if any) so we can overlay them on the spectrogram
        suffix_tmp = 'nrem' if hypno_int is not None else 'full_recording'
        burst_csv_tmp = (Path(participant_output_dir) /
                         f'{participant_id}_{session_name}_{suffix_tmp}_per_pulse_features.csv')
        burst_times_for_overlay = None
        if burst_csv_tmp.exists():
            try:
                _bdf = pd.read_csv(burst_csv_tmp)
                if 'burst_time_s' in _bdf.columns:
                    burst_times_for_overlay = _bdf['burst_time_s'].values
                del _bdf
            except Exception as exc:
                print(f'    Could not load burst times for spectrogram overlay: {exc}')

        safe_plot(plot_spectrogram, raw, hypno_int,session_name, participant_id, participant_output_dir,burst_times_for_overlay)
        suffix   = 'nrem' if hypno_int is not None else 'full_recording'
        burst_csv = (Path(participant_output_dir) /
                     f'{participant_id}_{session_name}_{suffix}_per_pulse_features.csv')
        bursts_df = pd.read_csv(burst_csv) if burst_csv.exists() else pd.DataFrame()

        focus_ch = None
        if not bursts_df.empty:
            focus_ch = safe_plot_returning(
                plot_erps, raw, bursts_df, freq_band,
                session_name, participant_id,
                participant_output_dir, suffix,
            )
            safe_plot(plot_erps_500ms, raw, bursts_df, freq_band, session_name, participant_id,participant_output_dir, suffix,)
            safe_plot(plot_erp_topomap_evolution, raw, bursts_df, freq_band, session_name, participant_id, participant_output_dir, suffix)
            safe_plot(plot_tfrs, raw, bursts_df, freq_band,session_name, participant_id,participant_output_dir, suffix,focus_ch,)
            safe_plot(compute_evoked_band_responses, raw, bursts_df, session_name, participant_id, participant_output_dir, suffix, hypno_int)

        del raw, hypno_up
        gc.collect()
        print('    Raw released')

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

    non_numeric = {'participant_id', 'session', 'analysis_scope',
                   'condition', 'group', 'brain_state'}
    for _, session_row in df.iterrows():
        if bool(session_row.get('is_adaptation', False)):
            continue
        session_name   = session_row.get('session')
        analysis_scope = session_row.get('analysis_scope', '')
        suffix         = 'nrem' if 'NREM' in str(analysis_scope) else 'full_recording'
        pulse_csv      = out_dir / f'{participant_id}_{session_name}_{suffix}_per_pulse_features.csv'
        if not pulse_csv.exists():
            print(f'  Boxplot/violin skipped ({session_name}): pulse CSV not found')
            continue
        try:
            pulse_df = pd.read_csv(pulse_csv)
        except Exception as exc:
            print(f'  Boxplot/violin skipped ({session_name}): could not read CSV — {exc}')
            continue
        for col in pulse_df.columns:
            if col not in non_numeric:
                pulse_df[col] = pd.to_numeric(pulse_df[col], errors='coerce')
        session_info = {'participant_id': participant_id, 'session_name': session_name}
        safe_plot(plot_spindle_boxplots, pulse_df, session_info, str(out_dir), suffix)
        safe_plot(plot_spindle_violins,  pulse_df, session_info, str(out_dir), suffix)
    safe_plot(plot_sw_locked_sigma_thalamus_vs_ventricle, participant_id, str(out_dir))
    safe_plot(plot_sw_locked_sigma_single_region, participant_id, 'thalamus', str(out_dir))
    safe_plot(plot_sw_locked_sigma_single_region, participant_id, 'ventricle', str(out_dir))
    safe_plot(plot_slowwave_region_comparison, participant_id, str(out_dir))
    safe_plot(plot_region_comparison_boxplots, participant_id, str(out_dir))

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
    else:
        print('\nNo participant tables produced.')
