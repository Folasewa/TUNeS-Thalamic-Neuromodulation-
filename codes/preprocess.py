"""
This script does:
For each participant and session:
  1. Find the raw BrainVision (.vhdr) files
  2. Downsample to RESAMPLE_FREQ (256 Hz) *before* loading into RAM
  3. Bandpass filter (0.1–40 Hz) + notch filter (50 Hz)
  4. Run ICA to remove ocular and cardiac artefacts
  5. Save the cleaned raw as  TUNES/preprocessed/<pid>/<session_target>_raw.fif
  6. Save a pre-ICA snapshot (first 5 min, VIZ_CHANNELS only) for QC plots                 
"""

import argparse
import gc
import re
import shutil
import time
import traceback
from pathlib import Path
 
import json
import mne
import numpy as np
 
mne.set_log_level('WARNING')

DATA_ROOT      = '/Users/folasewaabdulsalam/Downloads/TUNES/subjects'
PREPROCESSED_DIR = '/Users/folasewaabdulsalam/Downloads/TUNES/preprocessed'
LOCAL_WORK_DIR = '/Users/folasewaabdulsalam/Downloads/TUNES/tunes_work'
PARTICIPANTS   = ['03', '06', '08']

COPY_SESSIONS_TO_LOCAL = False
REFRESH_LOCAL_COPY     = False

RESAMPLE_FREQ  = 500    # Hz
BANDPASS_LOW   = 0.1
BANDPASS_HIGH  = 40.0
NOTCH_FREQ     = 50.0
 
ICA_N_COMPONENTS = 20
VIZ_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'Fz',
                 'C3',  'C4',  'Cz',
                 'P3',  'P4',  'Pz',
                 'O1',  'O2']

KNOWN_TARGETS = {'adapt', 'thalamus', 'ventricle', 'ventricles'}
SNAPSHOT_SECS = 300.0


def find_sessions(participant_folder):
    sessions = {'adapt': None, 'thalamus': None, 'ventricle': None}
    for folder in sorted(Path(participant_folder).iterdir()):
        if not folder.is_dir():
            continue
        text = folder.name.lower() + ' ' + ' '.join(
            p.name.lower() for p in folder.glob('*.vhdr')
        )
        if 'adapt' in text:
            sessions['adapt'] = str(folder)
        elif 'thalamus' in text:
            sessions['thalamus'] = str(folder)
        elif 'ventricle' in text or 'ventricles' in text:
            sessions['ventricle'] = str(folder)
    return sessions

def prepare_local_session(session_folder, participant_id, target):
    if not COPY_SESSIONS_TO_LOCAL:
        return str(session_folder)
    src = Path(session_folder)
    dst = Path(LOCAL_WORK_DIR) / participant_id / target / src.name
    if dst.exists() and not REFRESH_LOCAL_COPY:
        return str(dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return str(dst)

def find_vhdr_files(session_folder):
    files = list(Path(session_folder).glob('*.vhdr'))
    if not files:
        raise FileNotFoundError(f'No .vhdr files found in {session_folder}')
 
    def sort_key(path):
        match = re.match(r'([a-zA-Z_\-]+)(\d*)$', path.stem)
        base   = match.group(1) if match else path.stem
        number = int(match.group(2) or 0) if match else 0
        return base, number
 
    files  = sorted(files, key=sort_key)
    target = sort_key(files[0])[0]
    return [str(p) for p in files if sort_key(p)[0] == target], target

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

def load_and_resample(vhdr_path):
    """Load one block, resampling to RESAMPLE_FREQ.
    Returns (raw, original_sfreq) so callers can persist the hardware rate."""
    print(f'    Loading {Path(vhdr_path).name}')
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose=False)
    original_sfreq = raw.info['sfreq']
    print(f'    {len(raw.ch_names)} channels @ {original_sfreq:.0f} Hz')
    if original_sfreq > RESAMPLE_FREQ:
        print(f'    Resampling {original_sfreq:.0f} → {RESAMPLE_FREQ} Hz (lazy)')
        raw.resample(RESAMPLE_FREQ, npad='auto')
    raw.load_data()
    set_channel_types(raw)
    return raw, original_sfreq


def preprocess(raw):
    """Filter + ICA. Modifies raw in place, returns it."""
    raw.set_montage(
        mne.channels.make_standard_montage('standard_1020'), on_missing='ignore'
    )
    raw.filter(BANDPASS_LOW, BANDPASS_HIGH, verbose=False)
    raw.notch_filter(NOTCH_FREQ, verbose=False)
    # Re-reference to average
    raw.set_eeg_reference('average', projection=False, verbose=False)
    print('Re-referenced to Average')
 
    ART_THRESHOLD_UV = 150.0
    eeg_chs = [ch for idx, ch in enumerate(raw.ch_names)
               if mne.channel_type(raw.info, idx) == 'eeg']
    if eeg_chs:
        data_uv = raw.get_data(picks=eeg_chs) * 1e6
        flags   = np.abs(data_uv) > ART_THRESHOLD_UV
        total   = flags.any(axis=0).mean() * 100
        print(f'    Artefact rate (±{ART_THRESHOLD_UV} µV): {total:.1f}% of samples')
        del data_uv, flags
        gc.collect()
 
    if ICA_N_COMPONENTS is not None:
        try:
            eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
            n_comps   = min(ICA_N_COMPONENTS, len(eeg_picks) - 1)
            print(f'    ICA: fitting {n_comps} components …')
            ica = mne.preprocessing.ICA(
                n_components=n_comps, method='fastica',
                random_state=42, max_iter='auto',
            )
            raw_fit = raw.copy().filter(1.0, None, verbose=False)
            ica.fit(raw_fit, picks=eeg_picks, verbose=False)
            del raw_fit
            gc.collect()
 
            exclude = []
            eog_chs = [ch for ch in raw.ch_names
                       if mne.channel_type(raw.info, raw.ch_names.index(ch)) == 'eog']
            if eog_chs:
                try:
                    idx, _ = ica.find_bads_eog(raw, ch_name=eog_chs, verbose=False)
                    exclude.extend(idx)
                except Exception as e:
                    print(f'      EOG detection skipped: {e}')
            else:
                frontal = [ch for ch in ['Fp1', 'Fp2'] if ch in raw.ch_names]
                if frontal:
                    try:
                        idx, _ = ica.find_bads_eog(raw, ch_name=frontal, verbose=False)
                        exclude.extend(idx)
                    except Exception:
                        pass
            try:
                idx, _ = ica.find_bads_ecg(raw, method='correlation', verbose=False)
                exclude.extend(idx)
            except Exception as e:
                print(f'      ECG detection skipped: {e}')
 
            ica.exclude = list(set(exclude))
            print(f'    ICA: removing {len(ica.exclude)} components {ica.exclude}')
            ica.apply(raw)
            del ica
            gc.collect()
        except Exception as exc:
            print(f'    ICA failed ({exc}) — skipping')
 
    return raw


def preprocess_session(participant_id, session_folder, target, out_dir):
    """
    Full preprocessing for one session. Returns path to saved .fif or None.
    """
    fif_path = Path(out_dir) / f'{target}_raw.fif'
    if fif_path.exists():
        print(f'    Already preprocessed: {fif_path.name} — skipping')
        return str(fif_path)
 
    print(f'\n  [{target.upper()}] {session_folder}')
    t0 = time.time()
 
    try:
        vhdr_files, _ = find_vhdr_files(session_folder)
        loaded        = [load_and_resample(v) for v in vhdr_files]
        raws, sfreqs  = zip(*loaded)
        original_sfreq = sfreqs[0]   # hardware rate before resampling
        raw  = raws[0] if len(raws) == 1 else mne.concatenate_raws(list(raws))
        del raws, loaded
        gc.collect()

        # Save hardware sfreq so analysis.py never needs to open the .vhdr
        info_path = Path(out_dir) / f'{target}_info.json'
        with open(str(info_path), 'w') as _f:
            json.dump({'original_sfreq': original_sfreq}, _f)
        print(f'    Saved {info_path.name} (original_sfreq={original_sfreq:.0f} Hz)')
 
        viz_present  = [ch for ch in VIZ_CHANNELS if ch in raw.ch_names]
        snap_secs    = min(SNAPSHOT_SECS, raw.times[-1]) if SNAPSHOT_SECS else raw.times[-1]
        snap_samples = int(snap_secs * raw.info['sfreq'])
        if viz_present:
            snapshot = raw.get_data(picks=viz_present, start=0, stop=snap_samples) * 1e6
            snap_path = Path(out_dir) / f'{target}_raw_snapshot.npy'
            np.save(str(snap_path), snapshot)
            np.save(str(Path(out_dir) / f'{target}_snapshot_channels.npy'),
                    np.array(viz_present))
            del snapshot
            gc.collect()
            print(f'    Saved pre-ICA snapshot ({snap_secs:.0f} s, {len(viz_present)} ch)')
 
        raw = preprocess(raw)
 
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        raw.save(str(fif_path), overwrite=True, verbose=False)
        elapsed = (time.time() - t0) / 60
        print(f'    Saved → {fif_path.name}  ({elapsed:.1f} min)')
 
        del raw
        gc.collect()
        return str(fif_path)
 
    except Exception as exc:
        print(f'    FAILED: {exc}')
        traceback.print_exc()
        return None
    

def preprocess_participant(participant_id):
    print('\n' + '=' * 70)
    print(f'PARTICIPANT: {participant_id}')
    print('=' * 70)
 
    participant_folder = Path(DATA_ROOT) / participant_id
    out_dir            = Path(PREPROCESSED_DIR) / participant_id
    out_dir.mkdir(parents=True, exist_ok=True)
 
    if not participant_folder.exists():
        print(f'  Missing: {participant_folder}')
        return
 
    sessions = find_sessions(participant_folder)
    print('  Sessions:', {k: Path(v).name if v else None for k, v in sessions.items()})
 
    for target in ['adapt', 'thalamus', 'ventricle']:
        session_folder = sessions[target]
        if not session_folder:
            print(f'  Skipping missing {target} session')
            continue
        if not list(Path(session_folder).glob('*.vhdr')):
            print(f'  Skipping {target}: no .vhdr files')
            continue
 
        local = prepare_local_session(session_folder, participant_id, target)
        preprocess_session(participant_id, local, target, str(out_dir))
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TUNES preprocessing pipeline')
    parser.add_argument('--participants', nargs='+', default=None,
                        help='Override PARTICIPANTS list, e.g. --participants 01 02')
    args = parser.parse_args()
 
    participants = args.participants if args.participants else PARTICIPANTS
    print(f'Preprocessing {len(participants)} participant(s): {participants}')
 
    for pid in participants:
        preprocess_participant(pid)
 
    print('\nDone. Preprocessed files written to:', PREPROCESSED_DIR)
    print('Run analyze.py next.')