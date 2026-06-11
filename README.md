# TUNES — Thalamic Ultrasound Neuromodulation 

A two-stage EEG analysis pipeline for characterizing neural responses to transcranial ultrasound stimulation (TUS). The pipeline preprocesses multi-session BrainVision recordings, runs automated sleep staging, and extracts oscillatory features (spindles, slow waves, spectral power) to build a per-participant neural response profile.

---

## Background

This project addresses a methodological gap in TUS research: while subject-specific electric field modeling has been established in transcranial electrical stimulation (Kasten et al., 2019), no equivalent framework exists for TUS that links the realized acoustic dose at a neural target to downstream EEG responses.

TUNES implements a two-level characterization framework:

- **Level 1 — Acoustic dose**: quantifies what the ultrasound actually delivered to thalamic nuclei per participant
- **Level 2 — Neural response profile**: extracts sleep oscillatory features from pre- and post-stimulation EEG to characterize how the brain responded

Together, these levels allow us to ask whether the observed EEG response is coherent with the delivered ultrasound dose — at the single-subject level.

---

## Study Design

Each participant completes three sessions:

| Session | Role |
|---|---|
| `adapt` | Adaptation night — used to derive the individual spindle frequency |
| `thalamus` | Active TUS targeted at thalamic nuclei |
| `ventricle` | Control TUS targeted at the lateral ventricle (sham target) |

The pipeline processes each session in order. The adaptation session must be analyzed before the thalamus/ventricle sessions because it calibrates the spindle detection frequency band for that participant.

---

## Repository Structure

```
TUNES/
├── subjects/               # Raw BrainVision data (.vhdr, .vmrk, .eeg) per participant
│   └── <pid>/
│       ├── <adapt_folder>/
│       ├── <thalamus_folder>/
│       └── <ventricle_folder>/
├── preprocessed/           # Cleaned .fif files written by preprocess.py
│   └── <pid>/
│       ├── adapt_raw.fif
│       ├── thalamus_raw.fif
│       ├── ventricle_raw.fif
│       └── *_snapshot.npy  # Pre-ICA snapshots for QC
├── results/                # CSVs, figures, and epoch files written by analyze.py
│   └── <pid>/
├── preprocess.py
└── analyze.py
```

---

## Pipeline Overview

### Stage 1 — `preprocess.py`

Reads raw BrainVision files for each participant and session, and writes a clean `.fif` file ready for analysis.

Steps per session:
1. Discovers session folders and `.vhdr` files by matching on session keywords (`adapt`, `thalamus`, `ventricle`)
2. Resamples to 500 Hz before loading into RAM (reduces memory footprint)
3. Applies a 0.1–40 Hz bandpass filter and a 50 Hz notch filter
4. Runs FastICA (20 components) and auto-rejects ocular and cardiac artefacts using EOG/ECG detection
5. Saves a pre-ICA snapshot (first 5 minutes, frontal/central/parietal channels) as `.npy` for QC comparison
6. Writes the cleaned recording to `preprocessed/<pid>/<session>_raw.fif`

### Stage 2 — `analyze.py`

Reads the preprocessed `.fif` files and runs the full feature extraction and visualisation pipeline.

Steps per session:
1. **Sleep staging** — runs YASA on a lightweight 3-channel, 100 Hz copy of the raw data (C4, HEOG, EMG); hypnogram is cached to CSV so it only runs once
2. **Individual spindle frequency** — on the adaptation session only, fits a Welch PSD over NREM epochs and finds the spectral peak in the 12–15 Hz sigma band; the resulting personalized band (peak ± 1.5 Hz) is saved and reused for thalamus/ventricle sessions
3. **Spindle detection** — YASA spindle detector on C3/C4, restricted to NREM N2/N3; outputs density, amplitude, frequency, duration, and RMS per channel
4. **Slow-wave detection** — YASA slow-wave detector on F3/F4 (0.5–4 Hz), same NREM restriction
5. **Spectral band power** — Welch PSD across delta, theta, and sigma bands on a set of central/frontal/parietal channels
6. **Pulse-level analysis** — parses the `.vmrk` marker file, groups TUS bursts into active (60W) vs sham (1W) conditions, and extracts per-burst EEG features in a ±3s/5s epoch window; bursts occurring within 3.5 s of a detected spindle onset are excluded to avoid contamination; saves per-pulse feature CSV and MNE Epochs `.fif`
7. **Event-locked spindle features** — computes spindle probability and latency within the post-burst window for active vs sham bursts
8. **Visualisations** — QC raw-vs-preprocessed traces, spectrograms, topoplots, boxplots, violin plots, ERPs, and TFR topomaps

All features are assembled into a per-participant `<pid>_session_features.csv` and a group-level `all_session_features.csv`.

---

## Requirements

```
Python >= 3.9
mne
yasa
numpy
pandas
scipy
matplotlib
```

Install dependencies:

```bash
pip install mne yasa numpy pandas scipy matplotlib
```

---

## Configuration

Before running, update the path constants at the top of each script:

**`preprocess.py`**
```python
DATA_ROOT        = '/path/to/TUNES/subjects'
PREPROCESSED_DIR = '/path/to/TUNES/preprocessed'
LOCAL_WORK_DIR   = '/path/to/TUNES/tunes_work'   # only used if COPY_SESSIONS_TO_LOCAL = True
PARTICIPANTS     = ['03', '06', '08']
```

**`analyze.py`**
```python
DATA_ROOT        = '/path/to/TUNES/subjects'
PREPROCESSED_DIR = '/path/to/TUNES/preprocessed'
OUTPUT_DIR       = '/path/to/TUNES/results'
PARTICIPANTS     = ['03', '06', '08']
```

---

## Usage

**Run preprocessing for all participants:**
```bash
python preprocess.py
```

**Run preprocessing for specific participants:**
```bash
python preprocess.py --participants 03 06
```

**Run analysis (requires preprocessed .fif files):**
```bash
python analyze.py
```

**Run analysis for specific participants:**
```bash
python analyze.py --participants 03
```

> The adaptation session must be preprocessed and analyzed before the thalamus/ventricle sessions. Both scripts process sessions in `adapt → thalamus → ventricle` order by default.

---

## Output Files

| File | Description |
|---|---|
| `preprocessed/<pid>/<session>_raw.fif` | Cleaned MNE Raw object |
| `preprocessed/<pid>/<session>_raw_snapshot.npy` | Pre-ICA EEG snapshot for QC |
| `results/<pid>/<pid>_<session>_hypnogram.csv` | YASA sleep stage predictions |
| `results/<pid>/<pid>_<session>_spindles.csv` | Detected spindle events |
| `results/<pid>/<pid>_<session>_slowwaves.csv` | Detected slow-wave events |
| `results/<pid>/<pid>_individual_spindle_freq.csv` | Personalized sigma band |
| `results/<pid>/<pid>_<session>_*_per_pulse_features.csv` | Per-burst EEG features |
| `results/<pid>/<pid>_<session>_*_epochs-epo.fif` | MNE Epochs around TUS bursts |
| `results/<pid>/<pid>_session_features.csv` | Aggregated feature table per participant |
| `results/all_session_features.csv` | Group-level feature table |

---

## Notes

- Set `RUN_SLEEP_STAGING = False` in `analyze.py` to skip YASA staging (useful when the staging channel is missing or for a quick feature pass)
- Set `COPY_SESSIONS_TO_LOCAL = True` in `preprocess.py` if you are streaming data from an external drive and want a local working copy
- ICA failures are caught and logged; preprocessing continues without ICA rather than crashing
- All plots are saved non-interactively (`matplotlib Agg` backend) — no display required

---

## Reference

Kasten, F. H., et al. (2019). Integrating electric field modeling and neuroimaging to explain inter-individual variability of tACS effects. *Nature Communications*, 10(1), 5427.