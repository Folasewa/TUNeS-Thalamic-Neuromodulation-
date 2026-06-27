# =============================================================================
# dose_response.py  —  Level 3: Response Profile, Statistics & Dose–Response
# =============================================================================
#
# WHERE THIS SITS IN THE PIPELINE
# --------------------------------
#   Level 1 (acoustic_report script)
#       Reads kPlan RESULTS.h5, SimNIBS segmentation, and participant nucleus
#       masks.  Reports acoustic metrics for EVERY region the beam touches —
#       thalamus, ventricle, cortex, skull, eyes, etc. — across three tiers:
#           Tier1  SimNIBS tissue groups (brain, skull, scalp, eyes)
#           Tier2  Participant nucleus masks  
#           Tier3  Harvard-Oxford atlas regions
#
#       One CSV per H5 file, multiple rows per sonication (one row per region).
#       The CSV does NOT ship with participant_id or intended_target because
#       kPlan has no knowledge of those — they must be added manually.
#
#       Two columns added to Level 1 before saving:
#           participant_id   e.g. 'tunes-02'
#           intended_target  'thalamus' or 'ventricle'  ← what you aimed at
#                         
#
#       Saved as: {pid}_{intended_target}_ANALYSIS.csv
#       e.g.  tunes-02_thalamus_ANALYSIS.csv   (thalamus session, all regions)
#             tunes-02_ventricle_ANALYSIS.csv  (ventricle session, all regions)
#
#   Level 2 (analysis.py)
#       → all_session_features.csv
#         one row per participant × session
#         columns: participant_id, session, target, is_adaptation,
#                  event_locked_active_{stem},
#                  event_locked_sham_{stem},
#                  event_locked_active_minus_sham_{stem}, ...
#
#   Level 3 (dose_response_real.py)
#   ─────────────────────────────────────────────────────────────────────────
#   Step 1  Build response profile
#             — filters all_session_features.csv to experimental sessions only
#             — one row per participant with Comp 1–4 for each feature stem
#             — Comp 1: thalamus active − sham        (TUS effect at target)
#             — Comp 2: ventricle active − sham       (non-specific / control)
#             — Comp 3: (thal net) − (vent net)       ← TARGET SPECIFICITY key claim
#             — Comp 4: thal sham − vent sham         ← sanity check, should ≈ 0
#
#   Step 2  Run response statistics
#             — one-sample t-tests vs 0, FDR (BH), Cohen's d, 95% CI
#
#   Step 3  Load & link acoustic dose data
#             — reads all {pid}_{intended_target}_ANALYSIS.csv files
#             — extracts the row where TargetName matches the intended target
#               (e.g. thalamus session → the thalamus nucleus row = on-target dose)
#             — ALSO extracts the spillover row: thalamus session → ventricle row
#               (how much thalamus beam spilled into ventricle)
#             — pivots to one row per participant with columns:
#                 thal_session_ontarget_{metric}   ← thalamus row of thalamus CSV
#                 thal_session_spillover_{metric}  ← ventricle row of thalamus CSV
#                 vent_session_ontarget_{metric}   ← ventricle row of ventricle CSV
#                 vent_session_spillover_{metric}  ← thalamus row of ventricle CSV
#             — joins onto response profile on participant_id
#
#   Step 4  Dose–response correlations
#             — Pearson r + 95% CI (Fisher z) + Spearman ρ
#             — Four dose column families × all neural features × all comps
#             — FDR correction within each comparison family
#
#   Step 5  Plots
#             Fig A  Dose profile — on-target vs spillover per participant
#             Fig B  Paired dot plots — Comp 3 target specificity
#             Fig C  Bar + individual-point plots — Comp 1 & Comp 2
#             Fig D  Cohen's d heatmap — all features × all four comparisons
#             Fig E  Correlation heatmaps — dose × neural, per comp
#             Fig F  KEY THESIS FIGURE — on-target dose vs Comp 3
#             Fig G  Scatter plots for FDR-significant pairs
#
#
# OUTPUTS (all written to OUTPUT_DIR)
# ─────────────────────────────────────────────────────────────────────────────
#   response_profile.csv
#   response_statistics.csv
#   dose_response_linked.csv
#   dose_response_correlations.csv
#   response_profile_plots/
#   dose_response_plots/
# =============================================================================

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp_stats
from scipy.stats import pearsonr, spearmanr, ttest_1samp
from statsmodels.stats.multitest import multipletests


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Path to the all_session_features.csv produced by Level 2 (analyze.py)
ALL_SESSION_FEATURES_CSV = '/content/drive/MyDrive/TUNES/results/group/all_session_features.csv'

# Folder that contains per-participant sub-folders, each holding the
# Level 1 acoustic enriched CSV: {pid}_{target}_acoustic_report_enriched.csv
RESULTS_ROOT = '/content/drive/MyDrive/TUNES/results'

# Where to write all Level 3 outputs
OUTPUT_DIR = '/content/drive/MyDrive/TUNES/results/group/dose_response'

# Minimum participants for a statistical test / correlation to be reported
MIN_N = 4

# Which acoustic columns to treat as dose metrics (must exist in enriched CSV)
DOSE_METRICS = [
    'OnTarget_6dB_pct',           # PRIMARY
    'OnTarget_3dB_pct',
    'Coverage_6dB_pct',
    'Coverage_3dB_pct',
    'OffTarget_6dB_pct',
    'OffTarget_3dB_pct',
    'FocusVol_6dB_mm3',
    'FocusVol_3dB_mm3',
    'PeakPressure_Target_kPa',
    'MeanInt_Overlap_6dB_Wcm2',
    'MI',
    'PeakP_Overlap_6dB_kPa',      # pressure specifically in the focal overlap zone
    'MeanP_Overlap_6dB_kPa',
    'fwhm_axial_mm',               # beam dimensions — dose shape, not just magnitude
    'fwhm_lat_mean_mm',
    'skull_thickness_mm',          # beam path difficulty covariate
    'prefocal_ratio',              # pre-focal energy ratio — safety/specificity covariate
    'peak_axial_offset_mm',        # targeting error
    'peak_lateral_mm',
]

BEAM_METRICS = [
    'skull_thickness_mm',
    'path_length_mm',
    'peak_axial_offset_mm',
    'peak_lateral_mm',
    'fwhm_axial_mm',
    'fwhm_lat_mean_mm',
    'elongation',
    'prefocal_ratio',
    'prefocal_brain_peak',
    'prefocal_brain_pos_mm',
    'Isppa_global',
    'Isppa_brain',
    'dT_peak',
    'max_MI',
]

TARGET_NAME_MAP = {
    'thalamus': 'Left_Thalamus',
    'ventricle': 'Left_Lateral_Ventricle',
}

# Neural features to highlight in the Key Thesis Figure (Comp3 × OnTarget)
# These are the stems — e.g. 'spindle_rate_per_burst' maps to
# event_locked_active_minus_sham_spindle_rate_per_burst in the data
KEY_FEATURES = [
    'spindle_rate_per_burst',
    'spindle_amplitude_uv',
    'spindle_density_per_s',
    'total_spindles_in_windows',
]


# =============================================================================
# Shared helpers
# =============================================================================

def _el_stems(df: pd.DataFrame) -> list:
    """Return feature stems from event_locked_active_minus_sham_* columns."""
    prefix = 'event_locked_active_minus_sham_'
    return [c[len(prefix):] for c in df.columns if c.startswith(prefix)]


def _short(stem: str) -> str:
    """Human-readable short label for a feature stem."""
    return (stem
            .replace('spindle_', 'sp_')
            .replace('_per_burst', '/burst')
            .replace('_per_s', '/s')
            .replace('_uv', ' µV')
            .replace('_hz', ' Hz')
            .replace('_sec', ' s')
            .replace('total_spindles_in_windows', 'total_sp'))


def _cohens_d(values: np.ndarray, mu: float = 0.0) -> float:
    """One-sample Cohen's d (distance from mu in SD units)."""
    clean = values[~np.isnan(values)]
    if len(clean) < 2:
        return np.nan
    return float((np.mean(clean) - mu) / np.std(clean, ddof=1))


def _pearson_ci(r: float, n: int, alpha: float = 0.05):
    """Fisher-z 95 % CI for a Pearson r."""
    if n < 4 or np.isnan(r):
        return np.nan, np.nan
    z     = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se    = 1.0 / np.sqrt(n - 3)
    z_c   = sp_stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - z_c * se)), float(np.tanh(z + z_c * se))


def _sig_stars(p: float) -> str:
    if np.isnan(p):  return ''
    if p < 0.001:    return '***'
    if p < 0.01:     return '**'
    if p < 0.05:     return '*'
    return 'ns'


def _already_exists(path: Path) -> bool:
    return path.exists()

def _best_row(df, target_name, tier='Tier2_Nuclei'):
    """Select the best sonication row for a given target from a v4 CSV."""
    sub = df[
        (df['ReportingTier'] == tier) &
        (df['TargetName'].str.lower().str.contains(target_name.lower(), na=False))
    ]
    if sub.empty:
        return None
    # Best = highest on-target coverage (same logic as before)
    coverage_col = 'OnTarget_6dB_pct' if 'OnTarget_6dB_pct' in sub.columns else sub.columns[0]
    return sub.sort_values(coverage_col, ascending=False).iloc[0]

def _spillover_row(df, spillover_name, tier='Tier2_Nuclei'):
    """Select the spillover row — structure the beam crosses on the way in."""
    sub = df[
        (df['ReportingTier'] == tier) &
        (df['TargetName'].str.lower().str.contains(spillover_name.lower(), na=False)) &
        (df['BeamZone'].isin(['pre-focal', 'at focus']))
    ]
    if sub.empty:
        return None
    return sub.sort_values('Isppa_Target_Wcm2', ascending=False).iloc[0]

# =============================================================================
# STEP 1 — Build response profile
# =============================================================================

def build_response_profile(all_session_csv: str,
                           output_dir: str) -> pd.DataFrame:
    """
    Reshape all_session_features.csv into a response-profile table:
    one row per participant with Comp 1–4 columns for each feature stem.

    Adaptation rows (is_adaptation == TRUE) are excluded automatically.

    Comp 1: thalamus active − sham          (TUS effect at target)
    Comp 2: ventricle active − sham         (non-specific / control)
    Comp 3: (thal net) − (vent net)         ← target specificity — KEY THESIS CLAIM
    Comp 4: thal sham − vent sham           ← sanity check, should be ≈ 0
    """
    print('\n[Step 1] Building response profile …')
    df = pd.read_csv(all_session_csv)
    print(f'    Loaded {len(df)} session rows from {Path(all_session_csv).name}')

    # Exclude adaptation sessions
    if 'is_adaptation' in df.columns:
        n_before = len(df)
        df = df[df['is_adaptation'].astype(str).str.upper() != 'TRUE'].copy()
        print(f'    Excluded {n_before - len(df)} adaptation rows — '
              f'{len(df)} experimental rows remain')

    stems = _el_stems(df)
    if not stems:
        raise ValueError(
            'No event_locked_active_minus_sham_* columns found in '
            f'{all_session_csv}. Check that Level 2 ran correctly.'
        )
    print(f'    Found {len(stems)} event-locked feature stems: {stems}')

    rows = []
    for pid, p_df in df.groupby('participant_id'):
        thal_rows = p_df[p_df['target'].str.lower().str.contains('thal', na=False)]
        vent_rows = p_df[p_df['target'].str.lower().str.contains('vent', na=False)]

        if thal_rows.empty or vent_rows.empty:
            print(f'    [{pid}] Missing thalamus or ventricle session — skipping')
            continue

        thal = thal_rows.iloc[0]
        vent = vent_rows.iloc[0]
        row  = {'participant_id': pid}

        for stem in stems:
            ams_col    = f'event_locked_active_minus_sham_{stem}'
            active_col = f'event_locked_active_{stem}'
            sham_col   = f'event_locked_sham_{stem}'

            thal_active = pd.to_numeric(thal.get(active_col), errors='coerce')
            thal_sham   = pd.to_numeric(thal.get(sham_col),   errors='coerce')
            vent_active = pd.to_numeric(vent.get(active_col), errors='coerce')
            vent_sham   = pd.to_numeric(vent.get(sham_col),   errors='coerce')

            # Prefer the pre-computed net column; fall back to subtraction
            thal_net = pd.to_numeric(thal.get(ams_col), errors='coerce')
            vent_net = pd.to_numeric(vent.get(ams_col), errors='coerce')
            if np.isnan(thal_net) and not (np.isnan(thal_active) or np.isnan(thal_sham)):
                thal_net = thal_active - thal_sham
            if np.isnan(vent_net) and not (np.isnan(vent_active) or np.isnan(vent_sham)):
                vent_net = vent_active - vent_sham

            row[f'comp1_thal_net_{stem}']           = thal_net
            row[f'comp2_vent_net_{stem}']            = vent_net
            row[f'comp3_specificity_{stem}']         = (
                thal_net - vent_net if not (np.isnan(thal_net) or np.isnan(vent_net))
                else np.nan
            )
            row[f'comp4_sham_balance_{stem}']        = (
                thal_sham - vent_sham
                if not (np.isnan(thal_sham) or np.isnan(vent_sham))
                else np.nan
            )
            # Keep raw values for context / inspection
            row[f'thal_active_{stem}'] = thal_active
            row[f'thal_sham_{stem}']   = thal_sham
            row[f'vent_active_{stem}'] = vent_active
            row[f'vent_sham_{stem}']   = vent_sham

        rows.append(row)

    if not rows:
        raise RuntimeError('No participants had both thalamus and ventricle sessions.')

    profile_df = pd.DataFrame(rows)
    out_path   = Path(output_dir) / 'response_profile.csv'
    profile_df.to_csv(out_path, index=False)
    print(f'    Saved {len(profile_df)} participants → {out_path.name}')
    return profile_df, stems


# =============================================================================
# STEP 2 — Response statistics
# =============================================================================

def run_response_statistics(profile_df: pd.DataFrame,
                            stems: list,
                            output_dir: str) -> pd.DataFrame:
    """
    One-sample t-tests (vs 0) for each comparison × feature, with FDR
    correction (Benjamini-Hochberg) applied within each comparison family.

    Returns a DataFrame saved as response_statistics.csv.
    """
   

    print('\n[Step 2] Running response statistics …')

    comp_map = {
        'comp1_thal_net_':      'Comp1_Thal_NetEffect',
        'comp2_vent_net_':      'Comp2_Vent_NetEffect',
        'comp3_specificity_':   'Comp3_TargetSpecificity',
        'comp4_sham_balance_':  'Comp4_ShamSanity',
    }

    all_rows = []

    for prefix, comp_label in comp_map.items():
        comp_rows = []

        for stem in stems:
            col    = f'{prefix}{stem}'
            if col not in profile_df.columns:
                continue
            values = pd.to_numeric(profile_df[col], errors='coerce').values
            valid  = values[~np.isnan(values)]
            n      = len(valid)

            base = {
                'comparison': comp_label,
                'feature':    stem,
                'n':          n,
                'mean':       round(float(np.nanmean(values)), 6) if n > 0 else np.nan,
                'std':        round(float(np.nanstd(values, ddof=1)), 6) if n > 1 else np.nan,
                'median':     round(float(np.nanmedian(values)), 6) if n > 0 else np.nan,
                'sem':        round(float(np.nanstd(values, ddof=1) / np.sqrt(n)), 6) if n > 1 else np.nan,
            }

            if n < MIN_N:
                base.update({
                    'cohen_d': np.nan, 't_stat': np.nan,
                    'p_value': np.nan, 'p_fdr':  np.nan,
                    'ci_95_lo': np.nan, 'ci_95_hi': np.nan,
                    'significant': False, 'note': 'insufficient_data',
                })
            else:
                t_stat, p_val = ttest_1samp(valid, popmean=0.0)
                d             = _cohens_d(valid)
                # 95 % CI on the mean via t-distribution
                t_crit = sp_stats.t.ppf(0.975, df=n - 1)
                sem    = np.std(valid, ddof=1) / np.sqrt(n)
                base.update({
                    'cohen_d':   round(float(d),      4),
                    't_stat':    round(float(t_stat), 4),
                    'p_value':   round(float(p_val),  6),
                    'p_fdr':     np.nan,
                    'ci_95_lo':  round(float(np.mean(valid) - t_crit * sem), 6),
                    'ci_95_hi':  round(float(np.mean(valid) + t_crit * sem), 6),
                    'significant': False,
                    'note': '',
                })
            comp_rows.append(base)

        # FDR within this comparison family
        testable = [r for r in comp_rows if not np.isnan(r.get('p_value', np.nan))]
        if testable:
            pvals = np.array([r['p_value'] for r in testable])
            _, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh')
            for r, p_fdr in zip(testable, pvals_fdr):
                r['p_fdr']      = round(float(p_fdr), 6)
                r['significant'] = bool(p_fdr < 0.05)

        n_sig = sum(r['significant'] for r in comp_rows)
        print(f'    [{comp_label}] {len(comp_rows)} features '
              f'| {n_sig} significant after FDR')
        all_rows.extend(comp_rows)

    stats_df = pd.DataFrame(all_rows)
    out_path = Path(output_dir) / 'response_statistics.csv'
    stats_df.to_csv(out_path, index=False)
    print(f'    Saved → {out_path.name}')
    return stats_df


# =============================================================================
# STEP 3 — Load & link acoustic dose data
# =============================================================================

def load_and_link_acoustic(profile_df: pd.DataFrame,
                           results_root: str,
                           output_dir: str) -> pd.DataFrame:

    print('\n[Step 3] Loading & linking acoustic dose data …')

    targets       = ['thalamus', 'ventricle']
    acoustic_rows = []
    summary_rows  = []
    root_path     = Path(results_root)
    pids_in_profile = set(profile_df['participant_id'].astype(str).tolist())

    TARGET_FILE_MAP = {
        'thalamus': 'Left_Thalamus',
        'ventricle': 'Left_Lateral_Ventricle',
    }

    # ── 1. Load both CSV types for each participant × target ──────────────────
    for target in targets:
        structure_name = TARGET_FILE_MAP[target]

        # Group all CSVs found per participant so we can pick best across files
        pid_to_files = {}
        for csv_path in sorted(root_path.glob(f'**/*_{structure_name}*_analysis.csv')):
            # Skip old all-caps ANALYSIS files if any remain
            if csv_path.stem.endswith('_ANALYSIS'):
                continue

            # Extract pid: tunes08_Left_Thalamus_analysis → tunes08
            pid_raw = csv_path.stem.split(f'_{structure_name}')[0]
            # Normalise to match participant_id format: tunes08 → tunes-08
            pid = pid_raw[:5] + '-' + pid_raw[5:] if '-' not in pid_raw else pid_raw

            if pid not in pids_in_profile:
                continue

            if pid not in pid_to_files:
                pid_to_files[pid] = []
            pid_to_files[pid].append((pid_raw, csv_path))

        # Process each participant — pick best sonication across all files
        for pid, file_list in pid_to_files.items():
            best_row      = None
            best_coverage = -np.inf
            best_pid_raw  = None
            best_path     = None

            for pid_raw, csv_path in file_list:
                df_a = pd.read_csv(csv_path)

                # Keep only Tier2 nucleus rows
                if 'ReportingTier' in df_a.columns:
                    df_a = df_a[df_a['ReportingTier'] == 'Tier2_Nuclei'].copy()

                # Keep only rows matching this target structure
                if 'TargetName' in df_a.columns:
                    df_a_exact = df_a[df_a['TargetName'].str.startswith(
                        structure_name, na=False)].copy()

                    if df_a_exact.empty:
                        print(f'    [WARN] {pid} / {target} / {csv_path.name}: '
                              f'no TargetName starting with "{structure_name}" — '
                              f'available: {df_a["TargetName"].unique().tolist()} '
                              f'— falling back to substring match')
                        df_a_exact = df_a[
                            df_a['TargetName'].str.lower().str.contains(
                                target[:4], na=False)
                        ].copy()

                    df_a = df_a_exact

                if df_a.empty:
                    continue

                # Best sonication within this file
                if 'OnTarget_6dB_pct' in df_a.columns:
                    df_a = (df_a
                            .sort_values('OnTarget_6dB_pct', ascending=False)
                            .head(1)
                            .reset_index(drop=True))
                    coverage = pd.to_numeric(
                        df_a['OnTarget_6dB_pct'].iloc[0], errors='coerce')
                else:
                    df_a     = df_a.head(1).reset_index(drop=True)
                    coverage = -np.inf

                # Keep this file's best row if it beats the current best
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_row      = df_a
                    best_pid_raw  = pid_raw
                    best_path     = csv_path

            if best_row is None:
                print(f'    [WARN] {pid} / {target}: no valid rows found across '
                      f'{len(file_list)} file(s) — skipping')
                continue

            if len(file_list) > 1:
                print(f'    [{pid} / {target}] {len(file_list)} placement files found — '
                      f'selected {best_path.name} '
                      f'(OnTarget_6dB_pct = {best_coverage:.1f}%)')

            best_row['participant_id'] = pid
            best_row['_target']        = target
            acoustic_rows.append(best_row)

            # Sonication summary — look in same folder as the winning CSV
            summary_path = best_path.parent / \
                f'{best_pid_raw}_{structure_name}_sonication_summary.csv'

            # Fallback: try variant name (e.g. Left_Thalamus1_sonication_summary)
            if not summary_path.exists():
                alt_stem     = best_path.stem.replace('_analysis', '_sonication_summary')
                summary_path = best_path.parent / f'{alt_stem}.csv'

            if summary_path.exists():
                df_s = pd.read_csv(summary_path)

                if len(df_s) > 1 and 'coverage_6dB_pct' in df_s.columns:
                    df_s = (df_s
                            .sort_values('coverage_6dB_pct', ascending=False)
                            .head(1)
                            .reset_index(drop=True))

                df_s['participant_id'] = pid
                df_s['_target']        = target
                summary_rows.append(df_s)
            else:
                print(f'    [WARN] No sonication summary for {pid} / {target} '
                      f'at {summary_path} — beam metrics will be missing')

    if not acoustic_rows:
        print('    [WARN] No acoustic CSVs found — dose–response steps will be skipped.')
        print(f'    Expected pattern: {results_root}/<pid>/<target>/<pid>_<target>_analysis.csv')
        return profile_df.copy()

    # ── 2. Stack and check coverage ───────────────────────────────────────────
    acoustic_df = pd.concat(acoustic_rows, ignore_index=True)
    print(f'    Per-region rows loaded: {len(acoustic_df)} '
          f'({acoustic_df["participant_id"].nunique()} participants)')

    has_summary = bool(summary_rows)
    if has_summary:
        summary_df = pd.concat(summary_rows, ignore_index=True)
        print(f'    Sonication summary rows loaded: {len(summary_df)}')
    else:
        summary_df = pd.DataFrame()
        print('    [WARN] No sonication summary CSVs found — '
              'FWHM / skull / prefocal columns will be absent from linked table')

    acoustic_pids   = set(acoustic_df['participant_id'].astype(str).tolist())
    missing_acoustic = pids_in_profile - acoustic_pids
    if missing_acoustic:
        print(f'    [WARN] No acoustic data for: {sorted(missing_acoustic)}')

    # ── 3. Check which dose columns are actually present ──────────────────────
    available_dose = [c for c in DOSE_METRICS if c in acoustic_df.columns]
    missing_dose   = set(DOSE_METRICS) - set(available_dose)
    if missing_dose:
        print(f'    [WARN] Dose metric columns not found in per-region CSV: {missing_dose}')

    available_beam = ([c for c in BEAM_METRICS if c in summary_df.columns]
                      if has_summary else [])
    missing_beam = set(BEAM_METRICS) - set(available_beam)
    if missing_beam and has_summary:
        print(f'    [WARN] Beam metric columns not found in summary CSV: {missing_beam}')

    # ── 4. Pivot both wide — one column per target × metric ──────────────────
    pivot_rows = []
    all_pids   = acoustic_df['participant_id'].unique()

    for pid in all_pids:
        row = {'participant_id': str(pid)}

        pid_acoustic = acoustic_df[acoustic_df['participant_id'] == pid]
        for _, sess in pid_acoustic.iterrows():
            tgt = str(sess.get('_target', '')).lower()
            if tgt not in targets:
                continue
            for col in available_dose:
                row[f'{tgt}_{col}'] = pd.to_numeric(sess.get(col), errors='coerce')

        if has_summary:
            pid_summary = summary_df[summary_df['participant_id'] == pid]
            for _, sess in pid_summary.iterrows():
                tgt = str(sess.get('_target', '')).lower()
                if tgt not in targets:
                    continue
                for col in available_beam:
                    row[f'{tgt}_{col}'] = pd.to_numeric(sess.get(col), errors='coerce')

        pivot_rows.append(row)

    # ── 5. Merge onto response profile ───────────────────────────────────────
    acoustic_wide = pd.DataFrame(pivot_rows)
    linked        = profile_df.merge(acoustic_wide, on='participant_id', how='left')

    n_with_dose = linked[
        [f'thalamus_{c}' for c in available_dose if f'thalamus_{c}' in linked.columns]
    ].notna().any(axis=1).sum()
    n_with_beam = (linked[
        [f'thalamus_{c}' for c in available_beam if f'thalamus_{c}' in linked.columns]
    ].notna().any(axis=1).sum() if available_beam else 0)

    print(f'    Linked table: {len(linked)} participants '
          f'| {n_with_dose} with dose data '
          f'| {n_with_beam} with beam metrics')

    out_path = Path(output_dir) / 'dose_response_linked.csv'
    linked.to_csv(out_path, index=False)
    print(f'    Saved → {out_path.name}')
    return linked


# =============================================================================
# STEP 4 — Dose–response correlations
# =============================================================================

def run_dose_response_correlations(linked: pd.DataFrame,
                                   stems: list,
                                   output_dir: str) -> pd.DataFrame:
    """
    Pearson r (+ 95 % CI) and Spearman ρ for every:
        dose metric × neural response feature × comparison family

    Dose target mapping:
      Comp 1 (thalamus net)   → thalamus_{dose_metric}
      Comp 2 (ventricle net)  → ventricle_{dose_metric}
      Comp 3 (specificity)    → thalamus_{dose_metric}
          (better thalamus focus should predict stronger thalamus-specific effect)
      Comp 4 (sham balance)   → thalamus_{dose_metric}  (control)

    FDR correction within each comparison family, matching Step 2.
    Saves: dose_response_correlations.csv
    """
   

    # Check if any dose columns exist at all
    dose_cols_present = [c for c in linked.columns
                         if any(c == f'{tgt}_{dm}'
                                for tgt in ['thalamus', 'ventricle']
                                for dm in DOSE_METRICS)]
    if not dose_cols_present:
        print('\n[Step 4] No acoustic dose columns in linked table — '
              'correlations skipped.')
        return pd.DataFrame()

    print('\n[Step 4] Running dose–response correlations …')

    comp_map = {
        'comp1_thal_net_':     ('Comp1_Thal_NetEffect',      'thalamus'),
        'comp2_vent_net_':     ('Comp2_Vent_NetEffect',       'ventricle'),
        'comp3_specificity_':  ('Comp3_TargetSpecificity',    'thalamus'),
        'comp4_sham_balance_': ('Comp4_ShamSanity',           'thalamus'),
    }

    all_rows = []

    for prefix, (comp_label, dose_target) in comp_map.items():
        comp_rows = []

        for stem in stems:
            neural_col = f'{prefix}{stem}'
            if neural_col not in linked.columns:
                continue

            for dose_metric in DOSE_METRICS:
                dose_col = f'{dose_target}_{dose_metric}'
                if dose_col not in linked.columns:
                    continue

                x = pd.to_numeric(linked[dose_col],   errors='coerce').values
                y = pd.to_numeric(linked[neural_col], errors='coerce').values
                mask = ~(np.isnan(x) | np.isnan(y))
                n    = int(mask.sum())

                base = {
                    'comparison':     comp_label,
                    'dose_target':    dose_target,
                    'dose_metric':    dose_metric,
                    'neural_feature': stem,
                    'n':              n,
                }

                if n < MIN_N:
                    base.update({
                        'pearson_r': np.nan, 'pearson_p': np.nan,
                        'ci_lo': np.nan, 'ci_hi': np.nan,
                        'spearman_r': np.nan, 'spearman_p': np.nan,
                        'pearson_p_fdr': np.nan, 'spearman_p_fdr': np.nan,
                        'sig_pearson': False, 'sig_spearman': False,
                        'note': 'insufficient_data',
                    })
                else:
                    pr, pp   = pearsonr(x[mask], y[mask])
                    sr, sp_v = spearmanr(x[mask], y[mask])
                    lo, hi   = _pearson_ci(pr, n)
                    base.update({
                        'pearson_r':      round(float(pr),   4),
                        'pearson_p':      round(float(pp),   6),
                        'ci_lo':          round(float(lo),   4),
                        'ci_hi':          round(float(hi),   4),
                        'spearman_r':     round(float(sr),   4),
                        'spearman_p':     round(float(sp_v), 6),
                        'pearson_p_fdr':  np.nan,
                        'spearman_p_fdr': np.nan,
                        'sig_pearson':    False,
                        'sig_spearman':   False,
                        'note':           '',
                    })
                comp_rows.append(base)

        # FDR within this comparison family
        testable = [r for r in comp_rows if not np.isnan(r.get('pearson_p', np.nan))]
        if testable:
            pp_arr = np.array([r['pearson_p']  for r in testable])
            sp_arr = np.array([r['spearman_p'] for r in testable])
            _, fdr_p, _, _ = multipletests(pp_arr, method='fdr_bh')
            _, fdr_s, _, _ = multipletests(sp_arr, method='fdr_bh')
            for r, fp, fs in zip(testable, fdr_p, fdr_s):
                r['pearson_p_fdr']  = round(float(fp), 6)
                r['spearman_p_fdr'] = round(float(fs), 6)
                r['sig_pearson']    = bool(fp < 0.05)
                r['sig_spearman']   = bool(fs < 0.05)

        n_sig = sum(r['sig_pearson'] for r in comp_rows)
        print(f'    [{comp_label}] {len([r for r in comp_rows if r["n"] >= MIN_N])} '
              f'pairs tested | {n_sig} significant (Pearson FDR)')
        all_rows.extend(comp_rows)

    corr_df  = pd.DataFrame(all_rows)
    out_path = Path(output_dir) / 'dose_response_correlations.csv'
    corr_df.to_csv(out_path, index=False)
    print(f'    Saved → {out_path.name}')
    return corr_df


# =============================================================================
# STEP 5 — Plots
# =============================================================================

def plot_all(profile_df: pd.DataFrame,
             stats_df:   pd.DataFrame,
             linked:     pd.DataFrame,
             corr_df:    pd.DataFrame,
             stems:      list,
             output_dir: str) -> None:
    """
    Produce all Level 3 figures. Organised into two sub-folders:
      response_profile_plots/  — neural response characterisation
      dose_response_plots/     — acoustic dose × neural response
    """
    rp_dir = Path(output_dir) / 'response_profile_plots'
    dr_dir = Path(output_dir) / 'dose_response_plots'
    rp_dir.mkdir(parents=True, exist_ok=True)
    dr_dir.mkdir(parents=True, exist_ok=True)

    # Build a quick lookup: (comp_label, stem) → stats row
    stats_lut = {}
    if stats_df is not None and not stats_df.empty:
        for _, r in stats_df.iterrows():
            stats_lut[(r['comparison'], r['feature'])] = r

    def _sig_label(comp_label, stem):
        r = stats_lut.get((comp_label, stem))
        if r is None or np.isnan(r.get('p_fdr', np.nan)):
            return ''
        return _sig_stars(r['p_fdr'])

    n_stems = len(stems)
    n_cols  = 3
    n_rows  = max(1, int(np.ceil(n_stems / n_cols)))

    # ------------------------------------------------------------------
    # Fig B — Paired dot plots (Comp 3: target specificity)
    # ------------------------------------------------------------------
    print('\n    [Fig B] Paired dot plots — Comp 3 …')
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4.5, n_rows * 4.0),
                             squeeze=False)
    axes_flat = axes.ravel()

    for i, stem in enumerate(stems):
        ax        = axes_flat[i]
        thal_vals = pd.to_numeric(
            profile_df.get(f'comp1_thal_net_{stem}'), errors='coerce').values
        vent_vals = pd.to_numeric(
            profile_df.get(f'comp2_vent_net_{stem}'), errors='coerce').values

        for t, v in zip(thal_vals, vent_vals):
            if not (np.isnan(t) or np.isnan(v)):
                ax.plot([0, 1], [t, v], color='grey', lw=1.0, alpha=0.5, zorder=1)

        valid_t = thal_vals[~np.isnan(thal_vals)]
        valid_v = vent_vals[~np.isnan(vent_vals)]
        if len(valid_t):
            ax.scatter([0]*len(valid_t), valid_t,
                       color='#E04B4B', s=65, zorder=3, label='Thalamus')
            ax.hlines(np.mean(valid_t), -0.2, 0.2,
                      colors='#E04B4B', linewidths=2.5, zorder=4)
        if len(valid_v):
            ax.scatter([1]*len(valid_v), valid_v,
                       color='#4B7BE0', s=65, zorder=3, label='Ventricle')
            ax.hlines(np.mean(valid_v), 0.8, 1.2,
                      colors='#4B7BE0', linewidths=2.5, zorder=4)

        ax.axhline(0, color='black', lw=0.7, ls='--', alpha=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Thalamus\nnet effect', 'Ventricle\nnet effect'],
                           fontsize=8)
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(_short(stem), fontsize=9, fontweight='bold')
        sig = _sig_label('Comp3_TargetSpecificity', stem)
        if sig:
            ymax = ax.get_ylim()[1]
            ax.text(0.5, ymax * 0.92, sig, ha='center', fontsize=12,
                    fontweight='bold')
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        'Comp 3 — Target Specificity: Thalamus vs Ventricle net TUS effect\n'
        '(active − sham within each night)  |  lines = individual participants',
        fontsize=11, fontweight='bold'
    )
    fig.tight_layout()
    fig.savefig(rp_dir / 'comp3_paired_dots.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('      Saved comp3_paired_dots.png')

    # ------------------------------------------------------------------
    # Fig C — Bar + individual points (Comp 1 & Comp 2)
    # ------------------------------------------------------------------
    for comp_prefix, comp_label, color, title_str in [
        ('comp1_thal_net_', 'Comp1_Thal_NetEffect',
         '#E04B4B', 'Comp 1 — Thalamus night: active − sham net TUS effect'),
        ('comp2_vent_net_', 'Comp2_Vent_NetEffect',
         '#4B7BE0', 'Comp 2 — Ventricle night: active − sham net TUS effect'),
    ]:
        print(f'    [Fig C] Bar + dots — {comp_label} …')
        fig2, axes2 = plt.subplots(n_rows, n_cols,
                                   figsize=(n_cols * 4.0, n_rows * 3.8),
                                   squeeze=False)
        axes2_flat = axes2.ravel()

        for i, stem in enumerate(stems):
            ax   = axes2_flat[i]
            col  = f'{comp_prefix}{stem}'
            vals = pd.to_numeric(profile_df.get(col), errors='coerce').values
            valid = vals[~np.isnan(vals)]
            n_v   = len(valid)

            if n_v:
                mean_v = np.mean(valid)
                sem_v  = np.std(valid, ddof=1) / np.sqrt(n_v) if n_v > 1 else 0
                ax.bar([0], [mean_v], width=0.5, color=color, alpha=0.65,
                       yerr=[[0], [sem_v]], capsize=5,
                       error_kw=dict(elinewidth=1.2, ecolor='black'))
                rng    = np.random.default_rng(42)
                jitter = rng.uniform(-0.08, 0.08, n_v)
                ax.scatter(jitter, valid, color=color, s=55, zorder=4,
                           edgecolors='white', linewidths=0.6, alpha=0.9)

            ax.axhline(0, color='black', lw=0.7, ls='--', alpha=0.5)
            ax.set_xticks([])
            ax.set_title(_short(stem), fontsize=9, fontweight='bold')
            sig = _sig_label(comp_label, stem)
            if sig and n_v:
                ymax = ax.get_ylim()[1]
                ax.text(0.0, ymax * 0.88, sig, ha='center', fontsize=12,
                        fontweight='bold')

        for j in range(i + 1, len(axes2_flat)):
            axes2_flat[j].set_visible(False)

        fig2.suptitle(
            f'{title_str}\nBar = mean ± SEM  |  dots = individual participants  '
            f'|  * FDR < 0.05',
            fontsize=11, fontweight='bold'
        )
        fig2.tight_layout()
        fname2 = f'{comp_label.lower()}_bar_dots.png'
        fig2.savefig(rp_dir / fname2, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f'      Saved {fname2}')

    # ------------------------------------------------------------------
    # Fig D — Cohen's d heatmap (all features × all 4 comparisons)
    # ------------------------------------------------------------------
    print("    [Fig D] Cohen's d heatmap …")
    if stats_df is not None and not stats_df.empty:
        comp_order = ['Comp1_Thal_NetEffect', 'Comp2_Vent_NetEffect',
                      'Comp3_TargetSpecificity', 'Comp4_ShamSanity']
        pivot_d = (stats_df[stats_df['comparison'].isin(comp_order)]
                   .pivot(index='feature', columns='comparison', values='cohen_d')
                   .reindex(columns=comp_order)
                   .dropna(how='all'))
        pivot_p = (stats_df[stats_df['comparison'].isin(comp_order)]
                   .pivot(index='feature', columns='comparison', values='p_fdr')
                   .reindex(columns=comp_order)
                   .reindex(index=pivot_d.index))

        if not pivot_d.empty:
            nh = len(pivot_d.index)
            fig3, ax3 = plt.subplots(
                figsize=(len(comp_order) * 2.2 + 1, max(4, nh * 0.45 + 2))
            )
            vmax = max(0.5, np.nanpercentile(np.abs(pivot_d.values), 95))
            im = ax3.imshow(pivot_d.values, aspect='auto',
                            cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            for ri in range(nh):
                for ci in range(len(comp_order)):
                    d_val = pivot_d.values[ri, ci]
                    p_val = pivot_p.values[ri, ci]
                    if np.isnan(d_val):
                        continue
                    stars = _sig_stars(p_val) if not np.isnan(p_val) else ''
                    ax3.text(ci, ri, f'{d_val:.2f}{stars}',
                             ha='center', va='center', fontsize=7,
                             color='white' if abs(d_val) > vmax * 0.6 else 'black')
            ax3.set_xticks(range(len(comp_order)))
            ax3.set_xticklabels(
                ['Comp1\nThal net', 'Comp2\nVent net',
                 'Comp3\nSpecificity', 'Comp4\nSham check'],
                fontsize=9
            )
            ax3.set_yticks(range(nh))
            ax3.set_yticklabels([_short(s) for s in pivot_d.index], fontsize=8)
            fig3.colorbar(im, ax=ax3, label="Cohen's d", pad=0.01)
            ax3.set_title(
                "Cohen's d — all features × all comparisons\n"
                "* p_FDR<0.05   ** p_FDR<0.01   *** p_FDR<0.001",
                fontsize=11, fontweight='bold'
            )
            fig3.tight_layout()
            fig3.savefig(rp_dir / 'cohens_d_heatmap.png', dpi=150, bbox_inches='tight')
            plt.close(fig3)
            print('      Saved cohens_d_heatmap.png')

    # ------------------------------------------------------------------
    # Dose–response figures (only if acoustic data was linked)
    # ------------------------------------------------------------------
    dose_cols_present = any(
        f'thalamus_{dm}' in linked.columns or f'ventricle_{dm}' in linked.columns
        for dm in DOSE_METRICS
    )
    if not dose_cols_present:
        print('    [WARN] No acoustic dose columns in linked table — '
              'dose–response figures skipped.')
        return

    # Fig A — Dose profile per participant
    print('    [Fig A] Dose profile per participant …')
    primary_dose = 'OnTarget_6dB_pct'
    thal_d_col   = f'thalamus_{primary_dose}'
    vent_d_col   = f'ventricle_{primary_dose}'

    if thal_d_col in linked.columns or vent_d_col in linked.columns:
        pids      = linked['participant_id'].values
        thal_vals = pd.to_numeric(linked.get(thal_d_col,
                                             pd.Series([np.nan]*len(linked))),
                                  errors='coerce').values
        vent_vals = pd.to_numeric(linked.get(vent_d_col,
                                             pd.Series([np.nan]*len(linked))),
                                  errors='coerce').values
        idx   = np.arange(len(pids))
        width = 0.35

        fig_a, ax_a = plt.subplots(figsize=(max(7, len(pids) * 0.9 + 2), 5))
        bars_t = ax_a.barh(idx + width/2, thal_vals, width,
                           color='#E04B4B', alpha=0.8, label='Thalamus')
        bars_v = ax_a.barh(idx - width/2, vent_vals, width,
                           color='#4B7BE0', alpha=0.8, label='Ventricle')
        for bar, val in list(zip(bars_t, thal_vals)) + list(zip(bars_v, vent_vals)):
            if not np.isnan(val):
                ax_a.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                          f'{val:.1f}%', va='center', fontsize=7)
        ax_a.axvline(0, color='black', lw=0.8)
        ax_a.set_yticks(idx)
        ax_a.set_yticklabels(pids, fontsize=9)
        ax_a.set_xlabel(f'{primary_dose} (%)', fontsize=10)
        ax_a.set_title(
            f'Realized acoustic dose — {primary_dose}\n'
            'Higher % = better focal overlap with the target structure',
            fontsize=11, fontweight='bold'
        )
        ax_a.legend(fontsize=9)
        fig_a.tight_layout()
        fig_a.savefig(dr_dir / 'dose_profile_per_participant.png',
                      dpi=150, bbox_inches='tight')
        plt.close(fig_a)
        print('      Saved dose_profile_per_participant.png')

    # Fig E — Correlation heatmaps per comparison
    print('    [Fig E] Correlation heatmaps …')
    if corr_df is not None and not corr_df.empty:
        for comp_label in ['Comp1_Thal_NetEffect', 'Comp2_Vent_NetEffect',
                           'Comp3_TargetSpecificity', 'Comp4_ShamSanity']:
            sub = corr_df[(corr_df['comparison'] == comp_label) &
                          corr_df['pearson_r'].notna()].copy()
            if sub.empty:
                continue
            try:
                piv_r = sub.pivot_table(index='neural_feature', columns='dose_metric',
                                        values='pearson_r', aggfunc='first')
                piv_p = sub.pivot_table(index='neural_feature', columns='dose_metric',
                                        values='pearson_p_fdr', aggfunc='first')
            except Exception:
                continue
            piv_r = piv_r.dropna(how='all').dropna(how='all', axis=1)
            if piv_r.empty:
                continue

            nr, nc = piv_r.shape
            fig_e, ax_e = plt.subplots(
                figsize=(nc * 1.9 + 2, max(4, nr * 0.48 + 2))
            )
            vmax = max(0.3, np.nanpercentile(np.abs(piv_r.values), 95))
            im = ax_e.imshow(piv_r.values, aspect='auto',
                             cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            for ri in range(nr):
                for ci in range(nc):
                    r_v = piv_r.values[ri, ci]
                    p_v = piv_p.values[ri, ci] if piv_p is not None else np.nan
                    if np.isnan(r_v):
                        continue
                    stars = _sig_stars(p_v) if not np.isnan(p_v) else ''
                    ax_e.text(ci, ri, f'{r_v:.2f}{stars}',
                              ha='center', va='center', fontsize=7,
                              color='white' if abs(r_v) > vmax * 0.6 else 'black')
            ax_e.set_xticks(range(nc))
            ax_e.set_xticklabels(
                [c.replace('_percent', '%').replace('_', ' ')
                 for c in piv_r.columns],
                rotation=35, ha='right', fontsize=8
            )
            ax_e.set_yticks(range(nr))
            ax_e.set_yticklabels([_short(s) for s in piv_r.index], fontsize=8)
            fig_e.colorbar(im, ax=ax_e, label='Pearson r', pad=0.01)
            ax_e.set_title(
                f'Dose × Neural Response  |  {comp_label}\n'
                'Pearson r  |  * p_FDR<0.05  ** <0.01  *** <0.001',
                fontsize=10, fontweight='bold'
            )
            fig_e.tight_layout()
            fname_e = f'corr_heatmap_{comp_label.lower()}.png'
            fig_e.savefig(dr_dir / fname_e, dpi=150, bbox_inches='tight')
            plt.close(fig_e)
            print(f'      Saved {fname_e}')

    # Fig F — KEY THESIS FIGURE: OnTarget_6dB vs Comp3 key features
    print('    [Fig F] KEY THESIS FIGURE — OnTarget_6dB vs Comp3 …')
    dose_col_key   = f'thalamus_{primary_dose}'
    comp3_prefix   = 'comp3_specificity_'
    key_neural_cols = [f'{comp3_prefix}{f}' for f in KEY_FEATURES
                       if f'{comp3_prefix}{f}' in linked.columns]
    if not key_neural_cols:
        key_neural_cols = [c for c in linked.columns
                           if c.startswith(comp3_prefix)][:6]

    if dose_col_key in linked.columns and key_neural_cols:
        nk     = len(key_neural_cols)
        nc_k   = min(3, nk)
        nr_k   = max(1, int(np.ceil(nk / nc_k)))
        fig_f, axes_f = plt.subplots(nr_k, nc_k,
                                     figsize=(nc_k * 5, nr_k * 4.5),
                                     squeeze=False)
        axes_f_flat = axes_f.ravel()
        x_dose = pd.to_numeric(linked[dose_col_key], errors='coerce').values
        pids_f = linked['participant_id'].values

        for idx_k, ncol in enumerate(key_neural_cols):
            ax_f = axes_f_flat[idx_k]
            stem = ncol[len(comp3_prefix):]
            y    = pd.to_numeric(linked[ncol], errors='coerce').values
            mask = ~(np.isnan(x_dose) | np.isnan(y))
            n_k  = int(mask.sum())

            ax_f.axhline(0, color='grey', lw=0.7, ls='--', alpha=0.5)
            ax_f.axvline(50, color='grey', lw=0.5, ls=':', alpha=0.4,
                         label='50% coverage')

            if n_k >= MIN_N:
                pr, pp   = pearsonr(x_dose[mask], y[mask])
                lo, hi   = _pearson_ci(pr, n_k)
                coef     = np.polyfit(x_dose[mask], y[mask], 1)
                x_line   = np.linspace(x_dose[mask].min(), x_dose[mask].max(), 200)
                y_line   = np.polyval(coef, x_line)
                resid    = y[mask] - np.polyval(coef, x_dose[mask])
                se_r     = np.std(resid, ddof=2) if n_k > 2 else 0
                ax_f.plot(x_line, y_line, color='crimson', lw=1.8, zorder=3)
                ax_f.fill_between(x_line,
                                  y_line - 1.96 * se_r,
                                  y_line + 1.96 * se_r,
                                  color='crimson', alpha=0.12)
                label_r = (f'r = {pr:.2f}  [{lo:.2f}, {hi:.2f}]\n'
                           f'p = {pp:.3f}  {_sig_stars(pp)}  n = {n_k}')
                ax_f.text(0.05, 0.95, label_r, transform=ax_f.transAxes,
                          va='top', fontsize=9,
                          bbox=dict(boxstyle='round,pad=0.3',
                                    fc='white', alpha=0.85, ec='grey'))
            else:
                ax_f.text(0.5, 0.5, f'n={n_k}\n(too few)',
                          ha='center', va='center', transform=ax_f.transAxes,
                          color='grey', fontsize=9)

            ax_f.scatter(x_dose[mask], y[mask], color='#2C3E7A', s=70,
                         zorder=4, edgecolors='white', linewidths=0.7)
            for xi, yi, pid in zip(x_dose[mask], y[mask], pids_f[mask]):
                ax_f.annotate(str(pid), (xi, yi),
                              textcoords='offset points', xytext=(4, 2),
                              fontsize=6.5, color='#2C3E7A', alpha=0.85)

            ax_f.set_xlabel(f'Thalamus {primary_dose} (%)', fontsize=10)
            ax_f.set_ylabel('Comp 3 specificity score\n(thal − vent net effect)',
                            fontsize=9)
            ax_f.set_title(_short(stem), fontsize=10, fontweight='bold')

        for j in range(len(key_neural_cols), len(axes_f_flat)):
            axes_f_flat[j].set_visible(False)

        fig_f.suptitle(
            'KEY THESIS FIGURE — Does better thalamic focal coverage\n'
            'predict a stronger target-specific neural effect (Comp 3)?\n'
            f'x = thalamus {primary_dose}  |  y = thalamus − ventricle net TUS effect',
            fontsize=12, fontweight='bold'
        )
        fig_f.tight_layout(rect=[0, 0, 1, 0.92])
        fig_f.savefig(dr_dir / 'KEY_ontarget_vs_comp3.png',
                      dpi=150, bbox_inches='tight')
        plt.close(fig_f)
        print('      Saved KEY_ontarget_vs_comp3.png  ← KEY THESIS FIGURE')

    # Fig G — Scatter for every FDR-significant dose × response pair
    if corr_df is not None and not corr_df.empty:
        sig_rows = corr_df[corr_df['sig_pearson'] == True]
        if sig_rows.empty:
            print('    [Fig G] No FDR-significant pairs — skipping individual scatters.')
        else:
            print(f'    [Fig G] Scatter plots for {len(sig_rows)} significant pairs …')
            prefix_map = {v: k for k, v in {
                'comp1_thal_net_':     'Comp1_Thal_NetEffect',
                'comp2_vent_net_':     'Comp2_Vent_NetEffect',
                'comp3_specificity_':  'Comp3_TargetSpecificity',
                'comp4_sham_balance_': 'Comp4_ShamSanity',
            }.items()}

            for _, row in sig_rows.iterrows():
                comp_label  = row['comparison']
                dose_metric = row['dose_metric']
                stem        = row['neural_feature']
                dose_tgt    = row['dose_target']
                prefix      = prefix_map.get(comp_label, '')
                dose_col_s  = f'{dose_tgt}_{dose_metric}'
                neural_col_s = f'{prefix}{stem}'

                if (dose_col_s not in linked.columns or
                        neural_col_s not in linked.columns):
                    continue

                x    = pd.to_numeric(linked[dose_col_s],   errors='coerce').values
                y    = pd.to_numeric(linked[neural_col_s], errors='coerce').values
                pids = linked['participant_id'].values
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() < MIN_N:
                    continue

                xm, ym   = x[mask], y[mask]
                pr       = row['pearson_r']
                pp_fdr   = row['pearson_p_fdr']
                lo, hi   = row['ci_lo'], row['ci_hi']
                coef     = np.polyfit(xm, ym, 1)
                x_line   = np.linspace(xm.min(), xm.max(), 200)
                y_line   = np.polyval(coef, x_line)
                resid    = ym - np.polyval(coef, xm)
                se_r     = np.std(resid, ddof=2) if len(xm) > 2 else 0

                fig_g, ax_g = plt.subplots(figsize=(6, 5))
                ax_g.scatter(xm, ym, color='#2C3E7A', s=65, zorder=4,
                             edgecolors='white', linewidths=0.7)
                for xi, yi, pid in zip(xm, ym, pids[mask]):
                    ax_g.annotate(str(pid), (xi, yi),
                                  textcoords='offset points', xytext=(4, 2),
                                  fontsize=7, alpha=0.85)
                ax_g.plot(x_line, y_line, color='crimson', lw=1.8)
                ax_g.fill_between(x_line,
                                  y_line - 1.96 * se_r,
                                  y_line + 1.96 * se_r,
                                  color='crimson', alpha=0.12)
                ax_g.axhline(0, color='grey', lw=0.6, ls='--', alpha=0.5)
                ax_g.set_xlabel(
                    f'{dose_tgt.capitalize()} '
                    f'{dose_metric.replace("_", " ")}',
                    fontsize=10
                )
                ax_g.set_ylabel(_short(stem), fontsize=10)
                ax_g.set_title(
                    f'{comp_label}  |  {_short(stem)}\n'
                    f'r = {pr:.2f}  [{lo:.2f}, {hi:.2f}]  '
                    f'p_FDR = {pp_fdr:.3f}  {_sig_stars(pp_fdr)}',
                    fontsize=10, fontweight='bold'
                )
                fig_g.tight_layout()
                safe = lambda s: re.sub(r'[^A-Za-z0-9_]', '', s)
                fname_g = (f'scatter_{safe(dose_metric)[:25]}_'
                           f'{safe(stem)[:35]}_{safe(comp_label)[:18]}.png')
                fig_g.savefig(dr_dir / fname_g, dpi=150, bbox_inches='tight')
                plt.close(fig_g)
            print(f'      Saved {len(sig_rows)} scatter plots')

    print(f'\n    All figures saved to:\n'
          f'      {rp_dir}\n      {dr_dir}')


# =============================================================================
# Summary console report
# =============================================================================

def print_summary(stats_df: pd.DataFrame, corr_df: pd.DataFrame) -> None:
    """Print a concise summary of all statistical findings to the console."""
    print('\n' + '=' * 72)
    print('LEVEL 3 SUMMARY')
    print('=' * 72)

    if stats_df is not None and not stats_df.empty:
        print('\n── RESPONSE STATISTICS (one-sample t-tests vs 0) ──')
        for comp in ['Comp1_Thal_NetEffect', 'Comp2_Vent_NetEffect',
                     'Comp3_TargetSpecificity', 'Comp4_ShamSanity']:
            sub = stats_df[stats_df['comparison'] == comp].copy()
            sig = sub[sub['significant'] == True]
            print(f'\n  {comp}')
            print(f'    Tested: {len(sub[sub["n"] >= MIN_N])}  |  FDR sig: {len(sig)}')
            for _, r in sig.sort_values('cohen_d', key=abs, ascending=False).iterrows():
                print(f'      {_short(r["feature"]):<28s}  '
                      f'd={r["cohen_d"]:+.2f}  '
                      f't={r["t_stat"]:+.2f}  '
                      f'p_FDR={r["p_fdr"]:.3f}  {_sig_stars(r["p_fdr"])}')

    if corr_df is not None and not corr_df.empty:
        print('\n── DOSE–RESPONSE CORRELATIONS ──')
        for comp in ['Comp1_Thal_NetEffect', 'Comp2_Vent_NetEffect',
                     'Comp3_TargetSpecificity', 'Comp4_ShamSanity']:
            sub = corr_df[(corr_df['comparison'] == comp) &
                          corr_df['sig_pearson'] == True].copy()
            if sub.empty:
                continue
            print(f'\n  {comp}  — significant dose × neural pairs:')
            for _, r in sub.sort_values('pearson_r', key=abs, ascending=False).head(5).iterrows():
                arrow = '↑' if r['pearson_r'] > 0 else '↓'
                print(f'    {arrow} {r["dose_metric"]:<33s} × '
                      f'{_short(r["neural_feature"]):<22s} '
                      f'r={r["pearson_r"]:+.2f}  p_FDR={r["pearson_p_fdr"]:.3f}')

        print('\n  KEY THESIS: OnTarget_6dB_percent × Comp3_TargetSpecificity')
        key_sub = corr_df[
            (corr_df['comparison'] == 'Comp3_TargetSpecificity') &
            (corr_df['dose_metric'] == 'OnTarget_6dB_pct')
        ].sort_values('pearson_r', key=abs, ascending=False)
        for _, r in key_sub.head(5).iterrows():
            print(f'    {_short(r["neural_feature"]):<30s}  '
                  f'r={r["pearson_r"]:+.2f}  '
                  f'95%CI [{r["ci_lo"]:+.2f}, {r["ci_hi"]:+.2f}]  '
                  f'p_FDR={r["pearson_p_fdr"]:.3f}  {_sig_stars(r["pearson_p_fdr"])}')

    print('\n' + '=' * 72 + '\n')


# =============================================================================
# Main pipeline
# =============================================================================

def run_dose_response_pipeline(
    all_session_csv: str = ALL_SESSION_FEATURES_CSV,
    results_root:    str = RESULTS_ROOT,
    output_dir:      str = OUTPUT_DIR,
):
    """
    Run the full Level 3 pipeline in order.

    Returns
    -------
    profile_df, stats_df, linked_df, corr_df
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    profile_df, stems = build_response_profile(all_session_csv, output_dir)
    stats_df          = run_response_statistics(profile_df, stems, output_dir)
    linked_df         = load_and_link_acoustic(profile_df, results_root, output_dir)
    corr_df           = run_dose_response_correlations(linked_df, stems, output_dir)
    plot_all(profile_df, stats_df, linked_df, corr_df, stems, output_dir)
    print_summary(stats_df, corr_df)

    return profile_df, stats_df, linked_df, corr_df


if __name__ == '__main__':
    run_dose_response_pipeline()