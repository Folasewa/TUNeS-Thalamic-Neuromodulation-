"""
dose_response_single_participant.py — Per-participant dose vs. response report

WHAT THIS DOES (and does not do)
---------------------------------
For ONE participant at a time (mirrors analysis.py's process_participant style):

  1. Loads that participant's acoustic dose report (Level 1 kPlan CSVs) for
     the thalamus session and the ventricle session, and extracts:
        - on-target dose  (row matching the intended target structure)
        - spillover dose  (row matching the OTHER structure, i.e. how much
                            the beam leaked into it)

  2. Loads that participant's neural session-feature table (written by
     analysis.py as {pid}_session_features.csv) and computes, per feature:
        Comp 1: thalamus session, active − sham         (TUS effect at target)
        Comp 2: ventricle session, active − sham        (non-specific effect)
        Comp 3: Comp1 − Comp2                            (target specificity)
        Comp 4: thal sham − vent sham                    (sanity check ≈ 0)

  3. Writes ONE CSV and ONE figure per participant that puts dose and
     response side by side, so the write-up can make a qualitative /
     descriptive coherence claim: "this person got X% on-target coverage
     with Y% spillover, and showed a Comp3 of Z."

"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# USER CONFIGURATION — edit these paths
# =============================================================================

# Root folder that holds {pid}/{pid}_session_features.csv, written by analysis.py
ANALYSIS_OUTPUT_DIR = '/home/e_fabdulsa/Desktop/TUNeS_sharbie/results'

# Folder that contains the Level 1 acoustic per-region CSVs
# (…/{pid}_{Structure}_analysis.csv, possibly nested in sub-folders)
ACOUSTIC_ROOT = '/home/e_fabdulsa/Desktop/TUNeS_sharbie/acoustic_report'

# Where to write the per-participant dose-response reports
OUTPUT_DIR = '/home/e_fabdulsa/Desktop/TUNeS_sharbie/results/dose_response_single'

PARTICIPANTS = ['02', '03', '06', '08', '10']

TARGET_FILE_MAP = {
    'thalamus':  'Left_Thalamus',
    'ventricle': 'Left_Lateral_Ventricle',
}

# The dose metric used for the headline dose-profile figure.
PRIMARY_DOSE_METRIC = 'OnTarget_6dB_pct'

# Feature stems to highlight in the Comp 1-4 figure. If a stem isn't found
# in this participant's data it is silently skipped.
KEY_FEATURES = [
    'spindle_rate_per_burst',
    'spindle_amplitude_uv',
    'spindle_density_per_s',
    'total_spindles_in_windows',
]


# =============================================================================
# Small helpers
# =============================================================================

def _el_stems(df: pd.DataFrame) -> list:
    """Feature stems from event_locked_active_minus_sham_* columns."""
    prefix = 'event_locked_active_minus_sham_'
    return [c[len(prefix):] for c in df.columns if c.startswith(prefix)]


def _short(stem: str) -> str:
    return (stem
            .replace('spindle_', 'sp_')
            .replace('_per_burst', '/burst')
            .replace('_per_s', '/s')
            .replace('_uv', ' µV')
            .replace('_hz', ' Hz')
            .replace('_sec', ' s')
            .replace('total_spindles_in_windows', 'total_sp'))


def _best_row(df: pd.DataFrame, target_name: str, tier: str = 'Tier2_Nuclei'):
    """Best (highest on-target coverage) sonication row for a given target."""
    sub = df[
        (df.get('ReportingTier') == tier) &
        (df['TargetName'].str.lower().str.contains(target_name.lower(), na=False))
    ]
    if sub.empty:
        return None
    coverage_col = PRIMARY_DOSE_METRIC if PRIMARY_DOSE_METRIC in sub.columns else sub.columns[0]
    return sub.sort_values(coverage_col, ascending=False).iloc[0]


# =============================================================================
# STEP 1 — Load this participant's dose (on-target + spillover) per session
# =============================================================================

def load_participant_dose(participant_id: str, acoustic_root: str) -> dict:
    """
    Returns a dict:
        {
          'thalamus':  {'ontarget_pct': .., 'spillover_pct': ..},
          'ventricle': {'ontarget_pct': .., 'spillover_pct': ..},
        }
    'ontarget_pct'  = PRIMARY_DOSE_METRIC of the row matching the session's
                       own intended target.
    'spillover_pct' = PRIMARY_DOSE_METRIC of the row matching the OTHER
                       structure within the same session's CSV (i.e. how
                       much this session's beam crossed into the structure
                       it was NOT aimed at).
    """
    root = Path(acoustic_root)
    dose = {}

    for session_target, structure_name in TARGET_FILE_MAP.items():
        other_target = 'ventricle' if session_target == 'thalamus' else 'thalamus'
        other_structure = TARGET_FILE_MAP[other_target]

        candidates = sorted(root.glob(f'**/*{participant_id}*_{structure_name}*_analysis.csv'))
        if not candidates:
            print(f'  [{participant_id}/{session_target}] No acoustic CSV found — dose skipped')
            continue

        # If more than one file exists (multiple placements), pick whichever
        # gives the best on-target coverage for THIS session's own target.
        best_csv, best_row, best_cov = None, None, -np.inf
        for csv_path in candidates:
            df = pd.read_csv(csv_path)
            row = _best_row(df, structure_name)
            if row is None:
                continue
            cov = pd.to_numeric(row.get(PRIMARY_DOSE_METRIC), errors='coerce')
            if pd.notna(cov) and cov > best_cov:
                best_cov, best_row, best_csv = cov, row, csv_path

        if best_row is None:
            print(f'  [{participant_id}/{session_target}] No valid on-target row found')
            continue

        # Spillover: same CSV file, row matching the OTHER structure
        df_full = pd.read_csv(best_csv)
        spill_row = _best_row(df_full, other_structure)
        spill_val = (pd.to_numeric(spill_row.get(PRIMARY_DOSE_METRIC), errors='coerce')
                     if spill_row is not None else np.nan)

        dose[session_target] = {
            'ontarget_pct':  round(float(best_cov), 2),
            'spillover_pct': round(float(spill_val), 2) if pd.notna(spill_val) else np.nan,
            'source_file':   best_csv.name,
        }
        print(f'  [{participant_id}/{session_target}] on-target={best_cov:.1f}%  '
              f'spillover→{other_target}={spill_val:.1f}%  ({best_csv.name})')

    return dose


# =============================================================================
# STEP 2 — Compute this participant's Comp 1-4 per feature
# =============================================================================

def compute_participant_comps(participant_id: str, analysis_output_dir: str):
    """
    Reads {pid}_session_features.csv (written by analysis.py) and computes
    Comp 1-4 for each event-locked feature stem, for THIS participant only.
    Returns (comps_df, stems). comps_df has one row per feature.
    """
    csv_path = Path(analysis_output_dir) / participant_id / f'{participant_id}_session_features.csv'
    if not csv_path.exists():
        raise FileNotFoundError(
            f'Session feature table not found: {csv_path}\nRun analysis.py first.'
        )
    df = pd.read_csv(csv_path)

    if 'is_adaptation' in df.columns:
        df = df[df['is_adaptation'].astype(str).str.upper() != 'TRUE'].copy()

    thal_rows = df[df['target'].str.lower().str.contains('thal', na=False)]
    vent_rows = df[df['target'].str.lower().str.contains('vent', na=False)]
    if thal_rows.empty or vent_rows.empty:
        raise RuntimeError(
            f'{participant_id}: missing thalamus or ventricle experimental session '
            f'in {csv_path.name} — cannot compute Comp 1-4.'
        )
    thal, vent = thal_rows.iloc[0], vent_rows.iloc[0]

    stems = _el_stems(df)
    rows = []
    for stem in stems:
        ams_col    = f'event_locked_active_minus_sham_{stem}'
        active_col = f'event_locked_active_{stem}'
        sham_col   = f'event_locked_sham_{stem}'

        thal_net = pd.to_numeric(thal.get(ams_col), errors='coerce')
        vent_net = pd.to_numeric(vent.get(ams_col), errors='coerce')
        thal_sham = pd.to_numeric(thal.get(sham_col), errors='coerce')
        vent_sham = pd.to_numeric(vent.get(sham_col), errors='coerce')

        if np.isnan(thal_net):
            a, s = pd.to_numeric(thal.get(active_col), errors='coerce'), thal_sham
            thal_net = a - s if not (np.isnan(a) or np.isnan(s)) else np.nan
        if np.isnan(vent_net):
            a, s = pd.to_numeric(vent.get(active_col), errors='coerce'), vent_sham
            vent_net = a - s if not (np.isnan(a) or np.isnan(s)) else np.nan

        comp3 = thal_net - vent_net if not (np.isnan(thal_net) or np.isnan(vent_net)) else np.nan
        comp4 = thal_sham - vent_sham if not (np.isnan(thal_sham) or np.isnan(vent_sham)) else np.nan

        rows.append({
            'feature': stem,
            'comp1_thal_net':      round(thal_net, 4) if pd.notna(thal_net) else np.nan,
            'comp2_vent_net':      round(vent_net, 4) if pd.notna(vent_net) else np.nan,
            'comp3_specificity':   round(comp3, 4)    if pd.notna(comp3)    else np.nan,
            'comp4_sham_balance':  round(comp4, 4)    if pd.notna(comp4)    else np.nan,
        })

    return pd.DataFrame(rows), stems


# =============================================================================
# STEP 3 — Report (CSV + one figure)
# =============================================================================

def write_participant_report(participant_id, dose, comps_df, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- CSV: dose summary + comp table stacked for easy reading ----
    dose_rows = []
    for target, vals in dose.items():
        dose_rows.append({'session_target': target, **vals})
    dose_df = pd.DataFrame(dose_rows)
    dose_csv = out_dir / f'{participant_id}_dose_summary.csv'
    dose_df.to_csv(dose_csv, index=False)

    comps_csv = out_dir / f'{participant_id}_comp_summary.csv'
    comps_df.to_csv(comps_csv, index=False)
    print(f'  Saved {dose_csv.name}, {comps_csv.name}')

    # ---- Figure: dose bars (left) next to Comp1-4 bars for key features (right) ----
    key_rows = comps_df[comps_df['feature'].isin(KEY_FEATURES)]
    if key_rows.empty:
        key_rows = comps_df.head(6)  # fallback: just show whatever exists

    fig, (ax_dose, ax_comp) = plt.subplots(1, 2, figsize=(12, 5))

    # Dose panel
    targets = list(dose.keys())
    ontarget  = [dose[t]['ontarget_pct'] for t in targets]
    spillover = [dose[t]['spillover_pct'] for t in targets]
    x = np.arange(len(targets))
    width = 0.35
    ax_dose.bar(x - width/2, ontarget,  width, label='On-target %', color='#2C7BB6')
    ax_dose.bar(x + width/2, spillover, width, label='Spillover %', color='#C0392B')
    ax_dose.set_xticks(x)
    ax_dose.set_xticklabels([t.title() for t in targets])
    ax_dose.set_ylabel(f'{PRIMARY_DOSE_METRIC} (%)')
    ax_dose.set_title(f'{participant_id}: acoustic dose per session')
    ax_dose.legend(fontsize=8)
    ax_dose.spines[['top', 'right']].set_visible(False)

    # Comp panel
    labels = [_short(s) for s in key_rows['feature']]
    comp_cols = ['comp1_thal_net', 'comp2_vent_net', 'comp3_specificity', 'comp4_sham_balance']
    comp_colors = ['#E04B4B', '#4B7BE0', '#8E44AD', '#7F8C8D']
    y = np.arange(len(labels))
    bar_h = 0.18
    for i, (col, color) in enumerate(zip(comp_cols, comp_colors)):
        ax_comp.barh(y + i * bar_h, key_rows[col].values, height=bar_h,
                     label=col.replace('comp', 'C').replace('_', ' '), color=color)
    ax_comp.axvline(0, color='black', lw=0.8)
    ax_comp.set_yticks(y + 1.5 * bar_h)
    ax_comp.set_yticklabels(labels, fontsize=8)
    ax_comp.set_title(f'{participant_id}: neural response (Comp 1-4)')
    ax_comp.legend(fontsize=7, loc='lower right')
    ax_comp.spines[['top', 'right']].set_visible(False)

    fig.suptitle(
        f'{participant_id}: acoustic dose vs. neural response (single-participant view)\n'
        f'Descriptive only — no inferential stats at n=1',
        fontsize=11, fontweight='bold'
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig_path = out_dir / f'{participant_id}_dose_vs_response.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fig_path.name}')

    # ---- Console summary sentence, ready to paste into notes ----
    print(f'\n  ── {participant_id} summary ──')
    for t in targets:
        print(f'    {t.title()}: on-target={dose[t]["ontarget_pct"]}%  '
              f'spillover={dose[t]["spillover_pct"]}%')
    for _, r in key_rows.iterrows():
        print(f'    {_short(r["feature"]):<22s}  Comp3(specificity)={r["comp3_specificity"]:+.3f}')
    print()


# =============================================================================
# Per-participant runner
# =============================================================================

def process_participant(participant_id: str):
    print('\n' + '=' * 70)
    print(f'DOSE-RESPONSE (single participant): {participant_id}')
    print('=' * 70)
    try:
        dose = load_participant_dose(participant_id, ACOUSTIC_ROOT)
        if not dose:
            print(f'  {participant_id}: no dose data found — skipping report')
            return
        comps_df, stems = compute_participant_comps(participant_id, ANALYSIS_OUTPUT_DIR)
        write_participant_report(participant_id, dose, comps_df, OUTPUT_DIR)
    except Exception as exc:
        print(f'  {participant_id} failed: {exc}')


if __name__ == '__main__':
    for pid in PARTICIPANTS:
        process_participant(pid)