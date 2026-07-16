"""
dose_response_table.py — Per-participant dose + response summary table

WHAT THIS DOES (and deliberately does NOT do)
----------------------------------------------
For each participant, this script builds ONE merged row per session
(thalamus, ventricle) that places:

  - acoustic DOSE metrics (already computed by the k-Plan / Level-1
    acoustic pipeline: on-target coverage %, on-target intensity,
    plus a full list of every OTHER region the beam meaningfully
    reached during that session — its coverage %, intensity, and
    beam zone)

  side by side with

  - neural RESPONSE metrics (already computed by analysis.py: the
    active-minus-sham EEG features in {pid}_session_features.csv)

It does NOT compute any new derived quantity (no "specificity index",
no cross-session sham-balance check, no composite score). Those
values already exist in your pipeline outputs — this script only
reshapes and merges them into one readable table per participant,
which you can then write about directly:

    "Participant 03 received 55% on-target coverage (Isppa 5.9 W/cm2)
    at the thalamic session, and showed an active-minus-sham spindle
    amplitude change of X uV. The beam also reached the lateral
    ventricle (20% coverage, 2.1 W/cm2) and the left putamen (4%
    coverage, 1.3 W/cm2) during this session."

Output per participant:
  - {pid}_dose_response_table.csv   one row per session: on-target
    dose + response features (main table, for Results tables/prose)
  - {pid}_spillover_detail.csv      one row per session per OTHER
    region the beam reached (long format, for the Discussion section)

Plus combined versions across all participants.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# USER CONFIGURATION — edit these paths
# =============================================================================

ANALYSIS_OUTPUT_DIR = '/home/e_fabdulsa/Desktop/TUNeS_sharbie/results'
ACOUSTIC_ROOT = '/home/e_fabdulsa/Desktop/TUNeS_sharbie/acoustic_report'
OUTPUT_DIR = '/home/e_fabdulsa/Desktop/TUNeS_sharbie/results/dose_response_table'

PARTICIPANTS = ['02', '03', '06', '08', '10']

TARGET_FILE_MAP = {
    'thalamus':  'Left_Thalamus',
    'ventricle': 'Left_Lateral_Ventricle',
}

PRIMARY_DOSE_METRIC = 'OnTarget_6dB_pct'
INTENSITY_METRIC = 'Isppa_Target_Wcm2'

# Which ReportingTier values to scan for "other regions the beam reached".
# Tier1_Tissue is skull/scalp/brain-generic — usually not the kind of
# "spillover" you want to discuss (skull always gets dosed). Tier2_Nuclei
# and Tier3_Atlas are the named subcortical/cortical structures.
SPILLOVER_TIERS = ['Tier2_Nuclei', 'Tier3_Atlas']

# A region counts as "reached by the beam" if EITHER of these holds:
#   - its on-target coverage (% of -6dB focal volume overlapping it) > 0
#   - its peak in-region intensity is at/above this floor (W/cm2)
# 0.5 W/cm2 matches the "hot region" threshold already used by the
# report generator (isppa_overlay_threshold), so flagged regions here
# line up with what's visually flagged in the HTML report.
SPILLOVER_ISPPA_FLOOR = 0.5

# Response features to report. These must match the stems used in
# {pid}_session_features.csv as event_locked_active_minus_sham_{stem}.
RESPONSE_FEATURES = [
    'spindle_rate_per_burst',
    'spindle_amplitude_uv',
    'spindle_density_per_s',
    'spindle_frequency_hz',
    'spindle_duration_sec',
]


# =============================================================================
# Small helpers
# =============================================================================

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
# STEP 1 — Dose metrics per participant per session
#
#   on-target:  coverage % AND intensity for the session's own structure
#   spillover:  EVERY other named region (Tier2_Nuclei / Tier3_Atlas) that
#               the beam meaningfully reached, each with coverage %,
#               intensity, and beam zone (pre-focal / at focus / post-focal)
# =============================================================================

def load_participant_dose(participant_id: str, acoustic_root: str) -> dict:
    """
    Returns:
        {
          'thalamus':  {
              'ontarget_pct':   .., 'ontarget_isppa': ..,
              'source_file':    ..,
              'spillover': [
                  {'region': .., 'tier': .., 'ontarget_pct': ..,
                   'isppa_wcm2': .., 'beam_zone': ..},
                  ...  # sorted by intensity, descending
              ],
          },
          'ventricle': {...},
        }
    """
    root = Path(acoustic_root)
    dose = {}

    for session_target, structure_name in TARGET_FILE_MAP.items():
        candidates = sorted(root.glob(f'**/*{participant_id}*_{structure_name}*_analysis.csv'))
        if not candidates:
            print(f'  [{participant_id}/{session_target}] No acoustic CSV found — dose skipped')
            continue

        # Pick whichever candidate file gives the best on-target coverage
        # for this session's OWN structure (handles multiple placements).
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

        ontarget_isppa = pd.to_numeric(best_row.get(INTENSITY_METRIC), errors='coerce')

        # Scan the SAME file for every other named region the beam reached.
        df_full = pd.read_csv(best_csv)
        own_name_lower = structure_name.lower()

        spill_rows = []
        for tier in SPILLOVER_TIERS:
            sub = df_full[df_full.get('ReportingTier') == tier]
            for _, r in sub.iterrows():
                region_name = str(r.get('TargetName', r.get('name', '')))
                if not region_name or own_name_lower in region_name.lower():
                    continue  # skip the session's own target

                cov = pd.to_numeric(r.get(PRIMARY_DOSE_METRIC), errors='coerce')
                isppa = pd.to_numeric(r.get(INTENSITY_METRIC), errors='coerce')

                reached = (pd.notna(cov) and cov > 0) or (
                    pd.notna(isppa) and isppa >= SPILLOVER_ISPPA_FLOOR
                )
                if not reached:
                    continue

                spill_rows.append({
                    'region':      region_name,
                    'tier':        tier,
                    'ontarget_pct': round(float(cov), 2) if pd.notna(cov) else np.nan,
                    'isppa_wcm2':   round(float(isppa), 4) if pd.notna(isppa) else np.nan,
                    'beam_zone':    r.get('BeamZone', r.get('beam_zone', '—')),
                })

        spill_rows.sort(
            key=lambda x: x['isppa_wcm2'] if pd.notna(x['isppa_wcm2']) else -1,
            reverse=True,
        )

        dose[session_target] = {
            'ontarget_pct':   round(float(best_cov), 2),
            'ontarget_isppa': round(float(ontarget_isppa), 4) if pd.notna(ontarget_isppa) else np.nan,
            'source_file':    best_csv.name,
            'spillover':      spill_rows,
        }
        print(f'  [{participant_id}/{session_target}] on-target={best_cov:.1f}%  '
              f'{len(spill_rows)} other region(s) reached  ({best_csv.name})')

    return dose


# =============================================================================
# STEP 2 — Response metrics per session, read directly (no new math)
# =============================================================================

def load_participant_response(participant_id: str, analysis_output_dir: str) -> dict:
    """
    Returns:
        {
          'thalamus':  {stem: active_minus_sham_value, ...},
          'ventricle': {stem: active_minus_sham_value, ...},
        }
    Pulled directly from the event_locked_active_minus_sham_{stem} columns
    already written by analysis.py. No recomputation.
    """
    csv_path = Path(analysis_output_dir) / participant_id / f'{participant_id}_session_features.csv'
    if not csv_path.exists():
        print(f'  [{participant_id}] session_features.csv not found — response skipped')
        return {}

    df = pd.read_csv(csv_path)
    if 'is_adaptation' in df.columns:
        df = df[df['is_adaptation'].astype(str).str.upper() != 'TRUE'].copy()

    response = {}
    for session_target in TARGET_FILE_MAP:
        rows = df[df['target'].str.lower().str.contains(session_target[:4], na=False)]
        if rows.empty:
            print(f'  [{participant_id}/{session_target}] session row not found in feature table')
            continue
        row = rows.iloc[0]

        feat_vals = {}
        for stem in RESPONSE_FEATURES:
            col = f'event_locked_active_minus_sham_{stem}'
            val = pd.to_numeric(row.get(col), errors='coerce')
            feat_vals[stem] = round(float(val), 4) if pd.notna(val) else np.nan
        response[session_target] = feat_vals

    return response


# =============================================================================
# STEP 3 — Main table: one row per session (dose + response)
#
# The full spillover list is summarized here as a count + top region only,
# so this table stays compact enough for a Results table. The full list
# per region goes in the separate spillover-detail table (Step 4) for the
# Discussion section.
# =============================================================================

def build_participant_table(participant_id: str, dose: dict, response: dict) -> pd.DataFrame:
    rows = []
    for session_target in TARGET_FILE_MAP:
        d = dose.get(session_target, {})
        spill = d.get('spillover', [])

        row = {'participant': participant_id, 'session_target': session_target}
        row['ontarget_pct'] = d.get('ontarget_pct', np.nan)
        row['ontarget_isppa_wcm2'] = d.get('ontarget_isppa', np.nan)
        row['source_file'] = d.get('source_file', '')
        row['n_other_regions_reached'] = len(spill)
        row['top_spillover_region'] = spill[0]['region'] if spill else ''
        row['top_spillover_isppa_wcm2'] = spill[0]['isppa_wcm2'] if spill else np.nan
        row['top_spillover_pct'] = spill[0]['ontarget_pct'] if spill else np.nan

        row.update(response.get(session_target, {}))
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# STEP 4 — Spillover detail table: one row per session per OTHER region
# reached. This is the table to pull from for the Discussion section
# ("beyond the intended target, the beam also reached...").
# =============================================================================

def build_spillover_detail(participant_id: str, dose: dict) -> pd.DataFrame:
    rows = []
    for session_target in TARGET_FILE_MAP:
        d = dose.get(session_target, {})
        for s in d.get('spillover', []):
            rows.append({
                'participant': participant_id,
                'session_target': session_target,
                'region': s['region'],
                'tier': s['tier'],
                'ontarget_pct': s['ontarget_pct'],
                'isppa_wcm2': s['isppa_wcm2'],
                'beam_zone': s['beam_zone'],
            })
    return pd.DataFrame(rows)


# =============================================================================
# Per-participant runner
# =============================================================================

def process_participant(participant_id: str) -> tuple:
    print(f'\n{participant_id}: loading dose + response...')
    dose = load_participant_dose(participant_id, ACOUSTIC_ROOT)
    response = load_participant_response(participant_id, ANALYSIS_OUTPUT_DIR)

    table = build_participant_table(participant_id, dose, response)
    spill_detail = build_spillover_detail(participant_id, dose)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f'{participant_id}_dose_response_table.csv'
    table.to_csv(out_path, index=False)
    print(f'  Saved {out_path.name}')

    spill_path = out_dir / f'{participant_id}_spillover_detail.csv'
    spill_detail.to_csv(spill_path, index=False)
    print(f'  Saved {spill_path.name}')

    return table, spill_detail


if __name__ == '__main__':
    all_tables, all_spillover = [], []
    for pid in PARTICIPANTS:
        try:
            table, spill_detail = process_participant(pid)
            all_tables.append(table)
            all_spillover.append(spill_detail)
        except Exception as exc:
            print(f'  {pid} failed: {exc}')

    if all_tables:
        combined = pd.concat(all_tables, ignore_index=True)
        combined_path = Path(OUTPUT_DIR) / 'all_participants_dose_response_table.csv'
        combined.to_csv(combined_path, index=False)
        print(f'\nSaved combined table: {combined_path}')
        print(combined.to_string(index=False))

    if all_spillover:
        combined_spill = pd.concat(all_spillover, ignore_index=True)
        combined_spill_path = Path(OUTPUT_DIR) / 'all_participants_spillover_detail.csv'
        combined_spill.to_csv(combined_spill_path, index=False)
        print(f'\nSaved combined spillover detail: {combined_spill_path}')
        print(combined_spill.to_string(index=False))