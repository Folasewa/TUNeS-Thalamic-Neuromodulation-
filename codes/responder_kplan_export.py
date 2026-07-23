"""
responder_kplan_export.py

Does exactly the simple thing:
  1. Load a subject's per-pulse features CSV (from analysis.py)
  2. Keep only 60W ('active_60w') trials
  3. Split into responder (brain_state == 'spindle' or 'slow_wave')
     vs non-responder (brain_state == 'none')
  4. For each trial, use first_trigger_seq_all to look up that pulse's
     coordinates in the Localite trigger-marker XML
  5. Convert to kPlan's coordinate convention and write a .kps file

Only wrinkle: if this session's EEG recording was split into multiple
files, first_trigger_seq_all resets partway through and can no longer be
trusted to index directly into the trigger XML for later blocks. This
script detects that automatically (a backwards jump in burst_time_s) and:
  - safely exports every trial in the FIRST recording block
  - prints (and skips) trials from any later block, since those need the
    corresponding block's vmrk pulse count before they can be matched
    correctly. Fill that in later via --vmrk-pulse-counts once you have it.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import xml.etree.ElementTree as ET


# ============================================================
# CONFIG — edit these paths, then just run the script
# ============================================================

per_pulse_csv = "/Users/folasewaabdulsalam/Downloads/03_03_thalamus_nrem_per_pulse_features.csv"
trigger_xml   = "/Users/folasewaabdulsalam/Downloads/TriggerMarkers_Coil0_20260504231624578.xml"
output_dir    = "/Users/folasewaabdulsalam/Downloads/kplan_output_03_thalamus"

condition   = "active_60w"   # which trials count as "60W trials"
kplan_offset = 10.82          # same offset as your other kPlan conversion code
n_per_group  = 3               # max clearest responder trials per response type
                                # (spindle, slow_wave), each matched with the
                                # same number of non-responder trials


# ============================================================
# Trigger XML parsing (same logic as your coordinate-export code)
# ============================================================

def parse_matrix4d(trigger_element):
    matrix4d = trigger_element.find("Matrix4D")
    if matrix4d is None:
        return None
    matrix = np.zeros((4, 4), dtype=float)
    for row in range(4):
        for col in range(4):
            value = matrix4d.get(f"data{row}{col}")
            matrix[row, col] = np.nan if value is None else float(value)
    return matrix


def build_pulse_database(trigger_file):
    """One row per TriggerMarker, with x/y/z and a 1-based valid-pulse index."""
    tree = ET.parse(trigger_file)
    root = tree.getroot()
    trigger_markers = root.findall(".//TriggerMarker")

    rows = []
    for idx0, trig in enumerate(trigger_markers):
        matrix = parse_matrix4d(trig)
        x = y = z = np.nan
        valid = False
        if matrix is not None:
            x, y, z = matrix[0, 3], matrix[1, 3], matrix[2, 3]
            is_zero = np.isclose(x, 0.0) and np.isclose(y, 0.0) and np.isclose(z, 0.0)
            has_nan = pd.isna(x) or pd.isna(y) or pd.isna(z)
            valid = (not is_zero) and (not has_nan)
        rows.append({
            "original_index_0based": idx0,
            "is_valid_position": valid,
        })

    df = pd.DataFrame(rows)
    df["valid_pulse_index_1based"] = np.nan
    mask = df["is_valid_position"] == True
    df.loc[mask, "valid_pulse_index_1based"] = np.arange(1, mask.sum() + 1)
    return df, root, trigger_markers


# ============================================================
# kPlan conversion (same as your existing code)
# ============================================================

def convert_Localite_to_kPlan_position_matrix(Localite_position_matrix, offset):
    kPlan_position_matrix = Localite_position_matrix.copy()
    kPlan_position_matrix[0:3, 0] = -Localite_position_matrix[0:3, 1]
    kPlan_position_matrix[0:3, 1] = Localite_position_matrix[0:3, 2]
    kPlan_position_matrix[0:3, 2] = -Localite_position_matrix[0:3, 0]
    kPlan_position_matrix[0:3, 3] = (
        kPlan_position_matrix[0:3, 3] - offset * kPlan_position_matrix[0:3, 0]
    )
    kPlan_position_matrix[0:3, 3] = kPlan_position_matrix[0:3, 3] / 1000
    return kPlan_position_matrix


def create_kps_file_for_kPlan(kplan_position_matrix, output_path, kps_filename):
    position_matrix = np.transpose(kplan_position_matrix).reshape((1, 4, 4))
    output_filepath = os.path.join(output_path, kps_filename + ".kps")
    with h5py.File(output_filepath, "w") as f:
        dset = f.create_dataset("/1/position_transform", (1, 4, 4), dtype="float32")
        dset[:] = position_matrix.astype("float32")
        f["/1"].attrs.create("transform_label", np.bytes_(kps_filename, "utf-8"))
        f.attrs.create("application_name", np.bytes_("k-Plan", "utf-8"))
        f.attrs.create("file_type", np.bytes_("k-Plan Transducer Position", "utf-8"))
        f.attrs.create("number_transforms", np.array([1], dtype=np.uint64))
    return output_filepath


# ============================================================
# Recording-block detection (handles the split-file case automatically)
# ============================================================

def flag_recording_blocks(df):
    """
    Detects EEG-file restarts within one session's per-pulse CSV by looking
    for backwards jumps in burst_time_s (a session with a single continuous
    recording will have block == 0 for every row).
    """
    df = df.reset_index(drop=True).copy()
    resets = df["burst_time_s"].diff() < 0
    df["block"] = resets.cumsum()
    return df


# ============================================================
# Main selection + export
# ============================================================

def _add_change_columns(df):
    """Compute per-channel sigma/delta power change (post - pre) if not present."""
    df = df.copy()
    channels = ["C3", "C4", "Cz", "F3", "F4", "Fz"]
    for ch in channels:
        sig_col = f"{ch}_sigma_power_change"
        if sig_col not in df.columns and f"{ch}_pre_sigma_power" in df.columns:
            df[sig_col] = df[f"{ch}_post_sigma_power"] - df[f"{ch}_pre_sigma_power"]
        delta_col = f"{ch}_delta_power_change"
        if f"{ch}_pre_delta_power" in df.columns:
            df[delta_col] = df[f"{ch}_post_delta_power"] - df[f"{ch}_pre_delta_power"]
    return df


def _effect_strength(row, channels, suffix):
    """Average change across available channels for ranking 'clearest' trials."""
    vals = [row.get(f"{ch}_{suffix}") for ch in channels]
    vals = [v for v in vals if pd.notna(v)]
    return np.mean(vals) if vals else np.nan


def _pick_matched_nonresponders(responders, nonresponders, n):
    """For each chosen responder trial, pick the nearest-in-time unused non-responder."""
    pool = nonresponders.copy()
    picked = []
    for _, r in responders.iterrows():
        if pool.empty:
            break
        idx = (pool["burst_time_s"] - r["burst_time_s"]).abs().idxmin()
        picked.append(pool.loc[idx])
        pool = pool.drop(idx)
    picked_df = pd.DataFrame(picked)
    return picked_df.head(n)


def select_and_export(per_pulse_csv, trigger_xml, output_dir,
                       condition="active_60w", offset=10.82, n_per_group=3):
    df = pd.read_csv(per_pulse_csv)
    df = flag_recording_blocks(df)
    df = _add_change_columns(df)

    active = df[df["condition"] == condition].copy()

    n_blocks = active["block"].nunique()
    if n_blocks > 1:
        print(f"NOTE: {n_blocks} recording blocks detected in this session "
              f"(EEG file restart mid-session).")
        print("Only block 0 (the first block) can be safely matched to the "
              "trigger XML right now — later blocks need that block's vmrk "
              "pulse count first. Skipping these rows for now:\n")
        print(active.loc[active["block"] > 0,
                          ["burst_time_s", "first_trigger_seq_all", "brain_state", "block"]
                          ].to_string(index=False))
        print()

    safe = active[active["block"] == 0].copy()

    channels = ["C3", "C4", "Cz", "F3", "F4", "Fz"]
    spindle_all  = safe[safe["brain_state"] == "spindle"].copy()
    slowwave_all = safe[safe["brain_state"] == "slow_wave"].copy()
    nonresp_all  = safe[safe["brain_state"] == "none"].copy()

    spindle_all["effect_strength"] = spindle_all.apply(
        lambda r: _effect_strength(r, channels, "sigma_power_change"), axis=1)
    slowwave_all["effect_strength"] = slowwave_all.apply(
        lambda r: _effect_strength(r, channels, "delta_power_change"), axis=1)

    top_spindle  = spindle_all.sort_values("effect_strength", ascending=False).head(n_per_group)
    top_slowwave = slowwave_all.sort_values("effect_strength", ascending=False).head(n_per_group)

    pool = nonresp_all.copy()
    matched_for_spindle = _pick_matched_nonresponders(top_spindle, pool, n_per_group)
    pool = pool.drop(matched_for_spindle.index, errors="ignore")
    matched_for_slowwave = _pick_matched_nonresponders(top_slowwave, pool, n_per_group)

    selection = []
    for _, r in top_spindle.iterrows():
        selection.append((r, "responder", "spindle"))
    for _, r in matched_for_spindle.iterrows():
        selection.append((r, "nonresponder", "matched_to_spindle"))
    for _, r in top_slowwave.iterrows():
        selection.append((r, "responder", "slow_wave"))
    for _, r in matched_for_slowwave.iterrows():
        selection.append((r, "nonresponder", "matched_to_slow_wave"))

    print(f"Selected {len(top_spindle)} spindle responder(s), "
          f"{len(matched_for_spindle)} matched non-responder(s),")
    print(f"         {len(top_slowwave)} slow_wave responder(s), "
          f"{len(matched_for_slowwave)} matched non-responder(s)")
    print(f"Total placements to simulate: {len(selection)}\n")

    trig_df, root, trigger_markers = build_pulse_database(trigger_xml)

    os.makedirs(output_dir, exist_ok=True)
    exported_rows = []

    for row, response_group, pair_label in selection:
        seq = row["first_trigger_seq_all"]
        match = trig_df[trig_df["valid_pulse_index_1based"] == seq]
        if match.empty:
            print(f"  no trigger-XML match for pulse seq {seq} — skipping")
            continue

        trig_idx0 = int(match.iloc[0]["original_index_0based"])
        matrix = parse_matrix4d(trigger_markers[trig_idx0])
        if matrix is None:
            print(f"  no position matrix for pulse seq {seq} — skipping")
            continue

        kplan_matrix = convert_Localite_to_kPlan_position_matrix(matrix, offset)
        label = (f"{response_group}_{pair_label}_burst{int(row['burst_seq_all'])}"
                  f"_seq{int(seq)}")
        kps_path = create_kps_file_for_kPlan(kplan_matrix, output_dir, label)

        exported_rows.append({
            "burst_seq_all": row["burst_seq_all"],
            "burst_time_s": row["burst_time_s"],
            "brain_state": row["brain_state"],
            "response_group": response_group,
            "pair_label": pair_label,
            "first_trigger_seq_all": seq,
            "kps_path": kps_path,
        })
        print(f"Exported {label} -> {kps_path}")

    summary_df = pd.DataFrame(exported_rows)
    summary_csv = os.path.join(output_dir, "exported_pulses_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nWrote {len(summary_df)} .kps files. Summary: {summary_csv}")
    return summary_df


if __name__ == "__main__":
    select_and_export(
        per_pulse_csv, trigger_xml, output_dir,
        condition=condition, offset=kplan_offset, n_per_group=n_per_group,
    )
