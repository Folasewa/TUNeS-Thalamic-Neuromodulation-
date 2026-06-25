
"""

LEVEL 1: The Acoustic Reporting
"""

"""
================================================================================
Focused Ultrasound Post-Processing and Reporting Script
================================================================================

Inputs:
    - kPlan RESULTS.h5 file
    - SimNIBS segmentation (final_tissues.nii / .nii.gz)
    - Participant nuclei masks folder (one .nii.gz per nucleus)
    - Individualized cortical + subcortical atlases 
    - T1 MRI (optional, for visualization)

Reporting tiers (all in one CSV):
    Tier 1 — Tissue safety   : skull, scalp, eyes (MI flagged), brain (SimNIBS)
    Tier 2 — Nuclei targets  : all masks in TARGET_MASK_FOLDER
    Tier 3 — Brain regions   : Individualized cortical and subcortical regions

Focal zone definition:
    - Intensity field (W/cm²) computed from pressure
    - Peak intensity found WITHIN BRAIN MASK ONLY
    - -6 dB threshold = 25% of brain peak intensity
    - -3 dB threshold = 50% of brain peak intensity
    - Focal zone masks restricted to brain voxels

Pressure reporting:
    - Peak and mean pressure (kPa) per region
    - Peak and mean pressure in overlap zones (-6dB, -3dB)
    - Mechanical Index (MI) with safety flags

Author: Folasewa Abdulsalam
================================================================================
"""

# Imports
import os
import glob
import h5py
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_img
import matplotlib
#matplotlib.use("Agg")                   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import binary_dilation

# Configuration — edit these paths before running

H5_FILES = [
    "/content/drive/MyDrive/TUNES/tunes_h5/PLAN-150-1041-RESULTS.h5",

]

SIMNIBS_PATH        = "/content/drive/MyDrive/TUNES/final_tissues.nii.gz"
T1_PATH             = "/content/drive/MyDrive/TUNES/T1.nii.gz"
TARGET_MASK_FOLDER  = "/content/drive/MyDrive/TUNES/participant_nuclei"
OUTPUT_DIR          = "/content/drive/MyDrive/TUNES/results/150"
INDIVIDUALIZED_CORTICAL_SUBCORTICAL_MASKS = "/content/drive/MyDrive/TUNES/ho_masks"
# SimNIBS tissue label definitions

TISSUE_GROUPS = {
    "brain":        [1, 2, 3],   # white matter + gray matter + CSF
    "skull":        [7, 8],      # compact + spongy bone
    "scalp":        [5],
    "eyes":         [6],
}

# Safety thresholds
MI_LIMIT_BRAIN = 1.9
MI_LIMIT_EYES  = 0.4


#partipant_id
PARTICIPANT_ID = "tunes-02" # what you planned in kPlan 

#intented target
INTENDED_TARGET = 'thalamus'

# Acoustic impedance of tissue (Pa·s/m) — standard value used in kPlan
RHO_C = 1.5e6

# ════════════════════════════════════════════════════════════════════════════════
# build_affine
# ════════════════════════════════════════════════════════════════════════════════

def build_affine(voxel_size_mm, origin_mm):
    """
    Build a 4×4 isotropic NIfTI affine from voxel size and world-space origin.

    Parameters
    ----------
    voxel_size_mm : float
        Isotropic voxel spacing in millimetres.
    origin_mm : array-like, shape (3,)
        World-space XYZ offset of the grid origin in millimetres.

    Returns
    -------
    affine : np.ndarray, shape (4, 4)
    """
    affine = np.zeros((4, 4), dtype=np.float64)
    affine[0, 0] = voxel_size_mm
    affine[1, 1] = voxel_size_mm
    affine[2, 2] = voxel_size_mm
    affine[0:3, 3] = origin_mm
    affine[3, 3] = 1.0
    return affine


# ════════════════════════════════════════════════════════════════════════════════
# match_shape
# ════════════════════════════════════════════════════════════════════════════════

def match_shape(data, target_shape):
    """
    Force an array into exactly target_shape by zero-padding or clipping.

    Needed because floating-point rounding during resampling can shift array
    dimensions by ±1 voxel relative to the simulation grid.
    """
    out = np.zeros(target_shape, dtype=data.dtype)
    s = [min(a, b) for a, b in zip(data.shape, target_shape)]
    out[:s[0], :s[1], :s[2]] = data[:s[0], :s[1], :s[2]]
    return out


# ════════════════════════════════════════════════════════════════════════════════
# FUNCTION 1a: load_pressure
# ════════════════════════════════════════════════════════════════════════════════

def load_pressure(h5_path, sonication_index):
    """
    Load and scale the pressure amplitude field for one sonication.

    The raw array is stored as uint16 in the H5 file. Real pressures (Pa) are
    recovered using linear scaling attributes stored on the dataset:

        pressure_pa = raw_uint16 * scale_slope + scale_intercept

    Parameters
    ----------
    h5_path : str
        Path to the kPlan RESULTS.h5 file.
    sonication_index : int
        1-based sonication number.

    Returns
    -------
    pressure_pa : np.ndarray, shape (X, Y, Z), float32
        Pressure amplitude in Pascals, transposed to (X, Y, Z) voxel order.
    affine_sim : np.ndarray, shape (4, 4)
        Affine mapping simulation voxel indices → mm world coordinates.
    grid_shape : tuple of int
        Shape of the simulation grid (X, Y, Z).
    metadata : dict
        Keys: target_position_mm, frequency_hz, target_pressure_pa,
              sptp_pa, sptp_masked_pa
    """
    with h5py.File(h5_path, "r") as f:

        # Pressure field
        p_key   = f"sonications/{sonication_index}/simulated_field/pressure_amplitude"
        dataset = f[p_key]

        scale_slope     = float(dataset.attrs["scale_slope"].ravel()[0])
        scale_intercept = float(dataset.attrs["scale_intercept"].ravel()[0])

        raw         = dataset[:]                                    # uint16, (Z,Y,X)
        pressure_pa = (raw.astype(np.float32) * scale_slope
                       + scale_intercept)                          # Pa, still (Z,Y,X)
        pressure_pa = np.transpose(pressure_pa)                    # → (X,Y,Z)

        # Spatial metadata
        mm_attrs       = f["medium_properties/medium_mask"]
        grid_spacing_m = float(mm_attrs.attrs["grid_spacing"].ravel()[0])
        grid_spacing_mm = grid_spacing_m * 1000.0

        origin_m   = f["settings/grid/domain_position"][:].ravel()[:3]
        origin_mm  = origin_m * 1000.0
        affine_sim = build_affine(grid_spacing_mm, origin_mm)

        # Sonication parameters
        pk = f"sonications/{sonication_index}/sonication_parameters"
        sf = f"sonications/{sonication_index}/simulated_field"

        target_pos_m  = f[f"{pk}/target_position"][:].ravel()[:3]
        frequency_hz  = float(f[f"{pk}/driving_frequency"][:].ravel()[0])
        target_press  = float(f[f"{pk}/target_pressure"][:].ravel()[0])
        sptp_pa       = float(f[f"{sf}/pressure_amplitude_sptp"][:].ravel()[0])
        sptp_masked   = float(f[f"{sf}/pressure_amplitude_sptp_masked"][:].ravel()[0])

        metadata = {
            "target_position_mm":  target_pos_m * 1000.0,
            "frequency_hz":        frequency_hz,
            "target_pressure_pa":  target_press,
            "sptp_pa":             sptp_pa,
            "sptp_masked_pa":      sptp_masked,
        }

    return pressure_pa, affine_sim, pressure_pa.shape, metadata

# ════════════════════════════════════════════════════════════════════════════════
# FUNCTION 1b: load_thermal
# ════════════════════════════════════════════════════════════════════════════════
def scalar_from_h5(f, full_key):
    return float(f[full_key][:].ravel()[0]) if full_key in f else None

def load_thermal(h5_path, sonication_index):
    """
    Load kPlan thermal outputs for one sonication.

    Returns
    -------
    temp_3d_degC : np.ndarray, shape (X, Y, Z), float32
        Peak temperature reached at each voxel over the sonication duration,
        correctly scaled to degrees Celsius.
    meta_thermal : dict
        Scalar summaries:
            temp_at_target_degC     — temperature at the planned target point
            temp_at_peak_degC       — temperature at the globally hottest voxel
            temp_sptp_degC          — spatial-peak temporal-peak temperature
            temp_sptp_masked_degC   — same, brain-masked
            cem43_at_target         — CEM43 dose at target point (min)
            cem43_sptp              — peak CEM43 in field (min)
            cem43_sptp_masked       — peak CEM43 in brain (min)
    """
    with h5py.File(h5_path, "r") as f:
        sf = f"sonications/{sonication_index}/simulated_field"

        temp_ds = f[f"{sf}/temperature_maximum"]

        # Apply linear scaling — identical pattern to load_pressure
        scale_slope     = float(temp_ds.attrs["scale_slope"].ravel()[0])
        scale_intercept = float(temp_ds.attrs["scale_intercept"].ravel()[0])

        raw_temp     = temp_ds[:].astype(np.float32)
        temp_3d_degC = np.transpose(
            np.squeeze(raw_temp * scale_slope + scale_intercept)
        )

        # Sanity check against kPlan's own data_range attribute
        expected_min, expected_max = temp_ds.attrs["data_range"].ravel()
        actual_min  = float(temp_3d_degC.min())
        actual_max  = float(temp_3d_degC.max())
        tol = 0.05  # °C
        if abs(actual_min - expected_min) > tol or abs(actual_max - expected_max) > tol:
            print(f"  [warn] Thermal scaling mismatch on sonication {sonication_index}: "
                  f"got [{actual_min:.3f}, {actual_max:.3f}] °C, "
                  f"expected [{expected_min:.3f}, {expected_max:.3f}] °C")
        else:
            print(f"  [thermal] Scaled OK — range {actual_min:.3f} to {actual_max:.3f} °C")

        meta_thermal = {
            "temp_at_target_degC":   scalar_from_h5(f, f"{sf}/temperature_at_target"),
            "temp_at_peak_degC":     scalar_from_h5(f, f"{sf}/temperature_at_peak"),
            "temp_sptp_degC":        scalar_from_h5(f, f"{sf}/temperature_maximum_sptp"),
            "temp_sptp_masked_degC": scalar_from_h5(f, f"{sf}/temperature_maximum_sptp_masked"),
            "cem43_at_target":       scalar_from_h5(f, f"{sf}/thermal_dose_at_target"),
            "cem43_sptp":            scalar_from_h5(f, f"{sf}/thermal_dose_sptp"),
            "cem43_sptp_masked":     scalar_from_h5(f, f"{sf}/thermal_dose_sptp_masked"),
        }

    return temp_3d_degC, meta_thermal



# ════════════════════════════════════════════════════════════════════════════════
# FUNCTION 2: normalize_and_threshold
# ════════════════════════════════════════════════════════════════════════════════

def normalize_and_threshold(pressure_pa, brain_mask):
    """
    Derive intensity field and compute -3 dB / -6 dB focal zone masks.

    Parameters
    ----------
    pressure_pa : np.ndarray, shape (X, Y, Z)
        Pressure amplitude in Pascals.
    brain_mask : np.ndarray, shape (X, Y, Z), uint8
        Binary mask — 1 = brain voxel, 0 = outside brain.

    Returns
    -------
    intensity_full : np.ndarray, float32
        Intensity field across the whole grid (W/cm²).
    intensity_brain : np.ndarray, float32
        Intensity field zeroed outside the brain mask.
    mask_6dB : np.ndarray, uint8
        Binary focal zone at -6 dB (brain-restricted).
    mask_3dB : np.ndarray, uint8
        Binary focal zone at -3 dB (brain-restricted).
    peak_int_overall : float
        Global peak intensity (W/cm²) — whole field.
    peak_int_brain : float
        Peak intensity within brain (W/cm²) — used as normalisation reference.
    """
    # Step 1: full intensity field
    intensity_full  = (pressure_pa ** 2) / (2 * RHO_C) / 1e4   # W/cm²

    # Step 2: brain-restricted intensity
    intensity_brain = intensity_full.copy()
    intensity_brain[brain_mask != 1] = 0.0

    vals_brain = intensity_brain[brain_mask == 1]
    if vals_brain.size == 0:
        raise ValueError("Brain mask has no voxels — check SimNIBS resampling.")
    if vals_brain.max() == 0:
        raise ValueError("Peak brain intensity is zero — check pressure scaling.")

    peak_int_overall = float(intensity_full.max())
    peak_int_brain   = float(vals_brain.max())

    # Step 3: thresholds — intensity domain, brain-peak reference
    thr_6dB = 0.25 * peak_int_brain   # -6 dB
    thr_3dB = 0.50 * peak_int_brain   # -3 dB

    # Step 4: focal zone masks
    mask_6dB = (intensity_brain > thr_6dB).astype(np.uint8)
    mask_3dB = (intensity_brain > thr_3dB).astype(np.uint8)

    return intensity_full, intensity_brain, mask_6dB, mask_3dB, peak_int_overall, peak_int_brain


# ════════════════════════════════════════════════════════════════════════════════
# FUNCTION 3: make_tissue_masks
# ════════════════════════════════════════════════════════════════════════════════

def make_tissue_masks(seg_path, affine_sim, grid_shape):
    """
    Load the SimNIBS segmentation and resample it into simulation grid space,
    producing one binary mask per tissue group defined in TISSUE_GROUPS.

    Parameters
    ----------
    seg_path : str
        Path to SimNIBS final_tissues.nii or .nii.gz.
    affine_sim : np.ndarray, shape (4, 4)
        Affine of the simulation grid (target resampling space).
    grid_shape : tuple of int
        Shape (X, Y, Z) of the simulation grid.

    Returns
    -------
    tissue_masks : dict[str, np.ndarray]
        Binary uint8 arrays keyed by tissue group name,
        resampled and shaped to grid_shape.
    """
    seg_img  = nib.load(seg_path)
    seg_data = np.squeeze(seg_img.get_fdata()).astype(np.int16)
    seg_img_sq = nib.Nifti1Image(seg_data, seg_img.affine)

    tissue_masks = {}
    for group_name, labels in TISSUE_GROUPS.items():
        # Build binary mask in segmentation space
        group_mask = np.zeros(seg_data.shape, dtype=np.uint8)
        for lbl in labels:
            group_mask[seg_data == lbl] = 1

        group_img  = nib.Nifti1Image(group_mask, seg_img.affine)
        resampled  = resample_img(
            group_img,
            target_affine=affine_sim,
            target_shape=grid_shape,
            interpolation="nearest",
        )
        resampled_data = match_shape(
            np.squeeze(resampled.get_fdata()).astype(np.uint8),
            grid_shape,
        )
        tissue_masks[group_name] = resampled_data

    return tissue_masks



# ════════════════════════════════════════════════════════════════════════════════
# FUNCTION 4: make_row  — unified metrics for any binary region mask
# ════════════════════════════════════════════════════════════════════════════════

def make_row(
    region_mask,
    target_name,
    target_path,
    tier_label,
    son_i,
    pressure_pa,
    intensity_full,
    mask_6dB,
    mask_3dB,
    peak_int_overall,
    peak_int_brain,
    focus_vox_6,
    focus_vox_3,
    focus_vol_6dB,
    focus_vol_3dB,
    voxel_vol_mm3,
    frequency_hz,
    affine_sim,
    temp_3d_degC=None,
    meta_thermal=None,
    is_eye=False,
):
    """
    Compute all CSV metrics for one binary region mask.

    Intensity metrics  — Intensity metrics
    Pressure metrics   — new additions: peak/mean kPa, overlap pressure
    Safety metrics     — MI + MI_Flag (eye limit vs brain limit)
    Spatial metric     — XYZ mm of peak pressure voxel within region

    Skull mean pressure is reported as N/A (not meaningful — ultrasound
    passes through only a fraction of skull voxels).

    Parameters
    ----------
    region_mask : np.ndarray, uint8
        Binary mask for the region (already in simulation grid space).
    target_name : str
    target_path : str
    tier_label  : str
    son_i       : int
    pressure_pa : np.ndarray   Full pressure field (Pa)
    intensity_full : np.ndarray  Full intensity field (W/cm²)
    mask_6dB, mask_3dB : np.ndarray  Focal zone masks (brain-restricted)
    peak_int_overall : float   Global peak intensity
    peak_int_brain   : float   Brain peak intensity (threshold reference)
    focus_vox_6, focus_vox_3 : int   Focal zone voxel counts
    focus_vol_6dB, focus_vol_3dB : float  Focal zone volumes (mm³)
    voxel_vol_mm3 : float
    frequency_hz  : float
    affine_sim    : np.ndarray  (4,4) — used to convert voxel index → mm
    temp_3d_degC  :(X,Y,Z) float32 — temperature rise field (°C)

    meta_thermal  :dict of kPlan scalar summaries (floats or None)
    is_eye        : bool

    Returns
    -------
    dict  — one CSV row, or None if the region has no voxels.
    """
    tgt_vox = int(region_mask.sum())
    if tgt_vox == 0:
        return None

    tgt_vol = round(tgt_vox * voxel_vol_mm3, 2)
    is_skull = "skull" in target_name.lower()

    # Intensity within region
    int_in_region  = intensity_full[region_mask == 1]
    wcm2_max_tgt   = round(float(int_in_region.max()), 4)
    wcm2_mean_tgt  = round(float(int_in_region.mean()), 4)

    # Pressure within region
    p_in_region  = pressure_pa[region_mask == 1]
    peak_pa_reg  = float(p_in_region.max())
    peak_kpa     = round(peak_pa_reg / 1000.0, 3)
    mean_kpa     = "N/A" if is_skull else round(float(p_in_region.mean()) / 1000.0, 3)

    # Focal zone overlaps
    ov6_vox = int((mask_6dB * region_mask).sum())
    ov3_vox = int((mask_3dB * region_mask).sum())

    cov_6dB_mm3 = round(ov6_vox * voxel_vol_mm3, 2)
    cov_3dB_mm3 = round(ov3_vox * voxel_vol_mm3, 2)
    cov_6dB_pct = round(ov6_vox / tgt_vox * 100.0, 2) if tgt_vox > 0 else float("nan")
    cov_3dB_pct = round(ov3_vox / tgt_vox * 100.0, 2) if tgt_vox > 0 else float("nan")

    on_6dB  = round(ov6_vox / focus_vox_6 * 100.0, 2) if focus_vox_6 > 0 else float("nan")
    off_6dB = round(100.0 - on_6dB, 2)                if focus_vox_6 > 0 else float("nan")
    on_3dB  = round(ov3_vox / focus_vox_3 * 100.0, 2) if focus_vox_3 > 0 else float("nan")
    off_3dB = round(100.0 - on_3dB, 2)                if focus_vox_3 > 0 else float("nan")

    #  Mean intensity in overlap zone
    if ov6_vox > 0:
        mean_int_ov6 = round(float(
            (intensity_full * mask_6dB * region_mask).sum() / ov6_vox
        ), 4)
    else:
        mean_int_ov6 = float("nan")

    if ov3_vox > 0:
        mean_int_ov3 = round(float(
            (intensity_full * mask_3dB * region_mask).sum() / ov3_vox
        ), 4)
    else:
        mean_int_ov3 = float("nan")

    #  Pressure in overlap zones
    ov6_mask = (mask_6dB == 1) & (region_mask == 1)
    ov3_mask = (mask_3dB == 1) & (region_mask == 1)

    if ov6_mask.any():
        p_ov6 = pressure_pa[ov6_mask]
        peak_pressure_ov6_kpa = round(float(p_ov6.max())  / 1000.0, 3)
        mean_pressure_ov6_kpa = round(float(p_ov6.mean()) / 1000.0, 3)
    else:
        peak_pressure_ov6_kpa = float("nan")
        mean_pressure_ov6_kpa = float("nan")

    if ov3_mask.any():
        p_ov3 = pressure_pa[ov3_mask]
        peak_pressure_ov3_kpa = round(float(p_ov3.max())  / 1000.0, 3)
        mean_pressure_ov3_kpa = round(float(p_ov3.mean()) / 1000.0, 3)
    else:
        peak_pressure_ov3_kpa = float("nan")
        mean_pressure_ov3_kpa = float("nan")

    # XYZ of peak pressure voxel within region (mm)
    masked_p = pressure_pa.copy()
    masked_p[region_mask == 0] = 0.0
    peak_idx    = np.unravel_index(masked_p.argmax(), masked_p.shape)
    peak_idx_h  = np.array([peak_idx[0], peak_idx[1], peak_idx[2], 1.0])
    peak_xyz_mm = (affine_sim @ peak_idx_h)[:3]
    peak_xyz_str = (f"({peak_xyz_mm[0]:.1f}, "
                    f"{peak_xyz_mm[1]:.1f}, "
                    f"{peak_xyz_mm[2]:.1f})")

    #  Mechanical Index
    freq_mhz     = frequency_hz / 1e6
    peak_neg_mpa = peak_pa_reg / 1e6       # peak amplitude used as proxy
    mi           = round(peak_neg_mpa / np.sqrt(freq_mhz), 3)
    mi_limit     = MI_LIMIT_EYES if is_eye else MI_LIMIT_BRAIN
    mi_flag      = "EXCEEDS LIMIT" if mi > mi_limit else "OK"

    # Thermal metrics
    # kPlan scalar summaries — same value on every row for this sonication
    temp_at_target_fmt    = round(meta_thermal["temp_at_target_degC"],   3) if meta_thermal["temp_at_target_degC"] is not None else None
    temp_at_peak_fmt      = round(meta_thermal["temp_at_peak_degC"],     3) if meta_thermal["temp_at_peak_degC"] is not None else None
    cem43_at_target_fmt   = round(meta_thermal["cem43_at_target"],       3) if meta_thermal["cem43_at_target"] is not None else None
    cem43_sptp_fmt        = round(meta_thermal["cem43_sptp"],            3) if meta_thermal["cem43_sptp"] is not None else None
    cem43_peak_masked_fmt = round(meta_thermal["cem43_sptp_masked"],     3) if meta_thermal["cem43_sptp_masked"] is not None else None

    # Per-region statistics from the 3-D temperature field
    t_reg            = temp_3d_degC[region_mask == 1]
    peak_temp_region = round(float(t_reg.max()),  3)
    mean_temp_region = round(float(t_reg.mean()), 3)

    ov6_temp_mask = (mask_6dB == 1) & (region_mask == 1)
    if ov6_temp_mask.any():
        t_ov6         = temp_3d_degC[ov6_temp_mask]
        peak_temp_ov6 = round(float(t_ov6.max()),  3)
        mean_temp_ov6 = round(float(t_ov6.mean()), 3)
    else:
        peak_temp_ov6 = float("nan")
        mean_temp_ov6 = float("nan")


   # Thermal safety flag based on peak temperature in region
    # < 40°C = OK,  40–43°C = CAUTION,  >43°C = DANGER

    if peak_temp_region >= 43.0:
        thermal_flag = "DANGER"
    elif peak_temp_region >= 40.0:
        thermal_flag = "CAUTION"
    else:
        thermal_flag = "OK"

    return {
        # Identifiers
        "Sonication":                           son_i,
        "TargetName":                           target_name,
        "TargetPath":                           target_path,
        "ReportingTier":                        tier_label,

        #  Global intensity reference (same for all rows in one sonication) ─
        "MaxIntensity_Overall_Wcm2":            round(peak_int_overall, 4),
        "MaxIntensity_Brain_Wcm2":              round(peak_int_brain, 4),

        # intensity columns
        "MaxIntensity_Target_Wcm2":             wcm2_max_tgt,
        "MeanIntensity_Target_Wcm2":            wcm2_mean_tgt,
        "TargetVolume_mm3":                     tgt_vol,
        "FocusVolume_6dB_mm3":                  focus_vol_6dB,
        "FocusVolume_3dB_mm3":                  focus_vol_3dB,
        "TargetCoverage_6dB_mm3":               cov_6dB_mm3,
        "TargetCoverage_6dB_percent":           cov_6dB_pct,
        "TargetCoverage_3dB_mm3":               cov_3dB_mm3,
        "TargetCoverage_3dB_percent":           cov_3dB_pct,
        "OnTarget_6dB_percent":                 on_6dB,
        "OffTarget_6dB_percent":                off_6dB,
        "OnTarget_3dB_percent":                 on_3dB,
        "OffTarget_3dB_percent":                off_3dB,
        "MeanIntensity_TargetOverlap_6dB_Wcm2": mean_int_ov6,
        "MeanIntensity_TargetOverlap_3dB_Wcm2": mean_int_ov3,

        # Pressure metrics
        "PeakPressure_Target_kPa":              peak_kpa,
        "MeanPressure_Target_kPa":              mean_kpa,
        "PeakPressure_Overlap_6dB_kPa":         peak_pressure_ov6_kpa,
        "MeanPressure_Overlap_6dB_kPa":         mean_pressure_ov6_kpa,
        "PeakPressure_Overlap_3dB_kPa":         peak_pressure_ov3_kpa,
        "MeanPressure_Overlap_3dB_kPa":         mean_pressure_ov3_kpa,

        # Safety metrics
        "MI":                                   mi,
        "MI_Flag":                              mi_flag,

        # Spatial metric
        "PeakPressure_XYZ_mm":                  peak_xyz_str,

        # Thermal — kPlan scalar summaries (global reference values)

        "Temp_AtTarget_degC":                   temp_at_target_fmt,
        "Temp_AtPeak_degC":                     temp_at_peak_fmt,
        "CEM43_AtTarget_min":                   cem43_at_target_fmt,
        "CEM43_SPTP_min":                       cem43_sptp_fmt,
        "CEM43_SPTPMasked_min":                 cem43_peak_masked_fmt,


    # Thermal — per-region from 3D temperature field
        "PeakTemp_Region_degC":      peak_temp_region,
        "MeanTemp_Region_degC":      mean_temp_region,
        "PeakTemp_Overlap_6dB_degC": peak_temp_ov6,
        "MeanTemp_Overlap_6dB_degC": mean_temp_ov6,

        # Thermal safety flag
        "ThermalSafetyFlag":         thermal_flag,
    }

# ════════════════════════════════════════════════════════════════════════════════
# FUNCTION 5: visualize_fus
# ════════════════════════════════════════════════════════════════════════════════

def get_background(t1_path, affine_sim, grid_shape, pressure_pa):
    """
    Returns a float32 (X, Y, Z) greyscale volume in simulation grid space.
    Uses T1 when available; falls back to the pressure field (in kPa).
    """
    if t1_path and os.path.exists(t1_path):
        t1_img = nib.load(t1_path)
        t1_res = resample_img(
            t1_img,
            target_affine=affine_sim,
            target_shape=grid_shape,
            interpolation="linear",
        )
        bg = match_shape(
            np.squeeze(t1_res.get_fdata()).astype(np.float32), grid_shape
        )
        label = "T1 MRI"
    else:
        bg = (pressure_pa / 1000.0).astype(np.float32)  # kPa
        label = "Pressure (kPa)"

    bg_min, bg_max = bg.min(), bg.max()
    if bg_max > bg_min:
        bg = (bg - bg_min) / (bg_max - bg_min)
    return bg, label

def best_slices(mask, pressure_pa):
    """
    Return (cx, cy, cz) — indices of the axial/coronal/sagittal slices that
    pass through the centre of mass of the mask.
    Falls back to the global pressure peak if mask is empty.
    """
    if mask.sum() > 0:
        coords = np.argwhere(mask > 0)
        cx, cy, cz = coords.mean(axis=0).astype(int)
    else:
        peak = np.unravel_index(pressure_pa.argmax(), pressure_pa.shape)
        cx, cy, cz = peak
    return cx, cy, cz


def draw_contour(ax, mask_2d, color, linewidth=1.5, label=None):
    """
    Overlays a matplotlib contour outline on an existing imshow axis.
    Silently skips if the mask slice is all zero.
    """
    if mask_2d.sum() == 0:
        return
    ax.contour(
        mask_2d,
        levels=[0.5],
        colors=[color],
        linewidths=linewidth,
        linestyles="solid",
        label=label if label else "",
    )


def panel_A_focus_target_overlap(
    sonication_index,
    pressure_pa,
    affine_sim,
    grid_shape,
    mask_6dB,
    mask_3dB,
    target_mask_filepaths,
    df_results,
    output_dir,
    t1_path=None,
):
    """
    3-plane (axial / coronal / sagittal) figure showing:
      - Greyscale anatomical background (T1 or pressure)
      - -6 dB focal zone — filled, semi-transparent cyan
      - -3 dB focal zone — filled, semi-transparent yellow
      - Each Tier2 nucleus — contour outline in a distinct colour
      - Quantitative overlay: coverage % from df_results

    One row per nucleus (+1 summary row at the bottom).
    """
    print(f"  [viz A] Focus ↔ target overlap …")

    bg, bg_label = get_background(t1_path, affine_sim, grid_shape, pressure_pa)

    nuc_masks = {}
    for mp in target_mask_filepaths:
        nm = os.path.basename(mp).replace(".nii.gz", "").replace(".nii", "")
        img = nib.load(mp)
        res = resample_img(
            img,
            target_affine=affine_sim,
            target_shape=grid_shape,
            interpolation="nearest",
        ).get_fdata()
        nuc_masks[nm] = match_shape(
            (np.squeeze(res) > 0).astype(np.uint8), grid_shape
        )

    if not nuc_masks:
        print("    [warn] No nucleus masks found — skipping Panel A.")
        return

    n_nuc = len(nuc_masks)
    fig, axes = plt.subplots(
        n_nuc, 3,
        figsize=(12, 4 * n_nuc),
        facecolor="#0a0a0a",
    )
    if n_nuc == 1:
        axes = axes[np.newaxis, :]

    nucleus_colors = plt.cm.Set1.colors

    for row_i, (nuc_name, nuc_mask) in enumerate(nuc_masks.items()):
        cx, cy, cz = best_slices(nuc_mask, pressure_pa)
        planes = [
            ("Axial",    bg[cx, :, :],   nuc_mask[cx, :, :],   mask_6dB[cx, :, :],   mask_3dB[cx, :, :]),
            ("Coronal",  bg[:, cy, :],   nuc_mask[:, cy, :],   mask_6dB[:, cy, :],   mask_3dB[:, cy, :]),
            ("Sagittal", bg[:, :, cz],   nuc_mask[:, :, cz],   mask_6dB[:, :, cz],   mask_3dB[:, :, cz]),
        ]

        df_row = df_results[
            (df_results["Sonication"] == sonication_index) &
            (df_results["TargetName"] == nuc_name)
        ]
        cov_6 = df_row["TargetCoverage_6dB_percent"].values
        cov_3 = df_row["TargetCoverage_3dB_percent"].values
        on_6  = df_row["OnTarget_6dB_percent"].values
        cov_6 = f"{cov_6[0]:.1f}%" if len(cov_6) else "N/A"
        cov_3 = f"{cov_3[0]:.1f}%" if len(cov_3) else "N/A"
        on_6  = f"{on_6[0]:.1f}%"  if len(on_6)  else "N/A"

        nuc_color = nucleus_colors[row_i % len(nucleus_colors)]

        for col_i, (plane_label, bg_sl, nuc_sl, foc6_sl, foc3_sl) in enumerate(planes):
            ax = axes[row_i, col_i]
            ax.set_facecolor("#0a0a0a")
            ax.imshow(bg_sl.T, cmap="gray", origin="lower",
                      vmin=0, vmax=1, aspect="auto")

            rgba_6 = np.zeros((*foc6_sl.T.shape, 4), dtype=np.float32)
            rgba_6[foc6_sl.T > 0] = [0.0, 0.9, 0.9, 0.30]
            ax.imshow(rgba_6, origin="lower", aspect="auto")

            rgba_3 = np.zeros((*foc3_sl.T.shape, 4), dtype=np.float32)
            rgba_3[foc3_sl.T > 0] = [1.0, 0.9, 0.1, 0.40]
            ax.imshow(rgba_3, origin="lower", aspect="auto")

            draw_contour(ax, nuc_sl.T, color=mcolors.to_hex(nuc_color), linewidth=2.0)

            ax.set_xticks([])
            ax.set_yticks([])
            if col_i == 0:
                ax.set_ylabel(nuc_name, color="white", fontsize=8,
                              rotation=90, va="center")
            if row_i == 0:
                ax.set_title(plane_label, color="#aaaaaa", fontsize=9)

        axes[row_i, 2].text(
            1.02, 0.5,
            f"Cov -6dB: {cov_6}\nCov -3dB: {cov_3}\nOn-target -6dB: {on_6}",
            transform=axes[row_i, 2].transAxes,
            color="white", fontsize=7.5, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="#222", ec="none", alpha=0.8),
        )

    handles = [
        mpatches.Patch(color=(0.0, 0.9, 0.9, 0.55), label="-6 dB focal zone"),
        mpatches.Patch(color=(1.0, 0.9, 0.1, 0.65), label="-3 dB focal zone"),
    ]
    for row_i, nm in enumerate(nuc_masks.keys()):
        handles.append(mpatches.Patch(
            color=nucleus_colors[row_i % len(nucleus_colors)],
            label=nm,
        ))
    fig.legend(
        handles=handles, loc="lower center", ncol=4,
        facecolor="#1a1a1a", edgecolor="none",
        labelcolor="white", fontsize=8,
    )
    fig.suptitle(
        f"Sonication {sonication_index} — Focus ↔ Target Overlap\n"
        f"Background: {bg_label}",
        color="white", fontsize=12, y=1.01,
    )
    fig.tight_layout()

    son_tag  = f"son{sonication_index}"
    out_path = os.path.join(output_dir, f"{son_tag}_A_focus_target_overlap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    → {out_path}")


def panel_B_hot_regions_on_T1(
    sonication_index,
    pressure_pa,
    affine_sim,
    grid_shape,
    mask_6dB,
    intensity_full,
    all_region_masks,
    df_results,
    output_dir,
    t1_path=None,
    intensity_threshold_wcm2=3.0,
    top_n_regions=5,
):
    """
    For each region whose MaxIntensity > threshold:
      Row layout:  [Axial] [Coronal] [Sagittal] | metrics text
      Underlay   : T1 (or pressure)
      Overlay    : intensity_full as hot colourmap (alpha by intensity)
                   + -6dB contour (cyan dashed)
                   + region outline (coloured)
    """
    print(f"  [viz B] Hot regions (>{intensity_threshold_wcm2} W/cm²) on T1 …")

    bg, bg_label = get_background(t1_path, affine_sim, grid_shape, pressure_pa)

    df_son = df_results[df_results["Sonication"] == sonication_index].copy()
    df_hot = (
        df_son[df_son["MaxIntensity_Target_Wcm2"] > intensity_threshold_wcm2]
        .sort_values("MaxIntensity_Target_Wcm2", ascending=False)
        .head(top_n_regions)
    )

    if df_hot.empty:
        print(f"    [warn] No regions exceed {intensity_threshold_wcm2} W/cm² "
              f"— skipping Panel B. Lower intensity_threshold_wcm2 if needed.")
        return

    hot_names = df_hot["TargetName"].tolist()
    print(f"    Hot regions: {hot_names}")

    int_norm = intensity_full / (intensity_full.max() + 1e-12)
    hot_cmap = LinearSegmentedColormap.from_list(
        "hot_pressure", ["#000000", "#ff4500", "#ffff00", "#ffffff"]
    )

    n_hot = len(hot_names)
    fig, axes = plt.subplots(
        n_hot, 3, figsize=(13, 4.2 * n_hot), facecolor="#0a0a0a"
    )
    if n_hot == 1:
        axes = axes[np.newaxis, :]

    region_colors = plt.cm.tab10.colors

    for row_i, rname in enumerate(hot_names):
        rmask = all_region_masks.get(rname)
        if rmask is None:
            print(f"    [warn] Mask for '{rname}' not in all_region_masks — skipped.")
            continue

        cx, cy, cz = best_slices(rmask, pressure_pa)

        planes = [
            ("Axial",    bg[cx, :, :],  rmask[cx, :, :],  mask_6dB[cx, :, :],  int_norm[cx, :, :]),
            ("Coronal",  bg[:, cy, :],  rmask[:, cy, :],  mask_6dB[:, cy, :],  int_norm[:, cy, :]),
            ("Sagittal", bg[:, :, cz],  rmask[:, :, cz],  mask_6dB[:, :, cz],  int_norm[:, :, cz]),
        ]

        df_r     = df_hot[df_hot["TargetName"] == rname].iloc[0]
        mi_val   = df_r.get("MI", "N/A")
        mi_flag  = df_r.get("MI_Flag", "")
        peak_kpa = df_r.get("PeakPressure_Target_kPa", "N/A")
        cov_6    = df_r.get("TargetCoverage_6dB_percent", "N/A")
        max_int  = df_r.get("MaxIntensity_Target_Wcm2", "N/A")
        tier     = df_r.get("ReportingTier", "")

        rcolor   = region_colors[row_i % len(region_colors)]
        mi_color = "#ff4040" if str(mi_flag) == "EXCEEDS LIMIT" else "#44ff88"

        for col_i, (plane_label, bg_sl, reg_sl, foc6_sl, int_sl) in enumerate(planes):
            ax = axes[row_i, col_i]
            ax.set_facecolor("#0a0a0a")

            ax.imshow(bg_sl.T, cmap="gray", origin="lower",
                      vmin=0, vmax=1, aspect="auto")

            rgba_int = hot_cmap(int_sl.T)
            rgba_int[..., 3] = int_sl.T ** 0.6
            ax.imshow(rgba_int, origin="lower", aspect="auto")

            if foc6_sl.sum() > 0:
                ax.contour(foc6_sl.T, levels=[0.5],
                           colors=["#00ffff"], linewidths=1.2,
                           linestyles="dashed")

            draw_contour(ax, reg_sl.T, color=mcolors.to_hex(rcolor), linewidth=2.2)

            ax.set_xticks([])
            ax.set_yticks([])
            if col_i == 0:
                ax.set_ylabel(rname[:22], color="white", fontsize=8,
                              rotation=90, va="center")
            if row_i == 0:
                ax.set_title(plane_label, color="#aaaaaa", fontsize=9)

        flag_str  = f"  ⚠ {mi_flag}" if str(mi_flag) == "EXCEEDS LIMIT" else ""
        info_text = (
            f"Tier: {tier.split('_')[-1]}\n"
            f"MaxInt: {max_int:.3f} W/cm²\n"
            f"PeakP:  {peak_kpa} kPa\n"
            f"Cov-6dB: {cov_6:.1f}%\n"
            f"MI: {mi_val:.3f}{flag_str}"
        )
        axes[row_i, 2].text(
            1.02, 0.5, info_text,
            transform=axes[row_i, 2].transAxes,
            color=mi_color, fontsize=7.5, va="center", ha="left",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#181818",
                      ec=mi_color, alpha=0.9, linewidth=0.8),
        )

    sm = plt.cm.ScalarMappable(
        cmap=hot_cmap,
        norm=mcolors.Normalize(vmin=0, vmax=float(intensity_full.max())),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:, 2], shrink=0.6, pad=0.18,
                        aspect=30, location="right")
    cbar.set_label("Intensity (W/cm²)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    legend_handles = [
        mpatches.Patch(color="#00ffff", label="-6 dB focal zone (dashed)"),
    ]
    for ri, rn in enumerate(hot_names):
        legend_handles.append(
            mpatches.Patch(color=region_colors[ri % len(region_colors)], label=rn[:30])
        )
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               facecolor="#1a1a1a", edgecolor="none",
               labelcolor="white", fontsize=7.5)

    fig.suptitle(
        f"Sonication {sonication_index} — Top {n_hot} Hot Regions "
        f"(>{intensity_threshold_wcm2} W/cm²) on {bg_label}",
        color="white", fontsize=12, y=1.01,
    )
    fig.tight_layout()

    son_tag  = f"son{sonication_index}"
    out_path = os.path.join(output_dir, f"{son_tag}_B_hot_regions_on_T1.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    → {out_path}")


def panel_C_MI_safety(sonication_index,df_results,output_dir):
    """
    All regions sorted by MI (descending) within each tier.
    Red bars = MI_Flag == 'EXCEEDS LIMIT'.
    Vertical dashed lines at MI_LIMIT_BRAIN and MI_LIMIT_EYES.
    """
    print(f"  [viz C] MI safety chart …")

    df_son = df_results[df_results["Sonication"] == sonication_index].copy()
    if df_son.empty:
        print("    [warn] No results for this sonication — skipping Panel C.")
        return

    df_son = df_son.sort_values(
        ["ReportingTier", "MI"], ascending=[True, False]
    ).reset_index(drop=True)

    tier_colors = {
        "Tier1_Tissue":            "#5588bb",
        "Tier2_ParticipantNuclei": "#55bb88",
        "Tier3_cortical_subcortical_atlas":     "#bb8855",
    }
    bar_colors = []
    for _, row in df_son.iterrows():
        if str(row["MI_Flag"]) == "EXCEEDS LIMIT":
            bar_colors.append("#ff3030")
        else:
            bar_colors.append(tier_colors.get(row["ReportingTier"], "#888888"))

    n     = len(df_son)
    fig_h = max(5, n * 0.28)
    fig, ax = plt.subplots(figsize=(10, fig_h), facecolor="#0a0a0a")
    ax.set_facecolor("#111111")

    y_pos = np.arange(n)
    ax.barh(y_pos, df_son["MI"].values,
            color=bar_colors, height=0.7, edgecolor="none")

    ax.axvline(MI_LIMIT_BRAIN, color="#ffcc00", linewidth=1.5,
               linestyle="--", label=f"Brain limit ({MI_LIMIT_BRAIN})")
    ax.axvline(MI_LIMIT_EYES,  color="#ff6600", linewidth=1.5,
               linestyle=":",  label=f"Eye limit ({MI_LIMIT_EYES})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [n[:35] for n in df_son["TargetName"].values],
        color="white", fontsize=6.5,
    )
    ax.tick_params(axis="x", colors="white", labelsize=8)
    ax.set_xlabel("Mechanical Index (MI)", color="white", fontsize=9)
    ax.set_title(
        f"Sonication {sonication_index} — MI Safety Summary\n"
        f"({n} regions across all tiers)",
        color="white", fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333")
    ax.spines["bottom"].set_color("#333")

    tier_handles = [
        mpatches.Patch(color=c, label=t.split("_", 1)[1])
        for t, c in tier_colors.items()
    ]
    tier_handles += [
        mpatches.Patch(color="#ff3030", label="EXCEEDS LIMIT"),
        plt.Line2D([0], [0], color="#ffcc00", lw=1.5, ls="--",
                   label=f"Brain MI limit ({MI_LIMIT_BRAIN})"),
        plt.Line2D([0], [0], color="#ff6600", lw=1.5, ls=":",
                   label=f"Eye MI limit ({MI_LIMIT_EYES})"),
    ]
    ax.legend(handles=tier_handles, loc="lower right",
              facecolor="#1a1a1a", edgecolor="none",
              labelcolor="white", fontsize=7.5)

    fig.tight_layout()

    son_tag  = f"son{sonication_index}"
    out_path = os.path.join(output_dir, f"{son_tag}_C_MI_safety.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"     {out_path}")


def panel_D_pressure_crosssections(
    sonication_index,
    pressure_pa,
    affine_sim,
    grid_shape,
    mask_6dB,
    mask_3dB,
    tissue_masks,
    output_dir,
    t1_path=None,
):
    """
    3-plane cross-section through the global pressure peak voxel.
    Tissue outlines, focal zone isocontours, and crosshair all overlaid.
    """
    print(f"  [viz D] Pressure cross-sections …")

    bg, bg_label = get_background(t1_path, affine_sim, grid_shape, pressure_pa)

    peak_idx = np.unravel_index(pressure_pa.argmax(), pressure_pa.shape)
    px, py, pz = peak_idx
    peak_mm = (affine_sim @ np.array([px, py, pz, 1.0]))[:3]

    pressure_kpa = pressure_pa / 1000.0
    p_max = float(pressure_kpa.max())

    hot_cmap = LinearSegmentedColormap.from_list(
        "fus_hot", ["#000033", "#0050ff", "#ff4500", "#ffff00", "#ffffff"]
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor="#0a0a0a")
    plane_defs = [
        ("Axial",    bg[px, :, :],  pressure_kpa[px, :, :],
         mask_6dB[px, :, :],  mask_3dB[px, :, :],
         {k: v[px, :, :] for k, v in tissue_masks.items()},
         py, pz),
        ("Coronal",  bg[:, py, :],  pressure_kpa[:, py, :],
         mask_6dB[:, py, :],  mask_3dB[:, py, :],
         {k: v[:, py, :] for k, v in tissue_masks.items()},
         px, pz),
        ("Sagittal", bg[:, :, pz],  pressure_kpa[:, :, pz],
         mask_6dB[:, :, pz],  mask_3dB[:, :, pz],
         {k: v[:, :, pz] for k, v in tissue_masks.items()},
         px, py),
    ]

    tissue_outline_colors = {
        "brain": "#88aaff",
        "skull": "#ff8844",
        "scalp": "#cc66cc",
        "eyes":  "#ff4444",
    }

    for col_i, (plane_label, bg_sl, p_sl, f6_sl, f3_sl,
                tis_sls, cross_x, cross_y) in enumerate(plane_defs):
        ax = axes[col_i]
        ax.set_facecolor("#0a0a0a")

        ax.imshow(bg_sl.T, cmap="gray", origin="lower",
                  vmin=0, vmax=1, aspect="auto")

        rgba_p = hot_cmap(p_sl.T / (p_max + 1e-12))
        rgba_p[..., 3] = (p_sl.T / (p_max + 1e-12)) ** 0.5
        ax.imshow(rgba_p, origin="lower", aspect="auto")

        if f6_sl.sum() > 0:
            ax.contour(f6_sl.T, levels=[0.5], colors=["#00ffff"],
                       linewidths=1.5, linestyles="solid")
        if f3_sl.sum() > 0:
            ax.contour(f3_sl.T, levels=[0.5], colors=["#ffee00"],
                       linewidths=1.0, linestyles="solid")

        for tname, tsl in tis_sls.items():
            tc = tissue_outline_colors.get(tname, "#ffffff")
            draw_contour(ax, tsl.T, color=tc, linewidth=1.0)

        ax.axvline(cross_x, color="#ffffff", linewidth=0.7,
                   linestyle="--", alpha=0.5)
        ax.axhline(cross_y, color="#ffffff", linewidth=0.7,
                   linestyle="--", alpha=0.5)
        ax.plot(cross_x, cross_y, "w+", markersize=10, markeredgewidth=1.5)

        ax.set_title(
            f"{plane_label}  (slice through peak @ {plane_label[0]}-axis)",
            color="#aaaaaa", fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    sm = plt.cm.ScalarMappable(
        cmap=hot_cmap, norm=mcolors.Normalize(vmin=0, vmax=p_max)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.02, aspect=30)
    cbar.set_label("Pressure (kPa)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    legend_handles = [
        plt.Line2D([0], [0], color="#00ffff", lw=1.5, label="-6 dB focal zone"),
        plt.Line2D([0], [0], color="#ffee00", lw=1.0, label="-3 dB focal zone"),
    ]
    for tname, tc in tissue_outline_colors.items():
        legend_handles.append(
            plt.Line2D([0], [0], color=tc, lw=1.0, label=tname.capitalize())
        )
    legend_handles.append(
        plt.Line2D([0], [0], color="white", lw=0.7, ls="--",
                   marker="+", markersize=8, label="Pressure peak")
    )
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               facecolor="#1a1a1a", edgecolor="none",
               labelcolor="white", fontsize=8)

    fig.suptitle(
        f"Sonication {sonication_index} — Pressure Field Cross-Sections\n"
        f"Peak @ ({peak_mm[0]:.1f}, {peak_mm[1]:.1f}, {peak_mm[2]:.1f}) mm  |  "
        f"Background: {bg_label}",
        color="white", fontsize=11, y=1.02,
    )
    fig.tight_layout()

    son_tag  = f"son{sonication_index}"
    out_path = os.path.join(output_dir, f"{son_tag}_D_pressure_crosssections.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    → {out_path}")


def visualize_fus(
    sonication_index,
    pressure_pa,
    affine_sim,
    grid_shape,
    mask_6dB,
    mask_3dB,
    intensity_full,
    tissue_masks,
    target_mask_filepaths,
    all_region_masks,
    df_results,
    output_dir,
    t1_path=None,
    intensity_threshold_wcm2=3.0,
    top_n_regions=5,
):
    """
    Four-panel visualization suite for one sonication.

      Panel A - Focus ↔ Target overlap (all Tier2 nuclei, 3-plane)
      Panel B - Top-N hot regions on T1 (any region > threshold W/cm²)
      Panel C - MI safety bar chart — all tiers, flagged in red
      Panel D - Pressure field cross-sections (3-plane through peak)
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[visualize_fus] Sonication {sonication_index} — generating 4 panels …")

    panel_A_focus_target_overlap(
        sonication_index=sonication_index,
        pressure_pa=pressure_pa,
        affine_sim=affine_sim,
        grid_shape=grid_shape,
        mask_6dB=mask_6dB,
        mask_3dB=mask_3dB,
        target_mask_filepaths=target_mask_filepaths,
        df_results=df_results,
        output_dir=output_dir,
        t1_path=t1_path,
    )

    panel_B_hot_regions_on_T1(
        sonication_index=sonication_index,
        pressure_pa=pressure_pa,
        affine_sim=affine_sim,
        grid_shape=grid_shape,
        mask_6dB=mask_6dB,
        intensity_full=intensity_full,
        all_region_masks=all_region_masks,
        df_results=df_results,
        output_dir=output_dir,
        t1_path=t1_path,
        intensity_threshold_wcm2=intensity_threshold_wcm2,
        top_n_regions=top_n_regions,
    )

    panel_C_MI_safety(
        sonication_index=sonication_index,
        df_results=df_results,
        output_dir=output_dir,
    )

    panel_D_pressure_crosssections(
        sonication_index=sonication_index,
        pressure_pa=pressure_pa,
        affine_sim=affine_sim,
        grid_shape=grid_shape,
        mask_6dB=mask_6dB,
        mask_3dB=mask_3dB,
        tissue_masks=tissue_masks,
        output_dir=output_dir,
        t1_path=t1_path,
    )

    print(f"[visualize_fus] Done.")

# ════════════════════════════════════════════════════════════════════════════════
# FUNCTION 6: run_pipeline
# ════════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    """
    Main pipeline — loops over all H5 files and all sonications,
    applies three reporting tiers, and saves one CSV per H5 file.

    Step order per sonication
    ─────────────────────────
    1. load_pressure()              → pressure_pa, affine_sim, grid_shape, meta
    2. make_tissue_masks()          → tissue_masks (MUST precede normalize so
                                       brain mask is available for thresholding)
    3. normalize_and_threshold()    → intensity fields + focal zone masks

    4. make_row() × all regions     → one CSV row per region
    5. load_thermal()               → temp_3d_degC, cem43_3d, meta_thermal
    6. Save CSV, print summary
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Discover participant nucleus masks
    mask_niis   = sorted(glob.glob(os.path.join(TARGET_MASK_FOLDER, "*.nii")))
    mask_niigzs = sorted(glob.glob(os.path.join(TARGET_MASK_FOLDER, "*.nii.gz")))
    target_mask_filepaths = mask_niis + mask_niigzs
    if not target_mask_filepaths:
        raise FileNotFoundError(f"No NIfTI masks found in: {TARGET_MASK_FOLDER}")
    print(f"Found {len(target_mask_filepaths)} participant nucleus masks.")

    # Loop over H5 files
    for h5_path in H5_FILES:
        if not os.path.exists(h5_path):
            print(f"[SKIP] File not found: {h5_path}")
            continue

        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(h5_path)}")
        print(f"{'='*70}")

        all_rows = []

        with h5py.File(h5_path, "r") as f:
            n_sonications = len([k for k in f["sonications"].keys()
                                  if k.isdigit()])
        print(f"  Found {n_sonications} sonications.")

        # Load atlas once per H5 (resampling is slow)
        # We need grid info first, so load from sonication 1
        _, affine_sim_ref, grid_shape_ref, _ = load_pressure(h5_path, 1)
       # Using individualized cortical and subcortical masks
        atlas_masks = {}
        atlas_paths = sorted(glob.glob(os.path.join(INDIVIDUALIZED_CORTICAL_SUBCORTICAL_MASKS, "*.nii.gz")) +
                        glob.glob(os.path.join(INDIVIDUALIZED_CORTICAL_SUBCORTICAL_MASKS, "*.nii")))
        for p in atlas_paths:
            mname = os.path.basename(p).replace(".nii.gz", "").replace(".nii", "")
            img = nib.load(p)
            res = resample_img(
                img,
                target_affine=affine_sim_ref,
                target_shape=grid_shape_ref,
                interpolation="nearest",
            ).get_fdata()
            m = match_shape((np.squeeze(res) > 0).astype(np.uint8), grid_shape_ref)
            if m.sum() > 0:
                atlas_masks[mname] = m
        print(f"  [Atlas] {len(atlas_masks)} individualized regions loaded")
        # Loop over sonications
        for son_i in range(1, n_sonications + 1):
            print(f"\n  --- Sonication {son_i}/{n_sonications} ---")

            # Step 1: Load pressure
            pressure_pa, affine_sim, grid_shape, meta = load_pressure(
                h5_path, son_i
            )
            voxel_size_mm = float(affine_sim[0, 0])
            voxel_vol_mm3 = voxel_size_mm ** 3
            freq_hz       = meta["frequency_hz"]

            print(f"     Grid: {grid_shape}  |  Voxel: {voxel_size_mm:.3f} mm  "
                  f"|  Freq: {freq_hz/1e3:.0f} kHz")
            print(f"     Target (mm): {np.round(meta['target_position_mm'], 1)}")

            # Step 2: Tissue masks
            print(f"     Step 2: Building tissue masks...", end=" ", flush=True)
            tissue_masks = make_tissue_masks(SIMNIBS_PATH, affine_sim, grid_shape)
            brain_mask   = tissue_masks.get(
                "brain", np.zeros(grid_shape, dtype=np.uint8)
            )
            print("done")

            # Step 3: Normalize & threshold
            print(f"     Step 3: Computing focal zones...", end=" ", flush=True)
            (intensity_full, intensity_brain,
             mask_6dB, mask_3dB,
             peak_int_overall, peak_int_brain) = normalize_and_threshold(
                pressure_pa, brain_mask
            )

            focus_vox_6  = int(mask_6dB.sum())
            focus_vox_3  = int(mask_3dB.sum())
            focus_vol_6dB = round(focus_vox_6 * voxel_vol_mm3, 2)
            focus_vol_3dB = round(focus_vox_3 * voxel_vol_mm3, 2)
            print(f"done  "
                  f"(-6dB: {focus_vol_6dB} mm³  |  -3dB: {focus_vol_3dB} mm³)")
            # Step 4: Load thermal data
            print(f"     Step 4: Loading thermal data...", end=" ", flush=True)
            temp_3d_degC, meta_thermal = load_thermal(h5_path, son_i)

            print("done")

            # Shared kwargs for every make_row call
            row_kwargs = dict(
                son_i            = son_i,
                pressure_pa      = pressure_pa,
                intensity_full   = intensity_full,
                mask_6dB         = mask_6dB,
                mask_3dB         = mask_3dB,
                peak_int_overall = peak_int_overall,
                peak_int_brain   = peak_int_brain,
                focus_vox_6      = focus_vox_6,
                focus_vox_3      = focus_vox_3,
                focus_vol_6dB    = focus_vol_6dB,
                focus_vol_3dB    = focus_vol_3dB,
                voxel_vol_mm3    = voxel_vol_mm3,
                frequency_hz     = freq_hz,
                affine_sim       = affine_sim,
                temp_3d_degC     = temp_3d_degC,
                meta_thermal     = meta_thermal,
            )

            # Tissue safety
            print(f"     Tier 1: tissue masks...", end=" ", flush=True)
            for tname, tmask in tissue_masks.items():
                row = make_row(
                    region_mask = tmask,
                    target_name = tname,
                    target_path = f"SimNIBS_tissue/{tname}",
                    tier_label  = "Tier1_Tissue",
                    is_eye      = (tname == "eyes"),
                    **row_kwargs,
                )
                if row:
                    all_rows.append(row)
            print("done")

            # Participant nucleus masks
            print(f"     Tier 2: participant nuclei...", end=" ", flush=True)
            for mask_path in target_mask_filepaths:
                mname   = (os.path.basename(mask_path)
                           .replace(".nii.gz", "").replace(".nii", ""))
                tgt_img = nib.load(mask_path)
                tgt_res = resample_img(
                    tgt_img,
                    target_affine = affine_sim,
                    target_shape  = grid_shape,
                    interpolation = "nearest",
                ).get_fdata()
                tgt_bin = match_shape(
                    (np.squeeze(tgt_res) > 0).astype(np.uint8), grid_shape
                )
                row = make_row(
                    region_mask = tgt_bin,
                    target_name = mname,
                    target_path = mask_path,
                    tier_label  = "Tier2_ParticipantNuclei",
                    **row_kwargs,
                )
                if row:
                    all_rows.append(row)
            print("done")

            # Individualized cortical and subcortical regions
            if atlas_masks:
                print(f"     Tier 3: Individualized cortical and subcortical masks({len(atlas_masks)} regions)...",
                      end=" ", flush=True)
                for rname, rmask in atlas_masks.items():
                    row = make_row(
                        region_mask = rmask,
                        target_name = rname,
                        target_path = f"IndividualizedAtlas/{rname}",
                        tier_label  = "Tier3_cortical_subcortical_atlas",
                        **row_kwargs,
                    )
                    if row:
                        all_rows.append(row)
                print("done")
            # Visualize this sonication
            # Build nucleus masks dict in sim-space (needed by Panel A)
            nuc_masks_for_viz = {}
            for mask_path in target_mask_filepaths:
                mname = (os.path.basename(mask_path)
                         .replace(".nii.gz", "").replace(".nii", ""))
                tgt_img = nib.load(mask_path)
                tgt_res = resample_img(
                    tgt_img,
                    target_affine=affine_sim,
                    target_shape=grid_shape,
                    interpolation="nearest",
                ).get_fdata()
                nuc_masks_for_viz[mname] = match_shape(
                    (np.squeeze(tgt_res) > 0).astype(np.uint8), grid_shape
                )

            # Combined dict for Panel B (all tiers searchable)
            all_region_masks = {**tissue_masks, **nuc_masks_for_viz, **atlas_masks}

            # Partial DataFrame covering all rows collected so far
            df_so_far = pd.DataFrame(all_rows)

            visualize_fus(
                sonication_index      = son_i,
                pressure_pa           = pressure_pa,
                affine_sim            = affine_sim,
                grid_shape            = grid_shape,
                mask_6dB              = mask_6dB,
                mask_3dB              = mask_3dB,
                intensity_full        = intensity_full,
                tissue_masks          = tissue_masks,
                target_mask_filepaths = target_mask_filepaths,
                all_region_masks      = all_region_masks,
                df_results            = df_so_far,
                output_dir            = OUTPUT_DIR,
                t1_path               = T1_PATH,
            )

        # Save CSV
        if not all_rows:
            print("  [warn] No rows generated — check masks and H5 data.")
            continue

        df = pd.DataFrame(all_rows)

        # Column order
        ordered_cols = [
            "Sonication", "TargetName", "TargetPath", "ReportingTier",
            # global reference
            "MaxIntensity_Overall_Wcm2", "MaxIntensity_Brain_Wcm2",
            "MaxIntensity_Target_Wcm2", "MeanIntensity_Target_Wcm2",
            "TargetVolume_mm3",
            "FocusVolume_6dB_mm3", "FocusVolume_3dB_mm3",
            "TargetCoverage_6dB_mm3", "TargetCoverage_6dB_percent",
            "TargetCoverage_3dB_mm3", "TargetCoverage_3dB_percent",
            "OnTarget_6dB_percent",  "OffTarget_6dB_percent",
            "OnTarget_3dB_percent",  "OffTarget_3dB_percent",
            "MeanIntensity_TargetOverlap_6dB_Wcm2",
            "MeanIntensity_TargetOverlap_3dB_Wcm2",
            "PeakPressure_Target_kPa", "MeanPressure_Target_kPa",
            "PeakPressure_Overlap_6dB_kPa", "MeanPressure_Overlap_6dB_kPa",
            "PeakPressure_Overlap_3dB_kPa", "MeanPressure_Overlap_3dB_kPa",
            "MI", "MI_Flag",
            "PeakPressure_XYZ_mm",
            "Temp_AtTarget_degC", "Temp_AtPeak_degC",
            "CEM43_AtTarget_min", "CEM43_SPTP_min", "CEM43_SPTPMasked_min",
            "PeakTemp_Region_degC", "MeanTemp_Region_degC",
            "PeakTemp_Overlap_6dB_degC", "MeanTemp_Overlap_6dB_degC",
            "ThermalSafetyFlag",
        ]
        present = [c for c in ordered_cols if c in df.columns]
        df = df[present]

        # Sort: sonication → tier → descending peak intensity
        df = df.sort_values(
            ["Sonication", "ReportingTier", "MaxIntensity_Target_Wcm2"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
        df.insert(0, 'participant_id', PARTICIPANT_ID)
        df.insert(1, 'intended_target', INTENDED_TARGET)

        csv_name = f"{PARTICIPANT_ID}_{INTENDED_TARGET}_ANALYSIS.csv"
        csv_path = os.path.join(OUTPUT_DIR, csv_name)
        df.to_csv(csv_path, index=False)

        print(f"\n  [csv] Saved → {csv_path}")
        print(f"        {len(df)} rows  |  {df['Sonication'].nunique()} sonications")
        tier_counts = {
            t: int((df["ReportingTier"] == t).sum() // df["Sonication"].nunique())
            for t in df["ReportingTier"].unique()
        }
        print(f"        Regions per sonication: {tier_counts}")

        #  Console summary: key targets for Sonication 1
        print("\n  === KEY TARGETS — Sonication 1 ===")
        key_terms = ["thalamus", "anterior", "mediodorsal", "ant", "_md",
                     "ventricle", "eyes", "skull"]
        mask_key  = df["TargetName"].str.lower().str.contains(
            "|".join(key_terms), na=False
        )
        s1_key = df[(df["Sonication"] == 1) & mask_key][[
            "TargetName", "ReportingTier",
            "MaxIntensity_Target_Wcm2",
            "TargetCoverage_6dB_percent", "OnTarget_6dB_percent",
            "PeakPressure_Target_kPa",
            "MI", "MI_Flag",
            "PeakPressure_XYZ_mm",
        ]]
        if s1_key.empty:
            print("    (no key targets matched — check TargetName strings)")
        else:
            print(s1_key.to_string(index=False))

    print(f"\n{'='*70}")
    print(f"All done. Outputs saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")



# Entry point
if __name__ == "__main__":
    run_pipeline()

"""| Visual element           | Meaning                  |
| ------------------------ | ------------------------ |
| **Filled cyan blob**     | Beam spread (−6 dB)      |
| **Filled yellow blob**   | Beam core (−3 dB)        |
| **Thin colored outline** | Actual anatomical target |

"""