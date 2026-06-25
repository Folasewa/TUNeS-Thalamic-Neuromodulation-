#!/usr/bin/env python3
"""
================================================================================
Tuned:) Report Generator
================================================================================

A decision-support tool for choosing the final transducer placement in
transcranial ultrasound stimulation (TUS / tFUS) studies. Post-processes
kPlan acoustic simulation outputs (.h5) and produces a single, self-contained
HTML report per plan, designed so an operator can compare candidate placements
(sonications) and judge efficacy + safety at a glance.

Inputs
──────
  • kPlan RESULTS.h5                              (required)
  • SimNIBS final_tissues.nii.gz                 (optional — detailed tissues;
                                                  falls back to kPlan medium mask)
  • Primary target mask .nii/.nii.gz             (optional)
  • region_mask_folder/ of nuclei + HO atlases   (optional)
  • T1.nii.gz                                     (optional — plot background)

──────
Notetoself: add step for nuclei personalization to make this more self-sufficient
──────

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ANA~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================================================================================
"""

import os
import io
import glob
import base64
import datetime
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.ndimage import map_coordinates

# Optional deps — degrade gracefully if missing
try:
    import pandas as pd
    _HAVE_PANDAS = True
except Exception:
    _HAVE_PANDAS = False
try:
    import nibabel as nib
    from nilearn.image import resample_img
    _HAVE_NIBABEL = True
except Exception:
    _HAVE_NIBABEL = False


# ════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

CONFIG: Dict = {
    "h5_files": [
        "/Users/laptop/Documents/TUNeS/Alina/Simulations/PLAN-46-311-RESULTS.h5",
    ],
    "simnibs_path":     "/Users/laptop/Documents/TUNeS/Alina/final_tissues.nii.gz",   # None → kPlan mask
    "target_mask_path": "/Users/laptop/Documents/TUNeS/Alina/participant_nuclei/harvardoxford-subcortical_prob_Left_Thalamus.nii.gz",      # None → skip primary
    "target_name":      "Left_Thalamus",
    "region_mask_folder": "/Users/laptop/Documents/TUNeS/Alina/participant_nuclei",    # None → skip regions
    "t1_path":          "/Users/laptop/Documents/TUNeS/Alina/T1.nii.gz",               # None → tissue bg
    "output_dir":       "/Users/laptop/Documents/TUNeS/Alina/Output",
 
    "subject_id": "Alina",
    "session_id": "thal",
    "operator":   "",
    "notes":      "",
 
    # Pre-focal / region highlighting threshold (W/cm²).
    # Regions whose peak intensity exceeds this are flagged & contoured.
    "isppa_overlay_threshold": 0.5,
 
    # A pre-focal hotspot is concerning if its peak intensity reaches this
    # fraction of the focal peak intensity.
    "prefocal_ratio_flag": 0.5,
}


# ════════════════════════════════════════════════════════════════════════════
# PHYSICS & SAFETY CONSTANTS  (ITRUSST 2024 biophysical-safety consensus)
# ════════════════════════════════════════════════════════════════════════════

RHO_C        = 1.5e6     # Pa·s/m  acoustic impedance of soft tissue
BASELINE_T   = 37.0      # °C  assumed body baseline

MI_LIMIT          = 1.9   # MI / MItc non-significant-risk ceiling
T_RISE_LIMIT      = 2.0   # °C  peak temperature-rise ceiling
T_ABS_LIMIT       = 39.0  # °C  absolute-temperature ceiling
CEM43_LIMIT_BRAIN = 2.0   # CEM43
CEM43_LIMIT_BONE  = 16.0
CEM43_LIMIT_SKIN  = 21.0

# SimNIBS tissue label → report group
TISSUE_GROUPS: Dict[str, List[int]] = {
    "Brain (GM + WM + CSF)": [1, 2, 3],
    "Skull":                 [7, 8],
    "Scalp":                 [5],
    "Eyes":                  [6],
}

# Plot colours (dark theme)
CMAP_INT   = "inferno"
CMAP_BG    = "gray"
CMAP_TEMP  = "hot"
COL_6DB    = "#22d3ee"   # cyan   — −6 dB focus
COL_3DB    = "#fde047"   # yellow — −3 dB focus
COL_TARGET = "#4ade80"   # green  — primary target
COL_AXIS   = "#f472b6"   # pink   — beam axis
COL_FOCUS  = "#ffffff"   # white  — focus marker

HOT_REGION_PALETTE = [
    "#ff6b6b", "#a78bfa", "#fb923c", "#f472b6", "#38bdf8",
    "#e17055", "#fb7185", "#818cf8", "#cbd5e1", "#f43f5e",
    "#60a5fa", "#34d399",
]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def build_affine(voxel_mm: float, origin_mm: np.ndarray) -> np.ndarray:
    """4×4 isotropic NIfTI affine (voxel index → mm world)."""
    A = np.zeros((4, 4))
    A[0, 0] = A[1, 1] = A[2, 2] = voxel_mm
    A[:3, 3] = origin_mm
    A[3, 3] = 1.0
    return A


def match_shape(data: np.ndarray, target: tuple) -> np.ndarray:
    """Clip / zero-pad a 3-D array to exactly *target* shape."""
    out = np.zeros(target, dtype=data.dtype)
    s = tuple(min(a, b) for a, b in zip(data.shape, target))
    out[:s[0], :s[1], :s[2]] = data[:s[0], :s[1], :s[2]]
    return out


def voxel_to_mm(idx, affine: np.ndarray) -> np.ndarray:
    idx = np.asarray(idx, float)
    return (affine @ np.array([idx[0], idx[1], idx[2], 1.0]))[:3]


def mm_to_voxel(mm, affine: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(affine)
    mm = np.asarray(mm, float)
    return (inv @ np.array([mm[0], mm[1], mm[2], 1.0]))[:3]


def fig_to_b64(fig, dpi: int = 135) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    out = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return out


def fwhm_1d(profile: np.ndarray, half: float, voxel_mm: float) -> float:
    above = profile >= half
    if not above.any():
        return float("nan")
    idx = np.where(above)[0]
    return round((idx[-1] - idx[0] + 1) * voxel_mm, 2)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — H5 DATA LOADING  (pressure, thermal, transducer, pulse timing)
# ════════════════════════════════════════════════════════════════════════════

def _scaled_field(ds) -> np.ndarray:
    """Decode a uint16 kPlan field to physical units via stored scale attrs."""
    slope = float(ds.attrs["scale_slope"].ravel()[0])
    intcpt = float(ds.attrs["scale_intercept"].ravel()[0])
    return ds[:].astype(np.float32) * slope + intcpt


def load_h5_data(h5_path: str, son_idx: int) -> Dict:
    """
    Load everything needed for one sonication.

    Returns a dict with pressure & temperature fields (X,Y,Z), the kPlan
    medium mask, grid geometry, transducer geometry, pulse timing and all
    scalar parameters kPlan provides.
    """
    with h5py.File(h5_path, "r") as f:
        sf = f"sonications/{son_idx}/simulated_field"
        pk = f"sonications/{son_idx}/sonication_parameters"

        # ── Pressure amplitude (Pa) ────────────────────────────────────
        p_pa = np.transpose(_scaled_field(f[f"{sf}/pressure_amplitude"]))

        # ── Temperature field (°C), if present ─────────────────────────
        temp_c = None
        if f"{sf}/temperature_maximum" in f:
            temp_c = np.transpose(_scaled_field(f[f"{sf}/temperature_maximum"]))

        # ── Grid geometry ──────────────────────────────────────────────
        mm_ds = f["medium_properties/medium_mask"]
        dx_mm = float(mm_ds.attrs["grid_spacing"].ravel()[0]) * 1e3
        origin_mm = f["settings/grid/domain_position"][:].ravel()[:3] * 1e3
        affine = build_affine(dx_mm, origin_mm)
        shape = p_pa.shape

        # ── kPlan medium mask (0=bg, 1=head, 2=skull) ──────────────────
        med_mask = match_shape(np.transpose(mm_ds[:]), shape)
        try:
            med_labels = [x.decode() if isinstance(x, bytes) else str(x)
                          for x in np.ravel(f["medium_properties/medium_mask_labels"][:])]
        except Exception:
            med_labels = ["background", "head", "skull"]

        # ── Scalars ────────────────────────────────────────────────────
        def g(path, default=np.nan):
            try:
                return np.ravel(f[path][:])
            except Exception:
                return np.array([default])

        target_mm = g(f"{pk}/target_position")[:3] * 1e3
        freq_hz   = float(g(f"{pk}/driving_frequency")[0])
        focal_d   = float(g(f"{pk}/focal_distance")[0]) * 1e3      # mm
        tgt_p_pa  = float(g(f"{pk}/target_pressure")[0])
        sptp_pa   = float(g(f"{sf}/pressure_amplitude_sptp")[0])
        p_at_tgt  = float(g(f"{sf}/pressure_amplitude_at_target")[0])
        sptp_msk  = g(f"{sf}/pressure_amplitude_sptp_masked")       # [bg,head,skull]

        T_peak    = float(g(f"{sf}/temperature_at_peak")[0])
        T_target  = float(g(f"{sf}/temperature_at_target")[0])
        T_msk     = g(f"{sf}/temperature_maximum_sptp_masked")      # [bg,head,skull]
        dose_tgt  = float(g(f"{sf}/thermal_dose_at_target")[0])
        dose_sptp = float(g(f"{sf}/thermal_dose_sptp")[0])
        dose_msk  = g(f"{sf}/thermal_dose_sptp_masked")             # [bg,head,skull]

        # Pulse timing
        pst   = g(f"{pk}/pulse_sequence_timing")
        cool  = float(g(f"{pk}/pulse_sequence_cooling_time")[0])

        # ── Transducer geometry ────────────────────────────────────────
        pos_xform = g(f"{pk}/position_transform")
        pos_xform = (pos_xform.reshape(4, 4)
                     if pos_xform.size == 16 else np.eye(4))
        # translation (row-vector convention: last row) → transducer pivot
        transducer_mm = pos_xform[3, :3] * 1e3

        n_elements = 0
        try:
            n_elements = len([k for k in f["transducer/elements"].keys()
                              if k.isdigit()])
        except Exception:
            pass

        meta = {
            "target_mm": target_mm,
            "freq_hz": freq_hz, "freq_kHz": round(freq_hz / 1e3, 1),
            "focal_dist_mm": round(focal_d, 1),
            "tgt_p_kPa": round(tgt_p_pa / 1e3, 1),
            "sptp_kPa": round(sptp_pa / 1e3, 1),
            "p_at_target_kPa": round(p_at_tgt / 1e3, 1),
            "sptp_masked": sptp_msk,
            "med_labels": med_labels,
            "T_peak": T_peak, "T_target": T_target,
            "T_masked": T_msk,
            "dose_target": dose_tgt, "dose_sptp": dose_sptp,
            "dose_masked": dose_msk,
            "pulse_timing": pst, "cooling_time": cool,
            "transducer_mm": transducer_mm,
            "pos_xform": pos_xform,
            "n_elements": n_elements,
        }

    return {
        "pressure_pa": p_pa,
        "temp_c": temp_c,
        "affine_sim": affine,
        "grid_shape": shape,
        "medium_mask": med_mask,
        "voxel_mm": dx_mm,
        "voxel_vol": dx_mm ** 3,
        "meta": meta,
    }


def count_sonications(h5_path: str) -> int:
    with h5py.File(h5_path, "r") as f:
        return len([k for k in f["sonications"].keys() if k.isdigit()])


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MASKS  (SimNIBS if available, else kPlan medium-mask fallback)
# ════════════════════════════════════════════════════════════════════════════

def _resample_to_grid(img, affine_sim, grid_shape, interp="nearest"):
    res = resample_img(img, target_affine=affine_sim,
                       target_shape=grid_shape, interpolation=interp)
    return np.squeeze(res.get_fdata())


def build_tissue_masks(seg_path: Optional[str], affine_sim, grid_shape,
                       medium_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Detailed tissue masks from SimNIBS if available; otherwise derive a
    coarse fallback from the kPlan medium mask (head→brain proxy, 2→skull).
    """
    if seg_path and _HAVE_NIBABEL and os.path.exists(seg_path):
        seg = nib.load(seg_path)
        data = np.squeeze(seg.get_fdata()).astype(np.int16)
        masks = {}
        for name, labels in TISSUE_GROUPS.items():
            m = np.isin(data, labels).astype(np.uint8)
            rs = _resample_to_grid(nib.Nifti1Image(m, seg.affine),
                                   affine_sim, grid_shape, "nearest")
            masks[name] = match_shape((rs > 0.5).astype(np.uint8), grid_shape)
        masks["_source"] = "SimNIBS"
        return masks

    # Fallback: kPlan medium mask (0=bg, 1=head, 2=skull)
    masks = {
        "Brain (head proxy)": (medium_mask == 1).astype(np.uint8),
        "Skull":              (medium_mask == 2).astype(np.uint8),
        "_source": "kPlan medium mask (SimNIBS not provided)",
    }
    return masks


def get_brain_mask(tissue_masks: Dict[str, np.ndarray], grid_shape) -> np.ndarray:
    for key in ("Brain (GM + WM + CSF)", "Brain (head proxy)"):
        if key in tissue_masks:
            return tissue_masks[key]
    return np.zeros(grid_shape, np.uint8)


def load_binary_mask(path, affine_sim, grid_shape) -> Optional[np.ndarray]:
    if not (_HAVE_NIBABEL and path and os.path.exists(path)):
        return None
    rs = _resample_to_grid(nib.load(path), affine_sim, grid_shape, "nearest")
    return match_shape((rs > 0).astype(np.uint8), grid_shape)


def load_all_region_masks(folder, affine_sim, grid_shape) -> Dict[str, np.ndarray]:
    if not (_HAVE_NIBABEL and folder and os.path.isdir(folder)):
        return {}
    paths = sorted(glob.glob(os.path.join(folder, "*.nii")) +
                   glob.glob(os.path.join(folder, "*.nii.gz")))
    out = {}
    for p in paths:
        name = os.path.basename(p)
        for ext in (".nii.gz", ".nii"):
            if name.endswith(ext):
                name = name[:-len(ext)]; break
        name = (name.replace("harvardoxford-subcortical_prob_", "HO_sub_")
                    .replace("harvardoxford-cortical_prob_", "HO_ctx_")
                    .replace("harvardoxford-", "HO_"))
        try:
            rs = _resample_to_grid(nib.load(p), affine_sim, grid_shape, "nearest")
            m = match_shape((rs > 0).astype(np.uint8), grid_shape)
        except Exception as e:
            print(f"  [warn] {p}: {e}"); continue
        if m.sum() > 0:
            out[name] = m
    return out


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FOCAL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def compute_intensity_focus(pressure_pa, brain_mask, voxel_vol) -> Dict:
    """Intensity field (W/cm²) and −3/−6 dB focal-zone masks (brain-restricted)."""
    I_full = (pressure_pa ** 2) / (2.0 * RHO_C) / 1e4
    I_brain = I_full.copy()
    if brain_mask.sum() > 0:
        I_brain[brain_mask == 0] = 0.0
        ref = float(I_brain[brain_mask == 1].max())
    else:
        ref = float(I_full.max())          # no mask → use global
        I_brain = I_full.copy()

    if ref <= 0:
        raise ValueError("Peak intensity is zero — check pressure scaling/mask.")

    mask_6 = (I_brain > 0.25 * ref).astype(np.uint8)
    mask_3 = (I_brain > 0.50 * ref).astype(np.uint8)
    peak_idx = np.unravel_index(I_brain.argmax(), I_brain.shape)

    return {
        "I_full": I_full, "I_brain": I_brain,
        "mask_6dB": mask_6, "mask_3dB": mask_3,
        "Isppa": round(float(I_full.max()), 3),
        "Isppa_brain": round(ref, 3),
        "peak_idx": peak_idx,
        "focus_vol_6dB": round(float(mask_6.sum()) * voxel_vol, 1),
        "focus_vol_3dB": round(float(mask_3.sum()) * voxel_vol, 1),
    }


def compute_beam_dimensions(I_brain, peak_idx, affine_sim, voxel_mm) -> Dict:
    """Locate the (brain-restricted) intensity peak. Beam-aligned FWHM is
    computed later by `compute_beam_fwhm` once the beam axis is known."""
    return {"focus_center_mm": voxel_to_mm(peak_idx, affine_sim)}


def compute_beam_fwhm(I_field, beam: Dict, affine_sim, voxel_mm) -> Dict:
    """
    FWHM of the focus measured ALONG the beam axis (axial) and along two
    orthogonal in-plane directions (lateral) through the actual intensity peak.

    This is the physically meaningful decomposition for an obliquely-oriented
    transducer: cardinal-axis profiles would mix axial and lateral extents.
    """
    peak_mm = beam.get("peak_mm", beam.get("focus_center_mm"))
    axis = beam["axis"]
    up = np.eye(3)[np.argmin(np.abs(axis))]
    v1 = up - np.dot(up, axis) * axis; v1 /= (np.linalg.norm(v1) + 1e-9)
    v2 = np.cross(axis, v1)

    def _profile(direction, span=20.0):
        rs = np.arange(-span, span + 1e-3, voxel_mm * 0.5)
        pts = np.array([peak_mm + r * direction for r in rs])
        vox = np.array([mm_to_voxel(p, affine_sim) for p in pts]).T
        vals = map_coordinates(I_field, vox, order=1, mode="constant", cval=0.0)
        return rs, vals

    out = {}
    fwhms = {}
    for name, d in [("axial", axis), ("lat1", v1), ("lat2", v2)]:
        rs, vals = _profile(d)
        peakv = float(vals.max())
        if peakv <= 0:
            fwhms[name] = float("nan"); continue
        above = vals >= 0.5 * peakv
        idx = np.where(above)[0]
        fwhms[name] = round((idx[-1] - idx[0] + 1) * (voxel_mm * 0.5), 2) if idx.size else float("nan")
    lat = np.nanmean([fwhms["lat1"], fwhms["lat2"]])
    out["fwhm_axial_mm"] = fwhms["axial"]
    out["fwhm_lat1_mm"] = fwhms["lat1"]
    out["fwhm_lat2_mm"] = fwhms["lat2"]
    out["fwhm_lat_mean_mm"] = round(float(lat), 2) if not np.isnan(lat) else float("nan")
    out["elongation"] = (round(fwhms["axial"] / lat, 2)
                         if (not np.isnan(fwhms["axial"]) and lat > 0) else float("nan"))
    return out


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — BEAM GEOMETRY & PRE-FOCAL (BEAM-PATH) ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def compute_beam_geometry(meta: Dict, target_mm: np.ndarray,
                          peak_mm: np.ndarray,
                          medium_mask: np.ndarray, affine_sim,
                          voxel_mm: float) -> Dict:
    """
    Reconstruct the acoustic beam axis and the geometry needed for the
    pre-focal (beam-path) analysis.

    The axis is anchored on the PLANNED TARGET — the "area of interest" the
    operator chose — running transducer_pivot → target. This is deliberate:
    "pre-focal" must mean "between the transducer and the intended target",
    so that a near-field hotspot (which may itself be the global intensity
    peak) is correctly reported as pre-focal rather than redefining the focus.

    The actual intensity peak (`peak_mm`) is tracked separately and projected
    onto the same axis, so a focus that lands short of / beyond / off the
    target is immediately visible (targeting error + axial offset).

    `target_mm` falls back to `peak_mm` when no target mask is supplied.
    """
    transducer_mm = meta["transducer_mm"]
    ref_mm = np.asarray(target_mm if target_mm is not None else peak_mm, float)

    vec = ref_mm - transducer_mm
    L = float(np.linalg.norm(vec))
    axis = vec / L if L > 0 else np.array([0, 0, 1.0])
    target_proj = float(np.dot(ref_mm - transducer_mm, axis))   # == L

    # Where the actual intensity peak sits along (and off) the intended axis
    peak_rel = np.asarray(peak_mm, float) - transducer_mm
    peak_proj = float(np.dot(peak_rel, axis))
    peak_lateral = float(np.linalg.norm(peak_rel - peak_proj * axis))
    # +ve = peak is beyond the target (deeper); -ve = peak falls short (nearer)
    peak_axial_offset = round(peak_proj - target_proj, 1)

    # Walk the axis from transducer toward target; first skull (==2) voxel = entry
    entry_mm, entry_proj = None, None
    skull_thickness_mm = np.nan
    steps = np.arange(0, L + 25, voxel_mm * 0.5)
    in_skull = False
    skull_enter = skull_exit = None
    for s in steps:
        pt = transducer_mm + s * axis
        vx = np.round(mm_to_voxel(pt, affine_sim)).astype(int)
        if np.any(vx < 0) or np.any(vx >= np.array(medium_mask.shape)):
            continue
        val = medium_mask[vx[0], vx[1], vx[2]]
        if val == 2 and not in_skull:
            in_skull = True; skull_enter = s
            if entry_mm is None:
                entry_mm, entry_proj = pt.copy(), s
        elif val != 2 and in_skull:
            in_skull = False; skull_exit = s
    if skull_enter is not None and skull_exit is not None:
        skull_thickness_mm = round(skull_exit - skull_enter, 1)

    return {
        "transducer_mm": transducer_mm,
        "axis": axis,
        "path_length_mm": round(L, 1),
        # reference for "pre-focal" is the planned target
        "focus_proj": target_proj,        # kept name for downstream consumers
        "target_proj": target_proj,
        "target_mm": ref_mm,
        # actual intensity peak, on the same axis
        "peak_mm": np.asarray(peak_mm, float),
        "peak_proj": peak_proj,
        "peak_lateral_mm": round(peak_lateral, 1),
        "peak_axial_offset_mm": peak_axial_offset,
        "entry_mm": entry_mm,
        "entry_proj": entry_proj,
        "skull_thickness_mm": skull_thickness_mm,
    }


def classify_regions_along_beam(region_metrics: List[Dict],
                                region_masks: Dict[str, np.ndarray],
                                beam: Dict, affine_sim,
                                focus_mm: np.ndarray,
                                prefocal_margin_mm: float = 5.0) -> None:
    """
    Annotate each region row (in-place) with its position relative to the focus
    along the beam axis: 'pre-focal', 'at focus', or 'post-focal', plus the
    along-axis distance from the focus and lateral offset from the axis.
    """
    tx = beam["transducer_mm"]; axis = beam["axis"]; fproj = beam["focus_proj"]
    for r in region_metrics:
        m = region_masks.get(r["name"])
        if m is None or m.sum() == 0:
            r["beam_zone"] = "—"; r["along_focus_mm"] = float("nan")
            r["lateral_mm"] = float("nan"); continue
        idx = np.argwhere(m == 1)
        cen_vox = idx.mean(0)
        cen_mm = voxel_to_mm(cen_vox, affine_sim)
        rel = cen_mm - tx
        proj = float(np.dot(rel, axis))
        lateral = float(np.linalg.norm(rel - proj * axis))
        d = proj - fproj                     # <0 = before focus
        if d < -prefocal_margin_mm:
            zone = "pre-focal"
        elif d > prefocal_margin_mm:
            zone = "post-focal"
        else:
            zone = "at focus"
        r["beam_zone"] = zone
        r["along_focus_mm"] = round(d, 1)
        r["lateral_mm"] = round(lateral, 1)


def sample_along_beam(I_full, beam: Dict, affine_sim, voxel_mm,
                      brain_mask=None, focus_mask=None,
                      radius_mm: float = 3.0, extend_mm: float = 15.0) -> Dict:
    """
    Max intensity within a small disc perpendicular to the beam axis, sampled
    from the transducer pivot to just beyond the target. Reveals near-field /
    pre-focal hotspots along the propagation path.

    A "pre-focal hotspot" is energy that lies before the target AND outside the
    −6 dB focal zone — i.e. a genuine secondary deposition, not the rising edge
    of the main focus. Two figures are produced:
      • prefocal_peak       — strongest such on-axis intensity (any tissue);
      • prefocal_brain_peak — strongest such intensity in BRAIN tissue, the
                              clinically relevant "what brain is hit on the way
                              in". Used for the verdict ratio.
    """
    tx = beam["transducer_mm"]; axis = beam["axis"]
    L = beam["path_length_mm"]; fproj = beam["focus_proj"]
    up = np.eye(3)[np.argmin(np.abs(axis))]
    v1 = up - np.dot(up, axis) * axis; v1 /= (np.linalg.norm(v1) + 1e-9)
    v2 = np.cross(axis, v1)

    s_vals = np.arange(0, L + extend_mm, voxel_mm * 0.5)
    rr = np.arange(-radius_mm, radius_mm + 1e-3, voxel_mm)
    prof = np.zeros_like(s_vals)
    in_brain = np.zeros_like(s_vals, dtype=bool)
    in_focus = np.zeros_like(s_vals, dtype=bool)
    for i, s in enumerate(s_vals):
        pts = []
        for a in rr:
            for b in rr:
                if a * a + b * b > radius_mm * radius_mm:
                    continue
                pts.append(tx + s * axis + a * v1 + b * v2)
        pts = np.array(pts)
        vox = np.array([mm_to_voxel(p, affine_sim) for p in pts]).T
        prof[i] = float(map_coordinates(I_full, vox, order=1, mode="constant", cval=0.0).max())
        if brain_mask is not None:
            in_brain[i] = map_coordinates(brain_mask.astype(np.float32), vox,
                                          order=0, mode="constant", cval=0.0).max() > 0.5
        if focus_mask is not None:
            in_focus[i] = map_coordinates(focus_mask.astype(np.float32), vox,
                                          order=0, mode="constant", cval=0.0).max() > 0.5

    focal_peak = float(prof.max())
    focus_ref = float(prof[in_focus].max()) if in_focus.any() else focal_peak
    ratio_ref = max(focus_ref, 1e-9)

    # pre-focal = before target AND outside the −6 dB main lobe
    before = s_vals < fproj
    pre = before & (~in_focus)
    def _peak(mask):
        if not mask.any():
            return 0.0, float("nan")
        sub = np.where(mask, prof, -1)
        j = int(sub.argmax())
        return float(prof[j]), float(s_vals[j])
    prefocal_peak, prefocal_peak_pos = _peak(pre)
    prefocal_brain_peak, prefocal_brain_pos = _peak(pre & in_brain)
    brain_entry_proj = float(s_vals[in_brain][0]) if in_brain.any() else None

    return {
        "s_mm": s_vals, "profile": prof, "in_brain": in_brain, "in_focus": in_focus,
        "focus_proj": fproj,
        "peak_proj": beam.get("peak_proj"),
        "entry_proj": beam.get("entry_proj"),
        "brain_entry_proj": brain_entry_proj,
        "prefocal_peak": round(prefocal_peak, 3),
        "prefocal_peak_pos_mm": round(prefocal_peak_pos, 1),
        "prefocal_brain_peak": round(prefocal_brain_peak, 3),
        "prefocal_brain_pos_mm": round(prefocal_brain_pos, 1),
        "focus_ref": round(focus_ref, 3),
        "focal_peak": round(focal_peak, 3),
        "prefocal_ratio": round(prefocal_brain_peak / ratio_ref, 3),
        "prefocal_ratio_anytissue": round(prefocal_peak / ratio_ref, 3),
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — METRICS  (targeting, regions, mechanical & thermal safety)
# ════════════════════════════════════════════════════════════════════════════

def compute_targeting_metrics(target_mask, I_full, I_brain,
                              mask_6dB, mask_3dB, voxel_vol) -> Dict:
    tgt_vox = int(target_mask.sum())
    if tgt_vox == 0:
        return {"error": "Target mask empty in simulation space."}
    I_tgt = I_brain[target_mask == 1]
    fv6, fv3 = int(mask_6dB.sum()), int(mask_3dB.sum())
    ov6 = int((mask_6dB * target_mask).sum())
    ov3 = int((mask_3dB * target_mask).sum())
    on6 = round(ov6 / fv6 * 100, 1) if fv6 else 0.0
    on3 = round(ov3 / fv3 * 100, 1) if fv3 else 0.0
    return {
        "target_vol_mm3": round(tgt_vox * voxel_vol, 1),
        "Isppa_target": round(float(I_tgt.max()), 3),
        "Imean_target": round(float(I_tgt.mean()), 3),
        "overlap_vol_6dB": round(ov6 * voxel_vol, 1),
        "overlap_vol_3dB": round(ov3 * voxel_vol, 1),
        "on_target_6dB_pct": on6, "off_target_6dB_pct": round(100 - on6, 1),
        "coverage_6dB_pct": round(ov6 / tgt_vox * 100, 1),
        "on_target_3dB_pct": on3, "off_target_3dB_pct": round(100 - on3, 1),
        "coverage_3dB_pct": round(ov3 / tgt_vox * 100, 1),
    }


def compute_all_region_metrics(region_masks, I_full, I_brain,
                               mask_6dB, mask_3dB, voxel_vol,
                               isppa_threshold, target_name) -> Dict:
    rows, hot_regions, hot_colors = [], {}, {}
    for name, m in region_masks.items():
        tgt_vox = int(m.sum())
        if tgt_vox == 0:
            continue
        I_in = I_full[m == 1]               # use full field (regions may sit
        Isppa = round(float(I_in.max()), 3) # near boundaries / be non-brain)
        ov6 = int((mask_6dB * m).sum())
        ov3 = int((mask_3dB * m).sum())
        rtype = "Atlas (HO)" if ("HO_sub_" in name or "HO_ctx_" in name) else "Nuclei / ROI"
        rows.append({
            "name": name, "type": rtype,
            "target_vol_mm3": round(tgt_vox * voxel_vol, 1),
            "Isppa_target": Isppa,
            "Imean_target": round(float(I_in.mean()), 3),
            "overlap_vol_6dB": round(ov6 * voxel_vol, 1),
            "coverage_6dB_pct": round(ov6 / tgt_vox * 100, 1),
            "coverage_3dB_pct": round(ov3 / tgt_vox * 100, 1),
        })
    rows.sort(key=lambda r: r["Isppa_target"], reverse=True)
    ci = 0
    for r in rows:
        is_primary = r["name"].lower() == (target_name or "").lower()
        if r["Isppa_target"] > isppa_threshold and not is_primary:
            hot_regions[r["name"]] = region_masks[r["name"]]
            hot_colors[r["name"]] = HOT_REGION_PALETTE[ci % len(HOT_REGION_PALETTE)]
            ci += 1
    return {"rows": rows, "hot_regions": hot_regions, "hot_colors": hot_colors}


def compute_mechanical_safety(tissue_masks, pressure_pa, I_full, freq_hz) -> List[Dict]:
    """Per-tissue peak pressure, Isppa and Mechanical Index (MI = p_neg/√f_MHz)."""
    fmhz = freq_hz / 1e6
    rows = []
    for name, m in tissue_masks.items():
        if name.startswith("_") or np.sum(m) == 0:
            continue
        p = pressure_pa[m == 1]
        pp = float(p.max())
        mi = round(pp / 1e6 / np.sqrt(fmhz), 3)
        ok = mi <= MI_LIMIT
        rows.append({"tissue": name, "pp_kPa": round(pp / 1e3, 1),
                     "Isppa": round(float(I_full[m == 1].max()), 3),
                     "MI": mi, "ok": ok})
    return rows


def compute_thermal_safety(temp_c, tissue_masks, meta) -> Dict:
    """
    Evaluate thermal exposure against ITRUSST (ΔT≤2°C or T_abs≤39°C; CEM43
    ≤2 brain / 16 bone / 21 skin). Uses the 3-D temperature field when present
    plus the per-tissue scalars kPlan provides (sptp_masked = [bg,head,skull]).
    """
    out = {"available": temp_c is not None or not np.isnan(meta.get("T_peak", np.nan))}

    # Global peak from 3-D field if available, else scalar
    if temp_c is not None:
        T_peak = float(temp_c.max())
    else:
        T_peak = float(meta.get("T_peak", np.nan))
    out["T_peak"] = round(T_peak, 3)
    out["dT_peak"] = round(T_peak - BASELINE_T, 3)
    out["T_target"] = round(float(meta.get("T_target", np.nan)), 3)

    # Per-tissue peak T from sptp_masked [bg, head/brain, skull]
    Tm = np.ravel(meta.get("T_masked", []))
    per = {}
    labels = meta.get("med_labels", ["background", "head", "skull"])
    for i, lab in enumerate(labels):
        if i < Tm.size and lab.lower() != "background":
            per[lab] = round(float(Tm[i]), 3)
    out["T_per_tissue"] = per

    # Thermal dose (CEM43) — per tissue [bg, head, skull]
    Dm = np.ravel(meta.get("dose_masked", []))
    dose = {}
    for i, lab in enumerate(labels):
        if i < Dm.size and lab.lower() != "background":
            dose[lab] = float(Dm[i])
    out["dose_per_tissue"] = dose
    out["dose_target"] = float(meta.get("dose_target", np.nan))
    out["dose_sptp"] = float(meta.get("dose_sptp", np.nan))

    # Verdict (non-significant if ΔT≤2 OR T_abs≤39, AND dose within tissue limits)
    dT_ok = (out["dT_peak"] <= T_RISE_LIMIT) or (T_peak <= T_ABS_LIMIT)
    dose_ok = True
    for lab, d in dose.items():
        lim = (CEM43_LIMIT_BONE if "skull" in lab.lower() or "bone" in lab.lower()
               else CEM43_LIMIT_SKIN if "skin" in lab.lower() or "scalp" in lab.lower()
               else CEM43_LIMIT_BRAIN)
        if d > lim:
            dose_ok = False
    out["thermal_ok"] = bool(dT_ok and dose_ok)
    return out


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — DECISION VERDICT
# ════════════════════════════════════════════════════════════════════════════

def build_verdict(tm, beam, along, mech, thermal, cfg) -> Dict:
    """Synthesise a GO / CAUTION verdict with explicit reasons."""
    reasons, flags = [], []

    # Targeting
    cov = tm.get("coverage_6dB_pct", 0) if "error" not in tm else 0
    on  = tm.get("on_target_6dB_pct", 0) if "error" not in tm else 0
    if "error" not in tm:
        if cov < 20 or on < 30:
            flags.append("targeting"); reasons.append(
                f"Weak targeting (coverage {cov}%, on-target {on}%).")

    # Pre-focal exposure (energy deposited before the planned target)
    ratio = along.get("prefocal_ratio", 0)
    if ratio >= cfg.get("prefocal_ratio_flag", 0.5):
        flags.append("prefocal"); reasons.append(
            f"Beam deposits {int(ratio*100)}% of its on-axis peak intensity "
            f"before the target (pre-target brain peak at "
            f"{along.get('prefocal_brain_pos_mm')} mm from the transducer).")

    # Mechanical
    mech_ok = all(r["ok"] for r in mech) if mech else True
    if not mech_ok:
        flags.append("mechanical")
        bad = ", ".join(r["tissue"] for r in mech if not r["ok"])
        reasons.append(f"MI exceeds {MI_LIMIT} in: {bad}.")

    # Thermal
    if thermal.get("available") and not thermal.get("thermal_ok", True):
        flags.append("thermal"); reasons.append(
            f"Thermal exposure exceeds ITRUSST levels "
            f"(ΔT {thermal.get('dT_peak')} °C).")

    level = "go" if not flags else ("caution" if flags == ["targeting"] or
                                    len(flags) == 1 else "review")
    if not flags:
        reasons = ["All checks within ITRUSST non-significant-risk levels and "
                   "targeting is adequate."]
    return {"level": level, "flags": flags, "reasons": reasons,
            "mech_ok": mech_ok, "coverage": cov, "on_target": on,
            "prefocal_ratio": ratio}


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

def _tissue_rgb(tissue_masks, grid_shape):
    """Combined labelled background: scalp/bone/brain → grayscale levels."""
    bg = np.zeros(grid_shape, float)
    for key, lvl in [("Scalp", 0.35), ("Skull", 0.65),
                     ("Brain (GM + WM + CSF)", 0.5), ("Brain (head proxy)", 0.5)]:
        if key in tissue_masks:
            bg[tissue_masks[key] == 1] = lvl
    return bg


def _background(t1_path, medium_mask, affine_sim, grid_shape):
    if _HAVE_NIBABEL and t1_path and os.path.exists(t1_path):
        bg = match_shape(_resample_to_grid(nib.load(t1_path), affine_sim,
                                           grid_shape, "linear"), grid_shape).astype(float)
    else:
        bg = medium_mask.astype(float)
    mx = bg.max()
    return bg / mx if mx > 0 else bg


def plot_orthogonal_views(I_brain, mask_6dB, mask_3dB, target_mask, medium_mask,
                          affine_sim, grid_shape, peak_idx, t1_path=None,
                          hot_regions=None, hot_colors=None) -> str:
    hot_regions = hot_regions or {}; hot_colors = hot_colors or {}
    bg = _background(t1_path, medium_mask, affine_sim, grid_shape)
    px, py, pz = peak_idx
    Inorm = I_brain / (I_brain.max() + 1e-12)
    Idisp = np.where(mask_6dB == 1, Inorm, 0.0)
    tmask = target_mask if target_mask is not None else np.zeros(grid_shape, np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor="#0b0e14")
    views = [
        (bg[px, :, :], Idisp[px, :, :], mask_6dB[px, :, :], mask_3dB[px, :, :],
         tmask[px, :, :], "Sagittal", "sag"),
        (bg[:, py, :], Idisp[:, py, :], mask_6dB[:, py, :], mask_3dB[:, py, :],
         tmask[:, py, :], "Coronal", "cor"),
        (bg[:, :, pz], Idisp[:, :, pz], mask_6dB[:, :, pz], mask_3dB[:, :, pz],
         tmask[:, :, pz], "Axial", "axi"),
    ]
    for ax, (b, I, s6, s3, ts, title, key) in zip(axes, views):
        ax.set_facecolor("#0b0e14")
        ax.imshow(b.T, cmap=CMAP_BG, origin="lower", vmin=0, vmax=1, alpha=0.8)
        ax.imshow(np.ma.masked_where(I == 0, I).T, cmap=CMAP_INT, origin="lower",
                  alpha=0.9, vmin=0, vmax=1)
        for sl, c, lw, ls in [(s6, COL_6DB, 1.5, "solid"), (s3, COL_3DB, 1.2, "solid"),
                              (ts, COL_TARGET, 2.0, "dashed")]:
            if sl.any():
                ax.contour(sl.T, levels=[0.5], colors=[c], linewidths=lw,
                           linestyles=ls, origin="lower")
        for rname, m in hot_regions.items():
            sl = {"sag": m[px, :, :], "cor": m[:, py, :], "axi": m[:, :, pz]}[key]
            if sl.any():
                ax.contour(sl.T, levels=[0.5], colors=[hot_colors.get(rname, "#fff")],
                           linewidths=1.1, linestyles="dotted", origin="lower")
        ax.set_title(title, color="#e6edf3", fontsize=11, fontweight="bold", pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#222a36")
    sm = plt.cm.ScalarMappable(cmap=CMAP_INT, norm=mcolors.Normalize(0, 1)); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.02, shrink=0.8)
    cb.set_label("Normalised intensity", color="#e6edf3", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="#e6edf3", labelcolor="#e6edf3", labelsize=8)
    fig.suptitle("Acoustic focus — orthogonal views through the intensity peak",
                 color="#e6edf3", fontsize=11, y=1.02)
    return fig_to_b64(fig)


def plot_beam_corridor(I_full, tissue_masks, beam, affine_sim, grid_shape,
                       voxel_mm, isppa_thr, region_masks=None,
                       hot_colors=None) -> str:
    """
    Oblique reslice ALONG the beam axis — the 'acoustic corridor' view.
    Shows tissue background, intensity, transducer side, skull entry and the
    focus, so pre-focal energy deposition is directly visible.
    """
    tx = beam["transducer_mm"]; axis = beam["axis"]
    L = beam["path_length_mm"]; fproj = beam["focus_proj"]
    up = np.eye(3)[np.argmin(np.abs(axis))]
    v1 = up - np.dot(up, axis) * axis; v1 /= (np.linalg.norm(v1) + 1e-9)

    s = np.arange(-5, L + 22, voxel_mm * 0.6)
    t = np.arange(-32, 32, voxel_mm * 0.6)
    SS, TT = np.meshgrid(s, t, indexing="ij")
    world = (tx[None, None, :] + SS[..., None] * axis + TT[..., None] * v1)
    inv = np.linalg.inv(affine_sim)
    vox = world @ inv[:3, :3].T + inv[:3, 3]
    coords = [vox[..., 0], vox[..., 1], vox[..., 2]]

    tissue_bg = _tissue_rgb(tissue_masks, grid_shape)
    bg_s = map_coordinates(tissue_bg, coords, order=1, mode="constant", cval=0)
    I_s = map_coordinates(I_full, coords, order=1, mode="constant", cval=0)

    fig, ax = plt.subplots(figsize=(11, 5.6), facecolor="#0b0e14")
    ax.set_facecolor("#0b0e14")
    ext = [s.min(), s.max(), t.min(), t.max()]
    ax.imshow(bg_s.T, cmap=CMAP_BG, origin="lower", extent=ext, aspect="auto",
              vmin=0, vmax=1, alpha=0.85)
    Imax = I_s.max() + 1e-9
    ax.imshow(np.ma.masked_where(I_s < isppa_thr * 0.5, I_s).T, cmap=CMAP_INT,
              origin="lower", extent=ext, aspect="auto", alpha=0.92, vmin=0, vmax=Imax)
    ax.contour((I_s / Imax).T, levels=[0.25, 0.5], colors=[COL_6DB, COL_3DB],
               linewidths=[1.3, 1.1], extent=ext, origin="lower")

    # beam axis & markers
    ax.axhline(0, color=COL_AXIS, lw=1.0, ls=(0, (4, 3)), alpha=0.8)
    # planned target (the area of interest)
    ax.plot([fproj], [0], marker="o", ms=11, mfc="none", mec=COL_TARGET, mew=2.2)
    ax.annotate("target", (fproj, 0), color=COL_TARGET, fontsize=9,
                xytext=(fproj, 7), ha="center")
    # actual intensity peak (may sit short of / beyond / off-axis from target)
    ppx = beam.get("peak_proj"); ply = beam.get("peak_lateral_mm", 0.0)
    if ppx is not None and abs(ppx - fproj) > 1.0:
        ax.plot([ppx], [ply], marker="x", ms=11, color=COL_FOCUS, mew=2.2)
        ax.annotate("actual\npeak", (ppx, ply), color=COL_FOCUS, fontsize=8,
                    ha="center", xytext=(ppx, ply - 9))
    if beam.get("entry_proj") is not None:
        ax.axvline(beam["entry_proj"], color="#9ca3af", lw=1.0, ls=":")
        ax.annotate("skull\nentry", (beam["entry_proj"], t.min() + 4),
                    color="#9ca3af", fontsize=8, ha="center")
    ax.annotate("transducer →", (s.min() + 1, t.max() - 5), color=COL_AXIS, fontsize=9)

    # region intersections along corridor (contours)
    if region_masks and hot_colors:
        for rname, col in hot_colors.items():
            m = region_masks.get(rname)
            if m is None:
                continue
            ms = map_coordinates(m.astype(float), coords, order=0, mode="constant", cval=0)
            if ms.max() > 0:
                ax.contour(ms.T, levels=[0.5], colors=[col], linewidths=1.0,
                           linestyles="dotted", extent=ext, origin="lower")

    ax.set_xlabel("Distance along beam axis from transducer (mm)",
                  color="#9ca3af", fontsize=9)
    ax.set_ylabel("Lateral (mm)", color="#9ca3af", fontsize=9)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#222a36")
    ax.set_title("Beam-path corridor — what the beam crosses before the target",
                 color="#e6edf3", fontsize=11, fontweight="bold")
    return fig_to_b64(fig)


def plot_along_beam(along: Dict, isppa_thr: float) -> str:
    s = along["s_mm"]; p = along["profile"]; fproj = along["focus_proj"]
    fig, ax = plt.subplots(figsize=(11, 3.6), facecolor="#0b0e14")
    ax.set_facecolor("#11161f")
    # shade the in-brain portion of the path
    ib = along.get("in_brain")
    if ib is not None and np.any(ib):
        ax.fill_between(s, 0, p.max() * 1.18, where=ib, color="#3b82f6",
                        alpha=0.10, step="mid", label="brain on path")
    fz = along.get("in_focus")
    if fz is not None and np.any(fz):
        ax.fill_between(s, 0, p.max() * 1.18, where=fz, color=COL_6DB,
                        alpha=0.16, step="mid", label="−6 dB focal zone")
    ax.plot(s, p, color="#f59e0b", lw=2.0)
    ax.fill_between(s, 0, p, color="#f59e0b", alpha=0.15)
    ax.axvline(fproj, color=COL_TARGET, lw=1.4, ls="--", label="planned target")
    if along.get("peak_proj") is not None and abs(along["peak_proj"] - fproj) > 1.0:
        ax.axvline(along["peak_proj"], color=COL_FOCUS, lw=1.2, ls="-.",
                   label="actual peak")
    ax.axhline(isppa_thr, color=COL_6DB, lw=1.0, ls=(0, (4, 3)),
               label=f"threshold {isppa_thr} W/cm²")
    # in-brain pre-focal peak — the clinically relevant marker
    bpk = along.get("prefocal_brain_peak", 0)
    bpos = along.get("prefocal_brain_pos_mm", np.nan)
    if bpk > 0 and not np.isnan(bpos):
        ax.plot([bpos], [bpk], marker="v", ms=10, color="#f43f5e")
        ax.annotate(f"pre-target brain peak\n{bpk} W/cm²", (bpos, bpk),
                    color="#f43f5e", fontsize=8, ha="center",
                    xytext=(bpos, bpk + p.max() * 0.12))
    ax.set_ylim(0, p.max() * 1.25 + 1e-6)
    ax.set_xlabel("Distance along beam axis from transducer (mm)", color="#9ca3af", fontsize=9)
    ax.set_ylabel("Peak intensity (W/cm²)", color="#9ca3af", fontsize=9)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#222a36")
    ax.legend(facecolor="#11161f", edgecolor="#222a36", labelcolor="#e6edf3",
              fontsize=8, ncol=2, loc="upper right")
    ax.set_title("Intensity deposited along the beam path", color="#e6edf3",
                 fontsize=11, fontweight="bold")
    return fig_to_b64(fig)


def plot_temperature(temp_c, peak_idx, affine_sim, grid_shape) -> str:
    if temp_c is None:
        return ""
    px, py, pz = np.unravel_index(temp_c.argmax(), temp_c.shape)
    dT = temp_c - BASELINE_T
    vmax = max(0.1, float(dT.max()))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), facecolor="#0b0e14")
    for ax, sl, title in zip(axes,
                             [dT[px, :, :], dT[:, py, :], dT[:, :, pz]],
                             ["Sagittal", "Coronal", "Axial"]):
        ax.set_facecolor("#0b0e14")
        im = ax.imshow(np.ma.masked_where(sl <= 0.02, sl).T, cmap=CMAP_TEMP,
                       origin="lower", vmin=0, vmax=vmax)
        ax.set_title(title, color="#e6edf3", fontsize=11, fontweight="bold", pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#222a36")
    sm = plt.cm.ScalarMappable(cmap=CMAP_TEMP, norm=mcolors.Normalize(0, vmax)); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.02, shrink=0.8)
    cb.set_label("Temperature rise ΔT (°C)", color="#e6edf3", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="#e6edf3", labelcolor="#e6edf3", labelsize=8)
    fig.suptitle("Maximum temperature rise", color="#e6edf3", fontsize=11, y=1.02)
    return fig_to_b64(fig)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — HTML REPORT
# ════════════════════════════════════════════════════════════════════════════

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
 --bg:#0b0e14;--bg2:#11161f;--bg3:#1a212e;--bd:#222a36;
 --tx:#e6edf3;--mut:#8b98a9;--acc:#3b82f6;--acc2:#60a5fa;
 --go:#22c55e;--cau:#f59e0b;--rev:#ef4444;
 --mono:'IBM Plex Mono',monospace;--sans:'Inter',sans-serif;--r:10px}
body{background:var(--bg);color:var(--tx);font-family:var(--sans);font-size:14px;line-height:1.6}
.wrap{max-width:1180px;margin:0 auto;padding:0 22px 80px}
.head{padding:34px 22px 26px;border-bottom:1px solid var(--bd);
 background:radial-gradient(1200px 240px at 80% -40%,rgba(59,130,246,.16),transparent)}
.head .inner{max-width:1180px;margin:0 auto}
.tag{display:inline-block;font-family:var(--mono);font-size:11px;letter-spacing:.08em;
 color:var(--acc2);border:1px solid rgba(59,130,246,.4);background:rgba(59,130,246,.1);
 padding:3px 10px;border-radius:20px;margin-bottom:12px}
h1{font-size:25px;font-weight:700;letter-spacing:-.3px}
.sub{color:var(--mut);font-size:13px;margin-top:2px}
.meta{display:flex;gap:30px;flex-wrap:wrap;margin-top:18px}
.meta div{display:flex;flex-direction:column;gap:1px}
.meta .k{font-family:var(--mono);font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut)}
.meta .v{font-size:13px;font-weight:600}
.sec-title{font-family:var(--mono);font-size:12px;letter-spacing:.12em;text-transform:uppercase;
 color:var(--mut);margin:34px 0 14px;padding-bottom:7px;border-bottom:1px solid var(--bd)}
.card{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--r);padding:18px 20px;margin-bottom:14px}
.card h3{font-size:13px;font-weight:600;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.card h3::before{content:'';width:3px;height:15px;background:var(--acc);border-radius:2px;display:inline-block}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
@media(max-width:820px){.grid2,.grid3{grid-template-columns:1fr}}
.row{display:flex;justify-content:space-between;align-items:baseline;padding:6px 0;border-bottom:1px solid var(--bd)}
.row:last-child{border:0}
.row .k{color:var(--mut);font-size:12.5px}
.row .v{font-family:var(--mono);font-size:13px;font-weight:600;color:var(--acc2)}
.row .u{color:var(--mut);font-size:10.5px;margin-left:3px;font-family:var(--sans);font-weight:400}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{background:var(--bg3);color:var(--mut);font-family:var(--mono);font-size:10.5px;text-transform:uppercase;
 letter-spacing:.05em;font-weight:500;text-align:left;padding:9px 12px;border-bottom:1px solid var(--bd)}
td{padding:8px 12px;border-bottom:1px solid var(--bd)}
tr:last-child td{border:0}
tbody tr:hover{background:rgba(255,255,255,.025)}
td.n{font-family:var(--mono);color:var(--acc2)}
td.m{color:var(--mut);font-size:11.5px}
.badge{display:inline-block;padding:2px 9px;border-radius:20px;font-family:var(--mono);font-size:10.5px;font-weight:600}
.b-go{background:rgba(34,197,94,.15);color:var(--go);border:1px solid rgba(34,197,94,.4)}
.b-cau{background:rgba(245,158,11,.15);color:var(--cau);border:1px solid rgba(245,158,11,.4)}
.b-rev{background:rgba(239,68,68,.15);color:var(--rev);border:1px solid rgba(239,68,68,.4)}
.son-hd{display:flex;align-items:center;gap:14px;padding:14px 18px;border-radius:var(--r);margin:8px 0 18px;
 border:1px solid var(--bd);background:linear-gradient(90deg,rgba(59,130,246,.08),transparent)}
.son-no{font-family:var(--mono);font-weight:700;font-size:13px;color:#fff;background:var(--acc);padding:4px 12px;border-radius:20px}
.verdict{border-radius:var(--r);padding:16px 18px;margin-bottom:14px;border:1px solid}
.v-go{border-color:rgba(34,197,94,.35);background:rgba(34,197,94,.06)}
.v-cau{border-color:rgba(245,158,11,.35);background:rgba(245,158,11,.06)}
.v-rev{border-color:rgba(239,68,68,.35);background:rgba(239,68,68,.06)}
.verdict .lead{display:flex;align-items:center;gap:10px;font-size:15px;font-weight:700;margin-bottom:6px}
.verdict ul{margin:6px 0 0 2px;padding-left:18px;color:var(--mut);font-size:12.5px}
.chips{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px}
.chip{background:var(--bg3);border:1px solid var(--bd);border-radius:8px;padding:8px 12px;min-width:120px}
.chip .k{font-family:var(--mono);font-size:10px;text-transform:uppercase;color:var(--mut);letter-spacing:.04em}
.chip .v{font-family:var(--mono);font-size:18px;font-weight:600;margin-top:2px}
.fig{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--r);overflow:hidden;margin-bottom:14px}
.fig img{width:100%;display:block}
.fig .cap{padding:9px 14px;font-size:11.5px;color:var(--mut);border-top:1px solid var(--bd);font-style:italic}
.pill{display:inline-flex;align-items:center;gap:6px;background:rgba(255,255,255,.04);border:1px solid var(--bd);
 border-radius:20px;padding:3px 11px;font-size:11px;color:var(--mut);margin:0 6px 6px 0}
.dot{width:11px;height:3px;border-radius:2px;display:inline-block}
.rtbl-wrap{overflow-x:auto}.rtbl-wrap table{min-width:760px}
.filter{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:11px}
.filter input{flex:1;min-width:160px;background:var(--bg3);border:1px solid var(--bd);border-radius:7px;
 color:var(--tx);font-family:var(--mono);font-size:12px;padding:7px 11px;outline:none}
.tb{background:var(--bg3);border:1px solid var(--bd);border-radius:7px;color:var(--mut);font-family:var(--mono);
 font-size:11.5px;padding:6px 12px;cursor:pointer}
.tb.on{background:var(--acc);border-color:var(--acc);color:#fff}
tr.hot td{background:rgba(239,68,68,.06)}tr.hot td:first-child{border-left:3px solid var(--rev)}
td.hotv{color:var(--rev)!important;font-weight:700}
.zone-pre{color:var(--cau);font-weight:600}.zone-foc{color:var(--go);font-weight:600}.zone-post{color:var(--mut)}
.note{font-size:11px;color:var(--mut);margin-top:9px}
.foot{border-top:1px solid var(--bd);padding:22px;text-align:center;color:var(--mut);font-family:var(--mono);font-size:11px;margin-top:30px}
.cmp td.best{color:var(--go);font-weight:700}
"""


def _f(v, d=2, u=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "<span class='m'>n/a</span>"
    s = f"{v:.{d}f}" if isinstance(v, float) else str(v)
    return f"{s}<span class='u'>{u}</span>" if u else s


def _row(k, v, d=2, u=""):
    return f"<div class='row'><span class='k'>{k}</span><span class='v'>{_f(v,d,u)}</span></div>"


def _badge(level):
    return {"go": "<span class='badge b-go'>GO</span>",
            "caution": "<span class='badge b-cau'>CAUTION</span>",
            "review": "<span class='badge b-rev'>REVIEW</span>"}[level]


def _fig(b64, cap):
    return (f"<div class='fig'><img src='data:image/png;base64,{b64}'>"
            f"<div class='cap'>{cap}</div></div>") if b64 else ""


def _verdict_html(v):
    cls = {"go": "v-go", "caution": "v-cau", "review": "v-rev"}[v["level"]]
    lead = {"go": "Suitable placement", "caution": "Usable — review flags",
            "review": "Review before use"}[v["level"]]
    reasons = "".join(f"<li>{r}</li>" for r in v["reasons"])
    pf = int(v["prefocal_ratio"] * 100)
    return f"""
<div class='verdict {cls}'>
  <div class='lead'>{_badge(v['level'])} {lead}</div>
  <ul>{reasons}</ul>
  <div class='chips'>
    <div class='chip'><div class='k'>Coverage −6dB</div><div class='v'>{v['coverage']}%</div></div>
    <div class='chip'><div class='k'>On-target −6dB</div><div class='v'>{v['on_target']}%</div></div>
    <div class='chip'><div class='k'>Pre-focal / peak</div><div class='v'>{pf}%</div></div>
    <div class='chip'><div class='k'>Mechanical</div><div class='v'>{'OK' if v['mech_ok'] else 'OVER'}</div></div>
  </div>
</div>"""


def _placement_card(meta, beam, along):
    tx = beam["transducer_mm"]; fc = beam.get("focus_center_mm", meta["target_mm"])
    steer = round(float(np.linalg.norm(np.array(fc) - np.asarray(meta["target_mm"], float))), 1)
    return f"""
<div class='card'><h3>Transducer placement</h3>
 {_row("Elements", meta['n_elements'], 0)}
 {_row("Centre frequency", meta['freq_kHz'], 1, "kHz")}
 {_row("Geometric focal distance", meta['focal_dist_mm'], 1, "mm")}
 {_row("Beam path (transducer→target)", beam['path_length_mm'], 1, "mm")}
 {_row("Skull thickness on axis", beam['skull_thickness_mm'], 1, "mm")}
 {_row("Targeting error (peak→target)", steer, 1, "mm")}
 {_row("Peak axial offset (+deep/−short)", beam.get('peak_axial_offset_mm'), 1, "mm")}
 {_row("Peak off-axis distance", beam.get('peak_lateral_mm'), 1, "mm")}
 {_row("Transducer pivot X", tx[0], 1, "mm")}
 {_row("Transducer pivot Y", tx[1], 1, "mm")}
 {_row("Transducer pivot Z", tx[2], 1, "mm")}
</div>"""


def _focus_card(foc, beam, meta):
    fc = beam.get("focus_center_mm", [np.nan]*3)
    return f"""
<div class='card'><h3>Acoustic focus</h3>
 {_row("Isppa (global peak)", foc['Isppa'], 2, "W/cm²")}
 {_row("Isppa (in brain)", foc['Isppa_brain'], 2, "W/cm²")}
 {_row("Peak pressure (SPTP)", meta['sptp_kPa'], 1, "kPa")}
 {_row("Pressure at target", meta['p_at_target_kPa'], 1, "kPa")}
 {_row("Focal volume −6 dB", foc['focus_vol_6dB'], 1, "mm³")}
 {_row("Focal volume −3 dB", foc['focus_vol_3dB'], 1, "mm³")}
 {_row("FWHM axial (along beam)", beam.get('fwhm_axial_mm'), 2, "mm")}
 {_row("FWHM lateral (mean)", beam.get('fwhm_lat_mean_mm'), 2, "mm")}
 {_row("Beam elongation (axial/lateral)", beam.get('elongation'), 2)}
</div>"""


def _targeting_card(tm, name):
    if "error" in tm:
        return f"<div class='card'><h3>Targeting</h3><p class='m'>{tm['error']}</p></div>"
    return f"""
<div class='card'><h3>Targeting — {name}</h3>
 {_row("Target volume", tm['target_vol_mm3'], 1, "mm³")}
 {_row("Isppa in target", tm['Isppa_target'], 2, "W/cm²")}
 {_row("Mean intensity in target", tm['Imean_target'], 2, "W/cm²")}
 {_row("Coverage −6 dB", tm['coverage_6dB_pct'], 1, "%")}
 {_row("Coverage −3 dB", tm['coverage_3dB_pct'], 1, "%")}
 {_row("On-target −6 dB", tm['on_target_6dB_pct'], 1, "%")}
 {_row("Off-target −6 dB", tm['off_target_6dB_pct'], 1, "%")}
 {_row("Overlap volume −6 dB", tm['overlap_vol_6dB'], 1, "mm³")}
</div>"""


def _safety_card(mech, thermal):
    BADGE_OK = "<span class='badge b-go'>OK</span>"
    BADGE_OVER = "<span class='badge b-rev'>OVER</span>"
    mrows = ""
    for r in mech:
        status = BADGE_OK if r["ok"] else BADGE_OVER
        mrows += (f"<tr><td>{r['tissue']}</td><td class='n'>{r['pp_kPa']}</td>"
                  f"<td class='n'>{r['MI']}</td><td>{status}</td></tr>")
    # thermal block
    if thermal.get("available"):
        tpt = "".join(f"<div class='row'><span class='k'>{k}</span>"
                      f"<span class='v'>{_f(v,3,'°C')}</span></div>"
                      for k, v in thermal.get("T_per_tissue", {}).items())
        dpt = "".join(f"<div class='row'><span class='k'>CEM43 {k}</span>"
                      f"<span class='v'>{v:.2e}</span></div>"
                      for k, v in thermal.get("dose_per_tissue", {}).items())
        tbadge = ("<span class='badge b-go'>within ITRUSST</span>" if thermal['thermal_ok']
                  else "<span class='badge b-rev'>exceeds</span>")
        thermal_html = f"""
   {_row("Peak temperature", thermal['T_peak'], 3, "°C")}
   {_row("Peak rise ΔT", thermal['dT_peak'], 3, "°C")}
   {tpt}{dpt}
   <div class='row'><span class='k'>Thermal verdict</span><span class='v'>{tbadge}</span></div>"""
    else:
        thermal_html = "<p class='m'>No temperature field in this H5.</p>"
    return f"""
<div class='card'><h3>Safety — mechanical &amp; thermal</h3>
 <table><thead><tr><th>Tissue</th><th>Peak P (kPa)</th><th>MI</th><th>Status</th></tr></thead>
 <tbody>{mrows}</tbody></table>
 <div style='margin-top:12px'>{thermal_html}</div>
 <p class='note'>ITRUSST 2024 non-significant-risk levels: MI ≤ {MI_LIMIT};
 ΔT ≤ {T_RISE_LIMIT} °C or T ≤ {T_ABS_LIMIT} °C; CEM43 ≤ {CEM43_LIMIT_BRAIN} (brain) /
 {CEM43_LIMIT_BONE:.0f} (bone) / {CEM43_LIMIT_SKIN:.0f} (skin).</p>
</div>"""


def _prefocal_card(region_metrics, along, cfg):
    thr = cfg["isppa_overlay_threshold"]
    pre = [r for r in region_metrics if r.get("beam_zone") == "pre-focal"
           and r["Isppa_target"] >= thr]
    pre.sort(key=lambda r: r["Isppa_target"], reverse=True)
    if pre:
        rows = "".join(
            f"<tr class='hot'><td>{r['name']}</td><td class='m'>{r['type']}</td>"
            f"<td class='n hotv'>{r['Isppa_target']:.2f}</td>"
            f"<td class='n'>{abs(r['along_focus_mm'])}</td>"
            f"<td class='n'>{r['lateral_mm']}</td>"
            f"<td class='n'>{r['coverage_6dB_pct']:.0f}</td></tr>" for r in pre)
        body = (f"<table><thead><tr><th>Structure</th><th>Type</th>"
                f"<th>Isppa W/cm²</th><th>Before target (mm)</th>"
                f"<th>Off-axis (mm)</th><th>Cov −6dB %</th></tr></thead><tbody>{rows}</tbody></table>")
    else:
        body = "<p class='m'>No mapped structure between transducer and target exceeds the threshold.</p>"
    ratio = along.get("prefocal_ratio", 0)                      # in-brain vs focus
    rcls = "zone-pre" if ratio >= cfg["prefocal_ratio_flag"] else "zone-foc"
    bpk = along.get("prefocal_brain_peak", 0)
    bpos = along.get("prefocal_brain_pos_mm", "n/a")
    any_pk = along.get("prefocal_peak", 0)
    return f"""
<div class='card'><h3>Pre-focal exposure (brain crossed before the target)</h3>
 <div class='chips' style='margin-bottom:12px'>
   <div class='chip'><div class='k'>Pre-target brain peak</div><div class='v'>{bpk}<span class='u'> W/cm²</span></div></div>
   <div class='chip'><div class='k'>Position from transducer</div><div class='v'>{bpos}<span class='u'> mm</span></div></div>
   <div class='chip'><div class='k'>vs focus intensity</div><div class='v {rcls}'>{int(ratio*100)}%</div></div>
   <div class='chip'><div class='k'>Any-tissue peak</div><div class='v'>{any_pk}<span class='u'> W/cm²</span></div></div>
 </div>
 {body}
 <p class='note'>Brain structures the beam crosses on the way to the target. A pre-target
 brain intensity approaching the focus indicates substantial off-target neuromodulation —
 a reason to reconsider the placement. "Any-tissue peak" also counts near-field scalp/skull.</p>
</div>"""


def _region_table(rows, thr, hot_colors, tid):
    if not rows:
        return ""
    def sw(n):
        c = hot_colors.get(n)
        return (f"<span style='display:inline-block;width:9px;height:9px;border-radius:50%;"
                f"background:{c};margin-right:6px;vertical-align:middle'></span>") if c else ""
    def zspan(z):
        cls = {"pre-focal": "zone-pre", "at focus": "zone-foc",
               "post-focal": "zone-post"}.get(z, "")
        return f"<span class='{cls}'>{z}</span>"
    body = ""
    for r in rows:
        hot = r["Isppa_target"] > thr
        hot_cls = "hot" if hot else ""
        hotv_cls = "hotv" if hot else ""
        rtype = r["type"]
        body += (f"<tr class='{hot_cls}' data-type='{rtype}'>"
                 f"<td>{sw(r['name'])}{r['name']}</td><td class='m'>{rtype}</td>"
                 f"<td>{zspan(r.get('beam_zone','—'))}</td>"
                 f"<td class='n {hotv_cls}'>{r['Isppa_target']:.2f}</td>"
                 f"<td class='n'>{r['target_vol_mm3']}</td>"
                 f"<td class='n'>{r['coverage_6dB_pct']:.0f}</td></tr>")
    nh = sum(1 for r in rows if r["Isppa_target"] > thr)
    return f"""
<div class='card'><h3>All regions — intensity &amp; beam position</h3>
 <div class='filter'>
   <input id='{tid}_s' placeholder='Filter…' oninput="rf('{tid}')">
   <button class='tb on' id='{tid}_all' onclick="rt('{tid}','all')">All ({len(rows)})</button>
   <button class='tb' id='{tid}_hot' onclick="rt('{tid}','hot')">⚠ Hot ({nh})</button>
   <button class='tb' id='{tid}_pre' onclick="rt('{tid}','pre')">Pre-focal</button>
 </div>
 <div class='rtbl-wrap'><table id='{tid}'><thead><tr><th>Region</th><th>Type</th>
  <th>Beam zone</th><th>Isppa W/cm²</th><th>Vol mm³</th><th>Cov −6dB %</th></tr></thead>
  <tbody>{body}</tbody></table></div>
 <p class='note'>Sorted by peak intensity. Highlighted rows exceed {thr} W/cm² and are
 contoured on the views. "Beam zone" places each structure before / at / after the focus.</p>
</div>
<script>
window.rt=window.rt||function(t,m){{['all','hot','pre'].forEach(k=>{{var b=document.getElementById(t+'_'+k);if(b)b.classList.toggle('on',k===m);}});window['_m_'+t]=m;rf(t);}};
window.rf=window.rf||function(t){{var q=(document.getElementById(t+'_s').value||'').toLowerCase();var m=window['_m_'+t]||'all';
document.querySelectorAll('#'+t+' tbody tr').forEach(function(r){{var hot=r.classList.contains('hot');var pre=r.children[2].textContent.indexOf('pre-focal')>=0;
var ok=(m==='all')||(m==='hot'&&hot)||(m==='pre'&&pre);var s=!q||r.textContent.toLowerCase().includes(q);r.style.display=(ok&&s)?'':'none';}});}};
</script>"""


def _comparison_table(summaries: List[Dict], target_name: str) -> str:
    """Top-of-report comparison so the operator can pick the best placement."""
    if len(summaries) < 2:
        return ""
    # best = highest coverage among GO/CAUTION with lowest pre-focal ratio
    def score(s):
        penalty = 0 if s["verdict"]["level"] == "go" else (1 if s["verdict"]["level"] == "caution" else 2)
        return (-penalty, s["verdict"]["coverage"], -s["verdict"]["prefocal_ratio"])
    best_i = max(range(len(summaries)), key=lambda i: score(summaries[i]))
    rows = ""
    for i, s in enumerate(summaries):
        v = s["verdict"]; best = (i == best_i)
        cellcls = "best" if best else "n"
        star = " ★" if best else ""
        rows += (f"<tr><td class='{cellcls}'>Sonication {s['idx']}{star}</td>"
                 f"<td>{_badge(v['level'])}</td>"
                 f"<td class='{cellcls}'>{v['coverage']}%</td>"
                 f"<td class='n'>{v['on_target']}%</td>"
                 f"<td class='n'>{int(v['prefocal_ratio']*100)}%</td>"
                 f"<td class='n'>{s['Isppa_target']}</td>"
                 f"<td class='n'>{s['max_MI']}</td>"
                 f"<td class='n'>{_f(s['dT'],2)}</td></tr>")
    return f"""
<div class='sec-title'>Placement comparison</div>
<div class='card cmp'><h3>Candidate placements — {target_name}</h3>
 <table><thead><tr><th>Placement</th><th>Verdict</th><th>Coverage −6dB</th>
  <th>On-target</th><th>Pre-focal/focus</th><th>Isppa target</th><th>Max MI</th><th>ΔT °C</th></tr></thead>
  <tbody>{rows}</tbody></table>
 <p class='note'>★ = best candidate by verdict, then target coverage, then lowest
 pre-focal exposure. Review the full per-placement sections below before deciding.</p>
</div>"""


def _sonication_block(s: Dict, cfg: Dict) -> str:
    idx = s["idx"]; meta = s["meta"]; foc = s["foc"]; beam = s["beam"]
    tm = s["tm"]; mech = s["mech"]; thermal = s["thermal"]; along = s["along"]
    region_metrics = s["region_metrics"]; hot_colors = s["hot_colors"]
    name = cfg["target_name"]
    fc = beam.get("focus_center_mm", meta["target_mm"])

    region_tbl = (_region_table(region_metrics, cfg["isppa_overlay_threshold"],
                                hot_colors, f"rt{idx}") if region_metrics else "")
    prefocal = _prefocal_card(region_metrics, along, cfg) if region_metrics else ""
    pills = "".join(
        f"<span class='pill'><span class='dot' style='background:{c}'></span>{n}</span>"
        for n, c in hot_colors.items())

    return f"""
<div class='son-hd'><span class='son-no'>Sonication {idx}</span>
 <div><div style='font-weight:600'>Target: {name}</div>
 <div class='m' style='font-family:var(--mono);font-size:11.5px'>
  {meta['freq_kHz']} kHz · focus ({fc[0]:.0f}, {fc[1]:.0f}, {fc[2]:.0f}) mm ·
  path {beam['path_length_mm']} mm</div></div></div>
{_verdict_html(s['verdict'])}
<div class='grid2'>{_placement_card(meta, beam, along)}{_focus_card(foc, beam, meta)}</div>
<div class='grid2'>{_targeting_card(tm, name)}{_safety_card(mech, thermal)}</div>
{prefocal}
<div class='sec-title'>Beam path</div>
{_fig(s['fig_corridor'], "Oblique reslice along the beam axis (transducer on the left, "
      "focus marked). Tissue greyscale, intensity in colour, −6/−3 dB contours; dotted "
      "contours mark flagged structures the beam crosses.")}
{_fig(s['fig_along'], "Peak intensity sampled along the beam axis. A pre-focal peak "
      "approaching the focal peak indicates significant off-target deposition.")}
<div class='sec-title'>Focus &amp; targeting</div>
<div style='margin-bottom:8px'>
 <span class='pill'><span class='dot' style='background:{COL_6DB}'></span>−6 dB</span>
 <span class='pill'><span class='dot' style='background:{COL_3DB}'></span>−3 dB</span>
 <span class='pill'><span class='dot' style='background:{COL_TARGET}'></span>target</span>{pills}</div>
{_fig(s['fig_ortho'], "Orthogonal slices through the intensity peak with focus and "
      "target contours.")}
{_fig(s['fig_temp'], "Maximum temperature rise (ΔT above 37 °C baseline).") if s.get('fig_temp') else ""}
{region_tbl}
"""


def generate_html(report: Dict, cfg: Dict, out_path: str):
    date = datetime.datetime.now().strftime("%d %b %Y · %H:%M")
    blocks = "".join(_sonication_block(s, cfg) for s in report["sonications"])
    cmp_tbl = _comparison_table(report["sonications"], cfg["target_name"])
    src = report.get("tissue_source", "")
    op = cfg.get("operator", ""); notes = cfg.get("notes", "")
    html = f"""<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>tFUS Report — {cfg['subject_id']}</title><style>{_CSS}</style></head><body>
<div class='head'><div class='inner'>
 <div class='tag'>TUNeS · tFUS placement report</div>
 <h1>Acoustic simulation &amp; placement decision support</h1>
 <div class='sub'>{report['h5_name']}</div>
 <div class='meta'>
  <div><span class='k'>Subject</span><span class='v'>{cfg['subject_id']}</span></div>
  <div><span class='k'>Session</span><span class='v'>{cfg['session_id']}</span></div>
  <div><span class='k'>Target</span><span class='v'>{cfg['target_name']}</span></div>
  <div><span class='k'>Sonications</span><span class='v'>{len(report['sonications'])}</span></div>
  <div><span class='k'>Tissue model</span><span class='v'>{src}</span></div>
  <div><span class='k'>Generated</span><span class='v'>{date}</span></div>
  {f"<div><span class='k'>Operator</span><span class='v'>{op}</span></div>" if op else ""}
 </div>
 {f"<div class='note' style='margin-top:12px'><b style='color:var(--tx)'>Notes:</b> {notes}</div>" if notes else ""}
</div></div>
<div class='wrap'>
{cmp_tbl}
{blocks}
<div class='foot'>TUNeS FUS Report Generator v3.0 · safety levels per ITRUSST 2024
(Aubry et al.; Martin et al., Brain Stimul) · {date}</div>
</div></body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [html] {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 — PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def process_h5(h5_path: str, cfg: Dict) -> Dict:
    h5_name = os.path.splitext(os.path.basename(h5_path))[0]
    n_son = count_sonications(h5_path)
    print(f"\n=== {h5_name} — {n_son} sonication(s) ===")

    ref = load_h5_data(h5_path, 1)
    target_mask = load_binary_mask(cfg.get("target_mask_path"),
                                   ref["affine_sim"], ref["grid_shape"])
    region_masks = load_all_region_masks(cfg.get("region_mask_folder"),
                                         ref["affine_sim"], ref["grid_shape"])
    print(f"  target mask: {'loaded' if target_mask is not None else 'none'} | "
          f"regions: {len(region_masks)}")

    sonications, tissue_source = [], ""
    isppa_thr = cfg["isppa_overlay_threshold"]

    for i in range(1, n_son + 1):
        d = load_h5_data(h5_path, i)
        meta, vox_mm, vox_vol = d["meta"], d["voxel_mm"], d["voxel_vol"]
        tmasks = build_tissue_masks(cfg.get("simnibs_path"), d["affine_sim"],
                                    d["grid_shape"], d["medium_mask"])
        tissue_source = tmasks.get("_source", "")
        brain = get_brain_mask(tmasks, d["grid_shape"])

        foc = compute_intensity_focus(d["pressure_pa"], brain, vox_vol)
        beam_dim = compute_beam_dimensions(foc["I_brain"], foc["peak_idx"],
                                           d["affine_sim"], vox_mm)
        focus_mm = beam_dim["focus_center_mm"]          # actual intensity peak
        # Anchor the beam axis on the planned target (the area of interest) so
        # "pre-focal" = between transducer and target. Fall back to the peak if
        # the H5 carries no usable target position.
        target_mm = meta.get("target_mm")
        if target_mm is None or np.any(np.isnan(np.ravel(target_mm))):
            target_mm = focus_mm
        beam = compute_beam_geometry(meta, target_mm, focus_mm, d["medium_mask"],
                                     d["affine_sim"], vox_mm)
        beam.update(beam_dim)  # focus_center_mm
        beam.update(compute_beam_fwhm(foc["I_full"], beam, d["affine_sim"], vox_mm))

        tm = (compute_targeting_metrics(target_mask, foc["I_full"], foc["I_brain"],
              foc["mask_6dB"], foc["mask_3dB"], vox_vol)
              if target_mask is not None else {"error": "No primary target mask."})

        mech = compute_mechanical_safety(tmasks, d["pressure_pa"], foc["I_full"], meta["freq_hz"])
        thermal = compute_thermal_safety(d["temp_c"], tmasks, meta)

        rdata = (compute_all_region_metrics(region_masks, foc["I_full"], foc["I_brain"],
                 foc["mask_6dB"], foc["mask_3dB"], vox_vol, isppa_thr, cfg["target_name"])
                 if region_masks else {"rows": [], "hot_regions": {}, "hot_colors": {}})
        if rdata["rows"]:
            classify_regions_along_beam(rdata["rows"], region_masks, beam,
                                        d["affine_sim"], focus_mm)
        along = sample_along_beam(foc["I_full"], beam, d["affine_sim"], vox_mm,
                                  brain_mask=brain, focus_mask=foc["mask_6dB"])
        verdict = build_verdict(tm, beam, along, mech, thermal, cfg)

        print(f"  son {i}: verdict={verdict['level']:7s} "
              f"cov={verdict['coverage']}% prefocal={int(verdict['prefocal_ratio']*100)}% "
              f"Isppa={foc['Isppa']} ΔT={thermal.get('dT_peak')}")

        # figures
        fig_ortho = plot_orthogonal_views(foc["I_brain"], foc["mask_6dB"], foc["mask_3dB"],
                    target_mask, d["medium_mask"], d["affine_sim"], d["grid_shape"],
                    foc["peak_idx"], cfg.get("t1_path"),
                    rdata["hot_regions"], rdata["hot_colors"])
        fig_corr = plot_beam_corridor(foc["I_full"], tmasks, beam, d["affine_sim"],
                    d["grid_shape"], vox_mm, isppa_thr, rdata["hot_regions"], rdata["hot_colors"])
        fig_along = plot_along_beam(along, isppa_thr)
        fig_temp = plot_temperature(d["temp_c"], foc["peak_idx"], d["affine_sim"], d["grid_shape"])

        max_mi = max((r["MI"] for r in mech), default=float("nan"))
        sonications.append({
            "idx": i, "meta": meta, "foc": foc, "beam": beam, "tm": tm,
            "mech": mech, "thermal": thermal, "along": along, "verdict": verdict,
            "region_metrics": rdata["rows"], "hot_colors": rdata["hot_colors"],
            "fig_ortho": fig_ortho, "fig_corridor": fig_corr,
            "fig_along": fig_along, "fig_temp": fig_temp,
            "Isppa_target": tm.get("Isppa_target", foc["Isppa"]) if "error" not in tm else foc["Isppa"],
            "max_MI": round(max_mi, 2), "dT": thermal.get("dT_peak", float("nan")),
        })

    return {"h5_name": h5_name, "h5_path": h5_path,
            "sonications": sonications, "tissue_source": tissue_source}


def write_summary_csv(report: Dict, cfg: Dict, out_path: str):
    """One row per sonication with the key decision metrics (for cross-subject
    analysis). Written only if pandas is available."""
    if not _HAVE_PANDAS:
        return
    rows = []
    for s in report["sonications"]:
        meta, beam, tm = s["meta"], s["beam"], s["tm"]
        a, th, v = s["along"], s["thermal"], s["verdict"]
        rows.append({
            "sonication": s["idx"], "verdict": v["level"],
            "freq_kHz": meta["freq_kHz"],
            "target_x_mm": round(float(meta["target_mm"][0]), 1),
            "target_y_mm": round(float(meta["target_mm"][1]), 1),
            "target_z_mm": round(float(meta["target_mm"][2]), 1),
            "beam_path_mm": beam["path_length_mm"],
            "targeting_error_mm": round(float(np.linalg.norm(
                np.asarray(beam.get("focus_center_mm")) - np.asarray(meta["target_mm"]))), 1),
            "skull_thickness_mm": beam.get("skull_thickness_mm"),
            "fwhm_axial_mm": beam.get("fwhm_axial_mm"),
            "fwhm_lateral_mm": beam.get("fwhm_lat_mean_mm"),
            "Isppa_global_Wcm2": s["foc"]["Isppa"],
            "Isppa_brain_Wcm2": s["foc"]["Isppa_brain"],
            "Isppa_target_Wcm2": tm.get("Isppa_target") if "error" not in tm else None,
            "coverage_6dB_pct": tm.get("coverage_6dB_pct") if "error" not in tm else None,
            "on_target_6dB_pct": tm.get("on_target_6dB_pct") if "error" not in tm else None,
            "prefocal_brain_peak_Wcm2": a.get("prefocal_brain_peak"),
            "prefocal_ratio": a.get("prefocal_ratio"),
            "max_MI": s["max_MI"],
            "peak_temperature_C": th.get("T_peak"),
            "peak_dT_C": th.get("dT_peak"),
            "thermal_within_itrusst": th.get("thermal_ok"),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  [csv]  {out_path}")


def run():
    cfg = CONFIG
    os.makedirs(cfg["output_dir"], exist_ok=True)
    for h5_path in cfg["h5_files"]:
        if not os.path.exists(h5_path):
            print(f"[skip] not found: {h5_path}"); continue
        report = process_h5(h5_path, cfg)
        out = os.path.join(cfg["output_dir"], f"{report['h5_name']}_report.html")
        generate_html(report, cfg, out)
        write_summary_csv(report, cfg,
                          os.path.join(cfg["output_dir"], f"{report['h5_name']}_summary.csv"))
    print("\nDone.")


if __name__ == "__main__":
    run()