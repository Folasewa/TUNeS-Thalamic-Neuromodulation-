#!/usr/bin/env python3
"""
================================================================================
TUNeS Report Generator  v4.0
================================================================================

A decision-support tool for choosing the final transducer placement in
transcranial ultrasound stimulation (TUS / tFUS) studies. Post-processes
kPlan acoustic simulation outputs (.h5) and produces a single, self-contained
HTML report per plan, designed so an operator can compare candidate placements
(sonications) and judge efficacy + safety at a glance.


──────────────────────────────────────
  • Beam axis reconstruction  (transducer pivot → planned target)
  • Pre-focal brain-peak detection + ratio vs focal peak
  • Beam-aligned FWHM (axial & lateral) + elongation
  • Per-plan targeting-error (planned target vs actual intensity peak)
  • Skull thickness on-axis
  • BeamZone column per region  (pre-focal / at-focus / post-focal)
  • AlongFocus_mm + LateralOffset_mm per region
  • CEM43 ITRUSST-tiered thermal flags (brain 2 / bone 16 / skin 21 min)
  • Eye-specific MI limit (0.4) retained from Script 2
  • Per-region peak/mean pressure in −6/−3 dB overlap zones
  • Per-region peak/mean temperature in −6 dB overlap zone
  • XYZ mm of peak pressure voxel per region
  • Self-contained HTML report with dark design system (no external deps)
  • Comparison table across sonications in the HTML report

Inputs
──────
  • kPlan RESULTS.h5                              (required)
  • SimNIBS final_tissues.nii.gz                 (optional)
  • Participant nucleus masks folder             (optional)
  • Individualised cortical/subcortical masks    (optional)
  • T1.nii.gz                                     (optional)

Outputs
───────
  • <plan>_report.html     self-contained decision-support report
  • <plan>_analysis.csv    flat per-region metrics table

Author: merged & extended from TUNeS pipeline scripts
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
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import map_coordinates

try:
    import pandas as pd
    _HAVE_PANDAS = True
except ImportError:
    _HAVE_PANDAS = False

try:
    import nibabel as nib
    from nilearn.image import resample_img
    _HAVE_NIBABEL = True
except ImportError:
    _HAVE_NIBABEL = False


# ════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION  — edit before running
# ════════════════════════════════════════════════════════════════════════════

CONFIG: Dict = {
    "h5_files": [
        "/Users/laptop/Documents/TUNeS/Alina/Simulations/PLAN-46-311-RESULTS.h5",
    ],
    "simnibs_path":      "/Users/laptop/Documents/TUNeS/Alina/final_tissues.nii.gz",
    "target_mask_path":  "/Users/laptop/Documents/TUNeS/Alina/participant_nuclei/harvardoxford-subcortical_prob_Left_Thalamus.nii.gz",
    "target_name":       "Left_Thalamus",
    "region_mask_folder":"/Users/laptop/Documents/TUNeS/Alina/participant_nuclei",
    "atlas_mask_folder": "/Users/laptop/Documents/TUNeS/Alina/ho_masks",
    "t1_path":           "/Users/laptop/Documents/TUNeS/Alina/T1.nii.gz",
    "output_dir":        "/Users/laptop/Documents/TUNeS/Alina/Output",

    "participant_id": "Alina",
    "session_id":     "thal",
    "operator":       "",
    "notes":          "",

    # Intensity threshold (W/cm²) — regions above this are flagged & contoured
    "isppa_overlay_threshold": 0.5,

    # Pre-focal flag: fire if pre-target brain peak ≥ this fraction of focal peak
    "prefocal_ratio_flag": 0.5,
}


# ════════════════════════════════════════════════════════════════════════════
# PHYSICS & SAFETY CONSTANTS  (ITRUSST 2024)
# ════════════════════════════════════════════════════════════════════════════

RHO_C             = 1.5e6    # Pa·s/m — acoustic impedance of soft tissue
BASELINE_T        = 37.0     # °C

MI_LIMIT_BRAIN    = 1.9
MI_LIMIT_EYES     = 0.4
T_RISE_LIMIT      = 2.0      # °C
T_ABS_LIMIT       = 39.0     # °C
CEM43_LIMIT_BRAIN = 2.0      # min
CEM43_LIMIT_BONE  = 16.0
CEM43_LIMIT_SKIN  = 21.0

TISSUE_GROUPS: Dict[str, List[int]] = {
    "Brain (GM + WM + CSF)": [1, 2, 3],
    "Skull":                 [7, 8],
    "Scalp":                 [5],
    "Eyes":                  [6],
}

# Colour palette (dark theme)
CMAP_INT   = "inferno"
CMAP_BG    = "gray"
CMAP_TEMP  = "hot"
COL_6DB    = "#22d3ee"
COL_3DB    = "#fde047"
COL_TARGET = "#4ade80"
COL_AXIS   = "#f472b6"
COL_FOCUS  = "#ffffff"

HOT_REGION_PALETTE = [
    "#ff6b6b","#a78bfa","#fb923c","#f472b6","#38bdf8",
    "#e17055","#fb7185","#818cf8","#cbd5e1","#f43f5e",
    "#60a5fa","#34d399",
]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def build_affine(voxel_mm: float, origin_mm: np.ndarray) -> np.ndarray:
    A = np.zeros((4, 4))
    A[0,0] = A[1,1] = A[2,2] = voxel_mm
    A[:3, 3] = origin_mm
    A[3, 3]  = 1.0
    return A


def match_shape(data: np.ndarray, target: tuple) -> np.ndarray:
    out = np.zeros(target, dtype=data.dtype)
    s = tuple(min(a, b) for a, b in zip(data.shape, target))
    out[:s[0], :s[1], :s[2]] = data[:s[0], :s[1], :s[2]]
    return out


def voxel_to_mm(idx, affine: np.ndarray) -> np.ndarray:
    idx = np.asarray(idx, float)
    return (affine @ np.array([idx[0], idx[1], idx[2], 1.0]))[:3]


def mm_to_voxel(mm, affine: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(affine)
    mm  = np.asarray(mm, float)
    return (inv @ np.array([mm[0], mm[1], mm[2], 1.0]))[:3]


def fig_to_b64(fig, dpi: int = 135) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    out = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return out


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — H5 LOADING
# ════════════════════════════════════════════════════════════════════════════

def _scaled_field(ds) -> np.ndarray:
    slope  = float(ds.attrs["scale_slope"].ravel()[0])
    intcpt = float(ds.attrs["scale_intercept"].ravel()[0])
    return ds[:].astype(np.float32) * slope + intcpt


def _scalar(f, path, default=np.nan):
    try:
        return float(np.ravel(f[path][:])[0])
    except Exception:
        return default


def count_sonications(h5_path: str) -> int:
    with h5py.File(h5_path, "r") as f:
        return len([k for k in f["sonications"].keys() if k.isdigit()])


def load_h5_data(h5_path: str, son_idx: int) -> Dict:
    """Load pressure, temperature, grid geometry, transducer geometry and
    all kPlan scalar outputs for one sonication."""
    with h5py.File(h5_path, "r") as f:
        sf = f"sonications/{son_idx}/simulated_field"
        pk = f"sonications/{son_idx}/sonication_parameters"

        # ── Pressure (Pa) ─────────────────────────────────────────────
        p_pa = np.transpose(_scaled_field(f[f"{sf}/pressure_amplitude"]))

        # ── Temperature (°C) ──────────────────────────────────────────
        temp_c = None
        if f"{sf}/temperature_maximum" in f:
            temp_c = np.transpose(_scaled_field(f[f"{sf}/temperature_maximum"]))

        # ── Grid ──────────────────────────────────────────────────────
        mm_ds    = f["medium_properties/medium_mask"]
        dx_mm    = float(mm_ds.attrs["grid_spacing"].ravel()[0]) * 1e3
        origin   = f["settings/grid/domain_position"][:].ravel()[:3] * 1e3
        affine   = build_affine(dx_mm, origin)
        shape    = p_pa.shape
        med_mask = match_shape(np.transpose(mm_ds[:]), shape)

        try:
            med_labels = [x.decode() if isinstance(x, bytes) else str(x)
                          for x in np.ravel(f["medium_properties/medium_mask_labels"][:])]
        except Exception:
            med_labels = ["background", "head", "skull"]

        # ── Scalars ───────────────────────────────────────────────────
        target_mm  = np.ravel(f[f"{pk}/target_position"][:])[:3] * 1e3
        freq_hz    = _scalar(f, f"{pk}/driving_frequency")
        focal_d    = _scalar(f, f"{pk}/focal_distance") * 1e3
        tgt_p_pa   = _scalar(f, f"{pk}/target_pressure")
        sptp_pa    = _scalar(f, f"{sf}/pressure_amplitude_sptp")
        p_at_tgt   = _scalar(f, f"{sf}/pressure_amplitude_at_target")

        def _arr(path):
            try:
                return np.ravel(f[path][:])
            except Exception:
                return np.array([np.nan])

        sptp_msk   = _arr(f"{sf}/pressure_amplitude_sptp_masked")
        T_peak     = _scalar(f, f"{sf}/temperature_at_peak")
        T_target   = _scalar(f, f"{sf}/temperature_at_target")
        T_msk      = _arr(f"{sf}/temperature_maximum_sptp_masked")
        dose_tgt   = _scalar(f, f"{sf}/thermal_dose_at_target")
        dose_sptp  = _scalar(f, f"{sf}/thermal_dose_sptp")
        dose_msk   = _arr(f"{sf}/thermal_dose_sptp_masked")

        pst  = _arr(f"{pk}/pulse_sequence_timing")
        cool = _scalar(f, f"{pk}/pulse_sequence_cooling_time")

        # ── Transducer geometry ───────────────────────────────────────
        try:
            pos_xform = np.ravel(f[f"{pk}/position_transform"][:])
            pos_xform = (pos_xform.reshape(4,4) if pos_xform.size == 16
                         else np.eye(4))
        except Exception:
            pos_xform = np.eye(4)
        transducer_mm = pos_xform[3, :3] * 1e3

        n_elements = 0
        try:
            n_elements = len([k for k in f["transducer/elements"].keys()
                              if k.isdigit()])
        except Exception:
            pass

        meta = {
            "target_mm":       target_mm,
            "freq_hz":         freq_hz,
            "freq_kHz":        round(freq_hz / 1e3, 1),
            "focal_dist_mm":   round(focal_d, 1),
            "tgt_p_kPa":       round(tgt_p_pa / 1e3, 1),
            "sptp_kPa":        round(sptp_pa / 1e3, 1),
            "p_at_target_kPa": round(p_at_tgt / 1e3, 1),
            "sptp_masked":     sptp_msk,
            "med_labels":      med_labels,
            "T_peak":          T_peak,
            "T_target":        T_target,
            "T_masked":        T_msk,
            "dose_target":     dose_tgt,
            "dose_sptp":       dose_sptp,
            "dose_masked":     dose_msk,
            "pulse_timing":    pst,
            "cooling_time":    cool,
            "transducer_mm":   transducer_mm,
            "pos_xform":       pos_xform,
            "n_elements":      n_elements,
        }

    return {
        "pressure_pa":  p_pa,
        "temp_c":       temp_c,
        "affine_sim":   affine,
        "grid_shape":   shape,
        "medium_mask":  med_mask,
        "voxel_mm":     dx_mm,
        "voxel_vol":    dx_mm ** 3,
        "meta":         meta,
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MASKS
# ════════════════════════════════════════════════════════════════════════════

def _resample(img, affine_sim, shape, interp="nearest"):
    res = resample_img(img, target_affine=affine_sim,
                       target_shape=shape, interpolation=interp)
    return np.squeeze(res.get_fdata())


def build_tissue_masks(seg_path, affine_sim, shape, medium_mask) -> Dict:
    if seg_path and _HAVE_NIBABEL and os.path.exists(seg_path):
        seg  = nib.load(seg_path)
        data = np.squeeze(seg.get_fdata()).astype(np.int16)
        masks = {}
        for name, labels in TISSUE_GROUPS.items():
            m  = np.isin(data, labels).astype(np.uint8)
            rs = _resample(nib.Nifti1Image(m, seg.affine), affine_sim, shape)
            masks[name] = match_shape((rs > 0.5).astype(np.uint8), shape)
        masks["_source"] = "SimNIBS"
        return masks
    # Fallback
    return {
        "Brain (head proxy)": (medium_mask == 1).astype(np.uint8),
        "Skull":              (medium_mask == 2).astype(np.uint8),
        "_source": "kPlan medium mask",
    }


def get_brain_mask(tissue_masks, shape) -> np.ndarray:
    for k in ("Brain (GM + WM + CSF)", "Brain (head proxy)"):
        if k in tissue_masks:
            return tissue_masks[k]
    return np.zeros(shape, np.uint8)


def load_mask(path, affine_sim, shape) -> Optional[np.ndarray]:
    if not (_HAVE_NIBABEL and path and os.path.exists(path)):
        return None
    rs = _resample(nib.load(path), affine_sim, shape)
    return match_shape((rs > 0).astype(np.uint8), shape)


def load_mask_folder(folder, affine_sim, shape,
                     prefix_strip=None) -> Dict[str, np.ndarray]:
    """Load every .nii / .nii.gz in *folder* into simulation grid space."""
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
        if prefix_strip:
            for ps in (prefix_strip if isinstance(prefix_strip, list) else [prefix_strip]):
                name = name.replace(ps, "")
        try:
            rs = _resample(nib.load(p), affine_sim, shape)
            m  = match_shape((rs > 0).astype(np.uint8), shape)
        except Exception as e:
            print(f"  [warn] {p}: {e}"); continue
        if m.sum() > 0:
            out[name] = m
    return out


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — INTENSITY & FOCAL ZONES
# ════════════════════════════════════════════════════════════════════════════

def compute_intensity(pressure_pa, brain_mask, voxel_vol) -> Dict:
    I_full  = (pressure_pa ** 2) / (2.0 * RHO_C) / 1e4        # W/cm²
    I_brain = I_full.copy()
    if brain_mask.sum() > 0:
        I_brain[brain_mask == 0] = 0.0
        ref = float(I_brain[brain_mask == 1].max())
    else:
        ref = float(I_full.max())
        I_brain = I_full.copy()

    if ref <= 0:
        raise ValueError("Peak brain intensity is zero — check pressure scaling.")

    mask_6 = (I_brain > 0.25 * ref).astype(np.uint8)
    mask_3 = (I_brain > 0.50 * ref).astype(np.uint8)
    peak_idx = np.unravel_index(I_brain.argmax(), I_brain.shape)

    return {
        "I_full":        I_full,
        "I_brain":       I_brain,
        "mask_6dB":      mask_6,
        "mask_3dB":      mask_3,
        "Isppa":         round(float(I_full.max()), 4),
        "Isppa_brain":   round(ref, 4),
        "peak_idx":      peak_idx,
        "focus_vox_6":   int(mask_6.sum()),
        "focus_vox_3":   int(mask_3.sum()),
        "focus_vol_6dB": round(float(mask_6.sum()) * voxel_vol, 2),
        "focus_vol_3dB": round(float(mask_3.sum()) * voxel_vol, 2),
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — BEAM GEOMETRY & PRE-FOCAL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def compute_beam_geometry(meta: Dict, target_mm, peak_mm,
                          medium_mask, affine_sim, voxel_mm) -> Dict:
    """
    Reconstruct beam axis (transducer → planned target).
    The planned target anchors "pre-focal" so that a near-field hotspot
    is correctly reported as pre-focal rather than redefining the focus.
    """
    tx  = meta["transducer_mm"]
    ref = np.asarray(target_mm if target_mm is not None else peak_mm, float)

    vec  = ref - tx
    L    = float(np.linalg.norm(vec))
    axis = vec / L if L > 0 else np.array([0., 0., 1.])
    target_proj = float(np.dot(ref - tx, axis))

    # Actual intensity peak on the same axis
    pk_rel       = np.asarray(peak_mm, float) - tx
    peak_proj    = float(np.dot(pk_rel, axis))
    peak_lat     = float(np.linalg.norm(pk_rel - peak_proj * axis))
    peak_axial_offset = round(peak_proj - target_proj, 1)

    # Walk axis to find skull entry/exit → thickness
    steps = np.arange(0, L + 25, voxel_mm * 0.5)
    in_skull = False
    skull_enter = skull_exit = None
    entry_mm = entry_proj = None
    for s in steps:
        pt = tx + s * axis
        vx = np.round(mm_to_voxel(pt, affine_sim)).astype(int)
        if np.any(vx < 0) or np.any(vx >= np.array(medium_mask.shape)):
            continue
        val = medium_mask[vx[0], vx[1], vx[2]]
        if val == 2 and not in_skull:
            in_skull = True
            skull_enter = s
            if entry_mm is None:
                entry_mm, entry_proj = pt.copy(), s
        elif val != 2 and in_skull:
            in_skull = False
            skull_exit = s
    skull_thickness = (round(skull_exit - skull_enter, 1)
                       if skull_enter is not None and skull_exit is not None
                       else np.nan)
    return {
        "transducer_mm":       tx,
        "axis":                axis,
        "path_length_mm":      round(L, 1),
        "focus_proj":          target_proj,
        "target_proj":         target_proj,
        "target_mm":           ref,
        "peak_mm":             np.asarray(peak_mm, float),
        "peak_proj":           peak_proj,
        "peak_lateral_mm":     round(peak_lat, 1),
        "peak_axial_offset_mm":peak_axial_offset,
        "entry_mm":            entry_mm,
        "entry_proj":          entry_proj,
        "skull_thickness_mm":  skull_thickness,
    }


def compute_beam_fwhm(I_field, beam: Dict, affine_sim, voxel_mm) -> Dict:
    """Beam-aligned FWHM through the actual intensity peak (oblique interpolation)."""
    peak_mm = beam.get("peak_mm", beam.get("focus_center_mm"))
    axis = beam["axis"]
    up   = np.eye(3)[np.argmin(np.abs(axis))]
    v1   = up - np.dot(up, axis) * axis; v1 /= (np.linalg.norm(v1) + 1e-9)
    v2   = np.cross(axis, v1)

    def _profile(direction, span=20.0):
        rs  = np.arange(-span, span + 1e-3, voxel_mm * 0.5)
        pts = np.array([peak_mm + r * direction for r in rs])
        vox = np.array([mm_to_voxel(p, affine_sim) for p in pts]).T
        vals = map_coordinates(I_field, vox, order=1, mode="constant", cval=0.)
        return rs, vals

    fwhms = {}
    for name, d in [("axial", axis), ("lat1", v1), ("lat2", v2)]:
        _, vals = _profile(d)
        peakv   = float(vals.max())
        if peakv <= 0:
            fwhms[name] = float("nan"); continue
        above = vals >= 0.5 * peakv
        idx   = np.where(above)[0]
        fwhms[name] = round((idx[-1] - idx[0] + 1) * (voxel_mm * 0.5), 2) if idx.size else float("nan")

    lat = np.nanmean([fwhms["lat1"], fwhms["lat2"]])
    return {
        "fwhm_axial_mm":    fwhms["axial"],
        "fwhm_lat1_mm":     fwhms["lat1"],
        "fwhm_lat2_mm":     fwhms["lat2"],
        "fwhm_lat_mean_mm": round(float(lat), 2) if not np.isnan(lat) else float("nan"),
        "elongation":       (round(fwhms["axial"] / lat, 2)
                             if (not np.isnan(fwhms["axial"]) and lat > 0)
                             else float("nan")),
    }


def sample_along_beam(I_full, beam: Dict, affine_sim, voxel_mm,
                      brain_mask=None, focus_mask=None,
                      radius_mm: float = 3.0, extend_mm: float = 15.0) -> Dict:
    """
    Max intensity in a disc perpendicular to the beam axis at each step.
    Separates pre-focal brain peak from any-tissue peak for the verdict.
    """
    tx    = beam["transducer_mm"]; axis = beam["axis"]
    L     = beam["path_length_mm"]; fproj = beam["focus_proj"]
    up    = np.eye(3)[np.argmin(np.abs(axis))]
    v1    = up - np.dot(up, axis) * axis; v1 /= (np.linalg.norm(v1) + 1e-9)
    v2    = np.cross(axis, v1)

    s_vals = np.arange(0, L + extend_mm, voxel_mm * 0.5)
    rr     = np.arange(-radius_mm, radius_mm + 1e-3, voxel_mm)
    prof   = np.zeros_like(s_vals)
    in_brain = np.zeros_like(s_vals, dtype=bool)
    in_focus = np.zeros_like(s_vals, dtype=bool)

    for i, s in enumerate(s_vals):
        pts = [tx + s * axis + a * v1 + b * v2
               for a in rr for b in rr if a*a + b*b <= radius_mm**2]
        pts = np.array(pts)
        vox = np.array([mm_to_voxel(p, affine_sim) for p in pts]).T
        prof[i] = float(map_coordinates(I_full, vox, order=1,
                                        mode="constant", cval=0.).max())
        if brain_mask is not None:
            in_brain[i] = map_coordinates(
                brain_mask.astype(np.float32), vox,
                order=0, mode="constant", cval=0.).max() > 0.5
        if focus_mask is not None:
            in_focus[i] = map_coordinates(
                focus_mask.astype(np.float32), vox,
                order=0, mode="constant", cval=0.).max() > 0.5

    focal_peak = float(prof.max())
    focus_ref  = float(prof[in_focus].max()) if in_focus.any() else focal_peak
    ratio_ref  = max(focus_ref, 1e-9)

    before = s_vals < fproj
    pre    = before & (~in_focus)

    def _peak(mask):
        if not mask.any():
            return 0., float("nan")
        sub = np.where(mask, prof, -1)
        j   = int(sub.argmax())
        return float(prof[j]), float(s_vals[j])

    pf_peak,  pf_pos  = _peak(pre)
    pf_brain, pf_bpos = _peak(pre & in_brain)
    brain_entry = float(s_vals[in_brain][0]) if in_brain.any() else None

    return {
        "s_mm":                   s_vals,
        "profile":                prof,
        "in_brain":               in_brain,
        "in_focus":               in_focus,
        "focus_proj":             fproj,
        "peak_proj":              beam.get("peak_proj"),
        "entry_proj":             beam.get("entry_proj"),
        "brain_entry_proj":       brain_entry,
        "prefocal_peak":          round(pf_peak, 4),
        "prefocal_peak_pos_mm":   round(pf_pos, 1),
        "prefocal_brain_peak":    round(pf_brain, 4),
        "prefocal_brain_pos_mm":  round(pf_bpos, 1),
        "focus_ref":              round(focus_ref, 4),
        "focal_peak":             round(focal_peak, 4),
        "prefocal_ratio":         round(pf_brain / ratio_ref, 4),
        "prefocal_ratio_anytissue": round(pf_peak / ratio_ref, 4),
    }


def classify_regions_along_beam(region_rows: List[Dict],
                                region_masks: Dict,
                                beam: Dict,
                                affine_sim,
                                margin_mm: float = 5.0):
    """Annotate each region row in-place with BeamZone / AlongFocus_mm / LateralOffset_mm."""
    tx    = beam["transducer_mm"]
    axis  = beam["axis"]
    fproj = beam["focus_proj"]
    for r in region_rows:
        m = region_masks.get(r["name"])
        if m is None or m.sum() == 0:
            r["beam_zone"] = "—"; r["along_focus_mm"] = np.nan; r["lateral_mm"] = np.nan
            continue
        idx    = np.argwhere(m == 1)
        cen    = voxel_to_mm(idx.mean(0), affine_sim)
        rel    = cen - tx
        proj   = float(np.dot(rel, axis))
        lateral = float(np.linalg.norm(rel - proj * axis))
        d      = proj - fproj
        zone   = ("pre-focal" if d < -margin_mm
                  else "post-focal" if d > margin_mm
                  else "at focus")
        r["beam_zone"]      = zone
        r["along_focus_mm"] = round(d, 1)
        r["lateral_mm"]     = round(lateral, 1)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PER-REGION METRICS  (tiered CSV rows)
# ════════════════════════════════════════════════════════════════════════════

def _cem43_limit(name: str) -> float:
    n = name.lower()
    if any(x in n for x in ("skull", "bone", "compact", "spongy")):
        return CEM43_LIMIT_BONE
    if any(x in n for x in ("scalp", "skin")):
        return CEM43_LIMIT_SKIN
    return CEM43_LIMIT_BRAIN


def make_row(region_mask, target_name, target_path, tier_label,
             son_i, pressure_pa, intensity_full, mask_6dB, mask_3dB,
             peak_int_overall, peak_int_brain,
             focus_vox_6, focus_vox_3, focus_vol_6dB, focus_vol_3dB,
             voxel_vol, frequency_hz, affine_sim,
             temp_c=None, meta=None,
             is_eye=False,
             beam_zone="—", along_focus_mm=np.nan, lateral_mm=np.nan) -> Optional[Dict]:
    """Unified metrics row for any binary region mask."""
    tgt_vox = int(region_mask.sum())
    if tgt_vox == 0:
        return None

    tgt_vol  = round(tgt_vox * voxel_vol, 2)
    is_skull = "skull" in target_name.lower()

    # Intensity
    I_in       = intensity_full[region_mask == 1]
    Isppa_tgt  = round(float(I_in.max()), 4)
    Imean_tgt  = round(float(I_in.mean()), 4)

    # Pressure
    p_in     = pressure_pa[region_mask == 1]
    peak_pa  = float(p_in.max())
    peak_kpa = round(peak_pa / 1e3, 3)
    mean_kpa = "N/A" if is_skull else round(float(p_in.mean()) / 1e3, 3)

    # Focal overlaps
    ov6 = int((mask_6dB * region_mask).sum())
    ov3 = int((mask_3dB * region_mask).sum())
    cov6_mm3 = round(ov6 * voxel_vol, 2)
    cov3_mm3 = round(ov3 * voxel_vol, 2)
    cov6_pct = round(ov6 / tgt_vox * 100, 2) if tgt_vox else np.nan
    cov3_pct = round(ov3 / tgt_vox * 100, 2) if tgt_vox else np.nan
    on6  = round(ov6 / focus_vox_6 * 100, 2) if focus_vox_6 else np.nan
    on3  = round(ov3 / focus_vox_3 * 100, 2) if focus_vox_3 else np.nan
    off6 = round(100 - on6, 2) if not np.isnan(on6) else np.nan
    off3 = round(100 - on3, 2) if not np.isnan(on3) else np.nan

    # Mean intensity in overlap
    mean_int_ov6 = (round(float((intensity_full * mask_6dB * region_mask).sum() / ov6), 4)
                    if ov6 else np.nan)
    mean_int_ov3 = (round(float((intensity_full * mask_3dB * region_mask).sum() / ov3), 4)
                    if ov3 else np.nan)

    # Pressure in overlap zones
    ov6m = (mask_6dB == 1) & (region_mask == 1)
    ov3m = (mask_3dB == 1) & (region_mask == 1)
    if ov6m.any():
        p6 = pressure_pa[ov6m]
        pk_ov6  = round(float(p6.max())  / 1e3, 3)
        mn_ov6  = round(float(p6.mean()) / 1e3, 3)
    else:
        pk_ov6 = mn_ov6 = np.nan
    if ov3m.any():
        p3 = pressure_pa[ov3m]
        pk_ov3  = round(float(p3.max())  / 1e3, 3)
        mn_ov3  = round(float(p3.mean()) / 1e3, 3)
    else:
        pk_ov3 = mn_ov3 = np.nan

    # XYZ of peak pressure voxel
    masked_p = pressure_pa.copy(); masked_p[region_mask == 0] = 0.
    pidx     = np.unravel_index(masked_p.argmax(), masked_p.shape)
    peak_xyz = voxel_to_mm(pidx, affine_sim)
    peak_xyz_str = f"({peak_xyz[0]:.1f}, {peak_xyz[1]:.1f}, {peak_xyz[2]:.1f})"

    # Mechanical Index — eye limit vs brain limit
    freq_mhz = frequency_hz / 1e6
    mi       = round((peak_pa / 1e6) / np.sqrt(freq_mhz), 4)
    mi_limit = MI_LIMIT_EYES if is_eye else MI_LIMIT_BRAIN
    mi_flag  = "EXCEEDS LIMIT" if mi > mi_limit else "OK"

    # Thermal — per-region from 3D field
    if temp_c is not None:
        t_reg       = temp_c[region_mask == 1]
        pk_temp_reg = round(float(t_reg.max()), 3)
        mn_temp_reg = round(float(t_reg.mean()), 3)
        if ov6m.any():
            t6          = temp_c[ov6m]
            pk_temp_ov6 = round(float(t6.max()), 3)
            mn_temp_ov6 = round(float(t6.mean()), 3)
        else:
            pk_temp_ov6 = mn_temp_ov6 = np.nan
        dT_reg = pk_temp_reg - BASELINE_T
    else:
        pk_temp_reg = mn_temp_reg = pk_temp_ov6 = mn_temp_ov6 = np.nan
        dT_reg      = np.nan

    # Thermal flag — ITRUSST CEM43 tiered limits
    cem43_limits = _cem43_limit(target_name)
    dose_msk = np.ravel(meta.get("dose_masked", [])) if meta else np.array([])
    med_labels = meta.get("med_labels", []) if meta else []
    # Try to find CEM43 for this tissue type from kPlan scalars
    cem43_reg = np.nan
    for i, lab in enumerate(med_labels):
        if i < dose_msk.size and lab.lower() != "background":
            if any(x in target_name.lower() for x in [lab.lower(), lab[:4].lower()]):
                cem43_reg = float(dose_msk[i])
    # Primary thermal verdict
    dT_ok = (not np.isnan(dT_reg)) and ((dT_reg <= T_RISE_LIMIT) or (pk_temp_reg <= T_ABS_LIMIT))
    cem_ok = np.isnan(cem43_reg) or (cem43_reg <= cem43_limits)
    if np.isnan(dT_reg):
        thermal_flag = "NO DATA"
    elif pk_temp_reg >= 43.0:
        thermal_flag = "DANGER"
    elif pk_temp_reg >= T_ABS_LIMIT or (not dT_ok) or (not cem_ok):
        thermal_flag = "CAUTION"
    else:
        thermal_flag = "OK"

    # kPlan scalar summaries (same value repeated per row for convenience)
    def _ms(key):
        v = meta.get(key, np.nan) if meta else np.nan
        return round(float(v), 4) if not np.isnan(float(v) if v is not None else np.nan) else np.nan

    return {
        # Identifiers
        "Sonication":               son_i,
        "TargetName":               target_name,
        "TargetPath":               target_path,
        "ReportingTier":            tier_label,
        # Beam position (new in v4)
        "BeamZone":                 beam_zone,
        "AlongFocus_mm":            along_focus_mm,
        "LateralOffset_mm":         lateral_mm,
        # Global reference
        "Isppa_Overall_Wcm2":       round(peak_int_overall, 4),
        "Isppa_Brain_Wcm2":         round(peak_int_brain, 4),
        # Region intensity
        "Isppa_Target_Wcm2":        Isppa_tgt,
        "Imean_Target_Wcm2":        Imean_tgt,
        "TargetVol_mm3":            tgt_vol,
        # Focal zone volumes
        "FocusVol_6dB_mm3":         focus_vol_6dB,
        "FocusVol_3dB_mm3":         focus_vol_3dB,
        # Coverage
        "Coverage_6dB_mm3":         cov6_mm3,
        "Coverage_6dB_pct":         cov6_pct,
        "Coverage_3dB_mm3":         cov3_mm3,
        "Coverage_3dB_pct":         cov3_pct,
        "OnTarget_6dB_pct":         on6,
        "OffTarget_6dB_pct":        off6,
        "OnTarget_3dB_pct":         on3,
        "OffTarget_3dB_pct":        off3,
        "MeanInt_Overlap_6dB_Wcm2": mean_int_ov6,
        "MeanInt_Overlap_3dB_Wcm2": mean_int_ov3,
        # Pressure (new overlap columns)
        "PeakPressure_Target_kPa":  peak_kpa,
        "MeanPressure_Target_kPa":  mean_kpa,
        "PeakP_Overlap_6dB_kPa":    pk_ov6,
        "MeanP_Overlap_6dB_kPa":    mn_ov6,
        "PeakP_Overlap_3dB_kPa":    pk_ov3,
        "MeanP_Overlap_3dB_kPa":    mn_ov3,
        # Safety
        "MI":                       mi,
        "MI_Flag":                  mi_flag,
        # Spatial
        "PeakP_XYZ_mm":             peak_xyz_str,
        # Thermal — kPlan scalars
        "Temp_AtTarget_degC":       _ms("T_target"),
        "Temp_AtPeak_degC":         _ms("T_peak"),
        "CEM43_AtTarget_min":       _ms("dose_target"),
        "CEM43_SPTP_min":           _ms("dose_sptp"),
        # Thermal — per-region 3D field
        "PeakTemp_Region_degC":     pk_temp_reg,
        "MeanTemp_Region_degC":     mn_temp_reg,
        "PeakTemp_Overlap_6dB_degC":pk_temp_ov6,
        "MeanTemp_Overlap_6dB_degC":mn_temp_ov6,
        "ThermalFlag":              thermal_flag,
        "CEM43_Limit_min":          round(cem43_limits, 1),
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — SAFETY SUMMARIES
# ════════════════════════════════════════════════════════════════════════════

def compute_mechanical_safety(tissue_masks, pressure_pa, I_full, freq_hz) -> List[Dict]:
    fmhz = freq_hz / 1e6
    rows = []
    for name, m in tissue_masks.items():
        if name.startswith("_") or m.sum() == 0:
            continue
        pp  = float(pressure_pa[m == 1].max())
        mi  = round(pp / 1e6 / np.sqrt(fmhz), 4)
        lim = MI_LIMIT_EYES if "eye" in name.lower() else MI_LIMIT_BRAIN
        rows.append({"tissue": name, "pp_kPa": round(pp/1e3, 1),
                     "Isppa": round(float(I_full[m==1].max()), 4),
                     "MI": mi, "MI_limit": lim, "ok": mi <= lim})
    return rows


def compute_thermal_safety(temp_c, tissue_masks, meta) -> Dict:
    out = {"available": temp_c is not None or not np.isnan(meta.get("T_peak", np.nan))}
    T_peak = float(temp_c.max()) if temp_c is not None else float(meta.get("T_peak", np.nan))
    out["T_peak"]  = round(T_peak, 3)
    out["dT_peak"] = round(T_peak - BASELINE_T, 3)
    out["T_target"]= round(float(meta.get("T_target", np.nan)), 3)

    Tm = np.ravel(meta.get("T_masked", []))
    per = {}
    for i, lab in enumerate(meta.get("med_labels", [])):
        if i < Tm.size and lab.lower() != "background":
            per[lab] = round(float(Tm[i]), 3)
    out["T_per_tissue"] = per

    Dm   = np.ravel(meta.get("dose_masked", []))
    dose = {}
    for i, lab in enumerate(meta.get("med_labels", [])):
        if i < Dm.size and lab.lower() != "background":
            dose[lab] = float(Dm[i])
    out["dose_per_tissue"] = dose
    out["dose_target"]     = float(meta.get("dose_target", np.nan))
    out["dose_sptp"]       = float(meta.get("dose_sptp", np.nan))

    dT_ok   = (out["dT_peak"] <= T_RISE_LIMIT) or (T_peak <= T_ABS_LIMIT)
    dose_ok = all(d <= _cem43_limit(lab) for lab, d in dose.items())
    out["thermal_ok"] = bool(dT_ok and dose_ok)
    return out


def build_verdict(tm, beam, along, mech, thermal, cfg) -> Dict:
    reasons, flags = [], []

    cov = tm.get("Coverage_6dB_pct", 0) if "error" not in tm else 0
    on  = tm.get("OnTarget_6dB_pct", 0) if "error" not in tm else 0
    if "error" not in tm:
        if cov < 20 or on < 30:
            flags.append("targeting")
            reasons.append(f"Weak targeting (coverage {cov}%, on-target {on}%).")

    ratio = along.get("prefocal_ratio", 0)
    if ratio >= cfg.get("prefocal_ratio_flag", 0.5):
        flags.append("prefocal")
        reasons.append(
            f"Beam deposits {int(ratio*100)}% of focal peak intensity before the "
            f"target (brain peak at {along.get('prefocal_brain_pos_mm')} mm).")

    if mech and not all(r["ok"] for r in mech):
        flags.append("mechanical")
        bad = ", ".join(r["tissue"] for r in mech if not r["ok"])
        reasons.append(f"MI exceeds limit in: {bad}.")

    if thermal.get("available") and not thermal.get("thermal_ok", True):
        flags.append("thermal")
        reasons.append(f"Thermal exposure exceeds ITRUSST levels "
                       f"(ΔT {thermal.get('dT_peak')} °C).")

    level = ("go"      if not flags else
             "caution" if len(flags) == 1 else
             "review")
    if not flags:
        reasons = ["All checks within ITRUSST non-significant-risk levels."]
    return {"level": level, "flags": flags, "reasons": reasons,
            "mech_ok": all(r["ok"] for r in mech) if mech else True,
            "coverage": cov, "on_target": on, "prefocal_ratio": ratio}


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

def _get_bg(t1_path, medium_mask, affine_sim, shape):
    if _HAVE_NIBABEL and t1_path and os.path.exists(t1_path):
        bg = match_shape(
            _resample(nib.load(t1_path), affine_sim, shape, "linear"), shape
        ).astype(float)
    else:
        bg = medium_mask.astype(float)
    mx = bg.max()
    return bg / mx if mx > 0 else bg


def _tissue_rgb(tissue_masks, shape):
    bg = np.zeros(shape, float)
    for key, lvl in [("Scalp", 0.35), ("Skull", 0.65),
                     ("Brain (GM + WM + CSF)", 0.5), ("Brain (head proxy)", 0.5)]:
        if key in tissue_masks:
            bg[tissue_masks[key] == 1] = lvl
    return bg


def plot_orthogonal_views(I_brain, mask_6dB, mask_3dB, target_mask,
                          medium_mask, affine_sim, shape, peak_idx,
                          t1_path=None, hot_regions=None, hot_colors=None) -> str:
    hot_regions = hot_regions or {}; hot_colors = hot_colors or {}
    bg  = _get_bg(t1_path, medium_mask, affine_sim, shape)
    px, py, pz = peak_idx
    Inorm = I_brain / (I_brain.max() + 1e-12)
    Idisp = np.where(mask_6dB == 1, Inorm, 0.)
    tmask = target_mask if target_mask is not None else np.zeros(shape, np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor="#0b0e14")
    views = [
        (bg[px,:,:], Idisp[px,:,:], mask_6dB[px,:,:], mask_3dB[px,:,:], tmask[px,:,:], "Sagittal","sag"),
        (bg[:,py,:], Idisp[:,py,:], mask_6dB[:,py,:], mask_3dB[:,py,:], tmask[:,py,:], "Coronal", "cor"),
        (bg[:,:,pz], Idisp[:,:,pz], mask_6dB[:,:,pz], mask_3dB[:,:,pz],tmask[:,:,pz],"Axial",   "axi"),
    ]
    for ax, (b, I, s6, s3, ts, title, key) in zip(axes, views):
        ax.set_facecolor("#0b0e14")
        ax.imshow(b.T, cmap=CMAP_BG, origin="lower", vmin=0, vmax=1, alpha=0.8)
        ax.imshow(np.ma.masked_where(I==0,I).T, cmap=CMAP_INT, origin="lower", alpha=0.9, vmin=0, vmax=1)
        for sl, c, lw, ls in [(s6,COL_6DB,1.5,"solid"),(s3,COL_3DB,1.2,"solid"),(ts,COL_TARGET,2.,"dashed")]:
            if sl.any():
                ax.contour(sl.T, levels=[0.5], colors=[c], linewidths=lw, linestyles=ls, origin="lower")
        for rname, m in hot_regions.items():
            sl = {"sag":m[px,:,:],"cor":m[:,py,:],"axi":m[:,:,pz]}[key]
            if sl.any():
                ax.contour(sl.T, levels=[0.5], colors=[hot_colors.get(rname,"#fff")],
                           linewidths=1.1, linestyles="dotted", origin="lower")
        ax.set_title(title, color="#e6edf3", fontsize=11, fontweight="bold", pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor("#222a36")
    sm = plt.cm.ScalarMappable(cmap=CMAP_INT, norm=mcolors.Normalize(0,1)); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.02, shrink=0.8)
    cb.set_label("Normalised intensity", color="#e6edf3", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="#e6edf3", labelcolor="#e6edf3", labelsize=8)
    fig.suptitle("Acoustic focus — orthogonal views through intensity peak",
                 color="#e6edf3", fontsize=11, y=1.02)
    return fig_to_b64(fig)


def plot_beam_corridor(I_full, tissue_masks, beam, affine_sim, shape,
                       voxel_mm, isppa_thr,
                       region_masks=None, hot_colors=None) -> str:
    tx = beam["transducer_mm"]; axis = beam["axis"]
    L  = beam["path_length_mm"]; fproj = beam["focus_proj"]
    up = np.eye(3)[np.argmin(np.abs(axis))]
    v1 = up - np.dot(up, axis)*axis; v1 /= (np.linalg.norm(v1)+1e-9)

    s  = np.arange(-5, L+22, voxel_mm*0.6)
    t  = np.arange(-32, 32, voxel_mm*0.6)
    SS, TT = np.meshgrid(s, t, indexing="ij")
    world   = tx[None,None,:] + SS[...,None]*axis + TT[...,None]*v1
    inv     = np.linalg.inv(affine_sim)
    vox     = world @ inv[:3,:3].T + inv[:3,3]
    coords  = [vox[...,0], vox[...,1], vox[...,2]]

    bg_s = map_coordinates(_tissue_rgb(tissue_masks, shape), coords, order=1, mode="constant", cval=0)
    I_s  = map_coordinates(I_full, coords, order=1, mode="constant", cval=0)

    fig, ax = plt.subplots(figsize=(11,5.6), facecolor="#0b0e14")
    ax.set_facecolor("#0b0e14")
    ext = [s.min(), s.max(), t.min(), t.max()]
    ax.imshow(bg_s.T, cmap=CMAP_BG, origin="lower", extent=ext, aspect="auto", vmin=0, vmax=1, alpha=0.85)
    Imax = I_s.max()+1e-9
    ax.imshow(np.ma.masked_where(I_s < isppa_thr*0.5, I_s).T, cmap=CMAP_INT,
              origin="lower", extent=ext, aspect="auto", alpha=0.92, vmin=0, vmax=Imax)
    ax.contour((I_s/Imax).T, levels=[0.25,0.5], colors=[COL_6DB,COL_3DB],
               linewidths=[1.3,1.1], extent=ext, origin="lower")
    ax.axhline(0, color=COL_AXIS, lw=1., ls=(0,(4,3)), alpha=0.8)
    ax.plot([fproj],[0], marker="o", ms=11, mfc="none", mec=COL_TARGET, mew=2.2)
    ax.annotate("target",(fproj,0), color=COL_TARGET, fontsize=9,
                xytext=(fproj,7), ha="center")
    ppx = beam.get("peak_proj"); ply = beam.get("peak_lateral_mm",0.)
    if ppx is not None and abs(ppx-fproj) > 1.:
        ax.plot([ppx],[ply], marker="x", ms=11, color=COL_FOCUS, mew=2.2)
        ax.annotate("actual\npeak",(ppx,ply), color=COL_FOCUS, fontsize=8,
                    ha="center", xytext=(ppx, ply-9))
    if beam.get("entry_proj") is not None:
        ax.axvline(beam["entry_proj"], color="#9ca3af", lw=1., ls=":")
        ax.annotate("skull\nentry",(beam["entry_proj"], t.min()+4),
                    color="#9ca3af", fontsize=8, ha="center")
    if region_masks and hot_colors:
        for rname, col in hot_colors.items():
            m = region_masks.get(rname)
            if m is not None:
                ms = map_coordinates(m.astype(float), coords, order=0, mode="constant", cval=0)
                if ms.max() > 0:
                    ax.contour(ms.T, levels=[0.5], colors=[col], linewidths=1.,
                               linestyles="dotted", extent=ext, origin="lower")
    ax.set_xlabel("Distance along beam axis from transducer (mm)", color="#9ca3af", fontsize=9)
    ax.set_ylabel("Lateral (mm)", color="#9ca3af", fontsize=9)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#222a36")
    ax.set_title("Beam-path corridor", color="#e6edf3", fontsize=11, fontweight="bold")
    return fig_to_b64(fig)


def plot_along_beam(along: Dict, isppa_thr: float) -> str:
    s = along["s_mm"]; p = along["profile"]; fproj = along["focus_proj"]
    fig, ax = plt.subplots(figsize=(11,3.6), facecolor="#0b0e14")
    ax.set_facecolor("#11161f")
    ib = along.get("in_brain")
    if ib is not None and np.any(ib):
        ax.fill_between(s, 0, p.max()*1.18, where=ib, color="#3b82f6", alpha=0.10,
                        step="mid", label="brain on path")
    fz = along.get("in_focus")
    if fz is not None and np.any(fz):
        ax.fill_between(s, 0, p.max()*1.18, where=fz, color=COL_6DB, alpha=0.16,
                        step="mid", label="−6 dB focal zone")
    ax.plot(s, p, color="#f59e0b", lw=2.)
    ax.fill_between(s, 0, p, color="#f59e0b", alpha=0.15)
    ax.axvline(fproj, color=COL_TARGET, lw=1.4, ls="--", label="planned target")
    if along.get("peak_proj") and abs(along["peak_proj"]-fproj) > 1.:
        ax.axvline(along["peak_proj"], color=COL_FOCUS, lw=1.2, ls="-.", label="actual peak")
    ax.axhline(isppa_thr, color=COL_6DB, lw=1., ls=(0,(4,3)), label=f"threshold {isppa_thr} W/cm²")
    bpk  = along.get("prefocal_brain_peak",0)
    bpos = along.get("prefocal_brain_pos_mm", np.nan)
    if bpk > 0 and not np.isnan(bpos):
        ax.plot([bpos],[bpk], marker="v", ms=10, color="#f43f5e")
        ax.annotate(f"pre-target brain\n{bpk} W/cm²",(bpos,bpk), color="#f43f5e",
                    fontsize=8, ha="center", xytext=(bpos, bpk+p.max()*0.12))
    ax.set_ylim(0, p.max()*1.25+1e-6)
    ax.set_xlabel("Distance along beam axis from transducer (mm)", color="#9ca3af", fontsize=9)
    ax.set_ylabel("Peak intensity (W/cm²)", color="#9ca3af", fontsize=9)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#222a36")
    ax.legend(facecolor="#11161f", edgecolor="#222a36", labelcolor="#e6edf3",
              fontsize=8, ncol=2, loc="upper right")
    ax.set_title("Intensity deposited along the beam path", color="#e6edf3",
                 fontsize=11, fontweight="bold")
    return fig_to_b64(fig)


def plot_temperature(temp_c, peak_idx, affine_sim, shape) -> str:
    if temp_c is None:
        return ""
    px, py, pz = np.unravel_index(temp_c.argmax(), temp_c.shape)
    dT   = temp_c - BASELINE_T
    vmax = max(0.1, float(dT.max()))
    fig, axes = plt.subplots(1, 3, figsize=(15,5.), facecolor="#0b0e14")
    for ax, sl, title in zip(axes, [dT[px,:,:], dT[:,py,:], dT[:,:,pz]],
                             ["Sagittal","Coronal","Axial"]):
        ax.set_facecolor("#0b0e14")
        ax.imshow(np.ma.masked_where(sl<=0.02, sl).T, cmap=CMAP_TEMP,
                  origin="lower", vmin=0, vmax=vmax)
        ax.set_title(title, color="#e6edf3", fontsize=11, fontweight="bold", pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor("#222a36")
    sm = plt.cm.ScalarMappable(cmap=CMAP_TEMP, norm=mcolors.Normalize(0,vmax)); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.02, shrink=0.8)
    cb.set_label("Temperature rise ΔT (°C)", color="#e6edf3", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="#e6edf3", labelcolor="#e6edf3", labelsize=8)
    fig.suptitle("Maximum temperature rise", color="#e6edf3", fontsize=11, y=1.02)
    return fig_to_b64(fig)


def plot_mi_chart(mech: List[Dict], son_idx: int) -> str:
    if not mech:
        return ""
    names  = [r["tissue"]  for r in mech]
    mis    = [r["MI"]      for r in mech]
    limits = [r["MI_limit"]for r in mech]
    colors = ["#ef4444" if not r["ok"] else "#3b82f6" for r in mech]

    fig, ax = plt.subplots(figsize=(9, max(3, len(mech)*0.7)), facecolor="#0b0e14")
    ax.set_facecolor("#11161f")
    y = np.arange(len(mech))
    ax.barh(y, mis, color=colors, height=0.6, edgecolor="none")
    ax.axvline(MI_LIMIT_BRAIN, color="#fde047", lw=1.4, ls="--",
               label=f"Brain limit {MI_LIMIT_BRAIN}")
    ax.axvline(MI_LIMIT_EYES,  color="#fb923c", lw=1.2, ls=":",
               label=f"Eye limit {MI_LIMIT_EYES}")
    ax.set_yticks(y); ax.set_yticklabels(names, color="#e6edf3", fontsize=8)
    ax.tick_params(axis="x", colors="#9ca3af", labelsize=8)
    ax.set_xlabel("Mechanical Index", color="#9ca3af", fontsize=9)
    ax.set_title(f"Sonication {son_idx} — MI per tissue", color="#e6edf3",
                 fontsize=11, fontweight="bold")
    for sp in ax.spines.values(): sp.set_edgecolor("#222a36")
    ax.legend(facecolor="#11161f", edgecolor="#222a36", labelcolor="#e6edf3", fontsize=8)
    return fig_to_b64(fig)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — HTML / CSS REPORT
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
.wrap{max-width:1200px;margin:0 auto;padding:0 24px 80px}
/* ── Header ── */
.head{padding:36px 24px 28px;border-bottom:1px solid var(--bd);
 background:radial-gradient(1200px 260px at 75% -40%,rgba(59,130,246,.18),transparent)}
.head .inner{max-width:1200px;margin:0 auto}
.tag{display:inline-block;font-family:var(--mono);font-size:11px;letter-spacing:.08em;
 color:var(--acc2);border:1px solid rgba(59,130,246,.4);background:rgba(59,130,246,.1);
 padding:3px 11px;border-radius:20px;margin-bottom:14px}
h1{font-size:26px;font-weight:700;letter-spacing:-.3px}
.sub{color:var(--mut);font-size:13px;margin-top:3px}
.meta-bar{display:flex;gap:32px;flex-wrap:wrap;margin-top:20px}
.meta-bar div{display:flex;flex-direction:column;gap:1px}
.meta-bar .k{font-family:var(--mono);font-size:10.5px;text-transform:uppercase;
 letter-spacing:.06em;color:var(--mut)}
.meta-bar .v{font-size:13px;font-weight:600}
/* ── Section titles ── */
.sec{font-family:var(--mono);font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;
 color:var(--mut);margin:36px 0 16px;padding-bottom:8px;border-bottom:1px solid var(--bd)}
/* ── Cards ── */
.card{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--r);
 padding:20px 22px;margin-bottom:14px}
.card h3{font-size:13px;font-weight:600;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.card h3::before{content:'';width:3px;height:15px;background:var(--acc);
 border-radius:2px;display:inline-block}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
@media(max-width:840px){.g2,.g3{grid-template-columns:1fr}}
/* ── Key-value rows ── */
.row{display:flex;justify-content:space-between;align-items:baseline;
 padding:6px 0;border-bottom:1px solid var(--bd)}
.row:last-child{border:0}
.row .k{color:var(--mut);font-size:12.5px}
.row .v{font-family:var(--mono);font-size:13px;font-weight:600;color:var(--acc2)}
.row .u{color:var(--mut);font-size:10.5px;margin-left:3px;font-family:var(--sans);font-weight:400}
/* ── Tables ── */
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{background:var(--bg3);color:var(--mut);font-family:var(--mono);font-size:10.5px;
 text-transform:uppercase;letter-spacing:.05em;font-weight:500;
 text-align:left;padding:9px 12px;border-bottom:1px solid var(--bd)}
td{padding:8px 12px;border-bottom:1px solid var(--bd)}
tr:last-child td{border:0}
tbody tr:hover{background:rgba(255,255,255,.025)}
td.n{font-family:var(--mono);color:var(--acc2)}
td.m{color:var(--mut);font-size:11.5px}
/* ── Badges ── */
.badge{display:inline-block;padding:2px 9px;border-radius:20px;
 font-family:var(--mono);font-size:10.5px;font-weight:600}
.b-go {background:rgba(34,197,94,.15);color:var(--go);border:1px solid rgba(34,197,94,.4)}
.b-cau{background:rgba(245,158,11,.15);color:var(--cau);border:1px solid rgba(245,158,11,.4)}
.b-rev{background:rgba(239,68,68,.15); color:var(--rev);border:1px solid rgba(239,68,68,.4)}
.b-ok {background:rgba(34,197,94,.12);color:var(--go);border:1px solid rgba(34,197,94,.3)}
.b-bad{background:rgba(239,68,68,.12);color:var(--rev);border:1px solid rgba(239,68,68,.3)}
/* ── Verdict block ── */
.verdict{border-radius:var(--r);padding:18px 20px;margin-bottom:14px;border:1px solid}
.v-go {border-color:rgba(34,197,94,.35);background:rgba(34,197,94,.06)}
.v-cau{border-color:rgba(245,158,11,.35);background:rgba(245,158,11,.06)}
.v-rev{border-color:rgba(239,68,68,.35);background:rgba(239,68,68,.06)}
.verdict .lead{display:flex;align-items:center;gap:10px;font-size:15px;
 font-weight:700;margin-bottom:8px}
.verdict ul{margin:6px 0 0 2px;padding-left:18px;color:var(--mut);font-size:12.5px}
/* ── Chips ── */
.chips{display:flex;gap:12px;flex-wrap:wrap;margin-top:12px}
.chip{background:var(--bg3);border:1px solid var(--bd);border-radius:8px;
 padding:9px 14px;min-width:120px}
.chip .k{font-family:var(--mono);font-size:10px;text-transform:uppercase;
 color:var(--mut);letter-spacing:.04em}
.chip .v{font-family:var(--mono);font-size:18px;font-weight:600;margin-top:2px}
/* ── Figures ── */
.fig{background:var(--bg2);border:1px solid var(--bd);border-radius:var(--r);
 overflow:hidden;margin-bottom:14px}
.fig img{width:100%;display:block}
.fig .cap{padding:9px 14px;font-size:11.5px;color:var(--mut);
 border-top:1px solid var(--bd);font-style:italic}
/* ── Legend pills ── */
.pill{display:inline-flex;align-items:center;gap:6px;background:rgba(255,255,255,.04);
 border:1px solid var(--bd);border-radius:20px;padding:3px 11px;font-size:11px;
 color:var(--mut);margin:0 6px 6px 0}
.dot{width:11px;height:3px;border-radius:2px;display:inline-block}
/* ── Region table ── */
.rtbl-wrap{overflow-x:auto}.rtbl-wrap table{min-width:900px}
.filter{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:11px}
.filter input{flex:1;min-width:160px;background:var(--bg3);border:1px solid var(--bd);
 border-radius:7px;color:var(--tx);font-family:var(--mono);font-size:12px;
 padding:7px 11px;outline:none}
.tb{background:var(--bg3);border:1px solid var(--bd);border-radius:7px;color:var(--mut);
 font-family:var(--mono);font-size:11.5px;padding:6px 12px;cursor:pointer}
.tb.on{background:var(--acc);border-color:var(--acc);color:#fff}
tr.hot td{background:rgba(239,68,68,.06)}
tr.hot td:first-child{border-left:3px solid var(--rev)}
td.hotv{color:var(--rev)!important;font-weight:700}
.zone-pre{color:var(--cau);font-weight:600}
.zone-foc{color:var(--go);font-weight:600}
.zone-post{color:var(--mut)}
/* ── Sonication header ── */
.son-hd{display:flex;align-items:center;gap:14px;padding:14px 18px;
 border-radius:var(--r);margin:10px 0 20px;border:1px solid var(--bd);
 background:linear-gradient(90deg,rgba(59,130,246,.1),transparent)}
.son-no{font-family:var(--mono);font-weight:700;font-size:13px;color:#fff;
 background:var(--acc);padding:4px 13px;border-radius:20px}
/* ── Note / comparison ── */
.note{font-size:11px;color:var(--mut);margin-top:10px}
.cmp td.best{color:var(--go);font-weight:700}
/* ── Footer ── */
.foot{border-top:1px solid var(--bd);padding:24px;text-align:center;
 color:var(--mut);font-family:var(--mono);font-size:11px;margin-top:32px}
/* ── New: beam-position badge colours ── */
.bp-pre{color:#f59e0b;font-family:var(--mono);font-size:11px;font-weight:600}
.bp-foc{color:#22c55e;font-family:var(--mono);font-size:11px;font-weight:600}
.bp-post{color:#8b98a9;font-family:var(--mono);font-size:11px}
"""

# ── HTML helpers ──────────────────────────────────────────────────────────

def _f(v, d=2, u=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "<span class='m'>n/a</span>"
    s = f"{v:.{d}f}" if isinstance(v, float) else str(v)
    return f"{s}<span class='u'>{u}</span>" if u else s


def _row(k, v, d=2, u=""):
    return f"<div class='row'><span class='k'>{k}</span><span class='v'>{_f(v,d,u)}</span></div>"


def _badge(level):
    return {"go":  "<span class='badge b-go'>GO</span>",
            "caution": "<span class='badge b-cau'>CAUTION</span>",
            "review":  "<span class='badge b-rev'>REVIEW</span>"}[level]


def _fig(b64, cap):
    if not b64:
        return ""
    return (f"<div class='fig'><img src='data:image/png;base64,{b64}'>"
            f"<div class='cap'>{cap}</div></div>")


def _verdict_html(v):
    cls   = {"go":"v-go","caution":"v-cau","review":"v-rev"}[v["level"]]
    lead  = {"go":"Suitable placement","caution":"Usable — review flags",
             "review":"Review before use"}[v["level"]]
    items = "".join(f"<li>{r}</li>" for r in v["reasons"])
    pf    = int(v["prefocal_ratio"] * 100)
    return f"""
<div class='verdict {cls}'>
  <div class='lead'>{_badge(v['level'])} {lead}</div>
  <ul>{items}</ul>
  <div class='chips'>
    <div class='chip'><div class='k'>Coverage −6dB</div><div class='v'>{v['coverage']}%</div></div>
    <div class='chip'><div class='k'>On-target −6dB</div><div class='v'>{v['on_target']}%</div></div>
    <div class='chip'><div class='k'>Pre-focal / peak</div><div class='v'>{pf}%</div></div>
    <div class='chip'><div class='k'>Mechanical</div><div class='v'>{'OK' if v['mech_ok'] else 'OVER'}</div></div>
  </div>
</div>"""


def _placement_card(meta, beam, along):
    tx = beam["transducer_mm"]
    fc = beam.get("focus_center_mm", meta["target_mm"])
    steer = round(float(np.linalg.norm(np.array(fc) - np.asarray(meta["target_mm"], float))), 1)
    return f"""
<div class='card'><h3>Transducer placement</h3>
 {_row("Elements",                    meta['n_elements'], 0)}
 {_row("Centre frequency",            meta['freq_kHz'], 1, "kHz")}
 {_row("Geometric focal distance",    meta['focal_dist_mm'], 1, "mm")}
 {_row("Beam path (tx → target)",     beam['path_length_mm'], 1, "mm")}
 {_row("Skull thickness on-axis",     beam['skull_thickness_mm'], 1, "mm")}
 {_row("Targeting error (peak→tgt)",  steer, 1, "mm")}
 {_row("Peak axial offset (+deep/−short)", beam.get('peak_axial_offset_mm'), 1, "mm")}
 {_row("Peak off-axis distance",      beam.get('peak_lateral_mm'), 1, "mm")}
 {_row("Transducer pivot X",          tx[0], 1, "mm")}
 {_row("Transducer pivot Y",          tx[1], 1, "mm")}
 {_row("Transducer pivot Z",          tx[2], 1, "mm")}
</div>"""


def _focus_card(foc, beam, meta):
    fc = beam.get("focus_center_mm", [np.nan]*3)
    return f"""
<div class='card'><h3>Acoustic focus</h3>
 {_row("Isppa (global peak)",         foc['Isppa'], 3, "W/cm²")}
 {_row("Isppa (in brain)",            foc['Isppa_brain'], 3, "W/cm²")}
 {_row("Peak pressure (SPTP)",        meta['sptp_kPa'], 1, "kPa")}
 {_row("Pressure at target",          meta['p_at_target_kPa'], 1, "kPa")}
 {_row("Focal volume −6 dB",         foc['focus_vol_6dB'], 1, "mm³")}
 {_row("Focal volume −3 dB",         foc['focus_vol_3dB'], 1, "mm³")}
 {_row("FWHM axial (along beam)",     beam.get('fwhm_axial_mm'), 2, "mm")}
 {_row("FWHM lateral (mean)",         beam.get('fwhm_lat_mean_mm'), 2, "mm")}
 {_row("Beam elongation",             beam.get('elongation'), 2)}
</div>"""


def _targeting_card(tm, name):
    if "error" in tm:
        return f"<div class='card'><h3>Targeting</h3><p class='m'>{tm['error']}</p></div>"
    return f"""
<div class='card'><h3>Targeting — {name}</h3>
 {_row("Target volume",               tm['TargetVol_mm3'], 1, "mm³")}
 {_row("Isppa in target",             tm['Isppa_Target_Wcm2'], 3, "W/cm²")}
 {_row("Mean intensity in target",    tm['Imean_Target_Wcm2'], 3, "W/cm²")}
 {_row("Coverage −6 dB",             tm['Coverage_6dB_pct'], 1, "%")}
 {_row("Coverage −3 dB",             tm['Coverage_3dB_pct'], 1, "%")}
 {_row("On-target −6 dB",            tm['OnTarget_6dB_pct'], 1, "%")}
 {_row("Off-target −6 dB",           tm['OffTarget_6dB_pct'], 1, "%")}
 {_row("Overlap volume −6 dB",       tm['Coverage_6dB_mm3'], 1, "mm³")}
 {_row("Peak pressure in target",     tm['PeakPressure_Target_kPa'], 1, "kPa")}
 {_row("Peak P in −6dB overlap",      tm['PeakP_Overlap_6dB_kPa'], 1, "kPa")}
</div>"""


def _safety_card(mech, thermal):
    mrows = ""
    for r in mech:
        ok  = r["ok"]
        lbl = f"<span class='badge {'b-ok' if ok else 'b-bad'}'>{'OK' if ok else 'OVER'}</span>"
        mrows += (f"<tr><td>{r['tissue']}</td>"
                  f"<td class='n'>{r['pp_kPa']}</td>"
                  f"<td class='n'>{r['MI']}</td>"
                  f"<td class='n'>{r['MI_limit']}</td>"
                  f"<td>{lbl}</td></tr>")
    if thermal.get("available"):
        tpt = "".join(f"<div class='row'><span class='k'>{k}</span>"
                      f"<span class='v'>{_f(v,3,'°C')}</span></div>"
                      for k, v in thermal.get("T_per_tissue",{}).items())
        dpt = "".join(f"<div class='row'><span class='k'>CEM43 {k}</span>"
                      f"<span class='v'>{v:.2e} <span class='u'>min</span></span></div>"
                      for k, v in thermal.get("dose_per_tissue",{}).items())
        tb = ("<span class='badge b-ok'>within ITRUSST</span>" if thermal['thermal_ok']
              else "<span class='badge b-bad'>exceeds</span>")
        th_html = f"""
   {_row("Peak temperature",   thermal['T_peak'], 3, "°C")}
   {_row("Peak rise ΔT",       thermal['dT_peak'], 3, "°C")}
   {tpt}{dpt}
   <div class='row'><span class='k'>Thermal verdict</span><span class='v'>{tb}</span></div>"""
    else:
        th_html = "<p class='m'>No temperature field in this H5.</p>"
    return f"""
<div class='card'><h3>Safety — mechanical &amp; thermal</h3>
 <table><thead><tr><th>Tissue</th><th>Peak P (kPa)</th><th>MI</th>
  <th>Limit</th><th>Status</th></tr></thead>
 <tbody>{mrows}</tbody></table>
 <div style='margin-top:14px'>{th_html}</div>
 <p class='note'>ITRUSST 2024: MI ≤ {MI_LIMIT_BRAIN} (brain) / {MI_LIMIT_EYES} (eyes);
 ΔT ≤ {T_RISE_LIMIT} °C or T ≤ {T_ABS_LIMIT} °C;
 CEM43 ≤ {CEM43_LIMIT_BRAIN} brain / {CEM43_LIMIT_BONE:.0f} bone / {CEM43_LIMIT_SKIN:.0f} skin (min).</p>
</div>"""


def _prefocal_card(region_rows, along, cfg):
    thr = cfg["isppa_overlay_threshold"]
    pre = [r for r in region_rows
           if r.get("beam_zone") == "pre-focal" and r["Isppa_Target_Wcm2"] >= thr]
    pre.sort(key=lambda r: r["Isppa_Target_Wcm2"], reverse=True)

    if pre:
        rows_html = "".join(
            f"<tr class='hot'><td>{r['name']}</td><td class='m'>{r['type']}</td>"
            f"<td class='n hotv'>{r['Isppa_Target_Wcm2']:.3f}</td>"
            f"<td class='n'>{abs(r['along_focus_mm']):.1f}</td>"
            f"<td class='n'>{r['lateral_mm']:.1f}</td>"
            f"<td class='n'>{r['Coverage_6dB_pct']:.0f}</td></tr>"
            for r in pre)
        body = (f"<table><thead><tr>"
                f"<th>Structure</th><th>Type</th><th>Isppa W/cm²</th>"
                f"<th>Before target (mm)</th><th>Off-axis (mm)</th><th>Cov −6dB %</th>"
                f"</tr></thead><tbody>{rows_html}</tbody></table>")
    else:
        body = "<p class='m'>No mapped structure before the target exceeds the threshold.</p>"

    ratio = along.get("prefocal_ratio", 0)
    rcls  = "bp-pre" if ratio >= cfg["prefocal_ratio_flag"] else "bp-foc"
    bpk   = along.get("prefocal_brain_peak", 0)
    bpos  = along.get("prefocal_brain_pos_mm", "n/a")
    any_pk = along.get("prefocal_peak", 0)

    return f"""
<div class='card'><h3>Pre-focal exposure (brain crossed before the target)</h3>
 <div class='chips' style='margin-bottom:14px'>
   <div class='chip'><div class='k'>Pre-target brain peak</div>
    <div class='v'>{bpk}<span class='u'> W/cm²</span></div></div>
   <div class='chip'><div class='k'>Position from transducer</div>
    <div class='v'>{bpos}<span class='u'> mm</span></div></div>
   <div class='chip'><div class='k'>vs focus intensity</div>
    <div class='v {rcls}'>{int(ratio*100)}%</div></div>
   <div class='chip'><div class='k'>Any-tissue peak</div>
    <div class='v'>{any_pk}<span class='u'> W/cm²</span></div></div>
 </div>
 {body}
 <p class='note'>Pre-target brain intensity approaching the focal peak indicates
 substantial off-target neuromodulation. "Any-tissue peak" includes scalp/skull.</p>
</div>"""


def _region_table(rows, thr, hot_colors, tid):
    """Filterable region table with beam-zone column and new pressure/temp cols."""
    if not rows:
        return ""

    def sw(n):
        c = hot_colors.get(n)
        return (f"<span style='display:inline-block;width:9px;height:9px;border-radius:50%;"
                f"background:{c};margin-right:6px;vertical-align:middle'></span>") if c else ""

    def zspan(z):
        cls = {"pre-focal":"bp-pre","at focus":"bp-foc","post-focal":"bp-post"}.get(z,"")
        return f"<span class='{cls}'>{z}</span>"

    body = ""
    for r in rows:
        hot     = r["Isppa_Target_Wcm2"] > thr
        hcls    = "hot" if hot else ""
        hvcls   = "hotv" if hot else ""
        bz      = r.get("beam_zone", "—")
        alo_mm  = r.get("along_focus_mm", np.nan)
        lat_mm  = r.get("lateral_mm", np.nan)
        tf      = r.get("ThermalFlag","—")
        tf_col  = ("color:var(--rev)" if tf=="DANGER" else
                   "color:var(--cau)" if tf in ("CAUTION","EXCEEDS LIMIT") else
                   "color:var(--go)"  if tf=="OK" else "")
        body += (f"<tr class='{hcls}' data-type='{r['type']}' data-zone='{bz}'>"
                 f"<td>{sw(r['name'])}{r['name']}</td>"
                 f"<td class='m'>{r['type']}</td>"
                 f"<td>{zspan(bz)}</td>"
                 f"<td class='n'>{_f(alo_mm,1)}</td>"
                 f"<td class='n'>{_f(lat_mm,1)}</td>"
                 f"<td class='n {hvcls}'>{r['Isppa_Target_Wcm2']:.3f}</td>"
                 f"<td class='n'>{r['Coverage_6dB_pct']:.0f}</td>"
                 f"<td class='n'>{_f(r.get('PeakPressure_Target_kPa'),1)}</td>"
                 f"<td class='n'>{_f(r.get('PeakP_Overlap_6dB_kPa'),1)}</td>"
                 f"<td class='n' style='{tf_col}'>{tf}</td></tr>")

    nh  = sum(1 for r in rows if r["Isppa_Target_Wcm2"] > thr)
    npe = sum(1 for r in rows if r.get("beam_zone") == "pre-focal")
    return f"""
<div class='card'><h3>All regions — intensity, beam position &amp; safety</h3>
 <div class='filter'>
   <input id='{tid}_s' placeholder='Filter by name…' oninput="rf('{tid}')">
   <button class='tb on' id='{tid}_all' onclick="rt('{tid}','all')">All ({len(rows)})</button>
   <button class='tb'    id='{tid}_hot' onclick="rt('{tid}','hot')">⚠ Hot ({nh})</button>
   <button class='tb'    id='{tid}_pre' onclick="rt('{tid}','pre')">Pre-focal ({npe})</button>
 </div>
 <div class='rtbl-wrap'><table id='{tid}'><thead><tr>
  <th>Region</th><th>Tier</th><th>Beam zone</th>
  <th>Along focus (mm)</th><th>Lateral (mm)</th>
  <th>Isppa W/cm²</th><th>Cov −6dB %</th>
  <th>Peak P (kPa)</th><th>Peak P −6dB (kPa)</th>
  <th>Thermal</th></tr></thead>
  <tbody>{body}</tbody></table></div>
 <p class='note'>Sorted by Isppa. Highlighted rows exceed {thr} W/cm².
 Along-focus: negative = before target; positive = beyond target.
 Thermal flag uses ITRUSST CEM43 tissue-tiered limits.</p>
</div>
<script>
window.rt=window.rt||function(t,m){{
 ['all','hot','pre'].forEach(k=>{{var b=document.getElementById(t+'_'+k);
  if(b)b.classList.toggle('on',k===m);}});window['_m_'+t]=m;rf(t);}};
window.rf=window.rf||function(t){{
 var q=(document.getElementById(t+'_s').value||'').toLowerCase();
 var m=window['_m_'+t]||'all';
 document.querySelectorAll('#'+t+' tbody tr').forEach(function(r){{
  var hot=r.classList.contains('hot');
  var pre=r.dataset.zone==='pre-focal';
  var ok=(m==='all')||(m==='hot'&&hot)||(m==='pre'&&pre);
  var s=!q||r.textContent.toLowerCase().includes(q);
  r.style.display=(ok&&s)?'':'none';}});}};
</script>"""


def _comparison_table(summaries: List[Dict], target_name: str) -> str:
    if len(summaries) < 2:
        return ""
    def score(s):
        p = 0 if s["verdict"]["level"]=="go" else (1 if s["verdict"]["level"]=="caution" else 2)
        return (-p, s["verdict"]["coverage"], -s["verdict"]["prefocal_ratio"])
    best_i = max(range(len(summaries)), key=lambda i: score(summaries[i]))
    rows = ""
    for i, s in enumerate(summaries):
        v = s["verdict"]; best = (i == best_i)
        cc = "best" if best else "n"
        star = " ★" if best else ""
        rows += (f"<tr><td class='{cc}'>Sonication {s['idx']}{star}</td>"
                 f"<td>{_badge(v['level'])}</td>"
                 f"<td class='{cc}'>{v['coverage']}%</td>"
                 f"<td class='n'>{v['on_target']}%</td>"
                 f"<td class='n'>{int(v['prefocal_ratio']*100)}%</td>"
                 f"<td class='n'>{s['Isppa_target']}</td>"
                 f"<td class='n'>{s['max_MI']}</td>"
                 f"<td class='n'>{_f(s['dT'],2)}</td>"
                 f"<td class='n'>{_f(s['skull_mm'],1)}</td>"
                 f"<td class='n'>{_f(s['fwhm_ax'],2)}</td></tr>")
    return f"""
<div class='sec'>Placement comparison</div>
<div class='card cmp'><h3>Candidate placements — {target_name}</h3>
 <table><thead><tr>
  <th>Placement</th><th>Verdict</th><th>Coverage −6dB</th><th>On-target</th>
  <th>Pre-focal/focus</th><th>Isppa target</th><th>Max MI</th>
  <th>ΔT °C</th><th>Skull mm</th><th>FWHM axial mm</th></tr></thead>
 <tbody>{rows}</tbody></table>
 <p class='note'>★ = best candidate (verdict → coverage → lowest pre-focal ratio).
 Skull thickness and FWHM help compare beam path difficulty across placements.</p>
</div>"""


def _sonication_block(s: Dict, cfg: Dict) -> str:
    idx  = s["idx"]; meta = s["meta"]; foc = s["foc"]; beam = s["beam"]
    tm   = s["tm"];  mech = s["mech"]; thermal = s["thermal"]; along = s["along"]
    rows = s["region_rows"]; hot_colors = s["hot_colors"]
    name = cfg["target_name"]
    fc   = beam.get("focus_center_mm", meta["target_mm"])

    hot_regions_for_table = {r["name"]: None for r in rows if r.get("Isppa_Target_Wcm2",0) > cfg["isppa_overlay_threshold"]}
    pills = "".join(
        f"<span class='pill'><span class='dot' style='background:{c}'></span>{n}</span>"
        for n, c in hot_colors.items())

    rtbl = _region_table(rows, cfg["isppa_overlay_threshold"], hot_colors, f"rt{idx}")
    pre  = _prefocal_card(rows, along, cfg) if rows else ""

    return f"""
<div class='son-hd'>
 <span class='son-no'>Sonication {idx}</span>
 <div>
  <div style='font-weight:600'>Target: {name}</div>
  <div class='m' style='font-family:var(--mono);font-size:11.5px'>
   {meta['freq_kHz']} kHz · focus ({fc[0]:.0f}, {fc[1]:.0f}, {fc[2]:.0f}) mm ·
   path {beam['path_length_mm']} mm · skull {_f(beam.get('skull_thickness_mm'),1)} mm</div>
 </div>
</div>
{_verdict_html(s['verdict'])}
<div class='g2'>{_placement_card(meta,beam,along)}{_focus_card(foc,beam,meta)}</div>
<div class='g2'>{_targeting_card(tm,name)}{_safety_card(mech,thermal)}</div>
{pre}
<div class='sec'>Beam path</div>
{_fig(s['fig_corridor'],
  "Oblique reslice along the beam axis. Tissue greyscale, intensity coloured, "
  "−6/−3 dB contours. Dotted contours = flagged structures the beam crosses.")}
{_fig(s['fig_along'],
  "Intensity profile along the beam axis. Pre-focal brain peak (▼) marks "
  "energy deposited in brain before the intended target.")}
{_fig(s['fig_mi'],
  "Mechanical Index per tissue. Red bars exceed the tissue-appropriate MI limit.")}
<div class='sec'>Focus &amp; targeting</div>
<div style='margin-bottom:8px'>
 <span class='pill'><span class='dot' style='background:{COL_6DB}'></span>−6 dB</span>
 <span class='pill'><span class='dot' style='background:{COL_3DB}'></span>−3 dB</span>
 <span class='pill'><span class='dot' style='background:{COL_TARGET}'></span>target</span>{pills}
</div>
{_fig(s['fig_ortho'],
  "Orthogonal slices through the intensity peak with focus and target contours.")}
{_fig(s.get('fig_temp',''),
  "Maximum temperature rise (ΔT above 37 °C baseline).")}
{rtbl}
"""


def generate_html(report: Dict, cfg: Dict, out_path: str):
    date   = datetime.datetime.now().strftime("%d %b %Y · %H:%M")
    blocks = "".join(_sonication_block(s, cfg) for s in report["sonications"])
    cmp    = _comparison_table(report["sonications"], cfg["target_name"])
    op     = cfg.get("operator",""); notes = cfg.get("notes","")
    src    = report.get("tissue_source","")

    html = f"""<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>TUNeS Report — {cfg['participant_id']}</title>
<style>{_CSS}</style></head><body>
<div class='head'><div class='inner'>
 <div class='tag'>TUNeS · tFUS placement report</div>
 <h1>Acoustic simulation &amp; placement decision support</h1>
 <div class='sub'>{report['h5_name']}</div>
 <div class='meta-bar'>
  <div><span class='k'>Subject</span><span class='v'>{cfg['participant_id']}</span></div>
  <div><span class='k'>Session</span><span class='v'>{cfg['session_id']}</span></div>
  <div><span class='k'>Target</span><span class='v'>{cfg['target_name']}</span></div>
  <div><span class='k'>Sonications</span><span class='v'>{len(report['sonications'])}</span></div>
  <div><span class='k'>Tissue model</span><span class='v'>{src}</span></div>
  <div><span class='k'>Generated</span><span class='v'>{date}</span></div>
  {f"<div><span class='k'>Operator</span><span class='v'>{op}</span></div>" if op else ""}
 </div>
 {f"<div class='note' style='margin-top:14px'><b style='color:var(--tx)'>Notes:</b> {notes}</div>" if notes else ""}
</div></div>
<div class='wrap'>
{cmp}
{blocks}
<div class='foot'>TUNeS Report Generator v4.0 · safety levels per ITRUSST 2024
(Aubry et al.; Martin et al., Brain Stimul) · eye MI limit 0.4 ·
CEM43 limits: brain {CEM43_LIMIT_BRAIN} / bone {CEM43_LIMIT_BONE:.0f} / skin {CEM43_LIMIT_SKIN:.0f} min · {date}</div>
</div></body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [html] {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 — PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def process_h5(h5_path: str, cfg: Dict) -> Tuple[Dict, List[Dict]]:
    """Process one H5 file; return (report dict, list of CSV rows)."""
    h5_name = os.path.splitext(os.path.basename(h5_path))[0]
    n_son   = count_sonications(h5_path)
    print(f"\n=== {h5_name} — {n_son} sonication(s) ===")

    # Load shared masks once (use sonication 1 grid as reference)
    ref      = load_h5_data(h5_path, 1)
    aff_ref  = ref["affine_sim"]
    shp_ref  = ref["grid_shape"]

    target_mask = load_mask(cfg.get("target_mask_path"), aff_ref, shp_ref)
    nuc_masks   = load_mask_folder(cfg.get("region_mask_folder"), aff_ref, shp_ref,
                                   prefix_strip=["harvardoxford-subcortical_prob_",
                                                 "harvardoxford-cortical_prob_",
                                                 "harvardoxford-"])
    atlas_masks = load_mask_folder(cfg.get("atlas_mask_folder"), aff_ref, shp_ref)
    all_region_masks = {**nuc_masks, **atlas_masks}

    print(f"  target mask: {'loaded' if target_mask is not None else 'none'} | "
          f"nuclei: {len(nuc_masks)} | atlas: {len(atlas_masks)}")

    sonications = []
    all_csv_rows = []
    tissue_source = ""
    isppa_thr = cfg["isppa_overlay_threshold"]

    for i in range(1, n_son + 1):
        d    = load_h5_data(h5_path, i)
        meta = d["meta"]; vox_mm = d["voxel_mm"]; vox_vol = d["voxel_vol"]

        # Tissue masks & brain mask
        tmasks = build_tissue_masks(cfg.get("simnibs_path"),
                                    d["affine_sim"], d["grid_shape"], d["medium_mask"])
        tissue_source = tmasks.get("_source","")
        brain = get_brain_mask(tmasks, d["grid_shape"])

        # Intensity & focal zones
        foc = compute_intensity(d["pressure_pa"], brain, vox_vol)

        # Beam geometry
        focus_mm  = voxel_to_mm(foc["peak_idx"], d["affine_sim"])
        target_mm = meta.get("target_mm")
        if target_mm is None or np.any(np.isnan(np.ravel(target_mm))):
            target_mm = focus_mm
        beam = compute_beam_geometry(meta, target_mm, focus_mm,
                                     d["medium_mask"], d["affine_sim"], vox_mm)
        beam["focus_center_mm"] = focus_mm
        beam.update(compute_beam_fwhm(foc["I_full"], beam, d["affine_sim"], vox_mm))

        # Along-beam profile
        along = sample_along_beam(foc["I_full"], beam, d["affine_sim"], vox_mm,
                                  brain_mask=brain, focus_mask=foc["mask_6dB"])

        # Mechanical & thermal safety
        mech    = compute_mechanical_safety(tmasks, d["pressure_pa"], foc["I_full"], meta["freq_hz"])
        thermal = compute_thermal_safety(d["temp_c"], tmasks, meta)

        # ── Primary target metrics 
        tm_row = (make_row(
            target_mask, cfg["target_name"], cfg.get("target_mask_path",""), "Tier2_PrimaryTarget",
            i, d["pressure_pa"], foc["I_full"], foc["mask_6dB"], foc["mask_3dB"],
            foc["Isppa"], foc["Isppa_brain"],
            foc["focus_vox_6"], foc["focus_vox_3"], foc["focus_vol_6dB"], foc["focus_vol_3dB"],
            vox_vol, meta["freq_hz"], d["affine_sim"],
            d["temp_c"], meta)
            if target_mask is not None
            else {"error": "No primary target mask."})

        # ── All region rows (tiered) ───────────────────────────────────────
        region_rows_raw: List[Dict] = []
        hot_regions: Dict[str,np.ndarray] = {}
        hot_colors:  Dict[str,str]        = {}
        col_idx = 0

        for tier_label, mask_dict in [
            ("Tier1_Tissue",   {k:v for k,v in tmasks.items() if not k.startswith("_")}),
            ("Tier2_Nuclei",   nuc_masks),
            ("Tier3_Atlas",    atlas_masks),
        ]:
            for rname, rmask in mask_dict.items():
                is_eye = "eye" in rname.lower()
                row = make_row(
                    rmask, rname, tier_label, tier_label,
                    i, d["pressure_pa"], foc["I_full"],
                    foc["mask_6dB"], foc["mask_3dB"],
                    foc["Isppa"], foc["Isppa_brain"],
                    foc["focus_vox_6"], foc["focus_vox_3"],
                    foc["focus_vol_6dB"], foc["focus_vol_3dB"],
                    vox_vol, meta["freq_hz"], d["affine_sim"],
                    d["temp_c"], meta, is_eye=is_eye)
                if row is None:
                    continue
                row["name"] = rname; row["type"] = tier_label
                row["beam_zone"]      = "—"
                row["along_focus_mm"] = np.nan
                row["lateral_mm"]     = np.nan
                region_rows_raw.append(row)
                # Flag hot regions for contouring
                if (row["Isppa_Target_Wcm2"] > isppa_thr and
                        rname.lower() != cfg["target_name"].lower()):
                    hot_regions[rname] = rmask
                    hot_colors[rname]  = HOT_REGION_PALETTE[col_idx % len(HOT_REGION_PALETTE)]
                    col_idx += 1

        # Classify beam zones in-place
        if region_rows_raw:
            classify_regions_along_beam(region_rows_raw, all_region_masks,
                                        beam, d["affine_sim"])
            # Push updated beam-zone fields into row dicts
            for r in region_rows_raw:
                r["BeamZone"]       = r["beam_zone"]
                r["AlongFocus_mm"]  = r["along_focus_mm"]
                r["LateralOffset_mm"] = r["lateral_mm"]

        region_rows_raw.sort(key=lambda r: r["Isppa_Target_Wcm2"], reverse=True)

        # Verdict
        verdict = build_verdict(tm_row, beam, along, mech, thermal, cfg)

        print(f"  son {i}: verdict={verdict['level']:7s} "
              f"cov={verdict['coverage']}% "
              f"prefocal={int(verdict['prefocal_ratio']*100)}% "
              f"Isppa={foc['Isppa']} "
              f"skull={_f(beam.get('skull_thickness_mm'),1)} mm "
              f"ΔT={thermal.get('dT_peak')}")

        # Figures
        fig_ortho = plot_orthogonal_views(
            foc["I_brain"], foc["mask_6dB"], foc["mask_3dB"],
            target_mask, d["medium_mask"], d["affine_sim"], d["grid_shape"],
            foc["peak_idx"], cfg.get("t1_path"), hot_regions, hot_colors)
        fig_corr = plot_beam_corridor(
            foc["I_full"], tmasks, beam, d["affine_sim"],
            d["grid_shape"], vox_mm, isppa_thr, hot_regions, hot_colors)
        fig_along = plot_along_beam(along, isppa_thr)
        fig_temp  = plot_temperature(d["temp_c"], foc["peak_idx"],
                                     d["affine_sim"], d["grid_shape"])
        fig_mi    = plot_mi_chart(mech, i)

        max_mi = max((r["MI"] for r in mech), default=np.nan)
        sonications.append({
            "idx": i, "meta": meta, "foc": foc, "beam": beam,
            "tm": tm_row, "mech": mech, "thermal": thermal,
            "along": along, "verdict": verdict,
            "region_rows": region_rows_raw,
            "hot_colors":  hot_colors,
            "fig_ortho":   fig_ortho, "fig_corridor": fig_corr,
            "fig_along":   fig_along, "fig_temp":    fig_temp,
            "fig_mi":      fig_mi,
            # summary fields for comparison table
            "Isppa_target": (tm_row.get("Isppa_Target_Wcm2", foc["Isppa"])
                             if "error" not in tm_row else foc["Isppa"]),
            "max_MI":       round(max_mi, 3),
            "dT":           thermal.get("dT_peak", np.nan),
            "skull_mm":     beam.get("skull_thickness_mm"),
            "fwhm_ax":      beam.get("fwhm_axial_mm"),
        })

        # Accumulate CSV rows
        all_csv_rows.extend(region_rows_raw)
        if "error" not in tm_row:
            all_csv_rows.append(tm_row)

    return ({"h5_name": h5_name, "sonications": sonications,
             "tissue_source": tissue_source},
            all_csv_rows)


def write_csv(all_rows: List[Dict], cfg: Dict, out_path: str):
    if not _HAVE_PANDAS or not all_rows:
        return
    ordered = [
        "Sonication","TargetName","TargetPath","ReportingTier",
        "BeamZone","AlongFocus_mm","LateralOffset_mm",
        "Isppa_Overall_Wcm2","Isppa_Brain_Wcm2",
        "Isppa_Target_Wcm2","Imean_Target_Wcm2","TargetVol_mm3",
        "FocusVol_6dB_mm3","FocusVol_3dB_mm3",
        "Coverage_6dB_mm3","Coverage_6dB_pct",
        "Coverage_3dB_mm3","Coverage_3dB_pct",
        "OnTarget_6dB_pct","OffTarget_6dB_pct",
        "OnTarget_3dB_pct","OffTarget_3dB_pct",
        "MeanInt_Overlap_6dB_Wcm2","MeanInt_Overlap_3dB_Wcm2",
        "PeakPressure_Target_kPa","MeanPressure_Target_kPa",
        "PeakP_Overlap_6dB_kPa","MeanP_Overlap_6dB_kPa",
        "PeakP_Overlap_3dB_kPa","MeanP_Overlap_3dB_kPa",
        "MI","MI_Flag","CEM43_Limit_min","PeakP_XYZ_mm",
        "Temp_AtTarget_degC","Temp_AtPeak_degC",
        "CEM43_AtTarget_min","CEM43_SPTP_min",
        "PeakTemp_Region_degC","MeanTemp_Region_degC",
        "PeakTemp_Overlap_6dB_degC","MeanTemp_Overlap_6dB_degC",
        "ThermalFlag",
    ]
    df = pd.DataFrame(all_rows)
    df.insert(0, "participant_id", cfg["participant_id"])
    df.insert(1, "session_id",     cfg["session_id"])
    df.insert(2, "target_name",    cfg["target_name"])
    present = ["participant_id","session_id","target_name"] + [c for c in ordered if c in df.columns]
    df = df[present].sort_values(
        ["Sonication","ReportingTier","Isppa_Target_Wcm2"],
        ascending=[True,True,False]
    ).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  [csv]  {out_path}  ({len(df)} rows)")

# a sonication-level summary CSV
def write_sonication_summary(report: Dict, cfg: Dict, out_path: str):
    """
    One row per sonication with beam-geometry and pre-focal metrics.
    These live at sonication level (not per-region) so they can't go
    in the per-region analysis CSV — dose_response.py reads this separately
    and pivots it alongside the region dose metrics.
    """
    if not _HAVE_PANDAS:
        return
    rows = []
    for s in report["sonications"]:
        rows.append({
            "participant_id":        cfg["participant_id"],
            "session_id":            cfg["session_id"],
            "target_name":           cfg["target_name"],
            "sonication":            s["idx"],
            "verdict":               s["verdict"]["level"],
            # Beam geometry
            "skull_thickness_mm":    s["beam"].get("skull_thickness_mm"),
            "path_length_mm":        s["beam"].get("path_length_mm"),
            "peak_axial_offset_mm":  s["beam"].get("peak_axial_offset_mm"),
            "peak_lateral_mm":       s["beam"].get("peak_lateral_mm"),
            "fwhm_axial_mm":         s["beam"].get("fwhm_axial_mm"),
            "fwhm_lat_mean_mm":      s["beam"].get("fwhm_lat_mean_mm"),
            "elongation":            s["beam"].get("elongation"),
            # Pre-focal
            "prefocal_ratio":        s["along"].get("prefocal_ratio"),
            "prefocal_brain_peak":   s["along"].get("prefocal_brain_peak"),
            "prefocal_brain_pos_mm": s["along"].get("prefocal_brain_pos_mm"),
            # Global intensity reference
            "Isppa_global":          s["foc"]["Isppa"],
            "Isppa_brain":           s["foc"]["Isppa_brain"],
            # Coverage summary (mirrors verdict chips)
            "coverage_6dB_pct":      s["verdict"]["coverage"],
            "on_target_pct":         s["verdict"]["on_target"],
            # Safety
            "max_MI":                s["max_MI"],
            "dT_peak":               s["thermal"].get("dT_peak"),
            "thermal_ok":            s["thermal"].get("thermal_ok"),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  [sum]  {out_path}  ({len(rows)} sonication(s))")

def run():
    cfg = CONFIG
    os.makedirs(cfg["output_dir"], exist_ok=True)
    for h5_path in cfg["h5_files"]:
        if not os.path.exists(h5_path):
            print(f"[skip] not found: {h5_path}"); continue
        report, csv_rows = process_h5(h5_path, cfg)
        stem = report["h5_name"]
        generate_html(report, cfg,
                      os.path.join(cfg["output_dir"], f"{stem}_report.html"))
        write_csv(csv_rows, cfg,
                  os.path.join(cfg["output_dir"],
                               f"{cfg['participant_id']}_{cfg['target_name']}_analysis.csv"))
        write_sonication_summary(report, cfg,
                  os.path.join(cfg["output_dir"],
                               f"{cfg['participant_id']}_{cfg['target_name']}_sonication_summary.csv"))
    print("\nDone.")


if __name__ == "__main__":
    run()