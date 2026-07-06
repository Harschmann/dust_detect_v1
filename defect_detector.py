"""
defect_detector.py  (v2)
-------------------------
Multi-channel anomaly detection for camera-module inspection.

Why v2 - three real weaknesses of the plain global Z-score:

1.  ROBUST STATISTICS. mean/std get inflated by the defect itself
    (a big glue blob raises std -> threshold rises -> defect hides its
    own tail). v2 uses median + MAD (median absolute deviation), which
    barely move even when a defect covers a few % of the ROI.

2.  LOCAL DETREND, ONE-SIDED. Real lenses vignette - the rim is darker
    than the center even when perfectly clean. A single global mean
    makes the rim look "dark-anomalous". v2 subtracts a masked local
    mean (boxFilter) first, so only *local* deviations count - and
    only checks the BRIGHT side. Dust, thread, and glue are all
    white/bright against the module surface, and in some models the
    coating itself already looks whitish (that's exactly why this is a
    relative z-score check, not a fixed brightness cutoff - only a spot
    brighter than its own local surroundings counts). Dark deviations
    are never the actual defect here, so they're not flagged at all -
    this also quietly drops a lot of false alarms from lighting
    artifacts that have a dark side (shadows, the dim side of an arc).

3.  TEXTURE CHANNEL for same-color scratches. A scratch in the AR
    coating can have the SAME average brightness as the coating around
    it - invisible to any intensity threshold - but it still creates a
    coherent local gradient (one flank catches light, the other loses
    it). v2 runs Scharr gradient magnitude and Z-scores *that* inside
    the ROI. Uniform coating = low flat gradient; scratch = a bright
    line in gradient space even at zero mean-intensity difference.

Every surviving blob is classified:

    dust    - intensity anomaly, compact            (red)
    thread  - intensity anomaly, elongated          (yellow)
    scratch - texture-only anomaly (same-color)     (magenta)

Note: glue residue is white and compact, so it lands in "dust" - a
reliable glue-vs-dust split needs the circularity / stroke-width logic
and isn't attempted here. Everything found counts toward NG either way.

KNOWN OPEN ISSUE (not attempted here, revisit later): some models show
curved lighting-reflection arcs and/or concentric rings from the lens's
own construction, both fixed patterns tied to radius from the ROI
center rather than real defects. A radial-symmetry baseline was tried
and reverted - it starves for pixels near the ROI center (a real dust
speck there can dominate its own radius bin and hide itself) and
assumes the lighting is perfectly axi-symmetric, which real jigs often
aren't - both made real-photo detection worse, not better. Fixing the
arc/ring false-positives properly needs its own pass (e.g. a golden-
reference frame per model to subtract, since that cancels any fixed
pattern - radial, arc-shaped, or not - without these failure modes).

Optional extras:
    - edge margin: shrink each ROI a few px so the lens rim's natural
      bright/gradient ring never fires.
    - colored-reflection suppression: AR coatings throw magenta/green
      reflections; real dust/glue/thread is white/gray. If enabled,
      strongly *saturated* (colored) pixels are ignored by BOTH
      channels. Trade-off: a scratch lying inside a colored region is
      then missed too - so this stays OFF by default (with the vinyl
      removed the photo should be clean anyway) and is a rescue switch
      for lines where coating reflections cause false alarms.
"""

import cv2
import numpy as np

CLASS_COLORS_BGR = {
    "dust": (0, 0, 255),        # red
    "thread": (0, 255, 255),    # yellow
    "scratch": (255, 0, 255),   # magenta
}

DEFAULT_PARAMS = dict(
    sigma_intensity=3.0,     # |z| threshold on detrended intensity
    sigma_texture=5.5,       # z threshold on smoothed gradient magnitude (validated: 0 clean FP, catches 1.5-sigma-contrast scratches)
    min_pixel_area=6,        # contour area (px^2) below which a blob is noise
    edge_margin_px=6,        # shrink every ROI radius by this before analysis
    texture_enabled=True,    # the same-color-scratch channel
    suppress_colored=False,  # ignore strongly colored (saturated) areas entirely
    saturation_threshold=25, # "colored" if S exceeds the ROI median S by this much
    local_mean_window=0,     # 0 = auto (scales with each ROI's radius); >0 forces a fixed px window
    elongation_thread=3.5,   # minAreaRect long/short ratio => thread
)


class DetectionResult:
    def __init__(self):
        self.defects = []        # [{x,y,r,area,cls,roi_index,channels}]
        self.verdict = "OK"
        self.annotated = None    # BGR with ROI outlines + class-colored circles
        self.heatmap = None      # BGR anomaly heatmap (only if requested)
        self.class_counts = {}   # {"dust": n, "thread": n, "scratch": n}
        self.per_roi = []        # [{"roi_index": i, "count": n}]

    def summary(self):
        if not self.defects:
            return "no defects"
        return ", ".join(f"{v} {k}" for k, v in self.class_counts.items() if v)


# ----------------------------------------------------------------- helpers
def _robust_center_scale(values):
    """median + 1.4826*MAD; falls back to std if MAD collapses."""
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    scale = 1.4826 * mad
    if scale < 0.5:
        scale = max(float(np.std(values)), 1e-3)
    return med, scale


def _masked_local_mean(img_f32, mask_f32, window):
    """boxFilter local mean, masked to the ROI. Models slowly-varying
    background (vignetting, gentle lighting falloff) at the scale of
    `window`, without needing full radial symmetry - unlike a pure
    radius-based baseline, this still works if lighting isn't axi-
    symmetric, and doesn't risk a real defect near the ROI center
    dominating its own (tiny-population) radius bin and hiding itself."""
    k = (window, window)
    num = cv2.boxFilter(img_f32 * mask_f32, -1, k, normalize=False)
    den = cv2.boxFilter(mask_f32, -1, k, normalize=False)
    return num / np.maximum(den, 1e-6)


def _drop_small_components(mask_u8, min_px):
    """Remove connected components smaller than min_px pixels (kills
    isolated noise pixels without eroding thin lines like OPEN would)."""
    if min_px <= 1 or not mask_u8.any():
        return mask_u8
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for lbl in range(1, n_lbl):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_px:
            out[labels == lbl] = 255
    return out


def _roi_mask(shape_hw, cx, cy, r):
    m = np.zeros(shape_hw, dtype=np.uint8)
    cv2.circle(m, (int(round(cx)), int(round(cy))), max(int(round(r)), 1), 255, -1)
    return m


# ------------------------------------------------------------ single ROI
def _analyze_roi(gray_f32, sat_u8, cx, cy, r_analysis, p):
    """Returns (blobs, zmap_int, zmap_tex, mask) for one ROI.
    z-maps are full-frame float32, zero outside the ROI (for heatmap)."""
    h, w = gray_f32.shape
    mask = _roi_mask((h, w), cx, cy, r_analysis)
    inside = mask > 0
    if int(np.count_nonzero(inside)) < 100:
        return [], None, None, mask

    # ---- intensity channel: local detrend, then robust z - ONE-SIDED
    # (bright only). Dust, thread, and glue are all white/bright against
    # the module surface; the coating itself can look whitish too, which
    # is exactly why this is a relative (z-score) check and not a fixed
    # brightness cutoff - only a spot that's brighter than its own local
    # surroundings counts. Dark deviations are intentionally ignored:
    # they're never the defect here, only ever noise or a lighting
    # artifact's shadow side.
    win = int(p["local_mean_window"])
    if win <= 0:
        # auto: tie the detrend window to this lens's size in pixels, so
        # it works at any camera resolution. ~0.6x radius is far larger
        # than any dust speck or thread width (those survive the detrend)
        # but small enough to still flatten lens-scale lighting falloff.
        win = int(round(r_analysis * 0.6))
    win = max(15, win)
    if win % 2 == 0:
        win += 1
    mask_f = (mask.astype(np.float32)) / 255.0
    local_mean = _masked_local_mean(gray_f32, mask_f, win)
    detr = gray_f32 - local_mean
    vals = detr[inside]
    med, scale = _robust_center_scale(vals)
    z_int = np.zeros_like(gray_f32)
    z_int[inside] = (detr[inside] - med) / scale
    int_mask = ((z_int > p["sigma_intensity"]) & inside).astype(np.uint8) * 255

    colored = None
    colored_tex = None
    if p["suppress_colored"] and sat_u8 is not None:
        # Adaptive per-pixel gate: "colored" means saturated well ABOVE
        # this ROI's own median saturation. Intensity is gated strictly
        # per-pixel, so white dust sitting ON a colored sheen keeps its
        # own unsaturated pixels and is still caught. The texture gate
        # is dilated because a sheen's gradient slope extends slightly
        # past its colored pixels.
        med_s = float(np.median(sat_u8[inside]))
        colored = (sat_u8.astype(np.float32) > med_s + p["saturation_threshold"]) & inside
        int_mask[colored] = 0
        colored_u8 = colored.astype(np.uint8) * 255
        colored_tex = cv2.dilate(colored_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0

    # ---- texture channel: Scharr gradient magnitude, robust z (kept as
    # validated before - deprioritized for now, revisit with the arc/ring
    # problem later)
    z_tex = np.zeros_like(gray_f32)
    tex_mask = np.zeros((h, w), dtype=np.uint8)
    if p["texture_enabled"]:
        blur = cv2.GaussianBlur(gray_f32, (3, 3), 0)
        gx = cv2.Scharr(blur, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(blur, cv2.CV_32F, 0, 1)
        gmag = cv2.magnitude(gx, gy)
        # Coherence trick: a scratch is a LINE of gradient, noise is
        # isolated speckle. Averaging the magnitude over 3x3 integrates
        # the collinear signal (~keeps it) while incoherent noise drops
        # by ~sqrt(9) -> big SNR gain exactly for faint scratches.
        gmag = cv2.blur(gmag, (3, 3))
        gvals = gmag[inside]
        gmed, gscale = _robust_center_scale(gvals)
        z_tex[inside] = (gmag[inside] - gmed) / gscale
        tex_mask = ((z_tex > p["sigma_texture"]) & inside).astype(np.uint8) * 255
        if colored_tex is not None:
            tex_mask[colored_tex] = 0

    # ---- noise vs thin-structure handling:
    # A morphological OPEN would erase 1-2px threads/scratches, but pure
    # Gaussian noise leaves ~0.3% isolated pixels above 3 sigma, and a
    # CLOSE would merge those into fake blobs. So instead: drop tiny
    # connected components (noise singles/pairs) per channel first, THEN
    # close to bridge fragments of genuine long thin defects.
    int_mask = _drop_small_components(int_mask, 3)
    tex_mask = _drop_small_components(tex_mask, 5)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    int_mask = cv2.morphologyEx(int_mask, cv2.MORPH_CLOSE, k3)
    tex_mask = cv2.morphologyEx(tex_mask, cv2.MORPH_CLOSE, k5)
    combined = cv2.bitwise_or(int_mask, tex_mask)

    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    blobs = []
    for lbl in range(1, n_lbl):
        npix = int(stats[lbl, cv2.CC_STAT_AREA])   # true pixel count, honest for thin shapes
        if npix < p["min_pixel_area"]:
            continue
        ys, xs = np.where(labels == lbl)
        pts = np.column_stack([xs, ys]).astype(np.int32)
        blob_sel = labels == lbl
        frac_int = np.count_nonzero(int_mask[blob_sel]) / npix
        frac_tex = np.count_nonzero(tex_mask[blob_sel]) / npix

        (_, (rw, rh), _) = cv2.minAreaRect(pts)
        elong = max(rw, rh) / max(min(rw, rh), 1.0)

        channels = []
        if frac_int > 0.15:
            channels.append("intensity")
        if frac_tex > 0.15:
            channels.append("texture")
        if not channels:
            channels = ["texture"] if frac_tex >= frac_int else ["intensity"]

        if "intensity" not in channels:
            cls = "scratch"
        elif elong >= p["elongation_thread"]:
            cls = "thread"
        else:
            cls = "dust"

        (bx, by), br = cv2.minEnclosingCircle(pts)
        blobs.append({
            "x": float(bx), "y": float(by), "r": max(float(br), 3.0),
            "area": float(npix), "cls": cls, "elongation": float(elong),
            "channels": channels,
        })
    return blobs, z_int, z_tex, mask


# --------------------------------------------------------------- full run
def run_full_inspection(frame, rois, params=None, make_heatmap=False):
    """
    frame: BGR (H,W,3) or grayscale (H,W).
    rois:  [{"cx","cy","r"}, ...] full-frame pixel coords.
    params: dict overriding DEFAULT_PARAMS keys.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update({k: v for k, v in params.items() if v is not None})

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sat = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1] if p["suppress_colored"] else None
        annotated = frame.copy()
    else:
        gray = frame
        sat = None
        annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    gray_f32 = gray.astype(np.float32)

    # Resolution-scaled pre-blur: coating grain that's just sensor/surface
    # texture produces per-pixel speckle that lights up the z-score at
    # high resolution. A blur proportional to resolution smooths that
    # grain (a real dust speck is many pixels and survives) so we don't
    # drown in tiny false detects. Reference resolution is 1280px wide -
    # at/below that this is a no-op, matching the tuned low-res behavior.
    res_scale = max(gray.shape[0], gray.shape[1]) / 1280.0
    kb = int(round(res_scale))
    if kb % 2 == 0:
        kb += 1
    if kb >= 3:
        gray_f32 = cv2.GaussianBlur(gray_f32, (kb, kb), 0)

    result = DetectionResult()
    score = np.zeros_like(gray_f32) if make_heatmap else None

    for i, roi in enumerate(rois):
        r_analysis = max(roi["r"] - p["edge_margin_px"], 4)
        blobs, z_int, z_tex, mask = _analyze_roi(gray_f32, sat, roi["cx"], roi["cy"], r_analysis, p)
        for b in blobs:
            b["roi_index"] = i
        result.defects.extend(blobs)
        result.per_roi.append({"roi_index": i, "count": len(blobs)})

        cv2.circle(annotated, (int(roi["cx"]), int(roi["cy"])), int(roi["r"]), (200, 160, 60), 1)
        if make_heatmap and z_int is not None:
            s = np.maximum(np.abs(z_int) / p["sigma_intensity"],
                           z_tex / max(p["sigma_texture"], 1e-6))
            score = np.maximum(score, np.where(mask > 0, s, 0))

    for b in result.defects:
        color = CLASS_COLORS_BGR[b["cls"]]
        cv2.circle(annotated, (int(b["x"]), int(b["y"])), int(b["r"]) + 4, color, 2)

    counts = {"dust": 0, "thread": 0, "scratch": 0}
    for b in result.defects:
        counts[b["cls"]] += 1
    result.class_counts = counts
    result.verdict = "NG" if result.defects else "OK"

    # small legend so saved "after" images are self-explanatory
    y0 = 22
    for name, col in CLASS_COLORS_BGR.items():
        cv2.circle(annotated, (14, y0 - 4), 5, col, -1)
        cv2.putText(annotated, name, (26, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (235, 235, 235), 1, cv2.LINE_AA)
        y0 += 20
    result.annotated = annotated

    if make_heatmap:
        s8 = np.uint8(np.clip(score / 2.0, 0, 1) * 255)   # 2x threshold = full hot
        heat = cv2.applyColorMap(s8, cv2.COLORMAP_INFERNO)
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result.heatmap = cv2.addWeighted(base, 0.35, heat, 0.65, 0)
    return result
