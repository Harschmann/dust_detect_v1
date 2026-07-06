# Camera Dust Inspector (v2 - final)

Basler (webcam / synthetic fallback) live feed -> click to mark camera
centers -> per-model config -> **Capture & Inspect** (button or Space)
runs a multi-channel detector and files the result as NG/OK. Wrong
verdicts can be overridden by the operator, and the built-in **Tuning
Lab** replays saved photos against the current sliders to compute miss
rate (FNR) and false-alarm rate (FPR).

## Run

```
pip install -r requirements.txt        # or in a venv
python main.py
```

No camera needed to try it: Basler -> webcam -> synthetic test pattern,
automatic fallback.

## Files

- `main.py` - UI + app logic (run this)
- `defect_detector.py` - the detection algorithm
- `camera_manager.py` - Basler -> webcam -> synthetic chain
- `canvas_widget.py` - zoom/pan canvas used by every panel
- `storage.py` - model configs + capture folders + relabel

## Detection algorithm (what changed vs plain Z-score)

1. **Robust statistics (median + MAD)** - a big defect can no longer
   inflate the mean/std and hide itself under its own raised threshold.
2. **Local background detrend** (masked boxFilter) - lens vignetting
   no longer reads as fake bright-center / dark-rim anomalies.
3. **Texture channel for same-color AR scratches** - Scharr gradient
   magnitude, averaged 3x3 (a scratch is a *line* of gradient, noise is
   speckle - averaging integrates the line and suppresses the speckle),
   then robust Z-score. Catches scratches whose per-pixel contrast is
   only ~1.5x the noise floor - invisible to any intensity threshold.
4. **Thin-structure-safe noise filtering** - no morphological OPEN
   (that erases 1-2px threads); instead tiny connected components are
   dropped, then CLOSE bridges fragments. Blob size = true pixel count
   (contourArea under-reports thin shapes).
5. **Blob classification** by channel + elongation:
   dust/glue = red, thread = yellow, same-color scratch = magenta.
   (Glue lands in the dust class - both are compact white blobs;
   separating them reliably needs stroke-width logic.)
6. **Optional colored-reflection gate** (OFF by default) - ignores
   pixels saturated well above the ROI's median saturation. Rescue
   switch if AR coating reflections cause false alarms; note a scratch
   inside a colored region is then missed too.

Defaults (sigma_i=3.0, sigma_t=5.5, min 6 px, edge margin 6) were
validated on synthetic lenses: 0 false NG over clean seeds, 8/8 catch
of 1.5-sigma-contrast scratches that intensity-only misses 5/5.

## Workflow

1. Model name type/select -> live feed pe camera centers click karo,
   radius tune karo (Shift+Scroll ya slider).
2. **Save** - ROIs + all thresholds together under that model.
3. **Capture & Inspect** (ya Space) - verdict popup, original vs
   inspected side-by-side (zoom/pan synced).
4. Verdict galat lage to popup me **flip** dabao - files sahi folder
   me move ho jaati hain, isliye NG/OK folders hamesha human-confirmed
   truth rakhte hain.
5. **Tuning Lab** - saved originals browse karo, anomaly heatmap dekho,
   sliders change karke *Re-run*, phir **Evaluate ALL** se FNR/FPR
   + misclassified list + rule-of-3 note. Report export bhi hai.

## Folder layout

```
inspection_data/<model>/original/NG|OK/<model>_test<N>_<NG|OK>.png
inspection_data/<model>/after/NG|OK/...          (class-colored circles)
inspection_data/<model>/eval_report_*.txt
configs/<model>.json                              (ROIs + thresholds)
```

`testN` per model unique rehta hai (max-scan, count nahi) - relabel ya
delete ke baad bhi collision nahi.

## Controls (every image panel)

| Action | Effect |
|---|---|
| Click (no drag) | Add / select ROI (live feed only) |
| Drag | Pan |
| Scroll | Zoom at cursor |
| Shift+Scroll, `[` `]`, Up/Down | Selected ROI radius |
| Delete / Backspace | Remove selected ROI |
| Space | Capture & Inspect |

