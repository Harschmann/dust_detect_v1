"""
main.py  (v2)
--------------
Samsung camera-module inspection tool - final build.

Run:
    python main.py

Live feed (Basler -> webcam -> synthetic fallback) -> click to mark
camera centers -> per-model config (ROIs + all detection thresholds)
-> CAPTURE & INSPECT (button or Space) runs the multi-channel detector
and files the result away as NG/OK. The operator can override a wrong
verdict from the result popup; overrides move the files, so the NG/OK
folders always hold the HUMAN-confirmed truth. The Tuning Lab replays
saved originals against the current sliders, shows an anomaly heatmap,
and computes miss rate (FNR) / false-alarm rate (FPR) over everything
captured so far.

Defect classes drawn on the "after" image:
    dust / glue  - red      (compact intensity anomaly)
    thread       - yellow   (elongated intensity anomaly)
    scratch      - magenta  (texture-only anomaly, same-color AR scratch)
"""

import os
import threading
from datetime import datetime

import cv2
import customtkinter as ctk
from tkinter import messagebox, filedialog

from camera_manager import CameraManager
from defect_detector import run_full_inspection, DEFAULT_PARAMS
from storage import Storage
from canvas_widget import ViewState, ImageCanvas

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Palette: deep charcoal + optics-glass cyan; unambiguous red/green verdicts.
BG = "#15171a"
PANEL = "#1b1e22"
ACCENT = "#3fb8d6"
ACCENT_HOVER = "#2f96b0"
NG_COLOR = "#ef4444"
OK_COLOR = "#2fd66b"
MUTED = "#8a8f98"

ROI_COLOR = (210, 160, 40)      # BGR - unselected ROI outline
ROI_SELECTED = (0, 255, 255)    # BGR - selected ROI outline


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Camera Dust Inspector")
        self.geometry("1440x900")
        self.minsize(1160, 720)
        self.configure(fg_color=BG)

        self.storage = Storage()
        self.camera = CameraManager()
        self.camera.start()

        self.rois = []                # [{"cx","cy","r"}, ...] native frame coords
        self.selected_idx = None
        self.roi_row_widgets = []
        self._param_widgets = {}      # key -> (value_label, fmt, var)
        self._lab = None
        self._last_result_win = None
        self.static_image = None      # loaded-from-disk BGR frame, overrides live feed when set
        self.static_name = ""

        self.current_model = ctk.StringVar(value="")
        self.sigma_intensity = ctk.DoubleVar(value=DEFAULT_PARAMS["sigma_intensity"])
        self.sigma_texture = ctk.DoubleVar(value=DEFAULT_PARAMS["sigma_texture"])
        self.min_area = ctk.IntVar(value=DEFAULT_PARAMS["min_pixel_area"])
        self.edge_margin = ctk.IntVar(value=DEFAULT_PARAMS["edge_margin_px"])
        self.texture_enabled = ctk.BooleanVar(value=DEFAULT_PARAMS["texture_enabled"])
        self.suppress_colored = ctk.BooleanVar(value=DEFAULT_PARAMS["suppress_colored"])

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<space>", self._on_space)
        self._refresh_model_list()
        self._tick()

    # ------------------------------------------------------------- params
    def current_detect_params(self):
        return {
            "sigma_intensity": float(self.sigma_intensity.get()),
            "sigma_texture": float(self.sigma_texture.get()),
            "min_pixel_area": int(self.min_area.get()),
            "edge_margin_px": int(self.edge_margin.get()),
            "texture_enabled": bool(self.texture_enabled.get()),
            "suppress_colored": bool(self.suppress_colored.get()),
        }

    def _apply_detect_params(self, p):
        self.sigma_intensity.set(p.get("sigma_intensity", DEFAULT_PARAMS["sigma_intensity"]))
        self.sigma_texture.set(p.get("sigma_texture", DEFAULT_PARAMS["sigma_texture"]))
        self.min_area.set(int(p.get("min_pixel_area", DEFAULT_PARAMS["min_pixel_area"])))
        self.edge_margin.set(int(p.get("edge_margin_px", DEFAULT_PARAMS["edge_margin_px"])))
        self.texture_enabled.set(bool(p.get("texture_enabled", DEFAULT_PARAMS["texture_enabled"])))
        self.suppress_colored.set(bool(p.get("suppress_colored", DEFAULT_PARAMS["suppress_colored"])))
        self._refresh_param_labels()

    def _refresh_param_labels(self):
        for lbl, fmt, var in self._param_widgets.values():
            lbl.configure(text=fmt(var.get()))

    # ================================================================ UI
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(self, height=52, corner_radius=0, fg_color=PANEL)
        top.grid(row=0, column=0, columnspan=2, sticky="ew")
        ctk.CTkLabel(top, text="\U0001F52C  Camera Dust Inspector",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(side="left", padx=16, pady=10)
        self.status_label = ctk.CTkLabel(top, text="", font=ctk.CTkFont(size=13), text_color=MUTED)
        self.status_label.pack(side="left", padx=10)
        ctk.CTkButton(top, text="\U0001F504 Reconnect Camera", width=150, fg_color="transparent",
                      border_width=1, border_color=MUTED, hover_color="#262a2f",
                      command=self.camera.reconnect).pack(side="right", padx=(6, 16))
        ctk.CTkButton(top, text="\U0001F9EA Tuning Lab", width=120, fg_color="transparent",
                      border_width=1, border_color=ACCENT, text_color=ACCENT, hover_color="#1f3a42",
                      command=self._open_lab).pack(side="right", padx=6)
        self.back_live_btn = ctk.CTkButton(top, text="\U0001F534 Back to Live Feed", width=160,
                      fg_color="transparent", border_width=1, border_color=MUTED, hover_color="#262a2f",
                      command=self._back_to_live)
        ctk.CTkButton(top, text="\U0001F4C1 Open Image", width=140, fg_color="transparent",
                      border_width=1, border_color=MUTED, hover_color="#262a2f",
                      command=self._open_image_file).pack(side="right", padx=6)

        canvas_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=BG)
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        self.view_state = ViewState()
        self.live_canvas = ImageCanvas(
            canvas_frame, view_state=self.view_state, editable=True,
            on_click=self._on_canvas_click,
            on_radius_change=self._on_radius_nudge,
            on_delete=self._delete_selected_roi)
        self.live_canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 0))

        zoom_bar = ctk.CTkFrame(canvas_frame, fg_color="transparent")
        zoom_bar.grid(row=1, column=0, sticky="ew", pady=8, padx=10)
        ctk.CTkButton(zoom_bar, text="-", width=34, command=self._zoom_out).pack(side="left", padx=3)
        ctk.CTkButton(zoom_bar, text="\u2922 Fit", width=64,
                      command=self.live_canvas.fit_to_window).pack(side="left", padx=3)
        ctk.CTkButton(zoom_bar, text="+", width=34, command=self._zoom_in).pack(side="left", padx=3)
        ctk.CTkLabel(
            zoom_bar,
            text="Click = add/select ROI  \u2022  Drag = pan  \u2022  Scroll = zoom  \u2022  "
                 "Shift+Scroll = radius  \u2022  Del = remove  \u2022  Space = capture",
            text_color=MUTED, font=ctk.CTkFont(size=11)).pack(side="left", padx=16)

        sidebar = ctk.CTkScrollableFrame(self, width=340, corner_radius=0, fg_color=PANEL,
                                          scrollbar_button_color="#33373d")
        sidebar.grid(row=1, column=1, sticky="ns")
        self._build_sidebar(sidebar)

    def _section(self, parent, title):
        ctk.CTkLabel(parent, text=title.upper(), font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=ACCENT).pack(anchor="w", padx=6, pady=(18, 6))

    def _slider(self, parent, key, title, var, from_, to, fmt, steps=None):
        ctk.CTkLabel(parent, text=title, font=ctk.CTkFont(size=11)).pack(anchor="w", padx=6, pady=(8, 0))
        val = ctk.CTkLabel(parent, text=fmt(var.get()), text_color=MUTED, font=ctk.CTkFont(size=11))
        kwargs = {"number_of_steps": steps} if steps else {}
        ctk.CTkSlider(parent, from_=from_, to=to, variable=var,
                      command=lambda v, l=val, f=fmt: l.configure(text=f(float(v))),
                      **kwargs).pack(fill="x", padx=6)
        val.pack(anchor="e", padx=6)
        self._param_widgets[key] = (val, fmt, var)

    def _build_sidebar(self, parent):
        self._section(parent, "Phone model")
        self.model_combo = ctk.CTkComboBox(parent, values=[], variable=self.current_model,
                                            command=lambda _c: self._load_current_model())
        self.model_combo.pack(fill="x", padx=6)
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=6, padx=6)
        ctk.CTkButton(row, text="\U0001F4C2 Load", command=self._load_current_model
                      ).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ctk.CTkButton(row, text="\U0001F4BE Save", fg_color=ACCENT, hover_color=ACCENT_HOVER,
                      command=self._save_current_model).pack(side="left", expand=True, fill="x", padx=3)
        ctk.CTkButton(row, text="\U0001F5D1", width=36, fg_color="#5c1f1f", hover_color="#7a2828",
                      command=self._delete_current_model).pack(side="left", padx=(3, 0))
        self.save_status_label = ctk.CTkLabel(parent, text="", text_color=OK_COLOR,
                                               font=ctk.CTkFont(size=11))
        self.save_status_label.pack(anchor="w", padx=6)

        self._section(parent, "ROI points")
        self.roi_list_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.roi_list_frame.pack(fill="x", padx=6)
        self.radius_slider = ctk.CTkSlider(parent, from_=5, to=400, command=self._on_radius_slider)
        self.radius_slider.set(50)
        self.radius_slider.pack(fill="x", padx=6, pady=(10, 0))
        self.radius_value_label = ctk.CTkLabel(parent, text="Select an ROI to edit its radius",
                                                text_color=MUTED, font=ctk.CTkFont(size=11))
        self.radius_value_label.pack(anchor="w", padx=6)
        self._rebuild_roi_list()

        self._section(parent, "Detection settings")
        self._slider(parent, "sigma_intensity", "Intensity sigma (white/dark blobs)",
                     self.sigma_intensity, 1.5, 6.0, lambda v: f"{v:.1f}")
        self._slider(parent, "sigma_texture", "Texture sigma (same-color scratches)",
                     self.sigma_texture, 3.0, 9.0, lambda v: f"{v:.1f}")
        sw_row = ctk.CTkFrame(parent, fg_color="transparent")
        sw_row.pack(fill="x", padx=6, pady=(8, 0))
        ctk.CTkSwitch(sw_row, text="Scratch (texture) channel", variable=self.texture_enabled,
                      progress_color=ACCENT).pack(anchor="w")
        ctk.CTkSwitch(sw_row, text="Ignore colored AR reflections", variable=self.suppress_colored,
                      progress_color=ACCENT).pack(anchor="w", pady=(6, 0))
        self._slider(parent, "min_pixel_area", "Min defect size (pixels)",
                     self.min_area, 1, 200, lambda v: str(int(v)))
        self._slider(parent, "edge_margin_px", "Edge margin (px ignored at ROI rim)",
                     self.edge_margin, 0, 40, lambda v: str(int(v)), steps=40)

        self._section(parent, "Inspect")
        self.capture_btn = ctk.CTkButton(parent, text="\U0001F4F8  CAPTURE & INSPECT", height=46,
                      fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#0c1114",
                      font=ctk.CTkFont(size=14, weight="bold"),
                      command=self._capture_and_inspect)
        self.capture_btn.pack(fill="x", padx=6, pady=4)

        self._section(parent, "Stats (human-confirmed)")
        self.stats_label = ctk.CTkLabel(parent, text="No captures yet", justify="left", anchor="w",
                                         font=ctk.CTkFont(size=12))
        self.stats_label.pack(fill="x", padx=6)

        self._section(parent, "Recent captures")
        self.log_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.log_frame.pack(fill="x", padx=6, pady=(0, 20))
        self.log_rows = []

    # ==================================================== model management
    def _refresh_model_list(self):
        self.model_combo.configure(values=self.storage.list_models())

    def _load_current_model(self):
        model = self.current_model.get().strip()
        if not model:
            return
        cfg = self.storage.load_model_config(model)
        if cfg is None:
            self.rois = []
        else:
            self.rois = cfg.get("rois", [])
            self._apply_detect_params(cfg.get("detect_params", {}))
        self.selected_idx = None
        self._rebuild_roi_list()
        self._update_stats()

    def _save_current_model(self):
        model = self.current_model.get().strip()
        if not model:
            messagebox.showwarning("Model name required", "Pehle phone model ka naam type ya select karo.")
            return
        self.storage.save_model_config(model, self.rois, self.current_detect_params())
        self._refresh_model_list()
        self.current_model.set(model)
        self._update_stats()
        self.save_status_label.configure(text=f"\u2713 Saved '{model}' - {len(self.rois)} ROI(s) + thresholds")
        self.after(2500, lambda: self.save_status_label.configure(text=""))

    def _delete_current_model(self):
        model = self.current_model.get().strip()
        if not model:
            return
        if not messagebox.askyesno("Delete config",
                                    f"Delete saved config for '{model}'?\n(captured photos are NOT deleted)"):
            return
        self.storage.delete_model_config(model)
        self._refresh_model_list()
        self.current_model.set("")
        self.rois, self.selected_idx = [], None
        self._rebuild_roi_list()

    # ========================================================= ROI editing
    def _on_canvas_click(self, nx, ny, cx, cy):
        hit = self._hit_test(cx, cy)
        if hit is not None:
            self.selected_idx = hit
        else:
            img = self.live_canvas.source_image
            base = min(img.shape[1], img.shape[0]) if img is not None else 960
            self.rois.append({"cx": nx, "cy": ny, "r": max(20, int(base * 0.04))})
            self.selected_idx = len(self.rois) - 1
        self._rebuild_roi_list()

    def _hit_test(self, cx, cy, tol=14):
        best, best_d = None, tol
        for i, r in enumerate(self.rois):
            rx, ry = self.view_state.to_canvas(r["cx"], r["cy"])
            d = ((rx - cx) ** 2 + (ry - cy) ** 2) ** 0.5
            if d < best_d:
                best, best_d = i, d
        return best

    def _on_radius_nudge(self, delta_px):
        if self.selected_idx is None or self.selected_idx >= len(self.rois):
            return
        r = self.rois[self.selected_idx]
        r["r"] = max(4, r["r"] + delta_px)
        self._rebuild_roi_list()

    def _on_radius_slider(self, value):
        if self.selected_idx is None or self.selected_idx >= len(self.rois):
            return
        self.rois[self.selected_idx]["r"] = float(value)
        self.radius_value_label.configure(
            text=f"ROI #{self.selected_idx + 1} radius: {int(float(value))}px")
        if self.selected_idx < len(self.roi_row_widgets):
            self.roi_row_widgets[self.selected_idx].configure(
                text=f"#{self.selected_idx + 1}   r = {int(float(value))}px")

    def _delete_selected_roi(self):
        if self.selected_idx is None or self.selected_idx >= len(self.rois):
            return
        del self.rois[self.selected_idx]
        self.selected_idx = None
        self._rebuild_roi_list()

    def _select_roi(self, idx):
        self.selected_idx = idx
        self._rebuild_roi_list()

    def _delete_roi_at(self, idx):
        del self.rois[idx]
        if self.selected_idx == idx:
            self.selected_idx = None
        elif self.selected_idx is not None and self.selected_idx > idx:
            self.selected_idx -= 1
        self._rebuild_roi_list()

    def _rebuild_roi_list(self):
        for w in self.roi_list_frame.winfo_children():
            w.destroy()
        self.roi_row_widgets = []
        if not self.rois:
            ctk.CTkLabel(self.roi_list_frame, text="Click the live feed to add one.",
                         text_color=MUTED, font=ctk.CTkFont(size=11)).pack(anchor="w", pady=2)
        for i, r in enumerate(self.rois):
            selected = i == self.selected_idx
            row = ctk.CTkFrame(self.roi_list_frame,
                               fg_color=("#1f3a42" if selected else "transparent"))
            row.pack(fill="x", pady=1)
            btn = ctk.CTkButton(row, text=f"#{i + 1}   r = {int(r['r'])}px", anchor="w",
                                fg_color="transparent",
                                text_color=(ACCENT if selected else "gray85"),
                                hover_color="#262a2f",
                                command=lambda idx=i: self._select_roi(idx))
            btn.pack(side="left", expand=True, fill="x")
            self.roi_row_widgets.append(btn)
            ctk.CTkButton(row, text="\u2715", width=28, fg_color="transparent",
                          hover_color="#7a2828",
                          command=lambda idx=i: self._delete_roi_at(idx)).pack(side="right")
        if self.selected_idx is not None and self.selected_idx < len(self.rois):
            self.radius_slider.set(self.rois[self.selected_idx]["r"])
            self.radius_value_label.configure(
                text=f"ROI #{self.selected_idx + 1} radius: {int(self.rois[self.selected_idx]['r'])}px")
        else:
            self.radius_value_label.configure(text="Select an ROI to edit its radius")

    # =============================================================== zoom
    def _zoom_in(self):
        self.view_state.zoom_at(self.live_canvas.winfo_width() / 2,
                                 self.live_canvas.winfo_height() / 2, 1.2)

    def _zoom_out(self):
        self.view_state.zoom_at(self.live_canvas.winfo_width() / 2,
                                 self.live_canvas.winfo_height() / 2, 1 / 1.2)

    # ======================================================= image source
    def _open_image_file(self):
        path = filedialog.askopenfilename(
            title="Open an already-captured image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")])
        if not path:
            return
        self._load_image_path(path)

    def _load_image_path(self, path):
        """Loads a static image from disk and switches the live canvas into
        static mode: same ROI click/zoom/pan controls, just frozen on this
        file instead of the live feed, until Back to Live is pressed."""
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Load failed", f"Ye file image ki tarah nahi khuli:\n{path}")
            return False
        self.static_image = img
        self.static_name = os.path.basename(path)
        self.back_live_btn.pack(side="right", padx=6)
        self.capture_btn.configure(text="\U0001F50D  INSPECT THIS IMAGE")
        self.after(20, self.live_canvas.fit_to_window)
        return True

    def _back_to_live(self):
        self.static_image = None
        self.static_name = ""
        self.back_live_btn.pack_forget()
        self.capture_btn.configure(text="\U0001F4F8  CAPTURE & INSPECT")
        self.after(20, self.live_canvas.fit_to_window)

    # ==================================================== capture + inspect
    def _on_space(self, _event):
        w = self.focus_get()
        if w is not None and w.winfo_class() in ("Entry", "TEntry", "TCombobox", "Text", "Spinbox"):
            return  # typing somewhere - don't hijack the spacebar
        self._capture_and_inspect()

    def _capture_and_inspect(self):
        model = self.current_model.get().strip()
        if not model:
            messagebox.showwarning("Model name required", "Pehle phone model select ya type karo.")
            return
        if not self.rois:
            messagebox.showwarning("No ROI", "Kam se kam ek ROI point set karo (live feed pe click karke).")
            return
        if self.static_image is not None:
            frame = self.static_image
        else:
            frame = self.camera.get_frame()
            if frame is None:
                messagebox.showwarning("No frame", "Camera se frame nahi mila.")
                return
        try:
            result = run_full_inspection(frame, self.rois, self.current_detect_params())
            saved = self.storage.save_capture(model, frame, result.annotated, result.verdict)
        except Exception as e:
            messagebox.showerror("Inspection failed", f"Kuch galat ho gaya:\n{e}")
            return
        self._update_stats()
        color = NG_COLOR if result.verdict == "NG" else OK_COLOR
        detail = result.summary() if result.defects else "clean"
        self._add_log_row(f"test{saved['index']}  \u2022  {result.verdict}  \u2022  {detail}", color)
        self._last_result_win = ResultWindow(self, frame, result, saved, model)

    def _update_stats(self):
        model = self.current_model.get().strip()
        if not model:
            self.stats_label.configure(text="No captures yet")
            return
        s = self.storage.get_stats(model)
        self.stats_label.configure(
            text=f"Model: {model}\nTotal: {s['total']}    NG: {s['ng']}    OK: {s['ok']}\n"
                 f"NG rate: {s['ng_rate']:.1f}%")

    def _add_log_row(self, text, color):
        ts = datetime.now().strftime("%H:%M:%S")
        row = ctk.CTkLabel(self.log_frame, text=f"{text}  \u2022  {ts}",
                            text_color=color, anchor="w", font=ctk.CTkFont(size=11))
        if self.log_rows:
            row.pack(fill="x", anchor="w", before=self.log_rows[0])
        else:
            row.pack(fill="x", anchor="w")
        self.log_rows.insert(0, row)
        if len(self.log_rows) > 12:
            self.log_rows.pop().destroy()

    # ================================================================ lab
    def _open_lab(self):
        model = self.current_model.get().strip()
        if not model:
            messagebox.showwarning("Model name required", "Pehle model select karo, phir Lab kholo.")
            return
        if self._lab is not None and self._lab.winfo_exists():
            self._lab.lift()
            return
        self._lab = LabWindow(self)

    # ========================================================= render loop
    def _tick(self):
        frame = self.static_image if self.static_image is not None else self.camera.get_frame()
        if frame is not None:
            display = frame.copy()
            for i, r in enumerate(self.rois):
                color = ROI_SELECTED if i == self.selected_idx else ROI_COLOR
                center = (int(r["cx"]), int(r["cy"]))
                cv2.circle(display, center, int(r["r"]), color, 2)
                cv2.drawMarker(display, center, color, cv2.MARKER_CROSS, 10, 1)
                cv2.putText(display, str(i + 1), (center[0] - 6, center[1] - int(r["r"]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            self.live_canvas.set_image(display)
            if self.static_image is not None:
                self.status_label.configure(text=f"\U0001F5BC Static image: {self.static_name}")
            else:
                self.status_label.configure(text=self.camera.status_text())
        self.after(33, self._tick)

    def _on_close(self):
        self.camera.stop()
        self.destroy()


# ======================================================================
class ResultWindow(ctk.CTkToplevel):
    """Original vs inspected side by side (zoom/pan synced) + operator
    override. Overriding moves the files, so the NG/OK folders always
    reflect the human-confirmed truth."""

    def __init__(self, app, original_bgr, result, saved, model):
        super().__init__(app)
        self.app = app
        self.saved = saved
        self.model = model
        self.title(f"Result - {model} test{saved['index']}")
        self.geometry("1100x680")
        self.configure(fg_color=BG)

        verdict_color = NG_COLOR if result.verdict == "NG" else OK_COLOR
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=14, pady=12)
        ctk.CTkLabel(header, text=result.verdict, font=ctk.CTkFont(size=28, weight="bold"),
                     text_color=verdict_color).pack(side="left")
        ctk.CTkLabel(header,
                     text=f"   {result.summary()}  \u2022  test{saved['index']}  \u2022  {model}",
                     font=ctk.CTkFont(size=13), text_color=MUTED).pack(side="left", padx=10)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=14, pady=(0, 10))
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(body, text="Original", text_color=MUTED,
                     font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(body, text="Inspected (dust=red, thread=yellow, scratch=magenta)",
                     text_color=MUTED, font=ctk.CTkFont(size=12)).grid(row=0, column=1, sticky="w")

        shared = ViewState()
        left = ImageCanvas(body, view_state=shared, editable=False)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        right = ImageCanvas(body, view_state=shared, editable=False)
        right.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        left.set_image(original_bgr)
        right.set_image(result.annotated, fit_if_first=False)

        ctk.CTkLabel(self, text=f"Saved: {saved['after_path']}", text_color=MUTED,
                     font=ctk.CTkFont(size=11)).pack(pady=(0, 6))

        other = "OK" if saved["verdict"] == "NG" else "NG"
        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.pack(pady=(0, 12))
        ctk.CTkButton(btns, text="\u2713 Sahi hai - Close", width=160,
                      command=self.destroy).pack(side="left", padx=6)
        ctk.CTkButton(btns, text=f"\u2717 Galat verdict - flip to {other}", width=220,
                      fg_color="#5a4516", hover_color="#79601f",
                      command=self._flip).pack(side="left", padx=6)

    def _flip(self):
        old = self.saved["verdict"]
        new = "OK" if old == "NG" else "NG"
        self.app.storage.relabel_capture(self.model, self.saved["index"], old, new)
        self.app._update_stats()
        self.app._add_log_row(f"test{self.saved['index']}  \u2022  overridden \u2192 {new}", ACCENT)
        self.destroy()


# ======================================================================
class LabWindow(ctk.CTkToplevel):
    """Tuning Lab: replay saved originals against the CURRENT sliders in
    the main window, view the anomaly heatmap, fix wrong truth labels,
    and evaluate miss rate (FNR) / false-alarm rate (FPR) over the whole
    captured set."""

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.model = app.current_model.get().strip()
        self.title(f"Tuning Lab - {self.model}")
        self.geometry("1280x800")
        self.configure(fg_color=BG)

        self.sel = None
        self.rows = []
        self._last_img = None
        self._last_res = None
        self._eval_state = None
        self._last_report = ""
        self.heatmap_on = ctk.BooleanVar(value=False)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---- left: capture list
        left = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color=PANEL)
        left.grid(row=0, column=0, sticky="nsw")
        head = ctk.CTkFrame(left, fg_color="transparent")
        head.pack(fill="x", padx=8, pady=(10, 4))
        ctk.CTkLabel(head, text="CAPTURES", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=ACCENT).pack(side="left")
        ctk.CTkButton(head, text="\U0001F504", width=30, fg_color="transparent",
                      hover_color="#262a2f", command=self._refresh_list).pack(side="right")
        self.list_frame = ctk.CTkScrollableFrame(left, width=230, fg_color="transparent")
        self.list_frame.pack(fill="both", expand=True, padx=6, pady=(0, 10))

        # ---- right: viewer + evaluation
        right = ctk.CTkFrame(self, corner_radius=0, fg_color=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)
        right.grid_columnconfigure(1, weight=1)

        controls = ctk.CTkFrame(right, fg_color="transparent")
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ctk.CTkButton(controls, text="\u25B6 Re-run with current sliders", width=200,
                      fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#0c1114",
                      command=self._run_on_selection).pack(side="left", padx=(0, 6))
        ctk.CTkSwitch(controls, text="Anomaly heatmap", variable=self.heatmap_on,
                      progress_color=ACCENT, command=self._render).pack(side="left", padx=6)
        self.relabel_btn = ctk.CTkButton(controls, text="\u21C4 Flip truth label", width=140,
                                          fg_color="#5a4516", hover_color="#79601f",
                                          command=self._relabel_selected)
        self.relabel_btn.pack(side="left", padx=6)

        self.sel_info = ctk.CTkLabel(right, text="Select a capture on the left.",
                                      text_color=MUTED, font=ctk.CTkFont(size=12), anchor="w")
        self.sel_info.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 4))

        shared = ViewState()
        self.left_canvas = ImageCanvas(right, view_state=shared, editable=False)
        self.left_canvas.grid(row=2, column=0, sticky="nsew", padx=(0, 5))
        self.right_canvas = ImageCanvas(right, view_state=shared, editable=False)
        self.right_canvas.grid(row=2, column=1, sticky="nsew", padx=(5, 0))

        evalf = ctk.CTkFrame(right, fg_color=PANEL)
        evalf.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        bar = ctk.CTkFrame(evalf, fg_color="transparent")
        bar.pack(fill="x", padx=8, pady=(8, 4))
        self.eval_btn = ctk.CTkButton(bar, text="\U0001F4CA Evaluate ALL captures with current sliders",
                                       fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#0c1114",
                                       command=self._start_evaluation)
        self.eval_btn.pack(side="left")
        self.progress_label = ctk.CTkLabel(bar, text="", text_color=MUTED,
                                            font=ctk.CTkFont(size=11))
        self.progress_label.pack(side="left", padx=10)
        ctk.CTkButton(bar, text="\U0001F4BE Export report", width=140, fg_color="transparent",
                      border_width=1, border_color=MUTED, hover_color="#262a2f",
                      command=self._export_report).pack(side="right")
        self.report_box = ctk.CTkTextbox(evalf, height=170, fg_color="#101214",
                                          font=ctk.CTkFont(family="Courier", size=12))
        self.report_box.pack(fill="x", padx=8, pady=(0, 8))

        self._refresh_list()

    # ------------------------------------------------------------- list
    def _refresh_list(self):
        for w in self.list_frame.winfo_children():
            w.destroy()
        self.rows = self.app.storage.list_captures(self.model)
        if not self.rows:
            ctk.CTkLabel(self.list_frame, text="No captures yet.", text_color=MUTED,
                         font=ctk.CTkFont(size=11)).pack(anchor="w", pady=4)
        for row in self.rows:
            color = NG_COLOR if row["truth"] == "NG" else OK_COLOR
            ctk.CTkButton(self.list_frame, text=f"test{row['index']}  \u2022  {row['truth']}",
                          anchor="w", fg_color="transparent", text_color=color,
                          hover_color="#262a2f",
                          command=lambda r=row: self._select_row(r)).pack(fill="x", pady=1)

    def _select_row(self, row):
        self.sel = row
        self._run_on_selection()

    # ------------------------------------------------------------ viewer
    def _run_on_selection(self):
        if self.sel is None:
            self.sel_info.configure(text="Select a capture on the left.", text_color=MUTED)
            return
        if not self.app.rois:
            self.sel_info.configure(text="Main window me ROIs load/set karo pehle.",
                                     text_color=NG_COLOR)
            return
        img = cv2.imread(self.sel["original_path"])
        if img is None:
            self.sel_info.configure(text=f"File read failed: {self.sel['original_path']}",
                                     text_color=NG_COLOR)
            return
        res = run_full_inspection(img, [dict(r) for r in self.app.rois],
                                   self.app.current_detect_params(), make_heatmap=True)
        self._last_img, self._last_res = img, res
        self._render()
        match = res.verdict == self.sel["truth"]
        mark = "\u2713" if match else "\u2717"
        self.sel_info.configure(
            text=f"test{self.sel['index']}  \u2022  truth {self.sel['truth']}  \u2022  "
                 f"predicted {res.verdict} {mark}  \u2022  {res.summary()}",
            text_color=(OK_COLOR if match else NG_COLOR))

    def _render(self):
        if self._last_img is None:
            return
        right_img = self._last_res.heatmap if self.heatmap_on.get() else self._last_res.annotated
        self.left_canvas.set_image(self._last_img)
        self.right_canvas.set_image(right_img, fit_if_first=False)

    def _relabel_selected(self):
        if self.sel is None:
            return
        old = self.sel["truth"]
        new = "OK" if old == "NG" else "NG"
        self.app.storage.relabel_capture(self.model, self.sel["index"], old, new)
        self.app._update_stats()
        idx = self.sel["index"]
        self._refresh_list()
        for r in self.rows:
            if r["index"] == idx:
                self._select_row(r)
                break

    # -------------------------------------------------------- evaluation
    def _start_evaluation(self):
        rows = self.app.storage.list_captures(self.model)
        if not rows:
            self.progress_label.configure(text="No captures to evaluate.")
            return
        if not self.app.rois:
            self.progress_label.configure(text="Main window me ROIs load/set karo pehle.")
            return
        self.eval_btn.configure(state="disabled")
        self._eval_state = {"done": 0, "total": len(rows), "rows": [], "finished": False}
        params = self.app.current_detect_params()
        rois = [dict(r) for r in self.app.rois]
        threading.Thread(target=self._eval_worker, args=(rows, rois, params),
                         daemon=True).start()
        self._poll_eval()

    def _eval_worker(self, rows, rois, params):
        st = self._eval_state
        for r in rows:
            img = cv2.imread(r["original_path"])
            pred = None
            if img is not None:
                pred = run_full_inspection(img, rois, params).verdict
            st["rows"].append({**r, "pred": pred})
            st["done"] += 1
        st["finished"] = True

    def _poll_eval(self):
        st = self._eval_state
        self.progress_label.configure(text=f"Evaluating {st['done']}/{st['total']}...")
        if not st["finished"]:
            self.after(150, self._poll_eval)
            return
        self.progress_label.configure(text=f"Done - {st['total']} images evaluated.")
        self.eval_btn.configure(state="normal")
        self._show_eval_report()

    def _show_eval_report(self):
        rows = self._eval_state["rows"]
        tp = sum(1 for r in rows if r["truth"] == "NG" and r["pred"] == "NG")
        fn = sum(1 for r in rows if r["truth"] == "NG" and r["pred"] == "OK")
        fp = sum(1 for r in rows if r["truth"] == "OK" and r["pred"] == "NG")
        tn = sum(1 for r in rows if r["truth"] == "OK" and r["pred"] == "OK")
        unread = [r for r in rows if r["pred"] is None]
        n_ng, n_ok = tp + fn, fp + tn
        fnr = (fn / n_ng * 100) if n_ng else 0.0
        fpr = (fp / n_ok * 100) if n_ok else 0.0
        p = self.app.current_detect_params()

        lines = []
        lines.append(f"EVALUATION REPORT - {self.model}")
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("Params: " + ", ".join(f"{k}={v}" for k, v in p.items()))
        lines.append(f"Samples: {len(rows)} total  ({n_ng} NG-truth, {n_ok} OK-truth)")
        lines.append("")
        lines.append("                 predicted NG    predicted OK")
        lines.append(f"  truth NG      TP = {tp:<10}  FN = {fn}   <- MISSES")
        lines.append(f"  truth OK      FP = {fp:<10}  TN = {tn}")
        lines.append("")
        lines.append(f"  FNR (miss rate)        : {fnr:.1f}%")
        lines.append(f"  FPR (false-alarm rate) : {fpr:.1f}%")
        if fn == 0 and n_ng > 0:
            worst = min(3.0 / n_ng, 1.0) * 100
            lines.append(f"  Rule of 3: 0 misses in {n_ng} NG samples -> worst-case true "
                         f"miss rate <= {worst:.1f}% (95% confidence)")
        bad = [r for r in rows if r["pred"] is not None and r["pred"] != r["truth"]]
        if bad:
            lines.append("")
            lines.append("Misclassified:")
            for r in bad:
                lines.append(f"  test{r['index']}  (truth {r['truth']}, predicted {r['pred']})  "
                             f"{r['original_path']}")
        if unread:
            lines.append("")
            lines.append(f"Unreadable files skipped: {len(unread)}")
        self._last_report = "\n".join(lines)
        self.report_box.delete("1.0", "end")
        self.report_box.insert("1.0", self._last_report)

    def _export_report(self):
        if not self._last_report:
            self.progress_label.configure(text="Pehle Evaluate chalao, phir export.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.app.storage.base_dir, self.model, f"eval_report_{ts}.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(self._last_report + "\n")
        self.progress_label.configure(text=f"Report saved: {path}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
