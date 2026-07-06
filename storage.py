"""
storage.py  (v2)
-----------------
1. Per-model config under configs/<model>.json - ROI list + ALL
   detection parameters, so selecting a model restores the exact
   tuned state.

2. Capture directories (unchanged layout):

       <base>/<model>/original/NG|OK/<model>_test<N>_<verdict>.png
       <base>/<model>/after/NG|OK/<model>_test<N>_<verdict>.png

   The folder a capture lives in is treated as the HUMAN-CONFIRMED
   truth: when the operator overrides a verdict, relabel_capture()
   moves+renames both files. That makes the folders usable as ground
   truth for the tuning lab's FNR/FPR evaluation.
"""

import json
import os
import re
import cv2


class Storage:
    def __init__(self, base_dir="inspection_data", config_dir="configs"):
        self.base_dir = base_dir
        self.config_dir = config_dir
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)

    # ------------------------------------------------------ model config
    def _config_path(self, model):
        return os.path.join(self.config_dir, f"{model}.json")

    def list_models(self):
        return sorted(f[:-5] for f in os.listdir(self.config_dir) if f.endswith(".json"))

    def save_model_config(self, model, rois, detect_params):
        cfg = {"model": model, "rois": rois, "detect_params": detect_params}
        with open(self._config_path(model), "w") as f:
            json.dump(cfg, f, indent=2)
        return cfg

    def load_model_config(self, model):
        path = self._config_path(model)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            cfg = json.load(f)
        # tolerate v1 configs (sigma/min_pixel_area at top level, ghost keys)
        if "detect_params" not in cfg:
            cfg["detect_params"] = {
                "sigma_intensity": cfg.get("sigma", 3.0),
                "min_pixel_area": cfg.get("min_pixel_area", 6),
            }
        return cfg

    def delete_model_config(self, model):
        p = self._config_path(model)
        if os.path.exists(p):
            os.remove(p)

    # --------------------------------------------------------- captures
    def _dirs_for(self, model):
        dirs = {}
        for stage in ("original", "after"):
            for verdict in ("NG", "OK"):
                d = os.path.join(self.base_dir, model, stage, verdict)
                os.makedirs(d, exist_ok=True)
                dirs[(stage, verdict)] = d
        return dirs

    def next_test_index(self, model):
        """Indices are unique per model. Scan existing filenames for the
        max index instead of counting files, so deletions or relabels
        can never cause a collision."""
        dirs = self._dirs_for(model)
        max_idx = 0
        pat = re.compile(rf"^{re.escape(model)}_test(\d+)_(NG|OK)\.png$")
        for verdict in ("NG", "OK"):
            for f in os.listdir(dirs[("original", verdict)]):
                m = pat.match(f)
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
        return max_idx + 1

    def save_capture(self, model, original_bgr, annotated_bgr, verdict):
        dirs = self._dirs_for(model)
        idx = self.next_test_index(model)
        fname = f"{model}_test{idx}_{verdict}.png"
        orig_path = os.path.join(dirs[("original", verdict)], fname)
        after_path = os.path.join(dirs[("after", verdict)], fname)
        cv2.imwrite(orig_path, original_bgr)
        cv2.imwrite(after_path, annotated_bgr)
        return {"index": idx, "verdict": verdict,
                "original_path": orig_path, "after_path": after_path}

    def relabel_capture(self, model, index, old_verdict, new_verdict):
        """Operator override: move original+after to the other folder and
        rename _NG <-> _OK. Returns the new paths."""
        if old_verdict == new_verdict:
            return None
        dirs = self._dirs_for(model)
        old_name = f"{model}_test{index}_{old_verdict}.png"
        new_name = f"{model}_test{index}_{new_verdict}.png"
        out = {}
        for stage in ("original", "after"):
            src = os.path.join(dirs[(stage, old_verdict)], old_name)
            dst = os.path.join(dirs[(stage, new_verdict)], new_name)
            if os.path.exists(src):
                os.replace(src, dst)
                out[stage] = dst
        return out

    def list_captures(self, model):
        """[{index, truth, original_path, after_path}] sorted by index.
        `truth` = the folder the capture currently lives in (human-
        confirmed verdict)."""
        dirs = self._dirs_for(model)
        pat = re.compile(rf"^{re.escape(model)}_test(\d+)_(NG|OK)\.png$")
        rows = []
        for verdict in ("NG", "OK"):
            d = dirs[("original", verdict)]
            for f in os.listdir(d):
                m = pat.match(f)
                if not m:
                    continue
                idx = int(m.group(1))
                rows.append({
                    "index": idx, "truth": verdict,
                    "original_path": os.path.join(d, f),
                    "after_path": os.path.join(dirs[("after", verdict)], f),
                })
        rows.sort(key=lambda r: r["index"])
        return rows

    def get_stats(self, model):
        caps = self.list_captures(model)
        ng = sum(1 for c in caps if c["truth"] == "NG")
        ok = len(caps) - ng
        total = len(caps)
        return {"ng": ng, "ok": ok, "total": total,
                "ng_rate": (ng / total * 100) if total else 0.0}

