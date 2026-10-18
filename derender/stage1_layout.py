"""Stage 1 — Layout Detection.

Runs the PDF-Extract-Kit LayoutLMv3 detector (10 classes) on a poster image and writes
`layout_results.json` in the schema the later stages consume:

    [{"image_id": str, "boxes": [[x1,y1,x2,y2],...], "scores": [...], "classes": [...],
      "detections": [{"bbox": [...], "score": float, "class_id": int, "class_name": str}, ...]}]

This script runs in the *layout* environment (detectron2 + the Transformers version the vendored
LayoutLMv3 code expects); `run_derender.py` launches it with the interpreter named by RAP_LAYOUT_PYTHON
(default: the current interpreter). Weights are fetched once from the Hugging Face repo
opendatalab/PDF-Extract-Kit-1.0 (models/Layout/LayoutLMv3). The detector code itself lives in
../inference/layout_vendor (PDF-Extract-Kit, AGPL-3.0).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CLASSES = ["title", "plain text", "abandon", "figure", "figure_caption", "table",
           "table_caption", "table_footnote", "isolate_formula", "formula_caption"]
VENDOR = Path(__file__).resolve().parent.parent / "inference"
PEK_REPO = "opendatalab/PDF-Extract-Kit-1.0"
PEK_REVISION = "ed6b654c018d742e65a17671e379c5e6ecc87ec9"


def load_predictor(device: str = "cpu", score_thresh: float = 0.05, weights_dir: str | None = None):
    sys.path.insert(0, str(VENDOR))
    from transformers import modeling_utils, pytorch_utils             # legacy symbols the vendored code imports
    for name in ("find_pruneable_heads_and_indices", "prune_linear_layer"):
        if not hasattr(modeling_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from layout_vendor.model_init import add_vit_config
    import layout_vendor.backbone   # noqa: F401  (registers the ViT backbone)
    import layout_vendor.rcnn_vl    # noqa: F401  (registers the VL R-CNN)

    if weights_dir is None:
        from huggingface_hub import hf_hub_download
        for name in ("config.json", "model_final.pth"):
            p = hf_hub_download(PEK_REPO, f"models/Layout/LayoutLMv3/{name}", revision=PEK_REVISION)
        weights_dir = str(Path(p).parent)
    cfg = get_cfg(); add_vit_config(cfg)
    cfg.merge_from_file(str(VENDOR / "layout_vendor" / "layoutlmv3_base_inference.yaml"))
    cfg.MODEL.DEVICE = device
    cfg.MODEL.CONFIG_PATH = weights_dir
    cfg.MODEL.WEIGHTS = str(Path(weights_dir) / "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.freeze()
    return DefaultPredictor(cfg)


def detect(predictor, image_path: Path) -> dict:
    import numpy as np
    from PIL import Image
    im = Image.open(image_path).convert("RGB")
    out = predictor(np.asarray(im)[:, :, ::-1])["instances"].to("cpu")       # detectron2 expects BGR
    boxes = out.pred_boxes.tensor.tolist(); scores = out.scores.tolist(); classes = out.pred_classes.tolist()
    dets = [{"bbox": b, "score": float(s), "class_id": int(c), "class_name": CLASSES[int(c)]}
            for b, s, c in zip(boxes, scores, classes)]
    return {"image_id": image_path.stem, "boxes": boxes, "scores": scores, "classes": classes, "detections": dets,
            "image_size": [im.width, im.height]}


def visualize(image_path: Path, dets: list[dict], out_png: Path, conf_thres: float = 0.25) -> None:
    from PIL import Image, ImageDraw
    im = Image.open(image_path).convert("RGB"); d = ImageDraw.Draw(im)
    for det in dets:
        if det["score"] < conf_thres:
            continue
        x1, y1, x2, y2 = det["bbox"]; d.rectangle([x1, y1, x2, y2], outline="red", width=3)
        d.text((x1 + 3, y1 + 3), f'{det["class_name"]} {det["score"]:.2f}', fill="red")
    im.save(out_png)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, nargs="+", help="poster image(s)")
    ap.add_argument("--output_dir", required=True, nargs="+", help="one output dir per input (or a single dir for one input)")
    ap.add_argument("--device", default=os.environ.get("RAP_LAYOUT_DEVICE", "cpu"))
    ap.add_argument("--score_thresh", type=float, default=0.05)
    ap.add_argument("--weights_dir", default=os.environ.get("RAP_LAYOUT_WEIGHTS"), help="dir with config.json + model_final.pth (default: download)")
    ap.add_argument("--visualize", action="store_true")
    a = ap.parse_args()
    if len(a.output_dir) != len(a.input):
        raise SystemExit("--input and --output_dir must have the same number of entries")
    predictor = load_predictor(a.device, a.score_thresh, a.weights_dir)
    for img, out in zip(a.input, a.output_dir):
        img, out = Path(img), Path(out); out.mkdir(parents=True, exist_ok=True)
        page = detect(predictor, img)
        (out / "layout_results.json").write_text(json.dumps([page], indent=2), encoding="utf-8")
        if a.visualize:
            visualize(img, page["detections"], out / f"{img.stem}_layout.png")
        print(f"LAYOUT_DONE {img.name}: {len(page['detections'])} detections", flush=True)


if __name__ == "__main__":
    main()
