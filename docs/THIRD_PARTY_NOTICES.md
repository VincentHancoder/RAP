# Third-party and reused components

`inference/paper_side.py` and `inference/paper_sections.py` are the RAP training pipeline's paper parser
(caption matching, reading order and section-aware text budgets).

`inference/layout_vendor/` is copied from a PDF-Extract-Kit checkout, revision
`3ca95ad616a8eac40c1043f92c1bfe673a9090b8`, preserving original copyright headers.
Upstream: https://github.com/opendatalab/PDF-Extract-Kit . Its LayoutLMv3 components
include Microsoft/Google/Hugging Face Apache-2.0 code and VIT backbone code with a
GPL-3.0/commercial dual-license notice; those notices continue to apply.
The PDF-Extract-Kit checkout also includes the AGPL-3.0 license reproduced
verbatim in `layout_vendor/LICENSE.md`.

The PDF detector is the **same checkpoint** as used locally for training assets:
`opendatalab/PDF-Extract-Kit-1.0`, revision
`ed6b654c018d742e65a17671e379c5e6ecc87ec9`,
`models/Layout/LayoutLMv3/model_final.pth`, SHA-256
`cd2402ed4dd01de36c659a1e7dad35541adb2265df8599f0a071b1cb66bbbc5c`.
It is downloaded from upstream at first use, not redistributed here.

Detectron2 is installed from facebookresearch/detectron2 at
`1e3e13bbf607b54f62205c4c33922521822fb298` under its upstream Apache-2.0 license.
