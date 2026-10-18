"""Multimodal LLM backends for Stage 3 (Design Captioning) and Stage 4 (HTML Reconstruction).

API keys are read from environment variables only:

  anthropic : ANTHROPIC_API_KEY  (+ optional ANTHROPIC_BASE_URL)      default model claude-opus-4-6
  openai    : OPENAI_API_KEY     (+ optional OPENAI_BASE_URL)         any OpenAI-compatible server
              (OpenAI, OpenRouter, vLLM, SGLang, ...); default model from OPENAI_MODEL or qwen/qwen3.6-plus
  mock      : no network; returns canned outputs so the pipeline can be exercised without a key

RAP-25K templates were produced with the anthropic backend (claude-opus-4-6): temperature 0.2,
max_tokens 16384 for the design caption and 32768 for the HTML, poster images resized to ≤1568 px.
"""
from __future__ import annotations

import base64
import os
import time
from io import BytesIO

from PIL import Image

TEMPERATURE = 0.2
MAX_TOKENS_DESIGN = 16384
MAX_TOKENS_HTML = 32768
RETRY_DELAYS = [10, 20, 40, 80, 120]
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-6"
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "qwen/qwen3.6-plus")


# ---------------------------------------------------------------- image prep
def resize_for_api(image: Image.Image, max_dim: int = 1568, max_bytes: int = 3_500_000) -> Image.Image:
    """Longest side ≤ max_dim and JPEG ≤ max_bytes (what the production run sent to the model)."""
    w, h = image.size
    if max(w, h) > max_dim:
        s = max_dim / max(w, h)
        image = image.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)
    for quality in (85, 70, 55, 40):
        buf = BytesIO(); image.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= max_bytes:
            return image
    w, h = image.size
    return image.resize((int(w * 0.7), int(h * 0.7)), Image.Resampling.LANCZOS)


def image_to_base64(image: Image.Image, quality: int = 85, max_b64_bytes: int = 5_000_000) -> str:
    for q in (quality, 70, 55, 40, 30):
        buf = BytesIO(); image.save(buf, format="JPEG", quality=q)
        enc = base64.b64encode(buf.getvalue()).decode("ascii")
        if len(enc) <= max_b64_bytes or q == 30:
            return enc
    return enc  # pragma: no cover


def _require(var: str, backend: str) -> str:
    value = os.environ.get(var, "").strip()
    if not value:
        raise SystemExit(f"[derender] backend '{backend}' needs the environment variable {var} "
                         f"(export {var}=... ; keys are never read from files or code).")
    return value


# ------------------------------------------------------------------ backends
class MLLM:
    """Common interface: complete(system, user_text, image) -> response text."""
    name = "base"

    def complete(self, system: str, user_text: str, image: Image.Image, max_tokens: int) -> str:
        raise NotImplementedError

    def call(self, system: str, user_text: str, image: Image.Image, max_tokens: int, label: str = "") -> str:
        last = None
        for attempt, delay in enumerate(RETRY_DELAYS + [None]):
            try:
                text = self.complete(system, user_text, image, max_tokens)
                if not text or not text.strip():
                    raise RuntimeError("empty response")
                return text
            except Exception as e:                                  # noqa: BLE001
                last = e
                if delay is None:
                    break
                wait = delay * (2 if "429" in str(e) or "rate" in str(e).lower() else 1)
                print(f"  [{label or self.name}] attempt {attempt + 1} failed: {str(e)[:160]} -> retry in {wait}s", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"[{label or self.name}] failed after {len(RETRY_DELAYS) + 1} attempts: {last}")


class AnthropicBackend(MLLM):
    name = "anthropic"

    def __init__(self, model: str | None = None):
        import anthropic
        key = _require("ANTHROPIC_API_KEY", self.name)
        base = os.environ.get("ANTHROPIC_BASE_URL") or None
        self.client = anthropic.Anthropic(api_key=key, base_url=base, timeout=900, max_retries=0)
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)

    def complete(self, system, user_text, image, max_tokens):
        b64 = image_to_base64(resize_for_api(image))
        msg = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=TEMPERATURE, system=system,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": user_text}]}])
        return "\n".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class OpenAICompatibleBackend(MLLM):
    name = "openai"

    def __init__(self, model: str | None = None):
        from openai import OpenAI
        key = _require("OPENAI_API_KEY", self.name)
        base = os.environ.get("OPENAI_BASE_URL") or None
        self.client = OpenAI(api_key=key, base_url=base, timeout=900, max_retries=0)
        self.model = model or DEFAULT_OPENAI_MODEL

    def complete(self, system, user_text, image, max_tokens):
        b64 = image_to_base64(resize_for_api(image))
        resp = self.client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, temperature=TEMPERATURE,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": [
                          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                          {"type": "text", "text": user_text}]}])
        return resp.choices[0].message.content or ""


class MockBackend(MLLM):
    """Offline stand-in used by tests: a minimal design caption, and the scaffold echoed back as the HTML."""
    name = "mock"

    def __init__(self, model: str | None = None):
        self.model = "mock"

    def complete(self, system, user_text, image, max_tokens):
        import json
        if "Scaffold HTML:" in user_text:
            return user_text.split("Scaffold HTML:", 1)[1].strip()
        from .design_dimensions import DIMENSIONS, REGION_DIM_IDS
        first = {d.id: d.variants[0] for d in DIMENSIONS}
        region = {"id": "R-01", "type": "header", "title": "(no title)", "reading_order": 1,
                  "caption": "Mock region covering the whole poster.", "design_caption": "Mock design caption."}
        region.update({k: {"value": first[k], "notes": "mock"} for k in REGION_DIM_IDS})
        return json.dumps({"global_caption": "Mock caption.",
                           "global": {"decorations": [], "decorations_notes": "none",
                                      "palette": {"surface": "#FFFFFF", "primary": "#1F3A93", "accent": "#E67E22", "fg": "#111111"},
                                      "typography_family": "sans_modern", "typography_notes": "bold sans headings"},
                           "regions": [region]})


BACKENDS = {"anthropic": AnthropicBackend, "openai": OpenAICompatibleBackend, "mock": MockBackend}


def make_backend(name: str, model: str | None = None) -> MLLM:
    if name not in BACKENDS:
        raise SystemExit(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[name](model)
