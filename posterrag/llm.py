"""LLM / VLM backends for PosterRAG. Keys are read from the environment only.

    ANTHROPIC_API_KEY  [ANTHROPIC_BASE_URL] [ANTHROPIC_MODEL=claude-opus-4-6]
    OPENAI_API_KEY     [OPENAI_BASE_URL]    [OPENAI_MODEL=gpt-4o]
    GEMINI_API_KEY                           [GEMINI_MODEL=gemini-2.5-pro]
"""
import base64
import json
import os
import re
import time
from pathlib import Path

DEFAULT_MODELS = {'anthropic': 'claude-opus-4-6', 'openai': 'gpt-4o', 'gemini': 'gemini-2.5-pro'}


class LLM:
    """One client for text and vision calls. `backend` in {anthropic, openai, gemini}."""

    def __init__(self, backend='anthropic', model=None, max_retries=3):
        self.backend = backend
        env_model = os.environ.get({'anthropic': 'ANTHROPIC_MODEL', 'openai': 'OPENAI_MODEL', 'gemini': 'GEMINI_MODEL'}[backend])
        self.model = model or env_model or DEFAULT_MODELS[backend]
        self.max_retries = max_retries
        if backend == 'anthropic':
            import anthropic
            key = os.environ.get('ANTHROPIC_API_KEY')
            if not key: raise EnvironmentError('set ANTHROPIC_API_KEY')
            self._client = anthropic.Anthropic(api_key=key, base_url=os.environ.get('ANTHROPIC_BASE_URL') or None)
        elif backend == 'openai':
            import openai
            key = os.environ.get('OPENAI_API_KEY')
            if not key: raise EnvironmentError('set OPENAI_API_KEY')
            self._client = openai.OpenAI(api_key=key, base_url=os.environ.get('OPENAI_BASE_URL') or None)
        elif backend == 'gemini':
            from google import genai
            key = os.environ.get('GEMINI_API_KEY')
            if not key: raise EnvironmentError('set GEMINI_API_KEY')
            self._client = genai.Client(api_key=key)
        else:
            raise ValueError(f'unknown backend {backend}')

    # ------------------------------------------------------------------ public
    def text(self, prompt, system='', temperature=0.3, max_tokens=16384):
        return self._call(prompt, system, [], temperature, max_tokens)

    def vision(self, prompt, images, system='', temperature=0.2, max_tokens=16384):
        """images: list of (base64_data, media_type) or image file paths."""
        imgs = [load_image_base64(i) if isinstance(i, (str, Path)) else tuple(i) for i in images]
        return self._call(prompt, system, imgs, temperature, max_tokens)

    # ----------------------------------------------------------------- private
    def _call(self, prompt, system, images, temperature, max_tokens):
        last = None
        for attempt in range(self.max_retries):
            try:
                return getattr(self, f'_{self.backend}')(prompt, system, images, temperature, max_tokens)
            except Exception as e:                                                        # noqa: BLE001
                last = e; time.sleep(5 * (attempt + 1))
        raise RuntimeError(f'{self.backend} call failed after {self.max_retries} attempts: {last}')

    def _anthropic(self, prompt, system, images, temperature, max_tokens):
        content = [{'type': 'image', 'source': {'type': 'base64', 'media_type': m, 'data': d}} for d, m in images]
        content.append({'type': 'text', 'text': prompt})
        kw = dict(model=self.model, max_tokens=max_tokens, temperature=temperature, messages=[{'role': 'user', 'content': content}])
        if system: kw['system'] = system
        r = self._client.messages.create(**kw)
        return {'content': ''.join(b.text for b in r.content if getattr(b, 'type', '') == 'text'),
                'input_tokens': r.usage.input_tokens, 'output_tokens': r.usage.output_tokens}

    def _openai(self, prompt, system, images, temperature, max_tokens):
        content = [{'type': 'image_url', 'image_url': {'url': f'data:{m};base64,{d}'}} for d, m in images]
        content.append({'type': 'text', 'text': prompt})
        messages = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': content}]
        r = self._client.chat.completions.create(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        u = r.usage
        return {'content': r.choices[0].message.content or '', 'input_tokens': getattr(u, 'prompt_tokens', 0), 'output_tokens': getattr(u, 'completion_tokens', 0)}

    def _gemini(self, prompt, system, images, temperature, max_tokens):
        from google.genai import types
        parts = [types.Part.from_bytes(data=base64.b64decode(d), mime_type=m) for d, m in images] + [types.Part.from_text(text=prompt)]
        cfg = types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens, system_instruction=system or None)
        r = self._client.models.generate_content(model=self.model, contents=[types.Content(role='user', parts=parts)], config=cfg)
        u = getattr(r, 'usage_metadata', None)
        return {'content': r.text or '', 'input_tokens': getattr(u, 'prompt_token_count', 0) or 0, 'output_tokens': getattr(u, 'candidates_token_count', 0) or 0}


# --------------------------------------------------------------------- helpers
def load_image_base64(path, max_b64_bytes=3_500_000):
    """Return (base64, media_type); large images are downscaled to JPEG so they fit API limits."""
    path = Path(path); data = base64.b64encode(path.read_bytes()).decode()
    media = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp'}.get(path.suffix.lower(), 'image/png')
    if len(data) <= max_b64_bytes: return data, media
    import io
    from PIL import Image
    im = Image.open(path).convert('RGB')
    for scale in (0.75, 0.5, 0.4, 0.3, 0.25):
        buf = io.BytesIO(); im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS).save(buf, 'JPEG', quality=85)
        data = base64.b64encode(buf.getvalue()).decode()
        if len(data) <= max_b64_bytes: break
    return data, 'image/jpeg'


def extract_json(text):
    """First JSON object/array in an LLM reply (fenced or bare)."""
    m = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    cand = (m.group(1) if m else text).strip()
    start = min([i for i in (cand.find('{'), cand.find('[')) if i >= 0] or [-1])
    if start < 0: raise ValueError(f'no JSON in reply: {text[:200]}')
    cand = cand[start:]
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        end = max(cand.rfind('}'), cand.rfind(']'))
        return json.loads(cand[:end + 1])


def extract_html(text):
    m = re.search(r'```html\s*\n?(.*?)```', text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r'(<(?:!DOCTYPE|html)[^>]*>.*)', text, re.DOTALL | re.IGNORECASE)
    if m: return m.group(1).strip()
    raise ValueError(f'no HTML in reply: {text[:200]}')
