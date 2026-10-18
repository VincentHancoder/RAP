"""Render a self-contained poster HTML file to PNG with headless Chromium (playwright)."""
import shutil
import sys
from pathlib import Path


def render_png(html_path, png_path, max_side=2400, timeout_ms=30000):
    from playwright.sync_api import sync_playwright
    html = Path(html_path).read_text(encoding='utf-8')
    executable = shutil.which('chromium') or shutil.which('chromium-browser')   # None -> playwright's own chromium
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, executable_path=executable,
                                     args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-background-networking'])
        context = browser.new_context(viewport={'width': 2400, 'height': 1600}, java_script_enabled=False, offline=True)
        context.route('**/*', lambda route: route.abort())
        page = context.new_page(); page.set_default_timeout(timeout_ms)
        page.set_content(html, wait_until='load')
        dims = page.evaluate('''() => {const c = document.getElementById('canvas') || document.body;
            const r = c.getBoundingClientRect(); return {w: r.width, h: r.height};}''')
        w, h = max(100, round(dims['w'])), max(100, round(dims['h']))
        scale = min(1.0, max_side / max(w, h))
        page.set_viewport_size({'width': max(1, round(w * scale)), 'height': max(1, round(h * scale))})
        page.evaluate('''s => {document.body.style.margin = '0'; document.body.style.overflow = 'hidden';
            const c = document.getElementById('canvas'); if (c) {c.style.margin = '0'; c.style.position = 'relative';
            c.style.left = '0'; c.style.top = '0'; c.style.transformOrigin = '0 0'; c.style.transform = `scale(${s})`;}}''', scale)
        page.wait_for_timeout(300); page.screenshot(path=str(png_path)); browser.close()
    return str(png_path)


if __name__ == '__main__':
    print(render_png(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else Path(sys.argv[1]).with_suffix('.png')))
