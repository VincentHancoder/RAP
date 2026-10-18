"""Resolve only offered assets; render model HTML without scripts or networking."""
import base64
import json
import re
import shutil
import zipfile
from pathlib import Path
from bs4 import BeautifulSoup

CSP = "default-src 'none'; img-src data:; style-src 'unsafe-inline'; font-src 'none'; script-src 'none'; connect-src 'none'; form-action 'none'; base-uri 'none'"

def safe_html(raw, assets):
    raw=re.sub(r'^\s*```(?:html)?\s*','',raw,flags=re.I)
    start=re.search(r'<!doctype\s+html|<html\b|<style\b',raw,re.I)
    if not start:
        raise ValueError('The model has not produced an HTML document yet. Try Continue generation.')
    raw=raw[start.start():]
    end=re.search(r'</html\s*>',raw,re.I)
    if end:raw=raw[:end.end()]
    soup=BeautifulSoup(raw,'html.parser')
    for tag in list(soup.find_all(['script','iframe','object','embed','base','link','meta','form','input','button','textarea','video','audio','source','foreignObject'])):
        tag.decompose()
    by_id={a['id']:a for a in assets};used=[];missing=[]
    for tag in soup.find_all(True):
        for name in list(tag.attrs):
            if name.lower().startswith('on') or name.lower() in ('href','srcset','action','formaction','xlink:href','srcdoc'):
                del tag.attrs[name]
        if tag.name=='img':
            aid=Path(str(tag.get('src','')).split('?')[0]).stem
            if aid in by_id:
                data=Path(by_id[aid]['file']).read_bytes()
                tag['src']='data:image/png;base64,'+base64.b64encode(data).decode()
                if aid not in used:used.append(aid)
            else:
                missing.append(aid[:80]);tag.attrs.pop('src',None)
                tag['alt']='Unavailable asset'
        elif tag.has_attr('src'):
            del tag.attrs['src']
    if not soup.find(id='canvas'):
        raise ValueError('No poster canvas was generated yet. Try Continue generation.')
    if soup.html is None:
        wrapper=BeautifulSoup('<!doctype html><html><head></head><body></body></html>','html.parser')
        for child in list(soup.contents):wrapper.body.append(child.extract())
        soup=wrapper
    if soup.head is None:
        head=soup.new_tag('head');soup.html.insert(0,head)
    meta=soup.new_tag('meta');meta['http-equiv']='Content-Security-Policy';meta['content']=CSP
    soup.head.insert(0,meta)
    charset=soup.new_tag('meta');charset['charset']='utf-8';soup.head.insert(0,charset)
    return str(soup),used,missing

def render(raw, prepared):
    from playwright.sync_api import sync_playwright
    job=Path(prepared['job']);output=job/'output';output.mkdir(exist_ok=True)
    html,used,missing=safe_html(raw,prepared['assets'])
    html_path=output/'poster.html';html_path.write_text(html,encoding='utf-8')
    (output/'model_output.txt').write_text(raw,encoding='utf-8')
    png=output/'poster.png'
    width,height=prepared['canvas']
    executable=shutil.which('chromium') or shutil.which('chromium-browser')
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,executable_path=executable,
            args=['--no-sandbox','--disable-dev-shm-usage','--disable-background-networking'])
        context=browser.new_context(viewport={'width':width,'height':height},
                                   java_script_enabled=False,service_workers='block',offline=True)
        context.route('**/*',lambda route:route.abort())
        page=context.new_page();page.set_default_timeout(20000)
        page.set_content(html,wait_until='load',timeout=20000)
        dimensions=page.evaluate('''() => {const c=document.getElementById('canvas');
          const r=c.getBoundingClientRect();return {width:r.width,height:r.height};}''')
        w,h=round(dimensions['width']),round(dimensions['height'])
        if not 100<=w<=6000 or not 100<=h<=6000:
            browser.close();raise ValueError('Generated canvas dimensions are invalid.')
        scale=min(1,1800/max(w,h))
        page.set_viewport_size({'width':max(1,round(w*scale)),'height':max(1,round(h*scale))})
        page.evaluate('''s => {document.body.style.margin='0';document.body.style.padding='0';
          document.body.style.overflow='hidden';let c=document.getElementById('canvas');
          c.style.margin='0';c.style.position='relative';c.style.left='0';c.style.top='0';
          c.style.transformOrigin='0 0';c.style.transform=`scale(${s})`;}''',scale)
        page.wait_for_timeout(400);page.screenshot(path=str(png));browser.close()
    metadata={'model':'VincentHancoder/RAP-Qwen3.8-27B-RL','dtype':'BF16',
              'selected_assets':used,'unresolved_assets':missing,'generated_canvas':[w,h],
              'requested_canvas':prepared['canvas'],'complete_html':bool(re.search(r'</html\s*>',raw,re.I)),
              'instruction':prepared['instruction'],'layout_spec':prepared['spec']}
    (output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    archive=output/'poster_bundle.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ('poster.html','poster.png','model_output.txt','metadata.json'):
            z.write(output/name,name)
        for a in prepared['assets']:
            if a['id'] in used:z.write(a['file'],f'assets/{a["id"]}.png')
    return str(png),str(html_path),str(archive),metadata
