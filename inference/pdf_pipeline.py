"""Paper-only prompt construction. No poster GT or external generation API."""
import json
import re
import unicodedata
from pathlib import Path
from PIL import Image
import fitz
from paper_side import load_paper_assets, select_for_prompt
from paper_sections import load_paper_body, clean_text, caption_fallback, section_by_reference, section_of_asset

SYSTEM = ('You are an expert academic poster designer. Given a paper and a design '
    'instruction, produce the poster as a standalone HTML document: a <style> '
    'block, then a fixed-size <div id="canvas"> containing optional decorative '
    '<div class="deco"> panels followed by absolutely positioned <div class="block"> '
    'elements (text-like or image-like) in reading order, each with explicit '
    'left/top/width/height in px. Place a figure with <img src="ASSET_ID"/> using '
    'exactly the asset ids offered; never invent ids.')
DEFAULT_INSTRUCTION = ('Create a clear academic poster with a strong title, a short motivation, '
    'the main method, key quantitative results, and takeaways. Prioritize the most informative '
    'figures. Use concise text, readable fonts, and generous spacing. Do not invent results.')
CANVASES = {'Landscape · 3:2': (2400,1600), 'Landscape · 16:9':(2400,1350),
            'Portrait · 2:3':(1600,2400), 'Square':(2000,2000)}

def scrub(value):
    return re.sub(r'<[^>]{0,80}>',lambda m:m.group(0).replace('<','‹').replace('>','›'),value or '')

def normalize_accent(value, default='#584287'):
    """Accept whatever the colour picker sends (#RGB, #RRGGBB, rgb()/rgba(), hsl()) and return #RRGGBB."""
    v=str(value or '').strip()
    m=re.fullmatch(r'#?([0-9a-fA-F]{6})',v)
    if m: return '#'+m.group(1).lower()
    m=re.fullmatch(r'#?([0-9a-fA-F]{3})',v)
    if m: return '#'+''.join(c*2 for c in m.group(1)).lower()
    m=re.fullmatch(r'rgba?\(([^)]*)\)',v,re.I)
    if m:
        try:
            parts=[p.strip() for p in m.group(1).replace('/',',').split(',')][:3]
            rgb=[max(0,min(255,round(float(p[:-1])*2.55 if p.endswith('%') else float(p)))) for p in parts]
            if len(rgb)==3: return '#%02x%02x%02x'%tuple(rgb)
        except ValueError: pass
    m=re.fullmatch(r'hsla?\(([^)]*)\)',v,re.I)
    if m:
        try:
            import colorsys
            parts=[p.strip().rstrip('%') for p in m.group(1).replace('/',',').split(',')][:3]
            h,sat,l=float(parts[0])/360.0,float(parts[1])/100.0,float(parts[2])/100.0
            r,g,b=colorsys.hls_to_rgb(h,l,sat); return '#%02x%02x%02x'%(round(r*255),round(g*255),round(b*255))
        except (ValueError,IndexError): pass
    return default

def prepare(pdf, job, instruction, canvas, accent, columns, custom_spec=''):
    from layout_runtime import extract
    pdf=Path(pdf);job=Path(job)
    if pdf.stat().st_size>40*1024**2:
        raise ValueError('Please upload a PDF smaller than 40 MB.')
    with pdf.open('rb') as f:
        if b'%PDF-' not in f.read(1024):
            raise ValueError('The uploaded file is not a PDF.')
    if len(instruction)>4000 or len(custom_spec)>3000:
        raise ValueError('Please shorten your design instruction / layout specification.')
    accent=normalize_accent(accent)
    columns=max(1,min(4,int(columns))); width,height=CANVASES[canvas]
    extracted=job/'paper'
    extract(pdf, extracted)
    body,stats,sections=load_paper_body('paper',str(job),str(pdf),abstract_missing=True,total=22000)
    registry=load_paper_assets('paper',str(job),str(pdf))
    if not body or len(body)<100:
        raise ValueError('No usable paper text was found. Scanned PDFs are not supported in this demo.')
    chosen=select_for_prompt(registry['assets'],[],16,{'figure':10,'table':5,'formula':3})
    with fitz.open(pdf) as doc:
        first=doc[0].get_text('dict')
        lines=[[s for s in l['spans'] if s['text'].strip() and not re.search(r'arxiv:|preprint|under review',s['text'],re.I)]
               for b in first['blocks'] if 'lines' in b for l in b['lines'] if abs(l.get('dir',(1,0))[1])<.1]
        lines=[l for l in lines if l]
        largest=max((s['size'] for l in lines for s in l),default=0)
        # whole lines whose biggest span is title-sized: keeps small-caps titles (ICLR) intact instead of 'B F G L'
        def join_line(l):   # small-caps titles come as 'B'+'AYESIAN' spans: no space when the spans touch
            out=''
            for prev,s in zip([None]+l[:-1],l):
                gap=s['bbox'][0]-prev['bbox'][2] if prev else 0
                out+=('' if prev and gap<s['size']*.15 and not out.endswith(' ') and not s['text'].startswith(' ') else ' ')+s['text']
            return out
        title=clean_text(' '.join(join_line(l) for l in lines if max(s['size'] for s in l)>=largest*.85
                                  and l[0]['bbox'][1]<doc[0].rect.height*.45))[:400]
        meta_title=clean_text(doc.metadata.get('title',''))
        if meta_title and len(meta_title)>8 and (len(title)<8 or len(title.split())>3 and sum(len(w) for w in title.split())/len(title.split())<2.5):
            title=meta_title[:400]
        # authors: the first-page lines between the title block and the abstract (PDF metadata is usually empty,
        # and without an authors line the model may invent names)
        title_lines=[l for l in lines if max(s['size'] for s in l)>=largest*.85 and l[0]['bbox'][1]<doc[0].rect.height*.45]
        title_bottom=max((s['bbox'][3] for l in title_lines for s in l),default=0)
        author_lines=[]
        for l in sorted((l for l in lines if l[0]['bbox'][1]>=title_bottom-1 and l not in title_lines),key=lambda l:(round(l[0]['bbox'][1]),l[0]['bbox'][0])):
            txt=clean_text(join_line(l))
            if re.match(r'^\s*(abstract|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t)\b',txt,re.I) or l[0]['bbox'][1]>doc[0].rect.height*.5: break
            if re.search(r'published as a conference|under review|preprint|anonymous author|proceedings of|copyright|workshop',txt,re.I): continue
            author_lines.append(txt)
        authors=' '.join(author_lines)
        authors=re.sub(r'\{[^}]*\}?|\S*@\S+','',authors)            # e-mail lists
        authors=re.sub(r'\s+([\u0300-\u036f])(\w)',r'\2\1',authors)    # 'Balunovi ́c' -> 'Balunović' (mark belongs to the next letter)
        authors=clean_text(unicodedata.normalize('NFC',authors))[:500] or clean_text(doc.metadata.get('author',''))[:500]
        asset_lines=[];assets=[]
        for a in chosen:
            cap=clean_text(a.get('caption',''))
            if not cap and a['kind'] in ('figure','table'):
                cap=caption_fallback(doc,a['page'],a['bbox'],a['kind'],72/200)
            sec=section_by_reference(sections,cap,a['kind'])
            if sec is None:
                pw=doc[a['page']-1].rect.width*200/72
                sec=section_of_asset(sections,a['page'],a['bbox'],pw,pw*.9)
            if sec is not None:a['section_role']=sec['role']
            context=f", page {a['page']}"+(f', in section "{scrub(sec["heading"][:70])}" [{sec["role"]}]' if sec else '')
            budget={'figure':448**2,'table':672**2,'formula':336**2}[a['kind']]
            with Image.open(a['file']) as original:
                im=original.convert('RGB');w,h=im.size
                scale=min(1,(budget/(w*h))**.5)
                tw,th=max(28,int(w*scale)//28*28),max(28,int(h*scale)//28*28)
                thumb=job/'thumbs'/f'{a["id"]}.jpg';thumb.parent.mkdir(exist_ok=True)
                im.resize((tw,th),Image.Resampling.LANCZOS).save(thumb,quality=88)
            a['thumb']=str(thumb);a['caption']=cap
            asset_lines.append(f'  {a["id"]} ({a["kind"]}, aspect {tw/th:.2f}{context})'+(f' - {scrub(cap)}' if cap else ''))
            assets.append(a)
    instruction=f'Canvas {width}x{height}px, {columns} columns. '+(instruction.strip() or DEFAULT_INSTRUCTION)
    fids=[a['id'] for a in assets if a['kind']=='figure']
    figure_slots=' -> '.join(fids[:5]) or 'text-based key findings'
    spec=(custom_spec.strip() or
        f'Canvas {width}x{height} {"landscape" if width>=height else "portrait"}, {columns} columns, moderate density. '
        f'Header band (14% tall): title {round(width*.028)}px white on {accent} band, authors line if provided, no invented logos. '
        f'Reading order: Motivation and Contributions -> Method -> Results -> Conclusion. '
        f'Available key figures: {figure_slots}. Keep each figure aspect ratio and give it a short caption. '
        f'Style: white background, Arial, body {round(width*.012)}px, section headings {round(width*.017)}px {accent}, '
        '24px gutters, balanced column heights, subtle light panels, no clipping or overlapping text.')
    return {'title':title,'authors':authors,'body':body,'sections':sections,'assets':assets,
            'asset_lines':'\n'.join(asset_lines) or '  (none)','instruction':instruction,'spec':spec,
            'canvas':[width,height],'stats':stats,'job':str(job)}

def chat(prepared, body=None):
    p=prepared
    paper=f'title: {scrub(p["title"])}\n'+(f'authors: {scrub(p["authors"])}\n' if p['authors'] else '')
    paper+='body:\n'+scrub(p['body'] if body is None else body)
    content=[{'type':'text','text':f'<paper>\n{paper}\n</paper>\n\n<assets>\n{p["asset_lines"]}\n</assets>\n'}]
    content.extend({'type':'image','image':a['thumb']} for a in p['assets'])
    content.append({'type':'text','text':f'\n\n<instruction>\n{scrub(p["instruction"])}\n</instruction>\n\n<layout_spec>\n{scrub(p["spec"])}\n</layout_spec>'})
    return [{'role':'system','content':[{'type':'text','text':SYSTEM}]},{'role':'user','content':content}]
