"""CPU LayoutLMv3 extraction in an isolated legacy-Transformers environment."""
import json
import os
import sys
from pathlib import Path

def run(pdf, destination):
    import fitz
    import numpy as np
    import torch
    from PIL import Image
    from huggingface_hub import hf_hub_download
    from transformers import modeling_utils, pytorch_utils
    for name in ('find_pruneable_heads_and_indices', 'prune_linear_layer'):
        if not hasattr(modeling_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from layout_vendor.model_init import add_vit_config
    import layout_vendor.backbone
    import layout_vendor.rcnn_vl

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
    revision = 'ed6b654c018d742e65a17671e379c5e6ecc87ec9'
    model_dir = None
    for name in ('config.json', 'model_final.pth'):
        p = hf_hub_download('opendatalab/PDF-Extract-Kit-1.0',
            'models/Layout/LayoutLMv3/' + name, revision=revision, token=False)
        model_dir = str(Path(p).parent)
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(str(Path(__file__).parent/'layout_vendor/layoutlmv3_base_inference.yaml'))
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.CONFIG_PATH = model_dir
    cfg.MODEL.WEIGHTS = str(Path(model_dir)/'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    classes = ['title', 'plain text', 'abandon', 'figure', 'figure_caption', 'table',
               'table_caption', 'table_footnote', 'isolate_formula', 'formula_caption']
    kinds = {'figure':'figure','table':'table','isolate_formula':'formula'}
    out = Path(destination); (out/'assets').mkdir(parents=True, exist_ok=True)
    layout, assets, counters = [], [], {k:0 for k in kinds}
    def iou(a, b):
        w = max(0, min(a[2],b[2])-max(a[0],b[0]))
        h = max(0, min(a[3],b[3])-max(a[1],b[1]))
        return w*h/max(1, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-w*h)
    with fitz.open(pdf) as doc:
        if doc.needs_pass:
            raise ValueError('Password-protected PDFs are not supported.')
        n = min(12, len(doc))
        for i in range(n):
            page = doc[i]
            if page.rect.width*page.rect.height*(200/72)**2 > 20_000_000:
                raise ValueError('PDF page dimensions are too large.')
            pix = page.get_pixmap(matrix=fitz.Matrix(200/72,200/72), colorspace=fitz.csRGB, alpha=False)
            image = Image.frombytes('RGB', (pix.width,pix.height), pix.samples)
            pred = predictor(np.asarray(image)[:,:,::-1])['instances'].to('cpu')
            dets=[]
            for box, score, cls in zip(pred.pred_boxes.tensor.tolist(),pred.scores.tolist(),pred.pred_classes.tolist()):
                dets.append({'page':i+1,'class_name':classes[cls], 'bbox':box,'score':score})
            layout.extend(dets)
            selected=[]
            for d in sorted(dets,key=lambda d:-d['score']):
                b=d['bbox']; area=(b[2]-b[0])*(b[3]-b[1])
                if d['class_name'] not in kinds or d['score']<.60 or area<.004*pix.width*pix.height:
                    continue
                if any(iou(b,k['bbox'])>=.75 for k in selected):
                    continue
                selected.append(d)
                cls=d['class_name']; counters[cls]+=1; aid=f'{kinds[cls]}_{counters[cls]}'
                crop=image.crop((max(0,int(b[0])-6),max(0,int(b[1])-6),
                                 min(pix.width,int(b[2])+6),min(pix.height,int(b[3])+6)))
                crop.save(out/'assets'/f'{aid}.png')
                assets.append({'id':aid,'class':cls,'page':i+1,'bbox':b,'score':d['score'],
                               'w':crop.width,'h':crop.height,'aspect':round(crop.width/max(1,crop.height),2)})
            print(f'LAYOUT_PAGE {i+1}/{n}', flush=True)
        (out/'assets.json').write_text(json.dumps({'dpi':200,'n_pages':n,'assets':assets,'counts':counters}))
        (out/'layout.json').write_text(json.dumps(layout))
    print('LAYOUT_DONE',flush=True)

if __name__ == '__main__':
    run(sys.argv[1],sys.argv[2])
