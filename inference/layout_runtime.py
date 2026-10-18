"""Layout detection runs in its own Python environment (detectron2 + legacy Transformers), created on first use
under .runtime/layout_env or pointed to with RAP_LAYOUT_PYTHON."""
import os
import subprocess
import sys
import threading
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT/'.runtime'
_lock = threading.Lock()

def worker_environment():
    env = dict(os.environ)
    for key in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HF_API_TOKEN','PYTHONPATH','LD_PRELOAD'):
        env.pop(key,None)
    env.update(HF_HUB_DISABLE_IMPLICIT_TOKEN='1', FORCE_CUDA='0', MAX_JOBS='2',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    return env

def setup():
    override = os.environ.get('RAP_LAYOUT_PYTHON')                     # an existing env with detectron2 + transformers 4.44 (see README)
    if override:
        return override
    with _lock:
        envdir=RUNTIME/'layout_env'; python=envdir/'bin/python'
        marker=envdir/'rap_parser_ready'
        if marker.exists():
            return str(python)
        envdir.parent.mkdir(parents=True,exist_ok=True)
        venv.EnvBuilder(with_pip=True,system_site_packages=True).create(envdir)
        env=worker_environment()
        with (RUNTIME/'layout_setup.log').open('a') as log:
            for packages in [
                ['pip','setuptools','wheel','ninja'],
                ['transformers==4.44.2','huggingface-hub==0.35.3','numpy<2','opencv-python-headless<4.12'],
                ['--no-build-isolation','git+https://github.com/facebookresearch/detectron2.git@1e3e13bbf607b54f62205c4c33922521822fb298']
            ]:
                result=subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check',*packages],
                                      env=env,stdout=log,stderr=log,timeout=900)
                if result.returncode:
                    log.flush()
                    print('PARSER_SETUP_ERROR', (RUNTIME/'layout_setup.log').read_text(errors='replace')[-6000:],flush=True)
                    raise RuntimeError('LayoutLMv3 dependency setup failed. Please contact the demo maintainer.')
        marker.write_text('layoutlmv3-cpu-v1\n')
        return str(python)

def extract(pdf, out):
    python=setup()
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'extraction.log').open('w') as log:
        result=subprocess.run([python,str(ROOT/'layout_worker.py'),str(pdf),str(out)],
                              env=worker_environment(),stdout=log,stderr=log,timeout=900)
    if result.returncode:
        print('PARSER_EXTRACTION_ERROR', (out/'extraction.log').read_text(errors='replace')[-6000:],flush=True)
        raise RuntimeError('LayoutLMv3 could not parse this PDF. Please try a text-based research paper.')
