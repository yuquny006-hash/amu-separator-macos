"""Verify the bundled runtime can load the instruments/Pro pipeline.

Runs inside the packaged runtime (torch/einops/beartype/... installed), but does
NOT load the 1.37GB Mega53 checkpoint -- that stays on-demand like Windows.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
import yaml                      # noqa: F401  Mega53 config loader
import beartype                  # noqa: F401
import einops                    # noqa: F401
import rotary_embedding_torch    # noqa: F401
import torch                     # noqa: F401
from amu_msst import BSRoformer  # noqa: F401

spec = importlib.util.spec_from_file_location(
    'instrument_worker', ROOT / 'app' / 'instrument_worker.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert len(m.CHANNELS) >= 17, '乐器通道表不完整'
assert m.CKPT_NAME == 'mvsep_mega_53.ckpt', '权重文件名变了'
print(json.dumps({'msst_runtime': 'ok', 'torch': torch.__version__,
                  'channels': len(m.CHANNELS), 'fallback_stems': len(m.FALLBACK_STEMS)},
                 ensure_ascii=False))
