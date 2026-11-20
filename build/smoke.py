"""Run from the relocated app's bundled interpreter, using no login/points."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

resources = Path(sys.argv[1]).resolve()
source = resources / 'app'
os.environ['AMU_VENV'] = str(resources / 'runtime')
os.environ['HF_HOME'] = str(source / '_hfcache')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PATH'] = str(source / 'bin') + os.pathsep + os.environ.get('PATH', '')
sys.path.insert(0, str(source))
import numpy as np
import soundfile as sf
import separate_core
import harmony_separate
harmony_separate.SILENT = False

assert Path(separate_core.VENV_PY).is_file()
assert str(resources) in separate_core.VENV_PY
subprocess.run([separate_core.find_ffmpeg(), '-version'], check=True)
report = {'python': sys.version, 'cases': []}
with tempfile.TemporaryDirectory(prefix='Amu smoke 中文 ') as tmp:
    tmp = Path(tmp)
    t = np.arange(44100 * 8) / 44100
    signal = np.stack([0.1*np.sin(2*np.pi*220*t)+0.03*np.sin(2*np.pi*330*t),
                       0.1*np.sin(2*np.pi*220*t)+0.03*np.sin(2*np.pi*440*t)], axis=1)
    inp = tmp / 'input.wav'
    sf.write(inp, signal, 44100)
    expected = {
        'basic': ['vocals', 'no_vocals'],
        'four': ['vocals', 'no_vocals', 'drums', 'bass', 'other'],
        'multi': ['lead', 'harmony', 'no_vocals', 'drums', 'bass', 'guitar', 'piano', 'other'],
    }
    for mode, fmt in [('basic', 'wav'), ('four', 'wav'), ('multi', 'wav'), ('multi', 'mp3')]:
        result = separate_core.run_separation(str(inp), str(tmp/(mode+'-'+fmt)),
            mode=mode, device='cuda', fmt=fmt, shifts=1, overlap=0.25)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        assert result['ok'], result
        assert [s['name'] for s in result['stems']] == expected[mode], result
        for stem in result['stems']:
            p = Path(stem['path'])
            assert p.suffix == '.' + fmt and p.stat().st_size > 1000, stem
            if fmt == 'wav':
                x, rate = sf.read(p)
                assert np.isfinite(x).all() and rate == 44100
        if fmt == 'mp3':
            assert not list(Path(result['outdir']).rglob('*.wav'))
        report['cases'].append({'mode': mode, 'format': fmt,
            'stems': [s['name'] for s in result['stems']], 'elapsed': result['elapsed']})
report['status'] = 'passed'
Path(sys.argv[2]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

