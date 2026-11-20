import importlib.util
import os
import socket
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

APP = Path(__file__).resolve().parents[1] / 'app'
sys.path.insert(0, str(APP))


def load(name):
    spec = importlib.util.spec_from_file_location(name, APP / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MacRuntimeTests(unittest.TestCase):
    def test_busy_port_uses_another_local_port(self):
        with patch.dict(sys.modules, {'app': types.SimpleNamespace(main=lambda: None)}):
            launcher = load('launcher_mac')
        self.assertTrue(hasattr(launcher, 'choose_port'), 'missing occupied-port handling')
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1', 0))
            occupied.listen()
            port = occupied.getsockname()[1]
            self.assertNotEqual(launcher.choose_port(port), port)

    def test_no_browser_option_is_not_forwarded_as_port(self):
        seen = []
        stub = types.SimpleNamespace(main=lambda: seen.append(sys.argv[1:]))
        with patch.dict(sys.modules, {'app': stub}):
            launcher = load('launcher_mac')
            with patch.object(sys, 'argv', ['launcher_mac.py', '--no-browser', '18877']):
                launcher.main()
        self.assertEqual(seen, [['18877']])

    def test_uploads_use_writable_data_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'AMU_DATA_DIR': tmp}):
                app = load('app')
            self.assertEqual(Path(app.UPLOAD_DIR), Path(tmp) / 'uploads')

    def test_mac_cuda_request_reaches_worker_as_cpu(self):
        core = load('separate_core')
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            def worker(inp, outdir, model, cfg, device, *args):
                seen.append(device)
                for name in ['vocals', 'no_vocals']:
                    Path(outdir, name + '.wav').write_bytes(b'fixture')
                return True, None
            with patch.object(sys, 'platform', 'darwin'), \
                 patch.object(core, 'DEMUCS', __file__), \
                 patch.object(core, 'VENV_PY', sys.executable), \
                 patch.object(core, '_run_via_api', worker):
                result = core.run_separation('input.wav', tmp, mode='basic', device='cuda')
            self.assertTrue(result['ok'], result)
            self.assertEqual(seen, ['cpu'])


if __name__ == '__main__':
    unittest.main()
