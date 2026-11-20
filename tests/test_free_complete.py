"""免费完整版的接线回归：不加载 1.37GB Mega53 权重，只验证功能已接上。"""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / 'app'
sys.path.insert(0, str(APP))


def load(name):
    spec = importlib.util.spec_from_file_location(name, APP / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FreeCompleteTests(unittest.TestCase):
    def test_free_mode_is_on(self):
        app = load('app')
        self.assertTrue(app.FREE_MODE, 'FREE_MODE 必须为 True（完全免费）')
        self.assertTrue(app.INTERNAL)
        self.assertEqual(app.GATE_ST['ok'], True)

    def test_all_five_modes_registered(self):
        app = load('app')
        self.assertEqual(set(app.MODE_TRACKS), {'basic', 'four', 'multi', 'instruments', 'pro'})

    def test_instrument_pipeline_files_present(self):
        for name in ('instrument_worker.py', 'mega53_download.py',
                     'amu_msst/__init__.py', 'amu_msst/bs_roformer.py',
                     'amu_msst/LICENSE_MSST.txt'):
            self.assertTrue((APP / name).is_file(), '缺少乐器模块: %s' % name)

    def test_instrument_targets_count(self):
        core = load('separate_core')
        self.assertGreaterEqual(len(core.INSTRUMENT_TARGETS), 17, '乐器选项数量不足')

    def test_frontend_is_free_edition(self):
        html = (APP / 'web' / 'index.html').read_text(encoding='utf-8')
        # payNote 是「日后恢复商业化」的可逆文案，默认 hidden；首屏显示的是 freeNote
        self.assertIn('id="freeNote"', html)
        self.assertIn('id="payNote" hidden', html)
        self.assertIn('完全免费', html)
        self.assertIn('无需登录', html)

    def test_no_old_login_promise_in_manual(self):
        for name in ('使用说明.txt', '使用说明_Intel.txt'):
            t = (APP.parent / 'build' / name).read_text(encoding='utf-8')
            self.assertIn('完全免费', t)
            self.assertNotIn('仍需使用 Amu 账号登录', t)
            self.assertNotIn('积分规则与 Windows 商业版相同', t)


if __name__ == '__main__':
    unittest.main()
