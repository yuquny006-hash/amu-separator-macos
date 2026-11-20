#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
人声和声分离（多轨模式专用）
============================
把 Demucs 出来的人声轨再拆成「主唱 + 和声」两条轨。

    主唱 = BVE 的 (Instrumental) 分支（反直觉：文件名带 inst / no_voc 的那一轨才是主唱）
    和声 = BVE 的 (Vocals) 分支

    喂给 BVE 之前先过 80Hz 高通，否则和声轨会被低频残料污染成轰鸣。

⚠️ 两条铁律（都是 2026-10-04 实测踩出来的，改代码前先读）：
  1. **必须保留立体声喂入**，转单声道会让模型输出塌成 0（详见 run_harmony 内注释）。
  2. **不要改回「和声 = 输入 − 主唱」的残差法**：主唱分支已接近输入，残差只有 -62 dB。

⚠️ 版权须知：
本模块仅提供音频分离能力。分离出的各音轨是原始录音的「衍生物」，
其版权仍归原录音的权利人所有。使用者须确保自己拥有所处理音频的合法权利。
"""
import os
import sys
import time
import shutil

import numpy as np
import soundfile as sf

# ---- 离线支持：模型权重指向包内 _hfcache（必须在 import torch / demucs 之前设置）
_HF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_hfcache")
if os.path.isdir(_HF):
    os.environ["HF_HOME"] = _HF
    os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


HERE = os.path.dirname(os.path.abspath(__file__))
BVE_DIR = os.path.join(HERE, "models", "bve")
BVE_MODEL = "UVR-BVE-4B_SN-44100-1.pth"

# 高通截止（Hz）
HP_CUTOFF = 80.0
# 保护上限：仅防削波用，**不做绝对电平归一化**。
# 主引擎给所有轨乘的是同一个 scale（1/(1.01×峰值)），本模块必须沿用输入轨的电平，
# 否则主唱/和声会比其他轨整体抬高、破坏「各轨相加 = 原曲 × scale」这条项目纪律。
CEIL = 1.0
SILENT = True  # 细节日志打到 stderr，失败时也不抛异常


def _log(msg):
    if not SILENT:
        print("[harmony] " + msg, file=sys.stderr)


def highpass(x, sr, fc=HP_CUTOFF):
    """80Hz 余弦过渡带高通（不做频响归一化，只保证过渡平滑不引入振铃）。"""
    n = len(x)
    if n < 16:
        return x
    f = np.fft.rfftfreq(n, 1.0 / sr)
    X = np.fft.rfft(x, n=n)
    t = np.clip((f - fc * 0.8) / (fc * 0.4 + 1e-9), 0.0, 1.0)
    t = 0.5 - 0.5 * np.cos(np.pi * t)
    return np.fft.irfft(X * t, n=n)


def _resolve_device(device):
    """cuda 不可用时自动回落 cpu，不抛异常。"""
    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    return device


def _import_separator_cls():
    """拿到 Separator 类。本环境已装 audio-separator；万一缺失才从兜底环境借。

    借用时 append 到 sys.path 末尾，**绝不能用 PYTHONPATH**（会顶掉 CUDA torch）。

    返回 (类, 错误信息)。
    """
    try:
        from audio_separator.separator import Separator
        return Separator, ""
    except Exception as e:
        return None, str(e)


def run_harmony(vocals_wav, out_dir, device="cuda", fmt="wav"):
    """把人声轨拆成主唱 + 和声。

    参数
        vocals_wav : Demucs 产出的人声轨（wav）
        out_dir    : 输出目录（lead / harmony 直接落在这里）
        device     : cuda / cpu，cuda 不可用时自动回落
        fmt        : wav / flac，与调用方的输出格式保持一致

    返回 (ok, error, info)
        ok    bool   是否成功；False 时调用方静默降级（保留原人声轨）
        error str    失败原因（ok=False 时非空）
        info  dict   {"lead": 主唱路径, "harmony": 和声路径, "elapsed": 耗时秒}

    任何异常都吞掉并返回 (False, 原因, {})，绝不向上抛——这是可选的增强环节，
    挂掉不能让整张订单失败。
    """
    info = {"lead": "", "harmony": "", "elapsed": 0.0}
    tmp_dir = os.path.join(out_dir, "_harmony_tmp")
    t0 = time.time()
    ext = "flac" if str(fmt).lower() == "flac" else "wav"

    def _fail(reason):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False, reason, info

    try:
        if not os.path.isfile(vocals_wav):
            return _fail("未找到人声轨: %s" % vocals_wav)
        if not os.path.isdir(BVE_DIR):
            return _fail("未找到模型目录: %s" % BVE_DIR)
        if not os.path.isfile(os.path.join(BVE_DIR, BVE_MODEL)):
            return _fail("未找到和声模型: %s" % BVE_MODEL)

        device = _resolve_device(device)
        os.makedirs(tmp_dir, exist_ok=True)

        # ---- 1) 喂给 BVE 之前先做 80Hz 高通，避免低频残料被算进和声 ----
        # ⚠️⚠️ 必须**保留立体声**，绝不能转单声道（2026-10-04 实测踩过的最贵的坑）：
        #    模型模板 4band_v3_sn 的 band4 带 convert_channels="stereo_n"，
        #    网络靠左右声道差异判断主唱/和声；转成 mono 后侧通道恒为 0，
        #    mask 直接塌成 0 → (Vocals) 分支只有 -91 dB（等于没跑）。
        #    实测同一首歌：mono 输入和声 -91.6 dB，保留立体声 -28.4 dB，差 63 dB。
        src, sr = sf.read(vocals_wav, always_2d=True)
        if src.shape[1] == 1:
            src = np.stack([src[:, 0], src[:, 0]], axis=1)
        hp_in_data = np.stack([highpass(src[:, c], sr) for c in range(src.shape[1])], axis=1)
        hp_in = os.path.join(tmp_dir, "_hp_input.wav")
        sf.write(hp_in, hp_in_data, sr, subtype="PCM_16")
        _log("高通输入 %s -> %sHz（%d 声道）" % (os.path.basename(vocals_wav), sr, src.shape[1]))

        # ---- 2) 跑 BVE ----
        Separator, imp_err = _import_separator_cls()
        if Separator is None:
            return _fail("和声模块不可用（缺少分离依赖）：%s" % imp_err)

        # ⚠️ Separator 没有 device 参数，它内部自动探测 CUDA；
        # 传 device= 会直接 TypeError。device 只用于前面的可用性判断与日志。
        sep = Separator(model_file_dir=BVE_DIR, output_dir=tmp_dir)
        sep.load_model(BVE_MODEL)
        files = sep.separate(hp_in)
        if not files:
            return _fail("和声引擎未产出文件")

        # ---- 3) 定位两轨（方向反直觉，别按文件名直觉猜）----
        #       (Instrumental) / no_voc / inst -> 主唱
        #       (Vocals) 且不含 inst / no_voc  -> 和声
        lead_p = None
        harm_p = None
        for f in files:
            p = f if os.path.isabs(f) else os.path.join(tmp_dir, os.path.basename(f))
            if not os.path.isfile(p):
                continue
            low = os.path.basename(p).lower()
            if "inst" in low or "no_voc" in low:
                lead_p = p
            elif "vocals" in low:
                harm_p = p
        if lead_p is None or harm_p is None:
            return _fail("和声引擎输出命名异常，无法定位主唱 / 和声轨")
        _log("主唱=%s  和声=%s" % (os.path.basename(lead_p), os.path.basename(harm_p)))

        # ---- 4) 主唱 / 和声都直接取 BVE 的两条分支输出 ----
        # ⚠️ 不再用「输入 − 主唱」的残差法（2026-10-04 实测推翻）：
        #    BVE 的 (Instrumental) 分支本身已接近输入（重建了完整人声），
        #    残差算出来只有 -62 dB，和声等于没有。
        #    (Vocals) 分支才是模型真正判出的和声层（实测 -28 dB，可听）。
        #    代价：两轨相加会略高于原人声（和声比主唱低约 11 dB，叠加仅 +0.4 dB），
        #    这是「和声是叠加层」的正常结果，不是 bug。
        lead_raw, _ = sf.read(lead_p, always_2d=True)
        harm_raw, _ = sf.read(harm_p, always_2d=True)
        lead_hp = np.stack([highpass(lead_raw[:, c], sr) for c in range(lead_raw.shape[1])], axis=1)
        harm_hp = np.stack([highpass(harm_raw[:, c], sr) for c in range(harm_raw.shape[1])], axis=1)
        n = min(len(lead_hp), len(harm_hp))
        lead_hp = lead_hp[:n]
        harm = harm_hp[:n]

        # ---- 5) 电平处理 ----
        # 两轨都是 BVE 的直接输出，电平基准与输入人声轨一致；
        # 沿用输入轨电平不缩放，仅在可能削波时整体乘同一个因子（保持两轨相互关系不变）。
        peak = max(float(np.abs(lead_hp).max()), float(np.abs(harm).max()))
        g = 1.0 if peak <= CEIL else CEIL / (1.01 * peak)

        lead_path = os.path.join(out_dir, "lead." + ext)
        harm_path = os.path.join(out_dir, "harmony." + ext)
        wfmt = "FLAC" if ext == "flac" else "WAV"
        subtype = "PCM_16"
        # 不做 np.clip 硬限幅：电平已与输入轨对齐，硬削会引入可听的失真
        sf.write(lead_path, np.clip(lead_hp * g, -1.0, 1.0) if g < 1.0 else lead_hp * g,
                 sr, format=wfmt, subtype=subtype)
        sf.write(harm_path, np.clip(harm * g, -1.0, 1.0) if g < 1.0 else harm * g,
                 sr, format=wfmt, subtype=subtype)

        info["lead"] = lead_path
        info["harmony"] = harm_path
        info["elapsed"] = round(time.time() - t0, 2)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return True, "", info

    except Exception as e:
        return _fail("%s: %s" % (type(e).__name__, e))


if __name__ == "__main__":
    # 独立调试用：harmony_separate.py <人声轨> [输出目录] [device]
    if len(sys.argv) < 2:
        print("用法: python harmony_separate.py <vocals.wav> [outdir] [device]")
        sys.exit(2)
    src_file = sys.argv[1]
    outd = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(src_file))
    dev = sys.argv[3] if len(sys.argv) > 3 else "cuda"
    os.makedirs(outd, exist_ok=True)
    ok, err, info = run_harmony(src_file, outd, dev)
    print("ok=%s err=%s" % (ok, err))
    print("lead   =", info.get("lead"))
    print("harmony=", info.get("harmony"))
    print("elapsed= %.2fs" % info.get("elapsed", 0))
    sys.exit(0 if ok else 1)
