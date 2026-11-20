#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""乐器分离 worker：MVSep Mega 53 Stems（BS-RoFormer，MIT）

由 separate_core 用发布包内的 venv python 调用（本进程需要 torch / numpy / einops /
beartype / rotary_embedding_torch / yaml，以及包内 amu_msst 最小运行时）。

为什么必须用子进程：
  - 主服务进程跑在精简环境里，且 53 轨模型 + 一次推理会常驻数 GB 显存；
    子进程结束即彻底释放，不会污染后续 Demucs 任务。
  - 崩溃（如 CUDA OOM）不会拖垮 Web 服务。

三项关键优化（都是本机 RTX 3060 6GB 实测逼出来的）：
  1. **短上下文推理**：训练 chunk=10s 需要 7.5GB 显存；推理时把上下文缩到
     3s 只需 5.1GB。用 sqrt(hann) 窗重叠相加还原，质量损失很小。
  2. **只累积目标轨**：全 53 轨的累积缓冲在一首 4 分钟歌上要 4.4GB，
     直接爆显存。这里按用户选的目标轨建缓冲（通常 2~6 条），且**放在 CPU**，
     GPU 只保留当前窗口的前向中间量。
  3. **OOM 自动降档**：捕获显存不足后清空缓存、缩短 chunk 重试。

协议（与 demucs_api_run.py 保持一致）：
  进度   → stderr，每行一个 JSON：{"progress": 42.5, "seconds": 100}
  结果   → stdout **最后一行** JSON：{"ok":true,"stems":[{"name","path","size"}]...}

⚠️ 版权须知：本模块仅改变音频的表达形式，产出轨的版权仍归原录音权利人所有。
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time

# ---- 离线支持：模型权重指向包内 _hfcache（必须在 import torch 之前设置）
_HF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_hfcache")
if os.path.isdir(_HF):
    os.environ["HF_HOME"] = _HF
    os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 44100

# 权重文件名（放在 models/mega53/ 下）
CKPT_NAME = "mvsep_mega_53.ckpt"
YAML_NAME = "mega53.yaml"

# 模型输出通道顺序 = 训练配置 training.instruments 的顺序，**不能重排**
FALLBACK_STEMS = [
    'accordion', 'acoustic-guitar', 'back-vocal', 'banjo', 'bass', 'bassoon', 'bells',
    'bowed_strings', 'brass', 'cello', 'clarinet', 'congas', 'digital-piano', 'dobro',
    'double-bass', 'drums', 'electric-guitar', 'flute', 'french-horn', 'glockenspiel',
    'guitar', 'harmonica', 'harp', 'harpsichord', 'hh', 'keys', 'kick', 'lead-vocal',
    'mandolin', 'marimba', 'oboe', 'organ', 'percussion', 'piano', 'saxophone', 'sitar',
    'snare', 'strings', 'synth', 'tambourine', 'timpani', 'toms', 'triangle', 'trombone',
    'trumpet', 'tuba', 'ukulele', 'viola', 'violin', 'vocal', 'wind', 'wind-chimes',
    'woodwind',
]

# 界面选项 → 模型通道。一个选项可对应多个通道（同一件乐器常被判到不同的 53 类标签上，
# 如《醒》的贝斯落在 bass、《普通朋友》落在 double-bass，只取一个会拿到静音轨）。
# 与 separate_core.INSTRUMENT_TARGETS 一一对应；启动时会尝试以 separate_core 为准校准。
CHANNELS = {
    "electric-guitar": ["electric-guitar"],
    "acoustic-guitar": ["acoustic-guitar"],
    "bass":            ["bass", "double-bass"],
    "drums":           ["drums"],
    "percussion":      ["percussion"],
    "piano":           ["piano", "digital-piano"],
    "keys":            ["keys", "synth", "organ"],
    "lead-vocal":      ["lead-vocal"],
    "back-vocal":      ["back-vocal"],
    "violin":          ["violin"],
    "cello":           ["cello"],
    "harp":            ["harp"],
    "saxophone":       ["saxophone"],
    "trumpet":         ["trumpet"],
    "trombone":        ["trombone"],
    "flute":           ["flute"],
    "clarinet":        ["clarinet"],
}
try:                                    # 以 separate_core 的真值为准，避免两份表走偏
    sys.path.insert(0, HERE)
    from separate_core import INSTRUMENT_CHANNELS as _SC  # type: ignore
    if isinstance(_SC, dict) and _SC:
        CHANNELS = {k: list(v) for k, v in _SC.items()}
except Exception:
    pass


def _emit(obj):
    """进度行写 stderr（每行一个 JSON），父进程据此更新真实百分比。"""
    print(json.dumps(obj, ensure_ascii=False), file=sys.stderr, flush=True)


def _log(msg):
    print("[instr] " + str(msg), file=sys.stderr, flush=True)


def _user_model_dir():
    """用户级写盘兜底目录（与 mega53_download._user_model_dir 保持一致）。"""
    if os.name == "nt":
        app = os.environ.get("APPDATA", "").strip()
        if app:
            return os.path.join(app, "AmuTools", "models", "mega53")
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Application Support",
                            "AmuTools", "models", "mega53")
    return os.path.join(home, ".config", "AmuTools", "models", "mega53")


def resolve_model_dir(explicit=""):
    """定位权重目录。

    顺序：显式 --model-dir > 环境变量 AMU_MEGA53_DIR > 源码同级 models/mega53
    > 上级 models/mega53 > 用户目录（安装到 Program Files 时包内不可写，
    下载的模型会落在用户目录，see mega53_download._user_model_dir）。
    返回 (目录, 说明)；找不到时目录为 ''。
    """
    cands = []
    if explicit:
        cands.append(explicit)
    env = os.environ.get("AMU_MEGA53_DIR", "").strip()
    if env:
        cands.append(env)
    for base in (HERE, os.path.dirname(HERE)):
        cands.append(os.path.join(base, "models", "mega53"))
    cands.append(_user_model_dir())
    for c in cands:
        if c and os.path.isfile(os.path.join(c, CKPT_NAME)) and \
                os.path.isfile(os.path.join(c, YAML_NAME)):
            return c, ""
    tried = "\n  ".join(cands)
    return "", "未找到乐器分离模型权重（%s + %s）。已尝试：\n  %s" % (
        CKPT_NAME, YAML_NAME, tried)


def load_stems(yaml_path):
    """读取模型配置文件里的 instruments 顺序。

    这份顺序决定模型输出通道，必须从权重自带的 yaml 读，不能硬编码。
    """
    import yaml

    class _L(yaml.SafeLoader):
        pass
    # 配置文件里有 !!python/tuple 标签，注册构造器避免报错
    _L.add_constructor("tag:yaml.org,2002:python/tuple",
                       lambda l, n: tuple(l.construct_sequence(n)))
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=_L)
    stems = (((cfg.get("training") or {}).get("instruments")) or [])
    if len(stems) != 53:
        _log("配置文件 instruments 数量异常(%d)，回退内置顺序" % len(stems))
        return list(FALLBACK_STEMS)
    return list(stems)


def load_model(model_dir, device):
    """构建 BSRoformer 并加载权重。返回 (model, stems)。"""
    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from amu_msst import BSRoformer  # 包内最小运行时（MIT，来源见 amu_msst/LICENSE_MSST.txt）

    stems = load_stems(os.path.join(model_dir, YAML_NAME))
    import yaml

    class _L(yaml.SafeLoader):
        pass
    _L.add_constructor("tag:yaml.org,2002:python/tuple",
                       lambda l, n: tuple(l.construct_sequence(n)))
    with open(os.path.join(model_dir, YAML_NAME), "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=_L)

    model = BSRoformer(**dict(cfg["model"]))
    ck = torch.load(os.path.join(model_dir, CKPT_NAME), map_location="cpu",
                    weights_only=False)
    sd = ck.get("state_dict") or ck.get("model") or ck
    model.load_state_dict(sd, strict=True)
    del ck, sd
    gc.collect()
    model = model.to(device).eval()
    return model, stems


def pick_chunk(device):
    """按显卡总显存挑推理上下文长度（秒）。

    实测（RTX 3060 6GB，fp32 + autocast fp16）：
        10s→7548MB(爆) 6s→6177MB(爆) 4s→5488MB 3s→5150MB 2s→4801MB
    保守留 ~0.5GB 余量。
    """
    if device != "cuda":
        return 2.0
    try:
        import torch
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        return 3.0
    if total_gb >= 11.5:
        return 4.0
    if total_gb >= 8:
        return 3.5
    if total_gb >= 5.8:      # RTX 3060 6GB 实测 3s → 5150MB，刚好放得下
        return 3.0
    if total_gb >= 4.5:
        return 2.5
    return 2.0


def read_audio(path, ffmpeg, progress_from=0.0, progress_to=10.0):
    """用 ffmpeg 统一转成 44.1kHz 立体声 float32 读入。

    ffmpeg 是外部进程调用，不影响本工具许可；这里只做「转码 → 管道」，
    不落临时文件。返回 torch.Tensor shape (2, T)。
    """
    import numpy as np
    import torch

    cmd = [ffmpeg, "-v", "error", "-i", path, "-ac", "2", "-ar", str(SR),
           "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("音频解码失败：%s" % (r.stderr or b"")[-400:].decode(
            "utf-8", "replace"))
    arr = np.frombuffer(r.stdout, dtype=np.float32)
    if arr.size == 0:
        raise RuntimeError("音频解码结果为空（文件可能损坏或不是音频）")
    arr = arr.reshape(-1, 2)
    _emit({"progress": progress_to, "stage": "decode"})
    return torch.from_numpy(arr.T.copy())


def separate(model, mix, targets_idx, device, chunk_s=3.0, overlap=4,
             progress_from=10.0, progress_to=95.0):
    """重叠相加推理。

    targets_idx : [(缓冲下标 j, [模型通道下标...])] —— 只为选中的轨建缓冲区。
        一个界面选项可对应多个模型通道（如「贝斯」= bass + double-bass），
        这些通道相加输出。
    返回 numpy 数组 shape (n_sel, 2, T)，保持模型输出的相对电平关系。

    ⚠️ 累积缓冲放在 CPU：全轨（甚至 6 轨）的长音频缓冲放 GPU 会显著抬高显存峰值，
       而每个 chunk 的搬运量只有几 MB，实测几乎不影响速度。
    """
    import torch

    # 样本数必须能被 overlap 整除，否则最后一个块的步长跨越会错位。
    # 自动向下对齐（例如 2.5s→110250，overlap=4 时取 110248）。
    chunk = (int(SR * chunk_s) // overlap) * overlap
    step = chunk // overlap
    win = torch.hann_window(chunk, periodic=False, device=device) ** 0.5

    n = mix.shape[1]
    pad_l = chunk - step      # 左侧也要留余量，否则头部累积窗为 0
    pad_r = chunk
    mix = torch.nn.functional.pad(mix, (pad_l, pad_r))
    total = mix.shape[1]

    n_sel = len(targets_idx)
    out = torch.zeros(n_sel, 2, total, dtype=torch.float32)
    acc = torch.zeros(2, total, dtype=torch.float32)
    win_cpu = win.cpu()
    nchunk = (total - chunk) // step + 1
    t0 = time.time()
    _last_emit = [t0 - 9.0]      # 上次上报时刻：按秒节流，别按块数
    for i in range(nchunk):
        s = i * step
        x = mix[:, s:s + chunk].unsqueeze(0).to(device)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                res = model(x)
        L = res.shape[-1]
        piece = res[0].float().cpu()          # (53, 2, L)
        del x, res
        for j_out, j_ins in targets_idx:
            # 同一件乐器可能分散在多个模型通道上，这里相加（不是取最大）
            for j_in in j_ins:
                out[j_out, :, s:s + L] += piece[j_in][:, :L] * win_cpu[:L]
        acc[:, s:s + L] += win_cpu[:L]
        del piece
        # 每 ~1 秒上报一次（末块必报）。短音频也能得到平滑进度，
        # 长音频不会因为「每 20 块」而几分钟才动一格。
        now = time.time()
        if now - _last_emit[0] >= 1.0 or i == nchunk - 1:
            _last_emit[0] = now
            if device == "cuda":
                torch.cuda.empty_cache()
            done = min(n, max(0, (i + 1) * step - pad_l)) / SR
            pct = progress_from + (progress_to - progress_from) * ((i + 1) / nchunk)
            _emit({"progress": round(pct, 1), "seconds": round(done, 1),
                   "stage": "run", "elapsed": round(now - t0, 1)})

    core = acc[:, pad_l:pad_l + n]
    if float(core.min()) < 1e-4:
        raise SystemExit("累积窗最小值过小: %.3e（请检查 chunk / overlap 配置）" % core.min())
    out = out[:, :, pad_l:pad_l + n] / core.unsqueeze(0)
    return out.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True, help="最终轨道目录")
    ap.add_argument("--targets", required=True, help="目标轨名，逗号分隔")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=float, default=0.0, help="0=按显存自动")
    ap.add_argument("--overlap", type=int, default=4)
    ap.add_argument("--fmt", default="wav")
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--ffmpeg", default="")
    a = ap.parse_args()

    import numpy as np
    import torch

    # rotary_embedding_torch 在 torch 2.5 下会刷屏 FutureWarning，与结果无关
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    t0 = time.time()
    try:
        # 设备可用性
        device = a.device
        if device == "cuda":
            try:
                if not torch.cuda.is_available():
                    device = "cpu"
            except Exception:
                device = "cpu"
        if device == "cpu":
            _log("未检测到可用 CUDA，改用 CPU —— 速度会慢几十倍，建议确认显卡驱动")

        # 模型位置
        model_dir, why = resolve_model_dir(a.model_dir)
        if not model_dir:
            print(json.dumps({"ok": False, "error": why}, ensure_ascii=False))
            return
        _emit({"progress": 1.0, "stage": "load"})
        model, stems = load_model(model_dir, device)
        after_load = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else 0
        _log("模型就绪：%s | 常驻显存 %.0f MB" % (model_dir, after_load))

        # 目标轨校验 + 通道展开（一个选项可能对应多个模型通道，见 separate_core.INSTRUMENT_CHANNELS）
        want = [t.strip() for t in a.targets.split(",") if t.strip()]
        bad = [t for t in want if t not in CHANNELS]
        if bad:
            print(json.dumps({"ok": False,
                              "error": "不支持的目标乐器: %s" % ", ".join(bad)},
                             ensure_ascii=False))
            return
        missing_ch = sorted({c for t in want for c in CHANNELS[t] if c not in stems})
        if missing_ch:
            print(json.dumps({"ok": False,
                              "error": "权重通道表缺少 %s，请核对 mega53.yaml"
                                       % ", ".join(missing_ch)}, ensure_ascii=False))
            return
        # [(缓冲下标, [通道下标...])]
        targets_idx = [(j, [stems.index(c) for c in CHANNELS[t]])
                       for j, t in enumerate(want)]
        _emit({"progress": 6.0, "stage": "decode"})

        # 解码
        ff = a.ffmpeg
        if not ff or not os.path.isfile(ff):
            import shutil
            ff = shutil.which("ffmpeg") or "ffmpeg"
        mix = read_audio(a.input, ff)

        # 推理（显存不足自动缩短上下文重试）
        chunk = a.chunk if a.chunk > 0 else pick_chunk(device)
        last_err = None
        for attempt in range(3):
            try:
                _log("推理开始 %.1fs 音频 | chunk=%.1fs overlap=%d | 目标 %d 轨"
                     % (mix.shape[1] / SR, chunk, a.overlap, len(want)))
                arr = separate(model, mix, targets_idx, device, chunk_s=chunk,
                               overlap=a.overlap)
                break
            except RuntimeError as e:
                last_err = e
                msg = str(e).lower()
                if ("out of memory" not in msg) or chunk <= 1.5:
                    raise
                if device == "cuda":
                    torch.cuda.empty_cache()
                chunk = max(1.5, chunk - 1.0)
                _log("显存不足，降到 chunk=%.1fs 重试" % chunk)
        else:
            raise last_err or RuntimeError("推理失败")

        peak = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else 0
        dur_sec = mix.shape[1] / SR
        _log("推理完成 %.1fs 用时 %.1fs (RTF %.2f) | 显存峰值 %.0f MB"
             % (dur_sec, time.time() - t0, (time.time() - t0) / max(dur_sec, 1e-6), peak))
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # 保存：所有轨乘**同一个**缩放因子（沿用输入电平关系，不做逐轨归一化）
        from demucs.audio import save_audio
        ext = "flac" if str(a.fmt).lower() == "flac" else "wav"
        os.makedirs(a.outdir, exist_ok=True)
        gmax = max(float(np.abs(arr[j]).max()) for j in range(arr.shape[0]))
        scale = 1.0 / max(1.01 * gmax, 1.0)
        out = []
        for j, name in enumerate(want):
            x = torch.from_numpy(np.ascontiguousarray(arr[j] * scale))
            path = os.path.join(a.outdir, name + "." + ext)
            save_audio(x, path, SR, clip="none", bits_per_sample=16)
            size = os.path.getsize(path)
            out.append({"name": name, "path": path, "size": size})
            _log("  %-18s 峰值 %.4f  已保存 %.1f MB" % (name, gmax and float(
                np.abs(arr[j]).max()), size / 1048576.0))
            _emit({"progress": 95.0 + 4.0 * (j + 1) / len(want), "stage": "save"})

        print(json.dumps({"ok": True, "stems": out, "samplerate": SR,
                          "elapsed": round(time.time() - t0, 1),
                          "chunk": chunk, "peak_mb": round(peak)},
                         ensure_ascii=False))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(json.dumps({"ok": False, "error": repr(e)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
