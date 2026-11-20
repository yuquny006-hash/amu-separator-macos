#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
分离核心（CLI 与 Web 服务共用）
==============================
引擎：Demucs (Meta, MIT License) —— 代码与预训练权重均为 MIT，可闭源商用。

    三种分离模式（对应用户界面的三个档位）：
    基本模式 basic : 人声 / 伴奏                    （htdemucs + --two-stems=vocals，最快）
    四轨模式 four  : 人声 / 鼓 / 贝斯 / 其他          （htdemucs_ft，精修 4-stem）
    多轨模式 multi : 人声 / 鼓 / 贝斯 / 吉他 / 钢琴 / 其他 （htdemucs_6s，6-stem）

    输出格式：wav（无损）/ flac（无损）/ mp3（320k，体积最小）。
    MP3 转换调用本机 ffmpeg 作为外部进程（subprocess），不属于本工具代码的一部分，
    不产生衍生作品；MP3 格式专利已于 2017 年到期，可自由商用。

⚠️ 版权须知（重要）：
本工具仅提供音频分离能力。分离出的各音轨是原始录音的「衍生物」，
其版权仍归原录音的权利人所有，不属于本工具或使用者。
使用者须确保自己拥有所处理音频的合法权利（如自有录音、已授权作品）。
将他人受版权保护的歌曲输入本工具进行分离/传播，责任由使用者承担。
"""
import os
import sys
import json
import time
import threading
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


IS_WIN = (os.name == "nt")
EXE = ".exe" if IS_WIN else ""

# venv 目录名候选（Windows 发布包用 .venv312；mac/linux 习惯 .venv / venv）
VENV_NAMES = (".venv312", ".venv", "venv")


def _venv_ok(path):
    """判断 path 是否为一个可用的 Python 运行环境（跨平台）。"""
    if not os.path.isdir(path):
        return False
    py = os.path.join(path, "Scripts" if IS_WIN else "bin", "python" + EXE)
    return os.path.isfile(py)


def _find_venv_root():
    """自动定位 Python 运行环境（.venv312）所在目录。

    运行环境只保留一份，放在发布包内：
        <项目根>/交付给用户/①发给用户_安装包/AmuMusic音频分离软件_v1.0.0/_runtime/.venv312
    这里同时兼容三种位置，避免根目录与发布包各存一份 7.6G 的重复环境：
        1) 本目录下（源码与运行环境同级，即发布包 _runtime 那种布局）
        2) 上级目录的 .venv312（早期布局）
        3) 交付给用户/①发给用户_安装包/*/_runtime/.venv312（项目根目录布局）
    返回 venv 的绝对路径；都找不到则返回第一候选（调用方会报「未找到运行环境」）。
    Windows 与 macOS/Linux 均支持（macOS/Linux 下 venv 可执行文件在 bin/ 子目录）。
    环境变量 AMU_VENV 可显式指定 venv 路径（打包 / 非常规布局时用），优先级最高。
    """
    env = os.environ.get("AMU_VENV", "").strip()
    if env and _venv_ok(env):
        return env
    cands = []
    for base in (HERE, os.path.dirname(HERE)):
        for name in VENV_NAMES:
            cands.append(os.path.join(base, name))
    # 沿项目根目录向下找 交付给用户/①发给用户_安装包/*/_runtime/.venv312
    for base in (HERE, os.path.dirname(HERE)):
        pkg = os.path.join(base, "交付给用户")
        if os.path.isdir(pkg):
            inst = os.path.join(pkg, "①发给用户_安装包")
            if os.path.isdir(inst):
                for name in sorted(os.listdir(inst)):
                    for vn in VENV_NAMES:
                        p = os.path.join(inst, name, "_runtime", vn)
                        if os.path.isdir(p):
                            cands.append(p)
    for c in cands:
        if _venv_ok(c):
            return c
    return cands[0]


def _venv_bin(name):
    """venv 内可执行文件的完整路径（Windows 在 Scripts/、mac/linux 在 bin/）。"""
    return os.path.join(VENV_DIR, "Scripts" if IS_WIN else "bin", name + EXE)


VENV_DIR = _find_venv_root()
DEMUCS = _venv_bin("demucs")
# Python API worker：用 venv 内的 torch / demucs 跑，统一增益保存 + 生成「原曲−人声」伴奏轨
VENV_PY = _venv_bin("python")
API_WORKER = os.path.join(HERE, "demucs_api_run.py")
# 和声模块：把多轨模式下 Demucs 的人声轨拆成 主唱 / 和声
# 惰性导入——只有 multi 模式才会用到，basic / four 不该为它付出导入代价
HARMONY_MODULE = os.path.join(HERE, "harmony_separate.py")
# 乐器提取模块：MVSep Mega 53 轨（BS-RoFormer，MIT），跑在 venv 子进程里
INSTRUMENT_MODULE = os.path.join(HERE, "instrument_worker.py")
# 使用 HF 镜像加速模型权重首次下载（约 19MB/s）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---------------------------------------------------------------- 模式定义
MODES = {
    "basic": {
        "label": "基本模式",
        "model": "htdemucs",
        "two_stems": "vocals",
        # minus：伴奏 = 原曲 − 人声（纯乐器段落 100% 原样，零重建音准漂移）
        # 若改回 add：伴奏 = 鼓+贝斯+其他 三轨重建相加（电平被压、贝斯易飘音）
        "other_method": "minus",
        "stems": ["vocals", "no_vocals"],
        # demucs 在 other_method=minus 下输出的文件名是 minus_vocals，需映射回 no_vocals
        "alias": {"no_vocals": "minus_vocals"},
        "desc": "人声 / 伴奏（2 轨，最快）",
    },
    "four": {
        "label": "四轨模式",
        "model": "htdemucs_ft",
        "two_stems": None,
        "stems": ["vocals", "no_vocals", "drums", "bass", "other"],
        "desc": "人声 / 贝斯 / 鼓 / 其他（4 轨，精修）",
    },
    "multi": {
        "label": "多轨模式",
        "model": "htdemucs_6s",
        "two_stems": None,
        # vocals 在引擎侧保持原样输出；run_separation 跑完 Demucs 后会把人声轨
        # 交给和声模块拆成 lead / harmony，并把 vocals 从最终 stems 里替换掉。
        "stems": ["vocals", "no_vocals", "drums", "bass", "guitar", "piano", "other"],
        "desc": "主唱 / 和声 / 贝斯 / 鼓 / 吉他 / 钢琴 / 其他（7 轨，最全）",
    },
    # 乐器提取：不走 Demucs，交给 instrument_worker.py 用 53 轨 BS-RoFormer 跑。
    # stems 为空 = 由用户勾选的目标乐器动态决定（targets）。
    "instruments": {
        "label": "乐器提取",
        "model": "mega53",
        "engine": "worker",
        "two_stems": None,
        "stems": [],
        "desc": "从混音里单独提取指定乐器（如电吉他 / 木吉他）",
    },
    # 全分离 Pro：与乐器提取同走 53 轨 worker，但一次性把所有乐器都拆出来，
    # 由后端自动剔除「这首歌里没有的乐器」，只把有内容的轨交给界面。
    # 模型本就一次推理输出全部通道，所以 58 个目标 = 一次推理，耗时与单轨乐器提取同量级。
    "pro": {
        "label": "全分离 Pro",
        "model": "mega53",
        "engine": "worker",
        "two_stems": None,
        "stems": [],
        "pro": True,
        "desc": "自动拆分全部乐器（民乐 / 独奏全覆盖），只保留有内容的音轨",
    },
}

# ---------------------------------------------------------------- 乐器提取目标
# ⚠️ 这张表是实测标定的，改之前先看 docs/2.0_乐器分离技术选型方案.md。
#
# 一个「界面选项」可能对应 1 个或多个模型通道 —— 模型通道是 53 类细粒度标签，
# 同一件乐器常被判到不同的类（比如《醒》的贝斯落在 bass、《普通朋友》落在
# double-bass），只取其中一个会拿到近乎静音的轨。别名通道相加即可解决。
#
# 已被实测否决、永远不要放回界面的通道：
#   strings  → 兜底通道。《醒》里 strings ≈ electric-guitar（corr 0.981），
#              《普通朋友》里 ≈ acoustic-guitar（corr 0.999），它复制了当前歌曲
#              能量最大的吉他轨，不是独立弦乐。
#   guitar / vocal → 同为「合集通道」，包含了已经单独列出的细粒度轨，相加会重复计数。
INSTRUMENT_TARGETS = [
    # (选项 key, 中文名, 界面分组, 对应的模型通道)
    # ⚠️ 模型是 MVSep Mega53，固定 53 种声音类别（西洋乐器/鼓件/人声），
    #    没有专门的民乐训练。下方「民乐近似」分组是从模型里最接近的已有音轨提取，
    #    质量非专门级，已在界面标注「(近似)」，切勿对外宣称「民乐精准分离」。
    # —— 吉他 / 节奏组 ——
    ("electric-guitar", "电吉他",     "吉他",     ["electric-guitar"]),
    ("acoustic-guitar", "木吉他",     "吉他",     ["acoustic-guitar"]),
    ("bass",            "贝斯",       "节奏组",   ["bass", "double-bass"]),
    ("drums",           "鼓组",       "节奏组",   ["drums"]),
    ("percussion",      "打击乐",     "节奏组",   ["percussion"]),
    # —— 键盘 ——
    ("piano",           "钢琴",       "键盘",     ["piano", "digital-piano"]),
    ("keys",            "键盘/合成器","键盘",     ["keys", "synth", "organ"]),
    ("harpsichord",     "羽管键琴",   "键盘",     ["harpsichord"]),
    # —— 人声 ——
    ("lead-vocal",      "主唱人声",   "人声",     ["lead-vocal"]),
    ("back-vocal",      "和声人声",   "人声",     ["back-vocal"]),
    # —— 弦乐 ——
    ("violin",          "小提琴",     "弦乐",     ["violin"]),
    ("cello",           "大提琴",     "弦乐",     ["cello"]),
    ("harp",            "竖琴",       "弦乐",     ["harp"]),
    ("viola",           "中提琴",     "弦乐",     ["viola"]),
    ("strings",         "弦乐群",     "弦乐",     ["strings"]),
    ("bowed_strings",   "弓弦群",     "弦乐",     ["bowed_strings"]),
    # —— 管乐 ——
    ("saxophone",       "萨克斯",     "管乐",     ["saxophone"]),
    ("trumpet",         "小号",       "管乐",     ["trumpet"]),
    ("trombone",        "长号",       "管乐",     ["trombone"]),
    ("flute",           "长笛",       "管乐",     ["flute"]),
    ("clarinet",        "单簧管",     "管乐",     ["clarinet"]),
    ("french-horn",     "圆号",       "管乐",     ["french-horn"]),
    ("oboe",            "双簧管",     "管乐",     ["oboe"]),
    ("tuba",            "大号",       "管乐",     ["tuba"]),
    ("bassoon",         "巴松管",     "管乐",     ["bassoon"]),
    ("brass",           "铜管群",     "管乐",     ["brass"]),
    ("woodwind",        "木管群",     "管乐",     ["woodwind"]),
    # —— 打击乐 / 鼓件 ——
    ("marimba",         "马林巴",     "打击乐",   ["marimba"]),
    ("timpani",         "定音鼓",     "打击乐",   ["timpani"]),
    ("glockenspiel",    "钟琴",       "打击乐",   ["glockenspiel"]),
    ("bells",           "钟/铃",      "打击乐",   ["bells"]),
    ("tambourine",      "铃鼓",       "打击乐",   ["tambourine"]),
    ("congas",          "康加鼓",     "打击乐",   ["congas"]),
    ("triangle",        "三角铁",     "打击乐",   ["triangle"]),
    ("wind-chimes",     "风铃",       "打击乐",   ["wind-chimes"]),
    ("kick",            "底鼓",       "鼓件",     ["kick"]),
    ("snare",           "军鼓",       "鼓件",     ["snare"]),
    ("hh",              "踩镲",       "鼓件",     ["hh"]),
    ("toms",            "桶鼓",       "鼓件",     ["toms"]),
    # —— 其他西洋乐器 ——
    ("accordion",       "手风琴",     "其他西洋", ["accordion"]),
    ("harmonica",       "口琴",       "其他西洋", ["harmonica"]),
    ("mandolin",        "曼陀林",     "其他西洋", ["mandolin"]),
    ("banjo",           "班卓琴",     "其他西洋", ["banjo"]),
    ("ukulele",         "尤克里里",   "其他西洋", ["ukulele"]),
    ("dobro",           "多布罗琴",   "其他西洋", ["dobro"]),
    ("sitar",           "西塔琴",     "其他西洋", ["sitar"]),
    # —— 民乐近似（模型无专门民乐训练，从最接近音轨提取，质量非专门级）——
    ("erhu",            "二胡(近似)",     "民乐近似", ["bowed_strings"]),
    ("matouqin",        "马头琴(近似)",   "民乐近似", ["bowed_strings"]),
    ("guzheng",         "古筝(近似)",     "民乐近似", ["harp"]),
    ("pipa",            "琵琶(近似)",     "民乐近似", ["harp"]),
    ("ruan",            "阮(近似)",       "民乐近似", ["harp"]),
    ("guqin",           "古琴(近似)",     "民乐近似", ["harp"]),
    ("yangqin",         "扬琴(近似)",     "民乐近似", ["marimba"]),
    ("dizi",            "笛子/箫(近似)",  "民乐近似", ["flute"]),
    ("suona",           "唢呐(近似)",     "民乐近似", ["oboe"]),
    ("sheng",           "笙(近似)",       "民乐近似", ["organ"]),
    ("hulusi",          "葫芦丝/巴乌(近似)","民乐近似",["flute"]),
    ("luogu",           "锣鼓(近似)",     "民乐近似", ["percussion"]),
]
INSTRUMENT_STEMS = [k for k, _z, _g, _c in INSTRUMENT_TARGETS]
INSTRUMENT_ZH = {k: z for k, z, _g, _c in INSTRUMENT_TARGETS}
# 选项 → 模型通道列表。拆出来给 worker 用（worker 不 import separate_core）。
INSTRUMENT_CHANNELS = {k: list(c) for k, _z, _g, c in INSTRUMENT_TARGETS}
# 一次最多选几条轨：既是显存约束（每条轨都要一份累积缓冲），也是计费保护
MAX_TARGETS = 6
# 输出轨 RMS 低于此值判为「这首里没有该乐器」，界面灰显提示而不是给一条静音轨
SILENT_RMS_DB = -60.0


def normalize_targets(raw, limit=MAX_TARGETS):
    """把目标乐器整理成合法轨列表。

    入参既可以是逗号分隔的字符串（HTTP 头 / "all" 关键字），也可以是列表（内部调用）。
    过滤非法名字、去重、保持提交顺序。
    - limit：截断上限（乐器提取默认 MAX_TARGETS=6）。传 None 表示不截断（全分离 Pro 用）。
    - 入参为字符串 "all"：展开为全部乐器（INSTRUMENT_STEMS），用于 Pro 模式。
    返回 (targets, error)；targets 为空时 error 非空。
    """
    if isinstance(raw, (list, tuple)):
        items = [str(t).strip() for t in raw]
    else:
        raw_s = str(raw or "").strip()
        if raw_s == "all":
            items = list(INSTRUMENT_STEMS)
        else:
            items = raw_s.split(",")
    items = [t.strip() for t in items if str(t).strip()]
    if not items:
        return [], "乐器提取模式需要至少选择 1 个目标乐器"
    out = []
    for t in items:
        if t in INSTRUMENT_STEMS and t not in out:
            out.append(t)
    if not out:
        return [], "目标乐器无效（支持：%s）" % "、".join(
            INSTRUMENT_ZH[k] for k in INSTRUMENT_STEMS[:8]) + " 等"
    return (out[:limit] if limit else out), ""

# 模型 → 输出音轨清单（兼容旧接口 / 内部复用）
MODELS = {
    "htdemucs": {
        "stems": ["vocals", "drums", "bass", "other"],
        "desc": "4-stem 标准：人声 / 鼓 / 贝斯 / 其他伴奏（最快最稳）",
    },
    "htdemucs_ft": {
        "stems": ["vocals", "drums", "bass", "other"],
        "desc": "4-stem 精修版：质量更高，显存临界 6GB",
    },
    "htdemucs_6s": {
        "stems": ["vocals", "drums", "bass", "guitar", "piano", "other"],
        "desc": "6-stem：在 4-stem 基础上 + 吉他 / 钢琴",
    },
}

# 旧模型名 → 模式（向后兼容）
MODEL_TO_MODE = {
    "htdemucs": "four",
    "htdemucs_ft": "four",
    "htdemucs_6s": "multi",
}

# 实时倍率估算（本机 RTX 3060 实测：base ≈11.6x，ft / 6s ≈5x）
SPEED = {
    "basic": {"cuda": 11.0, "cpu": 0.30},
    "four":  {"cuda": 5.0,  "cpu": 0.15},
    "multi": {"cuda": 4.5,  "cpu": 0.13},
    # 53 轨模型实测 RTF ≈ 1.76（RTX 3060 6GB，chunk 3s）→ 实时倍率 0.57，留点余量
    # CPU 下这个模型基本不可用，给个悲观值让用户看到真实量级
    "instruments": {"cuda": 0.55, "cpu": 0.03},
    # 全分离 Pro 同走 53 轨模型，也是一次推理输出全部目标，速度量级与 instruments 一致
    "pro": {"cuda": 0.55, "cpu": 0.03},
}

# htdemucs_ft / htdemucs_6s 的 Transformer 训练最大 segment≈7.8s，必须显式分块，否则报错
NEED_SEGMENT = ("htdemucs_ft", "htdemucs_6s")

# ---------------------------------------------------------------- 质量参数
# shifts  ：测试时增强次数（对音频做随机位移后多次推理再平均）。
#           demucs 默认 1 = 不做平均 → 分块拼接处易出现「沙沙 / 水声 / 毛刺」类杂音。
# overlap ：分块重叠比例，默认 0.25，提高可显著减轻拼接相位跳变。
# 代价：耗时 ≈ ×shifts（×overlap 带来的少量额外）。
DEFAULT_SHIFTS = 2
DEFAULT_OVERLAP = 0.5

# 质量档位：shifts 越大杂音越少，耗时按 cost 倍数增长
QUALITY = {
    "normal": {"label": "标准", "shifts": 1, "overlap": 0.25, "cost": 1.0,
               "desc": "最快，分块拼接处偶有轻微杂音"},
    "high":   {"label": "高质", "shifts": 2, "overlap": 0.50, "cost": 1.8,
               "desc": "推荐，杂音明显减少"},
    "max":    {"label": "极致", "shifts": 5, "overlap": 0.75, "cost": 4.0,
               "desc": "最干净，耗时约 4 倍"},
}
DEFAULT_QUALITY = "high"

# ---------------------------------------------------------------- 导出采样率
# 引擎（Demucs）原生输出 44100 Hz。若 DAW 工程是 48 kHz（视频 / 影视 / 播客工程
# 常见），导入 44.1k 文件时若不做转换会「快约 8.8% + 升约 1.5 个半音」。
# 选 48 kHz 时在**导出阶段用 ffmpeg 重采样**：只改采样率，不改速度、不改音高。
SR_CHOICES = (44100, 48000)
DEFAULT_SR = 44100
SR_LABELS = {44100: "44.1 kHz", 48000: "48 kHz"}


def normalize_sr(sr):
    """导出采样率校验：非法/缺省值一律回落到 44.1kHz。"""
    try:
        v = int(sr)
    except (TypeError, ValueError):
        return DEFAULT_SR
    return v if v in SR_CHOICES else DEFAULT_SR


def quality_params(q):
    """质量档位名 → (shifts, overlap)；未知档位回落到默认。"""
    cfg = QUALITY.get(q, QUALITY[DEFAULT_QUALITY])
    return cfg["shifts"], cfg["overlap"]


def speed_of_with_quality(mode, device, q="high"):
    """按质量档位折算后的实时倍率（用于预估耗时）。"""
    return speed_of(mode, device) / QUALITY.get(q, QUALITY[DEFAULT_QUALITY])["cost"]


def speed_of(mode, device):
    """取某模式在指定设备上的实时倍率（用于预估耗时）。"""
    if mode in SPEED:
        return SPEED[mode].get(device, 1.0)
    if mode in MODEL_TO_MODE:
        return speed_of(MODEL_TO_MODE[mode], device)
    return 1.0


def silent_stems(stems, thr_db=None):
    """挑出「这首里压根没有该乐器」的轨。

    乐器提取是按用户勾选硬跑的：歌里没有萨克斯，模型照样会吐一条近乎全零的 WAV。
    不做检测，用户看到的就是一条播放没声音的轨 → 判定软件坏了。这里把这类轨标出来，
    交给界面灰显并说明原因。
    """
    thr = SILENT_RMS_DB if thr_db is None else thr_db
    out = []
    try:
        import numpy as np
        import soundfile as sf
    except Exception:
        return out
    for st in stems:
        p = st.get("path")
        if not p or not os.path.isfile(p):
            continue
        try:
            x, _sr = sf.read(p, dtype="float32", always_2d=True)
        except Exception:
            continue
        if x.size == 0:
            out.append(st.get("name"))
            continue
        rms = float(np.sqrt(float((x.astype("float64") ** 2).mean())))
        if 20.0 * np.log10(rms + 1e-12) < thr:
            out.append(st.get("name"))
    return out


def get_audio_duration(path):
    """读取音频时长（秒），失败返回 None。"""
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.frames / float(info.samplerate)
    except Exception:
        return None


def compute_peaks(path, buckets=900):
    """读取音频并降采样为波形峰值序列（0~1），供前端绘制波形。

    做法：多声道取均值 → 逐桶取绝对峰值 → 按全局最大值归一化。
    归一化后每轨都能看清细节（弱轨不会被压成一条线）。
    返回 {"duration", "sr", "peaks":[...]}；失败返回 {"error": ...}。
    """
    try:
        import numpy as np
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        if x.size == 0:
            return {"duration": 0.0, "sr": sr, "peaks": [], "error": "空音频"}
        mono = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
        n = int(mono.shape[0])
        step = max(1, n // int(buckets))
        m = (n // step) * step
        y = np.abs(mono[:m]).reshape(-1, step)
        pk = y.max(axis=1).astype("float32")
        mx = float(pk.max()) if pk.size else 0.0
        # 整轨近乎静音（-46dBFS 以下）时不高倍放大噪声
        if mx < 0.005:
            pk = pk * 0.0
            mx = 1.0
        pk = pk / mx
        return {
            "duration": n / float(sr),
            "sr": sr,
            "peaks": [round(float(v), 4) for v in pk],
        }
    except Exception as e:
        return {"duration": 0.0, "sr": 0, "peaks": [], "error": "波形读取失败: %s" % e}


def ensure_ffmpeg_on_path():
    """把 ffmpeg 所在目录补进 PATH，让「只会查 PATH」的三方库也能用上自带 ffmpeg。

    audio-separator 直接跑 subprocess(["ffmpeg","-version"])，不走我们的 find_ffmpeg()。
    手动命令行启动服务时 PATH 里没有包内 bin\\ → 它抛 FileNotFoundError，
    和声分离就静默降级成「只有一条普通人声轨」，用户完全不知道少了东西。
    在每个会用到这类三方库的地方调用一次即可。
    """
    import shutil
    if shutil.which("ffmpeg"):
        return True
    ff = find_ffmpeg()
    if not ff:
        return False
    d = os.path.dirname(ff)
    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    return shutil.which("ffmpeg") is not None


def instrument_model_status():
    """查询 53 轨模型是否就绪（供界面决定要不要显示下载引导）。

    返回 dict：ready / installed / size / need_mb / downloading / progress …
    模型缺失时不抛异常，由调用方决定提示方式。
    """
    try:
        import mega53_download as _md
        return _md.status()
    except Exception as e:
        return {"ready": False, "installed": False, "path": "", "size": 0,
                "expected": 0, "need_mb": 0, "check_ok": False,
                "check_msg": "模型检测模块不可用：%s" % e,
                "downloading": False,
                "progress": {"active": False, "done": False, "ok": False,
                             "error": "", "total": 0, "got": 0,
                             "speed": 0.0, "stage": "", "percent": 0.0}}


def instrument_model_download():
    """触发后台下载（幂等）。返回与 instrument_model_status() 同构的 dict。"""
    try:
        import mega53_download as _md
        return _md.start_download()
    except Exception as e:
        return {"ready": False, "installed": False, "error": "无法启动下载：%s" % e,
                "downloading": False,
                "progress": {"active": False, "done": True, "ok": False,
                             "error": str(e), "total": 0, "got": 0,
                             "speed": 0.0, "stage": "error", "percent": 0.0}}


def find_ffmpeg():
    """定位 ffmpeg 可执行文件，找不到返回 None。

    优先用**包内自带**的 ffmpeg（发布包会把 ffmpeg 放在 bin\\ 下，
    Windows 为 ffmpeg.exe、mac/linux 为 ffmpeg），这样用户无需自行安装即可导出 MP3；
    其次才查系统 PATH 与常见安装位置（Windows / Homebrew / 系统目录）。
    """
    import shutil
    for cand in (
        os.path.join(HERE, "bin", "ffmpeg" + EXE),
        os.path.join(os.path.dirname(HERE), "bin", "ffmpeg" + EXE),
    ):
        if os.path.isfile(cand):
            return cand
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    if IS_WIN:
        for cand in (
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"D:\ffmpeg\bin\ffmpeg.exe",
        ):
            if os.path.isfile(cand):
                return cand
    else:
        # macOS（Homebrew Apple Silicon / Intel）与 Linux 常见位置
        for cand in (
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/usr/bin/ffmpeg",
        ):
            if os.path.isfile(cand):
                return cand
    return None


def _wav_to_mp3(ffmpeg, wav_path, bitrate="320k", sr=DEFAULT_SR):
    """将单个 wav 转成同目录 mp3（320k），成功后删除 wav。

    返回 (mp3_path, error)。error 为空表示成功。
    MP3 格式专利已过期，ffmpeg 以外部进程调用，不污染本工具许可。
    """
    mp3_path = os.path.splitext(wav_path)[0] + ".mp3"
    # -ar 用目标采样率：44.1k 直接编码；48k 由 ffmpeg 重采样后编码（不改速度/音高）
    cmd = [ffmpeg, "-y", "-i", wav_path,
           "-b:a", bitrate, "-ar", str(int(sr)), "-vn", mp3_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, (r.stderr or r.stdout or "")[-800:]
    if not os.path.isfile(mp3_path):
        return None, "未生成 mp3 文件"
    try:
        os.remove(wav_path)
    except OSError:
        pass
    return mp3_path, None


def _resample_inplace(ffmpeg, path, sr, fmt):
    """把音频文件重采样到目标采样率（原地替换）。

    用 ffmpeg 做带限插值重采样 —— 只改采样率，**不改速度、不改音高**。
    返回 (ok, error)。
    """
    ext = os.path.splitext(path)[1].lower()
    tmp = path + ".rs.tmp" + ext
    cmd = [ffmpeg, "-y", "-i", path, "-ar", str(int(sr)),
           "-c:a", "flac" if ext == ".flac" else "pcm_s16le", tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False, (r.stderr or r.stdout or "")[-600:]
    os.replace(tmp, path)
    return True, None


def convert_audio(src, dst, fmt, sr=DEFAULT_SR):
    """把任意音频转成指定格式 + 采样率（导出阶段按需转换，供下载/打包调用）。

    src 源文件（通常为分离产出的母版 wav）；dst 目标文件（含扩展名）；
    fmt: wav / flac / mp3；sr: 目标采样率（44100 / 48000）。
    只改采样率不改速度/音高（-ar 带限插值）。返回 (ok, error)。
    """
    if fmt not in ("wav", "flac", "mp3"):
        return False, "不支持的导出格式: %s" % fmt
    ff = find_ffmpeg()
    if not ff:
        return False, "未找到 ffmpeg，无法转换格式"
    cmd = [ff, "-y", "-i", src, "-ar", str(int(sr))]
    if fmt == "flac":
        cmd += ["-c:a", "flac"]
    elif fmt == "mp3":
        cmd += ["-b:a", "320k"]
    else:
        cmd += ["-c:a", "pcm_s16le"]
    cmd += ["-vn", dst]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(dst):
        try:
            os.remove(dst)
        except OSError:
            pass
        return False, (r.stderr or r.stdout or "")[-600:]
    return True, None


def run_harmony(vocals_wav, out_dir, device="cuda", fmt="wav"):
    """惰性包装 harmony_separate.run_harmony。

    导入失败（如未安装和声依赖）时返回 (False, 原因, {})，让 multi 模式静默降级为
    只输出原始人声轨，而不是让整张订单失败。
    """
    if not os.path.isfile(HARMONY_MODULE):
        return False, "未找到和声模块: %s" % HARMONY_MODULE, {}
    try:
        # audio-separator 只会 subprocess(["ffmpeg"]) 查 PATH，这里先把自带 ffmpeg 补进去，
        # 否则手动命令行启动服务时和声会静默消失（只留下一条普通人声轨）。
        ensure_ffmpeg_on_path()
        from harmony_separate import run_harmony as _impl
    except Exception as e:
        return False, "和声模块不可用（%s）" % e, {}
    return _impl(vocals_wav, out_dir, device, fmt=fmt)


def _run_via_cli(input_path, output_dir, demucs_model, cfg, device, fmt, shifts, overlap):
    """兜底路径：调用 demucs.exe 子进程（与原行为一致）。"""
    cmd = [DEMUCS, "-n", demucs_model, "-d", device, "-o", output_dir]
    if demucs_model in NEED_SEGMENT:
        cmd += ["--segment", "7"]
    if cfg.get("two_stems"):
        cmd += ["--two-stems", cfg["two_stems"]]
        # 伴奏取「原曲 − 人声」而非「三轨重建相加」，消除音准漂移与电平压缩
        cmd += ["--other-method", cfg.get("other_method", "minus")]
    # 抗杂音：多次位移平均 + 加大分块重叠
    cmd += ["--shifts", str(int(shifts)), "--overlap", str(float(overlap))]
    if fmt == "flac":
        cmd += ["--flac"]
    cmd += [input_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        if err:
            return False, "分离引擎异常退出(code %d)：\n%s" % (r.returncode, err[-2000:])
        # 引擎连一行错误信息都没吐就退出 —— 旧版本在这里只丢一句
        # 「分离进程异常退出(code 1)」，用户完全无从下手。改成可操作的指引。
        return False, (
            "分离引擎异常退出(code %d)，未返回错误信息。常见原因：\n"
            "  · 显卡驱动 / CUDA 运行环境未就绪（重启软件或电脑后重试）\n"
            "  · 运行环境文件不完整（_runtime 与 bin 目录请勿删改、勿单独拷贝 exe）\n"
            "  · 音频文件损坏或格式不受支持\n"
            "建议：改用 WAV 格式重试；若仍失败，重启软件后再试一次。"
            % r.returncode)
    return True, None


def _run_via_api(input_path, final_dir, demucs_model, cfg, device, fmt, shifts, overlap, with_no_vocals):
    """默认路径：用 venv 的 torch / demucs 跑 Python API，统一增益保存 + 生成 no_vocals 伴奏轨。"""
    sources = [s for s in cfg["stems"] if s != "no_vocals"]
    cmd = [VENV_PY, API_WORKER, "--input", input_path, "--outdir", final_dir,
           "--model", demucs_model, "--device", device, "--shifts", str(int(shifts)),
           "--overlap", str(float(overlap)), "--sources", ",".join(sources),
           "--minus-vocals", "1" if with_no_vocals else "0", "--fmt", fmt]
    if demucs_model in NEED_SEGMENT:
        cmd += ["--segment", "7"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return False, "API worker 异常退出(code %d): %s" % (r.returncode, (r.stderr or "").strip()[-1500:])
    for line in reversed(r.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                j = json.loads(line)
            except Exception:
                continue
            if not j.get("ok"):
                return False, j.get("error", "API worker 返回失败")
            return True, None
    return False, "API worker 未返回有效 JSON"


# 乐器提取是显存大户（单进程峰值 >5GB），同一实例内绝不让两条乐器任务并发抢显存
# （否则第二条必 OOM）；用户双击只起一个实例，进程内锁已足够。
INSTR_LOCK = threading.Lock()


def _run_instruments(input_path, outdir, targets, device="cuda", fmt="wav",
                     overlapped="", progress_cb=None):
    """乐器提取：交给 instrument_worker.py 子进程跑 53 轨 BS-RoFormer。

    为什么必须走子进程：模型常驻 2.7GB 显存、单帧推理峰值超 5GB，跑在
    服务进程里一旦 OOM 会带崩整张订单；子进程结束即彻底释放。

    progress_cb : 可选回调 fn(percent: float)，worker 每秒左右吐一次真实进度。
    返回 (ok: bool, error: str, stems: list)
    """
    if not os.path.isfile(INSTRUMENT_MODULE):
        return False, "未找到乐器分离模块: %s" % INSTRUMENT_MODULE, []
    if not os.path.isfile(VENV_PY):
        return False, "未找到运行环境 Python: %s" % VENV_PY, []

    ff = find_ffmpeg()
    if not ff:
        return False, "未找到 ffmpeg，无法解码音频进行乐器提取", []

    # 模型本体不在时，给出可操作的提示，而不是让子进程抛出难以理解的英文异常
    st = instrument_model_status()
    if not st.get("ready"):
        if st.get("downloading"):
            return False, "分离模型正在下载中，请等下载完成后再试", []
        need = st.get("need_mb") or 0
        tip = "乐器提取需要额外的识别模型（约 %d MB），尚未安装。请先在界面上点击安装。" % need
        if st.get("check_msg"):
            tip = "分离模型文件异常：%s" % st["check_msg"]
        return False, tip, []

    # 权重实际所在优先（安装到 Program Files 时模型会落在用户目录，见
    # mega53_download._user_model_dir）；环境变量次之。
    env_dir = os.environ.get("AMU_MEGA53_DIR", "").strip()
    model_dir = env_dir if (env_dir and os.path.isdir(env_dir)) else ""
    if not model_dir:
        found = st.get("path") or ""
        if found and os.path.isfile(found):
            model_dir = os.path.dirname(found)

    # ── 关键修复：worker 子进程绝对只接触 ASCII 路径 ──
    # 历史上「模型运行完成但未产出任何音轨」的根因：含中文的 outdir / input 经
    # subprocess 传给 worker 子进程，在 Windows / launcher 启动下出现跨进程编码
    # 错乱 —— worker 把文件写到乱码目录，父进程按正确中文路径去找 → 找不到。
    # 根治办法：把输入与输出重定向到系统临时目录（纯 ASCII），跑完再把成品移回
    # uploads 目录；父进程用「按目标名 glob 实文件」兜底收集，彻底不依赖 worker
    # 回报的路径字符串。
    import tempfile
    import shutil

    in_ext = os.path.splitext(input_path)[1].lower() or ".wav"
    work = tempfile.mkdtemp(prefix="amu_instr_")
    inp_work = os.path.join(work, "input" + in_ext)
    out_work = os.path.join(work, "out")
    try:
        os.makedirs(out_work, exist_ok=True)
    except Exception:
        pass
    try:
        shutil.copyfile(input_path, inp_work)
    except Exception as e:
        shutil.rmtree(work, ignore_errors=True)
        return False, "无法准备输入音频（复制失败）：%s" % e, []

    cmd = [VENV_PY, INSTRUMENT_MODULE,
           "--input", inp_work, "--outdir", out_work,
           "--targets", ",".join(targets), "--device", device,
           "--fmt", fmt, "--ffmpeg", ff]
    if model_dir:
        cmd += ["--model-dir", model_dir]

    INSTR_LOCK.acquire()
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, errors="replace",
                             encoding="utf-8", bufsize=1)
    except Exception as e:
        INSTR_LOCK.release()
        shutil.rmtree(work, ignore_errors=True)
        return False, "乐器分离进程启动失败：%s" % e, []

    err_lines = []  # 收集 worker 的 stderr（进度 JSON 之外的诊断信息也留着）

    def _pump_err():
        """边跑边解析 worker 打到 stderr 的进度 JSON，同时留存诊断文本。"""
        try:
            for line in p.stderr:
                line = line.strip()
                if not line:
                    continue
                err_lines.append(line)
                if not line.startswith("{"):
                    print("[instrument_worker] " + line, file=sys.stderr)
                    continue
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                v = j.get("progress")
                if v is not None and progress_cb:
                    try:
                        try:
                            progress_cb(float(v), j.get("stage") or "")
                        except TypeError:      # 兼容只收一个参数的旧回调
                            progress_cb(float(v))
                    except Exception:
                        pass
        except (ValueError, OSError):
            pass

    t = threading.Thread(target=_pump_err, daemon=True)
    t.start()
    out_lines = []
    for line in p.stdout:
        out_lines.append(line.strip())
    p.wait()
    t.join(timeout=5)
    INSTR_LOCK.release()

    # 解析 worker 末行 JSON 作为结果
    payload = None
    for line in reversed(out_lines):
        if line.startswith("{"):
            try:
                payload = json.loads(line)
                break
            except Exception:
                continue

    # 兜底收集：无论 worker 回报的路径字符串是否可靠，都按「目标名」在 out_work 里
    # 找真实存在的成品文件。worker 写的是 ASCII 临时目录，名称 = 目标 key（纯 ASCII）。
    def _work_file(name):
        for e in (".wav", ".flac"):
            c = os.path.join(out_work, name + e)
            if os.path.isfile(c):
                return c
        return None

    # 把成品搬到最终 outdir（父进程自己写中文路径没问题；问题只在子进程），并校验存在
    stems = []
    try:
        os.makedirs(outdir, exist_ok=True)
    except Exception:
        pass
    for name in targets:
        src = _work_file(name)
        if not src:
            continue
        dest = os.path.join(outdir, name + os.path.splitext(src)[1])
        moved = False
        try:
            shutil.move(src, dest)
            moved = True
        except Exception:
            try:
                shutil.copyfile(src, dest)
                moved = True
            except Exception:
                dest = src   # 移动/复制都失败 → 先用临时目录里的文件兜底（不删 work）
        if os.path.isfile(dest):
            stems.append({"name": name, "path": dest,
                          "size": os.path.getsize(dest)})
        if not moved and dest == src:
            moved = True   # 已用 src 兜底，允许清理 work

    # 只有在全部目标都成功移出后才清理临时目录；否则保留现场便于排查
    if all(_work_file(n) is None for n in targets):
        shutil.rmtree(work, ignore_errors=True)
    else:
        # 仍有未移出的文件 → 保留 work，写一条诊断
        try:
            with open(r"F:/_amu_test/instr_dbg.log", "a", encoding="utf-8") as _d:
                _d.write("\n[%s] 残留未移出，保留临时目录 %s\n"
                         % (time.strftime("%H:%M:%S"), work))
        except Exception:
            pass

    if not stems:
        # 给出可操作的报错：优先用 worker 自己的错误信息，其次贴出 stderr 末尾
        if payload is not None and not payload.get("ok"):
            return False, str(payload.get("error") or "乐器分离失败"), []
        diag = "\n".join(l for l in err_lines if l.strip())[-800:]
        head = "乐器提取未产出任何音轨"
        if diag:
            head += "（诊断：%s）" % diag[-600:]
        else:
            head += "（worker 无输出，可能是模型加载或显存不足）"
        return False, head, []

    return True, "", stems


def run_separation(input_path, output_dir, mode="four", device="cuda", fmt="wav", model=None,
                   shifts=None, overlap=None, with_no_vocals=True, sr=DEFAULT_SR,
                   targets=None, progress_cb=None):
    """执行分离，返回结构化结果字典。

    参数
        mode    : basic / four / multi（也可用旧参数 model= 指定 Demucs 模型名）
        device  : cuda / cpu
        fmt     : wav / flac
        shifts  : 测试时增强次数，None = 用 DEFAULT_SHIFTS（越大杂音越少，耗时线性增加）
        overlap : 分块重叠比例，None = 用 DEFAULT_OVERLAP（0.25~0.75）
        sr      : 导出采样率（44100 / 48000），只影响导出文件，不改速度与音高

    返回字段：
        ok        bool   是否成功
        error     str    失败原因
        outdir    str    输出目录
        stems     list   [{name, path, size}]
        duration  float  原曲时长(秒)
        elapsed   float  分离耗时(秒)
        mode      str    实际使用的模式
    """
    # Mac 无 CUDA：强制回退 CPU（历史 Mac 版适配，免费版曾丢失）
    if sys.platform == "darwin" and device == "cuda":
        device = "cpu"

    result = {"ok": False, "error": "", "outdir": "", "stems": [],
              "duration": 0.0, "elapsed": 0.0, "mode": ""}

    # 兼容旧调用：run_separation(..., model="htdemucs_ft")，也接受直接传模式名
    if model is not None:
        mode = model if model in MODES else MODEL_TO_MODE.get(model, "four")

    if not os.path.isfile(DEMUCS):
        result["error"] = "未找到 demucs 运行环境，请先搭建 .venv312"
        return result
    if mode not in MODES:
        result["error"] = "未知分离模式: %s" % mode
        return result
    if fmt not in ("wav", "flac", "mp3"):
        result["error"] = "不支持的输出格式: %s（仅 wav / flac / mp3）" % fmt
        return result

    cfg = MODES[mode]
    demucs_model = cfg["model"]
    result["mode"] = mode

    base = os.path.splitext(os.path.basename(input_path))[0]
    outdir = os.path.join(output_dir, demucs_model, base)
    os.makedirs(outdir, exist_ok=True)

    # 乐器提取：完全不经过 Demucs，走独立的 53 轨 worker
    if cfg.get("engine") == "worker":
        # 全分离 Pro 不截断（一次拆全部乐器）；其余按 MAX_TARGETS 截断
        tgts, terr = normalize_targets(
            targets, limit=None if mode == "pro" else MAX_TARGETS)
        if terr:
            result["error"] = terr
            return result
        t0 = time.time()
        ok_w, err_w, stems_w = _run_instruments(
            input_path, outdir, tgts, device, fmt, progress_cb=progress_cb)
        result["elapsed"] = time.time() - t0
        result["engine"] = "instrument_worker"
        if not ok_w:
            result["error"] = err_w
            return result
        if not stems_w:
            result["error"] = "模型运行完成但未产出任何音轨"
            return result
        result["targets"] = tgts
        target_sr = normalize_sr(sr)
        stems = stems_w

        # MP3：wav 母版 → ffmpeg 转 320k；WAV/FLAC：选 48k 时原地重采样
        if fmt == "mp3":
            ff = find_ffmpeg()
            if not ff:
                result["error"] = ("未找到 ffmpeg，无法输出 MP3。请安装 ffmpeg 后重试，"
                                   "或改用 WAV / FLAC 格式。")
                return result
            conv = []
            for st in stems:
                mp3_path, cerr = _wav_to_mp3(ff, st["path"], sr=target_sr)
                if cerr:
                    result["error"] = "MP3 转换失败：" + cerr
                    return result
                conv.append({"name": st["name"], "path": mp3_path,
                             "size": os.path.getsize(mp3_path)})
            stems = conv
        elif target_sr != DEFAULT_SR:
            ff = find_ffmpeg()
            if not ff:
                result["error"] = ("未找到 ffmpeg，无法导出 %s 音频。请改用 44.1kHz，"
                                   "或安装 ffmpeg 后重试。"
                                   % SR_LABELS.get(target_sr, target_sr))
                return result
            for st in stems:
                ok_rs, rs_err = _resample_inplace(ff, st["path"], target_sr, fmt)
                if not ok_rs:
                    result["error"] = "重采样到 %s 失败：%s" % (
                        SR_LABELS.get(target_sr, target_sr), rs_err)
                    return result
                st["size"] = os.path.getsize(st["path"])

        if not stems:
            result["error"] = "模型运行完成但未产出任何音轨"
            return result
        result["ok"] = True
        result["outdir"] = outdir
        result["stems"] = stems
        result["sr"] = target_sr
        # 这首歌里其实没有某个勾选乐器 → 标出来，界面灰显说明，别给一条「坏了的静音轨」
        result["silent"] = silent_stems(stems)
        # 全分离 Pro：只把「这首歌里确实有」的乐器交给界面，静音轨直接不列出
        # （模型对没出现的乐器也会吐一条近乎全零的轨，留着只会让用户以为软件坏了）
        if cfg.get("pro"):
            silent_names = set(result["silent"])
            if silent_names:
                stems = [s for s in stems if s["name"] not in silent_names]
                result["stems"] = stems
                result["silent"] = []          # 已剔除，界面无需灰显
                result["dropped_silent"] = len(silent_names)
            # 写标记：服务重启后 _recover_jobs 从磁盘重建任务表时，被剔除的静音轨
            # 又会冒出来。让恢复逻辑读这个标记，保持与首次结果完全一致。
            try:
                with open(os.path.join(output_dir, ".amu_pro"), "w",
                          encoding="utf-8") as _f:
                    json.dump({"mode": "pro", "dropped": sorted(silent_names)},
                              _f, ensure_ascii=False)
            except Exception:
                pass
        result["duration"] = get_audio_duration(input_path) or 0.0
        return result

    shifts_v = int(shifts if shifts is not None else DEFAULT_SHIFTS)
    overlap_v = float(overlap if overlap is not None else DEFAULT_OVERLAP)

    # 优先 Python API（统一增益 + 伴奏轨）；环境缺失或失败时回退 demucs.exe（CLI）
    t0 = time.time()
    used_api = False
    api_err = ""
    if os.path.isfile(VENV_PY) and os.path.isfile(API_WORKER):
        used_api, api_err = _run_via_api(input_path, outdir, demucs_model, cfg,
                                         device, fmt, shifts_v, overlap_v,
                                         with_no_vocals)
        if not used_api:
            print("[separate_core] API 路径失败，回退 CLI：", api_err, file=sys.stderr)
    if not used_api:
        ok, err = _run_via_cli(input_path, output_dir, demucs_model, cfg,
                               device, fmt, shifts_v, overlap_v)
        if not ok:
            # 两条路径都失败时，把 API 的真实原因一并交出去。旧版本把 API 错误
            # 只打到服务端 stderr（用户看不到），界面只剩干巴巴的「code 1」。
            parts = [err]
            if api_err:
                parts.append("\n（主引擎先前也失败：%s）" % api_err[-800:])
            parts.append("\n若持续失败，请重启软件；仍不行可换「标准」质量档再试。")
            result["error"] = "".join(parts)
            return result
    result["elapsed"] = time.time() - t0
    result["engine"] = "api" if used_api else "cli"

    if not os.path.isdir(outdir):
        result["error"] = "未生成输出目录: " + outdir
        return result

    ext = "flac" if fmt == "flac" else "wav"
    alias = cfg.get("alias", {})
    stems = []
    for s in cfg["stems"]:
        p = os.path.join(outdir, s + "." + ext)
        # other_method=minus 时引擎输出名为 minus_vocals，改回 no_vocals 以保持对外名称稳定
        if not os.path.isfile(p) and s in alias:
            src = os.path.join(outdir, alias[s] + "." + ext)
            if os.path.isfile(src):
                os.replace(src, p)
        if os.path.isfile(p):
            stems.append({"name": s, "path": p, "size": os.path.getsize(p)})

    # 多轨模式：把 Demucs 的人声轨再拆成「主唱 + 和声」（必须在 MP3 转换之前做完，
    # 否则新轨进不了下面的转换循环，mp3 输出会漏轨）。
    # 失败静默降级：保留原 vocals 轨、任务仍算成功（和声是可选增强，不能拖垮整单）。
    if mode == "multi":
        t_h = time.time()
        vocals_p = os.path.join(outdir, "vocals." + ext)
        ok_h, err_h, info_h = run_harmony(vocals_p, outdir, device, fmt=ext)
        if ok_h:
            try:
                os.remove(vocals_p)
            except OSError:
                pass
            new_stems = []
            for st in stems:
                if st["name"] == "vocals":
                    new_stems.append({"name": "lead", "path": info_h["lead"],
                                      "size": os.path.getsize(info_h["lead"])})
                    new_stems.append({"name": "harmony", "path": info_h["harmony"],
                                      "size": os.path.getsize(info_h["harmony"])})
                else:
                    new_stems.append(st)
            stems = new_stems
            result["elapsed"] += time.time() - t_h
            print("[separate_core] 和声分离完成，耗时 %.1f 秒" % (time.time() - t_h), file=sys.stderr)
        else:
            print("[separate_core] 和声分离跳过（保留人声轨）：" + str(err_h), file=sys.stderr)

    # 导出采样率：默认 44.1kHz（引擎原生）；选 48kHz 时在导出阶段重采样
    target_sr = normalize_sr(sr)

    # MP3 后置转换：Demucs 产出 wav 后统一用 ffmpeg 转 320k
    if fmt == "mp3":
        ff = find_ffmpeg()
        if not ff:
            result["error"] = "未找到 ffmpeg，无法输出 MP3。请安装 ffmpeg 后重试，或改用 WAV / FLAC 格式。"
            return result
        converted = []
        for st in stems:
            mp3_path, cerr = _wav_to_mp3(ff, st["path"], sr=target_sr)
            if cerr:
                result["error"] = "MP3 转换失败：" + cerr
                return result
            converted.append({"name": st["name"], "path": mp3_path,
                              "size": os.path.getsize(mp3_path)})
        stems = converted
    elif target_sr != DEFAULT_SR:
        # WAV / FLAC：原地重采样到目标采样率（只改采样率，不改速度/音高）
        ff = find_ffmpeg()
        if not ff:
            result["error"] = ("未找到 ffmpeg，无法导出 %s 音频。请改用 44.1kHz，"
                               "或安装 ffmpeg 后重试。" % SR_LABELS.get(target_sr, target_sr))
            return result
        for st in stems:
            ok_rs, rs_err = _resample_inplace(ff, st["path"], target_sr, fmt)
            if not ok_rs:
                result["error"] = "重采样到 %s 失败：%s" % (
                    SR_LABELS.get(target_sr, target_sr), rs_err)
                return result
            st["size"] = os.path.getsize(st["path"])
        print("[separate_core] 已重采样到 %s" % SR_LABELS.get(target_sr, target_sr),
              file=sys.stderr)

    if not stems:
        result["error"] = "模型运行完成但未产出任何音轨"
        return result

    result["ok"] = True
    result["outdir"] = outdir
    result["stems"] = stems
    result["sr"] = target_sr
    # Demucs 固定输出各轨（用户没点名），歌里没有该乐器时模型照样吐一条近乎全零的
    # WAV。不标出来，用户会看到一条播放没声音的轨并判定「音质坏了 / 软件坏了」。
    # 与乐器提取保持一致：只灰显说明原因，不删除、不做任何电平归一化。
    result["silent"] = silent_stems(stems)
    result["duration"] = get_audio_duration(input_path) or 0.0
    return result
