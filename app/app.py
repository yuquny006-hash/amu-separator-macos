#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
本地 Web 服务（Amu Music · 音频分离软件 完整版）
=========================================
纯 Python 标准库实现，未引入任何第三方 Web 框架 —— 零额外许可证风险，避开 GPL 壳。
前端 HTML/CSS/JS 全部为自有代码，不传染任何后端许可证。

运行：
    .venv312\Scripts\python.exe app.py [端口]
默认端口 8777，浏览器打开 http://127.0.0.1:8777

接口：
    POST /api/separate            提交音频，返回 job id
    GET  /api/status?job=         查询进度 / 结果
    GET  /api/source?job=         原始上传音频（支持 Range，前端试听波形）
    GET  /api/stream?job=&stem=   某条音轨的音频流（支持 Range，可 seek）
    GET  /api/download?job=&stem=&dl=1&fmt=&sr=   下载单条音轨（fmt/sr = 导出格式与采样率，下载时选定，按需转码）
    GET  /api/export?job=&fmt=&sr=          打包全部音轨为 zip 下载（同样按需转码）
    GET  /api/peaks?job=&all=1    各轨波形峰值数据（JSON，用于画波形）

说明：
- 仅监听 127.0.0.1（本机），不上外网，符合「离线本地工具」的版权低风险定位。
- 完全免费（2026-11-14 用户定调）：FREE_MODE = True 时所有功能无需登录、
  不消耗积分、不限次数、不联网验活，任何人拿到就能用。
  登录门与按轨计费代码完整保留但不启用 —— 日后如需恢复商业化，
  只需把 FREE_MODE 改回 False，无需重写；amu_selfuse 外部开关仅在那时才有意义。
- 分离在后台线程执行；前端轮询 /api/status 获取进度。
- 五种模式：basic / four / multi（Demucs，MIT）、instruments / pro
  （Mega53 乐器识别，一次推理输出全部目标轨）。
- 分离恒产 44.1kHz WAV 母版，格式与采样率只在下载导出时按需转换。
"""
import os
import sys
import json
import uuid
import time
import zipfile
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import separate_core
import amu_login_gate
import amu_selfuse

# ---- 离线支持：模型权重指向包内 _hfcache（必须在 import torch / demucs 之前设置）
_HF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_hfcache")
if os.path.isdir(_HF):
    os.environ["HF_HOME"] = _HF
    os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")
UPLOAD_DIR = os.path.join(os.environ.get("AMU_DATA_DIR", HERE), "uploads")  # Mac: .app Resources 只读，数据落到 ~/Library/Application Support
os.makedirs(UPLOAD_DIR, exist_ok=True)

JOBS = {}
LOCK = threading.Lock()
EXPORT_LOCK = threading.Lock()   # 导出转码互斥：同一文件不并发跑两份 ffmpeg

CHUNK = 262144  # 流式发送块大小（256KB）

# ── 完全免费开关（2026-11-14 用户定调：取消积分版，不设任何门槛）──
# FREE_MODE = True  →  所有门禁恒放行：免登录、免积分、不限次数、不联网验活。
#                      登录门与计费代码保留但不执行，随时可改回 False 恢复商业化。
# 下面这个外部加密开关（amu_selfuse）是早期「作者自用免积分」方案，
# 在 FREE_MODE=True 下已完全失效：开着关着都一样免费。
# 保留 import 只是为了让历史代码可逆；校验异常也绝不会让软件起不来。
FREE_MODE = True
INTERNAL = FREE_MODE or amu_selfuse.is_enabled()

# ── 登录门与计费状态（FREE_MODE 下不参与准入，仅为可逆保留）──────────
# 令牌存 %APPDATA%/AmuTools（DPAPI 机器绑定加密），与官网会员账号同体系。
# 断网宽限：最近一次验活成功后 72 小时内断网仍可用（缓存验活快照）。
GATE = amu_login_gate.AmuLoginGate()
GATE_ST = {
    "ok": INTERNAL,
    "user": ({"nickname": "完全免费版", "email": "", "points": "∞"}
             if INTERNAL else None),
    "reason": "",
}
GATE_CACHE = os.path.join(
    os.environ.get("APPDATA", os.path.expanduser("~")),
    "AmuTools", ".amu_gate_cache.json")
GATE_GRACE = 72 * 3600
GATE_REVERIFY_SECS = 3600  # 运行期每小时重验一次

# ── 按轨计费常量：FREE_MODE=True 下完全不执行，保留仅为日后可逆 ──────
COST_PER_TRACK = 5           # 每轨消耗积分（改这一个数即可全网调价）
# 模式 → 本次分离计费的轨数（与界面模式卡片显示的轨数保持一致）
# "instruments" 按用户实际勾选的目标乐器数计费，这里置 None 由 call site 传具体值
# "pro" 全分离为统一定价（见 PRO_COST），不按轨数计，置 None
MODE_TRACKS = {"basic": 2, "four": 4, "multi": 7, "instruments": None, "pro": None}
# 全分离 Pro 固定积分：一次把全部乐器拆出来，统一价，不按轨数乘（53 轨 × 5 = 265 太贵）
PRO_COST = 50               # 改这个数即可调整 Pro 模式定价
SEPARATE_TASK = 'separator'  # 记账任务类型（积分流水里显示用）


def tracks_of(mode, ntargets=None):
    """该模式本次计入的轨数。乐器提取模式 = 用户勾选的目标乐器数量。

    全分离 Pro 为统一定价，不按轨数计，这里返回 0（真实积分见 cost_of）。
    """
    if mode == "instruments":
        return max(1, int(ntargets or 0))
    if mode == "pro":
        return 0
    return MODE_TRACKS.get(mode, MODE_TRACKS["basic"])


def cost_of(mode, ntargets=None):
    """该模式本次分离消耗的积分。

    全分离 Pro 走统一定价 PRO_COST；其余 = 轨数 × 每轨单价。
    """
    if mode == "pro":
        return PRO_COST
    return COST_PER_TRACK * tracks_of(mode, ntargets)


def _gate_cache_read():
    try:
        with open(GATE_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _gate_cache_write(user):
    try:
        os.makedirs(os.path.dirname(GATE_CACHE), exist_ok=True)
        with open(GATE_CACHE, "w", encoding="utf-8") as f:
            json.dump({"t": time.time(), "user": user}, f, ensure_ascii=False)
    except Exception:
        pass


def _gate_verify_token(tok):
    """验一个令牌。返回 (code, data, user)。code=0 表示网络不通。"""
    code, data = amu_login_gate._post("/api/auth/desktop-verify", {"token": tok})
    user = (data.get("user") or {}) if (code == 200 and data.get("ok")) else None
    return code, data, user


def gate_unlock(user):
    GATE_ST["ok"] = True
    GATE_ST["user"] = user or {}
    GATE_ST["reason"] = ""
    _gate_cache_write(GATE_ST["user"])


def gate_lock(reason=""):
    # 自用模式恒定解锁：退出登录、令牌失效都不能把自用版锁回去
    if INTERNAL:
        return
    GATE_ST["ok"] = False
    GATE_ST["user"] = None
    GATE_ST["reason"] = reason


def gate_startup():
    """启动门：本地令牌验活通过→解锁；网络不通且 72h 内验活过→离线宽限。"""
    if INTERNAL:
        return                      # 自用模式不联网、不验活
    tok = amu_login_gate.load_token()
    if not tok:
        gate_lock()
        return
    code, data, user = _gate_verify_token(tok)
    if user is not None:
        GATE.last_user = user
        gate_unlock(user)
    elif code == 0:  # 连不上验证服务器 → 离线宽限判定
        c = _gate_cache_read()
        if c.get("t") and time.time() - c["t"] < GATE_GRACE:
            gate_unlock(c.get("user") or {})
            GATE_ST["reason"] = "grace"
        else:
            gate_lock("offline")
    else:  # 令牌被明确拒绝（过期/吊销/封禁）
        amu_login_gate.clear_token()
        gate_lock()


def gate_reverify_loop():
    """运行期定时重验：只有服务端明确拒绝才落锁，网络抖动不误伤。"""
    if INTERNAL:
        return                      # 自用模式不启重验线程
    while True:
        time.sleep(GATE_REVERIFY_SECS)
        if not GATE_ST["ok"]:
            continue
        tok = amu_login_gate.load_token()
        if not tok:
            gate_lock()
            continue
        code, data, user = _gate_verify_token(tok)
        if user is not None:
            GATE_ST["user"] = user
            _gate_cache_write(user)
        elif code != 0:
            amu_login_gate.clear_token()
            gate_lock()


def _pub_user():
    u = GATE_ST.get("user") or {}
    return {"nickname": u.get("nickname"), "email": u.get("email"),
            "points": u.get("points")}


def gate_consume(note="", cost=None):
    """分离任务开始前扣积分。返回 (ok, http_code, data)。
    - 扣分成功 → 同步刷新本地余额（前端头部实时可见）
    - 402 积分不足 / 令牌失效 → 拒绝本次任务
    - 网络不通 → 拒绝本次任务（按次付费工具不允许离线白嫖）
    - 自用模式 → 直接放行，不联网、不扣分
    """
    if INTERNAL:
        return True, 200, {"ok": True, "balance": "∞"}
    tok = amu_login_gate.load_token()
    if not tok:
        return False, 401, {"error": "登录状态已失效，请重新登录后再使用"}
    cost = COST_PER_TRACK if cost is None else int(cost)
    body = {"token": tok, "task": SEPARATE_TASK, "cost": cost,
            "nonce": uuid.uuid4().hex}  # nonce 防超时重试重复扣
    if note:
        body["note"] = note[:120]
    try:
        code, data = amu_login_gate._post("/api/consume-point", body)
    except Exception:
        return False, 0, {"error": "暂时连不上积分服务器，请检查网络后重试（分离为按次计费，需联网扣分）"}
    if code == 200 and data.get("ok"):
        # 同步本地余额，头部积分显示实时更新
        if isinstance(GATE_ST.get("user"), dict):
            GATE_ST["user"]["points"] = data.get("balance")
            _gate_cache_write(GATE_ST["user"])
        return True, code, data
    if not isinstance(data, dict):
        data = {}
    # 兜底：服务器没给出具体原因时，按 HTTP 状态码区分，
    # 别一律「请稍后再试」——用户无从判断是网络问题还是服务器故障。
    if not data.get("error"):
        if code == 0:
            data["error"] = "连不上积分服务器，请检查网络后重试（按次计费需联网扣分）"
        elif code in (500, 502, 503, 504):
            data["error"] = "积分服务器暂时异常（%s），请稍后重试；已扣的分不会重复扣" % code
        else:
            data["error"] = "扣分未完成（服务器返回 %s），请稍后重试" % code
    data["code"] = code
    return False, code, data


def _safe_name(header, default):
    name = urllib.parse.unquote(header or default)
    # 仅保留文件名，杜绝路径穿越
    return os.path.basename(name) or default


def do_separate(job_id, input_path, mode, device, quality=None, targets=None):
    """分离主流程：固定产出 44.1kHz 母版 WAV。

    输出格式（wav/flac/mp3）与导出采样率（44.1/48k）已改为**下载时**按需转换，
    分离阶段不再关心这两项 —— 母版永远是无损 WAV，转码只发生在导出时。
    """
    if quality is None:
        quality = separate_core.DEFAULT_QUALITY
    shifts, overlap = separate_core.quality_params(quality)
    label = separate_core.MODES.get(mode, {}).get("label", mode)

    # 真实进度回调：53 轨模型一次要跑几分钟，前端不该只靠时间估算转圈
    def _progress(pct, stage=""):
        try:
            with LOCK:
                JOBS[job_id]["real_progress"] = max(0.0, min(99.0, float(pct)))
                JOBS[job_id]["real_stage"] = str(stage or "")
        except Exception:
            pass

    with LOCK:
        JOBS[job_id]["status"] = "separating"
        JOBS[job_id]["message"] = "%s · GPU 分离中…" % label
    out_dir = os.path.join(UPLOAD_DIR, job_id, "out")
    res = separate_core.run_separation(input_path, out_dir, mode=mode,
                                       device=device, fmt="wav",
                                       shifts=shifts, overlap=overlap,
                                       sr=separate_core.DEFAULT_SR,
                                       targets=targets, progress_cb=_progress)
    with LOCK:
        j = JOBS[job_id]
        if res["ok"]:
            res["fmt"] = "wav"
            j["status"] = "done"
            j["result"] = res
            j["message"] = "分离完成"
        else:
            j["status"] = "error"
            j["error"] = res["error"]
            j["message"] = "分离失败"
        j["progress"] = 100


class Handler(BaseHTTPRequestHandler):
    server_version = "AmuMusicSplitter/3.0"
    protocol_version = "HTTP/1.1"

    # ---------------- 工具 ----------------
    def _send_json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_file(self, path, ctype, download=False, filename=None):
        """发送文件，支持 HTTP Range（浏览器 <audio> seek 必需）与流式写出。"""
        if not os.path.isfile(path):
            self._send_json(404, {"error": "文件不存在"})
            return
        size = os.path.getsize(path)
        start, end, partial = 0, size - 1, False

        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            spec = rng[6:].split(",")[0].strip()
            try:
                s, _, e = spec.partition("-")
                if s:
                    start = int(s)
                    end = int(e) if e else size - 1
                else:                       # bytes=-N 取末尾 N 字节
                    start = max(0, size - int(e or 0))
                    end = size - 1
                if start >= size or start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", "bytes */%d" % size)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                end = min(end, size - 1)
                partial = True
            except ValueError:
                start, end, partial = 0, size - 1, False

        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        if download:
            fn = filename or os.path.basename(path)
            self.send_header(
                "Content-Disposition",
                "attachment; filename*=UTF-8''%s" % urllib.parse.quote(fn))
        elif size < 2 * 1024 * 1024:
            # 小文本文件（界面等）禁用缓存：改了前端刷新即生效，不用手动清浏览器缓存
            self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        if self.command == "HEAD":
            return
        try:
            with open(path, "rb") as f:
                f.seek(start)
                remain = length
                while remain > 0:
                    chunk = f.read(min(CHUNK, remain))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remain -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _ctype_of(self, path):
        p = path.lower()
        if p.endswith(".mp3"):
            return "audio/mpeg"
        if p.endswith(".flac"):
            return "audio/flac"
        if p.endswith(".zip"):
            return "application/zip"
        if p.endswith(".ico"):
            return "image/x-icon"
        if p.endswith(".png"):
            return "image/png"
        return "audio/wav"

    def log_message(self, *args):
        pass  # 静默日志

    # ---------------- 数据查询 ----------------
    def _job_and_result(self, job):
        with LOCK:
            j = JOBS.get(job)
            if not j:
                return None, None
            rj = j.get("result")
            return dict(j), (dict(rj) if rj else None)

    @staticmethod
    def _stem_of(rj, stem):
        if rj:
            for s in rj.get("stems", []):
                if s["name"] == stem:
                    return s["path"]
        return None

    def _converted(self, job, src, fmt, sr):
        """按需转出「指定格式 + 采样率」的文件（带磁盘缓存，结果放 job/_export/）。

        请求的就是母版规格（wav + 44.1k）时直接返回源文件，不做任何转换。
        返回 (path, error)。
        """
        if fmt == "wav" and sr == separate_core.DEFAULT_SR:
            return src, None
        cache_dir = os.path.join(UPLOAD_DIR, job, "_export")
        os.makedirs(cache_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(src))[0]
        ext = ".flac" if fmt == "flac" else (".mp3" if fmt == "mp3" else ".wav")
        dst = os.path.join(cache_dir, "%s_%s_%d%s" % (base, fmt, int(sr), ext))
        if not os.path.isfile(dst):
            with EXPORT_LOCK:
                if not os.path.isfile(dst):        # 双检：等锁期间别人可能已转好
                    ok, err = separate_core.convert_audio(src, dst, fmt, sr)
                    if not ok:
                        return None, (err or "导出转换失败")
        return dst, None

    @staticmethod
    def _dl_params(q):
        """解析导出参数：fmt（wav/flac/mp3）与 sr（44100/48000），非法值回落默认。"""
        fmt = q.get("fmt", ["wav"])[0]
        if fmt not in ("wav", "flac", "mp3"):
            fmt = "wav"
        sr = separate_core.normalize_sr(q.get("sr", [None])[0])
        return fmt, sr

    # ---------------- GET ----------------
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)

        # ── 准入状态（前端据此决定是否显示任何登录 / 积分元素）──
        if u.path == "/api/gate/status":
            # free = 完全免费版开关（前端唯一依据）；selfuse = 外部开关真实状态，
            # 仅用于诊断，FREE_MODE 下它开着关着都不影响准入。
            self._send_json(200, {"ok": GATE_ST["ok"], "reason": GATE_ST["reason"],
                                  "user": _pub_user(),
                                  "selfuse": amu_selfuse.is_enabled(),
                                  "free": FREE_MODE})
            return

        # 页面与品牌静态资源始终可发（前端按 status 决定显示登录页还是主界面）
        _static = u.path.lstrip("/")
        if _static in ("favicon.ico", "favicon-32.png", "apple-touch-icon.png",
                       "logo.png"):
            p = os.path.join(WEB_DIR, _static)
            if os.path.isfile(p):
                self._send_file(p, self._ctype_of(p))
                return
        if u.path not in ("/", "/index.html") and not GATE_ST["ok"]:
            self._send_json(401, {"error": "gate_locked"})
            return

        if u.path in ("/", "/index.html"):
            self._send_file(os.path.join(WEB_DIR, "index.html"),
                            "text/html; charset=utf-8")
            return

        # ── 乐器提取模型：就绪状态查询（549 时为下载中）──
        if u.path == "/api/instrument-model":
            self._send_json(200, separate_core.instrument_model_status())
            return

        if u.path == "/api/status":
            job = q.get("job", [""])[0]
            with LOCK:
                j = JOBS.get(job)
                if not j:
                    self._send_json(404, {"error": "任务不存在"})
                    return
                dur = j.get("duration") or 0.0
                dev = j.get("device", "cuda")
                spd = separate_core.speed_of_with_quality(
                    j.get("mode", "basic"), dev,
                    j.get("quality", separate_core.DEFAULT_QUALITY))
                est = (dur / spd) if dur and spd > 0 else 0.0
                el = time.time() - j["start"]
                out_msg = j.get("message", "")
                if j["status"] == "separating":
                    # 乐器提取模式下 worker 会吐真实百分比，优先用它；
                    # 其余模式（Demucs 不给进度）仍按时间估算。
                    real = j.get("real_progress")
                    prog = (int(real) if real is not None
                            else (min(98, int(el / est * 100)) if est > 0 else 5))
                    if real is not None and j.get("mode") in ("instruments", "pro"):
                        st = _STAGE_TEXT.get(j.get("real_stage", ""), "")
                        if st:
                            out_msg = st
                elif j["status"] in ("done", "error"):
                    prog = 100
                else:
                    prog = 0
                out = {
                    "status": j["status"],
                    "progress": prog,
                    "message": out_msg,
                    "elapsed": round(el, 1),
                    "estimated": round(est, 1),
                }
                if j["status"] == "done":
                    res = dict(j["result"])
                    # 本机绝对路径不下发前端
                    res["stems"] = [{"name": s["name"], "size": s["size"]}
                                    for s in res["stems"]]
                    out["result"] = res
                if j["status"] == "error":
                    out["error"] = j.get("error", "")
            self._send_json(200, out)
            return

        # 原始上传音频（波形试听用）
        if u.path == "/api/source":
            job = q.get("job", [""])[0]
            j, _ = self._job_and_result(job)
            if not j:
                self._send_json(404, {"error": "任务不存在"})
                return
            p = j.get("input_path")
            if not p or not os.path.isfile(p):
                self._send_json(404, {"error": "原始音频不存在"})
                return
            self._send_file(p, self._ctype_of(p))
            return

        # 音轨播放流 / 下载（fmt + sr = 下载时选定的导出格式与采样率）
        if u.path in ("/api/stream", "/api/download"):
            job = q.get("job", [""])[0]
            stem = q.get("stem", [""])[0]
            j, rj = self._job_and_result(job)
            if not j:
                self._send_json(404, {"error": "任务不存在"})
                return
            if stem in ("", "__mix__", "mix"):
                path = j.get("input_path")
            else:
                path = self._stem_of(rj, stem)
            if not path or not os.path.isfile(path):
                self._send_json(404, {"error": "音轨文件不存在"})
                return
            force_dl = q.get("dl", ["0"])[0] == "1"
            if u.path == "/api/download" and force_dl:
                # 下载才做导出转换；播放流（stream）永远用母版
                fmt, sr = self._dl_params(q)
                path, cerr = self._converted(job, path, fmt, sr)
                if not path:
                    self._send_json(500, {"error": cerr})
                    return
            fn = "%s.%s" % (stem if stem not in ("", "__mix__", "mix") else "原始音频",
                            os.path.splitext(path)[1].lstrip("."))
            self._send_file(path, self._ctype_of(path),
                            download=force_dl, filename=fn)
            return

        # 全部音轨打包 zip（支持按 stems 参数过滤，尊重前端的 S/M 选择；
        # fmt + sr = 下载时选定的导出格式与采样率，转换结果缓存复用）
        if u.path == "/api/export":
            job = q.get("job", [""])[0]
            j, rj = self._job_and_result(job)
            if not rj:
                self._send_json(404, {"error": "任务不存在或尚未完成"})
                return
            fmt, sr = self._dl_params(q)
            # 先把所选轨全部转成目标规格（命中缓存则秒回）
            conv = []
            for s in rj["stems"]:
                cp, cerr = self._converted(job, s["path"], fmt, sr)
                if not cp:
                    self._send_json(500, {"error": "「%s」转换失败：%s" % (s["name"], cerr)})
                    return
                conv.append({"name": s["name"], "path": cp})
            only = q.get("stems", [""])[0].strip()
            if only:
                wanted = [s.strip() for s in only.split(",") if s.strip()]
                sel = [c for c in conv if c["name"] in wanted]
            else:
                sel = conv
            if not sel:
                self._send_json(400, {"error": "没有可导出的音轨（参数无效或全部被静音）"})
                return
            readme = (
                "Amu Music 音频分离软件 完整版 · 导出包\r\n"
                "==============================\r\n"
                "说明：各音轨是原始录音的衍生物，版权仍归原录音权利人所有。\r\n"
                "请仅处理你拥有合法权利的内容。\r\n"
                "提示：导出采样率请与宿主工程保持一致（44.1kHz / 48kHz）。\r\n"
                "     采样率不匹配会变速变调；需要更改时回结果页重新下载即可，\r\n"
                "     无需重新分离，母版会被完整保留。\r\n"
            )
            if only:
                # 按选择导出：每次实时打包，文件名带筛选标识
                import io
                bio = io.BytesIO()
                try:
                    with zipfile.ZipFile(bio, "w", zipfile.ZIP_STORED) as z:
                        for i, s in enumerate(sel, 1):
                            ext = os.path.splitext(s["path"])[1]
                            z.write(s["path"], arcname="%02d_%s%s" % (i, s["name"], ext))
                        z.writestr("README.txt", readme.encode("utf-8"))
                except Exception as e:
                    self._send_json(500, {"error": "打包失败: %s" % e})
                    return
                tmpp = os.path.join(UPLOAD_DIR, job, "export_sel_%d.zip" % abs(hash(only)))
                with open(tmpp, "wb") as f:
                    f.write(bio.getvalue())
                fn = "Amu分离_%s_选中%d轨.zip" % (job, len(sel))
                self._send_file(tmpp, "application/zip", download=True, filename=fn)
                return
            zpath = os.path.join(UPLOAD_DIR, job,
                                 "export_%s_%d.zip" % (fmt, int(sr)))
            if not os.path.isfile(zpath):
                tmp = zpath + ".part"
                try:
                    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
                        for i, s in enumerate(sel, 1):
                            ext = os.path.splitext(s["path"])[1]
                            z.write(s["path"], arcname="%02d_%s%s" % (i, s["name"], ext))
                        z.writestr("README.txt", readme.encode("utf-8"))
                except Exception as e:
                    self._send_json(500, {"error": "打包失败: %s" % e})
                    return
                os.replace(tmp, zpath)
            self._send_file(zpath, "application/zip", download=True,
                            filename="Amu分离_%s_%s_%dk.zip" % (job, fmt, int(sr) // 1000))
            return

        # 波形峰值
        if u.path == "/api/peaks":
            job = q.get("job", [""])[0]
            j, rj = self._job_and_result(job)
            if not j:
                self._send_json(404, {"error": "任务不存在"})
                return
            cache_dir = os.path.join(UPLOAD_DIR, job, "peaks")
            os.makedirs(cache_dir, exist_ok=True)

            def peaks_of(key, path):
                cf = os.path.join(cache_dir, key + ".json")
                if os.path.isfile(cf):
                    try:
                        with open(cf, "r", encoding="utf-8") as f:
                            return json.load(f)
                    except Exception:
                        pass
                d = separate_core.compute_peaks(path)
                # 只缓存成功结果：失败（如缺依赖、文件被占）下次重算，
                # 否则一次坏结果会永远黏在缓存里
                if not d.get("error"):
                    try:
                        with open(cf, "w", encoding="utf-8") as f:
                            json.dump(d, f)
                    except Exception:
                        pass
                return d

            if q.get("all", ["0"])[0] == "1":
                out = {"mix": None, "stems": {}}
                inp = j.get("input_path")
                if inp and os.path.isfile(inp):
                    out["mix"] = peaks_of("__mix__", inp)
                for s in (rj["stems"] if rj else []):
                    out["stems"][s["name"]] = peaks_of(s["name"], s["path"])
                self._send_json(200, out)
                return

            stem = q.get("stem", [""])[0]
            if stem in ("", "__mix__", "mix"):
                path, key = j.get("input_path"), "__mix__"
            else:
                path, key = self._stem_of(rj, stem), stem
            if not path or not os.path.isfile(path):
                self._send_json(404, {"error": "音轨不存在"})
                return
            self._send_json(200, peaks_of(key, path))
            return

        self._send_json(404, {"error": "接口不存在"})

    # ---------------- POST ----------------
    def do_POST(self):
        u = urllib.parse.urlparse(self.path)

        # ── 乐器提取模型：触发下载（幂等，重复调用只是查状态）──
        if u.path == "/api/instrument-model/install":
            if not GATE_ST["ok"]:
                self._send_json(401, {"error": "gate_locked"})
                return
            st = separate_core.instrument_model_download()
            code = 200
            pr = st.get("progress") or {}
            if pr.get("done") and not pr.get("ok"):
                code = 502           # 下载失败：让前端能区分「已触发」与「出问题」
            self._send_json(code, st)
            return

        # ── 登录门：登录 / 退出 ──
        if u.path == "/api/gate/login":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode("utf-8", "ignore") or "{}")
            except Exception:
                body = {}
            email = str(body.get("email", "")).strip()
            pwd = str(body.get("password", ""))
            if not email or not pwd:
                self._send_json(400, {"error": "请输入邮箱和密码"})
                return
            ok, msg = GATE.login(email, pwd)
            if ok:
                gate_unlock(GATE.last_user or {})
                self._send_json(200, {"ok": True, "user": _pub_user()})
            else:
                self._send_json(401, {"error": msg or "登录失败，请稍后再试"})
            return

        if u.path == "/api/gate/logout":
            if INTERNAL:
                # 自用模式没有登录态可退，保持解锁
                self._send_json(200, {"ok": True, "selfuse": True})
                return
            GATE.logout()
            gate_lock()
            self._send_json(200, {"ok": True})
            return

        if not GATE_ST["ok"]:
            self._send_json(401, {"error": "gate_locked"})
            return

        if u.path != "/api/separate":
            self._send_json(404, {"error": "接口不存在"})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            raw_name = self.headers.get("X-Filename", "input")
            mode = self.headers.get("X-Mode", "basic")
            device = self.headers.get("X-Device", "cuda")
            quality = self.headers.get("X-Quality", separate_core.DEFAULT_QUALITY)
            if mode not in separate_core.MODES:
                mode = "basic"
            if quality not in separate_core.QUALITY:
                quality = separate_core.DEFAULT_QUALITY

            # 乐器提取 / 全分离 Pro：目标乐器来源不同
            targets = None
            if mode == "instruments":
                targets, terr = separate_core.normalize_targets(
                    self.headers.get("X-Targets", ""))
                if terr:
                    self._send_json(400, {"error": terr})
                    return
            elif mode == "pro":
                # 全部分离：后端展开为全部乐器（normalize_targets("all")）
                targets = "all"

            data = self.rfile.read(length)
            if not data:
                self._send_json(400, {"error": "空文件"})
                return

            # ── 计费分支：FREE_MODE / 自用模式下 need=0 且 gate_consume 直接放行 ──
            fname = _safe_name(raw_name, "input")
            ntrk = tracks_of(mode, len(targets) if mode == "instruments" else None)
            # 完全免费版实扣恒为 0，界面也不再显示任何积分信息
            need = 0 if INTERNAL else cost_of(
                mode, len(targets) if mode == "instruments" else None)
            ok, code, cd = gate_consume(note=fname, cost=need)
            if not ok:
                if code == 402:
                    if mode == "pro":
                        msg = ("积分不足：本次「全分离 Pro」需要 %d 积分，当前余额 %s。"
                               "可在官网每日签到攒积分。" % (need, cd.get("balance", "?")))
                    else:
                        msg = ("积分不足：本次分离需要 %d 积分（%d 轨 × %d 积分/轨），"
                               "当前余额 %s。可在官网每日签到攒积分。"
                               % (need, ntrk, COST_PER_TRACK, cd.get("balance", "?")))
                    self._send_json(402, {"error": msg,
                                          "balance": cd.get("balance"), "needed": need})
                elif code == 401:
                    gate_lock()
                    self._send_json(401, {"error": cd.get("error", "gate_locked"),
                                          "gate_locked": True})
                else:
                    # 保留服务器返回的真实文案 + 状态码，便于前端与日志定位
                    self._send_json(503 if code in (0, 500, 502, 503, 504) else 502,
                                    {"error": cd.get("error") or "扣分未完成，请稍后重试",
                                     "code": code,
                                     "balance": cd.get("balance")})
                print("  [计费] 扣分失败 code=%s detail=%s" % (code, cd.get("error")))
                return

            ext = os.path.splitext(fname)[1] or ".wav"
            job_id = uuid.uuid4().hex[:12]
            jdir = os.path.join(UPLOAD_DIR, job_id)
            os.makedirs(jdir, exist_ok=True)
            inp = os.path.join(jdir, "input" + ext)
            with open(inp, "wb") as f:
                f.write(data)

            dur = separate_core.get_audio_duration(inp) or 0.0
            with LOCK:
                JOBS[job_id] = {
                    "status": "queued", "start": time.time(), "duration": dur,
                    "device": device, "mode": mode, "quality": quality,
                    "input_path": inp,
                    "message": "已接收文件，准备分离",
                }
                if mode in ("instruments", "pro"):
                    JOBS[job_id]["targets"] = (
                        separate_core.INSTRUMENT_STEMS if mode == "pro" else targets)
            threading.Thread(target=do_separate,
                             args=(job_id, inp, mode, device, quality, targets),
                             daemon=True).start()
            spd = separate_core.speed_of_with_quality(mode, device, quality)
            self._send_json(200, {"job": job_id, "duration": round(dur, 2),
                                  "estimated": round(dur / spd, 1) if spd > 0 else 0,
                                  "cost": need,                   # 本次实扣积分
                                  "balance": cd.get("balance")})  # 带上扣费后余额，前端实时刷新
        except Exception as e:
            self._send_json(500, {"error": "服务器错误: %s" % e})


_STEM_ORDER = ["vocals", "lead", "harmony", "no_vocals", "drums",
               "bass", "guitar", "piano", "other"] + separate_core.INSTRUMENT_STEMS

# 乐器提取 worker 上报的阶段 → 界面文案（比干巴巴一句"GPU 分离中"更能让人安心等待）
_STAGE_TEXT = {
    "load":   "正在加载乐器识别模型…",
    "decode": "正在解码音频…",
    "run":    "正在逐段识别乐器…",
    "save":   "正在写入音轨…",
}


def _recover_jobs():
    """启动时扫描 uploads/，把历史上已完成的任务重新登记进 JOBS。

    进程重启后内存任务表清空，但磁盘上的成品还在 —— 恢复后旧结果页的
    播放 / 波形 / 下载全部继续可用（应对服务重启不丢用户已付费的结果）。
    """
    if not os.path.isdir(UPLOAD_DIR):
        return
    for name in os.listdir(UPLOAD_DIR):
        jdir = os.path.join(UPLOAD_DIR, name)
        outroot = os.path.join(jdir, "out")
        if len(name) != 12 or not os.path.isdir(outroot):
            continue
        stems = []
        for root, _dirs, files in os.walk(outroot):
            for fn in files:
                if not fn.lower().endswith((".wav", ".flac")):
                    continue
                p = os.path.join(root, fn)
                key = os.path.splitext(fn)[0]
                if key in _STEM_ORDER:
                    stems.append({"name": key, "path": p,
                                  "size": os.path.getsize(p)})
        if not stems:
            continue
        stems.sort(key=lambda s: _STEM_ORDER.index(s["name"]))
        inp = None
        for fn in sorted(os.listdir(jdir)):
            if fn.startswith("input."):
                inp = os.path.join(jdir, fn)
                break
        dur = separate_core.get_audio_duration(inp) if inp else None
        if not dur:
            dur = 0.0
        n = len(stems)
        names = {s["name"] for s in stems}
        pro_mark = os.path.join(outroot, ".amu_pro")
        dropped, is_pro = [], False
        if os.path.isfile(pro_mark):
            try:
                with open(pro_mark, "r", encoding="utf-8") as f:
                    dropped = json.load(f).get("dropped", [])
                is_pro = True
                # 首次结果里已剔除的静音轨，重启后不能又冒出来
                stems = [s for s in stems if s["name"] not in set(dropped)]
            except Exception:
                dropped, is_pro = [], False
        # 判据必须是「全部轨名都属于乐器清单」，不能用「有交集」。
        # multi 模式固定输出 bass / drums / piano / guitar 等，这些名字也在
        # INSTRUMENT_STEMS 里，用交集判断会把多轨任务误标成乐器提取。
        inst_set = set(separate_core.INSTRUMENT_STEMS)
        if is_pro:
            mode = "pro"
        elif names and names <= inst_set:
            mode = "instruments"
        else:
            mode = "multi" if n >= 7 else ("four" if n >= 4 else "basic")
        quality = separate_core.DEFAULT_QUALITY
        with LOCK:
            if name in JOBS:
                continue
            JOBS[name] = {
                "status": "done", "start": time.time(), "duration": dur,
                "device": "cuda", "mode": mode, "quality": quality,
                "targets": sorted(names & set(separate_core.INSTRUMENT_STEMS)),
                "input_path": inp, "message": "分离完成",
                "result": {"ok": True, "error": "", "outdir": outroot,
                           "stems": stems, "duration": dur, "elapsed": 0.0,
                           "mode": mode, "engine": "recovered",
                           "fmt": os.path.splitext(stems[0]["path"])[1].lstrip("."),
                           "sr": separate_core.DEFAULT_SR,
                           "silent": separate_core.silent_stems(stems),
                           "dropped_silent": len(dropped) if is_pro else 0},
            }
    if JOBS:
        print("  已恢复 %d 个历史任务" % len(JOBS))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8777
    gate_startup()
    _recover_jobs()
    threading.Thread(target=gate_reverify_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("=" * 52)
    print("  Amu Music · 音频分离软件 完整版（本地版）已启动")
    print("  浏览器打开 → http://127.0.0.1:%d" % port)
    print("  模式：基本(2轨) / 四轨(4轨) / 多轨(7轨) / 乐器提取(按勾选) / 全分离Pro(全部乐器)")
    if FREE_MODE:
        print("  使用门槛：完全免费 · 无需登录 · 无积分 · 不限次数 · 无需联网")
    elif INTERNAL:
        print("  计费：自用模式（免登录免积分）")
    else:
        print("  计费：%d 积分/轨（基本 %d · 四轨 %d · 多轨 %d · 全分离Pro %d）"
              "（官网每日签到可攒）"
              % (COST_PER_TRACK, cost_of("basic"), cost_of("four"),
                 cost_of("multi"), cost_of("pro")))
    w_dir = separate_core.find_ffmpeg()
    try:
        import mega53_download as _m53
        _md = _m53.installed_dir()
        _m53_txt = (("已就位 %s" % _md) if _md else
                    "未安装（界面可一键下载，约 %d MB）" % (_m53.CKPT_SIZE // 1048576))
    except Exception as _e:
        _m53_txt = "状态未知（%s）" % _e
    print("  乐器提取：模型%s  依赖 ffmpeg %s"
          % (_m53_txt, "已就位" if w_dir else "缺失"))
    if FREE_MODE:
        print("  运行方式：免费版（所有功能开放，登录门与积分计费均已停用）")
    elif INTERNAL:
        print("  运行方式：自用完整版（免登录 · 不消耗积分 · 不联网验活）")
    elif GATE_ST["ok"]:
        u = GATE_ST.get("user") or {}
        who = u.get("nickname") or u.get("email") or "?"
        tag = "（离线宽限）" if GATE_ST.get("reason") == "grace" else ""
        print("  已登录： %s %s" % (who, tag))
    else:
        print("  未登录 —— 打开页面后需先登录 Amu 账号")
    print("  仅监听本机 127.0.0.1，音频数据不出本机")
    print("  按 Ctrl+C 停止")
    print("=" * 52)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
