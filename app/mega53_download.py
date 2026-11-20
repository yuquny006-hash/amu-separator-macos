# -*- coding: utf-8 -*-
"""53 轨模型的就绪检查与按需下载（标准库实现，零第三方依赖）。

为什么要有这个模块
------------------
「乐器提取」模式依赖一份 1.37 GB 的权重。把它硬塞进安装包会让交付包体积失控，
不放则用户一上来就点不动这个功能。因此采取双轨：

    优先用包内预置的 models/mega53/ → 没有则引导用户下载一次

设计要点：断网重连后可续传、完成后做 SHA256 校验、先写 .part 临时文件再改名
—— 避免下载到一半的残缺权重被当成完整模型加载（那样不会报错，只会输出垃圾）。
"""
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

CKPT_NAME = "mvsep_mega_53.ckpt"
YAML_NAME = "mega53.yaml"

# ── 完整性黄金值 ────────────────────────────────────────────────────────
# 来源：ZFTurbo Music-Source-Separation-Training Release v1.0.21（MIT）
# 本机素材算得：1368919887 字节 / sha256 c6282089…（改权重时必须同步改这里）
CKPT_SIZE = 1368919887
CKPT_SHA256 = "c62820893bbf86d4e734f966bd142d9157cfc8bb8e79e9d8f9ea553f3ff3519f"

# ── 下载源 ─────────────────────────────────────────────────────────────
# 主源走 HF 国内镜像（本机实测 302 → CDN，可用）；备用源走原始 HF。
# 想自建下载服务器时，给这两个环境变量赋值即可永久覆盖，无需改代码。
_MIRROR = "https://hf-mirror.com/oulianov/mvsep_mega_53/resolve/main"
_ORIGIN = "https://huggingface.co/oulianov/mvsep_mega_53/resolve/main"
# 原作者仓库 Release v1.0.21 —— 官方原始出处，作兜底源
_GITHUB = ("https://github.com/ZFTurbo/Music-Source-Separation-Training"
           "/releases/download/v1.0.21")


def _sources():
    base = os.environ.get("AMU_MEGA53_URL", "").strip()
    ck = os.environ.get("AMU_MEGA53_URL_CKPT", "").strip()
    ym = os.environ.get("AMU_MEGA53_URL_YAML", "").strip()
    if ck and ym:
        return [(ck, ym)]
    if base:
        return [(base.rstrip("/") + "/" + "mvsep_mega_model_bs_roformer_53_stems_v1.ckpt",
                 base.rstrip("/") + "/" + "mvsep_mega_model_bs_roformer_53_stems.yaml")]
    return [
        (_MIRROR + "/mvsep_mega_model_bs_roformer_53_stems_v1.ckpt",
         _MIRROR + "/mvsep_mega_model_bs_roformer_53_stems.yaml"),
        # HF 镜像对同一文件的短时间内重复请求会限流，换不同 host 往往直接成功
        (_GITHUB + "/mvsep_mega_model_bs_roformer_53_stems_v1.ckpt",
         _GITHUB + "/mvsep_mega_model_bs_roformer_53_stems.yaml"),
        (_ORIGIN + "/mvsep_mega_model_bs_roformer_53_stems_v1.ckpt",
         _ORIGIN + "/mvsep_mega_model_bs_roformer_53_stems.yaml"),
    ]


def _user_model_dir():
    """用户级写盘兜底目录。

    为什么需要：安装包若是压缩版解压到任意目录，包内 models/mega53 都可写；
    但若程序位于 C:\\Program Files 这类受保护目录，包内目录**不可写**，
    下载就会卡死在「找不到可写入的模型目录」。因此最后加一个用户目录兜底，
    与登录令牌的存放位置（%APPDATA%\\AmuTools）保持一致。
    """
    if os.name == "nt":
        app = os.environ.get("APPDATA", "").strip()
        if app:
            return os.path.join(app, "AmuTools", "models", "mega53")
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Application Support",
                            "AmuTools", "models", "mega53")
    return os.path.join(home, ".config", "AmuTools", "models", "mega53")


def candidate_dirs():
    """模型目录候选（公开）。顺序与 instrument_worker.resolve_model_dir 保持一致。

    约定优先级：环境变量 > 包内 models/mega53 > 上级 models/mega53 > 用户目录。
    """
    env_dir = os.environ.get("AMU_MEGA53_DIR", "").strip()
    cands = []
    if env_dir:
        cands.append(env_dir)
    cands.append(os.path.join(HERE, "models", "mega53"))
    cands.append(os.path.join(os.path.dirname(HERE), "models", "mega53"))
    cands.append(_user_model_dir())
    return cands


def _candidate_dirs():
    """向后兼容的内部别名。"""
    return candidate_dirs()


def installed_dir():
    """已存在的模型目录（含完整权重），没有则返回 ''。"""
    for d in _candidate_dirs():
        if os.path.isfile(os.path.join(d, CKPT_NAME)) and \
           os.path.isfile(os.path.join(d, YAML_NAME)):
            return d
    return ""


def writable_dir():
    """挑选一个可写的落点目录：已有就复用，否则挑第一个能创建的。"""
    exist = installed_dir()
    if exist:
        return exist
    for d in _candidate_dirs():
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, ".wtest")
            with open(probe, "w") as f:
                f.write("x")
            os.remove(probe)
            return d
        except Exception:
            continue
    return ""


def _sha256_of(path, stop=None):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            h.update(b)
            n += len(b)
            if stop is not None and stop():
                return None, n
    return h.hexdigest(), n


def _probe_opener(timeout=20):
    """带 UA 的 opener。代理：跟随系统环境变量；
    若本机设置了会干扰下载的环境代理，可用 AMU_NO_PROXY=1 关闭。"""
    if os.environ.get("AMU_NO_PROXY"):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


class _Prog(object):
    """下载任务状态，供 API 轮询。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.active = False
        self.done = False
        self.ok = False
        self.error = ""
        self.total = CKPT_SIZE
        self.got = 0
        self.speed = 0.0
        self.stage = ""
        self.started = 0.0

    def snapshot(self):
        with self.lock:
            return {
                "active": self.active, "done": self.done, "ok": self.ok,
                "error": self.error, "total": self.total, "got": self.got,
                "speed": round(self.speed, 1), "stage": self.stage,
                "percent": (round(self.got * 100.0 / self.total, 1)
                            if self.total else 0.0),
            }

    def set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)


PROG = _Prog()
_LOCK = threading.Lock()


def status():
    """返回模型就绪状况。前端据此决定要不要显示下载卡片。"""
    st = {"ready": False, "installed": False, "path": "", "size": 0,
          "expected": CKPT_SIZE, "need_mb": round(CKPT_SIZE / 1048576.0),
          "check_ok": False, "check_msg": ""}
    d = installed_dir()
    if d:
        p = os.path.join(d, CKPT_NAME)
        sz = os.path.getsize(p)
        st["path"] = p
        st["size"] = sz
        st["installed"] = True
        if sz != CKPT_SIZE:
            st["check_msg"] = "权重体积不符（期望 %d 字节，实际 %d）——请删除后重新下载" % (CKPT_SIZE, sz)
        else:
            st["ready"] = True
            st["check_ok"] = True
    else:
        # 分开诊断：只有权重没配置时别笼统说「未安装」，否则很难排查
        for cand in _candidate_dirs():
            cp = os.path.join(cand, CKPT_NAME)
            yp = os.path.join(cand, YAML_NAME)
            if os.path.isfile(cp) and not os.path.isfile(yp):
                st["path"] = cp
                st["size"] = os.path.getsize(cp)
                st["check_msg"] = ("权重已下载但缺少配置文件 %s，"
                                   "请再安装一次补全（很快，约 4 KB）" % YAML_NAME)
                break
    st["downloading"] = PROG.active
    st["progress"] = PROG.snapshot()
    return st


def _config_ok(path):
    """配置文件结构校验。返回 (是否可用, 说明)。

    为什么不能只靠「文件存在」判断：上一版把期望字节数写死成权重大小，
    而 HF 对这份 4KB 的 yaml 是 chunked 传输（没有 Content-Length）→
    永远判定「下载中断」，全新安装 100% 失败。这里改为按内容校验，
    残缺 / 错页 / 报错页都会被拦下，且不受传输方式影响。

    没有 yaml 模块时不强求（本模块按零第三方依赖设计），此时放过。
    """
    if not (path and os.path.isfile(path)):
        return False, "配置文件不存在"
    try:
        import yaml
    except Exception:
        return True, "（无 yaml 模块，跳过结构校验）"
    try:
        class _L(yaml.SafeLoader):
            pass
        # 配置里有 !!python/tuple 标签，不注册构造器会直接抛 ConstructorError
        _L.add_constructor("tag:yaml.org,2002:python/tuple",
                           lambda l, n: tuple(l.construct_sequence(n)))
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=_L) or {}
        stems = ((cfg.get("training") or {}).get("instruments")) or []
        if not cfg.get("model"):
            return False, "缺少 model 段"
        if len(stems) < 50:
            return False, "instruments 数量异常（读到 %d 项）" % len(stems)
        return True, ""
    except Exception as e:
        return False, str(e)


def _fetch(url, dst, binary=True, tries=4, expect=None, report=True):
    """把 url 续传到 dst.part，完成后改名为 dst。返回 (ok, error)。

    binary=True 时做内容类型嗅探：下载源若把报错信息以 json / html 返回，
    要立刻报错重试，而不是把它当权重存下来（那样加载时才炸，难以定位）。
    ⚠️ 配置文件（yaml）本身 Content-Type 就是 text/plain，必须传 binary=False，
       否则会把正确的配置误判成错误页 —— 曾导致「权重下完但始终显示未安装」。

    expect：期望的字节数。权重必须给（= CKPT_SIZE）；小配置文件不要给，
    因为服务端可能是 chunked 传输（无 Content-Length），硬校验必然误判。
    report：是否把进度写进 PROG（配置文件太小，没必要污染进度条）。

    tries：自动重试次数。家用网络抖动远比机房频繁，且 HF 镜像对同一文件的
    短时间内重复请求还会限流 —— 短促地重试几次往往全部失败。实测需要一个
    稍长的退避（3/9/21/45 秒），已下载的部分不会因为重试而丢失。
    """
    last = ""
    for n in range(tries):
        ok, last = _fetch_once(url, dst, binary, expect, report)
        if ok:
            return True, ""
        if n < tries - 1:
            time.sleep(3.0 * (2 ** n) + n * 3)
    return False, last


def _fetch_once(url, dst, binary=True, expect=None, report=True):
    """单次下载尝试，失败返回 (False, 原因)。不要直接调用，用 _fetch。"""
    part = dst + ".part"
    tmp_size = os.path.getsize(part) if os.path.isfile(part) else 0

    opener = _probe_opener()
    headers = {"User-Agent": "AmuMusic-Separator/2.0 (+https://amumusic.com)"}
    if tmp_size:
        headers["Range"] = "bytes=%d-" % tmp_size

    try:
        req = urllib.request.Request(url, headers=headers)
        resp = opener.open(req, timeout=60)
    except urllib.error.HTTPError as e:
        # 416 = 服务器不认我们的续传起点（文件已下载完），当作从头来
        if e.code == 416 and tmp_size:
            # 服务器不认我们的续传起点（通常是 part 已下完），删掉从头来
            try:
                os.remove(part)
            except Exception:
                pass
            return _fetch_once(url, dst, binary)
        return False, "下载失败（HTTP %s）" % e.code
    except Exception as e:
        return False, "网络错误：%s" % e

    code = getattr(resp, "code", 200) or resp.getcode()
    if tmp_size and code != 206:
        # 服务端不支持续传，从头开始
        tmp_size = 0
        try:
            os.remove(part)
        except Exception:
            pass

    ctype = resp.headers.get("Content-Type", "")
    if binary and ("application/json" in ctype or "text/html" in ctype
                   or ctype.startswith("text/plain")):
        # HF 的错误页 / 重定向说明会以文本形式返回，而不是权重本体
        try:
            body = resp.read(512).decode("utf-8", "replace")
        except Exception:
            body = ""
        return False, "下载源返回的不是权重文件：%s" % body[:120]

    total = int(expect) if expect else 0
    cl = 0
    try:
        cl = int(resp.headers.get("Content-Length") or 0)
        if cl and not expect:
            total = tmp_size + cl
    except Exception:
        cl = 0

    got = tmp_size
    t0 = time.time()
    flag = open(part, "ab") if tmp_size else open(part, "wb")
    try:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            flag.write(chunk)
            got += len(chunk)
            el = max(time.time() - t0, 1e-6)
            if report:
                PROG.set(got=got, total=total, speed=(got - tmp_size) / el / 1048576.0)
    finally:
        try:
            flag.close()
        except Exception:
            pass
        try:
            resp.close()
        except Exception:
            pass

    if got <= 0:
        return False, "下载到空文件，请重试"
    if expect and got != int(expect):
        return False, "下载不完整（已收 %d / 期望 %d 字节），可重试继续" % (
            got, int(expect))
    if cl and got != tmp_size + cl:
        return False, "下载中断（已收 %d / 期望 %d 字节），可重试继续" % (
            got, tmp_size + cl)

    os.replace(part, dst)
    return True, ""


def run_download():
    """后台线程主函数。全程更新 PROG 供前端轮询。"""
    PROG.reset()
    PROG.set(active=True, stage="prepare", started=time.time())
    try:
        d = writable_dir()
        if not d:
            PROG.set(active=False, done=True, ok=False,
                     error="找不到可写入的模型目录，请检查安装目录权限")
            return
        if installed_dir():
            PROG.set(active=False, done=True, ok=True, stage="ready")
            return

        srcs = _sources()
        last_err = ""
        for ck_url, ym_url in srcs:
            # ── 配置文件（4KB）：已有且结构正确就复用，否则重下一次 ──
            PROG.set(stage="config")
            ydst = os.path.join(d, YAML_NAME)
            cfg_ok, why = _config_ok(ydst)
            if cfg_ok:
                ok, err = True, ""
            else:
                ok, err = _fetch(ym_url, ydst, binary=False, report=False)
                if ok:
                    cfg_ok, why = _config_ok(ydst)
                    if not cfg_ok:
                        ok, err = False, why
                        try:
                            os.remove(ydst)
                        except Exception:
                            pass
            if not ok:
                last_err = "配置文件下载失败：%s" % err
                continue

            # ── 权重本体 ──
            cdst = os.path.join(d, CKPT_NAME)
            cpart = cdst + ".part"
            PROG.set(stage="weights",
                     got=os.path.getsize(cpart) if os.path.isfile(cpart) else 0)
            if os.path.isfile(cdst) and os.path.getsize(cdst) == CKPT_SIZE:
                ok, err = True, ""
            else:
                # 权重必须严格按字节数校验：残缺权重加载时不报错，只会输出垃圾
                ok, err = _fetch(ck_url, cdst, binary=True, expect=CKPT_SIZE)
            if not ok:
                last_err = err
                continue

            # ── 完整性校验：体积 + SHA256 ──
            PROG.set(stage="verify")
            try:
                h, _n = _sha256_of(cdst)
            except Exception as e:
                last_err = "校验时读取失败：%s" % e
                continue
            if h != CKPT_SHA256:
                try:
                    os.remove(cdst)
                except Exception:
                    pass
                last_err = "权重校验不通过（SHA256 不符），已删除，请重试下载"
                continue

            PROG.set(active=False, done=True, ok=True, stage="ready")
            return

        PROG.set(active=False, done=True, ok=False, stage="error",
                 error=last_err or "所有下载源均不可用")
    except Exception as e:
        PROG.set(active=False, done=True, ok=False, stage="error",
                 error="下载异常：%s" % e)


def start_download():
    """启动下载（幂等：已在下载则忽略）。返回当前状态快照。"""
    with _LOCK:
        if PROG.active:
            return status()
        threading.Thread(target=run_download, daemon=True).start()
        time.sleep(0.05)
    return status()


def main(argv):
    """命令行自检：python mega53_download.py [--check|--download]"""
    cmd = argv[1] if len(argv) > 1 else "--check"
    st = status()
    if cmd == "--check":
        print(json.dumps(st, ensure_ascii=False, indent=2))
        return 0
    if cmd == "--download":
        t = threading.Thread(target=run_download, daemon=True)
        t.start()
        while True:
            s = PROG.snapshot()
            print("\r  [%s] %5.1f%%  %.1f MB/s  (%d/%d)" %
                  (s["stage"] or "-", s["percent"], s["speed"],
                   s["got"], s["total"]), end="", flush=True)
            if s["done"]:
                print()
                print("结果:", "OK" if s["ok"] else "FAIL " + s["error"])
                break
            time.sleep(0.5)
        t.join(timeout=5)
        return 0 if PROG.ok else 1
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
