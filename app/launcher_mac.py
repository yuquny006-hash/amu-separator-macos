#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Amu Music · 音频分离软件 —— macOS / Linux 启动器。

Windows 版用 launcher.py（系统托盘）；本文件是 mac/linux 的等价薄外壳：
    1. 启动本地 Web 服务（app.py，仅监听 127.0.0.1）
    2. 自动打开浏览器
    3. 常驻前台，Ctrl+C（或关闭终端）即整体退出，无孤儿进程

用法（两种都行）：
    python3 launcher_mac.py            # 默认端口 8777 并自动开浏览器
    python3 launcher_mac.py 8899       # 指定端口
    python3 launcher_mac.py --no-browser

打包成 .app：见 02_打包配置/AmuMusic_mac.spec 与
03_使用文档/Mac端编译运行说明.md。
引擎（demucs/torch）不在本进程内，由 separate_core 按 venv 路径子进程调用，
因此本启动器极轻、可被 PyInstaller 直接打包（spec 里已排除 torch 全家）。
"""
import os
import sys
import time
import threading
import webbrowser
import socket
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
if getattr(sys, "frozen", False):
    # PyInstaller 打包后：数据文件在 _MEIPASS（onedir 也在包内 Resources/MacOS）
    HERE = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))

APP_DIR = os.path.join(HERE, "_runtime") if os.path.isfile(
    os.path.join(HERE, "_runtime", "app.py")) else HERE
sys.path.insert(0, APP_DIR)

import app  # noqa: E402  （本地服务，纯标准库）


def _open_browser(port):
    url = "http://127.0.0.1:%d" % port
    for _ in range(120):
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except Exception:
            time.sleep(0.5)


def choose_port(preferred):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", preferred))
        except OSError:
            sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    args = [a for a in sys.argv[1:] if a != "--no-browser"]
    port = int(args[0]) if args else 8777
    if sys.platform == "darwin":
        # The native launcher may stop this group, including our own workers.
        try:
            os.setpgid(0, 0)
        except PermissionError:
            pass
        port = choose_port(port)
    port_file = os.environ.get("AMU_PORT_FILE")
    if port_file:
        with open(port_file, "w", encoding="utf-8") as handle:
            handle.write(str(port))
    no_browser = "--no-browser" in sys.argv[1:]
    sys.argv = [sys.argv[0], str(port)]
    if not no_browser:
        threading.Thread(target=_open_browser, args=(port,), daemon=True).start()
    app.main()


if __name__ == "__main__":
    main()
