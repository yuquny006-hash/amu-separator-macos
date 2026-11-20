#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线人声/伴奏 + 乐器分轨分离工具（命令行入口）

引擎：Demucs (Meta, MIT License) —— 代码与预训练权重均为 MIT，可闭源商用。
底层逻辑见 separate_core.py，本文件仅做命令行参数解析。
"""
import os
import sys
import argparse

import separate_core

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="离线人声/伴奏+乐器分轨分离 (Demucs)")
    ap.add_argument("input", help="输入音频 (mp3/wav/flac/m4a)")
    ap.add_argument("-o", "--output", default=os.path.join(HERE, "output"), help="输出目录")
    ap.add_argument("-m", "--mode", default="four",
                    help="分离模式: basic(人声/伴奏 2轨) / four(人声/贝斯/鼓/其他 4轨) / multi(6轨含吉他钢琴)")
    ap.add_argument("-d", "--device", default="cuda", help="cuda / cpu")
    ap.add_argument("-f", "--format", default="wav", help="输出格式: wav / flac")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print("ERROR: 输入文件不存在:", args.input)
        sys.exit(1)

    print(f"[分离] 模式={args.mode}  设备={args.device}  格式={args.format}")
    print(f"[分离] 输入={args.input}")

    res = separate_core.run_separation(
        args.input, args.output, mode=args.mode, device=args.device, fmt=args.format)

    if not res["ok"]:
        print("ERROR: 分离失败 →", res["error"])
        sys.exit(1)

    print(f"[完成] 耗时 {res['elapsed']:.1f}s")
    print(f"[输出目录] {res['outdir']}")
    for s in res["stems"]:
        print(f"   - {s['name']}.{args.format}  ({s['size'] // 1024} KB)")


if __name__ == "__main__":
    main()
