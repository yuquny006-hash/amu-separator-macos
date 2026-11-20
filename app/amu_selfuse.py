#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""自用模式开关（只读校验）。

设计要点
--------
* 开关文件放在「程序目录之外」的用户配置目录（%APPDATA%/AmuTools/
  或 ~/Library/Application Support/AmuTools），因此对外交付的安装包
  天然不含开关，客户安装后一律走正常的登录 / 积分流程。
* 文件内容经 DPAPI 加密（Windows，绑定当前 Windows 用户），载荷里还带
  本机指纹。跨机器、跨用户直接复制文件无效；明文造一个同名文件也无效。
* 本模块只提供「校验」，不提供「生成」能力。生成 / 关闭开关的工具留在
  04_工具 目录，不随软件发布，也不会被打进交付 zip。

这样「自用免积分」与「对外商业版」共用同一份程序、同一份源码，
不再需要维护 1.0 / 2.0 或商业版 / 内部版两套代码。
"""
import os
import sys
import json
import hashlib

import amu_login_gate

MAGIC = "AMU-SELFUSE-COMPLETE-v1"
FILE_NAME = ".amu_selfuse.bin"


def switch_path() -> str:
    """开关文件的绝对路径（程序目录之外）。"""
    return os.path.join(amu_login_gate._config_dir(), FILE_NAME)


def machine_fingerprint() -> str:
    """本机指纹：MAC + 主机名 + 用户目录，取 SHA256 前 32 位。"""
    import uuid
    import platform
    raw = "%s|%s|%s" % (uuid.getnode(), platform.node(),
                        os.path.expanduser("~"))
    return hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest()[:32]


def is_enabled() -> bool:
    """本机是否已开启自用模式。任何异常一律按「未开启」处理，绝不影响正常启动。"""
    try:
        path = switch_path()
        if not os.path.isfile(path):
            return False
        with open(path, "rb") as f:
            payload = f.read()
        if not payload:
            return False
        if sys.platform == "win32":
            plain = amu_login_gate._dpapi_unprotect(payload)
        else:
            plain = amu_login_gate._xor(payload)
        obj = json.loads(plain.decode("utf-8", "ignore"))
        return (obj.get("magic") == MAGIC
                and obj.get("fp") == machine_fingerprint()
                and obj.get("on") is True)
    except Exception:
        return False
