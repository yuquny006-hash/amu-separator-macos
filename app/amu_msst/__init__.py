# -*- coding: utf-8 -*-
r"""BS-RoFormer 最小推理运行时（用于 53 轨乐器分离）

来源：ZFTurbo / Music-Source-Separation-Training（MIT License，见 LICENSE_MSST.txt）
    https://github.com/ZFTurbo/Music-Source-Separation-Training

这里只保留推理所必需的 `BSRoformer` 与其注意力实现 `Attend`，
不引入仓库其它模型（mamba / conformer / upernet 等）及其额外依赖。

对原文件做的两处改动（务必保留，升级官方代码时要重新打上）：
1. `from models.bs_roformer.attend import Attend` → `from .attend import Attend`
   （脱离 MSST 仓库目录结构，同时避免与项目内的 `models/` 目录撞名）
2. `attend.py` 增加 torch<2.6 的 `sdpa_kernel(set_priority=...)` 兼容包装
   （本机 torch 2.5.1 无此参数，否则注意力调用直接 TypeError）
"""
from .bs_roformer import BSRoformer

__all__ = ["BSRoformer"]
__msst_license__ = "MIT"
__msst_source__ = "https://github.com/ZFTurbo/Music-Source-Separation-Training"
