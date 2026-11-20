# Amu 音频分离 · macOS 构建仓库（公开）

本仓库只用于 **GitHub Actions 编译 macOS 版**，不含用户音频、登录缓存、Windows 虚拟环境或任何密钥。
产品本体 **完全免费**：无需登录、无需注册、无积分、不限次数、不联网验活。

- 版本：完整版 2.0.0（basic / four / multi / instruments / pro 五模式，58 种乐器，全分离 Pro）
- 架构：Apple Silicon (`macos-15`) 与 Intel (`macos-15-intel`) 分别构建
- 模型：Demucs / BVE 权重由 `build/fetch_models.py` 从本仓库 release 的哈希锁定分片下载并校验；
  Mega53 体积较大，沿用「首次使用时按需下载」，不随包内置
- 触发：两个 workflow 均为 `workflow_dispatch` 手动触发
- 版权：第三方许可证全文见 `licenses/`。Demucs 代码与权重为 MIT；
  其余组件的授权状态见 `licenses/技术选型与版权尽调.md`

> 这是构建用的源码快照，不是发行包。下载成品请前往 Amu Music 官网。
