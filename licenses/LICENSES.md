# 第三方许可证与版权合规说明（伴奏分离器）

> 本文档用于商业化闭源分发时，履行开源依赖的许可证义务（保留声明 + NOTICE）。
> 本工具引擎为 **Demucs (Meta)**，代码与预训练模型权重均为 **MIT License**，可闭源、可商用。

---

## 1. 核心引擎

| 组件 | 许可证 | 商用 | 说明 |
|------|--------|------|------|
| Demucs (adefossez/demucs) | MIT | ✅ 可闭源商用 | 代码 + 预训练权重（htdemucs / htdemucs_ft / htdemucs_6s）均 MIT |
| Spleeter (Deezer) | MIT | ✅ | 备选，质量较弱未作为主力 |

---

## 2. Python 运行时依赖（均为 MIT / BSD / Apache，无 copyleft 污染）

| 包 | 许可证 | 用途 |
|----|--------|------|
| torch / torchaudio 2.5.1+cu121 | BSD-3-Clause | 推理引擎（CUDA 版，仅本地运行，不分发权重外的东西） |
| numpy 2.5.3 | BSD-3-Clause | 音频数组处理 |
| av (PyAV) 18.1.0 | BSD-3-Clause | 读取 mp3/wav/flac 等格式 |
| julius 0.2.8 | MIT | 音频前端 |
| einops 0.8.2 | MIT | 张量操作 |
| lameenc 1.8.4 | LGPL-1.1？见下方注 | MP3 编码（若输出 MP3） |
| huggingface_hub | Apache-2.0 | 模型权重下载 |
| tqdm / click / colorama / einops 等 | 各自 MIT | 工具链 |

> **注**：默认输出 **WAV / FLAC**（无专利费）。若需输出 MP3，lameenc 底层为 LAME。
> **规避建议**：分发时默认输出 WAV/FLAC/Opus，避开 MP3/AAC 的专利授权坑（尤其 AAC）。

---

## 3. 训练数据（MUSDB18）声明

Demucs 的预训练权重使用 MUSDB18 训练。MUSDB18 数据集本身为 **CC BY-NC（仅非商业）**，
但 **Meta 已发布的预训练权重以 MIT 授权**——即「使用现成权重做分离」不受 MUSDB18 的
NC 限制。只有在「你自己用 MUSDB18 重新训练模型」时才受 NC 约束。本工具不重训，安全。

---

## 4. 用户侧版权责任（ToS 要点，待法务润色）

1. 本软件为**离线本地工具**，音频不上传任何服务器，由用户本人在自己设备上处理。
2. 用户须保证：输入本工具的音频为其**自有作品**或已获权利人合法授权。
3. 分离产生的各音轨是原始录音的**衍生物**，版权归原权利人，不属于本软件或用户。
4. 用户不得将分离结果用于侵犯他人版权之用途（如未授权传播、商用翻唱获利等）。
5. 软件提供方不对用户违规使用导致的版权纠纷承担责任。

---

## 5. 分发义务（开发完成前 checklist）

- [ ] 随安装包附 `LICENSES.md`（本文件）
- [ ] 保留 Demucs 的 MIT LICENSE 原文（NOTICE 文件）
- [ ] 保留 torch / numpy / av 等的许可证文本（可聚合为 THIRDPARTY_LICENSES）
- [ ] 默认输出格式设为 WAV/FLAC（避开 MP3/AAC 专利）
- [ ] 不在代码中引入任何 GPL/AGPL 组件（如部分 ComfyUI 节点是 GPLv3，禁用）
- [ ] UI 内放置「版权须知」弹窗/页（引用第 4 节）
