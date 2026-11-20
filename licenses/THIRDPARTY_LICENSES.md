# 第三方依赖许可证聚合（THIRDPARTY LICENSES）

本产品「伴奏分离器」基于 Demucs 引擎构建。以下为运行所依赖的主要第三方组件及其许可证。

**原则**：MIT / BSD / Apache 类许可证允许闭源商用，但要求**保留原版权声明与许可证文本**——
本文件即履行该义务；各组件完整许可证文本见其官方仓库。MPL 为文件级弱 copyleft，亦允许闭源分发。

## 核心引擎
| 组件 | 版本 | 许可证 | 版权方 | 商用义务 |
|---|---|---|---|---|
| demucs | 4.1.0 | MIT | Alexandre Défossez / Meta Platforms, Inc. | 保留 MIT 声明 |
| torch | 2.5.1+cu121 | BSD-3-Clause | Meta Platforms, Inc. | 保留声明 |
| torchaudio | 2.5.1+cu121 | BSD-3-Clause | Meta Platforms, Inc. | 保留声明 |

## P4 人声增强引擎（RoFormer / Mel-Band RoFormer）
| 组件 | 版本 | 许可证 | 版权方 | 商用义务 |
|---|---|---|---|---|
| melband_roformer_vocals.onnx（权重） | — | **MIT** | 权重 © KimberleyJensen (MIT)；实现 © ZFTurbo / Music-Source-Separation-Training (MIT)；内嵌 STFT/iSTFT 模块 © Mixxx Dev Team (MIT, demucs fork) | 保留 MIT 声明 |
| onnxruntime-gpu | 1.20.2 | MIT | Microsoft Corporation | 保留声明 |
| onnxconverter-common | 1.16.0 | MIT | onnx 社区 | 保留声明（仅离线转换 fp16 用，不进推理链路） |

> **版权结论**：RoFormer 人声权重整链 MIT，可闭源商用。已**排除** viperx 的
> BS-RoFormer（权重授权含糊、训练数据含 MUSDB18HQ 研究用途）——绝不使用任何带
> CC BY-NC / 仅研究声明的权重。

## P5 人声和声分离引擎（VR Architecture / BVE）
| 组件 | 版本 | 许可证 | 版权方 | 商用义务 |
|---|---|---|---|---|
| UVR-BVE-4B_SN-44100-1.pth（权重） | — | **MIT** | 权重 © Anjok07 / KimberleyJSN (MIT)；VR Architecture 架构 © KUIELab (MIT)；来源 TRvlvr/model_repo（UVR 官方模型分发仓，MIT） | 保留 MIT 声明 |
| audio-separator | 0.47.0 | MIT | audio-separator 社区 | 保留声明 |
| nets-new / spec_utils（BVE 推理实现） | — | MIT | UVR project (MIT) | 保留声明 |

> **版权结论**：BVE-4B 权重与 VR 架构整链 MIT，可闭源商用；随包内置分发无授权障碍。
> **已排除**：mel_band_roformer_karaoke（aufr33 / viperx，权重授权含糊、许可不明）——
> 绝不使用任何带 CC BY-NC / 仅研究声明的权重。
>
> **模型指纹**：随包权重 MD5 = `3021a16d67bc11f4860f30eccc651558`。
> 注：audio-separator 官方参数表另登记有 `944950a9...` 一条，那是**另一个发布版本**的指纹，
> 与本包内文件不对属，属正常现象。
>
> **实现说明**：本模块只用到上述 MIT 组件的推理路径；界面不出现引擎名 / 模型名 / 许可证名。
> 和声轨采用残差法（`和声 = 高通输入 − 高通主唱`），两轨相加严格等于原人声轨。

## 自研组件（本项目代码，MIT）
| 组件 | 许可证 | 说明 |
|---|---|---|
| fuse_vocals.py | MIT（自研） | 人声融合：Demucs 干人声 × α + RoFormer 自然人声 × (1−α) 加权融合，纯 numpy + soundfile，无外部许可证风险 |
| separate_combined.py | MIT（自研） | 组合分离编排：Demucs(htdemucs_ft 或 htdemucs_6s) 乐器轨 + RoFormer 人声，支持 4/6-stem 与融合开关 |
| harmony_separate.py | MIT（自研） | 和声拆分编排：80Hz 高通 → BVE 推理 → 残差法得到主唱/和声，两轨统一增益、严格守恒 |
| app.py / web/ | MIT（自研） | 本地 Web 服务与前端，纯标准库，避开 GPL 壳 |

> **组合策略**：
> - 4-stem（C 档）：Demucs `htdemucs_ft`（鼓/贝斯/其他）+ RoFormer 人声
> - 6-stem（B 档）：Demucs `htdemucs_6s`（鼓/贝斯/其他/吉他/钢琴）+ RoFormer 人声
> - 融合开关：额外产出 `vocals_dry.wav`（Demucs 干）+ `vocals_fused.wav`（加权融合）
> - `htdemucs_ft` / `htdemucs_6s` 均为 Meta MIT 预训练权重，零新许可风险

## Demucs 直接依赖
| 组件 | 版本 | 许可证 | 版权方 | 商用义务 |
|---|---|---|---|---|
| einops | — | MIT | Alex Rogozhnikov | 保留声明 |
| huggingface-hub | 1.31.0 | Apache-2.0 | Hugging Face, Inc. | 保留声明 + NOTICE |
| julius | — | MIT | Facebook AI Research | 保留声明 |
| pyyaml | — | MIT | yaml.org / Kirill Simonov | 保留声明 |
| safetensors | — | Apache-2.0 | Hugging Face, Inc. | 保留声明 + NOTICE |
| sphn | — | MIT | Alexandre Défossez | 保留声明 |
| tqdm | 4.70.1 | MPL-2.0 | tqdm 作者 | 弱 copyleft（文件级），可闭源；保留通知 |

## 本项目附加依赖（音频时长读取）
| 组件 | 版本 | 许可证 | 版权方 | 商用义务 |
|---|---|---|---|---|
| numpy | 2.5.3 | BSD-3-Clause | NumPy Developers | 保留声明 |
| soundfile | 0.14.0 | BSD-3-Clause | Bastian Bechtold | 保留声明 |

## 和声模块附加依赖（audio-separator 依赖链）
| 组件 | 许可证 | 版权方 | 商用义务 |
|---|---|---|---|
| librosa | ISC | librosa 作者 | 保留声明 |
| scipy | BSD-3-Clause | SciPy Developers | 保留声明 |
| onnxruntime / pydub / tqdm 等 | MIT / 各自 | 各自上游 | 保留声明 |

> 上述均为宽松许可证，允许闭源商用。完整清单以随包环境 `pip list` 为准。

## ⚠️ 需重点关注项
- **lameenc / LAME（MP3 编码器）**：随 demucs 一同安装。LAME 为 LGPL-2.1 且涉及 MP3 专利。
  **本产品仅输出 WAV / FLAC，禁用 MP3，故该组件不会被激活。**
  分发前建议从运行环境移除以彻底规避 MP3 专利与 LGPL 义务：
  `.venv312\Scripts\pip.exe uninstall lameenc`
- **av / PyAV（LGPL-2.1）**：仅开发期检查脚本（check_quality.py）使用，未进入分离主链路，
  **不纳入最终分发包**。
- **CUDA 运行时**：torch cu121 含 NVIDIA CUDA 再分发组件，遵循 NVIDIA CUDA 许可
  （允许再分发，须保留 NVIDIA 声明）。

## 训练数据
- **MUSDB18（CC BY-NC 4.0，仅非商业）**：本产品直接使用 Demucs 官方 **MIT 预训练权重**，
  **不重新训练**，因此不受 MUSDB18 非商业条款约束。

## 分离结果版权归属
分离出的各音轨是原始录音的「衍生物」，其版权仍归原录音的权利人所有。
本产品（MIT）仅覆盖**代码与模型权重**，不授予任何录音版权，也不转移任何第三方的录音权利。
