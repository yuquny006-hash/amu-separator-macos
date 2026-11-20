本目录存放运行时模型权重，不进源码。

发布时由 04_工具/build_release.py 负责装配：
  bve/      UVR-BVE-4B_SN-44100-1.pth   和声分离（多轨模式用）约 214 MB
  mega53/   mvsep_mega_53.ckpt          乐器识别 53 轨，约 1306 MB

mega53 按「方案 A」不预置进安装包：用户首次使用乐器提取时，
由 mega53_download.py 一键下载（默认落到 %APPDATA%\AmuTools\models\mega53，
避免装在 Program Files 时目录不可写）。
