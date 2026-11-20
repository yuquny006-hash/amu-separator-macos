# -*- coding: utf-8 -*-
"""Demucs Python API worker：统一增益保存 + 生成「原曲 − 人声」伴奏轨。

由 separate_core.run_separation 用 .venv312\\Scripts\\python.exe 调用（本进程需 torch / demucs）。
优势：
  1. 模型返回的各轨**未做任何限幅**，由本脚本统一施加**同一个**缩放因子保存 →
     各轨相加 = 原曲 × scale，导入 DAW 后电平关系与原始完全一致（根治分轨电平错位）。
  2. 额外生成 no_vocals = 原曲 − 人声（原曲与 vocals 同帧对齐，纯乐器段 100% 原样）。
最后一行为 JSON 结果（stdout），其余信息走 stderr。
"""
import argparse, json, os, sys, time

# ---- 离线支持：模型权重指向包内 _hfcache（必须在 import torch / demucs 之前设置）
_HF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_hfcache")
if os.path.isdir(_HF):
    os.environ["HF_HOME"] = _HF
    os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)   # 最终轨道目录（<model>/<base>/）
    ap.add_argument("--model", default="htdemucs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shifts", type=int, default=2)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--segment", type=int, default=0)   # 0 = 模型默认
    ap.add_argument("--sources", default="")            # 要保存的模型轨（逗号分隔，空=全部）
    ap.add_argument("--minus-vocals", type=int, default=1)
    ap.add_argument("--fmt", default="wav")
    ap.add_argument("--jobs", type=int, default=0)
    a = ap.parse_args()

    import torch as th
    from demucs.api import Separator
    from demucs.audio import save_audio

    t0 = time.time()
    try:
        seg = a.segment if a.segment > 0 else None
        sep = Separator(model=a.model, device=a.device, shifts=a.shifts,
                        overlap=a.overlap, segment=seg, split=True,
                        jobs=a.jobs, progress=False)
        origin, res = sep.separate_audio_file(a.input)
        sr = sep.samplerate

        keep = [s for s in (a.sources.split(",") if a.sources else list(res.keys())) if s in res]
        tensors = {name: res[name] for name in keep}

        if a.minus_vocals and "vocals" in res:
            tensors["no_vocals"] = origin - res["vocals"]

        # 统一增益：取「所有待存轨 + 模型原生各轨相加」中的全局峰值，整体乘同一因子
        mix = th.stack([res[k] for k in keep]).sum(0)
        allmax = max([t.abs().max().item() for t in tensors.values()] +
                     [mix.abs().max().item()])
        scale = 1.0 / max(1.01 * allmax, 1.0)

        os.makedirs(a.outdir, exist_ok=True)
        ext = "flac" if a.fmt == "flac" else "wav"
        out = []
        for name, x in tensors.items():
            path = os.path.join(a.outdir, name + "." + ext)
            save_audio(x * scale, path, sr, clip="none", bits_per_sample=16)
            out.append({"name": name, "path": path, "size": os.path.getsize(path)})

        # 自检：各轨相加与原曲的吻合度（仅 informational，写 stderr）
        saved = {k: tensors[k] for k in keep}
        if saved:
            recon = th.stack(list(saved.values())).sum(0) * scale
            corr = float(th.corrcoef(th.stack([recon.mean(0), origin.mean(0)]))[0, 1])
            print("自检 分轨相加 vs 原曲 相关=%.4f scale=%.4f" % (corr, scale), file=sys.stderr)

        print(json.dumps({"ok": True, "stems": out, "samplerate": sr,
                          "elapsed": round(time.time() - t0, 1)}, ensure_ascii=False))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(json.dumps({"ok": False, "error": repr(e)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
