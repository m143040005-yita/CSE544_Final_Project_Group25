# fix_ctta_ckpt.py
import torch
from pathlib import Path


def main():
    # 這三個路徑請確認一下
    base = Path("runs/train/E0_coco_yolov5s_baseline/weights/best.pt")  # 原 YOLOv5 checkpoint
    ctta = Path("best_ctta.pt")                                         # CTTA 之後存的 state_dict
    out  = Path("best_ctta_fixed.pt")                                   # 要輸出的新檔案

    print(f"Loading base checkpoint from: {base}")
    # ★ 關鍵：weights_only=False（關掉新版 PyTorch 的安全限制）
    base_ckpt = torch.load(base, map_location="cpu", weights_only=False)

    print(f"Loading CTTA state from: {ctta}")
    ctta_ckpt = torch.load(ctta, map_location="cpu")  # 這個是我們自己存的 dict

    # base_ckpt['model'] 應該是 nn.Module（YOLOv5 的 DetectionModel）
    model = base_ckpt["model"]

    # ctta_ckpt["model"] 是 state_dict（OrderedDict）
    state_dict = ctta_ckpt["model"]

    print("Loading CTTA state_dict into base model (strict=False)...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("  missing keys:", len(missing))
    print("  unexpected keys:", len(unexpected))

    base_ckpt["model"] = model

    print(f"Saving fixed checkpoint to: {out}")
    torch.save(base_ckpt, out)
    print("Done.")


if __name__ == "__main__":
    main()

