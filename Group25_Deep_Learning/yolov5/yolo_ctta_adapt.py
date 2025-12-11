# yolo_ctta_adapt.py

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from utils.dataloaders import create_dataloader
from utils.general import check_dataset, check_yaml
from utils.torch_utils import select_device

from models.common import Conv as YoloConv  # 👈 要有這行

from ctta_adapter import (
    add_adapters_to_model,
    set_adapter_trainable,
    get_all_adapters,
    adapter_orth_loss,
)


def entropy_loss_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, N, 5+nc]，最後 nc 維是 class logits。
    """
    cls_logits = logits[..., 5:]
    prob = cls_logits.sigmoid()
    eps = 1e-8
    ent = -(prob * (prob + eps).log()).sum(dim=-1)
    return ent.mean()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/coco.yaml", help="dataset .yaml path")
    parser.add_argument("--weights", type=str, default="yolov5s.pt", help="YOLOv5 weights (ckpt)")
    parser.add_argument("--imgsz", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--batch-size", type=int, default=16, help="CTTA batch size")
    parser.add_argument("--epochs", type=int, default=1, help="CTTA epochs over target data")
    parser.add_argument("--lr", type=float, default=1e-4, help="CTTA learning rate")
    parser.add_argument("--min-ch", type=int, default=128, help="min out_channels to adapt")
    parser.add_argument("--rank", type=int, default=8, help="adapter bottleneck rank")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--cache", type=str, default="ram", help="dataloader cache: ram/disk")
    parser.add_argument("--save", action="store_true", help="save adapted weights")
    parser.add_argument("--save-path", type=str, default="best_ctta_da.pt", help="save path")
    return parser.parse_args()


def load_training_model(weights: str):
    """
    直接從 best.pt 裡拿出訓練用 DetectionModel。
    """
    print(f"[CTTA] Loading training checkpoint from: {weights}")
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model = ckpt["model"]
        print("[CTTA] Found 'model' in checkpoint dict.")
    elif isinstance(ckpt, nn.Module):
        model = ckpt
        print("[CTTA] Checkpoint is a raw nn.Module.")
    else:
        raise RuntimeError(f"Unexpected checkpoint type: {type(ckpt)}")

    # 補上可能缺的 fused 屬性（因為你有改過 models/common.py）
    for m in model.modules():
        if isinstance(m, YoloConv) and not hasattr(m, "fused"):
            m.fused = False

    model = model.float()
    model.train()

    nc = ckpt.get("nc", None) if isinstance(ckpt, dict) else None
    names = ckpt.get("names", None) if isinstance(ckpt, dict) else None

    return model, nc, names


def main():
    opt = parse_opt()
    device = select_device(opt.device)

    # 1. dataloader（這裡是 coco_c 的 motion blur val）
    data_dict = check_dataset(check_yaml(opt.data))
    val_path = data_dict["val"]

    dataloader = create_dataloader(
        val_path,
        opt.imgsz,
        opt.batch_size,
        stride=32,
        single_cls=False,
        pad=0.5,
        rect=False,
        prefix="CTTA: ",
        workers=4,
        shuffle=True,
        cache=opt.cache,
    )[0]

    ds_nc = int(data_dict["nc"])
    ds_names = data_dict["names"]

    # 2. 從訓練 ckpt 拿模型（先不搬到 GPU，等 adapter 掛完一起搬）
    model, ckpt_nc, ckpt_names = load_training_model(opt.weights)

    nc = ckpt_nc if ckpt_nc is not None else ds_nc
    names = ckpt_names if ckpt_names is not None else ds_names

    # 3. 掛 adapters
    add_adapters_to_model(model, min_channels=opt.min_ch, rank=opt.rank)
    adapters = get_all_adapters(model)
    print(f"Added {len(adapters)} domain-aware adapters to model.")

    # 4. 這裡才一次搬到 GPU（包含剛掛好的 adapters）
    model.to(device)
    set_adapter_trainable(model)

    # 5. optimizer（只會抓 requires_grad=True，也就是 adapters）
    adapter_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of adapter parameters: {sum(p.numel() for p in adapter_params)}")
    optimizer = torch.optim.SGD(
        adapter_params,
        lr=opt.lr,
        momentum=0.9,
        weight_decay=1e-4,
    )

    lambda_ent = 0.2
    lambda_orth = 1.0

    # 6. CTTA loop
    for epoch in range(opt.epochs):
        print(f"\n[CTTA] Epoch {epoch + 1}/{opt.epochs}")
        for i, (imgs, targets, paths, shapes) in enumerate(dataloader):
            imgs = imgs.to(device, non_blocking=True).float() / 255.0

            optimizer.zero_grad()

            # 直接用訓練用 DetectionModel 的 forward，不再經過 DetectMultiBackend
            out = model(imgs)
            if isinstance(out, (list, tuple)):
                preds = out[0]
            else:
                preds = out

            ent_loss = entropy_loss_from_logits(preds)
            orth_loss = adapter_orth_loss(adapters)
            loss = lambda_ent * ent_loss + lambda_orth * orth_loss

            if i == 0:
                print(
                    "DEBUG requires_grad:",
                    "loss:", loss.requires_grad,
                    "ent:", ent_loss.requires_grad,
                    "orth:", orth_loss.requires_grad,
                    "preds:", preds.requires_grad,
                )

            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                print(
                    f"  iter {i + 1}/{len(dataloader)}  "
                    f"ent = {ent_loss.item():.4f}  orth = {orth_loss.item():.4f}"
                )

    # 7. 存檔
    if opt.save:
        save_path = Path(opt.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        ckpt_out = {
            "model": model,
            "nc": nc,
            "names": names,
        }
        torch.save(ckpt_out, save_path)
        print(f"\n[CTTA] Adapted YOLO checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()

