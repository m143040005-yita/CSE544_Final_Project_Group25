import torch
import torch.nn as nn
import torch.nn.functional as F

# 用來偵測 YOLOv5 的 Conv block（必要）
from models.common import Conv as YoloConv


# ============================================================
# Domain-Aware Adapter
#   - invertible branch (Finv)
#   - spatial branch (Fsp)
# ============================================================
class DomainAwareAdapter(nn.Module):
    def __init__(self, channels: int, rank_inv: int = 8, rank_sp: int = 8):
        """
        channels: Conv block 輸出的 channel 數
        rank_inv: Finv bottleneck rank
        rank_sp: Fsp bottleneck rank
        """
        super().__init__()

        r_inv = max(1, rank_inv)
        r_sp = max(1, rank_sp)

        # ---- Invertible 分支 Finv ----
        self.down_inv = nn.Conv2d(channels, r_inv, kernel_size=1, bias=False)
        self.up_inv   = nn.Conv2d(r_inv, channels, kernel_size=1, bias=False)

        # ---- Spatial 分支 Fsp ----
        self.down_sp = nn.Conv2d(channels, r_sp, kernel_size=1, bias=False)
        self.up_sp   = nn.Conv2d(r_sp, channels, kernel_size=1, bias=False)

        # 融合時的 scale（初始化為 0）
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: feature map, [B, C, H, W]
        """
        # ------------------------------------------------
        # 🛡️ 最後防線：如果 adapter 權重還沒搬到 GPU，就自動跟上
        # ------------------------------------------------
        if self.down_inv.weight.device != x.device:
            self.to(x.device)

        # ---- invertible branch ----
        z_inv = F.relu(self.down_inv(x))
        y_inv = self.up_inv(z_inv)

        # ---- spatial branch ----
        z_sp = F.relu(self.down_sp(x))
        y_sp = self.up_sp(z_sp)

        # ---- adapter output ----
        return x + self.scale * (y_inv + y_sp)


# ============================================================
# 掛上 adapters（只對 YOLOv5 Conv block）
# ============================================================
def add_adapters_to_model(model: nn.Module, min_channels: int = 128, rank: int = 8):
    """
    在 YOLOv5 backbone/neck/head 中所有 Conv block 上掛上 DomainAwareAdapter
    若該層 out_channels < min_channels，則略過（太小沒必要）
    """
    adapters = []

    for m in model.modules():
        if isinstance(m, YoloConv):

            out_ch = m.conv.out_channels

            if out_ch >= min_channels:

                # 避免重複掛
                if not hasattr(m, "adapter"):
                    m.adapter = DomainAwareAdapter(
                        channels=out_ch,
                        rank_inv=rank,
                        rank_sp=rank,
                    )
                    adapters.append(m.adapter)

    return adapters


# ============================================================
# 只讓 adapter 可以訓練
# ============================================================
def set_adapter_trainable(model: nn.Module):
    """ Freeze YOLO backbone，啟動 adapter 參數 """
    for p in model.parameters():
        p.requires_grad = False

    for m in model.modules():
        if isinstance(m, DomainAwareAdapter):
            for p in m.parameters():
                p.requires_grad = True


# ============================================================
# 拿出所有 adapters（訓練時用）
# ============================================================
def get_all_adapters(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, DomainAwareAdapter)]


# ============================================================
# Orthogonality Loss（讓 Finv 與 Fsp 正交）
# ============================================================
def adapter_orth_loss(adapters):
    """
    L_orth = mean(|| W_inv * W_sp^T ||^2)
    確保兩個低秩子空間分離（避免 collapse）
    """
    loss = 0.0

    for ad in adapters:
        # 取 bottleneck weights
        W_inv = ad.down_inv.weight.view(ad.down_inv.out_channels, -1)
        W_sp  = ad.down_sp.weight.view(ad.down_sp.out_channels, -1)

        # 兩者的 Gram（交叉相關）
        G = W_inv @ W_sp.t()

        # Frobenius norm
        loss += (G.pow(2).mean())

    return loss / max(1, len(adapters))

