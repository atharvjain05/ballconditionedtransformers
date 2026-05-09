"""Small heatmap detector for SoccerNet player+ball tracking."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HeatmapBoxDetector(nn.Module):
    """A compact stride-4 detector with center heatmaps and box-size regression."""

    def __init__(self, num_classes: int = 2, base_channels: int = 42) -> None:
        super().__init__()
        c = base_channels
        self.stem = ConvBlock(3, c, stride=2)
        self.stage2 = ConvBlock(c, c * 2, stride=2)
        self.stage3 = ConvBlock(c * 2, c * 4, stride=2)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.SiLU(inplace=True),
        )
        self.fuse = ConvBlock(c * 4, c * 2)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, num_classes, kernel_size=1),
        )
        self.wh_head = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, 2, kernel_size=1),
            nn.Sigmoid(),
        )
        # Start with low center probabilities to reduce early false positives.
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x1 = self.stem(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        up = self.up(x3)
        feat = self.fuse(torch.cat([up, x2], dim=1))
        return {
            "heatmap_logits": self.heatmap_head(feat),
            "wh": self.wh_head(feat),
        }


def detection_loss(
    pred: dict[str, torch.Tensor],
    target_heatmap: torch.Tensor,
    target_wh: torch.Tensor,
    wh_mask: torch.Tensor,
    box_weight: float = 5.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Focal-style center loss plus masked L1 box-size loss."""
    heatmap_logits = pred["heatmap_logits"]
    pred_wh = pred["wh"]

    heatmap_loss = _modified_focal_loss(heatmap_logits, target_heatmap)
    mask = wh_mask.expand_as(target_wh)
    wh_loss = F.l1_loss(pred_wh * mask, target_wh * mask, reduction="sum") / mask.sum().clamp_min(
        1.0
    )
    loss = heatmap_loss + box_weight * wh_loss
    return loss, {
        "loss": float(loss.detach()),
        "heatmap_loss": float(heatmap_loss.detach()),
        "wh_loss": float(wh_loss.detach()),
    }


def _modified_focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CenterNet-style focal loss for Gaussian heatmap targets."""
    pred = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
    pos_inds = target.eq(1.0).to(logits.dtype)
    neg_inds = target.lt(1.0).to(logits.dtype)
    neg_weights = torch.pow(1.0 - target, 4)

    pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, 2) * pos_inds
    neg_loss = -torch.log(1.0 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    num_pos = pos_inds.sum().clamp_min(1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos
