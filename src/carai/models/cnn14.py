"""Minimal PANNs CNN14 implementation, adapted from
https://github.com/qiuqiangkong/audioset_tagging_cnn (MIT License).

Strips the AudioSet 527-class head; returns 2048-dim embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer: nn.Module) -> None:
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn: nn.BatchNorm2d) -> None:
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        for m in [self.conv1, self.conv2]:
            init_layer(m)
        for m in [self.bn1, self.bn2]:
            init_bn(m)

    def forward(self, x: torch.Tensor, pool_size=(2, 2)) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)


class Cnn14(nn.Module):
    """Logmel-input CNN14. Input: (B, T_frames, 64_mels). Output: 2048-dim."""

    def __init__(self) -> None:
        super().__init__()
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048)
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        x = logmel.unsqueeze(1)  # (B, 1, T, 64)
        x = x.transpose(1, 3)  # (B, 64, T, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)  # (B, 1, T, 64)
        x = self.conv_block1(x, (2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, (2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, (2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, (2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, (2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, (1, 1))
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)  # (B, C, T)
        x1 = F.max_pool1d(x, kernel_size=x.shape[-1]).flatten(1)
        x2 = F.avg_pool1d(x, kernel_size=x.shape[-1]).flatten(1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        return F.relu_(self.fc1(x))  # (B, 2048)
