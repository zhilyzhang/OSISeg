# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from typing import Type
import torch.nn.functional as F


class ComprehensiveAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, rank_dim=32, num_heads=8):
        super(ComprehensiveAdapter, self).__init__()
        # 低秩结构
        self.low_rank_linear1 = nn.Linear(input_dim, rank_dim)
        self.low_rank_linear2 = nn.Linear(rank_dim, output_dim)

        # Self-Attention
        self.self_attention = nn.MultiheadAttention(output_dim, num_heads)

        # LayerNorm和激活函数
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        # 低秩结构变换
        x = self.activation(self.low_rank_linear2(self.activation(self.low_rank_linear1(x))))

        # Self-Attention
        x = x.transpose(0, 1)  # Transpose for self-attention (sequence, batch, feature)
        attn_output, _ = self.self_attention(x, x, x)
        x = x + attn_output
        x = self.norm(x)

        return x.transpose(0, 1)  # Transpose back (batch, sequence, feature)


class AdapterWithAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8):
        super(AdapterWithAttention, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.self_attention = nn.MultiheadAttention(output_dim, num_heads)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear(x))
        x = x.transpose(0, 1)  # Transpose for self-attention (sequence, batch, feature)
        attn_output, _ = self.self_attention(x, x, x)
        x = x + attn_output
        x = self.norm(x)
        return x.transpose(0, 1)  # Transpose back (batch, sequence, feature)


class Adapter(nn.Module):
    def __init__(self, D_features, out_features=None, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        O_features = out_features if out_features is not None else D_features
        self.D_fc2 = nn.Linear(D_hidden_features, O_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class AdapterLen(nn.Module):
    def __init__(self, D_features, out_features=None, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        O_features = out_features if out_features is not None else D_features
        self.D_fc2 = nn.Linear(D_hidden_features, O_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.atrous_blocks = nn.ModuleList()
        for rate in atrous_rates:
            self.atrous_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
            )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(len(atrous_rates) * out_channels + out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        res = []
        for block in self.atrous_blocks:
            res.append(block(x))
        global_avg_pool = self.global_avg_pool(x)
        global_avg_pool = F.interpolate(global_avg_pool, size=res[0].size()[2:], mode='bilinear', align_corners=True)
        res.append(global_avg_pool)

        res = torch.cat(res, dim=1)
        res = self.conv1(res)
        res = self.bn1(res)
        res = self.relu(res)

        return self.dropout(res)


class SpatialAdapterConv(nn.Module):
    def __init__(self, D_features, out_features=None, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()

        # 卷积层捕捉局部特征
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels=D_features, out_channels=D_hidden_features, kernel_size=1, padding=0),
            act_layer(),
            ASPP(in_channels=D_hidden_features, out_channels=D_hidden_features),
            nn.Conv2d(in_channels=D_hidden_features, out_channels=D_features, kernel_size=1, padding=0),
        )

        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        O_features = out_features if out_features is not None else D_features
        self.D_fc2 = nn.Linear(D_hidden_features, O_features)

    def forward(self, x):
        # x is (BT, H, W, D)
        B, H, W, D = x.shape

        # 提取图像局部特征
        x_spatial = x.permute(0, 3, 1, 2)  # reshape to (B, D, H, W)
        # 卷积操作
        x_spatial = self.spatial_conv(x_spatial)
        x_spatial = x_spatial.permute(0, 2, 3, 1)

        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)

        if self.skip_connect:
            x = x + xs + x_spatial
        else:
            x = xs + x_spatial
        return x


class SpatialAdapterNew(nn.Module):
    def __init__(self, D_features, out_features=None, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()

        # 卷积层捕捉局部特征
        # self.conv = nn.Conv2d(in_channels=D_features, out_channels=D_features, kernel_size=1, padding=0)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels=D_features, out_channels=D_hidden_features, kernel_size=1, padding=0),
            act_layer(),
            nn.Conv2d(in_channels=D_hidden_features, out_channels=D_hidden_features, kernel_size=3,
                      padding=1, groups=D_hidden_features),
            act_layer(),
            nn.Conv2d(in_channels=D_hidden_features, out_channels=D_features, kernel_size=1, padding=0)
        )
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, 32, 32))

        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        O_features = out_features if out_features is not None else D_features
        self.D_fc2 = nn.Linear(D_hidden_features, O_features)
        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Parameter):
                nn.init.normal_(m, mean=0, std=0.02)

    def forward(self, x):
        # x is (BT, H, W, D)
        B, H, W, D = x.shape

        # 提取图像局部特征
        x_spatial = x.permute(0, 3, 1, 2)  # reshape to (B, D, H, W)
        # 调整位置编码大小
        pos_embed_resized = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        # 复制位置编码通道以匹配输入的通道数
        pos_embed_resized = pos_embed_resized.repeat(1, D, 1, 1)
        # 卷积操作并添加位置编码
        x_spatial = self.spatial_conv(x_spatial + pos_embed_resized)
        x_spatial = x_spatial.permute(0, 2, 3, 1)

        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)

        if self.skip_connect:
            x = x + xs + x_spatial
        else:
            x = xs + x_spatial
        return x


class SpatialAdapterNewLen(nn.Module):
    def __init__(self, D_features, out_features=None, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()

        # 卷积层捕捉局部特征
        # self.conv = nn.Conv2d(in_channels=D_features, out_channels=D_features, kernel_size=1, padding=0)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels=D_features, out_channels=D_hidden_features, kernel_size=1, padding=0),
            act_layer(),
            nn.Conv2d(in_channels=D_hidden_features, out_channels=D_hidden_features, kernel_size=3,
                      padding=1, groups=D_hidden_features),
            act_layer(),
            nn.Conv2d(in_channels=D_hidden_features, out_channels=D_features, kernel_size=1, padding=0)
        )
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, 32, 32))

        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        O_features = out_features if out_features is not None else D_features
        self.D_fc2 = nn.Linear(D_hidden_features, O_features)
        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Parameter):
                nn.init.normal_(m, mean=0, std=0.02)

    def forward(self, x):
        # x is (BT, H, W, D)
        B, S, D = x.shape
        H = int((S-1)**0.5)
        # 提取图像局部特征
        x_spatial = x[:, 1:, :].reshape((B, H, H, D)).permute(0, 3, 1, 2).contiguous()  # reshape to (B, D, H, W)
        # 调整位置编码大小
        pos_embed_resized = F.interpolate(self.pos_embed, size=(H, H), mode='bilinear', align_corners=False)
        # 复制位置编码通道以匹配输入的通道数
        pos_embed_resized = pos_embed_resized.repeat(1, D, 1, 1)
        # 卷积操作并添加位置编码
        x_spatial = self.spatial_conv(x_spatial + pos_embed_resized)
        x_spatial = x_spatial.permute(0, 2, 3, 1).reshape((B, -1, D)).contiguous()
        x_spatial = torch.cat([x[:, :1, :], x_spatial], dim=1)

        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)

        if self.skip_connect:
            x = x + xs + x_spatial
        else:
            x = xs + x_spatial
        return x


class SpatialAdapterV2(nn.Module):
    def __init__(self, D_features, out_features=None, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()

        # 多尺度卷积层捕捉局部特征
        self.spatial_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=D_features, out_channels=D_hidden_features, kernel_size=1, padding=0),
            act_layer(),
            nn.Conv2d(in_channels=D_hidden_features, out_channels=D_hidden_features, kernel_size=3, padding=1,
                      groups=D_hidden_features),
            nn.Conv2d(in_channels=D_hidden_features, out_channels=D_features, kernel_size=1, padding=0)
        )

        self.spatial_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=D_features, out_channels=D_hidden_features, kernel_size=1, padding=0),
            act_layer(),
            nn.Conv2d(in_channels=D_hidden_features, out_channels=D_hidden_features, kernel_size=5, padding=2,
                      groups=D_hidden_features),
            nn.Conv2d(in_channels=D_hidden_features, out_channels=D_features, kernel_size=1, padding=0)
        )

        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, D_features, 32, 32))

        # 线性层
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        O_features = out_features if out_features is not None else D_features
        self.D_fc2 = nn.Linear(D_hidden_features, O_features)

        # Layer Normalization
        self.norm = nn.LayerNorm(D_features)

    def forward(self, x):
        # x is (B, H, W, D)
        B, H, W, D = x.shape

        # 提取图像局部特征
        x_spatial = x.permute(0, 3, 1, 2)  # reshape to (B, D, H, W)

        # 调整位置编码大小
        pos_embed_resized = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)

        # 多尺度卷积操作并添加位置编码
        x_spatial1 = self.spatial_conv1(x_spatial + pos_embed_resized)
        x_spatial2 = self.spatial_conv2(x_spatial + pos_embed_resized)
        x_spatial = x_spatial1 + x_spatial2
        x_spatial = x_spatial.permute(0, 2, 3, 1)  # reshape back to (B, H, W, D)

        # Transformer MLP部分
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)

        if self.skip_connect:
            x = x + xs + x_spatial
        else:
            x = xs + x_spatial

        # Layer Normalization
        x = self.norm(x)
        return x


class Decorator(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)

        self.linear_based_layer = nn.Sequential(
            nn.Linear(D_features, D_hidden_features),
            act_layer(),
            nn.Linear(D_hidden_features, D_features)
        )
        self.channel_based_layer = nn.Sequential(
            nn.Conv2d(D_features, D_hidden_features, kernel_size=1, bias=False),
            LayerNorm2d(D_hidden_features),
            nn.Conv2d(D_hidden_features, D_hidden_features, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(D_hidden_features),
            act_layer(),
            nn.Conv2d(D_hidden_features, D_features, kernel_size=1, bias=False)
            )

        self.spatial_based_layer = nn.Sequential(
            nn.Conv2d(D_features, D_hidden_features, kernel_size=1, bias=False),
            LayerNorm2d(D_hidden_features),
            nn.Conv2d(D_hidden_features, D_hidden_features, kernel_size=3, padding=1, stride=2, bias=False),
            LayerNorm2d(D_hidden_features),
            act_layer(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(D_hidden_features, D_features, kernel_size=1, bias=False),
        )

    def forward(self, x, H, W):
        # x is (BT, HW+1, D)
        B = x.shape[0]
        xl = self.linear_based_layer(x)

        x_conv = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        xc = self.channel_based_layer(x_conv)
        xc = xc.permute(0, 2, 3, 1).reshape(B, H * W, -1)

        xs = self.spatial_based_layer(x_conv)
        xs = xs.permute(0, 2, 3, 1).reshape(B, H*W, -1)

        xs = (xl + xc + xs) / 3
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
