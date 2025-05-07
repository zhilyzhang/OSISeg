import torch
import torch.nn as nn

from typing import Type
from networks.sam.common import Adapter, LayerNorm2d
import math
import warnings


class AdapterFusedFeature(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(AdapterFusedFeature, self).__init__()

        self.adapter1 = Adapter(D_features=in_channels, skip_connect=False)
        self.adapter2 = Adapter(D_features=in_channels, skip_connect=False)
        self.adapter3 = Adapter(D_features=in_channels, skip_connect=False)
        self.adapter4 = Adapter(D_features=in_channels, skip_connect=False)

        self.feature_fuse = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, bias=False),
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel),
        )

        self.final_fuse = nn.Sequential(
            nn.Conv2d(out_channel * 2, out_channel, kernel_size=1, bias=False),
            LayerNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel))
        self.apply(self._init_weights)

    def forward(self, list_features):
        f1, f2, f3, f4, maps = list_features
        B, _, H, W = f1.shape
        f1 = f1.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        f1 = self.adapter1(f1) + f1
        f1 = f1.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        f2 = f2.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        f2 = self.adapter2(f2) + f2
        f2 = f2.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        f3 = f3.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        f3 = self.adapter3(f3) + f3
        f3 = f3.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        f4 = f4.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        f4 = self.adapter4(f4) + f4
        f4 = f4.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        fused_features = torch.cat([f1, f2, f3, f4], dim=1)
        fused_features = self.feature_fuse(fused_features)

        final_features = torch.cat([maps, maps+fused_features], dim=1)
        final_features = self.final_fuse(final_features)

        return final_features

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class AdapterFeatureDecoration(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(AdapterFeatureDecoration, self).__init__()

        self.adapter1 = Adapter(D_features=in_channels, skip_connect=False)
        self.adapter2 = Adapter(D_features=in_channels, skip_connect=False)
        self.adapter3 = Adapter(D_features=in_channels, skip_connect=False)
        self.adapter4 = Adapter(D_features=in_channels, skip_connect=False)

        self.feature_decorated1 = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, bias=False),
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel),
        )

        self.feature_decorated2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=False),
            LayerNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel))

        self.scale_param1 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.01)
        self.scale_param2 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.01)

        self.apply(self._init_weights)

    def forward(self, list_features):
        f1, f2, f3, f4, maps = list_features
        B, _, H, W = f1.shape
        f1 = f1.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        f1 = self.adapter1(f1) + f1
        f1 = f1.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        f2 = f2.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        f2 = self.adapter2(f2) + f2
        f2 = f2.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        f3 = f3.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        f3 = self.adapter3(f3) + f3
        f3 = f3.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        f4 = f4.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        f4 = self.adapter4(f4) + f4
        f4 = f4.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        fused_features = torch.cat([f1, f2, f3, f4], dim=1)
        decorated_feature1 = self.feature_decorated1(fused_features)

        decorated_feature2 = self.feature_decorated2(maps)

        final_features = maps + decorated_feature1 * self.scale_param1 + decorated_feature2 * self.scale_param2
        return final_features

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class AdapterFeatureDecorationUpdate(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(AdapterFeatureDecorationUpdate, self).__init__()

        self.adapter1 = Adapter(D_features=in_channels, skip_connect=False)
        self.res_layer12 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(in_channels))
        self.adapter2 = Adapter(D_features=in_channels, skip_connect=False)
        self.res_layer23 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(in_channels))
        self.adapter3 = Adapter(D_features=in_channels, skip_connect=False)
        self.res_layer34 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(in_channels))
        self.adapter4 = Adapter(D_features=in_channels, skip_connect=False)

        self.feature_decorated1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size=1, bias=False),
            LayerNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel)
        )

        self.feature_decorated2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=False),
            LayerNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel))

        self.scale_param1 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.01)
        self.scale_param2 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.01)

        self.apply(self._init_weights)

    def forward(self, list_features):
        f1, f2, f3, f4, maps = list_features
        B, _, H, W = f1.shape
        f1 = f1.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        f1 = self.adapter1(f1) + f1
        f1 = f1.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        f2 = self.res_layer12(f1) + f2
        f2 = f2.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        f2 = self.adapter2(f2) + f2
        f2 = f2.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        f3 = self.res_layer23(f2) + f3
        f3 = f3.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        f3 = self.adapter3(f3) + f3
        f3 = f3.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        f4 = self.res_layer34(f3) + f4
        f4 = f4.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        f4 = self.adapter4(f4) + f4
        f4 = f4.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        fused_features = f1 + f2 + f3 + f4
        decorated_feature1 = self.feature_decorated1(fused_features)

        decorated_feature2 = self.feature_decorated2(maps)

        final_features = maps + decorated_feature1 * self.scale_param1 + decorated_feature2 * self.scale_param2
        return final_features

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class AdapterFeatureDecorationUpdateV2(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(AdapterFeatureDecorationUpdateV2, self).__init__()

        self.adapter1 = Adapter(D_features=in_channels, skip_connect=False)
        self.res_layer12 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(in_channels))
        self.adapter2 = Adapter(D_features=in_channels, skip_connect=False)
        self.res_layer23 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(in_channels))
        self.adapter3 = Adapter(D_features=in_channels, skip_connect=False)
        self.res_layer34 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(in_channels))
        self.adapter4 = Adapter(D_features=in_channels, skip_connect=False)

        self.feature_decorated1 = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, bias=False),
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel),
        )

        self.feature_decorated2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=False),
            LayerNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel))

        self.scale_param1 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.01)
        self.scale_param2 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.01)

        self.apply(self._init_weights)

    def forward(self, list_features):
        f1, f2, f3, f4, maps = list_features
        B, _, H, W = f1.shape
        f1 = f1.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        f1 = self.adapter1(f1) + f1
        f1 = f1.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        f2 = self.res_layer12(f1) + f2
        f2 = f2.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        f2 = self.adapter2(f2) + f2
        f2 = f2.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        f3 = self.res_layer23(f2) + f3
        f3 = f3.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        f3 = self.adapter3(f3) + f3
        f3 = f3.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        f4 = self.res_layer34(f3) + f4
        f4 = f4.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        f4 = self.adapter4(f4) + f4
        f4 = f4.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        fused_features = torch.cat([f1, f2, f3, f4], dim=1)
        decorated_feature1 = self.feature_decorated1(fused_features)

        decorated_feature2 = self.feature_decorated2(maps)

        final_features = maps + decorated_feature1 * self.scale_param1 + decorated_feature2 * self.scale_param2
        return final_features

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class AdapterFeatureDecorationUpUp(nn.Module):
    def __init__(self, in_channels, out_channel, mlp_ratio=0.25):
        super(AdapterFeatureDecorationUpUp, self).__init__()
        D_hidden_features = int(in_channels * mlp_ratio)
        self.adapter1 = Adapter(D_features=in_channels, skip_connect=False)
        self.res_layer12 = nn.Sequential(
            nn.Conv2d(in_channels, D_hidden_features, kernel_size=1, bias=False),
            LayerNorm2d(D_hidden_features),
            nn.Conv2d(D_hidden_features, D_hidden_features, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(D_hidden_features),
            nn.Conv2d(D_hidden_features, in_channels, kernel_size=1, bias=False),
            LayerNorm2d(in_channels),
        )

        self.adapter2 = Adapter(D_features=in_channels, skip_connect=False)
        self.res_layer23 = nn.Sequential(
            nn.Conv2d(in_channels, D_hidden_features, kernel_size=1, bias=False),
            LayerNorm2d(D_hidden_features),
            nn.Conv2d(D_hidden_features, D_hidden_features, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(D_hidden_features),
            nn.Conv2d(D_hidden_features, in_channels, kernel_size=1, bias=False),
            LayerNorm2d(in_channels),
        )
        self.adapter3 = Adapter(D_features=in_channels, skip_connect=False)
        self.res_layer34 = nn.Sequential(
            nn.Conv2d(in_channels, D_hidden_features, kernel_size=1, bias=False),
            LayerNorm2d(D_hidden_features),
            nn.Conv2d(D_hidden_features, D_hidden_features, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(D_hidden_features),
            nn.Conv2d(D_hidden_features, in_channels, kernel_size=1, bias=False),
            LayerNorm2d(in_channels),
        )
        self.adapter4 = Adapter(D_features=in_channels, skip_connect=False)

        self.feature_decorated1 = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, bias=False),
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel)
        )

        self.feature_decorated2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=False),
            LayerNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel))

        self.scale_param1 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.01)
        self.scale_param2 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.01)

        self.apply(self._init_weights)

    def forward(self, list_features):
        f1, f2, f3, f4, maps = list_features
        B, _, H, W = f1.shape
        f1 = f1.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        f1 = self.adapter1(f1) + f1
        f1 = f1.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        f2 = self.res_layer12(f1) + f2
        f2 = f2.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        f2 = self.adapter2(f2) + f2
        f2 = f2.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        f3 = self.res_layer23(f2) + f3
        f3 = f3.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        f3 = self.adapter3(f3) + f3
        f3 = f3.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        f4 = self.res_layer34(f3) + f4
        f4 = f4.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        f4 = self.adapter4(f4) + f4
        f4 = f4.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        fused_features = torch.cat([f1, f2, f3, f4], dim=1)
        decorated_feature1 = self.feature_decorated1(fused_features)

        decorated_feature2 = self.feature_decorated2(maps)

        final_features = maps + decorated_feature1 * self.scale_param1 + decorated_feature2 * self.scale_param2
        return final_features

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class AdapterFeatureDecorationUpUpV2(nn.Module):
    def __init__(self, in_channels, out_channel, mlp_ratio=0.25):
        super(AdapterFeatureDecorationUpUpV2, self).__init__()
        D_hidden_features = int(in_channels * mlp_ratio)
        self.adapter1 = Adapter(D_features=in_channels, skip_connect=False)
        self.res_layer12 = nn.Sequential(
            nn.Conv2d(in_channels, D_hidden_features, kernel_size=1, bias=False),
            LayerNorm2d(D_hidden_features),
            nn.GELU(),
            nn.Conv2d(D_hidden_features, D_hidden_features, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(D_hidden_features),
            nn.GELU(),
            nn.Conv2d(D_hidden_features, in_channels, kernel_size=1, bias=False),
            LayerNorm2d(in_channels),
            nn.GELU()
        )

        self.adapter2 = Adapter(D_features=in_channels, skip_connect=False)
        self.res_layer23 = nn.Sequential(
            nn.Conv2d(in_channels, D_hidden_features, kernel_size=1, bias=False),
            LayerNorm2d(D_hidden_features),
            nn.GELU(),
            nn.Conv2d(D_hidden_features, D_hidden_features, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(D_hidden_features),
            nn.GELU(),
            nn.Conv2d(D_hidden_features, in_channels, kernel_size=1, bias=False),
            LayerNorm2d(in_channels),
            nn.GELU()
        )
        self.adapter3 = Adapter(D_features=in_channels, skip_connect=False)
        self.res_layer34 = nn.Sequential(
            nn.Conv2d(in_channels, D_hidden_features, kernel_size=1, bias=False),
            LayerNorm2d(D_hidden_features),
            nn.GELU(),
            nn.Conv2d(D_hidden_features, D_hidden_features, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(D_hidden_features),
            nn.GELU(),
            nn.Conv2d(D_hidden_features, in_channels, kernel_size=1, bias=False),
            LayerNorm2d(in_channels),
            nn.GELU()
        )
        self.adapter4 = Adapter(D_features=in_channels, skip_connect=False)

        self.feature_decorated1 = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, bias=False),
            LayerNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel)
        )

        self.feature_decorated2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=False),
            LayerNorm2d(out_channel),
            nn.GELU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel))

        self.scale_param1 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.01)
        self.scale_param2 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.01)

        self.apply(self._init_weights)

    def forward(self, list_features):
        f1, f2, f3, f4, maps = list_features
        B, _, H, W = f1.shape
        f1 = f1.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        f1 = self.adapter1(f1) + f1
        f1 = f1.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        f2 = self.res_layer12(f1) + f2
        f2 = f2.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        f2 = self.adapter2(f2) + f2
        f2 = f2.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        f3 = self.res_layer23(f2) + f3
        f3 = f3.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        f3 = self.adapter3(f3) + f3
        f3 = f3.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        f4 = self.res_layer34(f3) + f4
        f4 = f4.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        f4 = self.adapter4(f4) + f4
        f4 = f4.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        fused_features = torch.cat([f1, f2, f3, f4], dim=1)
        decorated_feature1 = self.feature_decorated1(fused_features)

        decorated_feature2 = self.feature_decorated2(maps)

        final_features = maps + decorated_feature1 * self.scale_param1 + decorated_feature2 * self.scale_param2
        return final_features

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "\
                      "The distribution of values may be incorrect.",\
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


'''
2.1 sam-微调交互分割；
AdapterFeatureDecorationUpUpV2：加上非线性修正单元没有明显效果；
AdapterFeatureDecorationUpUp：外部微调后，对下层进行积极影响；下层继续增量学习；conv1x1-3x3-1x1；0.8024；sam_ourdecUpup_data_update_std
AdapterFeatureDecorationUpdateV2：外部微调后，对下层进行积极影响；下层继续增量学习； f1 + f2 + f3 + f4 ；0.7948；sam_ourdecV3_data_update_std
AdapterFeatureDecorationUpdate: 外部微调后，对下层进行积极影响；下层继续增量学习； concatenate(f1, f2 , f3 ,f4) 0.7911；sam_ourdecV2_data_update_std
AdapterFeatureDecoration:外部微调后，对下层进行积极影响； concatenate(f1, f2 , f3 ,f4) 0.7998； sam_ourdec_data_update_std
AdapterFusedFeature：concatenate 直接融合。× sam_ours_data_update_std'''