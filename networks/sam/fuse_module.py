import torch
import torch.nn as nn

from typing import Type
from networks.sam.common import Adapter, LayerNorm2d, Decorator, AdapterWithAttention, ComprehensiveAdapter
import math
import warnings
import math
from functools import reduce
from operator import mul
from torch import Tensor
import torch.nn.functional as F


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


class AdapterDecoratedLayers(nn.Module):
    def __init__(self, in_channels, out_channel, num_layers=12):
        super(AdapterDecoratedLayers, self).__init__()
        adapter_layers = []
        for i in range(num_layers):
            adapter_layers.append(Adapter(D_features=in_channels, skip_connect=False))
        self.adapter_layers = nn.ModuleList(adapter_layers)
        self.num_layers = num_layers
        self.feature_decorated1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
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

        self.scale_param1 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.001)
        self.scale_param2 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.001)

        self.apply(self._init_weights)

    def forward(self, list_features):
        assert len(list_features) == (self.num_layers + 1)
        maps = list_features[-1]
        B, _, H, W = list_features[0].shape
        list_new_f = []
        for i in range(self.num_layers):
            f = list_features[i].permute(0, 2, 3, 1).reshape(B, H*W, -1)
            f = self.adapter_layers[i](f) + f
            f = f.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            list_new_f.append(f)

        fused_features = sum(list_new_f) / self.num_layers
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


class DecoratorLayers(nn.Module):
    def __init__(self, in_channels, out_channel, num_layers=12):
        super(DecoratorLayers, self).__init__()
        decorator_layers = []
        for i in range(num_layers):
            decorator_layers.append(Decorator(D_features=in_channels, skip_connect=True))
        self.decorator_layers = nn.ModuleList(decorator_layers)
        self.num_layers = num_layers
        self.feature_decorated1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
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

        self.scale_param1 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.001)
        self.scale_param2 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.001)

        self.apply(self._init_weights)

    def forward(self, list_features):
        assert len(list_features) == (self.num_layers + 1)
        maps = list_features[-1]
        B, _, H, W = list_features[0].shape
        list_new_f = []
        for i in range(self.num_layers):
            f = list_features[i].permute(0, 2, 3, 1).reshape(B, H*W, -1)
            f = self.decorator_layers[i](f, H, W)
            f = f.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            list_new_f.append(f)

        fused_features = sum(list_new_f) / self.num_layers
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


class DecoratorLayersFlow(nn.Module):
    def __init__(self, in_channels, out_channel, num_layers=12, mlp_ratio=0.25):
        super(DecoratorLayersFlow, self).__init__()
        D_hidden_features = int(in_channels * mlp_ratio)
        decorator_layers = []
        for i in range(num_layers):
            decorator_layers.append(Decorator(D_features=in_channels, skip_connect=False))
        self.decorator_layers = nn.ModuleList(decorator_layers)

        self.num_layers = num_layers
        self.feature_decorated1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel),
        )
        flow_layers = []
        for i in range(num_layers-1):
            flow_layers.append(
                nn.Sequential(
                nn.Conv2d(in_channels, D_hidden_features, kernel_size=1, bias=False),
                LayerNorm2d(D_hidden_features),
                nn.Conv2d(D_hidden_features, D_hidden_features, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(D_hidden_features),
                nn.Conv2d(D_hidden_features, in_channels, kernel_size=1, bias=False),
                LayerNorm2d(in_channels)
            )
            )
        self.flow_layers = nn.ModuleList(flow_layers)
        self.feature_decorated2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=False),
            LayerNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel))

        self.scale_param1 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.001)
        self.scale_param2 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.001)

        self.apply(self._init_weights)

    def forward(self, list_features):
        assert len(list_features) == (self.num_layers + 1)
        maps = list_features[-1]
        B, _, H, W = list_features[0].shape
        list_new_f = []
        for i in range(self.num_layers):
            if i > 0:
                list_features[i] = self.flow_layers[i-1](list_new_f[i-1]) + list_features[i]

            f = list_features[i].permute(0, 2, 3, 1).reshape(B, H*W, -1)
            f = self.decorator_layers[i](f, H, W)
            f = f.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            list_new_f.append(f)

        fused_features = sum(list_new_f) / self.num_layers

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


class AdapterDecoratedLayersFlow(nn.Module):
    def __init__(self, in_channels, out_channel, num_layers=12, mlp_ratio=0.25):
        super(AdapterDecoratedLayersFlow, self).__init__()
        D_hidden_features = int(in_channels * mlp_ratio)
        adapter_layers = []
        for i in range(num_layers):
            adapter_layers.append(Adapter(D_features=in_channels, skip_connect=False))
        self.adapter_layers = nn.ModuleList(adapter_layers)
        self.num_layers = num_layers
        self.feature_decorated1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel),
        )
        flow_layers = []
        for i in range(num_layers-1):
            flow_layers.append(
                nn.Sequential(
                nn.Conv2d(in_channels, D_hidden_features, kernel_size=1, bias=False),
                LayerNorm2d(D_hidden_features),
                nn.Conv2d(D_hidden_features, D_hidden_features, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(D_hidden_features),
                nn.Conv2d(D_hidden_features, in_channels, kernel_size=1, bias=False),
                LayerNorm2d(in_channels)
            )
            )
        self.flow_layers = nn.ModuleList(flow_layers)
        self.feature_decorated2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=False),
            LayerNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel))

        self.scale_param1 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.001)
        self.scale_param2 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.001)

        self.apply(self._init_weights)

    def forward(self, list_features):
        assert len(list_features) == (self.num_layers + 1)
        maps = list_features[-1]
        B, _, H, W = list_features[0].shape
        list_new_f = []
        for i in range(self.num_layers):
            if i > 0:
                list_features[i] = self.flow_layers[i-1](list_new_f[i-1]) + list_features[i]

            f = list_features[i].permute(0, 2, 3, 1).reshape(B, H*W, -1)
            f = self.adapter_layers[i](f) + f
            f = f.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            list_new_f.append(f)

        fused_features = sum(list_new_f) / self.num_layers

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


class AdapterDecoratedLayersFlowV2(nn.Module):
    def __init__(self, in_channels, out_channel, num_layers=12, mlp_ratio=0.25):
        super(AdapterDecoratedLayersFlowV2, self).__init__()
        D_hidden_features = int(in_channels * mlp_ratio)
        adapter_layers = []
        for i in range(num_layers):
            adapter_layers.append(Adapter(D_features=in_channels, skip_connect=False))
        self.adapter_layers = nn.ModuleList(adapter_layers)
        self.num_layers = num_layers
        self.feature_decorated1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel),
        )
        flow_layers = []
        for i in range(num_layers-1):
            flow_layers.append(
                nn.Sequential(
                nn.Conv2d(in_channels, D_hidden_features, kernel_size=1, bias=False),
                LayerNorm2d(D_hidden_features),
                nn.Conv2d(D_hidden_features, D_hidden_features, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(D_hidden_features),
                nn.Conv2d(D_hidden_features, in_channels, kernel_size=1, bias=False),
                LayerNorm2d(in_channels)
            ))

        self.flow_layers = nn.ModuleList(flow_layers)
        self.feature_decorated2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=False),
            LayerNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channel))

        self.scale_param1 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.001)
        self.scale_param2 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.001)

        self.apply(self._init_weights)

    def forward(self, list_features):
        assert len(list_features) == (self.num_layers + 1)
        maps = list_features[-1]
        B, _, H, W = list_features[0].shape
        list_new_f = []
        for i in range(self.num_layers):
            if i > 0:
                list_features[i] = self.flow_layers[i-1](list_new_f[i-1]) + list_features[i]
            f = list_features[i].permute(0, 2, 3, 1).reshape(B, H*W, -1)
            f = self.adapter_layers[i](f) + f
            f = f.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            list_new_f.append(f)

        fused_features = sum(list_new_f[-4:])

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


class OurDecorator(nn.Module):
    def __init__(self, in_channels, out_channel, num_layers=12, mlp_ratio=0.25):
        super(OurDecorator, self).__init__()
        D_hidden_features = int(in_channels * mlp_ratio)
        adapter_layers = []
        for i in range(num_layers):
            adapter_layers.append(Adapter(D_features=in_channels, skip_connect=False))
        self.adapter_layers = nn.ModuleList(adapter_layers)
        self.num_layers = num_layers
        self.feature_decorated1 = Adapter(D_features=in_channels, out_features=256, skip_connect=False)

        flow_layers = []
        for i in range(num_layers - 1):
            flow_layers.append(
                Adapter(D_features=in_channels, skip_connect=False))

        self.flow_layers = nn.ModuleList(flow_layers)
        self.feature_decorated2 = Adapter(D_features=256, skip_connect=False)

        self.scale_param1 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.001)
        self.scale_param2 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.001)

        self.apply(self._init_weights)

    def forward(self, list_features):
        assert len(list_features) == (self.num_layers + 1)
        maps = list_features[-1]
        B, _, H, W = list_features[0].shape
        list_new_f = []
        for i in range(self.num_layers):
            if i > 0:
                pre_f = list_new_f[i - 1].permute(0, 2, 3, 1).reshape(B, H*W, -1)
                f = list_features[i].permute(0, 2, 3, 1).reshape(B, H * W, -1)
                f = self.flow_layers[i - 1](pre_f) + f
            else:
                f = list_features[i].permute(0, 2, 3, 1).reshape(B, H*W, -1)

            f = self.adapter_layers[i](f) + f
            f = f.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            list_new_f.append(f)

        fused_features = sum(list_new_f[-4:])

        fused_features = fused_features.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        decorated_feature1 = self.feature_decorated1(fused_features)
        decorated_feature1 = decorated_feature1.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        map_fs = maps.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        decorated_feature2 = self.feature_decorated2(map_fs)
        decorated_feature2 = decorated_feature2.reshape(B, H, W, -1).permute(0, 3, 1, 2)

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


class OurDecoratorAdd(nn.Module):
    def __init__(self, in_channels, out_channel, num_layers=12, mlp_ratio=0.25):
        super(OurDecoratorAdd, self).__init__()
        D_hidden_features = int(in_channels * mlp_ratio)
        adapter_layers = []
        for i in range(num_layers):
            adapter_layers.append(Adapter(D_features=in_channels, skip_connect=False))
        self.adapter_layers = nn.ModuleList(adapter_layers)
        self.num_layers = num_layers
        self.feature_decorated = Adapter(D_features=in_channels, out_features=256, skip_connect=False)

        flow_layers = []
        for i in range(num_layers - 1):
            flow_layers.append(
                Adapter(D_features=in_channels, skip_connect=False))

        self.flow_layers = nn.ModuleList(flow_layers)
        # self.feature_decorated2 = Adapter(D_features=256, skip_connect=False)

        # self.scale_param1 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.001)
        # self.scale_param2 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.001)

        self.apply(self._init_weights)

    def forward(self, list_features):
        assert len(list_features) == (self.num_layers + 1)
        maps = list_features[-1]
        B, _, H, W = list_features[0].shape
        list_new_f = []
        for i in range(self.num_layers):
            if i > 0:
                pre_f = sum(list_new_f[:i])
                pre_f = pre_f.permute(0, 2, 3, 1).reshape(B, H*W, -1)
                # pre_f = list_new_f[i - 1].permute(0, 2, 3, 1).reshape(B, H*W, -1)
                f = list_features[i].permute(0, 2, 3, 1).reshape(B, H * W, -1)
                f = self.flow_layers[i - 1](pre_f) + f
            else:
                f = list_features[i].permute(0, 2, 3, 1).reshape(B, H*W, -1)

            f = self.adapter_layers[i](f) + f
            f = f.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            list_new_f.append(f)

        # fused_features = sum(list_new_f[-4:])
        fused_features = list_new_f[-1]
        fused_features = fused_features.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        decorated_feature = self.feature_decorated(fused_features)
        decorated_feature = decorated_feature.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # map_fs = maps.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        # decorated_feature2 = self.feature_decorated2(map_fs)
        # decorated_feature2 = decorated_feature2.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        final_features = maps + decorated_feature #* self.scale_param1 #+ decorated_feature2 * self.scale_param2
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


class DecoratorAtte(nn.Module):
    def __init__(self, in_channels, out_channel, num_layers=12, mlp_ratio=0.25):
        super(DecoratorAtte, self).__init__()
        D_hidden_features = int(in_channels * mlp_ratio)
        adapter_layers = []
        for i in range(num_layers):
            adapter_layers.append(AdapterWithAttention(input_dim=in_channels, output_dim=in_channels, num_heads=8))
        self.adapter_layers = nn.ModuleList(adapter_layers)
        self.num_layers = num_layers
        self.feature_decorated1 = AdapterWithAttention(input_dim=in_channels, output_dim=256, num_heads=8)

        flow_layers = []
        for i in range(num_layers - 1):
            flow_layers.append(
                AdapterWithAttention(input_dim=in_channels, output_dim=in_channels, num_heads=8))

        self.flow_layers = nn.ModuleList(flow_layers)
        self.feature_decorated2 = AdapterWithAttention(input_dim=256, output_dim=256, num_heads=8)

        self.scale_param1 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.001)
        self.scale_param2 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.001)

        self.apply(self._init_weights)

    def forward(self, list_features):
        assert len(list_features) == (self.num_layers + 1)
        maps = list_features[-1]
        B, _, H, W = list_features[0].shape
        list_new_f = []

        for i in range(self.num_layers):
            if i > 0:
                pre_f = list_new_f[i - 1].permute(0, 2, 3, 1).reshape(B, H*W, -1)
                f = list_features[i].permute(0, 2, 3, 1).reshape(B, H * W, -1)
                f = self.flow_layers[i - 1](pre_f) + f
            else:
                f = list_features[i].permute(0, 2, 3, 1).reshape(B, H*W, -1)

            f = self.adapter_layers[i](f) + f
            f = f.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            list_new_f.append(f)

        fused_features = sum(list_new_f[-4:])

        fused_features = fused_features.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        decorated_feature1 = self.feature_decorated1(fused_features)
        decorated_feature1 = decorated_feature1.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        map_fs = maps.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        decorated_feature2 = self.feature_decorated2(map_fs)
        decorated_feature2 = decorated_feature2.reshape(B, H, W, -1).permute(0, 3, 1, 2)

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


class DecoratorAtteNew(nn.Module):
    def __init__(self, in_channels, out_channel, num_layers=12, mlp_ratio=0.25):
        super(DecoratorAtteNew, self).__init__()
        D_hidden_features = int(in_channels * mlp_ratio)
        adapter_layers = []
        for i in range(num_layers):
            adapter_layers.append(ComprehensiveAdapter(input_dim=in_channels, output_dim=in_channels, num_heads=8))
        self.adapter_layers = nn.ModuleList(adapter_layers)
        self.num_layers = num_layers
        self.feature_decorated1 = ComprehensiveAdapter(input_dim=in_channels, output_dim=256, num_heads=8)

        flow_layers = []
        for i in range(num_layers - 1):
            flow_layers.append(
                ComprehensiveAdapter(input_dim=in_channels, output_dim=in_channels, num_heads=8))

        self.flow_layers = nn.ModuleList(flow_layers)
        self.feature_decorated2 = ComprehensiveAdapter(input_dim=256, output_dim=256, num_heads=8)

        self.scale_param1 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.001)
        self.scale_param2 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.001)

        self.apply(self._init_weights)

    def forward(self, list_features):
        assert len(list_features) == (self.num_layers + 1)
        maps = list_features[-1]
        B, _, H, W = list_features[0].shape
        list_new_f = []

        for i in range(self.num_layers):
            if i > 0:
                pre_f = list_new_f[i - 1].permute(0, 2, 3, 1).reshape(B, H * W, -1)
                f = list_features[i].permute(0, 2, 3, 1).reshape(B, H * W, -1)
                f = self.flow_layers[i - 1](pre_f) + f
            else:
                f = list_features[i].permute(0, 2, 3, 1).reshape(B, H * W, -1)

            f = self.adapter_layers[i](f) + f
            f = f.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            list_new_f.append(f)

        fused_features = sum(list_new_f[-4:])

        fused_features = fused_features.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        decorated_feature1 = self.feature_decorated1(fused_features)
        decorated_feature1 = decorated_feature1.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        map_fs = maps.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        decorated_feature2 = self.feature_decorated2(map_fs)
        decorated_feature2 = decorated_feature2.reshape(B, H, W, -1).permute(0, 3, 1, 2)

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


class Reins(nn.Module):
    def __init__(
        self,
        num_layers: int = 4,
        embed_dims: int = 768,
        patch_size: int = 16,
        query_dims: int = 256,
        token_length: int = 100,
        use_softmax: bool = True,
        link_token_to_query: bool = True,
        scale_init: float = 0.001,
        zero_mlp_delta_f: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f
        self.create_model()

    def create_model(self):
        self.learnable_tokens = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
            )
        )
        nn.init.uniform_(self.learnable_tokens.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))
        self.transform = nn.Linear(self.embed_dims, self.query_dims)
        self.merge = nn.Linear(self.query_dims * 3, self.query_dims)
        if self.zero_mlp_delta_f:
            del self.scale
            self.scale = 1.0
            nn.init.zeros_(self.mlp_delta_f.weight)
            nn.init.zeros_(self.mlp_delta_f.bias)

    def return_auto(self, feats):
        if self.link_token_to_query:
            tokens = self.transform(self.get_tokens(-1)).permute(1, 2, 0)
            tokens = torch.cat(
                [
                    F.max_pool1d(tokens, kernel_size=self.num_layers),
                    F.avg_pool1d(tokens, kernel_size=self.num_layers),
                    tokens[:, :, -1].unsqueeze(-1),
                ],
                dim=-1,
            )
            querys = self.merge(tokens.flatten(-2, -1))
            return feats, querys
        else:
            return feats

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.learnable_tokens
        else:
            return self.learnable_tokens[layer]

    def forward(
        self, feats: Tensor, layer: int, batch_first=True, has_cls_token=False
    ) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)
        tokens = self.get_tokens(layer)
        delta_feat = self.forward_delta_feat(
            feats,
            tokens,
            layer,
        )
        delta_feat = delta_feat * self.scale

        feats = feats + delta_feat
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats

    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn[:, :, 1:],
            self.mlp_token2feat(tokens[1:, :]),
        )
        delta_f = self.mlp_delta_f(delta_f + feats)
        return delta_f


class OurDecoratorTaken(nn.Module):
    def __init__(self, in_channels, out_channel, num_layers=12, mlp_ratio=0.25):
        super(OurDecoratorTaken, self).__init__()
        D_hidden_features = int(in_channels * mlp_ratio)
        # adapter_layers = []
        # for i in range(num_layers):
        #     adapter_layers.append(Adapter(D_features=in_channels, skip_connect=False))
        # self.adapter_layers = nn.ModuleList(adapter_layers)
        self.reins: Reins = Reins(num_layers=4, embed_dims=768)
        self.rein_enabled_layers = list(range(4))
        self.num_layers = num_layers
        self.feature_decorated1 = Adapter(D_features=in_channels, out_features=256, skip_connect=False)
        flow_layers = []
        for i in range(num_layers - 1):
            flow_layers.append(
                Adapter(D_features=in_channels, skip_connect=False))

        self.flow_layers = nn.ModuleList(flow_layers)
        self.feature_decorated2 = Adapter(D_features=256, skip_connect=False)

        self.scale_param1 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.001)
        self.scale_param2 = nn.Parameter(torch.ones(1, out_channel, 1, 1) * 0.001)

        self.apply(self._init_weights)

    def forward(self, list_features):
        assert len(list_features) == (self.num_layers + 1)
        maps = list_features[-1]
        B, C, H, W = list_features[0].shape
        list_new_f = []
        for i in range(self.num_layers):
            if i > 0:
                pre_f = list_new_f[i - 1].permute(0, 2, 3, 1).reshape(B, H*W, -1)
                f = list_features[i].permute(0, 2, 3, 1).reshape(B, H * W, -1)
                f = self.flow_layers[i - 1](pre_f) + f
            else:
                f = list_features[i].permute(0, 2, 3, 1).reshape(B, H*W, -1)

            # f = self.adapter_layers[i](f) + f
            if i in self.rein_enabled_layers:
                f = self.reins.forward(
                    f.view(B, -1, C),
                    self.rein_enabled_layers.index(i),
                    batch_first=True,
                    has_cls_token=False,
                )
            f = f.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            list_new_f.append(f)

        # if idx in self.rein_enabled_layers:
        #     x = self.reins.forward(
        #         x.view(B, -1, C),
        #         self.rein_enabled_layers.index(idx),
        #         batch_first=True,
        #         has_cls_token=False,
        #     ).view(B, H, W, C)

        fused_features = sum(list_new_f[-4:])

        fused_features = fused_features.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        decorated_feature1 = self.feature_decorated1(fused_features)
        decorated_feature1 = decorated_feature1.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        map_fs = maps.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        decorated_feature2 = self.feature_decorated2(map_fs)
        decorated_feature2 = decorated_feature2.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        final_features = maps + decorated_feature1 * self.scale_param1 + decorated_feature2 * self.scale_param2
        return self.reins.return_auto(final_features)

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
AdapterFeatureDecorationUpdateV2：外部微调后，对下层进行积极影响；下层继续增量学习； f1 + f2 + f3 + f4 ；0.7948；sam_ourdecV3_data_update_std
AdapterFeatureDecorationUpdate: 外部微调后，对下层进行积极影响；下层继续增量学习； concatenate(f1, f2 , f3 ,f4) 0.7911；sam_ourdecV2_data_update_std
AdapterFeatureDecoration:外部微调后，对下层进行积极影响； concatenate(f1, f2 , f3 ,f4) 0.7998； sam_ourdec_data_update_std
AdapterFusedFeature：concatenate 直接融合。× sam_ours_data_update_std'''

'''
有用：
AdapterFeatureDecoration:外部微调后，对下层进行积极影响； concatenate(f1, f2 , f3 ,f4) 0.7998； sam_ourdec_data_update_std
AdapterFeatureDecorationUpUp：外部微调后，对下层进行积极影响；下层继续增量学习；conv1x1-3x3-1x1；0.8024；sam_ourdecUpup_data_update_std
'''