from torch import nn
import torch
import torch.nn.functional as F
import math


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=[3, 6, 9]):
        super(ASPP, self).__init__()
        out_channels = in_channels
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class RefineDecoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super(RefineDecoder, self).__init__()

        self.conv0 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(in_channels=out_channels, atrous_rates=[3, 6, 9])

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.cls = nn.Conv2d(out_channels, 1, 1, bias=False)

        self.apply(self._init_weights)

    def forward(self, aux_seg):
        x = torch.sigmoid(aux_seg)
        x = self.conv0(x)

        x1 = self.aspp(self.conv1(x)) + x
        x1 = self.up1(x1)

        x2 = self.conv2(x1) + F.interpolate(x, size=x1.shape[-2:], mode='bilinear')
        x2 = self.up2(x2)
        x3 = self.conv3(x2) + F.interpolate(x, size=x2.shape[-2:], mode='bilinear')
        res = self.cls(x3)
        return res

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class RefineBoundaryDecoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super(RefineBoundaryDecoder, self).__init__()

        self.conv0 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.inter_completeness_layer = nn.Sequential(
            ASPP(in_channels=out_channels, atrous_rates=[3, 6, 9]),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.boundary_layer = nn.Sequential(
            ASPP(in_channels=out_channels, atrous_rates=[3, 6, 9]),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.fuse_layer = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.cls = nn.Conv2d(out_channels, 1, 1, bias=False)
        self.boundary_cls = nn.Conv2d(out_channels, 1, 1, bias=False)

        self.apply(self._init_weights)

    def forward(self, aux_seg):
        x = torch.sigmoid(aux_seg)
        x = self.conv0(x)

        x1 = self.conv1(x)

        x_boundary = self.boundary_layer(x1)

        x1 = self.inter_completeness_layer(x1) + x
        x1 = self.fuse_layer(torch.cat([x1, x_boundary], dim=1))
        x1 = self.up(x1)

        x2 = self.conv2(x1) + F.interpolate(x, size=x1.shape[-2:], mode='bilinear')
        x2 = self.up(x2)
        x3 = self.conv3(x2) + F.interpolate(x, size=x2.shape[-2:], mode='bilinear')
        res = self.cls(x3)

        boundary_res = self.boundary_cls(x_boundary)
        return res, boundary_res

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class RefineFeatureDecoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super(RefineFeatureDecoder, self).__init__()

        self.conv0 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.additional_feature = nn.Conv2d(256, out_channels, 1)
        self.fuse_layer = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(in_channels=out_channels, atrous_rates=[3, 6, 9])

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.cls = nn.Conv2d(out_channels, 1, 1, bias=False)

        self.apply(self._init_weights)

    def forward(self, aux_seg, prior_features):
        x = torch.sigmoid(aux_seg)
        x = self.conv0(x)
        prior_feat = self.additional_feature(prior_features)
        prior_feat = F.interpolate(prior_feat, size=x.shape[-2:], mode='bilinear')
        x1 = self.fuse_layer(torch.cat([x, prior_feat], dim=1))

        x1 = self.aspp(self.conv1(x1)) + x
        x1 = self.up1(x1)

        x2 = self.conv2(x1) + F.interpolate(x, size=x1.shape[-2:], mode='bilinear')
        x2 = self.up2(x2)

        x3 = self.conv3(x2) + F.interpolate(x, size=x2.shape[-2:], mode='bilinear')

        res = self.cls(x3)

        return res

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class RefineFeatureBoundaryDecoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super(RefineFeatureBoundaryDecoder, self).__init__()

        self.conv0 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.additional_feature = nn.Conv2d(256, out_channels, 1)
        self.fuse_layer = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.inter_completeness_layer = nn.Sequential(
            ASPP(in_channels=out_channels, atrous_rates=[3, 6, 9]),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.boundary_layer = nn.Sequential(
            ASPP(in_channels=out_channels, atrous_rates=[3, 6, 9]),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.fuse_layer = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.cls = nn.Conv2d(out_channels, 1, 1, bias=False)
        self.boundary_cls = nn.Conv2d(out_channels, 1, 1, bias=False)

        self.apply(self._init_weights)

    def forward(self, aux_seg, prior_features):
        x = torch.sigmoid(aux_seg)
        x = self.conv0(x)
        prior_feat = self.additional_feature(prior_features)
        prior_feat = F.interpolate(prior_feat, size=x.shape[-2:], mode='bilinear')
        x1 = self.fuse_layer(torch.cat([x, prior_feat], dim=1))

        x_boundary = self.boundary_layer(x1)

        x1 = self.inter_completeness_layer(x1) + x
        x1 = self.fuse_layer(torch.cat([x1, x_boundary], dim=1))
        x1 = self.up(x1)

        x2 = self.conv2(x1) + F.interpolate(x, size=x1.shape[-2:], mode='bilinear')
        x2 = self.up(x2)
        x3 = self.conv3(x2) + F.interpolate(x, size=x2.shape[-2:], mode='bilinear')
        res = self.cls(x3)

        boundary_res = self.boundary_cls(x_boundary)
        return res, boundary_res

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)