import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math
from .refine_decoder import ASPP
from networks.sam.common import LayerNorm2d


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class GatedAttentionUnit(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        self.w1 = nn.Sequential(
            DepthWiseConv2d(in_c, in_c, kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

        self.w2 = nn.Sequential(
            DepthWiseConv2d(in_c, in_c, kernel_size + 2, padding=(kernel_size + 2) // 2),
            nn.GELU()
        )
        self.wo = nn.Sequential(
            DepthWiseConv2d(in_c, out_c, kernel_size),
            nn.GELU()
        )

        self.cw = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        x1, x2 = self.w1(x), self.w2(x)
        out = self.wo(x1 * x2) + self.cw(x)
        return out


class DilatedGatedAttention(nn.Module):
    def __init__(self, in_c, out_c, k_size=3, dilated_ratio=[7, 5, 2, 1]):
        super().__init__()

        self.mda0 = nn.Conv2d(in_c // 4, in_c // 4, kernel_size=k_size, stride=1,
                              padding=(k_size + (k_size - 1) * (dilated_ratio[0] - 1)) // 2,
                              dilation=dilated_ratio[0], groups=in_c // 4)
        self.mda1 = nn.Conv2d(in_c // 4, in_c // 4, kernel_size=k_size, stride=1,
                              padding=(k_size + (k_size - 1) * (dilated_ratio[1] - 1)) // 2,
                              dilation=dilated_ratio[1], groups=in_c // 4)
        self.mda2 = nn.Conv2d(in_c // 4, in_c // 4, kernel_size=k_size, stride=1,
                              padding=(k_size + (k_size - 1) * (dilated_ratio[2] - 1)) // 2,
                              dilation=dilated_ratio[2], groups=in_c // 4)
        self.mda3 = nn.Conv2d(in_c // 4, in_c // 4, kernel_size=k_size, stride=1,
                              padding=(k_size + (k_size - 1) * (dilated_ratio[3] - 1)) // 2,
                              dilation=dilated_ratio[3], groups=in_c // 4)
        self.norm_layer = nn.GroupNorm(4, in_c)
        self.conv = nn.Conv2d(in_c, in_c, 1)

        self.gau = GatedAttentionUnit(in_c, out_c, 3)

    def forward(self, x):
        x = torch.chunk(x, 4, dim=1)
        x0 = self.mda0(x[0])
        x1 = self.mda1(x[1])
        x2 = self.mda2(x[2])
        x3 = self.mda3(x[3])
        x = F.gelu(self.conv(self.norm_layer(torch.cat((x0, x1, x2, x3), dim=1))))
        x = self.gau(x)
        return x


class EAblock(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, in_c, 1)

        self.k = in_c * 4
        self.linear_0 = nn.Conv1d(in_c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, in_c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)

        self.conv2 = nn.Conv2d(in_c, in_c, 1, bias=False)
        self.norm_layer = nn.GroupNorm(4, in_c)

    def forward(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n

        attn = self.linear_0(x)  # b, k, n
        attn = F.softmax(attn, dim=-1)  # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # # b, k, n
        x = self.linear_1(attn)  # b, c, n

        x = x.view(b, c, h, w)
        x = self.norm_layer(self.conv2(x))
        x = x + idn
        x = F.gelu(x)
        return x


class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3):
        att = torch.cat((self.avgpool(t1),
                         self.avgpool(t2),
                         self.avgpool(t3)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))

        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)

        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)

        return att1, att2, att3


class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                           nn.Sigmoid())

    def forward(self, t1, t2, t3):
        t_list = [t1, t2, t3]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2]


class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()

        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()

    def forward(self, t1, t2, t3):
        r1, r2, r3 = t1, t2, t3

        satt1, satt2, satt3= self.satt(t1, t2, t3)
        t1, t2, t3 = satt1 * t1, satt2 * t2, satt3 * t3

        r1_, r2_, r3_ = t1, t2, t3
        t1, t2, t3 = t1 + r1, t2 + r2, t3 + r3

        catt1, catt2, catt3 = self.catt(t1, t2, t3)
        t1, t2, t3 = catt1 * t1, catt2 * t2, catt3 * t3

        return t1 + r1_, t2 + r2_, t3 + r3_


class RefineUNet(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, c_list=[24, 32, 48, 64],
                 split_att='fc', bridge=True):
        super().__init__()

        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            EAblock(c_list[0]),
            DilatedGatedAttention(c_list[0], c_list[1]),
        )
        self.encoder3 = nn.Sequential(
            EAblock(c_list[1]),
            DilatedGatedAttention(c_list[1], c_list[2]),
        )
        self.encoder4 = nn.Sequential(
            EAblock(c_list[2]),
            DilatedGatedAttention(c_list[2], c_list[3]),
        )

        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            # print('SC_Att_Bridge was used')

        self.decoder1 = nn.Sequential(
            DilatedGatedAttention(c_list[3], c_list[2]),
            EAblock(c_list[2]),
        )
        self.decoder2 = nn.Sequential(
            DilatedGatedAttention(c_list[2], c_list[1]),
            EAblock(c_list[1]),
        )
        self.decoder3 = nn.Sequential(
            DilatedGatedAttention(c_list[1], c_list[0]),
            EAblock(c_list[0]),
        )

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])

        self.dbn1 = nn.GroupNorm(4, c_list[2])
        self.dbn2 = nn.GroupNorm(4, c_list[1])
        self.dbn3 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        input_x = torch.sigmoid(x)
        # 使用膨胀操作
        # dilated_x = F.max_pool2d(input_x, kernel_size=5, stride=1, padding=2)
        # # 使用腐蚀操作
        # eroded_x = -F.max_pool2d(-input_x, kernel_size=5, stride=1, padding=2)
        # new_x = torch.cat([eroded_x, input_x, dilated_x], dim=1)

        # res = self.encoder1(new_x)
        out = F.gelu(self.ebn1(self.encoder1(input_x)))
        t1 = out  # b, c2, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c3, H/16, W/16

        if self.bridge: t1, t2, t3 = self.scab(t1, t2, t3)

        out = F.gelu(self.encoder4(out))  # b, c5, H/32, W/32

        out3 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
        out3 = torch.add(out3, t3)  # b, c4, H/32, W/32

        out2 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16
        out2 = torch.add(out2, t2)  # b, c3, H/16, W/16

        out1 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        out1 = torch.add(out1, t1)  # b, c2, H/8, W/8


        # out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
        #                      align_corners=True)  # b, num_class, H, W

        out0 = self.final(out1)

        return out0 + x


class RefineUNetFeatures(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, c_list=[24, 32, 48, 64],
                 split_att='fc', bridge=True):
        super().__init__()

        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            EAblock(c_list[0]),
            DilatedGatedAttention(c_list[0], c_list[1]),
        )
        self.encoder3 = nn.Sequential(
            EAblock(c_list[1]),
            DilatedGatedAttention(c_list[1], c_list[2]),
        )
        self.encoder4 = nn.Sequential(
            EAblock(c_list[2]),
            DilatedGatedAttention(c_list[2], c_list[3]),
        )

        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            # print('SC_Att_Bridge was used')
        self.feature_aspp = nn.Sequential(
            nn.Conv2d(256, c_list[3], kernel_size=3, padding=1, bias=False),
            LayerNorm2d(c_list[3]),
            ASPP(in_channels=c_list[3], atrous_rates=[3, 6, 9])
        )
        self.fuse_features = nn.Sequential(
            nn.Conv2d(c_list[3] * 2, c_list[3], 3, stride=1, padding=1),
            nn.GroupNorm(4, c_list[3])
        )

        self.decoder1 = nn.Sequential(
            DilatedGatedAttention(c_list[3], c_list[2]),
            EAblock(c_list[2]),
        )
        self.decoder2 = nn.Sequential(
            DilatedGatedAttention(c_list[2], c_list[1]),
            EAblock(c_list[1]),
        )
        self.decoder3 = nn.Sequential(
            DilatedGatedAttention(c_list[1], c_list[0]),
            EAblock(c_list[0]),
        )

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])

        self.dbn1 = nn.GroupNorm(4, c_list[2])
        self.dbn2 = nn.GroupNorm(4, c_list[1])
        self.dbn3 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, image_features=None):
        input_x = torch.sigmoid(x)
        # 使用膨胀操作
        # dilated_x = F.max_pool2d(input_x, kernel_size=5, stride=1, padding=2)
        # # 使用腐蚀操作
        # eroded_x = -F.max_pool2d(-input_x, kernel_size=5, stride=1, padding=2)
        # new_x = torch.cat([eroded_x, input_x, dilated_x], dim=1)
        # res = self.encoder1(new_x)
        out = F.gelu(self.ebn1(self.encoder1(input_x)))
        t1 = out  # b, c2, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c3, H/16, W/16

        if self.bridge: t1, t2, t3 = self.scab(t1, t2, t3)

        out = F.gelu(self.encoder4(out))

        image_features = self.feature_aspp(image_features)
        out = self.fuse_features(torch.cat([out, image_features], dim=1))

        out3 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
        out3 = torch.add(out3, t3)  # b, c4, H/32, W/32

        out2 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16
        out2 = torch.add(out2, t2)  # b, c3, H/16, W/16

        out1 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        out1 = torch.add(out1, t1)  # b, c2, H/8, W/8


        # out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
        #                      align_corners=True)  # b, num_class, H, W
        out0 = self.final(out1)
        return out0 + x
