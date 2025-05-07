import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# class MoEConvLiner(nn.Module):
#     def __init__(self, in_channels, out_channels, num_experts):
#         super(MoEConvLiner, self).__init__()
#         self.num_experts = num_experts
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#
#         # Define the base weights
#         self.W0 = nn.Linear(in_channels, out_channels)
#         self.Wd = nn.Linear(out_channels, out_channels)
#         self.We = nn.Linear(in_channels, out_channels)
#
#         # Define experts
#         self.experts = nn.ModuleList([Expert(in_channels) for _ in range(num_experts)])
#
#         # Gating network
#         self.gating = nn.Linear(out_channels, num_experts)
#
#     def forward(self, x):
#         base_output = self.W0(x)
#         expert_inputs = self.We(x)
#
#         gating_scores = F.softmax(self.gating(expert_inputs), dim=-1)
#
#         expert_outputs = []
#         for i, expert in enumerate(self.experts):
#             expert_output = expert(expert_inputs)
#             expert_outputs.append(gating_scores[:, i].unsqueeze(1) * expert_output)
#
#         experts_combined = sum(expert_outputs)
#         output = base_output + self.Wd(experts_combined)
#
#         return output


class MoEConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts):
        super(MoEConv, self).__init__()
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Define the base convolutional layers
        self.W0 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.Wd = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.We = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Define experts
        self.experts = nn.ModuleList([Expert(in_channels) for _ in range(num_experts)])

        # Gating network
        self.gating = nn.Conv2d(out_channels, num_experts, kernel_size=1)

    def forward(self, x):
        base_output = self.W0(x)
        expert_inputs = self.We(x)

        gating_scores = F.softmax(self.gating(expert_inputs), dim=1)

        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(expert_inputs)
            expert_outputs.append(gating_scores[:, i:i + 1, :, :] * expert_output)

        experts_combined = sum(expert_outputs)
        output = base_output + self.Wd(experts_combined)

        return output


class Expert(nn.Module):
    def __init__(self, in_channels):
        super(Expert, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Interpolate up, convolve, interpolate down
        scale_factor = 4  # Example scale factor
        x_up = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=True)
        x_conv = self.conv(x_up)
        x_down = F.interpolate(x_conv, scale_factor=1 / scale_factor, mode='bilinear', align_corners=True)
        return x_down


class ConvLoRA(nn.Module):
    def __init__(self, D_features, out_features=None, num_experts=2, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        O_features = out_features if out_features is not None else D_features

        self.MoE_Conv = MoEConv(D_hidden_features, D_hidden_features, num_experts)

        self.D_fc2 = nn.Linear(D_hidden_features, O_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        B, H, W, _ = x.shape
        xs = self.D_fc1(x)
        xs = self.act(xs)

        xs = xs.permute(0, 3, 1, 2)
        xs = self.MoE_Conv(xs)
        xs = xs.permute(0, 2, 3, 1)

        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


if __name__ == '__main__':

    # Example usage
    in_channels = 64
    out_channels = 128
    num_experts = 4

    model = MoEConv(in_channels, in_channels, num_experts)
    input_tensor = torch.randn(8, in_channels, 32, 32)  # Example input
    output = model(input_tensor)
    print(output.shape)  # Should be [8, out_channels, 32, 32]
