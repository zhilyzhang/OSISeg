import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """
    将图像切分为若干个 patch，并进行线性映射
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        num_patches = self.grid_size[0] * self.grid_size[1]

        # 用 Conv2d 实现 patch embedding，相当于分块后投影
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = num_patches

    def forward(self, x):
        # B, C, H, W -> B, embed_dim, H/patch_size, W/patch_size
        x = self.proj(x)
        # flatten: B, embed_dim, Nx, Ny -> B, embed_dim, Nx*Ny
        x = x.flatten(2)
        # transpose: B, embed_dim, N -> B, N, embed_dim
        x = x.transpose(1, 2)
        return x


class Attention(nn.Module):
    """
    多头自注意力模块
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]  # 分割 Q, K, V

        # 注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)

        # 得到注意力输出
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """
    Transformer Block 内的前馈网络
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    """
    Transformer Block：Attention + MLP
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAEEncoder(nn.Module):
    """
    最简化的 MAE Encoder（ViT）
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True):
        super().__init__()

        # 1) Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 2) 可学习的位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # 3) Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, N, embed_dim]
        """
        # patch embedding
        x = self.patch_embed(x)  # B, N, embed_dim

        # 加上可学习的位置编码
        x = x + self.pos_embed

        # 经过多层 Transformer
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # 最后 LayerNorm
        return x


def load_pretrained_mae_encoder(model, checkpoint_path, encoder_prefix="encoder."):
    """
    只从预训练权重中提取并加载 MAE Encoder 相关权重

    :param model: MAEEncoder 模型实例
    :param checkpoint_path: 预训练权重文件的路径
    :param encoder_prefix: 表征 encoder 部分的前缀，默认 "encoder."
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # 可能是 'model' key，也可能是直接存的 state_dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # 只取 encoder 部分
    print([k for k in state_dict.keys()])
    encoder_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(encoder_prefix):
            # 移除前缀以对齐我们在 MAEEncoder 里定义的名字
            # 例如 "encoder.patch_embed.proj.weight" -> "patch_embed.proj.weight"
            new_k = k[len(encoder_prefix):]
            encoder_state_dict[new_k] = v

    # 将筛选出的 encoder_state_dict 加载到 model
    load_info = model.load_state_dict(encoder_state_dict, strict=False)

    print(f"Loaded encoder weights from: {checkpoint_path}")
    print(load_info)

    return model


if __name__ == "__main__":
    import torch

    # 1) 实例化 MAE Encoder
    mae_encoder = MAEEncoder(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True
    )

    # 2) 加载预训练的 Encoder 权重（只加载 "encoder." 部分）
    pretrained_path = "/home/zzl/codes/InterSegAdapter/pre_weights/cross_scale_mae_base_pretrain.pth"
    mae_encoder = load_pretrained_mae_encoder(
        model=mae_encoder,
        checkpoint_path=pretrained_path,
        encoder_prefix="encoder."  # 若实际前缀不同，请修改
    )

    # 3) 测试网络的 forward
    dummy_input = torch.randn(2, 3, 224, 224)  # batch_size=2
    with torch.no_grad():
        output = mae_encoder(dummy_input)
    print("Output shape from MAEEncoder:", output.shape)
    # 通常为 [2, (224/16)*(224/16)=196, 768]
