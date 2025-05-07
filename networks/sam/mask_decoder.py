# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .image_encoder import LayerNorm2d
from .transformer import TwoWayTransformer


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 1,  # 3
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor):
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        pred_logits, pred_masks, pred_iou = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        b, num, h, w = pred_masks.shape
        masks = (pred_logits @ pred_masks.view(b, num, h * w)).view(b, -1, h, w)

        return pred_logits, pred_masks, pred_iou, masks

    def test_forward(
        self,
        image_embeddings,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        is_vis_map=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        if is_vis_map:
            (pred_logits, pred_masks, pred_iou), features = self.predict_masks(
                x=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                is_vis_map=is_vis_map
            )
        else:
            pred_logits, pred_masks, pred_iou = self.predict_masks(
                x=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
            )

        b, num, h, w = pred_masks.shape
        tmp_masks = (pred_logits @ pred_masks.view(b, num, h * w)).view(b, -1, h, w)
        # Select the correct mask or masks for outptu
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = tmp_masks[:, mask_slice, :, :]
        iou_pred = pred_iou[:, mask_slice]

        # Prepare output
        if is_vis_map:
            return masks, iou_pred, features
        return masks, iou_pred

    def predict_masks(
        self,
        x: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        is_vis_map=False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        image_embeddings = x
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1) # [b, ]

        # Expand per-image data in batch direction to be per-mask
        # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        if image_embeddings.shape[-2:] != dense_prompt_embeddings.shape[-2:]:
            dense_prompt_embeddings = F.interpolate(dense_prompt_embeddings,
                                                    size=image_embeddings.shape[-2:],
                                                    mode='bilinear')
        if image_embeddings.shape[-2:] != image_pe.shape[-2:]:
            image_pe = F.interpolate(image_pe, size=image_embeddings.shape[-2:], mode='bilinear')

        src = image_embeddings + dense_prompt_embeddings  #
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        # b, c, h, w = upscaled_embedding.shape
        # masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if is_vis_map:
            return (hyper_in, upscaled_embedding, iou_pred), image_embeddings
        return (hyper_in, upscaled_embedding, iou_pred)


class ReinMaskDecoder(MaskDecoder):
    def __init__(self, replace_query_feat=True, **kwargs):
        super().__init__(**kwargs)
        transformer_dim = kwargs["transformer_dim"]
        # del self.query_embed
        # del self.mask_tokens
        self.vpt_transforms = nn.ModuleList()
        self.replace_query_feat = replace_query_feat
        if replace_query_feat:
            self.querys2feat = nn.Linear(transformer_dim, transformer_dim)

    def predict_masks(
        self,
        # image_embeddings: torch.Tensor,
        x: Tuple[torch.Tensor, torch.Tensor],
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        image_embeddings, query_embed = x

        if self.replace_query_feat:
            query_embed = self.querys2feat(query_embed)

        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, query_embed], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1) # [b, ]

        # Expand per-image data in batch direction to be per-mask
        # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        if image_embeddings.shape[-2:] != dense_prompt_embeddings.shape[-2:]:
            dense_prompt_embeddings = F.interpolate(dense_prompt_embeddings,
                                                    size=image_embeddings.shape[-2:],
                                                    mode='bilinear')
        if image_embeddings.shape[-2:] != image_pe.shape[-2:]:
            image_pe = F.interpolate(image_pe, size=image_embeddings.shape[-2:], mode='bilinear')

        src = image_embeddings + dense_prompt_embeddings  #
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        # b, c, h, w = upscaled_embedding.shape
        # masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return (hyper_in, upscaled_embedding, iou_pred)


class ReinMaskDecoderV2(MaskDecoder):
    def __init__(self, replace_query_feat=True, **kwargs):
        super().__init__(**kwargs)
        transformer_dim = kwargs["transformer_dim"]
        # del self.query_embed
        del self.mask_tokens
        self.vpt_transforms = nn.ModuleList()
        self.replace_query_feat = replace_query_feat
        if replace_query_feat:
            self.querys2feat = nn.Linear(100, self.num_mask_tokens)

    def predict_masks(
        self,
        # image_embeddings: torch.Tensor,
        x: Tuple[torch.Tensor, torch.Tensor],
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        image_embeddings, query_embed = x

        query_embed = self.querys2feat(query_embed.permute(1, 0))
        query_embed = query_embed.permute(1, 0)
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, query_embed], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1) # [b, ]

        # Expand per-image data in batch direction to be per-mask
        # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        if image_embeddings.shape[-2:] != dense_prompt_embeddings.shape[-2:]:
            dense_prompt_embeddings = F.interpolate(dense_prompt_embeddings,
                                                    size=image_embeddings.shape[-2:],
                                                    mode='bilinear')
        if image_embeddings.shape[-2:] != image_pe.shape[-2:]:
            image_pe = F.interpolate(image_pe, size=image_embeddings.shape[-2:], mode='bilinear')

        src = image_embeddings + dense_prompt_embeddings  #
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        # b, c, h, w = upscaled_embedding.shape
        # masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return (hyper_in, upscaled_embedding, iou_pred)


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


def build_mask_decoder(cfg, is_rein_mode=False):
    if is_rein_mode:
        mask_decoder = ReinMaskDecoderV2(
            num_multimask_outputs=cfg.num_multimask_outputs,
            transformer=TwoWayTransformer(
                depth=cfg.transformer_depth,
                embedding_dim=cfg.transformer_embedding_dim,
                mlp_dim=cfg.transformer_mlp_dim,
                num_heads=cfg.transformer_num_heads,
            ),
            transformer_dim=cfg.transformer_dim,
            iou_head_depth=cfg.iou_head_depth,
            iou_head_hidden_dim=cfg.iou_head_hidden_dim,
        )
    else:
        mask_decoder = MaskDecoder(
            num_multimask_outputs=cfg.num_multimask_outputs,
            transformer=TwoWayTransformer(
                depth=cfg.transformer_depth,
                embedding_dim=cfg.transformer_embedding_dim,
                mlp_dim=cfg.transformer_mlp_dim,
                num_heads=cfg.transformer_num_heads,
            ),
            transformer_dim=cfg.transformer_dim,
            iou_head_depth=cfg.iou_head_depth,
            iou_head_hidden_dim=cfg.iou_head_hidden_dim,
        )
    return mask_decoder
