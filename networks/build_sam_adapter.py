import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os

# sam
from networks.sam.prompt_encoder import PromptEncoder, build_prompt_encoder
from networks.sam.mask_decoder import MaskDecoder, build_mask_decoder
from networks.sam.weight_transform import load_weight_by_splited_models, load_weight_for_mask_prompt
from functools import partial
from networks.sam.fuse_module import AdapterDecoratedLayers, AdapterFeatureDecoration,\
    AdapterFeatureDecorationUpUp, OurDecorator, \
    AdapterDecoratedLayersFlowV2, OurDecoratorAdd

from networks.sam.refine_decoder import RefineDecoder, RefineFeatureDecoder
from networks.sam.refine_module import RefineUNet, RefineUNetFeatures


class SAM(nn.Module):
    mask_threshold: float = 0.0
    def __init__(self,
                 image_encoder: None,
                 prompt_encoder: PromptEncoder,
                 mask_decoder: MaskDecoder,
                 refine_decoder=None,
                 fuse_feature_module=None):
        super().__init__()

        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.fuse_feature_module = fuse_feature_module
        self.refine_decoder = refine_decoder

    @torch.no_grad()
    def test_mask_forward(self, image_embeddings, ref_label, img_pad_shape):
        low_res_masks, iou_predictions = self.mask_decoder.test_forward(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=ref_label['sparse_embeddings'],
            dense_prompt_embeddings=ref_label['dense_embeddings'],
            multimask_output=False
        )
        masks = F.interpolate(
            low_res_masks,
            img_pad_shape,
            mode="bilinear",
            align_corners=False,
        )
        masks = masks > self.mask_threshold
        return masks.float()


def build_backbone(cfg):
    if cfg.vit_name == 'vit_b':
        encoder_embed_dim = 768
        encoder_depth = 12
        encoder_num_heads = 12
        encoder_global_attn_indexes = [2, 5, 8, 11]
    elif cfg.vit_name == 'vit_l':
        encoder_embed_dim = 1024
        encoder_depth = 24
        encoder_num_heads = 16
        encoder_global_attn_indexes = [5, 11, 17, 23],
    elif cfg.vit_name == 'vit_h':
        encoder_embed_dim = 1280
        encoder_depth = 32
        encoder_num_heads = 16
        encoder_global_attn_indexes = [7, 15, 23, 31]
    else:
        print(f'vit_name: {cfg.vit_name} is not exist!')
        return
    img_size = cfg.image_size
    vit_patch_size = 16
    prompt_embed_dim = 256

    if len(cfg.encoder_global_attn_indexes) > 0:
        encoder_global_attn_indexes = cfg.encoder_global_attn_indexes

    if cfg.adapter == 'sam_self':
        from networks.sam.image_encoder import ImageEncoderViT
    elif cfg.adapter == 'sam_mix':
        from networks.sam.image_encoder_mix import ImageEncoderViT
    elif cfg.adapter == 'sam_mix_ours':
        from networks.sam.image_encoder_mix_ours import ImageEncoderViT

    elif cfg.adapter == 'sam_lora':
        from networks.sam.image_encoder_lora import ImageEncoderViT

    elif cfg.adapter == 'sam_lora_conv':
        from networks.sam.image_encoder_lora_conv import ImageEncoderViT

    elif cfg.adapter == 'sam_para_conv':
        from networks.sam.image_encoder_para_conv import ImageEncoderViT

    elif cfg.adapter == 'sam_para_ablation1':
        from networks.sam.image_encoder_para_ours_ablation1 import ImageEncoderViT

    elif cfg.adapter == 'sam_para_ablation2':
        from networks.sam.image_encoder_para_ours_ablation2 import ImageEncoderViT

    elif cfg.adapter == 'sam_para_two_conv':
        from networks.sam.image_encoder_para_two_conv import ImageEncoderViT

    elif cfg.adapter == 'sam_para_aspp':
        from networks.sam.image_encoder_para_adapter_aspp import ImageEncoderViT

    elif cfg.adapter == 'AdaptFormer':
        from networks.sam.AdaptFormer.image_encoder_AdaptFormer import ImageEncoderViT

    elif cfg.adapter == 'vpt':
        from networks.sam.AdaptFormer.image_encoder_VPT import ImageEncoderViT

    elif cfg.adapter == 'sam_adapter':
        from networks.sam.image_encoder_adapter import ImageEncoderViT

    elif cfg.adapter == 'sam_para':
        from networks.sam.image_encoder_para import ImageEncoderViT

    elif cfg.adapter == 'sam_series':
        from networks.sam.image_encoder_series import ImageEncoderViT

    elif cfg.adapter == 'sam_series_ours':
        from networks.sam.image_encoder_series_ours import ImageEncoderViT

    elif cfg.adapter == 'sam_rein':
        from networks.sam.image_encoder_rein import ImageEncoderViT
    elif cfg.adapter == 'sam_rein_lora':
        from networks.sam.image_encoder_rein_lora import ImageEncoderViT
    else:
        from networks.sam.image_encoder import ImageEncoderViT

    image_encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=img_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
        multi_outputs=cfg.multi_outputs
    )
    return image_encoder


def build_model(cfg):
    image_encoder = build_backbone(cfg.model.image_encoder)
    prompt_encoder = build_prompt_encoder(cfg.model.prompt_encoder)
    # mask_decoder = build_mask_decoder(cfg.model.mask_decoder, is_rein_mode=cfg.model.mask_decoder.is_rein_mode)
    mask_decoder = build_mask_decoder(cfg.model.mask_decoder, is_rein_mode=False)
    # if cfg.model.image_encoder.adapter == 'sam_ours':
    #     fuse_feature_module = AdapterFusedFeature(in_channels=cfg.model.image_encoder.encoder_embed_dim,
    #                                               out_channel=cfg.model.image_encoder.out_channels)
    if cfg.model.image_encoder.adapter == 'sam_ourdec':
        fuse_feature_module = AdapterFeatureDecoration(in_channels=cfg.model.image_encoder.encoder_embed_dim,
                                                  out_channel=cfg.model.image_encoder.out_channels)

    elif cfg.model.image_encoder.adapter == 'sam_decorator':
        num_layers = len(cfg.model.image_encoder.encoder_global_attn_indexes)
        fuse_feature_module = OurDecorator(in_channels=cfg.model.image_encoder.encoder_embed_dim,
                                                     out_channel=cfg.model.image_encoder.out_channels,
                                                     num_layers=num_layers)

    elif cfg.model.image_encoder.adapter == 'sam_decorator_add':
        num_layers = len(cfg.model.image_encoder.encoder_global_attn_indexes)
        fuse_feature_module = OurDecoratorAdd(in_channels=cfg.model.image_encoder.encoder_embed_dim,
                                                     out_channel=cfg.model.image_encoder.out_channels,
                                                     num_layers=num_layers)
    # elif cfg.model.image_encoder.adapter == 'sam_decorator_atte':
    #     num_layers = len(cfg.model.image_encoder.encoder_global_attn_indexes)
    #     fuse_feature_module = DecoratorAtte(in_channels=cfg.model.image_encoder.encoder_embed_dim,
    #                                                  out_channel=cfg.model.image_encoder.out_channels,
    #                                                  num_layers=num_layers)
    #
    # elif cfg.model.image_encoder.adapter == 'sam_decorator_att_new':
    #     num_layers = len(cfg.model.image_encoder.encoder_global_attn_indexes)
    #     fuse_feature_module = DecoratorAtteNew(in_channels=cfg.model.image_encoder.encoder_embed_dim,
    #                                                  out_channel=cfg.model.image_encoder.out_channels,
    #                                                  num_layers=num_layers)

    elif cfg.model.image_encoder.adapter == 'sam_ourdec_layer_flowV2':
        num_layers = len(cfg.model.image_encoder.encoder_global_attn_indexes)
        fuse_feature_module = AdapterDecoratedLayersFlowV2(in_channels=cfg.model.image_encoder.encoder_embed_dim,
                                                     out_channel=cfg.model.image_encoder.out_channels,
                                                     num_layers=num_layers)
    # elif cfg.model.image_encoder.adapter == 'sam_ourdecV2':
    #     fuse_feature_module = AdapterFeatureDecorationUpdate(in_channels=cfg.model.image_encoder.encoder_embed_dim,
    #                                               out_channel=cfg.model.image_encoder.out_channels)
    #
    # elif cfg.model.image_encoder.adapter == 'sam_ourdecV3':
    #     fuse_feature_module = AdapterFeatureDecorationUpdateV2(in_channels=cfg.model.image_encoder.encoder_embed_dim,
    #                                               out_channel=cfg.model.image_encoder.out_channels)

    elif cfg.model.image_encoder.adapter == 'sam_ourdecUpup':
        fuse_feature_module = AdapterFeatureDecorationUpUp(in_channels=cfg.model.image_encoder.encoder_embed_dim,
                                                  out_channel=cfg.model.image_encoder.out_channels)

    # elif cfg.model.image_encoder.adapter == 'sam_ourdecUpupV2':
    #     fuse_feature_module = AdapterFeatureDecorationUpUpV2(in_channels=cfg.model.image_encoder.encoder_embed_dim,
    #                                               out_channel=cfg.model.image_encoder.out_channels)
    else:
        fuse_feature_module = None

    if not cfg.test.predict_flag:
        use_lora = True if cfg.model.image_encoder.adapter in ['sam_lora', 'sam_lora_conv'] else False
        if os.path.exists(cfg.test.sam_checkpoint):
            print(f'Loading weights from path: {cfg.test.sam_checkpoint} !')
            load_weight_by_splited_models(cfg.test.sam_checkpoint, image_encoder,
                                          mask_decoder, prompt_encoder, use_lora=use_lora)
        else:
            print(f'Without trained weights!')
            exit(0)

    if cfg.model.use_refine_decoder == 'refine_decoder':
        # refine_decoder = RefineBoundaryDecoder(in_channels=1, out_channels=32)
        refine_decoder = RefineUNet(input_channels=1)
    elif cfg.model.use_refine_decoder == 'refine_feature_decoder':
        refine_decoder = RefineUNetFeatures(input_channels=1)
        # refine_decoder = RefineFeatureBoundaryDecoder(in_channels=1, out_channels=32)
    else:
        refine_decoder = None

    model = SAM(image_encoder=image_encoder,
                prompt_encoder=prompt_encoder,
                mask_decoder=mask_decoder,
                fuse_feature_module=fuse_feature_module,
                refine_decoder=refine_decoder)

    return model
    # return model.to(cfg.test.device)


def build_net(cfg):
    if cfg.model.image_encoder.vit_name == 'mae':
        from networks.backbone.mae import mae_vit_base_patch16
        image_encoder = mae_vit_base_patch16()
        pre_weight_path = '/home/zzl/codes/InterSegAdapter/pre_weights/mae_pretrain_vit_base.pth'
        image_encoder.use_pre_weights(pre_weight_path)

    elif cfg.model.image_encoder.vit_name == 'dk_vitb':
        from networks.backbone.dk_vision_transformer import vit_base
        image_encoder = vit_base()
        pre_weight_path = "/home/zzl/codes/InterSegAdapter/pre_weights/dk_vitb-16.pt"
        image_encoder.use_pre_weights(pre_weight_path)

    elif cfg.model.image_encoder.vit_name == 'cross_scale_mae':
        from networks.backbone.cross_scale_mae import vit_base
        image_encoder = vit_base()
        pre_weight_path = "/home/zzl/codes/InterSegAdapter/pre_weights/cross_scale_mae_base_pretrain.pth"
        image_encoder.use_pre_weights(pre_weight_path)

    elif cfg.model.image_encoder.vit_name == 'convnextv2':
        from networks.backbone.convnextv2 import convnextv2_base
        image_encoder = convnextv2_base()
        pre_weight_path = '/home/zzl/codes/InterSegAdapter/pre_weights/convnextv2_base_22k_384_ema.pt'
        image_encoder.use_pre_weights(pre_weight_path)

    elif cfg.model.image_encoder.vit_name == 'convnext':
        from networks.backbone.convnet import ConvNeXt
        image_encoder = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], drop_path_rate=0.3)
        pre_weight_path = '/home/zzl/codes/InterSegAdapter/pre_weights/epoch400_convnext_base.pth'
        image_encoder.load_state_dict(torch.load(pre_weight_path, map_location='cpu'), strict=False)

    elif cfg.model.image_encoder.vit_name == 'dinov2':
        from networks.backbone.dinov2.vision_transformer import vit_base
        image_encoder = vit_base()
        pre_weight_path = '/home/zzl/codes/InterSegAdapter/pre_weights/dinov2_vitb14_pretrain.pth'
        image_encoder.load_pre_trained_weights(pre_weight_path)

    elif cfg.model.image_encoder.vit_name == 'eva2':
        from networks.backbone.eva2 import EVA2
        image_encoder = EVA2(img_size=512)
        pre_weight_path = '/home/zzl/codes/InterSegAdapter/pre_weights/eva02_B_pt_in21k_medft_in21k_ft_in1k_p14.pt'
        image_encoder.load_pre_trained_weights(pre_weight_path)

    else:
        print(f'loading backbone error! {cfg.model.image_encoder.vit_name}')
        exit()

    prompt_encoder = build_prompt_encoder(cfg.model.prompt_encoder)
    mask_decoder = build_mask_decoder(cfg.model.mask_decoder, is_rein_mode=cfg.model.mask_decoder.is_rein_mode)

    if not cfg.test.predict_flag:
        use_lora = True if cfg.model.image_encoder.adapter in ['sam_lora', 'sam_lora_conv'] else False
        if os.path.exists(cfg.test.sam_checkpoint):
            print(f'Loading weights from path: {cfg.test.sam_checkpoint} !')
            load_weight_for_mask_prompt(cfg.test.sam_checkpoint,
                                          mask_decoder, prompt_encoder, use_lora=use_lora)
        else:
            print(f'Without trained weights!')
            exit(0)

    model = SAM(image_encoder=image_encoder,
                prompt_encoder=prompt_encoder,
                mask_decoder=mask_decoder)

    return model
    # return model.to(cfg.test.device)


if __name__ == '__main__':
    import importlib
    cfg = importlib.import_module('configs.rsseg')
    samples = torch.randn(2, 3, 512, 512)
    patches = {
        "scale_embedding": torch.ones(2, 3) * 19,
        "batch_boxes": torch.from_numpy(np.array([[[10, 10, 12, 13]]]).repeat(2, axis=0).repeat(3, axis=1))
    }
    print(patches['batch_boxes'].shape)
    patches['scale_embedding'] = patches['scale_embedding'].long()

    model = build_model(cfg)
    outputs = model(samples, patches)
    print(outputs.shape)

