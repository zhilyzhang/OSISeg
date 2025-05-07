import numpy as np
import torch
import os.path as osp


class common(object):
    task = 'EVLabBuilding_bboxs'  # GDBuilding  EVLabBuilding
    output_dir = '/home/zzl/experiments/InterSegAdapter/paper_projects'
    gpu_id = 0


class data(object):
    data_dir = '/home/zzl/datasets/building_data/EVLab_building'

    train_image_paths = []
    train_ann_paths = []
    val_image_paths = []
    val_ann_paths = []
    list_data_name = ['Taiwan', 'Guangdong', 'Chongqing', 'Zhengzhou', 'Wuhan', 'Xian']
    # list_data_name = ['Guangdong']
    for data_name in list_data_name:
        train_image_paths.append(
            f'{data_dir}/{data_name}/new_clip512_dataset/img_train')
        train_ann_paths.append(
            f'{data_dir}/{data_name}/new_clip512_dataset/train_sbt_boundary_3000')

        val_image_path = f'{data_dir}/{data_name}/new_clip512_dataset/img_val'

        val_ann_path = f'{data_dir}/{data_name}/new_clip512_dataset/val_sbt_boundary_900'
        if osp.exists(val_image_path) and osp.exists(val_ann_path):
            val_image_paths.append(val_image_path)
            val_ann_paths.append(val_ann_path)


class image_encoder(object):
    adapter = "sam_para"
    vit_name = 'vit_b'
    encoder_depth = 12
    encoder_embed_dim = 768
    image_size = (512, 512)
    patch_size = 16
    window_size = 14
    mlp_ratio = 4
    qkv_bias = True
    use_rel_pos = True
    out_channels = 256
    multi_outputs = True
    encoder_global_attn_indexes = [2, 5, 8, 11]  #default: vit_b 4
    # encoder_global_attn_indexes = [1,3,5,7,9,11]  # vit_b 改进版本，逐层修正。
    # encoder_global_attn_indexes = list(range(encoder_depth))  # vit_b 改进版本，逐层修正。


class prompt_encoder(object):
    embed_dim = 256
    input_image_size = image_encoder.image_size
    image_embedding_size = (input_image_size[0]//image_encoder.patch_size,
                            input_image_size[0]//image_encoder.patch_size)
    mask_in_chans = 16


class mask_decoder(object):
    transformer_depth = 2
    transformer_embedding_dim = 256
    transformer_mlp_dim = 2048
    transformer_num_heads = 8

    num_multimask_outputs = 3
    transformer_dim = 256
    iou_head_depth = 3
    iou_head_hidden_dim = 256


class model(object):

    image_encoder = image_encoder
    prompt_encoder = prompt_encoder
    mask_decoder = mask_decoder
    use_refine_decoder = ''  # refine_feature_decoder  refine_decoder


class train(object):
    seed = 666
    batch_size = 8
    num_workers = 8
    epochs = 36
    debug = False
    choice_point_type = 'center_point'
    prompt_types = [2]
    ''' 提示类型 ：
                    ① 只有一个正样本点；
                    ② 只有一个bbox
                    ③ 在①或②的基础上，增加1-3次正样本点和负样本点
                    ④ 一笔画
                    '''


class test(object):
    predict_flag = False
    checkpoint = ''
    sam_checkpoint = '/home/zzl/codes/InterSegAdapter/pre_weights/sam_vit_b_01ec64.pth'
    output_file = ''

