import torch
import torch.nn.functional as F


def weight_transfer_based_sam(model,
                              checkpoint_model):
    '''
    image_encoder: rel_pos h/w, pos_embed;
    others: all in
    '''
    state_dict = model.state_dict()
    pos_params = [k for k in checkpoint_model if 'pos' in k]
    for pos in pos_params:
        # new_pos = pos.replace('image_encoder', 'backbone')
        if 'pos_embed' in pos:
            ori_size = checkpoint_model[pos].shape[-2]
            # new_size = state_dict[new_pos].shape[-2]
            new_size = state_dict[pos].shape[-2]
            if new_size != ori_size:
                # print(f'{pos} interpolate from {ori_size}x{ori_size} to {new_size}x{new_size}')
                pos_tokens = checkpoint_model[pos].permute(0, 3, 1, 2)
                pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1)
                checkpoint_model[pos] = pos_tokens

        elif 'rel_pos' in pos:
            rel_pos = checkpoint_model[pos]
            # new_size = state_dict[new_pos].shape[0]
            new_size = state_dict[pos].shape[0]
            ori_size = rel_pos.shape[0]
            if new_size != ori_size:
                # print(f'{pos} interpolate from {rel_pos.shape} to {state_dict[new_pos].shape}')
                rel_pos_resized = F.interpolate(
                    rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
                    size=new_size,
                    mode="linear",
                )
                rel_pos_resized = rel_pos_resized.reshape(-1, new_size).permute(1, 0)
            else:
                rel_pos_resized = rel_pos
            checkpoint_model[pos] = rel_pos_resized

    # checkpoint_model to model
    match_k = [k for k in state_dict.keys()]
    num_load_cpts = 0
    for k, v in checkpoint_model.items():
        new_k = k.replace('image_encoder', 'backbone')
        # if new_k not in ['mask_decoder.iou_prediction_head.layers.1.weight',
        #                  'mask_decoder.iou_prediction_head.layers.1.bias',
        #                  'mask_decoder.mask_tokens.weight'] and new_k in state_dict.keys():
        if new_k in state_dict.keys():
            state_dict[new_k] = v
            num_load_cpts += 1
            match_k.remove(new_k)
        else:
            print(f'{new_k} is not matched!')

    print(f'SAM transfer weight: pre-num{len(checkpoint_model)}/load-num{num_load_cpts}')

    # load cac weight
    cac_weight_path = r'D:\codes\InterFormer-master\work_dirs\cac_weights\cac_based_sam_rsV2_model_best.pth'
    cac_weight = torch.load(cac_weight_path, map_location='cpu')
    cac_weight = cac_weight['model']
    for k, v in cac_weight.items():
        if 'cac_decoder' in k:
            if k in state_dict.keys():
                state_dict[k] = v
                num_load_cpts += 1
                match_k.remove(k)
            else:
                print(f'key: {k} error!')
    print(f'Loaded CAC weight: new-num: {len(state_dict)} / load-num: {num_load_cpts}')
    print(f'without matching keys: {match_k}')
    return state_dict


def weight_alter_name(model, checkpoint):
    # checkpoint = torch.load(cfg.VAL.resume, map_location='cpu')
    old_ck = checkpoint['model']
    new_ck = {}
    new_ck_dict = model.state_dict()
    n = 0
    for k, v in old_ck.items():
        if 'image_encoder' in k:
            k = k.replace('image_encoder', 'backbone')
        if k in new_ck_dict.keys():
            new_ck[k] = v
            n += 1
    # model.load_state_dict(checkpoint['net'])
    model.load_state_dict(new_ck)
    print(n, len(new_ck_dict))
    return model


def load_weight_by_splited_models(weight_path, image_encoder, mask_decoder, prompt_encoder, use_lora=False):
    old_ck = torch.load(weight_path, map_location='cpu')

    state_dict = image_encoder.state_dict()
    # print(state_dict.keys())
    # print(old_ck.keys())
    # exit()
    pos_params = [k for k in old_ck if 'pos' in k]
    for pos in pos_params:
        # new_pos = pos.replace('image_encoder', 'backbone')
        if 'pos_embed' in pos:
            ori_size = old_ck[pos].shape[-2]
            new_pos = pos[len('image_encoder.'):]
            new_size = state_dict[new_pos].shape[-2]
            if new_size != ori_size:
                # print(f'{pos} interpolate from {ori_size}x{ori_size} to {new_size}x{new_size}')
                pos_tokens = old_ck[pos].permute(0, 3, 1, 2)
                pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1)
                old_ck[pos] = pos_tokens

        elif 'rel_pos' in pos:
            rel_pos = old_ck[pos]
            new_pos = pos[len('image_encoder.'):]
            new_size = state_dict[new_pos].shape[0]
            ori_size = rel_pos.shape[0]
            if new_size != ori_size:
                # print(f'{pos} interpolate from {rel_pos.shape} to {state_dict[new_pos].shape}')
                rel_pos_resized = F.interpolate(
                    rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
                    size=new_size,
                    mode="linear",
                )
                rel_pos_resized = rel_pos_resized.reshape(-1, new_size).permute(1, 0)
            else:
                rel_pos_resized = rel_pos
            old_ck[pos] = rel_pos_resized

    num_iew = 0
    num_mdw = 0
    num_pew = 0
    image_encoder_dict = image_encoder.state_dict()
    mask_decoder_dict = mask_decoder.state_dict()
    prompt_encoder_dict = prompt_encoder.state_dict()
    unloaded_keys = []
    for k, v in old_ck.items():
        if k.startswith('image_encoder'):
            # image_encoder_weights.append(k)
            if use_lora and 'attn.qkv' in k:
                k = k.replace('attn.qkv', 'attn.qkv.qkv')
            if k[len('image_encoder')+1:] in image_encoder_dict.keys():
                image_encoder_dict[k[len('image_encoder')+1:]] = v
                num_iew += 1
            else:
                unloaded_keys.append(k)

        elif k.startswith('mask_decoder'):
            # mask_decoder_weights.append(k)
            new_k = k[len('mask_decoder')+1:]
            if new_k in mask_decoder_dict.keys():
                mask_decoder_dict[new_k] = v
                num_mdw += 1

        elif k.startswith('prompt_encoder'):
            # prompt_encoder_weights.append(k)
            prompt_encoder_dict[k[len('prompt_encoder')+1:]] = v
            num_pew += 1

    image_encoder.load_state_dict(image_encoder_dict)
    mask_decoder.load_state_dict(mask_decoder_dict)
    prompt_encoder.load_state_dict(prompt_encoder_dict)

    if num_iew == len(image_encoder_dict): #and num_mdw == len(mask_decoder_dict) and num_pew == len(prompt_encoder_dict):
        print(f'loading image encoder weight successfully!')
    else:
        print('loading image encoder weight question!')
        print(f'missing preweight: {unloaded_keys}')

    if num_mdw == len(mask_decoder_dict):
        print(f'loading mask_decoder weight successfully!')
    else:
        print(f'loaded/new_num:{num_mdw}/{len(mask_decoder_dict)}: loading mask_decoder weight failed!')

    if num_pew == len(prompt_encoder_dict):
        print(f'loading prompt_encoder weight successfully!')
    else:
        print('loading prompt_encoder weight failed!')


def load_weight_for_mask_prompt(weight_path, mask_decoder, prompt_encoder, use_lora=False):
    old_ck = torch.load(weight_path, map_location='cpu')

    num_mdw = 0
    num_pew = 0
    mask_decoder_dict = mask_decoder.state_dict()
    prompt_encoder_dict = prompt_encoder.state_dict()
    for k, v in old_ck.items():
        if k.startswith('mask_decoder'):
            # mask_decoder_weights.append(k)
            new_k = k[len('mask_decoder')+1:]
            if new_k in mask_decoder_dict.keys():
                mask_decoder_dict[new_k] = v
                num_mdw += 1

        elif k.startswith('prompt_encoder'):
            # prompt_encoder_weights.append(k)
            prompt_encoder_dict[k[len('prompt_encoder')+1:]] = v
            num_pew += 1
    mask_decoder.load_state_dict(mask_decoder_dict)
    prompt_encoder.load_state_dict(prompt_encoder_dict)

    if num_mdw == len(mask_decoder_dict):
        print(f'loading mask_decoder weight successfully!')
    else:
        print(f'loaded/new_num:{num_mdw}/{len(mask_decoder_dict)}: loading mask_decoder weight failed!')

    if num_pew == len(prompt_encoder_dict):
        print(f'loading prompt_encoder weight successfully!')
    else:
        print('loading prompt_encoder weight failed!')


def load_cac_decoder_weight(cac_decoder):
    cac_decoder_num = 0
    cac_weight_path = r'D:\codes\InterFormer-master\work_dirs\cac_weights\cac_based_sam_rsV2_model_best.pth'
    cac_weight = torch.load(cac_weight_path, map_location='cpu')
    cac_weight = cac_weight['model']
    cac_decoder_dict = cac_decoder.state_dict()
    for k, v in cac_weight.items():
        if 'cac_decoder' in k:
            cac_decoder_dict[k[len('cac_decoder') + 1:]] = v
            cac_decoder_num += 1
    cac_decoder.load_state_dict(cac_decoder_dict)
    if cac_decoder_num == len(cac_decoder_dict):
        print('loading cac weight successfully!')
    else:
        print('loading cac weight failed!')


if __name__ == '__main__':
    '''split sam weight
    image_encoder, mask_decoder, prompt_encoder
    '''
    # load cac weight

