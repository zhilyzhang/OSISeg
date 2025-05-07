import torch
import torch.nn.functional as F


def load_weight_by_splited_models(weight_path, image_encoder, mask_decoder, prompt_encoder):
    old_ck = torch.load(weight_path, map_location='cpu')

    state_dict = image_encoder.state_dict()
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

    for k, v in old_ck.items():
        if k.startswith('image_encoder'):
            # image_encoder_weights.append(k)
            image_encoder_dict[k[len('image_encoder')+1:]] = v
            num_iew += 1
        elif k.startswith('mask_decoder'):
            # mask_decoder_weights.append(k)
            mask_decoder_dict[k[len('mask_decoder')+1:]] = v
            num_mdw += 1
        elif k.startswith('prompt_encoder'):
            # prompt_encoder_weights.append(k)
            prompt_encoder_dict[k[len('prompt_encoder')+1:]] = v
            num_pew += 1

    image_encoder.load_state_dict(image_encoder_dict)
    mask_decoder.load_state_dict(mask_decoder_dict)
    prompt_encoder.load_state_dict(prompt_encoder_dict)

    if num_iew == len(image_encoder_dict) and num_mdw == len(mask_decoder_dict) and num_pew == len(prompt_encoder_dict):
        print('loading base weight successfully!')
    else:
        print('loading base weight failed!')


if __name__ == '__main__':
    '''split sam weight
    image_encoder, mask_decoder, prompt_encoder
    '''
    # load cac weight

