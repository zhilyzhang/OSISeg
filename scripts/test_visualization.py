import os, sys
import os.path as osp
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, root)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
import torch
# from dataset.interactive_segment_loader import InterSegDataset, batch_collate_fn
from dataset.one_prompt_all_loader import InterSegOnePrompt, image_transform
# from models.build_net import get_network
from networks.build_sam_adapter import build_model
import scripts.measures as measures
import importlib
import datetime
import time
from tqdm import tqdm
import random
import numpy as np
import torch.backends.cudnn as cudnn
import warnings
import torch.optim as optim
import torch.nn.functional as F
from utils.vis_tensor import vis_image, vis_tensor
from torch.utils.data import DataLoader
warnings.filterwarnings(action='ignore')
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train(args, is_vis=True, predict_times=3, n_sample=1000):
    cfg = importlib.import_module('configs.prompt_types.' + args.config_file)
    if args.gpu_id in list(range(8)):
        cfg.common.gpu_id = args.gpu_id
        print(f'gpu_id: {args.gpu_id}')

    checkpoint_path = ''
    if not os.path.exists(args.checkpoints):
        checkpoint_path = cfg.test.checkpoint
    else:
        checkpoint_path = args.checkpoints
    if not os.path.exists(checkpoint_path):
        print('load weight errors!')
        exit()

    cfg.test.predict_flag = True if args.predict_flag == 'true' else False
    cfg.model.image_encoder.adapter = args.adapter
    cudnn.benchmark = False
    cudnn.deterministic = True
    cudnn.enabled = True
    device = torch.device('cuda:{}'.format(cfg.common.gpu_id))

    net = build_model(cfg).to(device)
    net.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    net.eval()

    # val_dataset = InterSegOnePrompt(
    #     image_paths=cfg.data.val_image_paths,
    #     ann_paths=cfg.data.val_ann_paths,
    #     debug=False,
    #     mode='val',
    #     model=net,
    #     device=device,
    #     batch_size=1,
    #     adapter=cfg.model.image_encoder.adapter,
    #     prompt_types=cfg.train.prompt_types
    # )
    for i_predict in range(predict_times):
        save_vis_dir = osp.join(osp.dirname(checkpoint_path), f'vis_sample/test_{i_predict}')
        os.makedirs(save_vis_dir, exist_ok=True)

        val_dataset = InterSegOnePrompt(
            image_paths=cfg.data.train_image_paths,
            ann_paths=cfg.data.train_ann_paths,
            debug=False,
            mode='train',
            model=net,
            device=device,
            batch_size=1,
            adapter=cfg.model.image_encoder.adapter,
            prompt_types=cfg.train.prompt_types
        )

        print(f'Num of samples: {val_dataset.length}')
        val_gen = val_dataset.generate_sample_std()
        for i_batch, batch in tqdm(enumerate(val_gen)):
            if n_sample != -1 and i_batch > n_sample:
                break
            imgs = batch['image'].to(dtype=torch.float32, device=device)
            labels = batch['label'].to(dtype=torch.float32, device=device)
            filename = batch['filename'][0]
            points = batch['prompt_point']
            boxes = batch['prompt_box']
            prompt_masks = batch['prompt_mask']
            prompt_type = batch['prompt_type']
            with torch.no_grad():
                imges = net.image_encoder(imgs)
                if cfg.model.image_encoder.adapter in ['sam_decorator', 'sam_ourdec_layer_flowV2']:
                    imge = net.fuse_feature_module(imges)
                else:
                    imge = imges[-1]

                se, de = net.prompt_encoder(
                    points=points,
                    boxes=boxes,
                    masks=prompt_masks,
                )
                pred, _ = net.mask_decoder.test_forward(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=False)

                if cfg.model.use_refine_decoder != '':
                    if cfg.model.use_refine_decoder == 'refine_feature_decoder':
                        pred = net.refine_decoder(pred, imge)
                    else:
                        pred = net.refine_decoder(pred)

            pred = F.interpolate(pred, size=(args.out_size, args.out_size), mode='bilinear', align_corners=False)
            predict = torch.sigmoid(pred)

            if is_vis:
                alpha = 0.75
                marker_size = 200
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Change to 1 row, 4 columns
                fig.tight_layout()
                image = image_transform(batch['image'][0])
                gt_mask = labels.cpu().numpy()[0][0]
                pred_mask = predict.cpu().numpy()[0][0] > 0.5

                # image
                axs[0].imshow(image, cmap='jet')
                axs[0].set_title('image')
                axs[0].axis('off')

                # gt-mask
                rgb_color = [255, 255, 0]  # 黄色
                rgb_mask = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
                rgb_mask[gt_mask > 0] = rgb_color
                gt_result = image * (1 - gt_mask[:, :, np.newaxis]) + \
                         (1 - alpha) * gt_mask[:, :, np.newaxis] * image + \
                         alpha * rgb_mask
                gt_result = gt_result.astype(np.uint8)
                axs[1].imshow(gt_result)
                axs[1].set_title('gt')
                axs[1].axis('off')

                # image/gt-check
                rgb_color = [255, 255, 0]  # 黄色
                rgb_mask = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
                rgb_mask[pred_mask > 0] = rgb_color
                result = image * (1 - pred_mask[:, :, np.newaxis]) + \
                         (1 - alpha) * pred_mask[:, :, np.newaxis] * image + \
                         alpha * rgb_mask
                result = result.astype(np.uint8)
                axs[2].imshow(result)

                if prompt_type == 1:  # point
                    points = batch['prompt_point'][0][0].cpu().numpy()
                    x = [point[0] for point in points]
                    y = [point[1] for point in points]
                    axs[2].scatter(x, y, color='g', marker='*', s=marker_size)  # mistaken_points

                elif prompt_type == 2:  # box
                    box = batch['prompt_box'][0].cpu().numpy()
                    x1, y1, x2, y2 = box
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='green', facecolor='none')
                    axs[2].add_patch(rect)

                elif prompt_type == 4:
                    points = batch['prompt_point'][0][0].cpu().numpy()
                    x = [point[0] for point in points]
                    y = [point[1] for point in points]
                    point_labels = batch['prompt_point'][1][0]
                    for i in range(len(x)):
                        if point_labels[i] == 1:
                            axs[2].scatter(x[i], y[i], color='g', marker='*', s=marker_size)  # mistaken_points
                        else:
                            axs[2].scatter(x[i], y[i], color='r', marker='*', s=marker_size)  # mistaken_points

                else:
                    points = batch['prompt_point'][0][0].cpu().numpy()
                    x = [point[0] for point in points]
                    y = [point[1] for point in points]
                    point_labels = batch['prompt_point'][1][0]
                    if point_labels == 1:
                        axs[2].scatter(x, y, color='g', marker='*', s=marker_size)  # mistaken_points
                    else:
                        axs[2].scatter(x, y, color='r', marker='*', s=marker_size)  # mistaken_points

                axs[2].set_title('pred-mask')
                axs[2].axis('off')
                plt.show()
                # plt.savefig(osp.join(save_vis_dir, filename + '.tif'), dpi=200)
                plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='cfg_evlab_building_ours')
    parser.add_argument("--net", type=str, default='sam')
    parser.add_argument("--adapter", type=str, default='sam_para')  # none: sam
    parser.add_argument("--gpu_id", type=int, default=7)
    parser.add_argument("--out_size", type=int, default=512)
    parser.add_argument('--predict_flag', default='true' ,type=str)
    parser.add_argument('--checkpoints', default='',
                        help='sam checkpoint address')
    args = parser.parse_args()

    train(args)
