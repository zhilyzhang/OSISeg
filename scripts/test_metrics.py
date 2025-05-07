import os, sys
import os.path as osp
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, root)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
import torch
# from dataset.interactive_segment_loader import InterSegDataset, batch_collate_fn
from dataset.one_prompt_all_loader import InterSegOnePrompt
# from models.build_net import get_network
from networks.build_sam_adapter import build_model, build_net
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train(args):
    cfg = importlib.import_module('configs.adapter_locations.' + args.config_file)
    if args.gpu_id in list(range(8)):
        cfg.common.gpu_id = args.gpu_id
        print(f'gpu_id: {args.gpu_id}')

    cfg.test.predict_flag = True if args.predict_flag == 'true' else False
    cfg.model.image_encoder.adapter = args.adapter
    cudnn.benchmark = False
    cudnn.deterministic = True
    cudnn.enabled = True
    device = torch.device('cuda:{}'.format(cfg.common.gpu_id))

    if not os.path.exists(args.checkpoints):
        checkpoint_path = cfg.test.checkpoint
    else:
        checkpoint_path = args.checkpoints
    if not os.path.exists(checkpoint_path):
        print('load weight errors!')
        exit()

    net = build_model(cfg).to(device)
    net.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    print(f'loaded weight: {checkpoint_path}')
    net.eval()
    TP, Acc, PN, GN = [], [], [], []

    val_dataset = InterSegOnePrompt(
        image_paths=cfg.data.val_image_paths,
        ann_paths=cfg.data.val_ann_paths,
        debug=False,
        mode='test',
        model=net,
        device=device,
        batch_size=1,
        adapter=cfg.model.image_encoder.adapter,
        choice_point_type=cfg.train.choice_point_type,
        prompt_types=cfg.train.prompt_types
    )

    print(f'Num of samples: {val_dataset.length}')
    val_gen = val_dataset.generate_sample_std()
    for i_batch, batch in tqdm(enumerate(val_gen)):
        imgs = batch['image'].to(dtype=torch.float32, device=device)
        labels = batch['label'].to(dtype=torch.float32, device=device)
        filename = batch['filename']
        points = batch['prompt_point']
        if points is not None:
            point_xy, point_label = list(points)
            point_xy = point_xy.to(device)
            point_label = point_label.to(device)
            points = (point_xy, point_label)
        boxes = batch['prompt_box']
        if boxes is not None:
            boxes = boxes.to(device)
        prompt_masks = batch['prompt_mask'].to(dtype=torch.float32, device=device) if batch['prompt_mask'] is not None else None
        prompt_type = batch['prompt_type']
        with torch.no_grad():
            imges = net.image_encoder(imgs)
            if cfg.model.image_encoder.adapter in ['sam_decorator', 'sam_ourdec_layer_flowV2']:
                imge = net.fuse_feature_module(imges)

            elif cfg.model.image_encoder.adapter in ['sam_rein', 'sam_rein_lora']:
                imge = imges

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
        tp, num_predict, num_gt, acc = measures.calculate_tp_pn_gn_accuracy(predict.squeeze(1), labels.squeeze(1),
                                                                            threshold=0.5)
        TP.append(tp)
        PN.append(num_predict)
        GN.append(num_gt)
        Acc.append(acc)

    IoU = sum(TP) / (sum(PN) + sum(GN) - sum(TP) + 1e-9)
    Recall = sum(TP) / (sum(GN) + 1e-9)
    Precision = sum(TP) / (sum(PN) + 1e-9)
    F1 = 2 * Precision * Recall / (Precision + Recall)
    Accuracy = sum(Acc) / len(Acc)

    print('IoU: %.3f%%' % (IoU.item() * 100))
    print('Recall: %.3f%%' % (Recall.item() * 100))
    print('Precision: %.3f%%' % (Precision.item() * 100))
    print('Accuracy: %.3f%%' % (Accuracy.item() * 100))
    print('F1: %.3f%%' % (F1.item() * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='cfg_evlab_building_ours')
    parser.add_argument("--net", type=str, default='vit_b')
    parser.add_argument("--adapter", type=str, default='sam_para_conv')  # none: sam
    parser.add_argument("--gpu_id", type=int, default=5)
    parser.add_argument("--out_size", type=int, default=512)
    parser.add_argument('--predict_flag', default='true',type=str)
    parser.add_argument('--checkpoints', default='',
                        help='sam checkpoint address')
    args = parser.parse_args()

    train(args)

