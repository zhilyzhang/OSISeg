import os, sys
import os.path as osp
root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, root)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
import torch
from dataset.landcover_wood_dataset import WoodLandDataset
from networks.build_sam_adapter import build_model
from tensorboardX import SummaryWriter
from utils.logger import get_logger
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
from utils.vis_tensor import vis_pure_tensor, vis_tensor
from torch.utils.data import DataLoader
import scripts.measures as measures
warnings.filterwarnings(action='ignore')
torch.multiprocessing.set_sharing_strategy('file_system')
# 在代码里面设置线程数
torch.set_num_threads(1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def dice_loss(y_pred, y_true):
    smooth = 1e-5
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true)
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice_score
    return dice_loss


def train(args):
    cfg = importlib.import_module('configs.landcover_woodland_task.' + args.config_file)
    if args.gpu_id in list(range(8)):
        cfg.common.gpu_id = args.gpu_id
        print(f'gpu_id: {args.gpu_id}')
    cfg.model.image_encoder.adapter = args.adapter
    set_seed(cfg.train.seed)
    device = torch.device('cuda:{}'.format(cfg.common.gpu_id))

    output_dirs = osp.join(cfg.common.output_dir,
                           f'{cfg.common.task}_epo_{cfg.train.epochs}_bs_{cfg.train.batch_size}')

    os.makedirs(osp.join(output_dirs, 'train_vis'), exist_ok=True)
    tb_writer = SummaryWriter(output_dirs + '/log')
    filename = datetime.datetime.now().strftime('log_' + '%m_%d_%H')
    logger = get_logger(os.path.join(output_dirs, filename + ".log"))
    logger.info(str(args))
    net = build_model(cfg).to(device)
    # old_ck = torch.load('/home/zzl/experiments/InterSegAdapter/new_projects/sam_decorator_data_update_std/'
    #                     'GDBuilding_epo_36_bs_8/weight_epoch_17_IoU_0.8202.pth', map_location='cpu')
    # new_state_dict = net.state_dict()
    # unload_keys = []
    # for k, v in old_ck.items():
    #     if k.startswith('image_encoder'):
    #         new_state_dict[k] = v
    #
    #     elif k.startswith('mask_decoder'):
    #         new_state_dict[k] = v
    #
    #     elif k.startswith('prompt_encoder'):
    #         new_state_dict[k] = v
    #
    #     elif k.startswith('fuse_feature_module'):
    #         new_state_dict[k] = v
    #
    #     else:
    #         unload_keys.append(k)
    # print(unload_keys)
    # net.load_state_dict(new_state_dict)
    debug = False
    train_dataset = WoodLandDataset(
        image_dir=cfg.data.train_image_path,
        gt_mask_dir=cfg.data.train_ann_path,
        debug=debug,
        mode='train',
        model=net,
        device=device,
        batch_size=cfg.train.batch_size,
        adapter=cfg.model.image_encoder.adapter,
        choice_point_type=cfg.train.choice_point_type,
        prompt_types=cfg.train.prompt_types
    )

    val_dataset = WoodLandDataset(
        image_dir=cfg.data.val_image_path,
        gt_mask_dir=cfg.data.val_ann_path,
        debug=debug,
        mode='test',
        model=net,
        device=device,
        batch_size=1,
        adapter=cfg.model.image_encoder.adapter,
        choice_point_type=cfg.train.choice_point_type,
        prompt_types=cfg.train.prompt_types
    )

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0,
                           amsgrad=False)
    pos_weight = torch.ones([1]).cuda(device=device) * 2
    BCEloss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    max_epoch = cfg.train.epochs
    start_time = time.time()
    best_performance = 0.0
    print(f'Num of samples: {train_dataset.length}')

    for epoch_num in range(1, max_epoch+1):
        train_gen = train_dataset.generate_sample_std()
        net.train()
        for names, param in net.image_encoder.named_parameters():
            if "Adapter" in names:
                param.requires_grad = True
            elif "LoRA" in names:
                param.requires_grad = True
            elif "reins" in names:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for name, module in net.prompt_encoder.named_parameters():
            module.requires_grad = False

        # for name, module in net.fuse_feature_module.named_parameters():
        #     module.requires_grad = False
        #
        # for name, module in net.mask_decoder.named_parameters():
        #     module.requires_grad = False

        n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('Training number of params (M): %.2f' % (n_parameters / 1.e6))
        # exit()
        logger.info('training epoch: %d' % epoch_num)
        epoch_loss = 0
        ind = 0
        vis = 50
        num_print = 30

        for i_batch, batch in enumerate(train_gen):
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

            imges = net.image_encoder(imgs)
            if cfg.model.image_encoder.adapter in ['sam_decorator', 'sam_ourdec_layer_flowV2']:
                imge = net.fuse_feature_module(imges)

            elif cfg.model.image_encoder.adapter == 'sam_rein':
                imge = imges

            else:
                imge = imges[-1]

            with torch.no_grad():
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
                multimask_output=False
            )
            loss = 0
            if cfg.model.use_refine_decoder != '':
                if cfg.model.use_refine_decoder == 'refine_feature_decoder':
                    refine_pred = net.refine_decoder(pred, imge)
                else:
                    refine_pred = net.refine_decoder(pred)

                refine_pred = F.interpolate(refine_pred, size=(args.out_size, args.out_size), mode='bilinear',
                                            align_corners=False)
                sigmoid_refine_pred = torch.sigmoid(refine_pred)
                refine_bce_loss = BCEloss(refine_pred, labels)
                loss += refine_bce_loss
                refine_d_loss = dice_loss(sigmoid_refine_pred, labels)
                loss += refine_d_loss

            pred = F.interpolate(pred, size=(args.out_size, args.out_size), mode='bilinear', align_corners=False)
            sigmoid_outputs = torch.sigmoid(pred)
            bce_loss = BCEloss(pred, labels)
            d_loss = dice_loss(sigmoid_outputs, labels)
            loss += bce_loss
            loss +=  d_loss

            epoch_loss += loss.item()
            loss.backward()
            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

            if cfg.model.use_refine_decoder != '':
                pred = sigmoid_refine_pred
            else:
                pred = sigmoid_outputs

            predict = (pred > 0.5).float()
            non_zero_missed = (labels!=0)

            acc = ((predict == labels) & non_zero_missed).sum().item() / \
                         (non_zero_missed.sum().item() + 1e-9)

            tb_writer.add_scalar('info/loss', loss.item(), ind)
            tb_writer.add_scalar('info/acc', acc, ind)

            if ind % num_print == 0:
                logger.info(
                    'i/iter_epoch/epoch %d/%d/%d : loss : %.4f, acc : %.4f'
                    % (ind, train_dataset.iter_num_per_epoch, epoch_num, loss.item(), acc))
            if vis:
                if ind % vis == 0:
                    vis_pure_tensor(imgs, pred, labels,
                                    save_path=os.path.join(output_dirs, 'train_vis',
                                                      'epoch+' + str(epoch_num) + '.jpg'))
            # pbar.update()
            ind += 1

        net.eval()

        TP, Acc, PN, GN = [], [], [], []
        # 在此次重新加载net，因为权重在更新。当前验证的权重，必须是当前训练的。
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

                elif cfg.model.image_encoder.adapter == 'sam_rein':
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
                    multimask_output=False
                )
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
        IoU = IoU.item()
        logger.info(f'training epoch: {epoch_num}, IoU: {round(IoU * 100, 3)} %')
        save_weight = False
        if IoU > best_performance or (IoU > 0.8 and IoU + 0.05 > best_performance):
            if IoU > best_performance: best_performance = IoU
            save_mode_path = os.path.join(output_dirs, f'weight_epoch_{epoch_num}_IoU_{round(IoU, 4)}.pth')
            torch.save(net.state_dict(), save_mode_path)
            logger.info("save model to {}".format(save_mode_path))
            save_weight = True

        if not save_weight and epoch_num % 10 == 0:
            save_mode_path = os.path.join(output_dirs, f'weight_epoch_{epoch_num}.pth')
            torch.save(net.state_dict(), save_mode_path)
            logger.info("save model to {}".format(save_mode_path))

        duration1 = time.time() - start_time
        start_time = time.time()
        logger.info('Train running time: %.2f(minutes)' % (duration1 / 60))

    tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='cfg_ours')
    parser.add_argument("--net", type=str, default='sam')
    parser.add_argument("--adapter", type=str, default='sam_para_conv')
    #  sam_self, sam_mix, sam_lora sam_para, sam_series, sam_decorator, sam_ourdec_layer_flowV2
    parser.add_argument("--gpu_id", type=int, default=5)
    parser.add_argument("--out_size", type=int, default=512)
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-distributed', default='none',type=str,help='multi GPU ids to use')
    parser.add_argument('-sam_ckpt', default='/home/zzl/codes/InterSegAdapter/pre_weights/sam_vit_b_01ec64.pth',
                        help='sam checkpoint address')
    args = parser.parse_args()

    train(args)
