import importlib
import os.path

import torch
from glob import glob
import copy
import os.path as osp
import numpy as np
from dataset.utils import polys2mask, random_click, scale_bbox_xywh_to_xyxy, \
    cal_center_point_for_binary_mask, \
    calculate_iou, random_dilate_or_erode, \
    select_random_pointsV2
import json
from PIL import Image
import matplotlib.pyplot as plt
from dataset.before_data_sampler.scribble_sampler import generate_spline_point_within_mask
from tqdm import tqdm


class InterSegOnePrompt(object):
    '''
    points, bbox, 一次交互情况，
    '''
    def __init__(self,
                 image_paths,
                 ann_paths,
                 debug=False,
                 mode='train',
                 model=None,
                 device=None,
                 box_scales=[0.8, 1.2],
                 box_ext_pixel=6,
                 batch_size=8,
                 adapter='sam_ours',
                 choice_point_type='center_point',
                 prompt_types=[1]):
        assert len(image_paths) == len(ann_paths)
        self.image_paths = image_paths
        self.ann_paths = ann_paths
        self.mode = mode
        self.data_pts = []
        if self.mode == 'test':
            self.data_pts = glob(osp.join('/home/zzl/datasets/building_data/EVLab_building/test_data', '*.pt'))
        self.gt_json_paths = [self.get_samples_from_json_path(p) for p in self.ann_paths]

        if debug:
            if self.mode == 'test':
                self.data_pts = self.data_pts[:200]
            else:
                gt_json_paths = copy.deepcopy(self.gt_json_paths)
                self.gt_json_paths = [s[:20] for s in gt_json_paths if len(s) > 6]
                del gt_json_paths
        self.model = model
        self.device = device
        self.box_scales = box_scales
        self.box_ext_pixel = box_ext_pixel

        if self.mode == 'test':
            self.length = len(self.data_pts)
        else:
            self.length = sum([len(p) for p in self.gt_json_paths])

        self.prompt_types = prompt_types
        self.choice_point_type = choice_point_type  # 'center_point'
        self.image_embeddings = None
        self.adapter = adapter
        self.batch_size = batch_size
        self.iter_num_per_epoch = self.length // batch_size
        self.IoU_threshold = 0.6

    def image_transform(self, image):
        pixel_mean = [123.675, 116.28, 103.53]
        pixel_std = [58.395, 57.12, 57.375]
        pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        image = torch.from_numpy(np.array(image).transpose((2, 0, 1)))
        x = (image - pixel_mean) / pixel_std
        return x

    def mask_transform(self, mask):
        '''shape: m×n, 0-1'''
        tensor = torch.from_numpy(np.asarray(mask)).float().unsqueeze(0)
        return tensor

    def get_samples_from_json_path(self, json_path):
        list_json_path = glob(osp.join(json_path, '*.json'))
        return list_json_path

    def rand(self, a=0., b=1.):
        return np.random.rand()*(b-a) + a

    def ann2mask(self, ann, shape):
        polys = []
        for segmentation in ann['segmentation']:
            poly = np.array(segmentation).reshape((-1, 2))
            if poly.shape[0] < 3: continue
            polys.append(poly)
        mask = polys2mask(polys, shape=shape)
        return mask

    def ann2bbox(self, ann, shape):
        scale = self.rand(self.box_scales[0], self.box_scales[1])
        bbox = scale_bbox_xywh_to_xyxy(ann['bbox'],
                                       W=shape[1],
                                       H=shape[0],
                                       scale=scale,
                                       ext_pixel=self.box_ext_pixel)
        return bbox

    def pred_gt_mask2prompt_point(self, pred, gt_mask):
        pred_mask = pred.cpu().numpy()[0][0]
        missed_mask = np.maximum(gt_mask - pred_mask, 0)
        mistaken_mask = np.maximum(pred_mask - gt_mask, 0)
        # mode 1:
        if np.sum(mistaken_mask) > np.sum(missed_mask):
            point_label = 0
            # point_coords = sample_points_from_binary_mask(mistaken_mask, min_area=20, border_width=2)
            # if len(point_coords) == 0:
            if self.choice_point_type == 'center_point':
                point_coord = cal_center_point_for_binary_mask(mistaken_mask)
            else:
                point_coord = random_click(np.array(mistaken_mask), value=1)
            point_coords = [point_coord]
        else:
            point_label = 1
            # point_coords = sample_points_from_binary_mask(missed_mask, min_area=20, border_width=2)
            # if len(point_coords) == 0:
                # point_coord = random_click(missed_mask, value=1)
            if self.choice_point_type == 'center_point':
                point_coord = cal_center_point_for_binary_mask(missed_mask)
            else:
                point_coord = random_click(np.array(missed_mask), value=1)
            point_coords = [point_coord]

        return (point_coords, [point_label])

    def point_mask_prompt2mask(self, pred, points):
        (point_coords, point_label) = points
        coords_torch = torch.as_tensor([point_coords], dtype=torch.float).to(self.device)
        labels_torch = torch.as_tensor([point_label], dtype=torch.int).to(self.device)
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=(coords_torch, labels_torch),
                                                                            boxes=None,
                                                                            masks=None)
            _, dense_embeddings = self.model.prompt_encoder.interact_forward(masks=pred)

            out = self.model.test_mask_forward(image_embeddings=self.image_embeddings,
                                               ref_label={'sparse_embeddings': sparse_embeddings,
                                                          'dense_embeddings': dense_embeddings},
                                               img_pad_shape=pred.shape[-2:])
        return out

    def interactive_sample(self, ann, shape, mask):
        point_box_prompt_type = np.random.randint(2)
        # point_box_prompt_type = 1
        if point_box_prompt_type == 0:
            if self.choice_point_type == 'center_point':
                pt = cal_center_point_for_binary_mask(mask)
            else:
                pt = random_click(np.array(mask), value=1)
            coords_torch = torch.as_tensor([pt], dtype=torch.float).to(self.device)
            labels_torch = torch.as_tensor([1], dtype=torch.int).to(self.device)
            coords_torch, labels_torch = coords_torch.unsqueeze(1), labels_torch.unsqueeze(1)
            points = (coords_torch, labels_torch)
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=points,
                                                                                boxes=None,
                                                                                masks=None)
        else:
            bbox = self.ann2bbox(ann, shape)
            boxes = torch.as_tensor([bbox], dtype=torch.float)
            boxes = boxes.to(self.device)
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=None,
                                                                                boxes=boxes,
                                                                                masks=None)
        with torch.no_grad():
            out = self.model.test_mask_forward(image_embeddings=self.image_embeddings,
                                               ref_label={'sparse_embeddings': sparse_embeddings,
                                                          'dense_embeddings': dense_embeddings},
                                               img_pad_shape=shape)
        points = self.pred_gt_mask2prompt_point(pred=out, gt_mask=mask)
        iter_num = np.random.randint(1, 4)
        while iter_num > 1:
            out = self.point_mask_prompt2mask(pred=out, points=points)
            points = self.pred_gt_mask2prompt_point(pred=out, gt_mask=mask)
            iter_num -= 1
        return out, points
        # 获取下一步点，继续交互时是否使用mask提示？还是使用已有提示的累积？
        # batch_size × num × 256； Bbox：2, Point：1
        # mask_prompts, point_prompts, box_prompts

    def make_sample(self, image_path, gt_json_path):
        # prompt_type = np.random.randint(1, self.type_num)
        prompt_type = 2
        ''' 提示类型：
                ① 只有一个正样本点；
                ② 只有一个bbox
                ③ 在①或②的基础上，增加1-3次正样本点和负样本点
                '''
        image = Image.open(image_path).convert('RGB')
        filename = os.path.basename(image_path).split('.')[0]
        image_tensor = self.image_transform(image)
        shape = image_tensor.shape[-2:]

        if prompt_type == 3:
            with torch.no_grad():
                image_embeddings = self.model.image_encoder(image_tensor.unsqueeze(0).to(self.device))
                if self.adapter in ['sam_decorator_layer', 'sam_ourdec', 'sam_ourdec_layer_flow',
                                    'sam_decorator_layer_flow', 'sam_ourdecUpup', 'sam_ourdec_layer_flowV2']:
                    self.image_embeddings = self.model.fuse_feature_module(image_embeddings)
                else:
                    self.image_embeddings = image_embeddings[-1]

        with open(gt_json_path, 'r') as file:
            data = json.load(file)
        annotations = data['annotations']
        image_tensors = []
        mask_tensors = []
        prompt_infos = []
        point_coords = []
        point_labels = []
        boxes = []
        masks = []

        for i in range(self.batch_size):
            ann_id = np.random.randint(len(annotations))
            ann = annotations[ann_id]
            mask = self.ann2mask(ann, shape)
            if prompt_type == 1:
                if self.choice_point_type == 'center_point':
                    pt = cal_center_point_for_binary_mask(mask)
                else:
                    pt = random_click(np.array(mask), value=1)
                point_coords.append([pt])
                point_labels.append([1])

            elif prompt_type == 2:
                bbox = self.ann2bbox(ann, shape)
                boxes.append(bbox)

            else:
                prompt_mask, (point_coord, point_label) = self.interactive_sample(ann, shape, mask)
                point_coords.append(point_coord)
                point_labels.append(point_label)
                masks.append(prompt_mask)

            prompt_infos.append(prompt_type)
            mask_tensor = self.mask_transform(mask)
            mask_tensors.append(mask_tensor)
            image_tensors.append(image_tensor)

        if len(point_coords) == 0:
            prompt_points = None
        else:
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float).to(self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int).to(self.device)
            prompt_points = (coords_torch, labels_torch)

        if len(boxes) == 0:
            prompt_boxes = None
        else:
            prompt_boxes = torch.as_tensor(boxes, dtype=torch.float).to(self.device)

        if len(masks) != 0:
            prompt_masks = torch.cat(masks, dim=0)
        else:
            prompt_masks = None

        return {
            'image': torch.stack(image_tensors, dim=0),
            'label': torch.stack(mask_tensors, dim=0),
            'prompt_point': prompt_points,
            'prompt_box': prompt_boxes,
            'prompt_mask': prompt_masks,
            'filename': filename,
            'prompt_type': prompt_type
        }

    def add_mask_noise(self, pred_mask, noise_anns, shape):
        for ann in noise_anns:
            noise_mask = self.ann2mask(ann, shape)
            pred_mask[noise_mask>0] = 1
        return pred_mask

    def drag_hole_to_mask(self, binary_image):
        # 获取所有非零点的坐标
        non_zero_points = np.argwhere(binary_image > 0)

        # 随机选择一个非零点作为中心点
        random_index = np.random.randint(0, len(non_zero_points) - 1)
        random_point = non_zero_points[random_index]

        # 定义挖掉区域的大小
        hole_size = np.random.choice([20, 40, 50, 80, 100, 120, 150, 200])

        # 确保挖掉区域不超出图像边界
        min_row = max(0, random_point[0] - hole_size // 2)
        max_row = min(binary_image.shape[0], random_point[0] + hole_size // 2)
        min_col = max(0, random_point[1] - hole_size // 2)
        max_col = min(binary_image.shape[1], random_point[1] + hole_size // 2)

        # 挖掉区域
        binary_image[min_row:max_row, min_col:max_col] = 0
        return binary_image

    def random_dilate_or_erode_threshold(self, pred_mask, gt_mask):
        # 尝试直到 IoU 低于 0.6
        iou = 1.0
        attempts = 0
        while iou >= 0.6 and attempts < 5:
            pred_mask = random_dilate_or_erode(pred_mask)
            iou = calculate_iou(gt_mask, pred_mask)
            attempts += 1
        return pred_mask

    def sample_point_by_mask(self, binary_image, num_points, min_distance, label_value):
        selected_points = select_random_pointsV2(binary_image, num_points)
        if len(selected_points) > 0:
            point_coords = [point[::-1] for point in selected_points]
            point_labels = [label_value for _ in range(len(point_coords))]
            return (point_coords, point_labels)
        else:
            return []

    def set_seed_val(self):
        if self.mode == 'val':
            np.random.seed(seed=666)

    def make_batch_sample(self, batch_image_path, batch_gt_json_path):
        # prompt_type = np.random.randint(1, self.type_num)
        # prompt_type = 4
        prompt_type = np.random.choice(self.prompt_types)
        ''' 提示类型 ：
                ① 只有一个正样本点；
                ② 只有一个bbox
                ③ 在①或②的基础上，增加1-3次正样本点和负样本点
                ④ 一笔画
                '''
        image_tensors = []
        mask_tensors = []
        prompt_infos = []
        point_coords = []
        point_labels = []
        boxes = []
        masks = []
        batch_filename = []

        # if self.mode == 'val':
        #     positive_num = 1
        #     negative_num = 2
        #     add_building_num = 0
        # else:
        positive_num = np.random.choice([1, 2, 3])
        # positive_num = 3
        negative_num = np.random.choice([0, 1, 2, 3])
        # negative_num = 3
        add_building_num = np.random.choice([0, 0, 0, 1, 2, 3])
        # add_building_num = 2
        for i in range(self.batch_size):
            image = Image.open(batch_image_path[i]).convert('RGB')
            filename = os.path.basename(batch_image_path[i]).split('.')[0]
            image_tensor = self.image_transform(image)
            shape = image_tensor.shape[-2:]
            # if prompt_type == 3:
            #     with torch.no_grad():
            #         image_embeddings = self.model.image_encoder(image_tensor.unsqueeze(0).to(self.device))
            #         if self.adapter in ['sam_decorator_layer', 'sam_ourdec', 'sam_ourdec_layer_flow',
            #                             'sam_decorator_layer_flow', 'sam_ourdecUpup', 'sam_ourdec_layer_flowV2']:
            #             self.image_embeddings = self.model.fuse_feature_module(image_embeddings)
            #         else:
            #             self.image_embeddings = image_embeddings[-1]

            with open(batch_gt_json_path[i], 'r') as file:
                data = json.load(file)
            annotations = data['annotations']
            ann_ids = list(range(len(annotations)))
            ann_id = np.random.choice(ann_ids)  # np.random.randint(len(annotations))
            ann = annotations[ann_id]
            gt_mask = self.ann2mask(ann, shape)

            if prompt_type == 1:
                if self.choice_point_type == 'center_point':
                    pt = cal_center_point_for_binary_mask(gt_mask)
                else:
                    pt = random_click(np.array(gt_mask), value=1)
                point_coords.append([pt])
                point_labels.append([1])

            elif prompt_type == 2:
                bbox = self.ann2bbox(ann, shape)
                boxes.append(bbox)

            elif prompt_type == 4:
                points = generate_spline_point_within_mask(np.array(gt_mask), N=5)
                point_coords.append(points)
                point_labels.append([1 for _ in range(len(points))])

            else:
                # prompt_mask, (point_coord, point_label) = self.interactive_sample(ann, shape, gt_mask)
                # point_coords.append(point_coord)
                # point_labels.append(point_label)
                # masks.append(prompt_mask)
                pred_mask = copy.deepcopy(gt_mask)
                pred_mask = self.drag_hole_to_mask(pred_mask)
                if add_building_num > 0:
                    ann_ids.remove(ann_id)
                    if add_building_num < len(ann_ids):
                        noise_ann_ids = np.random.choice(ann_ids, add_building_num)
                    else:
                        noise_ann_ids = ann_ids
                    noise_anns = [annotations[i_noise] for i_noise in noise_ann_ids]
                    pred_mask = self.add_mask_noise(pred_mask, noise_anns, shape)

                pred_mask = self.random_dilate_or_erode_threshold(pred_mask, gt_mask)
                # 计算交集
                intersection = np.logical_and(gt_mask, pred_mask).astype(np.uint8)
                missing_mask = gt_mask - intersection
                pos_point_coord, pos_point_label = [], []
                if positive_num > 0 and np.max(missing_mask) == 1:
                    point_coord = cal_center_point_for_binary_mask(missing_mask)
                    pos_point_coord = pos_point_coord + [point_coord]
                    pos_point_label = pos_point_label + [1]
                    if positive_num > 1:
                        point_coord, point_label = self.sample_point_by_mask(missing_mask,
                                                                             num_points=positive_num-1,
                                                                             min_distance=10, label_value=1)
                        pos_point_coord = pos_point_coord + point_coord
                        pos_point_label = pos_point_label + point_label
                else:
                    point_coord = cal_center_point_for_binary_mask(gt_mask)
                    pos_point_coord = pos_point_coord + [point_coord]
                    pos_point_label = pos_point_label + [1]
                # if positive_num == 1:
                #     point_coord = cal_center_point_for_binary_mask(missing_mask)
                #     pos_point_coord, pos_point_label = [point_coord], [1]
                #
                # else:
                #     if np.max(missing_mask) == 0:
                #         pos_point_coord, pos_point_label = [], []
                #     else:
                #         pos_point_coord, pos_point_label = self.sample_point_by_mask(missing_mask, num_points=positive_num,
                #                                                                      min_distance=10, label_value=1)
                if negative_num > 0:
                    mistaken_mask = pred_mask - intersection

                    if np.max(mistaken_mask) == 0:
                        neg_point_coord, neg_point_label = [], []
                    else:
                        neg_point_coord, neg_point_label = self.sample_point_by_mask(mistaken_mask, num_points=negative_num,
                                                                                     min_distance=10, label_value=0)
                else:
                    neg_point_coord, neg_point_label = [], []

                # neg_point_coord, neg_point_label = [], []
                # mistaken_mask = pred_mask - intersection
                # if negative_num > 0 and np.max(mistaken_mask) == 1:
                #     mistaken_mask = pred_mask - intersection
                #     point_coord = cal_center_point_for_binary_mask(mistaken_mask)
                #     neg_point_coord = neg_point_coord + [point_coord]
                #     neg_point_label = neg_point_label + [0]
                #     if negative_num > 1:
                #         point_coord, point_label = self.sample_point_by_mask(mistaken_mask,
                #                                                              num_points=negative_num-1,
                #                                                              min_distance=10, label_value=0)
                #         neg_point_coord = neg_point_coord + point_coord
                #         neg_point_label = neg_point_label + point_label

                coords = pos_point_coord + neg_point_coord
                p_labels = pos_point_label + neg_point_label
                num = positive_num + negative_num

                # 保持coords和p_labels的对应关系，扩展到相同长度
                while len(coords) < num:
                    # 随机选择已有的一个索引
                    index = np.random.randint(0, len(coords))
                    coords.append(coords[index])
                    p_labels.append(p_labels[index])

                point_coords.append(coords)
                point_labels.append(p_labels)
                masks.append(self.mask_transform(pred_mask).unsqueeze(0))

            prompt_infos.append(prompt_type)
            mask_tensor = self.mask_transform(gt_mask)
            mask_tensors.append(mask_tensor)
            image_tensors.append(image_tensor)
            batch_filename.append(filename)

        if len(point_coords) == 0:
            prompt_points = None
        else:
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float).to(self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int).to(self.device)
            prompt_points = (coords_torch, labels_torch)

        if len(boxes) == 0:
            prompt_boxes = None
        else:
            prompt_boxes = torch.as_tensor(boxes, dtype=torch.float).to(self.device)

        if len(masks) != 0:
            prompt_masks = torch.cat(masks, dim=0)
        else:
            prompt_masks = None

        return {
            'image': torch.stack(image_tensors, dim=0),
            'label': torch.stack(mask_tensors, dim=0),
            'prompt_point': prompt_points,
            'prompt_box': prompt_boxes,
            'prompt_mask': prompt_masks,
            'filename': batch_filename,
            'prompt_type': prompt_type
        }

    def generate_sample(self):
        if self.mode == 'val':
            np.random.seed(seed=666)
        for item in range(self.length):
            n_dataset = len(self.image_paths)
            n_pre_sample = 0
            for data_id in range(n_dataset):
                if item < n_pre_sample + len(self.gt_json_paths[data_id]):
                    gt_json_path = self.gt_json_paths[data_id][item - n_pre_sample]
                    filename = osp.basename(gt_json_path).split('.')[0] + '.tif'
                    image_path = osp.join(self.image_paths[data_id], filename)
                    sample = self.make_sample(image_path, gt_json_path)
                    yield sample
                else:
                    n_pre_sample += len(self.image_paths[data_id])

    def shuffle_samples(self):
        for i in range(len(self.gt_json_paths)):
            np.random.shuffle(self.gt_json_paths[i])

    def generate_sample_std(self):
        if self.mode == 'val':
            np.random.seed(seed=666)

        self.shuffle_samples()

        for item in range(self.iter_num_per_epoch):
            if self.mode == 'test':
                assert self.batch_size == 1
                pt_path = self.data_pts[item]
                sample = torch.load(pt_path)
                yield sample
            else:
                n_dataset = len(self.image_paths)
                sample_ids = [item * self.batch_size + dx for dx in range(self.batch_size)]
                batch_image_path = []
                batch_gt_json_path = []
                for sample_id in sample_ids:
                    n_pre_sample = 0
                    for data_id in range(n_dataset):
                        if sample_id < n_pre_sample + len(self.gt_json_paths[data_id]):
                            gt_json_path = self.gt_json_paths[data_id][sample_id - n_pre_sample]
                            filename = osp.basename(gt_json_path).split('.')[0] + '.tif'
                            image_path = osp.join(self.image_paths[data_id], filename)
                            batch_image_path.append(image_path)
                            batch_gt_json_path.append(gt_json_path)
                            break
                        else:
                            n_pre_sample += len(self.gt_json_paths[data_id])
                assert len(batch_image_path) == self.batch_size
                assert len(batch_gt_json_path) == self.batch_size
                # print(batch_image_path)
                sample = self.make_batch_sample(batch_image_path, batch_gt_json_path)
                yield sample


def image_transform(image):
    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]
    mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
    std = torch.Tensor(pixel_std).view(-1, 1, 1)
    image = image * std + mean
    # Convert tensor image back to numpy array
    x = image.numpy().transpose((1, 2, 0)).astype(np.uint8)
    return x


if __name__ == '__main__':
    #  set data parameters
    import argparse
    from scipy.interpolate import splprep, splev
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='cfg_evlab_building')
    parser.add_argument("--net", type=str, default='sam')
    parser.add_argument("--adapter", type=str, default='sam_ours')  # none: sam
    parser.add_argument("--gpu_id", type=int, default=6)
    parser.add_argument("--out_size", type=int, default=512)
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-distributed', default='none', type=str, help='multi GPU ids to use')
    parser.add_argument('-sam_ckpt', default='/home/zzl/codes/sam_adapter_rs/pre_weights/sam_vit_b_01ec64.pth',
                        help='sam checkpoint address')
    args = parser.parse_args()

    train_image_paths = []
    train_ann_paths = []
    val_image_paths = []
    val_ann_paths = []
    data_dir = '/home/zzl/datasets/building_data/EVLab_building'
    # list_data_name = ['Taiwan', 'Guangdong', 'Chongqing', 'Zhengzhou', 'Wuhan', 'Xian']
    list_data_name = ['Guangdong', 'Wuhan']
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

    cfg = importlib.import_module('configs.' + args.config_file)
    if args.gpu_id in list(range(8)):
        cfg.common.gpu_id = args.gpu_id
        print(f'gpu_id: {args.gpu_id}')
    cfg.model.image_encoder.adapter = args.adapter

    device = torch.device('cuda:{}'.format(cfg.common.gpu_id))

    # net = build_model(cfg).to(device)
    net = None

    # iter_seg_data = InterSegOnePrompt(image_paths=train_image_paths,
    #                                    ann_paths=train_ann_paths,
    #                                    debug=True,
    #                                    mode='val',
    #                                    model=net,
    #                                    device=device)
    save_test_sample_dir = '/home/zzl/datasets/building_data/EVLab_building/vis_sample_show'
    os.makedirs(osp.join(save_test_sample_dir, 'images'), exist_ok=True)
    os.makedirs(osp.join(save_test_sample_dir, 'interactive_infos'), exist_ok=True)
    os.makedirs(osp.join(save_test_sample_dir, 'masks'), exist_ok=True)

    iter_seg_data = InterSegOnePrompt(
        image_paths=train_image_paths,
        ann_paths=train_ann_paths,
        debug=False,
        mode='train',
        model=net,
        device=device,
        batch_size=1,
        adapter='sam_para',
        prompt_types=[1,2,4]
    )
    # prompt_types = [1,2,3,4]
    ''' 提示类型 ：
                    ① 只有一个正样本点；
                    ② 只有一个bbox
                    ③ 在①或②的基础上，增加1-3次正样本点和负样本点
                    ④ 一笔画
                    '''

    length = iter_seg_data.length
    print(f'data size: {length}')
    # {
    #     'image': torch.cat(image_tensors, dim=0),
    #     'label': torch.cat(mask_tensors, dim=0),
    #     'prompt_point': prompt_points,
    #     'prompt_box': prompt_boxes,
    #     'prompt_mask': prompt_masks,
    #     'filename': filename,
    # }

    data_generator = iter_seg_data.generate_sample_std()
    num = 0
    for i, data in enumerate(tqdm(data_generator)):
        if i > 200: break
        # prompt_type = data['prompt_type']
        # print(i, prompt_type)
        # torch.save(data, f'{save_test_sample_dir}/data_{i}.pt')
        # continue
        alpha = 0.75
        marker_size = 50
        prompt_type = data['prompt_type']
        print(i, prompt_type)
        gt_mask = data['label'].cpu().numpy()[0][0]
        image = image_transform(data['image'][0])
        filename = data['filename'][0]

        fig, axs = plt.subplots(1, 1, figsize=(5, 5))  # Change to 1 row, 4 columns
        fig.tight_layout(pad=0)
        axs.imshow(image)
        # axs.set_title('image')
        axs.axis('off')
        # 保存图像，去掉 bbox_inches='tight' 参数以保留边界空白
        plt.savefig(f'{save_test_sample_dir}/images/{filename}.tif', bbox_inches='tight', dpi=150)
        plt.close()

        fig, axs = plt.subplots(1, 1, figsize=(5, 5))  # Change to 1 row, 4 columns
        fig.tight_layout(pad=0)
        rgb_color = [255, 255, 0]  # 黄色
        rgb_mask = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        rgb_mask[gt_mask > 0] = rgb_color
        gt_result = image * (1 - gt_mask[:, :, np.newaxis]) + \
                 (1 - alpha) * gt_mask[:, :, np.newaxis] * image + \
                 alpha * rgb_mask
        gt_result = gt_result.astype(np.uint8)
        axs.imshow(gt_result)
        axs.axis('off')
        # plt.savefig('output_image.png', bbox_inches='tight', dpi=150)
        plt.savefig(f'{save_test_sample_dir}/masks/{filename}.tif', bbox_inches='tight', dpi=150)
        plt.close()
        # image/pred
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))  # Change to 1 row, 4 columns
        fig.tight_layout(pad=0)
        if prompt_type != 3:
            result = image.astype(np.uint8)
        else:
            rgb_color = [255, 255, 0]  # 黄色
            rgb_mask = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
            pred_mask = data['prompt_mask'][0][0].cpu().numpy()
            rgb_mask[pred_mask > 0] = rgb_color
            result = image * (1 - pred_mask[:, :, np.newaxis]) + \
                     (1 - alpha) * pred_mask[:, :, np.newaxis] * image + \
                     alpha * rgb_mask
            result = result.astype(np.uint8)
        axs.imshow(result)

        if prompt_type == 1:  # point
            points = data['prompt_point'][0][0].cpu().numpy()
            x = [point[0] for point in points]
            y = [point[1] for point in points]
            axs.scatter(x, y, color='yellow', marker='*', s=marker_size)  # mistaken_points

        elif prompt_type == 2:  # box
            box = data['prompt_box'][0].cpu().numpy()
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='yellow', facecolor='none')
            axs.add_patch(rect)

        elif prompt_type == 4:  # scribble
            points = data['prompt_point'][0][0].cpu().numpy()
            x = [point[0] for point in points]
            y = [point[1] for point in points]

            # # 使用散点图显示点
            # axs[2].scatter(x, y, color='yellow', marker='*', s=marker_size)
            # 使用样条插值生成平滑曲线
            tck, u = splprep([x, y], s=0)
            unew = np.linspace(0, 1, 100)
            out = splev(unew, tck)

            # 将样条插值点转换为整数坐标
            x_spline = np.array(out[0], dtype=int)
            y_spline = np.array(out[1], dtype=int)

            # 绘制平滑曲线
            axs.plot(x_spline, y_spline, color='yellow', linestyle='-', linewidth=2)  # 平滑曲线

        else:
            points = data['prompt_point'][0][0].cpu().numpy()
            x = [point[0] for point in points]
            y = [point[1] for point in points]
            point_labels = data['prompt_point'][1][0]
            for i in range(len(x)):
                if point_labels[i] == 1:
                    axs.scatter(x[i], y[i], color='g', marker='*', s=marker_size)
                else:
                    axs.scatter(x[i], y[i], color='r', marker='*', s=marker_size)

        axs.axis('off')
        plt.savefig(f'{save_test_sample_dir}/interactive_infos/{filename}.tif', bbox_inches='tight', dpi=150)
        plt.close()


