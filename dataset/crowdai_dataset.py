import os
import numpy as np
from pycocotools.coco import COCO
from skimage import io
import cv2
from matplotlib.patches import Polygon, Rectangle
import os.path as osp
from glob import glob
import torch
from dataset.utils import polys2mask, random_click, cal_center_point_for_binary_mask,\
    calculate_iou, random_dilate_or_erode,\
    select_random_pointsV2
from PIL import Image
import matplotlib.pyplot as plt
from dataset.before_data_sampler.scribble_sampler import generate_spline_point_within_mask
import copy
from tqdm import tqdm
import torch.nn.functional as F


class CrowdAiDataset:
    def __init__(self, image_dir, ann_path,
                 debug=False,
                 mode='train',
                 model=None,
                 device=None,
                 shape=(320, 320),
                 box_scales=[0.8, 1.2],
                 box_ext_pixel=6,
                 batch_size=8,
                 adapter='sam_ours',
                 choice_point_type='center_point',
                 prompt_types=[1]):

        self.IoU_threshold = 0.6
        self.min_building_area = 100
        self.image_shape = shape
        # 读取数据
        self.coco = COCO(ann_path)
        self.image_dir = image_dir
        self.imgIds = self.coco.getImgIds()
        self.image_info = []
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.load_data()

        self.mode = mode
        self.data_pts = []
        if self.mode == 'test':
            self.data_pts = glob(osp.join('/home/zzl/datasets/building_data/CrowdAI_building/test_data', '*.pt'))
        # 数据信息
        if debug:
            if self.mode == 'test':
                self.data_pts = self.data_pts[:200]
            else:
                self.image_info = self.image_info[:200]
        if self.mode == 'test':
            np.random.seed(666)
            np.random.shuffle(self.data_pts)
            self.data_pts = self.data_pts[:2000]
            np.random.seed(None)

        self.model = model
        self.device = device
        self.box_scales = box_scales
        self.box_ext_pixel = box_ext_pixel

        if self.mode == 'test':
            self.length = len(self.data_pts)
        else:
            self.length = len(self.image_info)

        self.prompt_types = prompt_types
        self.choice_point_type = choice_point_type  # 'center_point'
        self.image_embeddings = None
        self.adapter = adapter
        self.batch_size = batch_size
        self.iter_num_per_epoch = self.length // batch_size


    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path):
        image_info = {
            "source": source,
            "id": image_id,
            "path": path,
        }
        self.image_info.append(image_info)

    def load_data(self):
        self.add_class("buildings", 1, "building")
        reduce_data_num = 0
        for item in self.imgIds:
            img = self.coco.loadImgs(item)[0]
            path = os.path.join(self.image_dir, img['file_name'])

            annIds = self.coco.getAnnIds(imgIds=item, iscrowd=None)
            if type(annIds) == int:
                annIds = [annIds]

            # 加载注释
            annotations = self.coco.loadAnns(annIds)
            annotations = [item for item in annotations if item['area'] > self.min_building_area]
            if len(annotations) == 0:
                reduce_data_num += 1
                continue
            self.add_image("buildings", image_id=item, path=path)
        print(f'loading data finish, invalid data num: {reduce_data_num}')

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array."""
        info = self.image_info[image_id]
        image = io.imread(info['path'])
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "buildings":
            return info
        else:
            return super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Load instance masks for the image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        annIds = self.coco.getAnnIds(imgIds=info['id'], iscrowd=None)
        if type(annIds) == int:
            annIds = [annIds]
        anns = self.coco.loadAnns(annIds)
        anns = [item for item in anns if item['area'] != 0]
        masks = np.stack([self.coco.annToMask(item) for item in anns], axis=-1)
        _, _, num_masks = masks.shape
        class_ids = np.asarray([1] * num_masks)
        return masks, class_ids

    def draw_annotations_on_image(self, image_id):
        """Draw annotations (polygons) on the image."""
        # 获取图像信息和注释ID
        info = self.image_info[image_id]
        annIds = self.coco.getAnnIds(imgIds=info['id'], iscrowd=None)
        if type(annIds) == int:
            annIds = [annIds]

        # 加载注释
        anns = self.coco.loadAnns(annIds)

        # 读取图像
        image = cv2.imread(info['path'])  # 确保info['path']是图像的实际路径

        # # 绘制多边形
        # for ann in anns:
        #     if 'segmentation' in ann:
        #         for seg in ann['segmentation']:
        #             if isinstance(seg, list):  # 如果是多边形
        #                 pts = np.array(seg).reshape((-1, 1, 2)).astype(np.int32)
        #                 cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        #
        # return image
        # 创建一个绘图对象
        fig, ax = plt.subplots(1)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 绘制多边形
        for ann in anns:
            if 'segmentation' in ann:
                for seg in ann['segmentation']:
                    if isinstance(seg, list):  # 如果是多边形
                        pts = np.array(seg).reshape((-1, 2))
                        polygon = Polygon(pts, closed=True, edgecolor='green', linewidth=2, fill=None)
                        ax.add_patch(polygon)

                        # 使用segmentation计算bbox
                        x_min, y_min = np.min(pts, axis=0)
                        x_max, y_max = np.max(pts, axis=0)
                        width = x_max - x_min
                        height = y_max - y_min
                        rect = Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='blue', facecolor='none')
                        ax.add_patch(rect)
            # # 绘制边界框
            # if 'bbox' in ann:
            #     bbox = ann['bbox']
            #     # bbox格式为 [x, y, width, height]
            #     x, y, width, height = bbox
            #     rect = Rectangle((x, y), width, height, linewidth=2, edgecolor='red', facecolor='none')
            #     ax.add_patch(rect)

        plt.axis('off')
        plt.show()

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

    def ann2mask(self, ann, shape):
        polys = []
        for segmentation in ann['segmentation']:
            poly = np.array(segmentation).reshape((-1, 2))
            if poly.shape[0] < 3: continue
            polys.append(poly)
        mask = polys2mask(polys, shape=shape)
        return mask

    def scale_bbox_xywh_to_xyxy(self, bbox, W, H, scale=1.0, ext_pixel=6):
        """
        将xywh格式的框进行缩放转为xyxy格式，并确保框的范围在图像大小内。

        :param x: 框的x坐标
        :param y: 框的y坐标
        :param width: 框的宽度
        :param height: 框的高度
        :param W: 图像的宽度
        :param H: 图像的高度
        :param ext_pixel: 缩放的像素大小，默认为5
        :return: 缩放后的框（xyxy格式）
        """
        # 缩放框的大小
        x, y, width, height = bbox
        cx, cy = x + width / 2, y + height / 2
        wh = np.array([width, height], dtype=np.float32) * scale
        width, height = wh.tolist()
        x1, y1, x2, y2 = cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2
        dx = np.random.choice(np.arange(-ext_pixel, ext_pixel + 1, 2))
        dy = np.random.choice(np.arange(-ext_pixel, ext_pixel + 1, 2))
        x1 = max(x1 - dx, 0)
        y1 = max(y1 - dy, 0)
        x2 = min(x2 + dx, W)
        y2 = min(y2 + dy, H)

        return [x1, y1, x2, y2]

    def ann2bbox(self, ann, shape):
        scale = self.rand(self.box_scales[0], self.box_scales[1])
        segmentations = ann['segmentation']
        # print(f'segmentation_len: {len(segmentations)}')
        # for segmentation in ann['segmentation']:
        pts = np.array(segmentations[0]).reshape((-1, 2))
        # 使用segmentation计算bbox
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        width = x_max - x_min
        height = y_max - y_min
        bbox = [x_min, y_min, width, height]
        bbox = self.scale_bbox_xywh_to_xyxy(bbox,
                                            W=shape[1],
                                            H=shape[0],
                                            scale=scale,
                                            ext_pixel=self.box_ext_pixel)

        return bbox

    def rand(self, a=0., b=1.):
        return np.random.rand()*(b-a) + a

    def shuffle_samples(self):
        np.random.shuffle(self.image_info)

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

    def add_mask_noise(self, pred_mask, noise_anns, shape):
        for ann in noise_anns:
            noise_mask = self.ann2mask(ann, shape)
            pred_mask[noise_mask>0] = 1
        return pred_mask

    def sample_point_by_mask(self, binary_image, num_points, min_distance, label_value):
        selected_points = select_random_pointsV2(binary_image, num_points)
        if len(selected_points) > 0:
            point_coords = [point[::-1] for point in selected_points]
            point_labels = [label_value for _ in range(len(point_coords))]
            return (point_coords, point_labels)
        else:
            return []

    def random_dilate_or_erode_threshold(self, pred_mask, gt_mask):
        # 尝试直到 IoU 低于 0.6
        iou = 1.0
        attempts = 0
        while iou >= 0.6 and attempts < 5:
            pred_mask = random_dilate_or_erode(pred_mask)
            iou = calculate_iou(gt_mask, pred_mask)
            attempts += 1
        return pred_mask

    def expand_image(self, image_pil):
        # 创建一个新的图像，填充颜色为白色
        # new_image = Image.new("RGB", self.image_shape, (255, 255, 255))
        new_image = Image.new("RGB", self.image_shape, (0, 0, 0))
        # 将原始图像粘贴到新的图像上
        new_image.paste(image_pil, (0, 0))
        return new_image

    def make_batch_sample(self, batch_image_info):
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

        positive_num = np.random.choice([1, 2, 3])
        negative_num = np.random.choice([0, 1, 2, 3])
        add_building_num = np.random.choice([0, 0, 0, 1, 2, 3])

        # info = self.image_info[image_id]
        # annIds = self.coco.getAnnIds(imgIds=info['id'], iscrowd=None)
        # if type(annIds) == int:
        #     annIds = [annIds]
        #
        # # 加载注释
        # anns = self.coco.loadAnns(annIds)
        #
        # # 读取图像
        # image = cv2.imread(info['path'])  # 确保info['path']是图像的实际路径

        for i in range(self.batch_size):
            info = batch_image_info[i]
            image = Image.open(info['path']).convert('RGB')
            image = self.expand_image(image)
            filename = os.path.basename(info['path']).split('.')[0]
            image_tensor = self.image_transform(image)
            shape = image_tensor.shape[-2:]

            annIds = self.coco.getAnnIds(imgIds=info['id'], iscrowd=None)
            if type(annIds) == int:
                annIds = [annIds]

            # 加载注释
            annotations = self.coco.loadAnns(annIds)
            annotations = [item for item in annotations if item['area'] > self.min_building_area]
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
                try:
                    points = generate_spline_point_within_mask(np.array(gt_mask), N=5)
                except Exception as e:
                    print(f'error: {e}')
                    ann_ids.remove(ann_id)
                    ann_id = np.random.choice(ann_ids)  # np.random.randint(len(annotations))
                    ann = annotations[ann_id]
                    gt_mask = self.ann2mask(ann, shape)
                    points = generate_spline_point_within_mask(np.array(gt_mask), N=5)

                point_coords.append(points)
                point_labels.append([1 for _ in range(len(points))])

            else:
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
                                                                             num_points=positive_num - 1,
                                                                             min_distance=10, label_value=1)
                        pos_point_coord = pos_point_coord + point_coord
                        pos_point_label = pos_point_label + point_label
                else:
                    point_coord = cal_center_point_for_binary_mask(gt_mask)
                    pos_point_coord = pos_point_coord + [point_coord]
                    pos_point_label = pos_point_label + [1]

                if negative_num > 0:
                    mistaken_mask = pred_mask - intersection

                    if np.max(mistaken_mask) == 0:
                        neg_point_coord, neg_point_label = [], []
                    else:
                        neg_point_coord, neg_point_label = self.sample_point_by_mask(mistaken_mask,
                                                                                     num_points=negative_num,
                                                                                     min_distance=10, label_value=0)
                else:
                    neg_point_coord, neg_point_label = [], []

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

    def generate_sample_std(self):
        if self.mode == 'val':
            np.random.seed(seed=666)

        self.shuffle_samples()

        for item in range(self.iter_num_per_epoch):
            if self.mode == 'test':
                assert self.batch_size == 1
                pt_path = self.data_pts[item]
                sample = torch.load(pt_path)
                image_tensor = sample['image']
                if image_tensor.shape[-1] == 300:
                    image_tensor = F.pad(image_tensor, (0, 20, 0, 20))
                    mask_tensor = sample['label']
                    mask_tensor = F.pad(mask_tensor, (0, 20, 0, 20))
                    prompt_mask = sample['prompt_mask']
                    if prompt_mask is not None:
                        prompt_mask = F.pad(prompt_mask, (0, 20, 0, 20))
                    sample.update(
                        {
                            'image': image_tensor,
                            'label': mask_tensor,
                            'prompt_mask': prompt_mask}
                    )
                yield sample
            else:
                sample_ids = [item * self.batch_size + dx for dx in range(self.batch_size)]
                batch_image_info = []
                for sample_id in sample_ids:
                    batch_image_info.append(self.image_info[sample_id])

                assert len(batch_image_info) == self.batch_size
                sample = self.make_batch_sample(batch_image_info)
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
    # Example usage
    image_dir = '/home/zzl/datasets/building_data/CrowdAI_building/val/images'
    ann_path = '/home/zzl/datasets/building_data/CrowdAI_building/val/annotation.json'

    # Initialize dataset
    iter_seg_data = CrowdAiDataset(image_dir=image_dir, ann_path=ann_path,
                             debug=False,
                             mode='val',
                             model=None,
                             device=None,
                             box_scales=[0.8, 1.2],
                             box_ext_pixel=6,
                             batch_size=1,
                             adapter='sam_ours',
                             choice_point_type='center_point',
                             prompt_types=[1,2,3,4]
                             )
    # iter_seg_data.draw_annotations_on_image(0)
    # exit()

    # prompt_types = [1,2,3,4]
    ''' 提示类型 ：
                    ① 只有一个正样本点；
                    ② 只有一个bbox
                    ③ 在①或②的基础上，增加1-3次正样本点和负样本点
                    ④ 一笔画
                    '''

    length = iter_seg_data.length
    print(f'data size: {length}')
    save_test_sample_dir = '/home/zzl/datasets/building_data/CrowdAI_building/test_data'
    os.makedirs(save_test_sample_dir, exist_ok=True)
    data_generator = iter_seg_data.generate_sample_std()
    num = 0
    for i, data in enumerate(tqdm(data_generator)):
        # prompt_type = data['prompt_type']
        # filename = data['filename'][0]
        # print(i, prompt_type)
        # torch.save(data, f'{save_test_sample_dir}/{filename}.pt')
        # continue

        alpha = 0.75
        marker_size = 100
        prompt_type = data['prompt_type']
        print(i, prompt_type)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Change to 1 row, 4 columns
        fig.tight_layout()
        image = image_transform(data['image'][0])
        # print(image.shape)
        # continue
        gt_mask = data['label'].cpu().numpy()[0][0]
        filename = data['filename'][0]
        axs[0].imshow(image, cmap='jet')
        axs[0].set_title('image')
        axs[0].axis('off')

        rgb_color = [255, 255, 0]  # 黄色
        rgb_mask = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        rgb_mask[gt_mask > 0] = rgb_color
        gt_result = image * (1 - gt_mask[:, :, np.newaxis]) + \
                    (1 - alpha) * gt_mask[:, :, np.newaxis] * image + \
                    alpha * rgb_mask
        gt_result = gt_result.astype(np.uint8)
        if prompt_type == 3:
            points = data['prompt_point'][0][0].cpu().numpy()
            x = [point[0] for point in points]
            y = [point[1] for point in points]
            point_labels = data['prompt_point'][1][0]

            for i in range(len(x)):
                if point_labels[i] == 1:
                    axs[1].scatter(x[i], y[i], color='g', marker='*', s=marker_size)  # mistaken_points
                else:
                    axs[1].scatter(x[i], y[i], color='r', marker='*', s=marker_size)  # mistaken_points
        axs[1].imshow(gt_result)
        axs[1].set_title('gt')
        axs[1].axis('off')

        # image/gt-check
        if prompt_type != 3:
            # rgb_color = [255, 255, 0]  # 黄色
            # rgb_mask = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
            # rgb_mask[gt_mask > 0] = rgb_color
            # result = image * (1 - gt_mask[:, :, np.newaxis]) + \
            #          (1 - alpha) * gt_mask[:, :, np.newaxis] * image + \
            #          alpha * rgb_mask
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
        axs[2].imshow(result)

        if prompt_type == 1:  # point
            points = data['prompt_point'][0][0].cpu().numpy()
            x = [point[0] for point in points]
            y = [point[1] for point in points]
            axs[2].scatter(x, y, color='g', marker='*', s=marker_size)  # mistaken_points

        elif prompt_type == 2:  # box
            box = data['prompt_box'][0].cpu().numpy()
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='green', facecolor='none')
            axs[2].add_patch(rect)
        else:
            points = data['prompt_point'][0][0].cpu().numpy()
            x = [point[0] for point in points]
            y = [point[1] for point in points]
            point_labels = data['prompt_point'][1][0]
            # if point_labels == 1:
            #     axs[2].scatter(x, y, color='g', marker='*', s=marker_size)  # mistaken_points
            # else:
            #     axs[2].scatter(x, y, color='r', marker='*', s=marker_size)  # mistaken_points
            for i in range(len(x)):
                if point_labels[i] == 1:
                    axs[2].scatter(x[i], y[i], color='g', marker='*', s=marker_size)  # mistaken_points
                else:
                    axs[2].scatter(x[i], y[i], color='r', marker='*', s=marker_size)  # mistaken_points

        axs[2].set_title('pred_mask')
        axs[2].axis('off')

        plt.show()
        # plt.savefig(osp.join(f'{save_test_sample_dir}/test_{i_time}', filename+'.tif'), dpi=150)
        plt.close()
