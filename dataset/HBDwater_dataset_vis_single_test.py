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
from concurrent.futures import ThreadPoolExecutor
import scipy.ndimage as ndimage


class HBDWaterDataset:
    def __init__(self, image_dir, gt_mask_dir,
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

        self.IoU_threshold = 0.6
        self.min_building_area = 200

        # 读取数据
        self.image_dir = image_dir
        self.gt_mask_dir = gt_mask_dir
        self.mode = mode
        self.filenames = []
        # self.load_data()
        self.load_data_ref()

        self.data_pts = []
        if self.mode == 'test':
            self.data_pts = glob(osp.join('/home/zzl/datasets/waterbody_data/HBD_water_data/test_data',
                                          '*.pt'))
        # 数据信息
        if debug:
            if self.mode == 'test':
                self.data_pts = self.data_pts[:200]
            else:
                self.filenames = self.filenames[:200]

        self.model = model
        self.device = device
        self.box_scales = box_scales
        self.box_ext_pixel = box_ext_pixel

        if self.mode == 'test':
            self.length = len(self.data_pts)
        else:
            self.length = len(self.filenames)

        self.prompt_types = prompt_types
        self.choice_point_type = choice_point_type  # 'center_point'
        self.image_embeddings = None
        self.adapter = adapter
        self.batch_size = batch_size
        self.iter_num_per_epoch = self.length // batch_size

    def load_data(self):
        image_paths = glob(osp.join(self.image_dir, '*.tif'))

        with ThreadPoolExecutor() as executor:
            results = executor.map(self.process_image, image_paths)

        # Collect results
        reduce_data_num = 0
        for tif_name in results:
            if tif_name:
                self.filenames.append(tif_name)
            else:
                reduce_data_num += 1
        print(f'loading data finish, invalid data num: {reduce_data_num}')


    def load_data_ref(self):
        image_paths = glob(osp.join(f'{self.image_dir}_ref', '*.tif'))

        with ThreadPoolExecutor() as executor:
            results = executor.map(self.process_image, image_paths)

        # Collect results
        reduce_data_num = 0
        for tif_name in results:
            if tif_name:
                self.filenames.append(tif_name)
            else:
                reduce_data_num += 1
        print(f'loading data finish, invalid data num: {reduce_data_num}')

    def process_image(self, image_path):
        tif_name = osp.basename(image_path)
        gt_mask = cv2.imread(osp.join(self.gt_mask_dir, tif_name), 0)

        # 标记所有连通区域
        labeled_array, num_features = ndimage.label(gt_mask == 255)

        if num_features == 0:  # 没有连通区域
            return None

        # 创建一个与原始掩膜相同大小的全零掩膜
        filtered_mask = np.zeros_like(gt_mask)

        # 遍历所有连通区域并过滤面积小于200的区域
        for label_id in range(1, num_features + 1):
            region_area = np.sum(labeled_array == label_id)
            if region_area >= self.min_building_area / 2:
                filtered_mask[labeled_array == label_id] = 255

        # 计算过滤后的总建筑面积
        total_area = np.sum(filtered_mask == 255)
        if total_area > self.min_building_area:
            return tif_name
        return None

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
        np.random.shuffle(self.filenames)

    def drag_hole_to_mask(self, binary_image):
        # 获取所有非零点的坐标
        non_zero_points = np.argwhere(binary_image > 0)
        assert len(non_zero_points) > 3, f'the num non-zero-points of binary_map < 3!,' \
                                         f' max-value: {np.max(binary_image)}, non_zero_points: {non_zero_points}'
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

    def add_mask_noise2mask_tif(self, pred_mask, labeled_array, noise_label_ids):
        for label_id in noise_label_ids:
            pred_mask[labeled_array == label_id] = 1
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

    def bbox2transform(self, bbox, shape):
        scale = self.rand(self.box_scales[0], self.box_scales[1])
        bbox = self.scale_bbox_xywh_to_xyxy(bbox,
                                            W=shape[1],
                                            H=shape[0],
                                            scale=scale,
                                            ext_pixel=self.box_ext_pixel)

        return bbox

    def make_batch_sample(self, batch_filenames):
        # if self.mode == 'val':
        #     np.random.seed(seed=666)
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
        line_coords = []
        positive_num = np.random.choice([1, 2, 3])
        negative_num = np.random.choice([0, 1, 2, 3])
        add_building_num = np.random.choice([0, 0, 0, 1, 2, 3])

        for i in range(self.batch_size):
            tif_name = batch_filenames[i]
            image = Image.open(osp.join(self.image_dir, tif_name)).convert('RGB')
            filename = tif_name.split('.')[0]
            image_tensor = self.image_transform(image)
            # 加载注释
            gt_masks = cv2.imread(osp.join(self.gt_mask_dir, tif_name), 0)
            shape = gt_masks.shape
            gt_masks = np.asarray(gt_masks, dtype=np.uint8)
            gt_masks[gt_masks == 255] = 1

            labeled_array, num_features = ndimage.label(gt_masks == 1)

            # 遍历所有连通区域并过滤面积小于200的区域
            if prompt_type == 4 or prompt_type == 1:
                list_num_label = []  # 符合条件的连通区域
                max_id = 0
                max_area = 0
                for label_id in range(1, num_features + 1):
                    region_area = np.sum(labeled_array == label_id)
                    if region_area >= max_area:
                        max_area = region_area
                        max_id = label_id
                        # filtered_mask[labeled_array == label_id] = 255
                list_num_label.append(max_id)
            else:
                # 遍历所有连通区域并过滤面积小于200的区域
                list_num_label = []  # 符合条件的连通区域
                for label_id in range(1, num_features + 1):
                    region_area = np.sum(labeled_array == label_id)
                    if region_area >= self.min_building_area / 2:
                        # filtered_mask[labeled_array == label_id] = 255
                        list_num_label.append(label_id)

            assert len(list_num_label) > 0, '没有符合条件的连通区域'

            # 随机选择一个连通区域
            selected_label = np.random.choice(list_num_label)
            # 创建一个与原始掩膜相同大小的全零掩膜
            gt_mask = np.zeros_like(gt_masks)

            # 将选定的连通区域设置为255
            gt_mask[labeled_array == selected_label] = 1
            assert np.max(gt_mask) > 0, 'gt_mask为全0！'
            # 获取该连通区域的非零区域坐标
            coords = np.where(gt_mask == 1)
            y_min, x_min = np.min(coords[0]), np.min(coords[1])
            y_max, x_max = np.max(coords[0]), np.max(coords[1])
            bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)

            if prompt_type == 1:
                if self.choice_point_type == 'center_point':
                    pt = cal_center_point_for_binary_mask(gt_mask)
                else:
                    pt = random_click(np.array(gt_mask), value=1)
                point_coords.append([pt])
                point_labels.append([1])

            elif prompt_type == 2:
                new_bbox = self.bbox2transform(bbox, shape)
                boxes.append(new_bbox)

            elif prompt_type == 4:
                # points = generate_spline_point_within_mask(np.array(gt_mask), N=5)
                points, lines = generate_spline_point_within_mask(np.array(gt_mask), N=5, keep_lines=True)
                point_coords.append(points)
                point_labels.append([1 for _ in range(len(points))])

            else:
                pred_mask = np.asarray(copy.deepcopy(gt_mask), dtype=np.uint8)
                pred_mask = self.drag_hole_to_mask(pred_mask)
                if add_building_num > 0 and len(list_num_label) > 1:
                    list_num_label.remove(selected_label)
                    if add_building_num < len(list_num_label):
                        noise_label_ids = np.random.choice(list_num_label, add_building_num)
                    else:
                        noise_label_ids = list_num_label
                    pred_mask = self.add_mask_noise2mask_tif(pred_mask, labeled_array, noise_label_ids)

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

        if prompt_type == 4:
            line_coords.append(lines)

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
            'prompt_lines': line_coords,
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
                yield sample
            else:
                sample_ids = [item * self.batch_size + dx for dx in range(self.batch_size)]
                batch_filenames = []
                for sample_id in sample_ids:
                    batch_filenames.append(self.filenames[sample_id])

                assert len(batch_filenames) == self.batch_size
                sample = self.make_batch_sample(batch_filenames)
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
    image_dir = '/home/zzl/datasets/waterbody_data/HBD_water_data/test/image'
    ann_path = '/home/zzl/datasets/waterbody_data/HBD_water_data/test/label_checked'

    # Initialize dataset
    iter_seg_data = HBDWaterDataset(image_dir=image_dir, gt_mask_dir=ann_path,
                             debug=False,
                             mode='val',
                             model=None,
                             device=None,
                             box_scales=[0.8, 1.2],
                             box_ext_pixel=6,
                             batch_size=1,
                             adapter='sam_ours',
                             choice_point_type='center_point',
                             prompt_types=[1, 3, 4]
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

    save_test_sample_dir = '/home/zzl/datasets/waterbody_data/HBD_water_data/test_data_vis_ref_latest_blue'
    os.makedirs(save_test_sample_dir, exist_ok=True)
    data_generator = iter_seg_data.generate_sample_std()
    num = 0
    for i, data in enumerate(data_generator):
        prompt_type = data['prompt_type']
        filename = data['filename'][0]
        print(length, i, prompt_type)
        save_test_data = osp.join(save_test_sample_dir, 'test_vis_data')
        os.makedirs(save_test_data, exist_ok=True)
        torch.save(data, f'{save_test_data}/{filename}.pt')
        # continue

        alpha = 0.75
        marker_size = 300

        lines = []
        if prompt_type == 4:
            lines = data['prompt_lines'][0]

        image = image_transform(data['image'][0])
        gt_mask = data['label'].cpu().numpy()[0][0]

        if True:
            save_dirs = osp.join(save_test_sample_dir, 'paper_vis_results', 'image-gt_masks')
            os.makedirs(save_dirs, exist_ok=True)
            fig, axs = plt.subplots(1, 1, figsize=(5, 5))  # Change to 1 row, 4 columns
            fig.tight_layout()
            rgb_color = [0, 0, 255]  # [255, 255, 0] 黄色
            rgb_mask = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
            rgb_mask[gt_mask > 0] = rgb_color
            gt_result = image * (1 - gt_mask[:, :, np.newaxis]) + \
                        (1 - alpha) * gt_mask[:, :, np.newaxis] * image + \
                        alpha * rgb_mask
            gt_result = gt_result.astype(np.uint8)

            axs.imshow(gt_result)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除子图间的空白
            # axs[1].set_title('gt')
            axs.axis('off')
            # plt.show()
            plt.savefig(osp.join(save_dirs, filename + '.tif'), dpi=150)
            plt.close()

        # image/gt-check
        save_dirs = osp.join(save_test_sample_dir, 'paper_vis_results', 'image-prompts')
        os.makedirs(save_dirs, exist_ok=True)
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))  # Change to 1 row, 4 columns
        fig.tight_layout()

        if prompt_type != 3:
            result = image.astype(np.uint8)
        else:
            rgb_color = [0, 0, 255]  # [255, 255, 0] 黄色
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
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='yellow', facecolor='none')
            axs.add_patch(rect)

        elif prompt_type == 4:
            # 绘制样条曲线，仅连接在掩膜区域内的点
            for i in range(len(lines) - 1):
                pt1 = lines[i]
                pt2 = lines[i + 1]
                axs.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='yellow', linewidth=3, linestyle='-')
            # points = data['prompt_point'][0][0].cpu().numpy()
            # xs = [point[0] for point in points]
            # ys = [point[1] for point in points]
            # # 在图形上绘制采样点
            # axs.scatter(xs, ys, color='g', marker='*', s=marker_size)

        else:
            points = data['prompt_point'][0][0].cpu().numpy()
            x = [point[0] for point in points]
            y = [point[1] for point in points]
            point_labels = data['prompt_point'][1][0]

            for i in range(len(x)):
                if point_labels[i] == 1:
                    axs.scatter(x[i], y[i], color='yellow', marker='*', s=marker_size)  # mistaken_points
                else:
                    axs.scatter(x[i], y[i], color='r', marker='*', s=marker_size)  # mistaken_points

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除子图间的空白
        # axs[1].set_title('gt')
        axs.axis('off')
        # plt.show()
        plt.savefig(osp.join(save_dirs, filename + '.tif'), dpi=150)
        plt.close()
