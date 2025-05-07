import importlib
import torch
from typing import List, Dict
from glob import glob
import copy
import os.path as osp
import torchvision.transforms as transforms
import numpy as np
from dataset.utils import polys2mask, random_click, scale_bbox_xywh_to_xyxy
from torch.utils.data import Dataset
import json
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')


class InterSegDataset(Dataset):
    def __init__(self,
                 image_paths,
                 ann_paths,
                 debug=False,
                 mode='train',
                 prompt_type=0,
                 prompt_encoder=None):
        assert len(image_paths) == len(ann_paths)
        self.image_paths = image_paths
        self.ann_paths = ann_paths
        self.mode = mode
        self.gt_json_paths = [self.get_samples_from_json_path(p) for p in self.ann_paths]
        self.prompt_type = prompt_type  # 0:point, 1: bbox
        if debug:
            gt_json_paths = copy.deepcopy(self.gt_json_paths)
            self.gt_json_paths = [s[:20] for s in gt_json_paths if len(s) > 6]
            del gt_json_paths
        self.prompt_encoder = prompt_encoder
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406],
        #                          [0.229, 0.224, 0.225])])
        self.length = sum([len(p) for p in self.gt_json_paths])

    def __len__(self):
        return self.length

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

    def make_sample(self, image_path, gt_json_path):
        with open(gt_json_path, 'r') as file:
            data = json.load(file)
        if self.mode == 'val':
            np.random.seed(666)
        annotations = data['annotations']
        ann_id = np.random.randint(len(annotations))
        ann = annotations[ann_id]

        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_transform(image)
        shape = image_tensor.shape[-2:]
        mask = self.ann2mask(ann, shape)
        pt = random_click(np.array(mask), value=1)
        pt = pt[::-1].copy()
        point_label = 1
        if self.mode == 'val':
            np.random.seed(seed=None)
        # scale = self.rand(0.85, 1.2)
        # bbox = scale_bbox_xywh_to_xyxy(ann['bbox'],
        #                                W=shape[1],
        #                                H=shape[0],
        #                                scale=scale,
        #                                ext_pixel=6)

        mask_tensor = self.mask_transform(mask)
        name = osp.basename(gt_json_path).split('.')[0]

        return {
            'image': image_tensor,
            'label': mask_tensor,
            'p_label': point_label,
            'pt': pt,  #
            'filename': name,
        }

    def __getitem__(self, item):
        item = item % self.length
        n_dataset = len(self.image_paths)
        n_pre_sample = 0
        for data_id in range(n_dataset):
            if item < n_pre_sample + len(self.gt_json_paths[data_id]):
                gt_json_path = self.gt_json_paths[data_id][item - n_pre_sample]
                filename = osp.basename(gt_json_path).split('.')[0] + '.tif'
                image_path = osp.join(self.image_paths[data_id], filename)
                sample = self.make_sample(image_path, gt_json_path)
                return sample
            else:
                n_pre_sample += len(self.image_paths[data_id])


def batch_collate_fn(batch):
    inp_data = {}
    filename = [b['filename'] for b in batch if 'filename' in b]

    images = []
    labels = []
    p_labels = []
    pts = []
    for b in batch:
        images.append(b['image'])
        labels.append(b['label'])
        p_labels.append(torch.from_numpy(np.array(b['p_label'])))
        pts.append(torch.from_numpy(np.array(b['pt'])))

    inp_data.update(
        {
            'image': torch.stack(images),
            'label': torch.stack(labels),
            'p_label': torch.stack(p_labels),
            'pt': torch.stack(pts),
            'filename': filename

        }
    )

    return inp_data

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
    train_image_paths = [r'F:\project-datasets\building_datasets\EVLabNB6Ds\EVLab-BGZ\train\images']
    train_ann_paths = [r'F:\project-datasets\building_datasets\EVLabNB6Ds\EVLab-BGZ\train\gt']


    iter_seg_data = InterSegDataset(image_paths=train_image_paths,
                                    ann_paths=train_ann_paths, debug=False, prompt_type=0)
    train_loader = DataLoader(iter_seg_data, batch_size=1, shuffle=True,
                              num_workers=1, collate_fn=batch_collate_fn,
                              pin_memory=False)
    length = iter_seg_data.length
    print(f'data size: {length}')
    prompt_type = 0
    # for i in range(length):
    #     data = iter_seg_data.__getitem__(i)
    #     alpha = 0.75
    #     marker_size = 200
    #     points = data['pt']
    #
    #     fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Change to 1 row, 4 columns
    #     fig.tight_layout()
    #     image = image_transform(data['image'])
    #     gt_mask = data['label'].cpu().numpy()[0]
    #
    #     axs[0].imshow(image, cmap='jet')
    #     axs[0].set_title('image')
    #     axs[0].axis('off')
    #
    #     axs[1].imshow(gt_mask)
    #     axs[1].set_title('gt')
    #     axs[1].axis('off')
    #
    #     # image/gt-check
    #     rgb_color = [255, 255, 0]  # 黄色
    #     rgb_mask = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    #     rgb_mask[gt_mask > 0] = rgb_color
    #
    #     result = image * (1 - gt_mask[:, :, np.newaxis]) + \
    #              (1 - alpha) * gt_mask[:, :, np.newaxis] * image + \
    #              alpha * rgb_mask
    #     result = result.astype(np.uint8)
    #     axs[2].imshow(result)
    #
    #     if prompt_type == 1:
    #         x1, y1, x2, y2 = points
    #         rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='green', facecolor='none')
    #         axs[2].add_patch(rect)
    #     else:
    #         x = [point[0] for point in [points]]
    #         y = [point[1] for point in [points]]
    #         axs[2].scatter(x, y, color='r', marker='*', s=marker_size)  # mistaken_points
    #
    #     axs[2].set_title('image/gt')
    #     axs[2].axis('off')
    #
    #     plt.show()
    #     plt.close()
    #     prompt_type = 0 # i % 2
    #     iter_seg_data.prompt_type = prompt_type

    for i, data in enumerate(train_loader):
        # data = iter_seg_data.__getitem__(i)
        alpha = 0.75
        marker_size = 200
        points = data['pt'][0]
        print(prompt_type, points)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Change to 1 row, 4 columns
        fig.tight_layout()
        image = image_transform(data['image'][0])
        gt_mask = data['label'].cpu().numpy()[0][0]

        axs[0].imshow(image, cmap='jet')
        axs[0].set_title('image')
        axs[0].axis('off')

        axs[1].imshow(gt_mask)
        axs[1].set_title('gt')
        axs[1].axis('off')

        # image/gt-check
        rgb_color = [255, 255, 0]  # 黄色
        rgb_mask = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        rgb_mask[gt_mask > 0] = rgb_color

        result = image * (1 - gt_mask[:, :, np.newaxis]) + \
                 (1 - alpha) * gt_mask[:, :, np.newaxis] * image + \
                 alpha * rgb_mask
        result = result.astype(np.uint8)
        axs[2].imshow(result)

        if prompt_type == 1:
            x1, y1, x2, y2 = points
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='green', facecolor='none')
            axs[2].add_patch(rect)
        else:
            x = [point[0] for point in [points]]
            y = [point[1] for point in [points]]
            axs[2].scatter(x, y, color='r', marker='*', s=marker_size)  # mistaken_points

        axs[2].set_title('image/gt')
        axs[2].axis('off')

        plt.show()
        plt.close()
        prompt_type = i % 2
        iter_seg_data.prompt_type = prompt_type
