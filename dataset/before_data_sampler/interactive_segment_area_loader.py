import importlib
import torch
from typing import List, Dict
from glob import glob
import copy
import os.path as osp
import torchvision.transforms as transforms
import numpy as np
from dataset.utils import polys2mask, random_click, scale_bbox_xywh_to_xyxy, annotations2mask
from torch.utils.data import Dataset
import json
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')


class InterSegBasedAreaDataset(Dataset):
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
        prompt_mask = self.ann2mask(ann, shape)
        gt_label = annotations2mask(annotations, shape)
        gt_label_tensor = self.mask_transform(gt_label)
        prompt_mask_tensor = self.mask_transform(prompt_mask)
        name = osp.basename(gt_json_path).split('.')[0]

        return {
            'image': image_tensor,
            'prompt_mask': prompt_mask_tensor,
            'label': gt_label_tensor,
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
                n_pre_sample += len(self.gt_json_paths[data_id])


def batch_collate_fn(batch):
    inp_data = {}
    filename = [b['filename'] for b in batch if 'filename' in b]

    images = []
    labels = []
    prompt_masks = []
    for b in batch:
        images.append(b['image'])
        prompt_masks.append(b['prompt_mask'])
        labels.append(b['label'])

    inp_data.update(
        {
            'image': torch.stack(images),
            'label': torch.stack(labels),
            'prompt_mask': torch.stack(prompt_masks),
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
    train_image_paths = []
    train_ann_paths = []
    val_image_paths = []
    val_ann_paths = []
    # list_data_name = ['Taiwan', 'Guangdong', 'Chongqing', 'Zhengzhou', 'Wuhan', 'Xian']
    list_data_name = ['Guangdong']
    for data_name in list_data_name:
        train_image_paths.append(
            f'/home/zzl/datasets/building_datasets/evlab_buildings/{data_name}/new_clip512_dataset/img_train')
        train_ann_paths.append(
            f'/home/zzl/datasets/building_datasets/evlab_buildings/{data_name}/new_clip512_dataset/train_sbt_boundary_3000')

        val_image_path = f'/home/zzl/datasets/building_datasets/evlab_buildings/{data_name}/new_clip512_dataset/img_val'

        val_ann_path = f'/home/zzl/datasets/building_datasets/evlab_buildings/{data_name}/new_clip512_dataset/val_sbt_boundary_900'
        if osp.exists(val_image_path) and osp.exists(val_ann_path):
            val_image_paths.append(val_image_path)
            val_ann_paths.append(val_ann_path)


    iter_seg_data = InterSegBasedAreaDataset(image_paths=train_image_paths,
                                    ann_paths=train_ann_paths, debug=False, prompt_type=0)
    train_loader = DataLoader(iter_seg_data, batch_size=1, shuffle=True,
                              num_workers=1, collate_fn=batch_collate_fn,
                              pin_memory=False)
    length = iter_seg_data.length
    print(f'data size: {length}')

    for i, data in enumerate(train_loader):
        alpha = 0.75
        marker_size = 200
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Change to 1 row, 4 columns
        fig.tight_layout()
        image = image_transform(data['image'][0])
        prompt_mask = data['prompt_mask'].cpu().numpy()[0][0]
        gt_label = data['label'].cpu().numpy()[0][0]

        axs[0].imshow(image, cmap='jet')
        axs[0].set_title('image')
        axs[0].axis('off')

        rgb_color = [255, 255, 0]  # 黄色
        rgb_mask = np.zeros((*prompt_mask.shape, 3), dtype=np.uint8)
        rgb_mask[prompt_mask > 0] = rgb_color

        result = image * (1 - prompt_mask[:, :, np.newaxis]) + \
                 (1 - alpha) * prompt_mask[:, :, np.newaxis] * image + \
                 alpha * rgb_mask
        result = result.astype(np.uint8)

        axs[1].imshow(result)
        axs[1].set_title('prompt_mask')
        axs[1].axis('off')

        # image/gt-check
        rgb_color = [255, 255, 0]  # 黄色
        rgb_mask = np.zeros((*gt_label.shape, 3), dtype=np.uint8)
        rgb_mask[gt_label > 0] = rgb_color

        result = image * (1 - gt_label[:, :, np.newaxis]) + \
                 (1 - alpha) * gt_label[:, :, np.newaxis] * image + \
                 alpha * rgb_mask
        result = result.astype(np.uint8)
        axs[2].imshow(result)

        axs[2].set_title('image/gt')
        axs[2].axis('off')
        plt.show()
        plt.close()

