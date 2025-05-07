import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob


def load_image_and_label(image_path, label_path):
    image = Image.open(image_path)
    label = Image.open(label_path)
    return np.array(image), np.array(label)


def save_image_and_label(image, label, save_dir, prefix, index):
    image = Image.fromarray(image)
    label = Image.fromarray(label)
    image.save(os.path.join(save_dir, 'image', f"{prefix}_{index}.tif"))
    label.save(os.path.join(save_dir, 'label', f"{prefix}_{index}.tif"))


def crop_image_and_label(image, label, crop_size, overlap_rate, save_dir, prefix):
    stride = int(crop_size * (1 - overlap_rate))
    h, w = image.shape[:2]
    index = 0

    for i in tqdm(range(0, h, stride)):
        for j in range(0, w, stride):
            if i + crop_size > h and j + crop_size < w:
                crop_img = image[-crop_size:, j:j + crop_size]
                crop_label = label[-crop_size:, j:j + crop_size]
            elif j + crop_size > w and i + crop_size < h:
                crop_img = image[i:i + crop_size, -crop_size:]
                crop_label = label[i:i + crop_size, -crop_size:]
            elif j + crop_size > w and i + crop_size > h:
                crop_img = image[-crop_size:, -crop_size:]
                crop_label = label[-crop_size:, -crop_size:]
            else:
                crop_img = image[i:i + crop_size, j:j + crop_size]
                crop_label = label[i:i + crop_size, j:j + crop_size]
            save_image_and_label(crop_img, crop_label, save_dir, prefix, index)
            index += 1


def process_dataset(image_dir, label_dir, train_indices, val_indices, crop_size, train_overlap, val_overlap, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx in train_indices:
        image_path = os.path.join(image_dir, f"top_mosaic_09cm_area{idx}.tif")
        label_path = os.path.join(label_dir, f"top_mosaic_09cm_area{idx}.tif")
        image, label = load_image_and_label(image_path, label_path)
        crop_image_and_label(image, label, crop_size, train_overlap, os.path.join(save_dir, 'train'), f"sample_{idx}")

    for idx in val_indices:
        image_path = os.path.join(image_dir, f"top_mosaic_09cm_area{idx}.tif")
        label_path = os.path.join(label_dir, f"top_mosaic_09cm_area{idx}.tif")
        image, label = load_image_and_label(image_path, label_path)
        crop_image_and_label(image, label, crop_size, val_overlap, os.path.join(save_dir, 'val'), f"sample_{idx}")


# 定义 RGB 到类别标记的映射关系
rgb_to_label = {
    (255, 255, 255): 1,  # Impervious surface
    (0, 0, 255): 2,  # Building
    (0, 255, 255): 3,  # Low Vegetation
    (0, 255, 0): 4,  # Tree
    (255, 255, 0): 5,  # Car
    (255, 0, 0): 0  # Clutter/background
}


def rgb_to_single_channel(label_image):
    # 获取图像的宽和高
    height, width, _ = label_image.shape
    # 初始化单通道标签图像
    single_channel_label = np.zeros((height, width), dtype=np.uint8)

    # 遍历每一个像素点，并进行映射
    for rgb, label in rgb_to_label.items():
        mask = np.all(label_image == rgb, axis=-1)
        single_channel_label[mask] = label

    return single_channel_label


def label_transform_to_255(label_path, value=4, save_dir=''):
    label = Image.open(label_path)
    label = np.array(label)
    label255 = np.zeros_like(label, dtype=np.uint8)
    label255[label==value] = 255
    label255 = Image.fromarray(label255)
    label255.save(
        os.path.join(save_dir, os.path.basename(label_path))
    )


def process_label_images(label_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.tif')]

    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        label_image = np.array(Image.open(label_path))
        single_channel_label = rgb_to_single_channel(label_image)

        save_path = os.path.join(save_dir, label_file)
        Image.fromarray(single_channel_label).save(save_path)


if __name__ == '__main__':
    # 参数设置
    # image_dir = r"/home/zzl/datasets/ISPRS_data/Vaihingen_data/top"
    # label_dir = r"/home/zzl/datasets/ISPRS_data/Vaihingen_data/gts"
    # train_indices = [1, 3, 5, 7, 13, 17, 21, 23, 26, 32, 37]
    # # train_indices = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 16, 17, 20, 21, 22, 23, 24, 26, 27, 29, 31, 32, 33, 35, 37,
    # #                  38]
    #
    # # val_indices = [11, 15, 28, 30, 34]
    # val_indices = []
    # crop_size = 512
    # train_overlap = 0.5
    # val_overlap = 0.25
    # save_dir = r"/home/zzl/datasets/ISPRS_data/Vaihingen_data/vaihingen_dataset"
    # os.makedirs(save_dir, exist_ok=True)
    # os.makedirs(os.path.join(save_dir, 'train', 'image'), exist_ok=True)
    # os.makedirs(os.path.join(save_dir, 'val', 'image'), exist_ok=True)
    # os.makedirs(os.path.join(save_dir, 'train', 'label'), exist_ok=True)
    # os.makedirs(os.path.join(save_dir, 'val', 'label'), exist_ok=True)
    # # 处理数据集
    # process_dataset(image_dir, label_dir, train_indices, val_indices,
    #                 crop_size, train_overlap, val_overlap, save_dir)

    '''rgb2gray'''
    # # 参数设置
    # label_dir = "/home/zzl/datasets/ISPRS_data/Vaihingen_data/rgb_gts"
    # save_dir = "/home/zzl/datasets/ISPRS_data/Vaihingen_data/gts"
    # os.makedirs(save_dir, exist_ok=True)
    # # 处理标签图像
    # process_label_images(label_dir, save_dir)

    label_dir = '/home/zzl/datasets/ISPRS_data/Vaihingen_data/vaihingen_dataset/train_large/label'
    save_dir = '/home/zzl/datasets/ISPRS_data/Vaihingen_data/vaihingen_dataset/train_large/label_woodland'
    os.makedirs(save_dir, exist_ok=True)
    list_label_path = glob(os.path.join(label_dir, '*.tif'))
    for label_path in tqdm(list_label_path):
        label_transform_to_255(label_path, value=4, save_dir=save_dir)