import torch
import os.path as osp
import os
from tqdm import tqdm
from glob import glob


test_data_cuda_dir = '/home/zzl/datasets/building_data/EVLab_building/test_data'
list_test_data = glob(osp.join(test_data_cuda_dir, '*.pt'))
device = torch.device('cpu')
save_test_cpu_dir = '/home/zzl/datasets/building_data/EVLab_building/test_data_cpu'
os.makedirs(save_test_cpu_dir, exist_ok=True)
for pt_path in tqdm(list_test_data):
    batch = torch.load(pt_path)
    points = batch['prompt_point']
    if points is not None:
        point_xy, point_label = list(points)
        point_xy = point_xy.to(device)
        point_label = point_label.to(device)
        points = (point_xy, point_label)
    boxes = batch['prompt_box']
    if boxes is not None:
        boxes = boxes.to(device)
    batch['prompt_point'] = points
    batch['prompt_box'] = boxes
    filename = batch['filename'][0]
    # torch.save(data, f'{save_test_sample_dir}/data_{i}.pt')
    torch.save(batch, f'{save_test_cpu_dir}/{filename}.pt')