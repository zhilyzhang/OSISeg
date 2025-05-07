import torch
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from dataset.before_data_sampler.interactive_segment_loader import image_transform


def vis_image(imgs, pred_masks, gt_masks, save_path, reverse = False, points = None):

    b ,c ,h ,w = pred_masks.size()
    dev = pred_masks.get_device()
    row_num = min(b, 4)

    if torch.max(pred_masks) > 1 or torch.min(pred_masks) < 0:
        pred_masks = torch.sigmoid(pred_masks)

    if reverse == True:
        pred_masks = 1 - pred_masks
        gt_masks = 1 - gt_masks
    if c == 2:
        pred_disc, pred_cup = pred_masks[: ,0 ,: ,:].unsqueeze(1).expand(b ,3 ,h ,w), pred_masks[: ,1 ,: ,:].unsqueeze \
            (1).expand(b ,3 ,h ,w)
        gt_disc, gt_cup = gt_masks[: ,0 ,: ,:].unsqueeze(1).expand(b ,3 ,h ,w), gt_masks[: ,1 ,: ,:].unsqueeze \
            (1).expand(b ,3 ,h ,w)
        tup = \
        (imgs[:row_num, :, :, :], pred_disc[:row_num, :, :, :], pred_cup[:row_num, :, :, :], gt_disc[:row_num, :, :, :],
        gt_cup[:row_num, :, :, :])
        # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        compose = torch.cat((pred_disc[:row_num, :, :, :], pred_cup[:row_num, :, :, :], gt_disc[:row_num, :, :, :],
                             gt_cup[:row_num, :, :, :]), 0)
        vutils.save_image(compose, fp=save_path, nrow=row_num, padding=10)
    else:
        # imgs = torchvision.transforms.Resize((h, w))(imgs)
        # if imgs.size(1) == 1:
        #     imgs = imgs[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
        pred_masks = pred_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
        gt_masks = gt_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
        if points != None:
            for i in range(b):
                p = np.round(points.cpu() / 512 * 512).to(dtype=torch.int)
                # gt_masks[i,:,points[i,0]-5:points[i,0]+5,points[i,1]-5:points[i,1]+5] = torch.Tensor([255, 0, 0]).to(dtype = torch.float32, device = torch.device('cuda:' + str(dev)))
                gt_masks[i, 0, p[i, 0] - 5:p[i, 0] + 5, p[i, 1] - 5:p[i, 1] + 5] = 0.5
                gt_masks[i, 1, p[i, 0] - 5:p[i, 0] + 5, p[i, 1] - 5:p[i, 1] + 5] = 0.1
                gt_masks[i, 2, p[i, 0] - 5:p[i, 0] + 5, p[i, 1] - 5:p[i, 1] + 5] = 0.4
        tup = (imgs[:row_num, :, :, :], pred_masks[:row_num, :, :, :], gt_masks[:row_num, :, :, :])
        # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        compose = torch.cat(tup, 0)
        vutils.save_image(compose, fp=save_path, nrow=row_num, padding=10)

    return


def vis_tensor(imgs, pred_masks, gt_masks, points, save_path=''):
    batch = imgs.shape[0]
    num = min(2, batch)
    alpha = 0.75
    marker_size = 200
    fig, axs = plt.subplots(1*num, 4, figsize=(20, 5*num))  # Change to 1 row, 4 columns
    # fig.tight_layout()
    for i in range(num):
        pts = points[i].numpy()
        image = image_transform(imgs[i].cpu())
        gt_mask = gt_masks[i].cpu().numpy()[0]
        pred_mask = pred_masks[i].detach().cpu().numpy()[0]

        axs[i][0].imshow(image, cmap='jet')
        axs[i][0].set_title('image')
        axs[i][0].axis('off')

        # gt
        rgb_color = [255, 255, 0]  # 黄色
        rgb_mask = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        rgb_mask[gt_mask > 0] = rgb_color

        result = image * (1 - gt_mask[:, :, np.newaxis]) + \
                 (1 - alpha) * gt_mask[:, :, np.newaxis] * image + \
                 alpha * rgb_mask
        result = result.astype(np.uint8)
        axs[i][1].imshow(result)
        axs[i][1].imshow(result)

        if len(pts) == 4:
            x1, y1, x2, y2 = pts
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='green', facecolor='none')
            axs[i][1].add_patch(rect)
        else:
            x = [point[0] for point in [pts]]
            y = [point[1] for point in [pts]]
            axs[i][1].scatter(x, y, color='r', marker='*', s=marker_size)  # mistaken_points
        axs[i][1].set_title('image/gt')
        axs[i][1].axis('off')

        axs[i][2].imshow(gt_mask)
        axs[i][2].set_title('gt')
        axs[i][2].axis('off')

        axs[i][3].imshow(pred_mask)
        axs[i][3].set_title('pred')
        axs[i][3].axis('off')
    # plt.show()
    plt.savefig(save_path, dpi=150)
    plt.close()


def vis_tensor_single_point(imgs, pred_masks, gt_masks, points):
    batch = imgs.shape[0]
    num = min(1, batch)
    alpha = 0.75
    marker_size = 200
    fig, axs = plt.subplots(1*num, 4, figsize=(20, 5*num))  # Change to 1 row, 4 columns
    fig.tight_layout()
    for i in range(num):
        pts = points[i].numpy()
        image = image_transform(imgs[i].cpu())
        gt_mask = gt_masks[i].cpu().numpy()[0]
        pred_mask = pred_masks[i].detach().cpu().numpy()[0]

        axs[0].imshow(image, cmap='jet')
        axs[0].set_title('image')
        axs[0].axis('off')

        # gt
        rgb_color = [255, 255, 0]  # 黄色
        rgb_mask = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        rgb_mask[gt_mask > 0] = rgb_color

        result = image * (1 - gt_mask[:, :, np.newaxis]) + \
                 (1 - alpha) * gt_mask[:, :, np.newaxis] * image + \
                 alpha * rgb_mask
        result = result.astype(np.uint8)
        axs[1].imshow(result)
        axs[1].imshow(result)

        x = [point[0] for point in [pts]]
        y = [point[1] for point in [pts]]
        axs[1].scatter(x, y, color='r', marker='*', s=marker_size)  # mistaken_points
        axs[1].set_title('image/gt')
        axs[1].axis('off')

        axs[2].imshow(gt_mask)
        axs[2].set_title('gt')
        axs[2].axis('off')

        axs[3].imshow(pred_mask)
        axs[3].set_title('pred')
        axs[3].axis('off')
    plt.show()
    # plt.savefig(save_path, dpi=150)
    plt.close()


def vis_pure_tensor(imgs, pred_masks, gt_masks, save_path=''):
    batch = imgs.shape[0]
    num = min(2, batch)
    alpha = 0.75
    marker_size = 200
    fig, axs = plt.subplots(1*num, 4, figsize=(20, 5*num))  # Change to 1 row, 4 columns
    # fig.tight_layout()
    for i in range(num):
        image = image_transform(imgs[i].cpu())
        gt_mask = gt_masks[i].cpu().numpy()[0]
        pred_mask = pred_masks[i].detach().cpu().numpy()[0]

        axs[i][0].imshow(image, cmap='jet')
        axs[i][0].set_title('image')
        axs[i][0].axis('off')

        # gt
        rgb_color = [255, 255, 0]  # 黄色
        rgb_mask = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        rgb_mask[gt_mask > 0] = rgb_color

        result = image * (1 - gt_mask[:, :, np.newaxis]) + \
                 (1 - alpha) * gt_mask[:, :, np.newaxis] * image + \
                 alpha * rgb_mask
        result = result.astype(np.uint8)
        axs[i][1].imshow(result)
        axs[i][1].imshow(result)
        axs[i][1].set_title('image/gt')
        axs[i][1].axis('off')

        axs[i][2].imshow(gt_mask)
        axs[i][2].set_title('gt')
        axs[i][2].axis('off')

        axs[i][3].imshow(pred_mask)
        axs[i][3].set_title('pred')
        axs[i][3].axis('off')
    # plt.show()
    plt.savefig(save_path, dpi=150)
    plt.close()


def vis_pure_tensor_boundary(imgs, pred_masks, pred_boundary_masks, gt_masks, save_path=''):
    batch = imgs.shape[0]
    num = min(3, batch)
    alpha = 0.75
    marker_size = 200
    fig, axs = plt.subplots(1*num, 5, figsize=(25, 5*num))  # Change to 1 row, 4 columns
    # fig.tight_layout()
    for i in range(num):
        image = image_transform(imgs[i].cpu())
        gt_mask = gt_masks[i].cpu().numpy()[0]
        pred_mask = pred_masks[i].detach().cpu().numpy()[0]
        pred_boundary_mask = pred_boundary_masks[i].detach().cpu().numpy()[0]

        axs[i][0].imshow(image, cmap='jet')
        axs[i][0].set_title('image')
        axs[i][0].axis('off')

        # gt
        rgb_color = [255, 255, 0]  # 黄色
        rgb_mask = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        rgb_mask[gt_mask > 0] = rgb_color

        result = image * (1 - gt_mask[:, :, np.newaxis]) + \
                 (1 - alpha) * gt_mask[:, :, np.newaxis] * image + \
                 alpha * rgb_mask
        result = result.astype(np.uint8)
        axs[i][1].imshow(result)
        axs[i][1].imshow(result)
        axs[i][1].set_title('image/gt')
        axs[i][1].axis('off')

        axs[i][2].imshow(gt_mask)
        axs[i][2].set_title('gt')
        axs[i][2].axis('off')

        axs[i][3].imshow(pred_mask)
        axs[i][3].set_title('pred')
        axs[i][3].axis('off')

        axs[i][4].imshow(pred_boundary_mask)
        axs[i][4].set_title('pred_boundary')
        axs[i][4].axis('off')
    # plt.show()
    plt.savefig(save_path, dpi=150)
    plt.close()


def vis_prompt_area_tensor(imgs, prompt_masks, gt_masks, pred_masks, save_path=''):
    batch = imgs.shape[0]
    num = min(2, batch)
    alpha = 0.75
    marker_size = 200
    fig, axs = plt.subplots(1*num, 4, figsize=(20, 5*num))  # Change to 1 row, 4 columns
    # fig.tight_layout()
    for i in range(num):
        image = image_transform(imgs[i].cpu())
        gt_mask = gt_masks[i].cpu().numpy()[0]
        prompt_mask = prompt_masks[i].cpu().numpy()[0]
        pred_mask = pred_masks[i].detach().cpu().numpy()[0]

        axs[i][0].imshow(image, cmap='jet')
        axs[i][0].set_title('image')
        axs[i][0].axis('off')

        # prompt_mask
        rgb_color = [255, 255, 0]  # 黄色
        rgb_mask = np.zeros((*prompt_mask.shape, 3), dtype=np.uint8)
        rgb_mask[prompt_mask > 0] = rgb_color

        result = image * (1 - prompt_mask[:, :, np.newaxis]) + \
                 (1 - alpha) * prompt_mask[:, :, np.newaxis] * image + \
                 alpha * rgb_mask
        result = result.astype(np.uint8)
        axs[i][1].imshow(result)
        axs[i][1].set_title('prompt_mask')
        axs[i][1].axis('off')

        # predict
        rgb_color = [255, 255, 0]  # 黄色
        rgb_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        rgb_mask[pred_mask > 0] = rgb_color

        result = image * (1 - pred_mask[:, :, np.newaxis]) + \
                 (1 - alpha) * pred_mask[:, :, np.newaxis] * image + \
                 alpha * rgb_mask
        result = result.astype(np.uint8)
        axs[i][2].imshow(result)
        axs[i][2].set_title('predict')
        axs[i][2].axis('off')

        # gt
        rgb_color = [255, 255, 0]  # 黄色
        rgb_mask = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        rgb_mask[gt_mask > 0] = rgb_color

        result = image * (1 - gt_mask[:, :, np.newaxis]) + \
                 (1 - alpha) * gt_mask[:, :, np.newaxis] * image + \
                 alpha * rgb_mask
        result = result.astype(np.uint8)
        axs[i][3].imshow(result)
        axs[i][3].set_title('gt')
        axs[i][3].axis('off')
    # plt.show()
    plt.savefig(save_path, dpi=150)
    plt.close()
