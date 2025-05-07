import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev


# def generate_spline_point_within_mask(mask, N=5):
#     # 获取掩膜中的非零像素点
#     points = np.column_stack(np.where(mask > 0))
#
#     # 在非零像素点中均匀采样N个点
#     indices = np.linspace(0, len(points) - 1, 7, dtype=int)
#     sampled_points = points[indices]
#
#     # 使用样条插值生成平滑曲线
#     tck, u = splprep([sampled_points[:, 1], sampled_points[:, 0]], s=0)
#     unew = np.linspace(0, 1, 50)
#     out = splev(unew, tck)
#
#     # 将样条插值点转换为整数坐标
#     x_spline = np.array(out[0], dtype=int)
#     y_spline = np.array(out[1], dtype=int)
#
#     # 使用布尔索引检查哪些点在掩膜区域内
#     valid_mask = (y_spline >= 0) & (y_spline < mask.shape[0]) & (x_spline >= 0) \
#                  & (x_spline < mask.shape[1]) & (mask[y_spline, x_spline] > 0)
#     valid_x_spline = x_spline[valid_mask]
#     valid_y_spline = y_spline[valid_mask]
#
#     # 在 valid_x_spline 和 valid_y_spline 中随机采样N个点
#     if len(valid_x_spline) > N:
#         # sampled_indices = np.random.choice(len(valid_x_spline), N, replace=False)
#         sampled_indices = np.linspace(0, len(valid_x_spline) - 1, N, dtype=int)
#         sampled_valid_x_spline = valid_x_spline[sampled_indices]
#         sampled_valid_y_spline = valid_y_spline[sampled_indices]
#     else:
#         repeat_factor = (N + len(valid_x_spline) - 1) // len(valid_x_spline)
#         sampled_valid_x_spline = np.tile(valid_x_spline, repeat_factor)[:N]
#         sampled_valid_y_spline = np.tile(valid_y_spline, repeat_factor)[:N]
#     prompt_points = [(sampled_valid_x_spline[i], sampled_valid_y_spline[i]) for i in range(len(sampled_valid_y_spline))]
#     return prompt_points

def generate_spline_point_within_mask(mask, N=5, keep_lines=False):
    # 获取掩膜中的非零像素点
    points = np.column_stack(np.where(mask > 0))

    # 在非零像素点中均匀采样N个点
    indices = np.linspace(0, len(points) - 1, 7, dtype=int)
    sampled_points = points[indices]

    # 使用样条插值生成平滑曲线
    tck, u = splprep([sampled_points[:, 1], sampled_points[:, 0]], s=0)
    unew = np.linspace(0, 1, 50)
    out = splev(unew, tck)

    # 将样条插值点转换为整数坐标
    x_spline = np.array(out[0], dtype=int)
    y_spline = np.array(out[1], dtype=int)

    # 使用布尔索引检查哪些点在掩膜区域内
    valid_mask = (y_spline >= 0) & (y_spline < mask.shape[0]) & (x_spline >= 0) \
                 & (x_spline < mask.shape[1])

    # 应用有效掩码
    x_spline = x_spline[valid_mask]
    y_spline = y_spline[valid_mask]

    # 进一步检查哪些点在掩膜区域内
    valid_mask = mask[y_spline, x_spline] > 0
    valid_x_spline = x_spline[valid_mask]
    valid_y_spline = y_spline[valid_mask]

    # 在 valid_x_spline 和 valid_y_spline 中随机采样N个点
    if len(valid_x_spline) > N:
        sampled_indices = np.linspace(0, len(valid_x_spline) - 1, N, dtype=int)
        sampled_valid_x_spline = valid_x_spline[sampled_indices]
        sampled_valid_y_spline = valid_y_spline[sampled_indices]
    else:
        repeat_factor = (N + len(valid_x_spline) - 1) // len(valid_x_spline)
        sampled_valid_x_spline = np.tile(valid_x_spline, repeat_factor)[:N]
        sampled_valid_y_spline = np.tile(valid_y_spline, repeat_factor)[:N]

    prompt_points = [(sampled_valid_x_spline[i], sampled_valid_y_spline[i]) for i in range(len(sampled_valid_y_spline))]
    if not keep_lines:
        return prompt_points
    else:
        lines = [(valid_x_spline[i], valid_y_spline[i]) for i in range(len(valid_y_spline))]
        return prompt_points, lines


def generate_spline_point_within_maskV2(mask, N=5, keep_lines=False):
    # 获取掩膜中的非零像素点
    points = np.column_stack(np.where(mask > 0))

    # 在非零像素点中均匀采样N个点，确保N小于或等于点数
    if len(points) < N:
        raise ValueError("掩膜中的非零像素点数量少于采样点数量。")

    indices = np.linspace(0, len(points) - 1, N, dtype=int)
    sampled_points = points[indices]

    # 使用较大的光滑度参数s进行样条插值以生成简笔画
    tck, u = splprep([sampled_points[:, 1], sampled_points[:, 0]], s=2)
    unew = np.linspace(0, 1, 100)
    out = splev(unew, tck)

    # 将样条插值点转换为整数坐标
    x_spline = np.array(out[0], dtype=int)
    y_spline = np.array(out[1], dtype=int)

    # 检查哪些点在掩膜区域内
    valid_mask = (y_spline >= 0) & (y_spline < mask.shape[0]) & (x_spline >= 0) & (x_spline < mask.shape[1])
    x_spline = x_spline[valid_mask]
    y_spline = y_spline[valid_mask]

    valid_mask = mask[y_spline, x_spline] > 0
    valid_x_spline = x_spline[valid_mask]
    valid_y_spline = y_spline[valid_mask]

    # 提取有效点
    prompt_points = [(valid_x_spline[i], valid_y_spline[i]) for i in range(len(valid_y_spline))]

    if not keep_lines:
        return prompt_points
    else:
        lines = [(valid_x_spline[i], valid_y_spline[i]) for i in range(len(valid_y_spline))]
        return prompt_points, lines


def generate_spline_point_within_mask_image_show(image_path, gt_path, N):
    # 加载影像和掩膜
    image = cv2.imread(image_path)
    mask = cv2.imread(gt_path, 0)  # 读取掩膜，灰度模式

    # 获取掩膜中的非零像素点
    points = np.column_stack(np.where(mask > 0))

    # 在非零像素点中均匀采样N个点，确保N小于或等于点数
    if len(points) < N:
        raise ValueError("掩膜中的非零像素点数量少于采样点数量。")

    indices = np.linspace(0, len(points) - 1, N, dtype=int)
    sampled_points = points[indices]

    # 使用较大的光滑度参数s进行样条插值以生成简笔画
    tck, u = splprep([sampled_points[:, 1], sampled_points[:, 0]], s=2)
    unew = np.linspace(0, 1, 100)
    out = splev(unew, tck)

    # 将样条插值点转换为整数坐标
    x_spline = np.array(out[0], dtype=int)
    y_spline = np.array(out[1], dtype=int)

    # 检查哪些点在掩膜区域内
    valid_mask = (y_spline >= 0) & (y_spline < mask.shape[0]) & (x_spline >= 0) & (x_spline < mask.shape[1])
    x_spline = x_spline[valid_mask]
    y_spline = y_spline[valid_mask]

    valid_mask = mask[y_spline, x_spline] > 0
    valid_x_spline = x_spline[valid_mask]
    valid_y_spline = y_spline[valid_mask]

    # # 获取掩膜中的非零像素点
    # points = np.column_stack(np.where(gt > 0))
    #
    # # 在非零像素点中均匀采样N个点
    # indices = np.linspace(0, len(points) - 1, 7, dtype=int)
    # sampled_points = points[indices]
    #
    # # 使用样条插值生成平滑曲线
    # tck, u = splprep([sampled_points[:, 1], sampled_points[:, 0]], s=0)
    # unew = np.linspace(0, 1, 50)
    # out = splev(unew, tck)
    #
    # # 将样条插值点转换为整数坐标
    # x_spline = np.array(out[0], dtype=int)
    # y_spline = np.array(out[1], dtype=int)
    #
    # # 使用布尔索引检查哪些点在掩膜区域内
    # valid_mask = (y_spline >= 0) & (y_spline < gt.shape[0]) & (x_spline >= 0) & (x_spline < gt.shape[1]) & (gt[y_spline, x_spline] > 0)
    # valid_x_spline = x_spline[valid_mask]
    # valid_y_spline = y_spline[valid_mask]

    # 在 valid_x_spline 和 valid_y_spline 中随机采样N个点
    if len(valid_x_spline) > N:
        # sampled_indices = np.random.choice(len(valid_x_spline), N, replace=False)
        sampled_indices = np.linspace(0, len(valid_x_spline) - 1, N, dtype=int)
        sampled_valid_x_spline = valid_x_spline[sampled_indices]
        sampled_valid_y_spline = valid_y_spline[sampled_indices]
    else:
        sampled_valid_x_spline = valid_x_spline
        sampled_valid_y_spline = valid_y_spline

    # 绘制样条曲线，仅连接在掩膜区域内的点
    for i in range(len(valid_x_spline) - 1):
        pt1 = (valid_x_spline[i], valid_y_spline[i])
        pt2 = (valid_x_spline[i + 1], valid_y_spline[i + 1])
        cv2.line(image, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)  # 蓝色线条

    # # 在影像上绘制采样点
    # for i in range(len(valid_x_spline)):
    #     cv2.circle(image, (valid_x_spline[i], valid_y_spline[i]), 3, (255, 255, 0), -1, cv2.LINE_AA)  # 绿色圆点

    # 在影像上绘制采样点
    for i in range(N):
        cv2.circle(image, (sampled_valid_x_spline[i], sampled_valid_y_spline[i]), 5, (0, 255, 0), -1, cv2.LINE_AA)  # 绿色圆点

    # 显示结果
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image with Randomly Sampled Points and Constrained Spline Curve')
    plt.show()


if __name__ == '__main__':

    # 调用函数
    image_path = '/home/zzl/datasets/diffusion_model_refined_dataset/' \
                 'test/image/EH021541_128460_120824_023828_380.tif'
    gt_path = '/home/zzl/datasets/diffusion_model_refined_dataset/' \
              'test/gt/EH021541_128460_120824_023828_380.tif'
    N = 5
    generate_spline_point_within_mask_image_show(image_path, gt_path, N)
