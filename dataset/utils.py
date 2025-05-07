import numpy as np
import cv2
import skimage.measure
from scipy.ndimage import binary_erosion, center_of_mass
from scipy.spatial.distance import cdist


def segmentation2mask(segmentation,
                      mask=np.zeros((512, 512), dtype=np.uint8)):

    # 对于该对象的每个多边形，用 1 填充第一个多边形，然后用 0 填充剩下的多边形
    for i, poly in enumerate(segmentation):
        # 由于 COCO 的分割坐标是一维的，我们需要将它们转换为一个 N*2 的数组
        poly = np.array(poly).reshape(-1, 2)
        # 如果是外部多边形，我们用 1 填充，如果是内部多边形（空洞），我们用 0 填充
        color = 0 if i > 0 else 1
        # 使用 cv2.fillPoly 函数将多边形绘制到 mask 上
        cv2.fillPoly(mask, [poly.astype(int)], color)
    return mask


def annotations2mask(annotations, shape=(512, 512)):
    # 创建一个全零数组，大小为图像的高度和宽度
    mask = np.zeros(shape, dtype=np.uint8)
    for ann in annotations:
        # 对于该对象的每个多边形，用 1 填充第一个多边形，然后用 0 填充剩下的多边形
        mask = segmentation2mask(ann['segmentation'], mask)
    return mask


def polys2mask(polys, shape=(512, 512)):
    mask = np.zeros(shape, dtype=np.uint8)
    for i, poly in enumerate(polys):
        # 由于 COCO 的分割坐标是一维的，我们需要将它们转换为一个 N*2 的数组
        poly = poly.reshape(-1, 2)
        # 如果是外部多边形，我们用 1 填充，如果是内部多边形（空洞），我们用 0 填充
        color = 0 if i > 0 else 1
        # 使用 cv2.fillPoly 函数将多边形绘制到 mask 上
        cv2.fillPoly(mask, [poly.astype(int)], color)
    return mask


def polys2boundarymask(polys, shape=(512, 512), expansion_pixels=2):
    mask = np.zeros(shape, dtype=np.uint8)

    for i, poly in enumerate(polys):
        # 将多边形坐标转换为 N*2 的数组
        poly = np.array(poly).reshape(-1, 2)
        # 绘制多边形的边界
        cv2.polylines(mask, [poly.astype(int)], isClosed=True, color=1, thickness=1)

    # 创建扩张核（3x3的正方形结构元素）
    kernel = np.ones((2 * expansion_pixels + 1, 2 * expansion_pixels + 1), np.uint8)

    # 扩张mask边界
    boundary_mask = cv2.dilate(mask, kernel, iterations=1)

    return boundary_mask


def cal_center_point_for_binary_mask(binary_mask):
    binary_mask = np.array(binary_mask > 0, dtype=np.int)
    if np.sum(binary_mask) == 0:
        # If the mask is empty (all zeros), return a default value or handle appropriately
        return [0, 0]  # You can change this to another default value if needed

    # Calculate the center of mass for the binary_mask
    convex_center = center_of_mass(binary_mask)

    # Ensure the center point is within the connected region
    if np.isnan(convex_center[0]) or np.isnan(convex_center[1]):
        return [0, 0]  # Return a default value if the center of mass is NaN

    # Ensure the center point is within the connected region
    center_point = [int(convex_center[0]), int(convex_center[1])]

    if not binary_mask[tuple(center_point)]:
        # If the calculated center is not within the connected region, adjust the center
        distances = np.sqrt((np.indices(binary_mask.shape)[0] - convex_center[0]) ** 2 +
                            (np.indices(binary_mask.shape)[1] - convex_center[1]) ** 2)
        mask = binary_mask > 0
        adjusted_center = np.unravel_index(np.argmin(distances * mask + (1 - mask) * distances.max()),
                                           binary_mask.shape)
        center_point = adjusted_center

    return center_point[::-1]


def random_click(mask, value=1):
    indices = np.argwhere(mask == value)
    point = indices[np.random.randint(len(indices))]
    return point[::-1].copy()  # x, y


def scale_bbox_xywh_to_xyxy(bbox, W, H, scale=1.0, ext_pixel=6):
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
    cx, cy = x + width/2, y+height/2
    wh = np.array([width, height], dtype=np.float32) * scale
    width, height = wh.tolist()
    x1, y1, x2, y2 = cx - width/2, cy - height/2, cx + width/2, cy + height/2
    dx = np.random.choice(np.arange(-ext_pixel, ext_pixel+1, 2))
    dy = np.random.choice(np.arange(-ext_pixel, ext_pixel+1, 2))
    x1 = max(x1 - dx, 0)
    y1 = max(y1 - dy, 0)
    x2 = min(x2 + dx, W)
    y2 = min(y2 + dy, H)

    return [x1, y1, x2, y2]


def sample_points_from_binary_mask(binary_image, min_area=20, border_width=2):
    """
    在二值图像的最大连通域内部，非边界处选择距离边界最远的一个点。

    :param binary_image: 二值图像。
    :param min_area: 最小连通域面积。
    :param border_width: 边界宽度，用于排除边界。
    :return: 距离边界最远的点，或None。
    """
    # 标记连通域
    labeled_image = skimage.measure.label(binary_image)
    regions = skimage.measure.regionprops(labeled_image)

    # 找到最大的连通域
    max_region = max(regions, key=lambda r: r.area, default=None)

    # 检查连通域面积是否足够大
    if max_region is None or max_region.area < min_area:
        return []

    # 排除边界
    max_region_mask = (labeled_image == max_region.label)
    eroded_mask = binary_erosion(max_region_mask, iterations=border_width)

    # 获取内部点和边界点
    inner_points = np.argwhere(eroded_mask)
    boundary_points = np.argwhere(max_region_mask & ~eroded_mask)

    if len(inner_points) == 0 or len(boundary_points) == 0:
        return []

    # 计算所有内部点到所有边界点的距离
    distances = cdist(inner_points, boundary_points)

    # 找到距离边界最远的点
    farthest_point_index = np.argmax(np.min(distances, axis=1))
    farthest_point = inner_points[farthest_point_index]

    return [(farthest_point[1], farthest_point[0])]


def calculate_iou(image1, image2):
    intersection = np.logical_and(image1, image2)
    union = np.logical_or(image1, image2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def random_dilate_or_erode(binary_image):
    kernel_size = np.random.randint(7, 14)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    operation = np.random.choice([cv2.MORPH_DILATE, cv2.MORPH_ERODE])
    processed_image = cv2.morphologyEx(binary_image, operation, kernel)
    return processed_image


def is_valid_point(new_point, selected_points, min_distance):
    if len(selected_points) == 0:
        return True
    selected_points_array = np.array(selected_points)
    distances = np.linalg.norm(selected_points_array - np.array(new_point), axis=1)
    return np.all(distances >= min_distance)


def select_random_points(binary_image, num_points, min_distance):
    # 获取所有非零点的坐标
    non_zero_points = np.argwhere(binary_image == 1)
    selected_points = []

    while len(selected_points) < num_points and len(non_zero_points) > 0:
        # 随机选取一个非零点
        idx = np.random.randint(0, len(non_zero_points))
        new_point = tuple(non_zero_points[idx])

        if is_valid_point(new_point, selected_points, min_distance):
            selected_points.append(new_point)
            # 移除已选点
            non_zero_points = np.delete(non_zero_points, idx, 0)

    return selected_points


def select_uniform_points(binary_image, num_points):
    # 获取所有非零点的坐标
    non_zero_points = np.argwhere(binary_image == 1)

    if len(non_zero_points) == 0:
        raise ValueError("No non-zero points found in the binary image.")
    if len(non_zero_points) < num_points:
        return non_zero_points.tolist()

    # 均匀采样非零点
    indices = np.linspace(0, len(non_zero_points) - 1, num_points, dtype=int)
    selected_points = non_zero_points[indices]

    return selected_points.tolist()


def select_random_pointsV2(binary_image, num_points):
    # 获取所有非零点的坐标
    non_zero_points = np.argwhere(binary_image == 1)

    if len(non_zero_points) == 0:
        raise ValueError("No non-zero points found in the binary image.")

    if len(non_zero_points) < num_points:
        num_points = len(non_zero_points)

    # 随机选择 num_points 个非零点
    selected_indices = np.random.choice(len(non_zero_points), num_points, replace=False)
    selected_points = non_zero_points[selected_indices]

    return selected_points.tolist()