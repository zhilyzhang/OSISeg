import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
from shapely.geometry import box

# 读取 GeoJSON 真值
geojson_path = "/home/zzl/datasets/building_data/Tanzania-building-data/label/grid_001.geojson"
buildings = gpd.read_file(geojson_path)

# 读取影像
image_path = "/home/zzl/datasets/building_data/Tanzania-building-data/image/5afeda152b6a08001185f11b.tif"
image = rasterio.open(image_path)

# 裁剪一个区域，例如选择第一个建筑物的 bounding box
building_bbox = buildings.geometry[0].bounds
minx, miny, maxx, maxy = building_bbox
bbox = box(minx, miny, maxx, maxy)
buildings_clipped = buildings[buildings.intersects(bbox)]

# 裁剪影像到这个区域
window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=image.transform)
image_clipped = image.read(window=window)

# 显示影像和建筑物
plt.figure(figsize=(10, 10))

# 显示裁剪的影像
show(image_clipped, transform=image.window_transform(window), ax=plt.gca())

# 叠加显示建筑物轮廓
buildings_clipped.plot(ax=plt.gca(), facecolor="none", edgecolor="red", linewidth=1)

plt.title("Cropped Image and Buildings")
plt.show()
