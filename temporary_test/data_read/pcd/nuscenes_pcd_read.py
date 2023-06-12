from pyntcloud import PyntCloud

# 读取 PCD 文件
cloud = PyntCloud.from_file("../n008-2018-08-01-15-16-36-0400__RADAR_BACK_LEFT__1533151603522238.pcd")

# 获取点云数据
points = cloud.points

colums = cloud.points.columns
# 打印点的坐标
print(colums)
print(points[["x", "y", "z"]])
