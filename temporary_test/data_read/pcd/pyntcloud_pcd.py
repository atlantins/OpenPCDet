from pyntcloud import PyntCloud

# 读取 PCD 文件
cloud = PyntCloud.from_file("../1686039063.144342823.pcd")

# 获取点云数据
points = cloud.points

colums = cloud.points.columns
# 打印点的坐标
print(colums)
print(points[["x", "y", "z"]])
