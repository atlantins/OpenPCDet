import struct
import time

'''
# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z _ intensity t reflectivity ring ambient range
SIZE 4 4 4 1 4 4 2 2 2 1 4 1
TYPE F F F U F U U U U U U U
COUNT 1 1 1 4 1 1 1 1 1 0 1 14
WIDTH 2048
HEIGHT 64
VIEWPOINT 0 0 0 1 0 0 0
POINTS 131072
'''
# 打开PCD文件
with open('../1686039063.144342823.pcd', 'rb') as file:
    # 跳过PCD文件头部
    for _ in range(11):
        if _ == 3:
            size = file.readline()
            string = size.decode('utf-8')   # 这里如果不这么写，写成str()就会变成 b'SIZE 4 4 4 1 4 4 2 2 2 1 4 1\n',然后需要用replace方法去掉\n
            # numbers = [int(num) for id,num in enumerate(string.split()[1:]) if id != 3]  # 不计算下划线
            numbers = [int(num) for id,num in enumerate(string.split()[1:])]
            print(numbers)
            data_size = sum(numbers)
            print(data_size)
        else:    
            file.readline()

    # 读取点云数据
    data = file.read()

# 解析每个点的数据
point_size = 34 # 每个点的大小，根据字段定义计算得出  F F F U F U U U U U U U    4F 8U
num_points = 131072  # 点云数量，根据POINTS行获取
for i in range(num_points):
    point_data = data[i * point_size:(i + 1) * point_size]
    print(point_data)

    # 使用struct.unpack解析每个点的数据
    x, y, z, _, intensity, t, reflectivity, ring, ambient, range_value = struct.unpack('fff H f HHHHHHH', point_data)

    # 处理数据
    # 示例：打印点的坐标和其他字段的值
    print(f"Point {i + 1}: x={x}, y={y}, z={z}, intensity={intensity}, t={t}, reflectivity={reflectivity}, ring={ring}, ambient={ambient}, range={range_value}")

    time.sleep(5)