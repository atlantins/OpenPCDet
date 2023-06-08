import struct

file_path = '../../data/test_minikitti/train/velodyne/000001.bin'

with open(file_path, 'rb') as file:
    data = file.read()

    # 每个数据点的字节数（每个点有 x、y、z 和 intensity 四个值）
    point_size = 16

    # 计算数据点的数量
    num_points = len(data) // point_size

    # 逐个打印每个数据点的坐标
    for i in range(num_points):
        offset = i * point_size
        point_data = data[offset:offset + point_size]

        # 从字节数据中解析出每个数据点的 x、y、z 和 intensity 值
        x = struct.unpack('f', point_data[0:4])[0]
        y = struct.unpack('f', point_data[4:8])[0]
        z = struct.unpack('f', point_data[8:12])[0]
        intensity = struct.unpack('f', point_data[12:16])[0]

        print(f"Point {i+1}: x={x}, y={y}, z={z}, intensity={intensity}")
