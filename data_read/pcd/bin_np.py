import numpy as np

np.set_printoptions(suppress=True,threshold=np.inf)
file_path = '../../data/v1.0-mini/v1.0-mini/sweeps/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603947935.pcd.bin'
pointcould = np.fromfile(file_path, dtype=np.float32).reshape([-1,5])
for i in range(3):
    x,y,z,intensity,ring = pointcould[i]
    print(f'Point {i+1}: x={x}, y={y}, z={z}, intensity={intensity}, ring={ring}')
print(pointcould.shape)