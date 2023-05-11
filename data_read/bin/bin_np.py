import numpy as np

np.set_printoptions(suppress=True,threshold=np.inf)
file_path = '../../data/test_minikitti/train/velodyne/000001.bin'
pointcould = np.fromfile(file_path, dtype=np.float32).reshape([-1,4])
print(pointcould[0:3])
print(pointcould.shape)