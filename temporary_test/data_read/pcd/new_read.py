import numpy as np
def read_pcd_bin(pcd_file):
    with open(pcd_file, 'rb') as f:
        data = f.read()
        print(data)
        data_binary = data[data.find(b"DATA binary") + 12:]
        points = np.frombuffer(data_binary, dtype=np.float32).reshape(-1, 3)
        points = points.astype(np.float32)
    return points
xyz = read_pcd_bin("../n008-2018-08-01-15-16-36-0400__RADAR_BACK_LEFT__1533151603522238.pcd")  # (N, 3)