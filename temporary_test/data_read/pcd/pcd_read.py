import time
import struct
import numpy as np

def read_bin(pcd_file):
    list = []
    with open(pcd_file,'rb') as file:
        data = file.read()
        pc_iter = struct.iter_unpack('ffff',data)
        for i,point in enumerate(pc_iter):
            list.append(point)
    return np.array(list)

a = read_bin('../1686039063.144342823.pcd')
print(a)