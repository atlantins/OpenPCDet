import pickle
import pprint

path = './kitti_infos_train.pkl'  # pkl文件所在路径
f = open(path, 'rb')
data = pickle.load(f, encoding="utf8")
pprint.pprint(data)
print(len(data))
f.close()