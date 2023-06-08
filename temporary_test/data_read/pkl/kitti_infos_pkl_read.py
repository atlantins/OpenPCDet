import pickle
import pprint

file_path = './kitti_infos_train.pkl'
with open(file_path,'rb') as f:
    data = pickle.load(f)
# print(data[0])
pprint.pprint(data[0:2])
pprint.pprint('================================================================')
pprint.pprint(data[0])
pprint.pprint('================================================================')
pprint.pprint(data[0]['annos'])
# pprint.pprint(data[0]['annos'])
# for item in data.items():
#     print(item)
# for key in data.keys():
#     print(key)
# for value in data.values():
#     print(value)