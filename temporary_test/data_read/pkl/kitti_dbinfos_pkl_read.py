import pickle
import pprint

file_path = './kitti_dbinfos_train.pkl'
with open(file_path,'rb') as f:
    data = pickle.load(f)
pprint.pprint(data['Car'][0])
print(len(data['Car']))
for i in range(2):
    pprint.pprint(data['Car'][i])
for key in data.keys():
    pprint.pprint(key)
