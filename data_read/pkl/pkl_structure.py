import pickle
import pprint

file_path = './kitti_dbinfos_train.pkl'
with open(file_path,'rb') as f:
    data = pickle.load(f)

def display_dict_structure(data, indent=0):
    for key, value in data.items():
        print(f"{' ' * indent}{key}:")
        if isinstance(value, dict):
            display_dict_structure(value, indent + 2)
        else:
            print(f"{' ' * (indent + 2)}{type(value)}")


# 显示字典结构
display_dict_structure(data)
