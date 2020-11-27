import os
import numpy as np
from ops.os_operation import mkdir
def Calculate_Mean_STD(dataset_dir):
    save_dir_info=os.path.join(dataset_dir,'STAT')
    mkdir(save_dir_info)
    mean_path=os.path.join(save_dir_info,'mean.npy')
    std_path=os.path.join(save_dir_info,'std.npy')
    if os.path.exists(mean_path) and os.path.exists(std_path):
        mean_value=np.load(mean_path)
        std_value=np.load(std_path)
        return mean_value,std_value
    listfiles=[x for x in os.listdir(dataset_dir) if "trainset" in x]
    #channel_info=[]
    all_data=[]
    for item in listfiles:
        tmp_path=os.path.join(dataset_dir,item)
        tmp_example=np.load(tmp_path)
        all_data.append(tmp_example)
    all_data=np.array(all_data)
    mean_value=[]
    std_value=[]
    for k in range(3):
        mean_value.append(np.mean(all_data[:,:,:,k]))
        std_value.append(np.std(all_data[:,:,:,k]))
    mean_value=np.array(mean_value)
    std_value=np.array(std_value)
    np.save(mean_path,mean_value)
    np.save(std_path,std_value)
    return mean_value,std_value



