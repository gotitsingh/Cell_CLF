import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import os
from ops.os_operation import mkdir
import datetime
import time
from Model.Resnet import resnet20
from Training.utils import Logger,save_checkpoint
from Data_Processing.Calculate_Mean_STD import Calculate_Mean_STD
import os
from ops.os_operation import mkdir
#from scipy import misc
from PIL import Image
import numpy as np
#from libtiff import TIFF
import pandas as pd
import random
from Data_Processing.Generate_Segemented_Image import Extract_Coord_list,Save_Segment_Image
from torch.autograd import Variable
import torch.nn.functional as F
from Segment.Unsupervise_Segment_Image import Unsupervise_Segment_Image
from Segment.Overall_Predict_Img import Overall_Predict_Img
def Segment_Predict_Img(params, input_img_path,  model_path):
    log_path = os.path.join(os.getcwd(), 'Predict_Result')
    mkdir(log_path)
    split_path = os.path.split(input_img_path)
    log_path = os.path.join(log_path, split_path[1])
    mkdir(log_path)
    model = resnet20(num_class=params['class'])

    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    # reload the model
    model_state_dict = torch.load(model_path)
    if 'ema_state_dict' in model_state_dict.keys():
        print("Load EMA model")
        model.load_state_dict(model_state_dict['ema_state_dict'])
    else:
        print("Load common model")
        model.load_state_dict(model_state_dict['state_dict'])
    # we do not use this finally, use imagenet calculated standard mean and std normalize
    mean_value = (0.59187051, 0.53104666, 0.56797799)
    std_value = (0.19646512, 0.23195337, 0.20233912)
    im = Image.open(input_img_path)
    imarray = np.array(im)
    height = params['height']
    width = params['width']
    seg_height=params['resize_height']
    seg_width=params['resize_width']
    save_path = log_path
    scan_x=int(np.ceil(imarray.shape[0]/seg_width))
    scan_y=int(np.ceil(imarray.shape[1]/seg_height))
    count_img=0
    Overall_Segment_Array=np.zeros([imarray.shape[0],imarray.shape[1]])
    for j in range(scan_x):
        for k in range(scan_y):
            start_x=j*seg_width
            start_y=k*seg_height
            end_x=(j+1)*seg_width if (j+1)*seg_width<imarray.shape[0] else imarray.shape[0]
            end_y=(k+1)*seg_height if (k+1)*seg_height<imarray.shape[1] else imarray.shape[1]
            study_array=imarray[start_x:end_x,start_y:end_y]
            #add contrast to augment
            study_array=Augment(study_array)
            Label_Segment_Array,count_img=Unsupervise_Segment_Image(study_array, save_path, count_img,params)
            Overall_Segment_Array[start_x:end_x,start_y:end_y]=Label_Segment_Array
    #predict using my model
    Overall_Predict_Img(model,height,width,Overall_Segment_Array,imarray,mean_value,std_value,save_path,params)

from Augment.Transformations import Contrast
def Augment(study_array):
    image=Image.fromarray(study_array)
    image2=Contrast(image,1.6)
    imarray = np.array(image2)
    return imarray

