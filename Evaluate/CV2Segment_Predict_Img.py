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
import cv2 as cv
from matplotlib import pyplot as plt
from Segment.CV2_overall_predict import CV2_overall_predict
from Model.Load_CPU_Model import Load_CPU_Model
import shutil
def CV2Segment_Predict_Img(params, input_img_path,  model_path):
    #if resize is required


    log_path = os.path.join(os.getcwd(), 'Predict_Result')
    mkdir(log_path)
    split_path = os.path.split(input_img_path)
    origin_img_name=split_path[1][:-4]
    log_path = os.path.join(log_path, split_path[1])
    mkdir(log_path)
    log_path=os.path.join(log_path,"Filter_"+str(params['filter_size']))
    mkdir(log_path)
    origin_img_name+="Filter_"+str(params['filter_size'])
    log_path = os.path.join(log_path, "threshold_" + str(params['threshold']))
    mkdir(log_path)
    origin_img_name += "threshold_" + str(params['threshold'])
    log_path = os.path.join(log_path, "Removepixel_" + str(params['remove_pixel']))
    mkdir(log_path)
    origin_img_name+="Removepixel_" + str(params['remove_pixel'])
    model = resnet20(num_class=params['class'])
    if params['choose']!="-1":
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
    # reload the model
    model_state_dict = torch.load(model_path, map_location='cpu')
    if 'ema_state_dict' in model_state_dict.keys():
        print("Load EMA model")
        if params['choose'] != "-1":
            model.load_state_dict(model_state_dict['ema_state_dict'])
        else:
            model=Load_CPU_Model(model_state_dict['ema_state_dict'],model)
    else:
        print("Load common model")
        if params['choose'] != "-1":
            model.load_state_dict(model_state_dict['state_dict'])
        else:
            model = Load_CPU_Model(model_state_dict['state_dict'], model)

    save_path = log_path
    if params['resize']:
        print("We are doing resizing here")
        img2=Image.open(input_img_path)
        img2=img2.resize([params['resize_width'],params['resize_height']])
        input_img_path_resize=os.path.join(save_path,'resize_img.png')
        img2.save(input_img_path_resize)
        input_img_path=input_img_path_resize
    Markers= CV2Segment_Image(input_img_path, save_path, params)
    mean_value = (0.59187051, 0.53104666, 0.56797799)
    std_value = (0.19646512, 0.23195337, 0.20233912)
    original_path = os.path.join(save_path, 'Original.png')
    im = Image.open(original_path)
    imarray = np.array(im.getdata(),dtype=np.uint8)
    height = params['height']
    width = params['width']
    print("Markers shape", Markers.shape)  # same as image shape
    print("Image shape", im.size)
    CV2_overall_predict(model, height, width, Markers, imarray, mean_value, std_value, save_path, params,origin_img_name)

def CV2Segment_Image(input_img_path, save_path, params):
    #https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html
    original_path=os.path.join(save_path,'Original.png')
    #os.system("cp "+input_img_path+" "+original_path)
    shutil.copy(input_img_path,original_path)
    img = cv.imread(input_img_path)
    if img is None:
        print("READING IMAGE FAILED!!")
        exit()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if params['type']==0:
        ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    else:
        ret, thresh = cv.threshold(gray, params['threshold'], 255, cv.THRESH_BINARY_INV)
    # noise removal
    filter_size=params['filter_size']
    kernel = np.ones((filter_size, filter_size), np.uint8)
    #detailed instructions in https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 3)
    ret, sure_fg = cv.threshold(dist_transform, 3, 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    img1=img.copy()
    markers = cv.watershed(img1, markers)
    #check the markers id
    max_id=np.max(markers)
    if params['remove_pixel']!=0:
        remove_threshold=params['remove_pixel']

        for k in range(1,max_id+1):
            remove_index=np.argwhere(markers == k)
            area_size=len(remove_index)
            if area_size<remove_threshold:
                markers[remove_index[:,0],remove_index[:,1]]=-3#marked as not visiable

    img1[markers == -1] = [255, 0, 0]
    #save the image with watershed
    tmp_image=Image.fromarray(img1)
    extracted_path=os.path.join(save_path,"Filtered_watershed.png")
    tmp_image.save(extracted_path)
    #colored markers
    markers_plot=np.array(markers,dtype=np.uint8)
    heat_map=cv.applyColorMap(markers_plot,cv.COLORMAP_JET)
    plt.imshow(heat_map,alpha=0.5)
    markers_path=os.path.join(save_path,"Markers_visualization.png")
    plt.savefig(markers_path)
    #show the extracted plots, block other parts
    remained_image=np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(markers[i,j]>1):
                remained_image[i,j,:]=img[i,j,:]
    remained_image=np.array(remained_image,dtype=np.uint8)
    imgshow=Image.fromarray(remained_image)
    extracted_path=os.path.join(save_path,'Extracted_area.png')
    imgshow.save(extracted_path)
    return markers