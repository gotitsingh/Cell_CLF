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
from Model.Load_CPU_Model import Load_CPU_Model
def Predict_Img(params,input_img_path,location_info_path,model_path):
    """
    :param params:
    :param input_img_path:
    :param location_info_path:
    :param model_path:
    :return:
    """
    log_path=os.path.join(os.getcwd(),'Predict_Result')
    mkdir(log_path)
    split_path=os.path.split(input_img_path)
    log_path=os.path.join(log_path,split_path[1])
    mkdir(log_path)
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

    # we do not use this finally, use imagenet calculated standard mean and std normalize
    mean_value=(0.59187051,0.53104666,0.56797799)
    std_value=(0.19646512,0.23195337,0.20233912)
    # first segmented image to small image
    im = Image.open(input_img_path)
    imarray = np.array(im)
    coord_list =Extract_Coord_list(location_info_path)
    count_image=0
    height=params['height']
    width=params['width']
    save_path=log_path
    count_image=Save_Segment_Image(imarray, save_path, count_image, height, width, coord_list, 0)
    if count_image==len(coord_list):
        print("Successfully segmented image and saved!!!")
    else:
        print("Segmented part can not work, please have a check")
        return
    # then predict
    All_Predict_Img=[]
    for k in range(count_image):
        tmp_trainset_path=os.path.join(save_path,'trainset'+str(k)+'.npy')
        tmp_array=np.load(tmp_trainset_path)
        All_Predict_Img.append(tmp_array)
    All_Predict_Img=np.array(All_Predict_Img)
    from Data_Processing.Training_Dataset import SingleTestDataset
    valid_dataset = SingleTestDataset(All_Predict_Img, mean_value, std_value)
    test_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=params['batch_size'],
                                                   shuffle=False, num_workers=int(params['num_workers']),
                                                   drop_last=False, pin_memory=True)
    Label_List=[]
    Prob_List=[]
    model.eval()  # very important, fix batch normalization
    for i,inputs in enumerate(test_dataloader):
        if params['choose']!="-1":
            inputs=inputs.cuda()
            inputs = Variable(inputs, volatile=True)
        outputs, p1 = model(inputs)
        outputs = F.softmax(outputs, dim=1)
        if params['choose'] != "-1":
            outputs=outputs.cpu().detach().numpy()
        else:
            outputs=outputs.detach().numpy()
        for k in range(len(outputs)):
            tmp_pred=outputs[k]
            tmp_label=int(np.argmax(tmp_pred))
            Label_List.append(tmp_label)
            Prob_List.append(tmp_pred[tmp_label])
    # first, write a file for the predicted results
    pred_txt=os.path.join(save_path,'Predict.txt')
    with open(pred_txt,'w') as file:
        file.write('Coord0\tCoord_1\tPredict_Label\tProbability\n')
        for k in range(len(coord_list)):
            coord_info=coord_list[k]
            pred_info=Label_List[k]
            prob_info=Prob_List[k]
            file.write(str(coord_info[0])+"\t"+str(coord_info[1])+"\t"+str(pred_info)+"\t"+str(prob_info)+"\n")
    # relabel the segmented image
    for k in range(count_image):
        tmp_img_path = os.path.join(save_path, str(0) + "_" + str(k) + '.png')
        now_img_path=os.path.join(save_path,str(Label_List[k])+ "_" + str(k) + '.png')
        os.system("mv "+str(tmp_img_path)+" "+now_img_path)
    #label that on image
    Visualize_Predict_Image(imarray, save_path, height, width, coord_list,Label_List)


from PIL import Image,ImageFont,ImageDraw


def Visualize_Predict_Image(imarray, save_path, height, width, coord_list,Label_List):
    overall_width = imarray.shape[0]
    overall_height = imarray.shape[1]
    tmp_array = np.zeros([overall_width, overall_height, 3])
    coord_list = np.array(coord_list)
    max_x = np.max(coord_list[:, 0])#coord based on image
    max_y = np.max(coord_list[:, 1])
    print("1 Checking agreement of shape:%d/%d,%d/%d" % (max_y, overall_width, max_x, overall_height))
    for tmp_coord in coord_list:
        tmp_x=tmp_coord[1]
        tmp_y=tmp_coord[0]
        tmp_left=tmp_x-int(width/2)
        tmp_bottom=tmp_y-int(height/2)
        right_end = tmp_left + width if tmp_left + width < overall_width else overall_width
        upper_end = tmp_bottom + height if tmp_bottom + height < overall_height else overall_height
        # print(right_end,upper_end)
        left_start = 0 if tmp_left < 0 else tmp_left
        bottom_start = 0 if tmp_bottom < 0 else tmp_bottom
        tmp_width = int(right_end - left_start)
        tmp_height = int(upper_end - bottom_start)
        # print(tmp_width,tmp_height)
        tmp_left_start = int((width - tmp_width) / 2)
        tmp_bottom_start = int((height - tmp_height) / 2)
        tmp_array[left_start:right_end, bottom_start:upper_end, :] = \
            imarray[left_start:right_end, bottom_start:upper_end, :]
    img = Image.fromarray(tmp_array.astype(np.uint8))
    tmp_img_path = os.path.join(save_path, "Overall_Segment.png")
    img.save(tmp_img_path)
    img_origin=Image.fromarray(imarray.astype(np.uint8))
    tmp_img_path = os.path.join(save_path, "Original.png")
    img_origin.save(tmp_img_path)
    draw = ImageDraw.Draw(img)
    for k in range(len(Label_List)):
        tmp_coord=coord_list[k]
        draw.text((tmp_coord[0], tmp_coord[1]),str(Label_List[k]) , fill=(255, 255, 255))
    tmp_img_path = os.path.join(save_path, "Overall_Predict.png")
    img.save(tmp_img_path)





