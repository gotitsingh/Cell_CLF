
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from torch import nn
from torch import optim
from Data_Processing.Generate_Segemented_Image import Save_Segment_Image
import numpy as np
import os
from Evaluate.Predict_Img import Visualize_Predict_Image
from PIL import Image,ImageFont,ImageDraw
import shutil
def CV2_overall_predict(model,height,width,
                        Overall_Segment_Array,imarray,mean_value,std_value,save_path,params,added_name):
    coord_list=Build_Coord_List(Overall_Segment_Array)#this coord for imarray not for image

    #write a coordinate list to somewhere
    new_coord_list = []
    for coord in coord_list:
        new_coord_list.append([coord[1], coord[0]])
    coord_list = new_coord_list
    coord_list=np.array(coord_list)#coord now is based on image, instead of array
    tmp_coord_path=os.path.join(save_path,'Coord_Info.txt')
    np.savetxt(tmp_coord_path,coord_list)
    print("DEBUG INFO: im array type",type(imarray))
    #draw a picture for coordinates
    tmp_coord_figure_path=os.path.join(save_path,"Coord_Info.png")
    Draw_Coord_Figure(tmp_coord_figure_path,coord_list,imarray)
    count_image1=0
    count_image1 = Save_Segment_Image(imarray, save_path, count_image1, height, width, coord_list, 0)
    if count_image1==len(coord_list):
        print("Successfully segmented image and saved!!!")
    else:
        print("Segmented part can not work, please have a check")
        return
    All_Predict_Img = []
    for k in range(count_image1):
        tmp_trainset_path = os.path.join(save_path, 'trainset' + str(k) + '.npy')
        tmp_array = np.load(tmp_trainset_path)
        All_Predict_Img.append(tmp_array)

    All_Predict_Img = np.array(All_Predict_Img)
    from Data_Processing.Training_Dataset import SingleTestDataset
    valid_dataset = SingleTestDataset(All_Predict_Img, mean_value, std_value)
    test_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=params['batch_size'],
                                                  shuffle=False, num_workers=int(params['num_workers']),
                                                  drop_last=False, pin_memory=True)
    Label_List = []
    Prob_List = []
    model.eval()  # very important, fix batch normalization
    for i, inputs in enumerate(test_dataloader):
        if params['choose'] != "-1":
            inputs = inputs.cuda()
            inputs = Variable(inputs, volatile=True)
        outputs, p1 = model(inputs)
        outputs = F.softmax(outputs, dim=1)
        if params['choose'] != "-1":
            outputs = outputs.cpu().detach().numpy()
        else:
            outputs=outputs.detach().numpy()
        for k in range(len(outputs)):
            tmp_pred = outputs[k]
            tmp_label = int(np.argmax(tmp_pred))
            Label_List.append(tmp_label)
            Prob_List.append(tmp_pred[tmp_label])
    # first, write a file for the predicted results
    #based on image
    pred_txt = os.path.join(save_path, 'Predict_'+str(added_name)+'.txt')
    count_positive=0
    with open(pred_txt, 'w') as file:
        file.write('Coord0\tCoord_1\tPredict_Label\tProbability\n')
        for k in range(len(coord_list)):
            coord_info = coord_list[k]
            pred_info = Label_List[k]
            if pred_info==1:
                count_positive+=1
            prob_info = Prob_List[k]
            file.write(
                str(coord_info[0]) + "\t" + str(coord_info[1]) + "\t" + str(pred_info) + "\t" + str(prob_info) + "\n")

    pred_txt = os.path.join(save_path, 'Predict_' + str(added_name) + '_pcount.txt')
    with open(pred_txt, 'w') as file:
        file.write('Total Positive:%d\n'%count_positive)

    for k in range(count_image1):
        tmp_img_path = os.path.join(save_path, str(0) + "_" + str(k) + '.png')
        now_img_path=os.path.join(save_path,str(Label_List[k])+ "_" + str(k) + '.png')
        if now_img_path==tmp_img_path:
            continue
        if os.path.exists(now_img_path) and now_img_path!=tmp_img_path:
            os.remove(now_img_path)
        #os.system("mv "+str(tmp_img_path)+" "+now_img_path)
        shutil.move(tmp_img_path,now_img_path)
    #label that on image
    Visualize_Predict_Image(imarray, save_path, height, width, coord_list,Label_List)
    Visualize_Detail_Predict_Image(imarray, save_path, height, width, coord_list,Label_List,Overall_Segment_Array)

def Visualize_Detail_Predict_Image(imarray, save_path, height, width, coord_list,Label_List,Overall_Segment_Array):
    overall_width = imarray.shape[0]
    overall_height = imarray.shape[1]
    tmp_array = np.zeros([overall_width, overall_height, 3])
    for j in range(overall_width):
        for k in range(overall_height):
            if Overall_Segment_Array[j,k]<=1:
                continue
            tmp_array[j,k]=imarray[j,k]
    img = Image.fromarray(tmp_array.astype(np.uint8))
    tmp_img_path = os.path.join(save_path, "Detail_Segment.png")
    img.save(tmp_img_path)
    draw = ImageDraw.Draw(img)
    for k in range(len(Label_List)):
        tmp_coord = coord_list[k]
        draw.text((tmp_coord[0], tmp_coord[1]), str(Label_List[k]), fill=(255, 255, 255))
    tmp_img_path = os.path.join(save_path, "Detailed_Overall_Predict.png")
    img.save(tmp_img_path)

    tmp_img_path = os.path.join(save_path, "Detailed_Segment_Predict.png")
    #img = Image.fromarray(tmp_array.astype(np.uint8))
    #img[Overall_Segment_Array == -1] = [255, 0, 0]
    tmp_array = np.zeros([overall_width, overall_height, 3])
    for j in range(overall_width):
        for k in range(overall_height):
            if Overall_Segment_Array[j, k] ==- 1:
                tmp_array[j, k]=[255,0,0]
            else:
                tmp_array[j, k] = imarray[j, k]
    img = Image.fromarray(tmp_array.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for k in range(len(Label_List)):
        tmp_coord = coord_list[k]
        draw.text((tmp_coord[0], tmp_coord[1]), str(Label_List[k]), fill=(255, 255, 255))
    img.save(tmp_img_path)

    tmp_img_path = os.path.join(save_path, "Detailed_Segment_Predict2.png")
    # img = Image.fromarray(tmp_array.astype(np.uint8))
    # img[Overall_Segment_Array == -1] = [255, 0, 0]
    tmp_array = np.zeros([overall_width, overall_height, 3])
    for j in range(overall_width):
        for k in range(overall_height):
            if Overall_Segment_Array[j, k] == - 1 or Overall_Segment_Array[j, k] == - 3:
                tmp_array[j, k] = [255, 0, 0]
            else:
                tmp_array[j, k] = imarray[j, k]
    img = Image.fromarray(tmp_array.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for k in range(len(Label_List)):
        tmp_coord = coord_list[k]
        draw.text((tmp_coord[0], tmp_coord[1]), str(Label_List[k]), fill=(255, 255, 255))
    img.save(tmp_img_path)

def Draw_Coord_Figure(tmp_coord_figure_path,coord_list,imarray):
    #print(imarray)
    #print(imarray.shape)
    modify_imarray=np.array(imarray,dtype=np.uint8)
    img=Image.fromarray(modify_imarray)
    #img = Image.fromarray(imarray.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for k in range(len(coord_list)):
        tmp_coord = coord_list[k]
        draw.text((tmp_coord[0], tmp_coord[1]), "C", fill=(255, 255, 255))
    img.save(tmp_coord_figure_path)


def Build_Coord_List(Overall_Segment_Array):
    Final_Coord=[]
    label_list=np.unique(Overall_Segment_Array)
    print("In total, we have %d segmented areas waiting to be predicted"%(len(label_list)-3))
    for tmp_label in label_list:
        if tmp_label<=1:
            continue
        Coord_List=np.argwhere(Overall_Segment_Array==tmp_label)
        X_list=[]
        Y_list=[]
        for tmp_coord in Coord_List:
            X_list.append(tmp_coord[0])
            Y_list.append(tmp_coord[1])
        X_list=np.array(X_list)
        Y_list=np.array(Y_list)
        if len(X_list)==0:
            continue
        X_mean=int(np.mean(X_list))
        Y_mean=int(np.mean(Y_list))
        Final_Coord.append([X_mean,Y_mean])
    return Final_Coord
