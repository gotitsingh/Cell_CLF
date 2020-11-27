
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
from Segment.CV2_overall_predict import Draw_Coord_Figure
def Predict_img_Coord(model,height,width,coord_list,imarray,mean_value,std_value,save_path,params):
    #write a coordinate list to somewhere
    coord_list=np.array(coord_list)#coordinates based on image
    tmp_coord_path=os.path.join(save_path,'Coord_Info.txt')
    np.savetxt(tmp_coord_path,coord_list)
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
        inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)
        outputs, p1 = model(inputs)
        outputs = F.softmax(outputs, dim=1)
        outputs = outputs.cpu().detach().numpy()
        for k in range(len(outputs)):
            tmp_pred = outputs[k]
            tmp_label = int(np.argmax(tmp_pred))
            Label_List.append(tmp_label)
            Prob_List.append(tmp_pred[tmp_label])
    # first, write a file for the predicted results
    pred_txt = os.path.join(save_path, 'Predict.txt')
    with open(pred_txt, 'w') as file:
        file.write('Coord0\tCoord_1\tPredict_Label\tProbability\n')
        for k in range(len(coord_list)):
            coord_info = coord_list[k]#based on image coord, instead of array
            pred_info = Label_List[k]
            prob_info = Prob_List[k]
            file.write(
                str(coord_info[0]) + "\t" + str(coord_info[1]) + "\t" + str(pred_info) + "\t" + str(prob_info) + "\n")
    for k in range(count_image1):
        tmp_img_path = os.path.join(save_path, str(0) + "_" + str(k) + '.png')
        now_img_path=os.path.join(save_path,str(Label_List[k])+ "_" + str(k) + '.png')
        if now_img_path==tmp_img_path:
            continue
        if os.path.exists(now_img_path) and now_img_path!=tmp_img_path:
            os.remove(now_img_path)
        os.system("mv "+str(tmp_img_path)+" "+now_img_path)
    #label that on image
    Visualize_Predict_Image(imarray, save_path, height, width, coord_list,Label_List)