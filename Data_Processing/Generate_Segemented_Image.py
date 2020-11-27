import os
from ops.os_operation import mkdir
#from scipy import misc
from PIL import Image
import numpy as np
#from libtiff import TIFF
import pandas as pd
import random

def Generate_Segemented_Image_Update(dir_path,params):
    dir_path = os.path.abspath(dir_path)
    listfiles = os.listdir(dir_path)
    listfiles.sort()
    save_path = os.path.join(os.getcwd(), 'Training_Data')
    mkdir(save_path)
    record_path1 = os.path.join(save_path, 'Train_record.txt')
    record_path2 = os.path.join(save_path, 'Test_record.txt')
    save_path1 = os.path.join(save_path, 'Training')
    save_path2 = os.path.join(save_path, 'Testing')
    mkdir(save_path1)
    mkdir(save_path2)
    count_image = 0
    height = params['height']
    width = params['width']
    list_train = []
    list_test = []
    first_study_path=os.path.join(dir_path,'data1')
    listfiles2=os.listdir(first_study_path)
    if True:
        tmp_dir_path=first_study_path

        print(tmp_dir_path)

        allimage_list=[x for x in os.listdir(tmp_dir_path) if "tif" in x]
        allnega_infofile_list=[x for x in os.listdir(tmp_dir_path) if "nega" in x or "non-osteoclasts" in x]
        allposi_infofile_list=[x for x in os.listdir(tmp_dir_path) if "posi" in x or " osteoclasts" in x]
        allimage_list.sort()
        allnega_infofile_list.sort()
        allposi_infofile_list.sort()

        #CANCEL assert because some only include nega examples
        #assert len(allimage_list)==len(allnega_infofile_list) and len(allimage_list)==len(allposi_infofile_list)
        for k in range(len(allimage_list)):
            tmp_image_path=os.path.join(tmp_dir_path,allimage_list[k])
            tmp_pos_path=os.path.join(tmp_dir_path,allposi_infofile_list[k])
            tmp_nega_path=os.path.join(tmp_dir_path,allnega_infofile_list[k])

            prob=random.random()
            if prob<0.8:#for training
                count_image=Gen_Train_Image(tmp_image_path,tmp_pos_path,tmp_nega_path,save_path1,count_image,height,width)
                list_train.append(tmp_image_path)
            else:
                count_image = Gen_Train_Image(tmp_image_path, tmp_pos_path, tmp_nega_path, save_path2, count_image,
                                              height, width)
                list_test.append(tmp_image_path)
    tmp_pos_path = os.path.join(dir_path, "totalposi.csv")
    tmp_nega_path = os.path.join(dir_path, "totalnega.csv")
    posi_dict=Build_Coord_dict(tmp_pos_path)
    nega_dict=Build_Coord_dict(tmp_nega_path)
    for item in listfiles:
        if '.tif' not in item:
            continue
        tmp_image_path = os.path.join(dir_path, item)
        item=item.strip("'")
        item = item.strip('"')
        tmp_posi_list=posi_dict[item]
        tmp_nega_list=nega_dict[item]
        prob = random.random()
        if prob < 0.8:  # for training
            count_image = Gen_Train_Image_WithCoord(tmp_image_path, tmp_posi_list,tmp_nega_list, save_path1, count_image, height,width)
            list_train.append(tmp_image_path)
        else:
            count_image = Gen_Train_Image_WithCoord(tmp_image_path, tmp_posi_list,tmp_nega_list, save_path2, count_image,
                                          height, width)
            list_test.append(tmp_image_path)
    with open(record_path1,'w') as file:
        for item in list_train:
            file.write(item+'\n')
    with open(record_path2, 'w') as file:
        for item in list_test:
            file.write(item + '\n')
def Build_Coord_dict(table_path):
    #df = (pd.read_excel(table_path))
    #it not followed the actual xls format, so sad!!!
    from collections import defaultdict
    coord_dict=defaultdict(list)
    with open(table_path,'r') as file:
        line=file.readline()
        line=file.readline()
        while line:
            line=line.strip()
            if "," in line:
                split_lists=line.split(',')
            else:
                split_lists=line.split()
            if len(split_lists)<2:
                print("Incorrect format!!%s, %s"%(table_path,line))
                line=file.readline()
                continue
            img_name=split_lists[0]
            tmp_coord=[]
            tmp_coord.append(int(float(split_lists[-2])))
            tmp_coord.append(int(float(split_lists[-1])))
            coord_dict[img_name].append(tmp_coord)
            line=file.readline()
    return coord_dict#for image based ,not array

def Generate_Segemented_Image(dir_path,params):
    dir_path=os.path.abspath(dir_path)
    listfiles=os.listdir(dir_path)
    listfiles.sort()
    save_path=os.path.join(os.getcwd(),'Training_Data')
    mkdir(save_path)
    #save_path = os.path.join(save_path, 'Extract_Image')
    #mkdir(save_path)
    record_path1=os.path.join(save_path,'Train_record.txt')
    record_path2 = os.path.join(save_path, 'Test_record.txt')
    save_path1=os.path.join(save_path, 'Training')
    save_path2=os.path.join(save_path, 'Testing')
    mkdir(save_path1)
    mkdir(save_path2)
    count_image=0
    height=params['height']
    width=params['width']
    list_train=[]
    list_test=[]
    for item in listfiles:
        tmp_dir_path=os.path.join(dir_path,item)
        if not os.path.isdir(tmp_dir_path):
            continue
        print(tmp_dir_path)

        allimage_list=[x for x in os.listdir(tmp_dir_path) if "tif" in x]
        if item!="new nega and posi cells":
            allnega_infofile_list=[x for x in os.listdir(tmp_dir_path) if "nega" in x or "non-osteoclasts" in x]
            allposi_infofile_list=[x for x in os.listdir(tmp_dir_path) if "posi" in x or " osteoclasts" in x]
            allimage_list.sort()
            allnega_infofile_list.sort()
            allposi_infofile_list.sort()

        #CANCEL assert because some only include nega examples
        #assert len(allimage_list)==len(allnega_infofile_list) and len(allimage_list)==len(allposi_infofile_list)
        for k in range(len(allimage_list)):
            tmp_image_path=os.path.join(tmp_dir_path,allimage_list[k])
            if item != "new nega and posi cells":
                tmp_pos_path=os.path.join(tmp_dir_path,allposi_infofile_list[k])
                tmp_nega_path=os.path.join(tmp_dir_path,allnega_infofile_list[k])
            else:
                tmp_item=allimage_list[k]
                tmp_pos_path = os.path.join(tmp_dir_path, tmp_item[:-4]+" new posi.csv")
                tmp_nega_path=os.path.join(tmp_dir_path,tmp_item[:-4]+" new nega.csv")
            prob=random.random()
            if prob<0.8:#for training
                count_image=Gen_Train_Image(tmp_image_path,tmp_pos_path,tmp_nega_path,save_path1,count_image,height,width)
                list_train.append(tmp_image_path)
            else:
                count_image = Gen_Train_Image(tmp_image_path, tmp_pos_path, tmp_nega_path, save_path2, count_image,
                                              height, width)
                list_test.append(tmp_image_path)
    with open(record_path1,'w') as file:
        for item in list_train:
            file.write(item+'\n')
    with open(record_path2, 'w') as file:
        for item in list_test:
            file.write(item + '\n')
        #count_image=Gen_Train_Image(dir_path,save_path,allimage_list,allnega_infofile_list,allposi_infofile_list,count_image)
def Extract_Coord_list(table_path):
    #df = (pd.read_excel(table_path))
    #it not followed the actual xls format, so sad!!!
    coord_list=[]
    with open(table_path,'r') as file:
        line=file.readline()
        line=file.readline()
        while line:
            line=line.strip()
            if "," in line:
                split_lists=line.split(',')
            else:
                split_lists=line.split()
            if len(split_lists)<2:
                print("Incorrect format!!%s, %s"%(table_path,line))
                line=file.readline()
                continue
            tmp_coord=[]
            tmp_coord.append(int(float(split_lists[-2])))
            tmp_coord.append(int(float(split_lists[-1])))
            coord_list.append(tmp_coord)
            line=file.readline()
    return coord_list#for image based ,not array

def Save_Segment_Image(imarray,save_path,count_image,height,width,pos_coord_list,label):
    overall_width=imarray.shape[0]
    overall_height=imarray.shape[1]
    #print(overall_width,overall_height)
    pos_coord_list=np.array(pos_coord_list)
    max_x=np.max(pos_coord_list[:,0])
    max_y = np.max(pos_coord_list[:, 1])
    print("Checking agreement of shape:%d/%d,%d/%d"%(max_y,overall_width,max_x,overall_height))
    for tmp_coord in pos_coord_list:
        tmp_x=tmp_coord[1]
        tmp_y=tmp_coord[0]#must switch to make sure matched with the correct postion in array
        ##in this label x is y, y is x

        #print(tmp_x,tmp_y)
        tmp_left=tmp_x-int(width/2)
        tmp_bottom=tmp_y-int(height/2)
        #print(tmp_left,tmp_bottom)
        tmp_array=np.zeros([width,height,3])
        right_end=tmp_left+width if tmp_left+width<overall_width else overall_width
        upper_end=tmp_bottom+height if tmp_bottom+height<overall_height else overall_height
        #print(right_end,upper_end)
        left_start=0 if tmp_left<0 else tmp_left
        bottom_start=0 if tmp_bottom<0 else tmp_bottom
        tmp_width=int(right_end-left_start)
        tmp_height=int(upper_end-bottom_start)
        #print(tmp_width,tmp_height)
        tmp_left_start=int((width-tmp_width)/2)
        tmp_bottom_start=int((height-tmp_height)/2)
        tmp_array[tmp_left_start:tmp_left_start+tmp_width,tmp_bottom_start:tmp_height+tmp_bottom_start,:]=\
            imarray[left_start:right_end,bottom_start:upper_end,:]#set to center for new image
        #tmp_array=imarray[tmp_left:tmp_left+width,tmp_bottom:tmp_bottom+height,:]
        tmp_train_path=os.path.join(save_path,'trainset'+str(count_image)+'.npy')
        tmp_aim_path=os.path.join(save_path,'aimset'+str(count_image)+'.npy')
        np.save(tmp_train_path,tmp_array)
        np.save(tmp_aim_path, np.array(label))
        img=Image.fromarray(tmp_array.astype(np.uint8))
        tmp_img_path=os.path.join(save_path,str(label)+"_"+str(count_image)+'.png')
        img.save(tmp_img_path)
        count_image+=1
    return count_image

def Visualize_Segment_Image(imarray,save_path,count_image,height,width,pos_coord_list,label):
    overall_width=imarray.shape[0]
    overall_height=imarray.shape[1]
    print(overall_width,overall_height)
    for tmp_coord in pos_coord_list:
        tmp_x=tmp_coord[1]
        tmp_y=tmp_coord[0]
        ##in this label x is y, y is x

        #print(tmp_x,tmp_y)
        tmp_left=tmp_x-int(width/2)
        tmp_bottom=tmp_y-int(height/2)
        #print(tmp_left,tmp_bottom)
        tmp_array=np.zeros([overall_width,overall_height,3])
        right_end=tmp_left+width if tmp_left+width<overall_width else overall_width
        upper_end=tmp_bottom+height if tmp_bottom+height<overall_height else overall_height
        #print(right_end,upper_end)
        left_start=0 if tmp_left<0 else tmp_left
        bottom_start=0 if tmp_bottom<0 else tmp_bottom
        tmp_width=int(right_end-left_start)
        tmp_height=int(upper_end-bottom_start)
        #print(tmp_width,tmp_height)
        tmp_left_start=int((width-tmp_width)/2)
        tmp_bottom_start=int((height-tmp_height)/2)
        tmp_array[left_start:right_end,bottom_start:upper_end,:]=\
            imarray[left_start:right_end,bottom_start:upper_end,:]#set to center for new image
        #tmp_array=imarray[tmp_left:tmp_left+width,tmp_bottom:tmp_bottom+height,:]
        img=Image.fromarray(tmp_array.astype(np.uint8))
        tmp_img_path=os.path.join(save_path,"Overall_"+str(label)+"_"+str(count_image)+'.png')
        img.save(tmp_img_path)
        count_image+=1
    return count_image

def Gen_Train_Image(tmp_image_path,tmp_pos_path,tmp_nega_path,save_path,count_image,height,width):
    im = Image.open(tmp_image_path)
    imarray = np.array(im)
    #then read the excel to get coordinates
    if os.path.exists(tmp_pos_path):
        pos_coord_list=Extract_Coord_list(tmp_pos_path)
        count_image=Save_Segment_Image(imarray,save_path,count_image,height,width,pos_coord_list,1)
    if os.path.exists(tmp_nega_path):
        nega_coord_list = Extract_Coord_list(tmp_nega_path)
        count_image = Save_Segment_Image(imarray, save_path, count_image, height, width, nega_coord_list, 0)
    return count_image
def Gen_Train_Image_WithCoord(tmp_image_path,pos_coord_list,nega_coord_list,save_path,count_image,height,width):
    im = Image.open(tmp_image_path)
    imarray = np.array(im)
    #then read the excel to get coordinates
    count_image=Save_Segment_Image(imarray,save_path,count_image,height,width,pos_coord_list,1)
    count_image = Save_Segment_Image(imarray, save_path, count_image, height, width, nega_coord_list, 0)
    return count_image


def Visualize_Train_Image(tmp_image_path,tmp_pos_path,tmp_nega_path,save_path,count_image,height,width):
    im = Image.open(tmp_image_path)
    imarray = np.array(im)
    # then read the excel to get coordinates
    pos_coord_list = Extract_Coord_list(tmp_pos_path)
    count_image = Visualize_Segment_Image(imarray, save_path, count_image, height, width, pos_coord_list, 1)
    nega_coord_list = Extract_Coord_list(tmp_nega_path)
    count_image =Visualize_Segment_Image(imarray, save_path, count_image, height, width, nega_coord_list, 0)
