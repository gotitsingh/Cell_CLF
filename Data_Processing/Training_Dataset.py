from torch.utils.data.dataset import Dataset
import os
import numpy as np
from sklearn.model_selection import train_test_split
import random
from PIL import Image, PILLOW_VERSION, ImageEnhance
import torch
class TrainDataset(Dataset):
    def __init__(self,dataset_dir,portion,mean,std,augment_method=None,train_label=True,rand_state=888):
        super(TrainDataset, self).__init__()
        #self.trainsetFile = []
        #self.aimsetFile = []
        listfiles = os.listdir(dataset_dir)
        trainlist = [os.path.join(dataset_dir, x) for x in listfiles if "trainset" in x]
        aimlist = [os.path.join(dataset_dir, x) for x in listfiles if "aimset" in x]
        trainlist.sort()
        aimlist.sort()
        X_train, X_test, y_train, y_test = train_test_split(trainlist, aimlist, test_size=1-portion,
                                                            random_state=rand_state)
        if train_label:
            self.trainsetFile = X_train
            self.aimsetFile = y_train
        else:
            self.trainsetFile = X_test
            self.aimsetFile = y_test
        self.mean=mean
        self.std=std
        self.augment_method=augment_method
    def normalise(self,x, mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616) ):
        x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
        x -= mean * 255
        x *= 1.0 / (255 * std)
        return x
    def denormalize(self,x, mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)):
        x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
        x=x*(255*std)
        x+=mean*255
        return x
    def __getitem__(self, index):
        train_path = self.trainsetFile[index]
        aim_path = self.aimsetFile[index]
        img1 = np.load(train_path)
        target=np.load(aim_path)
        img1 = Image.fromarray(img1.astype(np.uint8))
        img1_t=None
        if self.augment_method is not None:
            img1_t=self.augment_method(img1)
            img1_t = np.array(img1_t).astype('float32')
            img1_t = self.normalise(img1_t)
            img1_t = img1_t.transpose((2, 0, 1))
            img1_t = torch.from_numpy(img1_t)
        img1 = np.array(img1).astype('float32')
        img1 = self.normalise(img1)
        img1 = img1.transpose((2, 0, 1))
        img1 = torch.from_numpy(img1)
        return img1,img1_t,target


    def __len__(self):
        return len(self.aimsetFile)

class TestDataset(Dataset):
    def __init__(self,dataset_dir,mean,std):
        super(TestDataset, self).__init__()
        #self.trainsetFile = []
        #self.aimsetFile = []
        listfiles = os.listdir(dataset_dir)
        trainlist = [os.path.join(dataset_dir, x) for x in listfiles if "trainset" in x]
        aimlist = [os.path.join(dataset_dir, x) for x in listfiles if "aimset" in x]
        trainlist.sort()
        aimlist.sort()
        self.mean=mean
        self.std=std
        self.trainsetFile=trainlist
        self.aimsetFile=aimlist

    def normalise(self,x, mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616) ):
        x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
        x -= mean * 255
        x *= 1.0 / (255 * std)
        return x
    def denormalize(self,x, mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)):
        x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
        x=x*(255*std)
        x+=mean*255
        return x
    def __getitem__(self, index):
        train_path = self.trainsetFile[index]
        aim_path = self.aimsetFile[index]
        img1 = np.load(train_path)
        target=np.load(aim_path)
        img1 = Image.fromarray(img1.astype(np.uint8))
        img1_t=0
        img1 = np.array(img1).astype('float32')
        img1 = self.normalise(img1)
        img1 = img1.transpose((2, 0, 1))
        img1 = torch.from_numpy(img1)
        return img1,img1_t,target


    def __len__(self):
        return len(self.aimsetFile)

class SingleTestDataset(Dataset):

    def __init__(self, trainsetfile, mean, std):
        super(SingleTestDataset, self).__init__()
        # self.trainsetFile = []
        # self.aimsetFile = []

        self.mean = mean
        self.std = std
        self.trainsetFile = trainsetfile


    def normalise(self, x, mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)):
        x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
        x -= mean * 255
        x *= 1.0 / (255 * std)
        return x

    def denormalize(self, x, mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)):
        x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
        x = x * (255 * std)
        x += mean * 255
        return x

    def __getitem__(self, index):
        train_path = self.trainsetFile[index]
        img1 = train_path
        img1 = Image.fromarray(img1.astype(np.uint8))
        img1 = np.array(img1).astype('float32')
        img1 = self.normalise(img1)
        img1 = img1.transpose((2, 0, 1))
        img1 = torch.from_numpy(img1)
        return img1
    def __len__(self):
        return len(self.trainsetFile)


