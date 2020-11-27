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
from Training.train_epoch import train_epoch
from Training.Val_epoch import val_epoch
import numpy as np
def Evaluate_Model(params,testing_path,model_path,type):
    """
    :param params:
    :param training_path:
    :param type: decides the augmentation type
    :return:
    """
    #first generate log path
    log_path=Get_log_path(params)
    #build the model
    model=resnet20(num_class=params['class'])

    model=model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    #reload the model
    model_state_dict=torch.load(model_path)
    if params['use_ema']:
        model.load_state_dict(model_state_dict['ema_state_dict'])
    else:
        model.load_state_dict(model_state_dict['state_dict'])
    #build logger
    val_logger=init_logger(log_path)
    #build dataloader
    #1st calculate mean and variance
    #mean_value,std_value=Calculate_Mean_STD(dataset_dir=training_path)#use this to get the previous mean and std
    #print("Mean value:",mean_value/255)
    #print("STD value:",std_value/255)
    # we do not use this finally, use imagenet calculated standard mean and std normalize
    mean_value = np.array([150.92697946, 135.41689821, 144.83438764])
    std_value = np.array([50.09860559, 59.14810949, 51.59647557])

    print("Mean value:",mean_value/255)
    print("STD value:",std_value/255)
    #predataloader
    #portion=params['portion']
    val_loader=Prepare_Dataloader(testing_path,mean_value/255,std_value/255,params)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    #evaluating
    best_acc=0
    #for i in range(params['epoch']):
    validation_loss,val_acc = val_epoch(0, val_loader, model, criterion,
                                     val_logger,collect_wrong=True)
    print("Our testing accuracy: %.5f"%val_acc)



def Prepare_Dataloader(training_path,mean_value,std_value,params):
    from Data_Processing.Training_Dataset import TestDataset
    valid_dataset=TestDataset(training_path,mean_value, std_value)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=params['batch_size'],
                                                  shuffle=True, num_workers=int(params['num_workers']),
                                                  drop_last=True, pin_memory=True)
    return valid_dataloader
def init_logger(log_path):

    val_logger = Logger(
        os.path.join(log_path, 'val.log'), ['epoch', 'loss', 'acc'])
    return val_logger
def Get_log_path(params):
    type = params['type']
    log_path = os.path.join(os.getcwd(), "test_log")
    mkdir(log_path)
    log_path=os.path.join(log_path,'Type_'+str(type))
    mkdir(log_path)
    today = datetime.date.today()
    formatted_today = today.strftime('%y%m%d')
    now = time.strftime("%H:%M:%S")
    log_path = os.path.join(log_path, formatted_today + now)
    mkdir(log_path)
    return log_path