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
from Model.EMA import WeightEMA
from Training.train_ema_epoch import train_ema_epoch
import numpy as np
def Generate_Resnet_Model(params,training_path,type):
    """
    :param params:
    :param training_path:
    :param type: decides the augmentation type
    :return:
    """
    #first generate log path
    log_path,result_path=Get_log_path(params)
    #build the model
    model=resnet20(num_class=params['class'])

    model=model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    if params['use_ema']:
        #ema_model=WeightEMA()
        ema_model=resnet20(num_class=params['class'])
        for param in ema_model.parameters():
            param.detach_()
        ema_model = ema_model.cuda()
        ema_model = nn.DataParallel(ema_model, device_ids=None)
        ema_optimizer=WeightEMA(model,params['lr'], ema_model, num_classes=params['class'],alpha=0.999)
    #build logger
    train_batch_logger, train_logger, val_logger=init_logger(log_path)
    #build dataloader
    #1st calculate mean and variance
    #we do not use this finally, use imagenet calculated standard mean and std normalize
    #mean_value,std_value=Calculate_Mean_STD(dataset_dir=training_path)
   # print("Mean value:",mean_value/255)
    #print("STD value:",std_value/255)
    mean_value = np.array([150.92697946, 135.41689821, 144.83438764])
    std_value = np.array([50.09860559, 59.14810949, 51.59647557])
    #predataloader
    portion=params['portion']
    train_loader,val_loader=Prepare_Dataloader(training_path,portion,mean_value/255,std_value/255,params)
    reg=params['reg']
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    parameters = model.parameters()
    optimizer = optim.Adam(parameters,
                           lr=params['lr'])
    #training
    best_acc=0
    for i in range(params['epoch']):
        if params['use_ema']:
            train_ema_epoch(i, train_loader, model, ema_model,criterion, optimizer,ema_optimizer,
                        train_logger, train_batch_logger, result_path, reg)
            validation_loss0, val_acc0 = val_epoch(i, val_loader, model, criterion,
                                                 val_logger)
            validation_loss, val_acc = val_epoch(i, val_loader, ema_model, criterion,
                                                 val_logger)

            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            save_checkpoint({
                'epoch': i + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict':ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=result_path)
        else:
            train_epoch(i, train_loader, model, criterion, optimizer,
                     train_logger, train_batch_logger, result_path, reg)
            validation_loss,val_acc = val_epoch(i, val_loader, model, criterion,
                                     val_logger)
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            save_checkpoint({
                'epoch': i + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=result_path)

def Prepare_Dataloader(training_path,portion,mean_value,std_value,params):
    from Augment.Augment_method import Augmentation
    augment_method=Augmentation()
    from Data_Processing.Training_Dataset import TrainDataset
    train_dataset=TrainDataset(training_path, portion, mean_value, std_value, augment_method, train_label=True, rand_state=params['seed'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'],
                                                  shuffle=True, num_workers=int(params['num_workers']),
                                                  drop_last=True, pin_memory=True)
    valid_dataset=TrainDataset(training_path, portion, mean_value, std_value, augment_method, train_label=False, rand_state=params['seed'])
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=params['batch_size'],
                                                  shuffle=True, num_workers=int(params['num_workers']),
                                                  drop_last=True, pin_memory=True)
    return train_dataloader,valid_dataloader
def init_logger(log_path):
    train_logger = Logger(
        os.path.join(log_path, 'train.log'),
        ['epoch', 'loss', 'regloss', 'acc', 'lr'])
    train_batch_logger = Logger(
        os.path.join(log_path, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    val_logger = Logger(
        os.path.join(log_path, 'val.log'), ['epoch', 'loss', 'acc'])
    return train_batch_logger,train_logger,val_logger
def Get_log_path(params):
    learning_rate = params['lr']
    reg = params['reg']
    type = params['type']
    log_path = os.path.join(os.getcwd(), "train_log")
    mkdir(log_path)
    log_path=os.path.join(log_path,'Type_'+str(type))
    mkdir(log_path)
    log_path=os.path.join(log_path,'lr_'+str(learning_rate)+"reg_"+str(reg))
    mkdir(log_path)
    today = datetime.date.today()
    formatted_today = today.strftime('%y%m%d')
    now = time.strftime("%H:%M:%S")
    log_path = os.path.join(log_path, formatted_today + now)
    mkdir(log_path)
    result_path = os.path.join(log_path,'model')
    mkdir(result_path)
    return log_path,result_path