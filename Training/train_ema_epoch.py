import torch
from torch.autograd import Variable
import time
import os
import sys

from Training.utils import AverageMeter, calculate_accuracy

def train_ema_epoch(epoch, data_loader, model, ema_model,criterion, optimizer,ema_optimizer,
                epoch_logger, batch_logger,result_path,reg):
    print('train at epoch {}'.format(epoch))

    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    Regloss=AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs,input_t, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        #targets = targets.cuda(async=True)

        if i==0:
            print(inputs.size())
            print(targets)
        targets = targets.cuda()
        inputs=inputs.cuda()
        input_t=input_t.cuda()
        inputs = Variable(inputs)
        input_t=Variable(input_t)
        targets = Variable(targets)
        #outputs,p1 = model(inputs)
        #if i==0:
        #    print(outputs)

        l2_crit = torch.nn.MSELoss(size_average=False)#sum the reg part
        reg_loss = 0
        for param in model.parameters():
            other=torch.zeros(param.size())
            other=other.cuda()
            reg_loss += l2_crit(param,other)
        #loss = criterion(outputs, targets)
        #add transformed cross entropy
        outputs_t,p2=model(input_t)
        loss=criterion(outputs_t,targets)
        loss += reg* reg_loss
        Regloss.update(reg*reg_loss.item(),inputs.size(0))
        if i==0:
            print(loss.data)
        with torch.no_grad():
            outputs, p1 = model(inputs)
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        #if model_type>0:
        torch.nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()
        ema_optimizer.step()
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Reg_Loss {regloss.val:.4f} ({regloss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,regloss=Regloss,
                  acc=accuracies))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'regloss':Regloss.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    ema_optimizer.step(bn=True)