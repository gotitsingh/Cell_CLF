import torch
from torch.autograd import Variable
import time

from Training.utils import AverageMeter, calculate_accuracy
from collections import defaultdict
import numpy as np
import os
from ops.os_operation import mkdir
from PIL import Image
def val_epoch(epoch, data_loader, model, criterion, logger,collect_wrong=False):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    if collect_wrong:
        All_Wrong_Collection=defaultdict(list)

    for i, (inputs,_, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        #targets = targets.cuda(async=True)
        targets = targets.cuda()
        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)
        outputs,p1 = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)
        batch_size=inputs.size(0)
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        if collect_wrong:
            inputs=inputs.cpu().detach().numpy()
            targets=targets.cpu().detach().numpy()
            outputs=outputs.cpu().detach().numpy()
            for k in range(batch_size):
                tmp_pred=outputs[k]
                tmp_pred_label=int(np.argmax(tmp_pred))
                tmp_target=int(targets[k])
                if tmp_pred_label!=tmp_target:
                    All_Wrong_Collection[tmp_target].append(inputs[k])
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))

    logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
    if collect_wrong:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        tmp_path=os.path.join(os.getcwd(),'Eval_Wrong_Example')
        mkdir(tmp_path)
        count_image=0
        for key in All_Wrong_Collection.keys():
            tmp_list=All_Wrong_Collection[key]
            for tmp_item in tmp_list:
                tmp_img_path=os.path.join(tmp_path,str(key)+"_"+str(count_image)+'.png')
                tmp_item = tmp_item.transpose((1, 2, 0))
                tmp_item, mean, std = [np.array(a, np.float32) for a in (tmp_item, mean, std)]
                tmp_item = tmp_item * (255 * std)
                tmp_item += mean * 255
                img = Image.fromarray(tmp_item.astype(np.uint8))
                img.save(tmp_img_path)
                count_image+=1
    return losses.avg,accuracies.avg