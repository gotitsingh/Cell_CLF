import numpy as np
import torch
from torch.autograd import Variable
from skimage import segmentation
from Segment.MyNet import MyNet
import torch.optim as optim
from PIL import Image
import os
from Segment.Find_Segment_Area import Find_Segment_Area
from Segment.Locate_Cell_Area import Locate_Cell_Area
def Unsupervise_Segment_Image(imarray, save_path, count_img,params):
    data = torch.from_numpy(np.array([imarray.transpose((2, 0, 1)).astype('float32') / 255.]))
    data = data.cuda()
    data = Variable(data)
    labels = segmentation.slic(imarray, compactness=params['compactness'], n_segments=params['num_superpixels'])
    labels = labels.reshape(imarray.shape[0] * imarray.shape[1])
    u_labels = np.unique(labels)
    l_inds = []
    for i in range(len(u_labels)):
        l_inds.append(np.where(labels == u_labels[i])[0])
    model = MyNet(data.size(1),params['nChannel'],params['nConv'])
    model.cuda()
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=params['unsupervise_lr'], momentum=0.9)
    label_colours = np.random.randint(255, size=(100, 3))
    for batch_idx in range(params['maxIter']):
        # forwarding
        optimizer.zero_grad()
        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, params['nChannel'])
        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
        # superpixel refinement
        # TODO: use Torch Variable instead of numpy for faster calculation
        for i in range(len(l_inds)):
            labels_per_sp = im_target[l_inds[i]]
            u_labels_per_sp = np.unique(labels_per_sp)
            hist = np.zeros(len(u_labels_per_sp))
            for j in range(len(hist)):
                hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
            im_target[l_inds[i]] = u_labels_per_sp[np.argmax(hist)]
        target = torch.from_numpy(im_target)
        target = target.cuda()
        target = Variable(target)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # print (batch_idx, '/', args.maxIter, ':', nLabels, loss.data[0])
        print (batch_idx, '/', params['maxIter'], ':', nLabels, loss.item())

        if nLabels <= params['minLabels']:
            print ("nLabels", nLabels, "reached minLabels", params['minLabels'], ".")
            break
        if loss.item()<=0.25 and nLabels<=7:
            print("Reach the level we set to avoid further combination")
            break
    model.eval()
    output = model(data)[0]
    output = output.permute(1, 2, 0).contiguous().view(-1, params['nChannel'])
    ignore, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
    im_target_rgb = im_target_rgb.reshape(imarray.shape).astype(np.uint8)
    image= Image.fromarray(np.uint8(im_target_rgb))
    segment_path=os.path.join(save_path,'Segmented_'+str(count_img)+'.png')
    image.save(segment_path)
    origin_img_path=os.path.join(save_path,'Origin_'+str(count_img)+'.png')
    image = Image.fromarray(np.uint8(imarray))
    image.save(origin_img_path)
    #according to im_target_rgb to get individual segmented cell areas
    #1st find those area: 2nd largest color area
    Segment_Array=Find_Segment_Area(im_target_rgb)
    tmp_label_array_path = os.path.join(save_path, 'Segment_array' + str(count_img) + '.png')
    Visualize_Segment(Segment_Array, tmp_label_array_path)
    Label_Segment_Array=Locate_Cell_Area(Segment_Array,params)
    #useing count_img to update the array,
    Label_Segment_Array,count_img=Update_Label_Segment(Label_Segment_Array,count_img)
    #img = Image.fromarray(Label_Segment_Array, 'L')
    #im = Image.fromarray(Label_Segment_Array, 'RGB')
    #tmp_label_array_path=os.path.join(save_path,'Label_array'+str(count_img)+'.png')
    #img.save(tmp_label_array_path)
    tmp_label_array_path = os.path.join(save_path, 'Label_array' + str(count_img) + '.png')
    Visualize_Segment(Label_Segment_Array,tmp_label_array_path)
    return Label_Segment_Array,count_img

def Visualize_Segment(Overall_Segment_Array,tmp_label_array_path):
    overall_width = Overall_Segment_Array.shape[0]
    overall_height = Overall_Segment_Array.shape[1]
    tmp_array = np.zeros(overall_width*overall_height)
    for j in range(overall_width):
        for k in range(overall_height):
            if Overall_Segment_Array[j, k] < 0:
                continue
            tmp_array[j*overall_height+k] = Overall_Segment_Array[j, k]
    label_colours = np.random.randint(255, size=(255, 3))
    im_target_rgb = np.array([label_colours[int(c) % 255] for c in tmp_array])
    im_target_rgb = im_target_rgb.reshape([overall_width,overall_height,3]).astype(np.uint8)
    image = Image.fromarray(np.uint8(im_target_rgb))
    image.save(tmp_label_array_path)


def Update_Label_Segment(Label_Segment_Array,count_img):
    different_labels=np.unique(Label_Segment_Array)
    print("We have those different values, in total %d"%len(different_labels),different_labels)
    for tmp_label in different_labels:
        if tmp_label<0:
            continue
        tmp_coord=np.argwhere(Label_Segment_Array==tmp_label)
        for tmp_tmp_coord in tmp_coord:
            Label_Segment_Array[tmp_tmp_coord[0],tmp_tmp_coord[1]]=count_img
        count_img+=1
    return Label_Segment_Array,count_img


