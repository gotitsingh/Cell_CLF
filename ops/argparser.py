#
# Copyright (C) 2018 Xiao Wang
# Email:xiaowang20140001@gmail.com
#

import parser
import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-F',type=str, required=True,help='training data path')
    parser.add_argument('-F1', type=str,  help='testing data path')
    parser.add_argument('--mode',type=int,required=True,help='0: calculating iRMSD and then prepare input randomly')
    parser.add_argument('--type',type=int,default=0,help='setting type: 0: common setting, 1: including large cells')
    parser.add_argument('--choose',type=str,default='0',help='gpu id choose for training, if you use -1 means you do not use gpu')
    parser.add_argument('--lr', type=float, default='0.002', help='learning rate for training')
    parser.add_argument('--reg', type=float, default='1e-5', help='REG for training')
    parser.add_argument('--class', type=int, default='2', help='number of classes')
    parser.add_argument('--cardinality',default=32, type=int,help='ResNeXt cardinality')
    parser.add_argument('--batch_size', type=int, default='128', help='batch size for training')
    parser.add_argument('--model', type=int, default=0, help='model type for training: 0:resnet20 1:resnet50 2:resnet101 3:resnet152')
    parser.add_argument('-M', type=str,  help='model path to resume the unexpectedly stopped training')  # File path for our MAINMAST code
    parser.add_argument('--resume',type=int,default=0,help='Resume or not')
    parser.add_argument('--width',type=int,default=50,help="Width of classification image")
    parser.add_argument('--height',type=int,default=50,help="Height of classification image")
    parser.add_argument('--seed',type=int,default=888,help="the random seed for spliting data and so on")
    parser.add_argument('--num_workers', type=int, default=4, help="number of workers for the dataloader")
    parser.add_argument('--portion',type=float,default=0.8,help="portion of training data for training, others for validation")
    parser.add_argument('--epoch',type=int,default=100,help="The number of epochs for training")
    parser.add_argument('--use_ema',type=int,default=0,help="use ema model or not")
    parser.add_argument('--num_superpixels', metavar='K', default=10000, type=int,
                        help='number of superpixels')
    parser.add_argument('--compactness', metavar='C', default=100, type=float,
                        help='compactness of superpixels')
    parser.add_argument('--nConv', metavar='M', default=2, type=int,
                        help='number of convolutional layers in unsupervised part')
    parser.add_argument('--nChannel', metavar='N', default=100, type=int,
                        help='number of channels')
    parser.add_argument('--unsupervise_lr',default=0.1,type=float,help="Unsupervised learning rate")
    parser.add_argument('--maxIter',default=200, type=int,
                    help='number of maximum iterations')
    parser.add_argument('--minLabels', metavar='minL', default=5, type=int,
                        help='minimum number of labels')
    parser.add_argument('--resize',default=0,type=int,help="if we need resize the input or not to make predictions clear")
    parser.add_argument('--resize_height',default=200,type=int,help="The required image height used for the segmentation")
    parser.add_argument('--resize_width', default=200, type=int,help="The required image width used for the segmentation")
    parser.add_argument('--filter_size',default=3,type=int,help="user can adjust their own filter size to have different segmentation results")
    parser.add_argument('--threshold',default=195,type=int,help="Threshold used to do image segmentation (Suggested 150-210 for big cell cases)")
    parser.add_argument('--remove_pixel',default=500,type=int,help="remove positive segmented area")
    args = parser.parse_args()
    # try:
    #     import ray,socket
    #     rayinit()
    # except:
    #     print('ray need to be installed')#We do not need this since GAN can't be paralleled.
    params = vars(args)
    return params