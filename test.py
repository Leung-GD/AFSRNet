'''this Hard_loss is intercepted from HardNet
"Working hard to know your neighbor's margins: Local descriptor learning loss" '''

import torch
import torch.nn as nn
import sys
import math
from MSCNet_brown import test
from Get_data import create_train_loaders,create_test_loaders
#from Get_data_patch_face import create_train_loaders,create_test_loaders

#model = torch.load("/home/yons/Code/RAL-Net/RAL-Net_result_models/RAL512d_30_1000000_500_augliberty/RAL512d_29_1000000_500.th").cuda()
model = torch.load("/media/amax/ef914467-feed-4743-b2ec-58c3abb24e7a/LHW/MSC-net_org_3ch_64/paper_ASOSR/baeline_new/0.15ASR_50_1000000_512liberty_aug/0.15ASR_45_1000000_512.pth").cuda()
traindata='liberty'
testdata1='yosemite'
testdata2='notredame'
data_root='/media/amax/ef914467-feed-4743-b2ec-58c3abb24e7a/LHW/LJH/MSC-net/Brown'
#data_root='/home/yons/Code/Facescape_dataset'
saving_root='/home/yons/Code/RAL-Net/test_save'
# test(model, create_test_loaders(testdata1,512,data_root),traindata,testdata1,saving_root,1)
test(model, create_test_loaders(testdata2,512,data_root),traindata,testdata1,saving_root,1)
#test(model, create_test_loaders(traindata,512,data_root),traindata,testdata1,saving_root)

#rows=3
#a = torch.ones(rows,1).float().cuda()
#N = torch.ones(rows,1).float().cuda()
#D = torch.ones(rows,1).float().cuda()
#D=D*2
#for i in range(0,rows):
#        a[i]=(N[i])/(D[i])
#print(a)


