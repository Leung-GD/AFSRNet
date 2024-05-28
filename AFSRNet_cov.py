from __future__ import division, print_function
from FRN import FRN,TLU
import torch
import torch.nn.init

from Utils import L2Norm, np_reshape

import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, input_channels,output_channels,kernel_size,stride,padding,dilation):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation),
            nn.BatchNorm2d(num_features=output_channels), 
            nn.ReLU(),
            #nn.Dropout(0.1)
            )

    def forward(self, input):
        return self.conv(input)

class backbone(nn.Module):
    """HardNet model definition
    """
    def __init__(self):
        super(backbone, self).__init__()
        self.features = nn.Sequential(
            #nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
            #nn.BatchNorm2d(32, affine=False),
            #nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU()
            #nn.Dropout(0.3),
            #nn.Conv2d(128, 128, kernel_size=8, bias = False),
            #nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)

    def forward(self, input):
        x_features = self.features(input)
        return x_features

class backbone2(nn.Module):
    """HardNet model definition
    """
    def __init__(self):
        super(backbone2, self).__init__()
        self.features = nn.Sequential(
            #nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
            #nn.BatchNorm2d(32, affine=False),
            #nn.ReLU(),
            #FRN(32, is_bias=True),
            #TLU(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            FRN(32, is_bias=True),
            TLU(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            FRN(64, is_bias=True),
            TLU(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            FRN(64, is_bias=True),
            TLU(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = False),
            FRN(128, is_bias=True),
            TLU(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            FRN(128, is_bias=True),
            TLU(128)
            #nn.Dropout(0.3),
            #nn.Conv2d(128, 128, kernel_size=8, bias = False),
            #nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)

    def forward(self, input):
        x_features = self.features(input)
        return x_features

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)



class MSCNet(nn.Module):
    def __init__(self):
        super(MSCNet,self).__init__()
        self.epsilon = 1e-4
        self.swish=Swish()
        # Weight
        self.p13_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p13_w_relu = nn.ReLU()
        self.p23_w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p23_w_relu = nn.ReLU()
        self.p33_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p33_w_relu = nn.ReLU()
        self.pc_w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.pc_w_relu = nn.ReLU()


        self.input1_cov1=ConvBlock(input_channels=1,output_channels=32,kernel_size=3,stride=2,padding=3,dilation=3)
        self.input2_cov1=ConvBlock(input_channels=1,output_channels=32,kernel_size=11,stride=1,padding=2,dilation=2)
        self.input3_cov1=ConvBlock(input_channels=1,output_channels=32,kernel_size=3,stride=1,padding=1,dilation=1)

        #self.input1_cov2=ConvBlock(input_channels=32,output_channels=32,kernel_size=3,stride=1,padding=1,dilation=1)
        #self.input2_cov2=ConvBlock(input_channels=32,output_channels=32,kernel_size=3,stride=1,padding=1,dilation=1)
        #self.input3_cov2=ConvBlock(input_channels=32,output_channels=32,kernel_size=3,stride=1,padding=1,dilation=1)
        # self.input1_cov2=backbone2()
        # self.input2_cov2=backbone2()
        # self.input3_cov2=backbone2()
        self.input1_cov2=backbone()
        self.input2_cov2=backbone()
        self.input3_cov2=backbone()

        self.input1_cov3=ConvBlock(input_channels=128,output_channels=128,kernel_size=1,stride=1,padding=0,dilation=1)
        self.input2_cov3=ConvBlock(input_channels=128,output_channels=128,kernel_size=1,stride=1,padding=0,dilation=1)
        self.input3_cov3=ConvBlock(input_channels=128,output_channels=128,kernel_size=1,stride=1,padding=0,dilation=1)

        self.c1=ConvBlock(input_channels=128,output_channels=512,kernel_size=3,stride=1,padding=1,dilation=1)
        self.descriptornet=nn.Sequential(   nn.Dropout(0.2),
                                            nn.Conv2d(512, 384, kernel_size=8, bias = False),#512->128
                                            nn.BatchNorm2d(384, affine=False),
                                            #nn.Conv2d(128, 128, kernel_size=8, bias = False),#512->128
                                            #nn.BatchNorm2d(128, affine=False),
                                        )
    def forward(self,input1,input2,input3):
        x11=self.input1_cov1(input1)
        x21=self.input2_cov1(input2)
        x31=self.input3_cov1(input3)
        # print(x11.shape)
        # print(x21.shape)
        # print(x31.shape)
        x12=self.input1_cov2(x11)
        x22=self.input2_cov2(x21)
        x32=self.input3_cov2(x31)
        
        # # Weights for P13 
        p13_w = self.p13_w_relu(self.p13_w)
        weight = p13_w / (torch.sum(p13_w, dim=0) + self.epsilon)
        # Connections
        # print(weight.shape)
        # print(weight[0].shape)
        # print(weight[1].shape)


        x13 = self.input1_cov3(self.swish(weight[0] * x12 + weight[1] * x22))

        ## Weights for P23 
        p23_w = self.p23_w_relu(self.p23_w)
        weight = p23_w / (torch.sum(p23_w, dim=0) + self.epsilon)
        # Connections 
        x23 = self.input2_cov3(self.swish(weight[0] * x12 + weight[1] * x22+weight[2] * x32))

        ## Weights for P33 
        p33_w = self.p33_w_relu(self.p33_w)
        weight = p33_w / (torch.sum(p33_w, dim=0) + self.epsilon)
        # Connections 
        x33 = self.input3_cov3(self.swish(weight[0] * x22 + weight[1] * x32))
        

        # PC 
        pc_w = self.pc_w_relu(self.pc_w)
        weight = pc_w / (torch.sum(pc_w, dim=0) + self.epsilon)
        # Connections 
        c1 = self.c1(self.swish(weight[0] * x13 + weight[1] * x23+weight[2] * x33))
        #c1 = self.c1(self.swish(weight[0] * x12 + weight[1] * x22+weight[2] * x32))
        descripor=self.descriptornet(c1)
        descripor = descripor.view(descripor.size(0), -1)
        #print(descripor.size())
        #assert 1==0
        return L2Norm()(descripor)


class MSCNet_concate(nn.Module):
    def __init__(self):
        super(MSCNet_concate, self).__init__()
        self.epsilon = 1e-4
        self.swish = Swish()
        # Weight
        self.p13_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p13_w_relu = nn.ReLU()
        self.p23_w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p23_w_relu = nn.ReLU()
        self.p33_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p33_w_relu = nn.ReLU()
        self.pc_w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.pc_w_relu = nn.ReLU()

        self.input1_cov1 = ConvBlock(input_channels=1, output_channels=32, kernel_size=3, stride=2, padding=3,
                                     dilation=3)
        self.input2_cov1 = ConvBlock(input_channels=1, output_channels=32, kernel_size=11, stride=1, padding=2,
                                     dilation=2)
        self.input3_cov1 = ConvBlock(input_channels=1, output_channels=32, kernel_size=3, stride=1, padding=1,
                                     dilation=1)

        # self.input1_cov2=ConvBlock(input_channels=32,output_channels=32,kernel_size=3,stride=1,padding=1,dilation=1)
        # self.input2_cov2=ConvBlock(input_channels=32,output_channels=32,kernel_size=3,stride=1,padding=1,dilation=1)
        # self.input3_cov2=ConvBlock(input_channels=32,output_channels=32,kernel_size=3,stride=1,padding=1,dilation=1)
        self.input1_cov2 = backbone2()
        self.input2_cov2 = backbone2()
        self.input3_cov2 = backbone2()

        self.input1_cov3 = ConvBlock(input_channels=128, output_channels=128, kernel_size=1, stride=1, padding=0,
                                     dilation=1)
        self.input2_cov3 = ConvBlock(input_channels=128, output_channels=128, kernel_size=1, stride=1, padding=0,
                                     dilation=1)
        self.input3_cov3 = ConvBlock(input_channels=128, output_channels=128, kernel_size=1, stride=1, padding=0,
                                     dilation=1)

        self.c1 = ConvBlock(input_channels=128, output_channels=512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.descriptornet = nn.Sequential(nn.Dropout(0.2),
                                           nn.Conv2d(128, 128, kernel_size=8, bias=False),  # 512->128
                                           nn.BatchNorm2d(128, affine=False),
                                           # nn.Conv2d(128, 128, kernel_size=8, bias = False),#512->128
                                           # nn.BatchNorm2d(128, affine=False),
                                           )

    def forward(self, input1, input2, input3):
        x11 = self.input1_cov1(input1)
        x21 = self.input2_cov1(input2)
        x31 = self.input3_cov1(input3)
        # print(x11.shape)
        # print(x21.shape)
        # print(x31.shape)
        x12 = self.input1_cov2(x11)
        x22 = self.input2_cov2(x21)
        x32 = self.input3_cov2(x31)

        x13=self.descriptornet(x12)
        x23=self.descriptornet(x22)
        x33=self.descriptornet(x32)

        descripor=torch.cat((x13,x23,x33),1)
        # # Weights for P13
        # p13_w = self.p13_w_relu(self.p13_w)
        # weight = p13_w / (torch.sum(p13_w, dim=0) + self.epsilon)
        # # Connections
        # # print(weight.shape)
        # # print(weight[0].shape)
        # # print(weight[1].shape)
        #
        # x13 = self.input1_cov3(self.swish(weight[0] * x12 + weight[1] * x22))
        #
        # ## Weights for P23
        # p23_w = self.p23_w_relu(self.p23_w)
        # weight = p23_w / (torch.sum(p23_w, dim=0) + self.epsilon)
        # # Connections
        # x23 = self.input2_cov3(self.swish(weight[0] * x12 + weight[1] * x22 + weight[2] * x32))
        #
        # ## Weights for P33
        # p33_w = self.p33_w_relu(self.p33_w)
        # weight = p33_w / (torch.sum(p33_w, dim=0) + self.epsilon)
        # # Connections
        # x33 = self.input3_cov3(self.swish(weight[0] * x22 + weight[1] * x32))
        #
        # # PC
        # pc_w = self.pc_w_relu(self.pc_w)
        # weight = pc_w / (torch.sum(pc_w, dim=0) + self.epsilon)
        # # Connections
        # c1 = self.c1(self.swish(weight[0] * x13 + weight[1] * x23 + weight[2] * x33))
        # # c1 = self.c1(self.swish(weight[0] * x12 + weight[1] * x22+weight[2] * x32))
        # descripor = self.descriptornet(c1)
        descripor = descripor.view(descripor.size(0), -1)
        # print(descripor.size())
        # assert 1==0
        return L2Norm()(descripor)


class MSCNet2(nn.Module):
    def __init__(self):
        super(MSCNet2,self).__init__()
        self.epsilon = 1e-4
        self.swish=Swish()
        # Weight
        self.pc_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.pc_w_relu = nn.ReLU()



        self.input1_cov1=ConvBlock(input_channels=1,output_channels=32,kernel_size=3,stride=2,padding=2,dilation=2)
        self.input2_cov1=ConvBlock(input_channels=1,output_channels=32,kernel_size=3,stride=1,padding=1,dilation=1)

        self.input1_cov2=backbone2()
        self.input2_cov2=backbone2()

        self.input1_cov3=ConvBlock(input_channels=128,output_channels=128,kernel_size=1,stride=1,padding=0,dilation=1)
        self.input2_cov3=ConvBlock(input_channels=128,output_channels=128,kernel_size=1,stride=1,padding=0,dilation=1)

        self.c1=ConvBlock(input_channels=128,output_channels=128,kernel_size=3,stride=1,padding=1,dilation=1)
        self.descriptornet=nn.Sequential(   #nn.Dropout(0.1),
                                            nn.Conv2d(128, 512, kernel_size=8, bias = False),
                                            nn.BatchNorm2d(512, affine=False)
                                        )
    def forward(self,input1,input2):
        x11=self.input1_cov1(input1)
        x21=self.input2_cov1(input2)
        
        x12=self.input1_cov2(x11)
        x22=self.input2_cov2(x21)
        
        # PC 
        pc_w = self.pc_w_relu(self.pc_w)
        weight = pc_w / (torch.sum(pc_w, dim=0) + self.epsilon)
        # Connections 
        c1 = self.c1(self.swish(weight[0] * x12 + weight[1]*x22))
        descripor=self.descriptornet(c1)
        descripor = descripor.view(descripor.size(0), -1)
        #print(descripor.size())
        #assert 1==0
        return L2Norm()(descripor)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=0.6)
        try:
            nn.init.constant_(m.bias.data, 0.01)
        except:
            pass
    return

def get_L2_conv():
    return MSCNet()
    # return MSCNet_concate()
