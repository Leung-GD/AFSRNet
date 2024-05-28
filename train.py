from __future__ import absolute_import, division
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
from torch.autograd import Variable
from Utils import str2bool
#from Evaluation_L2 import EvalL2
from Evaluation_hynet import Evalhynet
from MSC_net_cov import get_L2_conv
#from RAL_net_cov import get_L2_conv
#from RAL_loss import loss_RAL
from Get_data import create_train_loaders,create_test_loaders
#from Get_data_patch_face import create_train_loaders,create_test_loaders
from tqdm import tqdm
import argparse
import os
import numpy as np
import time
from Losses import loss_HardNet,loss_hynet,loss_SOSNet
parser = argparse.ArgumentParser(description='RAL_Net')


parser.add_argument('--data-root', default='/media/amax/ef914467-feed-4743-b2ec-58c3abb24e7a/LHW/LJH/MSC-net/Brown', type=str, metavar='D',
#                    help='the path to your data')
#parser.add_argument('--data-root', default='/home/yons/Code/Facescape_dataset', type=str, metavar='D',
                    help='the path to your data')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=70, metavar='E',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--batch-size', type=int, default=500, metavar='BS',
                    help='input batch size for training (default: 512)')
parser.add_argument('--test-batch-size', type=int, default=500, metavar='BST',
                    help='input batch size for testing (default: 512)')
parser.add_argument('--n-pairs', type=int, default=1000000, metavar='N',
                    help='how many pairs will generate from the dataset')
# 1000000
parser.add_argument('--loss-type', type=str, default="0.15ASR",
                    help='type of training loss: "RAL_loss","Hard_loss","loss_hynet"')

parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='same as HardNet')

parser.add_argument('--augmentation', type=str2bool, default=True,
                    help='augmentation of random flip or rotation of 90 deg')

# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=413, metavar='S',
                    help='random seed (default: 0)')

args = parser.parse_args()
cudnn.benchmark = True
torch.cuda.manual_seed_all(args.seed)





os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_id)

def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        group['lr'] = args.lr * (
        1.0 - float(group['step']) * float(args.batch_size) / (args.n_pairs * float(args.epochs)))
    return

def create_optimizer(model, new_lr):
    optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=1e-4)
    return optimizer

def train(model, train_gen, epoch,optimizer):
        model.train()
        pbar = tqdm(enumerate(train_gen))
        i = 0
        lossdata = 0
        eye = Variable(torch.eye(args.batch_size).cuda())
        for batch_idx, data in pbar:
            # if i>=1:
            #     break
            data, target ,negative= data
            #data, target = data
            i = i + 1

            data1=data[:,:,8:56,8:56]
            data2=data[:,:,16:48,16:48]
            target1=target[:,:,8:56,8:56]
            target2=target[:,:,16:48,16:48]
            #data1=data[:,:,16:48,16:48]  #2input
            #target1=target[:,:,16:48,16:48]
            # print(data.shape)
            # print(data1.shape)
            # print(data2.shape)
            data = data.float().cuda()
            target = target.float().cuda()
            data1 = data1.float().cuda()
            target1 = target1.float().cuda()
            data2 = data2.float().cuda()
            target2 = target2.float().cuda()

            data = Variable(data)
            target = Variable(target)
            data1 = Variable(data1)
            target1 = Variable(target1)
            data2 = Variable(data2)
            target2 = Variable(target2)
            # print(data.shape)
            imgf1= model(data,data1,data2)
            imgf2 = model(target,target1,target2)
            #imgf1= model(data,data1)
            #imgf2 = model(target,target1)
            if args.loss_type == "RAL_loss":
                loss = loss_RAL(eye,imgf1,imgf2)
            elif args.loss_type == "Hard_loss":
                loss = loss_HardNet(eye,imgf1,imgf2)
            elif args.loss_type == "loss_hynet":
                loss = loss_hynet(eye,imgf1,imgf2)
            else:
                #loss = loss_HardNet(eye,imgf1,imgf2)
                loss = loss_hynet(eye,imgf1,imgf2)
                

            lossdata = lossdata + loss.data.detach().cpu()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            adjust_learning_rate(optimizer)
            if batch_idx % 20 == 0:
                pbar.set_description(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_gen.dataset),
                               100. * batch_idx / len(train_gen),
                        lossdata/i))
        #print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, lossdata/i))

def test(model, test_gen,traindata,testdata,result_root,epoch):
        model.eval()
        pbar = tqdm(enumerate(test_gen))
        i = 0
        fsp1 = torch.ones(1, 384).float().cuda()
        fsp2 = torch.ones(1, 384).float().cuda()
        label = torch.ones(1).long()
        for batch_idx, (data_a, data_p, label1) in pbar:
            # if i>=3:
            #     break
            # i = i+1
            data_a1=data_a[:, :, 8:56, 8:56]
            # print(data_a1.shape)
            data_p1=data_p[:, :, 8:56, 8:56]
            # print(data_p1.shape)
            data_a2=data_a[:, :, 16:48, 16:48]
            data_p2=data_p[:, :, 16:48, 16:48]
            # print(data_a.shape)
            # data1 = data[:, :, 8:56, 8:56]
            # data2 = data[:, :, 16:48, 16:48]
            # target1 = target[:, :, 8:56, 8:56]
            # target2 = target[:, :, 16:48, 16:48]
            #data_a1=data_a[:,:,16:48,16:48]  #2input
            #data_p1=data_p[:,:,16:48,16:48]

            data_a = Variable(data_a.float().cuda())
            data_p = Variable(data_p.float().cuda())
            data_a1 = Variable(data_a1.float().cuda())
            data_p1 = Variable(data_p1.float().cuda())
            data_a2 = Variable(data_a2.float().cuda())
            data_p2 = Variable(data_p2.float().cuda())
 
            imgf1 = torch.squeeze(model(data_a,data_a1,data_a2)).data
            imgf2 = torch.squeeze(model(data_p,data_p1,data_p2)).data
            
            #imgf1 = torch.squeeze(model(data_a,data_a1)).data
            #imgf2 = torch.squeeze(model(data_p,data_p1)).data

            fsp1 = torch.cat((fsp1,imgf1),0)
            fsp2 = torch.cat((fsp2,imgf2),0)
            label = torch.cat((label,label1),0)
            if batch_idx % 20 == 0:
                pbar.set_description(
                               str(100.*batch_idx / len(test_gen)))


        fsp1 = fsp1[1:fsp1.size(0)]
        fsp2 = fsp2[1:fsp2.size(0)]
        label = label[1:label.size(0)]
        print('finish get data, please wait')
        #EvalL2(fsp1, fsp2,label, traindata, testdata,result_root)
        Evalhynet(fsp1, fsp2,label, traindata, testdata,result_root,epoch)

def main(traindata,testdata1,testdata2):
    from datetime import date
    today=date.today()
    if args.augmentation == True:
        saving_root = str(today)+"_duichenAll_paper_ASOSR/"+ args.loss_type + "_" + str(args.epochs) + "_" + str(args.n_pairs) + \
        "_" + str(args.batch_size)+traindata + "_" + "aug"
    else:
        saving_root = str(today)+"_duichenAll_paper_ASOSR/"+ args.loss_type + "_" + str(args.epochs) + "_" + str(args.n_pairs) + \
        "_" + str(args.batch_size)+traindata

    if not os.path.exists(saving_root):
        os.makedirs(saving_root)

    start = args.start_epoch
    end = start + args.epochs
    if start != 0:
        model = torch.load(
            "/media/amax/ef914467-feed-4743-b2ec-58c3abb24e7a/LHW/MSC-net_org_3ch_64/paper_ASOSR/baeline_new/0.15ASR_50_1000000_512liberty_aug/0.15ASR_{}_1000000_512.pth".format(
                start)).cuda()
    else:
        model = get_L2_conv().cuda()

    optimizer1 = create_optimizer(model, args.lr)




    for epoch in range(start, end):
        train_loader = create_train_loaders(traindata,args.batch_size,args.n_pairs,
                                            args.augmentation,args.data_root)
        # iterate over test loaders and test results
        train( model,train_loader, epoch, optimizer1)
        torch.save(model.state_dict(), saving_root + "/" + args.loss_type + "_" + str(epoch) + "_" + str(args.n_pairs) + \
        "_" + str(args.batch_size) + '.pth')

        # release ram
        train_loader=0
        # if epoch%2==0:
        test(model, create_test_loaders(testdata1, args.batch_size, args.data_root), traindata, testdata1,
             saving_root, epoch)
        # time.sleep(10)
        test(model, create_test_loaders(testdata2, args.batch_size, args.data_root), traindata, testdata2,
             saving_root, epoch)



            # model = torch.load(saving_root + "/" + args.loss_type + "_" + str(epoch) + "_" + str(args.n_triplets) + \
    #     "_" + str(args.batch_size) + '.pth').cuda()
    test(model, create_test_loaders(testdata1,args.batch_size,args.data_root),traindata,testdata1,saving_root,epoch)
        #time.sleep(10)
    test(model, create_test_loaders(testdata2,args.batch_size,args.data_root),traindata,testdata2,saving_root,epoch) 
        #time.sleep(10)
    # test(model, create_test_loaders(traindata,args.batch_size,args.data_root),traindata,traindata,saving_root,epoch)



if __name__=='__main__':
    # time.sleep(800)
    main('liberty', 'notredame', 'yosemite')
    #
    # main('yosemite','liberty','notredame')
    # main('notredame','yosemite','liberty')
