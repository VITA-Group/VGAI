import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
from models import Darknet
import os
import torch.nn.functional as F
from yolo_utils.utils import *


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, \
                     stride=stride, padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    "5x5 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, \
                     stride=stride, padding=2, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class DAGNN(torch.nn.Module):
    def __init__(self, filter_length=3, n_vis_out=6):
        super(DAGNN, self).__init__()
        self.filter_length = filter_length
        self.conv1 = conv5x5(in_planes=3, out_planes=16, stride=(2, 1))
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d((2,1), stride=(2,1))

        ### residual-block1
        downsample1 = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=1, stride=(2,1), bias=False),
                nn.BatchNorm2d(16))
        self.block1 = BasicBlock(16, 16, stride=(2,1), downsample=downsample1)
        ### residual-block2
        downsample2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=1, stride=(2,1), bias=False),
                nn.BatchNorm2d(32))
        self.block2 = BasicBlock(16, 32, stride=(2,1), downsample=downsample2)

        ### residual-block3
        downsample3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=1, stride=(2,1), bias=False),
                nn.BatchNorm2d(64))
        self.block3 = BasicBlock(32, 64, stride=(2,1), downsample=downsample3)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1024))
        self.convlast = nn.Conv2d(64, 1, kernel_size=1)

        self.n_vis_out = n_vis_out
        self.vis_predict = torch.nn.Linear(1024, n_vis_out)

        # Second part
        layer_sizes = [self.filter_length * n_vis_out, 256, 256, 256, 2]
        self.n_layers = len(layer_sizes) - 1
        #self.dropout = nn.Dropout(0.5)

        self.hidden = []
        for n in range(0, self.n_layers - 1):
            self.hidden.append(torch.nn.Linear(layer_sizes[n], layer_sizes[n + 1]))
        self.hidden = torch.nn.ModuleList(self.hidden)

        # output layer
        self.predict = torch.nn.Linear(layer_sizes[self.n_layers - 1], layer_sizes[self.n_layers])


    def forward(self, input, a_nets):

        x_agg = torch.zeros(input.shape[0], self.n_vis_out * self.filter_length).cuda()
        for t in range(self.filter_length):
            a_net = a_nets[:, :, t] # 50 x 50
            #print('input shape = {}'.format(input.shape))
            x = input[:, :, :, :, t] # 50 x 384
            #print('x shape = {}'.format(x.shape))
            ### DroneNet
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.avgpool(x)
            x = self.convlast(x)
            x = x.view(x.size(0), -1)
            x = self.vis_predict(x)

            a_net_comm = a_net.unsqueeze(2)
            x = x * a_net_comm # 50 x 6  * 50 x 50 x 1  
            x = torch.sum(x, dim=1) # 50 x 50 x 1
            x_agg = x_agg[:, :-self.n_vis_out].unsqueeze(1) # 50 x 1 x 12
            x_agg = x_agg.repeat(1, a_net.shape[0], 1) # 50 x 50 x 12
            x_agg = torch.sum(x_agg * a_net_comm, dim=1) # 50 x 12
            x_agg = torch.cat((x, x_agg), 1) # 50 x 18
        u = x_agg
        for n in range(0, self.n_layers - 1):
            u = torch.nn.functional.relu((self.hidden[n](u)))
            #u = self.dropout(u)
        u = self.predict(u)
        #u = torch.clamp(u, -0.5, 0.5)
        
        return x_agg, u      


def vis_dagnn(K=3, pretrained=False, **kwargs):
    # n = 6
    model = DAGNN(filter_length=K, n_vis_out=kwargs['n_vis_out'])
    return model




class GRNN(torch.nn.Module):
    def __init__(self, filter_length=3, n_vis_out=6):
        super(GRNN, self).__init__()
        self.filter_length = filter_length
        self.conv1 = conv5x5(in_planes=3, out_planes=16, stride=(2, 1))
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d((2,1), stride=(2,1))
        self.a = torch.nn.Parameter(torch.ones(filter_length))
        self.b = torch.nn.Parameter(torch.ones(filter_length))

        ### residual-block1
        downsample1 = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=1, stride=(2,1), bias=False),
                nn.BatchNorm2d(16))
        self.block1 = BasicBlock(16, 16, stride=(2,1), downsample=downsample1)
        ### residual-block2
        downsample2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=1, stride=(2,1), bias=False),
                nn.BatchNorm2d(32))
        self.block2 = BasicBlock(16, 32, stride=(2,1), downsample=downsample2)

        ### residual-block3
        downsample3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=1, stride=(2,1), bias=False),
                nn.BatchNorm2d(64))
        self.block3 = BasicBlock(32, 64, stride=(2,1), downsample=downsample3)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1024))
        self.convlast = nn.Conv2d(64, 1, kernel_size=1)

        self.n_vis_out = n_vis_out
        self.vis_predict = torch.nn.Linear(1024, n_vis_out)

        # Second part
        layer_sizes = [n_vis_out, 256, 256, 256, 2]
        self.n_layers = len(layer_sizes) - 1
        #self.dropout = nn.Dropout(0.5)

        self.hidden = []
        for n in range(0, self.n_layers - 1):
            self.hidden.append(torch.nn.Linear(layer_sizes[n], layer_sizes[n + 1]))
        self.hidden = torch.nn.ModuleList(self.hidden)

        # output layer
        self.predict = torch.nn.Linear(layer_sizes[self.n_layers - 1], layer_sizes[self.n_layers])


    def forward(self, input, a_nets, input_state):
        states = torch.zeros(input.shape[0], self.n_vis_out, self.filter_length).cuda()
        u = 0

        #x_agg = torch.zeros(input.shape[0], self.n_vis_out * self.filter_length).cuda()
        for t in range(self.filter_length):
            a_net = a_nets[:, :, t] # 50 x 50
            a_net_comm = a_net.unsqueeze(2)
            x = input[:, :, :, :, t] # 50 x 384
            #x = input[:, :, t] # 50 x 384
            ### DroneNet
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.avgpool(x)
            x = self.convlast(x)
            x = x.view(x.size(0), -1)
            x = self.vis_predict(x)
            x = x * a_net_comm # 50 x F  * 50 x 50 x 1  
            x = torch.sum(x, dim=1) # 50 x F

            input_state = input_state.unsqueeze(1) # 50 x 1 x F
            input_state = input_state.repeat(1, a_net.shape[0], 1) # 50 x 50 x F
            input_state = torch.sum(input_state * a_net_comm, dim=1) # 50 x F
            input_state = torch.tanh(x * self.a[t] + input_state  * self.b[t])
            states[:, :, t] = input_state
        u = input_state
        for n in range(0, self.n_layers - 1):
            u = torch.nn.functional.relu((self.hidden[n](u)))
        u = self.predict(u)
        
        return states, u      


def vis_grnn(K=3, **kwargs):
    # n = 6
    model = GRNN(filter_length=K, n_vis_out=kwargs['n_vis_out'])
    return model


class ReluNet(torch.nn.Module):
    def __init__(self, filter_length=3):
        super(ReluNet, self).__init__()
        n_feature = filter_length * 6       
        layer_sizes = [n_feature, 256, 256, 2]
        self.n_layers = len(layer_sizes) - 1
        self.filter_length = filter_length

        # hidden layers
        self.hidden = []
        for n in range(0, self.n_layers - 1):
            self.hidden.append(torch.nn.Linear(layer_sizes[n], layer_sizes[n + 1]))
        self.hidden = torch.nn.ModuleList(self.hidden)

        # output layer
        self.predict = torch.nn.Linear(layer_sizes[self.n_layers - 1], layer_sizes[self.n_layers])

    def forward(self, input, a_nets):
        x_agg = torch.zeros(input.shape[0], 6 * self.filter_length).cuda()
        for t in range(self.filter_length):
            a_net = a_nets[:, :, t] # 50 x 50
            x = input[:, :, t] # 50 x 6
            a_net_comm = a_net.unsqueeze(2)
            x = x * a_net_comm # 50 x 6  * 50 x 50 x 1  
            x = torch.sum(x, dim=1) # 50 x 50 x 1
            x_agg = x_agg[:, :-6].unsqueeze(1) # 50 x 1 x 12
            x_agg = x_agg.repeat(1, a_net.shape[0], 1) # 50 x 50 x 12
            x_agg = torch.sum(x_agg * a_net_comm, dim=1) # 50 x 12
            x_agg = torch.cat((x, x_agg), 1) # 50 x 18


        x_agg = torch.clamp(x_agg, -30, 30 )
        for n in range(0, self.n_layers - 1):
            x_agg = torch.nn.functional.relu((self.hidden[n](x_agg)))
        return self.predict(x_agg)

def loc_dagnn(K=3, **kwargs):
    # n = 6
    model = ReluNet(filter_length=K)
    return model


def pad_to_square(img, pad_value=0.):
    # 需要channel-first
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad




class Yolo_DAGNN(torch.nn.Module):
    def __init__(self, filter_length=3, n_vis_out=6):
        super(Yolo_DAGNN, self).__init__()
        self.filter_length = 3

        self.n_vis_out = n_vis_out
        self.vis_predict = torch.nn.Linear(36, n_vis_out)

        # Second part
        layer_sizes = [self.filter_length * n_vis_out, 256, 256, 256, 2]
        self.n_layers = len(layer_sizes) - 1
        #self.dropout = nn.Dropout(0.5)

        self.hidden = []
        for n in range(0, self.n_layers - 1):
            self.hidden.append(torch.nn.Linear(layer_sizes[n], layer_sizes[n + 1]))
        self.hidden = torch.nn.ModuleList(self.hidden)

        # output layer
        self.predict = torch.nn.Linear(layer_sizes[self.n_layers - 1], layer_sizes[self.n_layers])






        # wz-detactor loading
        detector_ckpt_dir = 'yolov3_ckpt_100.pth'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        print('self.device',self.device)
                
        self.detector_head = Darknet('config/yolov3.cfg').to(self.device)
        self.detector_head.load_state_dict(torch.load(detector_ckpt_dir, map_location=lambda storage, loc: storage))
        self.detector_head.eval()
        print('model being loaded')


    def sort_features_pad_len3(self, bbox):
        if len(bbox.shape)!=2:
            raise

        xys = []
        if len(bbox)==1:
            xys0 = torch.stack([*bbox[0,:2], bbox[0,2]*bbox[0,3]*bbox[0,4]**2*1e2], 0)
            xys.append(xys0)
            xys.append(xys0*0)
            xys.append(xys0*0)
        elif len(bbox)==2:
            xys0 = torch.stack([*bbox[0,:2], bbox[0,2]*bbox[0,3]*bbox[0,4]**2*1e2], 0)
            xys1 = torch.stack([*bbox[1,:2], bbox[1,2]*bbox[1,3]*bbox[1,4]**2*1e2], 0)
            xys.append(xys0)
            xys.append(xys1)
            xys.append(xys0*0)
        else:
            for bv in bbox:
                xysv = torch.stack([*bv[:2], bv[2]*bv[3]*bv[4]**2*1e2], 0)
                xys.append(xysv)


        xys = torch.stack(xys, 0)
        xord = torch.argsort(xys[:,2], descending=True)
        xys = xys[xord]
        

        return xys

    def avg_of_xys(self, xys_segment):
        x = xys_segment[:,0]
        y = xys_segment[:,1]
        s = xys_segment[:,2]
        mean_x = torch.sum(x*s)/(torch.sum(s)+1e-8)
        mean_y = torch.sum(y*s)/(torch.sum(s)+1e-8)
        mean_s = torch.mean(s)
        return torch.stack([mean_x, mean_y, mean_s])

    def xys_13a_1im(self, bbox):
        # 功能：输入single image的label （type=tensor) ，返回features
        # returns: dim=9
            # 最大S的x/y/s
            # 前3大S的 x/y/s 均值
            # 所有detection的 x/y/s 均值
        # print('check bbox:',bbox)
        xys = self.sort_features_pad_len3(bbox) # shape: [N>=3, 3=x/y/s]

        res = []
        res.append(xys[0])
        res.append(self.avg_of_xys(xys[:3]))
        res.append(self.avg_of_xys(xys))
        xys_v9 = torch.cat(res, 0)
        return xys_v9


    def xys_features_1uav(self, imgs):
        # input  img shape:     [4=前后左右四张图,   3,144,256]
        imgs = torch.stack([pad_to_square(img)[0] for img in imgs])
        # imgs shape: [batch_size, 3, 256, 256]
        # imgs = torch.stack([resize(img, 416) for img in imgs])
        # imgs shape: [batch_size, 3, 416, 416]

        outputs = self.detector_head(imgs)  # 是个tensor，shape = [4=img个数, 4032, 85]
        bbox_list = non_max_suppression(outputs, conf_thres=0.5, nms_thres=0.4)  # 是一个list，元素个数=imgs的个数，每个元素是tensor或者None，如果是tensor则0123是location，4是confidence，
        # save_padded_ims_labels(imgs, bbox_list)



        # 此处bbox 是一个list，里面是tensor，每个tensor shape:  [n_detected, 7] ; 每个tensor的 0123 列是xyxy数值，且没有归一化，处于0~416之间； 如果没探测到就是None
        
        res_features_v36 = []
        for ib, bbox in enumerate(bbox_list): # 每次迭代对应一个img （前后左右4个摄像机共4次）
            if bbox is None:
                # print('this bbox is None!')
                res_features_v36.append(torch.zeros(9, device=self.device))
                continue
            bbox[..., :4] = xyxy2xywh_norm(bbox[..., :4], int(imgs.shape[-1]))
            xys_v9 = self.xys_13a_1im(bbox)  # shape: [9, ]
            res_features_v36.append(xys_v9)




        res_features_v36 = torch.cat(res_features_v36, 0) # shape: [9个feature*4个摄像机, ] = [36, ]

        return res_features_v36






    def forward(self, input, a_nets):
        x_agg = torch.zeros(input.shape[0], self.n_vis_out * self.filter_length).to(self.device)
        for t in range(self.filter_length):
            a_net = a_nets[:, :, t] # 50 x 50
            x = input[..., t]
            # input.shape:  [50, 3, 144, 1024, 3] = [UAV, img, filter]


            # imgs = Variable(imgs.type(Tensor), requires_grad=False)  # shape: [batch_size=8, 3, 416, 416]




            with torch.no_grad():
                x_list = []
                for iuav in range(x.shape[0]):
                    this_uav = x[iuav]  # shape [3, 144, 1024], type is tensor
                    imgs = this_uav.reshape(3,144,4,256).transpose(1,2).transpose(0,1)  # shape:   [4=前后左右四张图,   3,144,256]
                    featurev = self.xys_features_1uav(imgs)
                    x_list.append(featurev)
                x = torch.stack(x_list, 0)    # shape: [50, 36]

            x = self.vis_predict(x)
            a_net_comm = a_net.unsqueeze(2)
            x = x * a_net_comm # [50, 24] * [50, 50, 1] = [50, 50, 24]
            x = torch.sum(x, dim=1) # 50 x 50 x 1
            x_agg = x_agg[:, :-self.n_vis_out].unsqueeze(1) # 50 x 1 x 12
            x_agg = x_agg.repeat(1, a_net.shape[0], 1) # 50 x 50 x 12
            x_agg = torch.sum(x_agg * a_net_comm, dim=1) # 50 x 12
            x_agg = torch.cat((x, x_agg), 1) # 50 x 18
        u = x_agg  # shape: [50, 72]
        for n in range(0, self.n_layers - 1):  #self.n_layers=4
            u = torch.nn.functional.relu((self.hidden[n](u)))
            #u = self.dropout(u)
        u = self.predict(u)   # [50, 256] -> [50,2]
        #u = torch.clamp(u, -0.5, 0.5)
        
        return x_agg, u      



def vis_yolo_dagnn(K=3, pretrained=False, **kwargs):
    # n = 6
    model = Yolo_DAGNN(filter_length=K, n_vis_out=kwargs['n_vis_out'])
    return model






class GRNNReluNet(torch.nn.Module):
    def __init__(self, filter_length=3):
        super(GRNNReluNet, self).__init__()
        n_feature = 6
        layer_sizes = [n_feature, 256, 256, 2]
        self.n_layers = len(layer_sizes) - 1
        self.filter_length = filter_length
        self.a = torch.nn.Parameter(torch.ones(filter_length))
        self.b = torch.nn.Parameter(torch.ones(filter_length))

        # hidden layers
        self.hidden = []
        for n in range(0, self.n_layers - 1):
            self.hidden.append(torch.nn.Linear(layer_sizes[n], layer_sizes[n + 1]))
        self.hidden = torch.nn.ModuleList(self.hidden)

        # output layer
        self.predict = torch.nn.Linear(layer_sizes[self.n_layers - 1], layer_sizes[self.n_layers])

    def forward(self, input, a_nets, input_state):
        states = torch.zeros(input.shape[0], 6, self.filter_length).cuda()
        for t in range(self.filter_length):
            a_net = a_nets[:, :, t] # 50 x 50
            x = input[:, :, t] # 50 x 6
            a_net_comm = a_net.unsqueeze(2)
            x = x * a_net_comm # 50 x F  * 50 x 50 x 1  
            x = torch.sum(x, dim=1) # 50 x F
            input_state = input_state.unsqueeze(1) # 50 x 1 x F
            input_state = input_state.repeat(1, a_net.shape[0], 1) # 50 x 50 x F
            input_state = torch.sum(input_state * a_net_comm, dim=1) # 50 x F
            input_state = torch.tanh(x * self.a[t] + input_state  * self.b[t])
            states[:, :, t] = input_state
        u = input_state
        for n in range(0, self.n_layers - 1):
            u = torch.nn.functional.relu((self.hidden[n](u)))
        return states, self.predict(u)

def loc_grnn(K=3, **kwargs):
    # n = 6
    model = GRNNReluNet(filter_length=K)
    return model





def main():
    print('test model')
    k = 3  # timesteps ( filter_length)
    x = np.random.uniform(-10,10, (50, 6, k))
    input_state = np.random.uniform(-10,10, (50, 6))

    a_nets = np.random.choice([0,1], (50, 50, k))
    x = torch.from_numpy(x).float().cuda()
    input_state = torch.from_numpy(input_state).float().cuda()
    a_nets = torch.from_numpy(a_nets).float().cuda()


    model = loc_grnn(K=3, n_vis_out=6).cuda()
    u = model(x, a_nets, input_state)
    print(u[0].shape)

if __name__ == "__main__":
    main()
