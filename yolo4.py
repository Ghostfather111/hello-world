import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
from CSPDarknet import *
import cv2

# CBL的构建
def conv2d(filter_in, filter_out, kernel_size, stride=1):
    '''
    ps:此conv2d实际上包括：卷积层+batch Normal+leaky relu激活函数
    :param filter_in: 输入通道数
    :param filter_out:输出通道数
    :param kernel_size:卷积核大小
    :param stride:步长
    :return:
    '''
    pad = (kernel_size - 1) // 2 if kernel_size else 0 #当卷积核为3时，填充为1，其余为0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)), #bn层只需要输入通道数作为参数
        ("relu", nn.LeakyReLU(0.1)), #参数为<0时的斜率
    ]))


#---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])
        '''
        SPP采用3个最大池化后拼接，池化的核的大小为5、9、13
        '''

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]] #[::-1]表示倒序
        features = torch.cat(features + [x], dim=1)

        return features

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        '''
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')

        )
        '''
        scale_factor=2表示输出尺寸为输入的2倍
        '''

    def forward(self, x,):
        x = self.upsample(x)
        return x


#---------------------------------------------------#
#   三次卷积块
#   [512, 1024]
#---------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    '''
    :param filters_list: 中间及最后输出的通道数
    :param in_filters: 输入通道数
    :return: m为Sequential类型
    '''
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m



#---------------------------------------------------#
#   五次卷积块
#---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m



#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    '''
    包含CBL和单独的一次卷积
    :param filters_list: 中间通道及输出通道
    :param in_filters: 输入通道
    :return:
    '''
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()
        #  backbone
        self.backbone = darknet53(None) #构造身体部分

        self.conv1 = make_three_conv([512,1024],1024) #105-107层
        self.SPP = SpatialPyramidPooling() #108-113层？
        self.conv2 = make_three_conv([512,1024],2048) #114-116层，输出通道数为512

        self.upsample1 = Upsample(512,256) #第117-118层
        self.conv_for_P4 = conv2d(512,256,1) #120层
        self.make_five_conv1 = make_five_conv([256, 512],512) #122-126层

        self.upsample2 = Upsample(256,128) #127-128层
        self.conv_for_P3 = conv2d(256,128,1) #129层
        self.make_five_conv2 = make_five_conv([128, 256],256) #132-136层
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        # 4+1+num_classes
        final_out_filter2 = num_anchors * (5 + num_classes) #一个头有3个anchors，num_anchors=3
        self.yolo_head3 = yolo_head([256, final_out_filter2],128) #137-138层

        self.down_sample1 = conv2d(128,256,3,stride=2) #141层
        self.make_five_conv3 = make_five_conv([256, 512],512) #143-147层
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter1 =  num_anchors * (5 + num_classes)
        self.yolo_head2 = yolo_head([512, final_out_filter1],256) #148-149层


        self.down_sample2 = conv2d(256,512,3,stride=2) #152层
        self.make_five_conv4 = make_five_conv([512, 1024],1024) #154-158层
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter0 =  num_anchors * (5 + num_classes)
        self.yolo_head1 = yolo_head([1024, final_out_filter0],512) #159-160层



    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)
        '''
        x0：104层的输出
        x1：85层的输出
        x2：54层的输出
        '''

        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        P5 = self.conv2(P5)

        P5_upsample = self.upsample1(P5)
        P4 = self.conv_for_P4(x1)
        P4 = torch.cat([P4,P5_upsample],axis=1)
        P4 = self.make_five_conv1(P4)

        P4_upsample = self.upsample2(P4)
        P3 = self.conv_for_P3(x2)
        P3 = torch.cat([P3,P4_upsample],axis=1)
        P3 = self.make_five_conv2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample,P4],axis=1)
        P4 = self.make_five_conv3(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample,P5],axis=1)
        P5 = self.make_five_conv4(P5)

        out2 = self.yolo_head3(P3)
        out1 = self.yolo_head2(P4)
        out0 = self.yolo_head1(P5)

        return out0, out1, out2
        #out0:19*19
        #out1:38*38
        #out2:76*76



if __name__ == '__main__':
    model = YoloBody(3, 1) #3表示每个头有3个anchors
    load_model_pth(model, '/media/xwd/HDD/yolov4/yolov4-pytorch1/chk/Epoch_400_Loss_33.9423.pth')

