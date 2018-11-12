# -*- coding: utf-8 -*-

'''
Resnet block
'''

from mxnet import nd, gluon, init
from mxnet.gluon import loss as gloss, nn

class Residual(nn.HybridBlock):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def hybrid_forward(self, F, X, *args, **kwargs):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.HybridSequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk

# 64x64 --> 64x64
def FCN():
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Conv2D(48,kernel_size=3,strides=2,padding=1),nn.BatchNorm(),nn.Activation('relu'))
        net.add(nn.Conv2D(48,kernel_size=1,strides=1,padding=0),nn.BatchNorm(),nn.Activation('relu'))
        net.add(nn.Conv2D(96,kernel_size=3,strides=2,padding=1),nn.BatchNorm(),nn.Activation('relu'))
        net.add(nn.Conv2D(96,kernel_size=1,strides=1,padding=0),nn.BatchNorm(),nn.Activation('relu'))
        net.add(nn.Conv2D(256,kernel_size=3,strides=2,padding=0),nn.BatchNorm(),nn.Activation('relu'))
        net.add(nn.Conv2DTranspose(256,kernel_size=3,strides=1,padding=0),nn.BatchNorm(),nn.Activation('relu'))
        net.add(nn.Conv2DTranspose(128, kernel_size=3, strides=1, padding=0), nn.BatchNorm(), nn.Activation('relu'))
        net.add(nn.Conv2DTranspose(96, kernel_size=6, strides=1, padding=0), nn.BatchNorm(), nn.Activation('relu'))
        net.add(nn.Conv2DTranspose(1, kernel_size=8, strides=4, padding=2), nn.BatchNorm(), nn.Activation('relu'))
    return net