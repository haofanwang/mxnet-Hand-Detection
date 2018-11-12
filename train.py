# -*- coding: utf-8 -*-

'''
Training
'''

import mxnet as mx
from mxnet import autograd, contrib, gluon, image, init, nd
from mxnet.gluon import loss as gloss, nn
from resnet import *
import os,argparse,time,logging
from mxnet.gluon.block import HybridBlock
import numpy as np
import cv2
import random

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Training.')
    parser.add_argument('--lst', dest='lst_path', help='Path to idx file',
                        default="train.lst", type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size',
                        default=256, type=int)
    parser.add_argument('--epoch', dest='epoch_num', help='epoch num',
                        default=51, type=int)
    parser.add_argument('--dev', dest='dev', help='0:cpu,1:gpu',
                        default=1, type=int)
    args = parser.parse_args()
    return args
    
# Iterator for image-heatmap pair
def read_data(lst):
    with open(lst) as fin:
        labels = []
        datas = []
        count = 0
        for line in iter(fin.readline, ''):
            line = line.strip().split('\t')
            label = np.array(cv2.imread(line[0],0))
            data = np.array(cv2.imread(line[1],0))
            labels.append(label)
            datas.append(data)
            count += 1
            if count % 500 == 0:
              print(count)
    return (datas,labels)

def create_pairs(x,label):
    pairs = []
    for d in range(len(x)):
            pairs.append([x[d],label[d]])
    return np.array(pairs)

class PairDataIter(mx.io.DataIter):
    def __init__(self, batch_size, lst_path, mode='train'):
        super(PairDataIter, self).__init__()
        self.lst_path = lst_path
        self.batch_size = batch_size
        self.provide_label = [('label',(batch_size, 64, 64))]
        self.provide_data = [('data', (batch_size, 64, 64))]
        if mode == 'train':
            (self.datas,self.labels) = read_data(self.lst_path)

        self.pairs = create_pairs(self.datas,self.labels)
        self.end_idx = len(self.pairs) // self.batch_size
        self.count = 0

    def reset(self):
        self.count = 0
        indexes = range(len(self.pairs))
        random.shuffle(indexes)
        self.pairs = self.pairs[indexes]

    def next(self):
        if self.count == self.end_idx:
            raise StopIteration
        pair_data = self.pairs[self.count * self.batch_size:(self.count + 1) * self.batch_size]
        self.count += 1

        return mx.io.DataBatch(
            data=[mx.nd.array(pair_data[:,0]).reshape((self.batch_size,1,64,64))],
            label=[mx.nd.array(pair_data[:, 1]).reshape((self.batch_size,1,64,64))],
            provide_data=self.provide_data,
            provide_label=self.provide_label
        )

if __name__ == '__main__':
    args = parse_args()
    train_lst = args.lst_path
    batch_size = args.batch_size
    
    resize_w, resize_h = 64, 64
    channel = 1
    begin_epoch = 1
    
    # Loading
    # Note: you could take a while to load all data
    train_data = PairDataIter(lst_path=train_lst,batch_size=batch_size)

    # Network
    net = FCN()

    # Init
    epoch_num = args.epoch_num
    
    dev = args.dev
    if dev == -1:
      ctx = mx.cpu()
    else:
      ctx = mx.gpu(dev)
      
    net.initialize(init=init.Xavier(), ctx=ctx)
    net.hybridize()
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(net.collect_params(),
                            'adam', {'learning_rate': 0.0001, 'wd': 5e-4})

    lr_period = int(epoch_num / 5)
    lr_decay = 0.1
    NEAR_0 = 1e-10
    
    reg_loss = gloss.L1Loss()
    
    # Training
    for epoch in range(epoch_num):
        train_data.reset()
        tic = time.time()
        i = 0

        # Dynamic lr
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        
        for i, batch in enumerate(train_data):
            with autograd.record():
              pred = net(batch.data[0].as_in_context(ctx))
              label = batch.label[0].as_in_context(ctx)
              loss = reg_loss(pred,label)
              
            # backpropagate
            loss.backward()
            trainer.step(batch_size)
            
            if i % 100 == 0:
                print('epoch %d, batch: %d,Losses: %.4f,time: %.4f '
                      % (epoch, i, loss.mean().asscalar(), time.time() - tic))
            i += 1
            
        # Save model
        if epoch % 10 == 0:
          net.export('/model/model_'+str(epoch))
