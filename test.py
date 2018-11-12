import mxnet as mx
from mxnet import autograd, contrib, gluon, image, init, nd
from mxnet.gluon import loss as gloss, nn
from resnet import *
import os,argparse,time,logging
from mxnet.gluon.block import HybridBlock
import numpy as np
import cv2
import random
from collections import namedtuple
    
ctx = mx.cpu()
resize_w, resize_h = 64,64
channel = 1
batch_size = 1

symnet = mx.symbol.load('/model/model_50-symbol.json')
mod = mx.mod.Module(symbol=symnet, context=ctx)
mod.bind(data_shapes=[('data', (batch_size, 1, 64, 64))])
mod.load_params('/model/model_50-0000.params')
Batch = namedtuple('Batch', ['data'])

image_names = os.listdir('/test_images/')
for image_name in image_names:
  image_file = '/test_images/' + image_name
  img = cv2.imread(image_file, 0)

  if img is None:
    print('Fail to open image:', image_file)
    continue

  img = cv2.resize(img, (64,64))
  img = img[np.newaxis]
  img = img[np.newaxis]

  # net forward
  mod.forward(Batch([mx.nd.array(img)]),is_train=False)
  # get result
  pred = mod.get_outputs()
  pred = pred[0].asnumpy()[0][0]*255

  cv2.imwrite('/test_results/'+image_name,pred)