'''
Note: We will not release code for Data Augmentation here.
      Only provide function below to generate heatmap.
'''


import numpy as np
import math

# Compute gaussian kernel
# (c_x,c_y) is the center of hand bbox, scale is the length of hand bbox side
def CenterGaussianHeatMap(img_height, img_width, c_x, c_y, scale_x, scale_y):
    gaussian_map = np.zeros([img_height, img_width],dtype=np.uint8)
    
    mask_len = min(scale_x,scale_y)
    x,y =np.meshgrid(np.linspace(-1,1,mask_len),np.linspace(-1,1,mask_len))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 0.5, 0.0
    mask = np.exp(-((d-mu)**2 / (2.0* sigma**2)))
    
    x_left = int(math.ceil(c_x - mask_len/2.0))
    x_right = int(math.ceil(c_x + mask_len / 2.0))
    y_down = int(math.ceil(c_y - mask_len / 2.0))
    y_up = int(math.ceil(c_y + mask_len / 2.0))
    
    if x_right > gaussian_map.shape[1] or y_up > gaussian_map.shape[0]:
      return None
    else:
      gaussian_map[y_down:y_up,x_left:x_right] = mask * 255
      return gaussian_map
    
  