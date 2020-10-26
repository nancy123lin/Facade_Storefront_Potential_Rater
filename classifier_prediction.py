# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:27:33 2020

@author: narch
"""

# for loading model
from keras.models import load_model
# for processing predicting data
import cv2
import os 
import pandas as pd
import numpy as np


model = load_model('model_baselinelmf350_32channels_slow_reexcite_fullcolour_p1.h5')


# load predicting data
images = []
imgnms = []
PATH = "C:\\City Design\\pix2pix-tensorflow\\facades_storefront01"
facade_dir = os.path.join(PATH,'test')

# If you think there is enough main memory on you system please replace the first line with following: for i in range(0,len(plant_dir)):
for fname in os.listdir(facade_dir):
    img = cv2.imread(os.path.join(facade_dir,fname))
    if img is not None:
        h, w = img.shape[:2]
        img = img[:,int(w/2):w,:]
        img = cv2.resize(img, (256,256))
        # may need to enlage image size
        img = img/255
        images.append(img)
        imgnms.append(int(fname.split('.')[0].split('_')[0]))
        
y_pred = model.predict([images])
df_result = pd.DataFrame(y_pred, index=imgnms, columns=[0,1])

print(df_result)