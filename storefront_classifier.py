# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 18:03:13 2020

@author: narch
"""

import cv2
import os 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix

import keras
import keras.utils as ku
from keras import backend as K
#Importing the keras Deep learning Neural network layers for building a neural network model
from keras.layers import Dropout, Input, Dense, Activation,GlobalMaxPooling2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, LeakyReLU
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
import random

#Few more imports for Keras NN training process. For evaluation, learning rate, checkpointing the model and decision of an early stop if convergence is achieved
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

trial_name = 'lmf700_32channels_slow_reexcite_p1'
dampen_top = False
feature_highlight = False

images, imgidx = [], []
#To store the type of plant in the imges
# label = []   ### add labels
PATH = "C:\\City Design\\pix2pix-tensorflow\\Datasets\\facades_test_ca1lmf700lmf"
PATHlist = ["C:\\City Design\\pix2pix-tensorflow\\Datasets\\CMP_facade_DB_base\\classifier", 
            "C:\\City Design\\pix2pix-tensorflow\\Datasets\\ECPFD\\classifier",
            "C:\\City Design\\pix2pix-tensorflow\\Datasets\\ParisArtDecoFacadesDataset\\classifier",
            "C:\\City Design\\pix2pix-tensorflow\\Datasets\\CMP_facade_DB_extended\\classifier", 
            "C:\\City Design\\pix2pix-tensorflow\\Datasets\\graz50_facade_dataset\\classifier",
            "C:\\City Design\\pix2pix-tensorflow\\Datasets\\etrims\\classifier", 
            "C:\\City Design\\pix2pix-tensorflow\\Datasets\\labelmefacade\\classifier"]
crop = True
#List of directories where image resides
facade_dir = os.path.join(PATH,'images')
if feature_highlight:
    facade_dir = os.path.join(facade_dir,'featured')
#If you think there is enough main memory on you system please replace the first line with following: for i in range(0,len(plant_dir)):
for fname in os.listdir(facade_dir):
    if fname.split('.')[0][-8:] == '-outputs':
        idx = fname[:-14]
        img = cv2.imread(os.path.join(facade_dir,fname))
        if img is not None:
            imgidx.append(idx)
            if crop:
                h, w = img.shape[:2]
                top = max(h-w,0)
                left = int(max((w-h)/2,0))
                right = w - left
                img = img[top:h,left:right,:]
            img = cv2.resize(img, (256,256))
            if dampen_top:
                for i in range(img.shape[0]):
                    img[i,:,:] = img[i,:,:] * i / 255
            # may need to enlage image size
            img = img/255
            images.append(img)
            # boo_store = fname.split('.')[0]
            # boo_store = int(boo_store.split('_')[1])
            # label.append(boo_store)
sr_labels = pd.Series(index=imgidx)

#for fname in os.listdir(facade_dir):
#    idx = fname.split('_')[0]
#    sr_labels.loc[idx] = fname.split('_')[1][0]
# for label_dir in PATHlist:
for fname in os.listdir(os.path.join(PATH,'images')):
    base, extension = os.path.splitext(fname)
    if base[-7:] == '-inputs':
        idx = base[:-9]
        tg = int(base[-8])
        if tg != 0 and tg != 1:
            print(idx + ' class label error')
        if idx not in sr_labels.index:
            print(idx + ' not in index')
        else:
            sr_labels.loc[idx] = tg
label = sr_labels.values.astype(int)

#Use of label binarizer to conver class names to numeric format.
#This task is required because neural network needs output in numeric format to evaluate the modelfrom sklearn.preprocessing import LabelBinarizer
images = np.array([np.array(img) for img in images])
lb = LabelBinarizer().fit(label)
label = lb.transform(label)
label = ku.to_categorical(label)
pct_1 = sum(label[:,1]) / label.shape[0]
print('percent 1: %s' %pct_1)

#Split the data into training set and testing set for analysis.
#90% of data would be used for training; the rest is for testing
from sklearn.model_selection import train_test_split
trainX, validX, trainY, validY = train_test_split(images, label, test_size=0.2, stratify=label)
        
        
#Convo and dense  layer blocks
# Dense layers set
def dense_set(inp_layer, n, activation, drop_rate=0.5):
    dp = Dropout(drop_rate)(inp_layer)
    dns = Dense(n)(dp)
    bn = BatchNormalization(axis=-1)(dns)
    act = Activation(activation=activation)(bn)
    return act
# Conv. layers set
def conv_layer(feature_batch, feature_map, kernel_size=(3, 3),strides=(1,1), zp_flag=False):
    if zp_flag:
        zp = ZeroPadding2D((1,1))(feature_batch)
    else:
        zp = feature_batch
    conv = Conv2D(filters=feature_map, kernel_size=kernel_size, strides=strides)(zp)
    bn = BatchNormalization(axis=3)(conv)
    act = LeakyReLU(1/10)(bn)
    return act
        

#Building the Deep Learning Neural Network model.
inp_img = Input(shape=(256, 256, 3))
# 256
conv1 = conv_layer(inp_img, 8, kernel_size=(3, 3),strides=(1,1), zp_flag=False)
conv2 = conv_layer(conv1, 8, kernel_size=(5, 5),strides=(2,2), zp_flag=False)
# mp1p = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(conv1)
mp1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv2)
# 62
conv3 = conv_layer(mp1, 16, kernel_size=(3, 3),strides=(1,1), zp_flag=False)
conv4 = conv_layer(conv3, 16, kernel_size=(3, 3),strides=(1,1), zp_flag=False)
# mp2p = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(conv3)
mp2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv4)
# 13
conv7 = conv_layer(mp2, 32, kernel_size=(3, 3),strides=(1,1), zp_flag=False)
conv8 = conv_layer(conv7, 32, kernel_size=(3, 3),strides=(2,2), zp_flag=False)
# mp3p = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv8)
conv9 = conv_layer(conv8, 32, kernel_size=(3, 3),strides=(1,1), zp_flag=False)
mp3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv9)
# 1
# dense layers
flt = Flatten()(mp3)
ds1 = dense_set(flt, 16, activation='tanh')
out = dense_set(flt, 2, activation='softmax')

model = Model(inputs=inp_img, outputs=out)

# The first 50 epochs are used by Adam opt.
# Then 30 epochs are used by SGD opt.

mypotim = Adam(lr=1 * 1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#mypotim = SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
               optimizer=mypotim,
               metrics=['accuracy'])
model.summary()
        

#Defining the learning rate
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
#Defining the early stopping point parameters
earlystop = EarlyStopping(patience=3)
#Create models and save them if they are better.
#This would iteratively create new models and test them
#modelsave = ModelCheckpoint(
#    filepath='model_baseline-', save_best_only=True, verbose=1)
    
class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.validation_data[0])
        y_pred = np.round(y_pred)
        y_true = self.validation_data[1]
        print(confusion_matrix(y_true[:,1], y_pred[:,1]))
        print(trial_name)
    def on_test_begin(self, logs={}):
        pass
    def on_test_end(self, logs={}):
        pass
    def on_test_batch_begin(self, batch, logs={}):
        pass
    def on_test_batch_end(self, batch, logs={}):
        pass
    
class ReExciteCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs['accuracy'] - logs['val_accuracy'] > 0.1:
            session = K.get_session()
            count = 0
            for layer in self.model.layers:
                for v in layer.__dict__:
                    v_arg = getattr(layer,v)
                    if hasattr(v_arg, 'initializer') and random.random() < 0.01:
                        v_arg.initializer.run(session=session)
                        count += 1
            print('%d weights re-excited' %count)
    def on_test_begin(self, logs={}):
        pass
    def on_test_end(self, logs={}):
        pass
    def on_test_batch_begin(self, batch, logs={}):
        pass
    def on_test_batch_end(self, batch, logs={}):
        pass
    
checkpoint = ModelCheckpoint('model_baseline'+trial_name+'.h5', monitor='val_accuracy', save_best_only=True, mode='max')

history = model.fit(
    trainX, trainY, batch_size=32,
    epochs=2000, 
    validation_data=(validX, validY),
    callbacks=[checkpoint, 
               ConfusionMatrixCallback(), ReExciteCallback()]
)

#model.save('sf_cl-'+trial_name+'.h5')


from matplotlib import pyplot as plt
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()

