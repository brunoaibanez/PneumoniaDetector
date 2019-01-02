from keras.models import load_model
import numpy as np 
import pydicom
from PIL import Image
import os
import sys
import glob
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator #Data augmentation and preprocessing
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from keras.utils import to_categorical
import keras
import skimage.exposure as hist
from keras.utils import np_utils
import pandas as pd
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
import keras.backend as K
import pickle
import numpy as np
import warnings
import os
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import SGD

from keras.layers import Input
from keras import layers
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D, Input, BatchNormalization, GlobalAveragePooling2D

ROOT_DIR= '/veu4/usuaris27/pae2018/GitHub/'

def mostrar_imatge(imatge_a_mostrar):
    plt.imshow(imatge_a_mostrar, cmap=plt.cm.bone)
    plt.show()  

classes = ["Healthy", "Not_Healthy"]

train_datagen=ImageDataGenerator(samplewise_center=False, 
                              samplewise_std_normalization=False, 
                              horizontal_flip = True, 
                              vertical_flip = False,
                              validation_split= 0.1, 
                              height_shift_range = 0.05, 
                              width_shift_range = 0.02, 
                              rotation_range = 3, 
                              shear_range = 0.01,
                              fill_mode = 'nearest',
                              zoom_range = 0.05,
                              preprocessing_function=preprocess_input)

test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator= train_datagen.flow_from_directory(
    directory= ROOT_DIR+ "DataBase/Train", 
    subset="training", 
    batch_size=16, 
    shuffle=True, 
    class_mode="categorical",
    classes = classes,
    color_mode= 'rgb',
    target_size=(224,224))


valid_generator= train_datagen.flow_from_directory(
    directory=ROOT_DIR+"DataBase/Train",
    subset="validation",
    batch_size=16,
    shuffle=True,
    class_mode="categorical",
    color_mode= 'rgb',
    classes = classes,
    target_size=(224,224))

test_generator= test_datagen.flow_from_directory(
    directory=ROOT_DIR+"DataBase/Test",
    batch_size=16,
    class_mode="categorical",
    color_mode= 'rgb',
    classes = classes,
    target_size=(224,224))

model_Dense = InceptionResNetV2(include_top= False,   weights = 'imagenet', pooling= 'avg' )
model_Dense.trainable = True

model = Sequential()
model.add(model_Dense)

model.add(Dropout(0.2))
model.add(Dense(activation = 'relu', units = 64))
model.add(Dropout(0.1))
model.add(Dense(2, activation = 'softmax'))


model.summary()



from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format(ROOT_DIR + 'dataPretrainedModels/InceptionResNetV2')


checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                   patience=3, verbose=1, mode='auto', 
                                   epsilon=0.0001, cooldown=5, min_lr=0.00001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min",
                      patience=12) 
callbacks_list = [checkpoint, early, reduceLROnPlat]



history = model.fit_generator(generator= train_generator,
                           epochs=20,
                           callbacks=callbacks_list,
                           shuffle=True,
                           steps_per_epoch= 3500,
                           validation_data = valid_generator,
                           verbose= 1)

model.load_weights(weight_path)


model.save_weights(weight_path)
model.save(ROOT_DIR + 'dataPretrainedModels/InceptionResNetV2.h5')

val_accu= model.evaluate_generator(generator= test_generator , verbose= 1)
print('The testing accuracy is :', val_accu[1]*100, '%')

