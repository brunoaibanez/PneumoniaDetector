import numpy as np 
import pydicom
from PIL import Image
import os
import sys
import glob
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input
import csv
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from keras.utils import to_categorical
import skimage.exposure as hist
import tensorflow as tf
import pandas as pd
import pickle
import numpy as np # forlinear algebra
import matplotlib.pyplot as plt #for plotting things
import os
from PIL import Image
import sys
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D
# Keras Libraries
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import np_utils

IMG_SIZE = (224, 224, 1) # [(224, 224), (384, 384), (512, 512), (640, 640)]
BATCH_SIZE = 24 # [1, 8, 16, 24]
DENSE_COUNT = 128 # [32, 64, 128, 256]
DROPOUT = 0.25 # [0, 0.25, 0.5]
LEARN_RATE = 1e-4 # [1e-4, 1e-3, 4e-3]

USE_ATTN = False # [True, False]

ROOT_DIR = '/veu4/usuaris27/pae2018/GitHub/'

x = []
y = []

num_classes = 2
trainshape = 224
SAVEIMAGES = 1;  #1 if save; 0 if open

pkl_file = open(ROOT_DIR + 'created_Data1/dataX_Train_No_Mtdt_All_224.pkl', 'rb')
x_train = pickle.load(pkl_file)
pkl_file.close()
print('x_train', np.array(x_train).shape)
pkl_file = open(ROOT_DIR + 'created_Data1/dataY_Train_No_Mtdt_All_224.pkl', 'rb')
y_train = pickle.load(pkl_file)
print('y_train', np.array(y_train).shape)
pkl_file.close()

pkl_file = open(ROOT_DIR + 'created_Data1/dataX_Test_No_Mtdt_All_224.pkl', 'rb')
x_test = pickle.load(pkl_file)
pkl_file.close()
print('x_test', np.array(x_test).shape)
pkl_file = open(ROOT_DIR + 'created_Data1/dataY_Test_No_Mtdt_All_224.pkl', 'rb')
y_test = pickle.load(pkl_file)
print('y_test', np.array(y_test).shape)
pkl_file.close()
print("Files loaded correctly")

x_train = x_train/255

x_test = x_test/255



x_train_class0= x_train[y_train == 0, ...]
y_train_class0= y_train[y_train == 0, ...]

x_train_class1= x_train[y_train == 1, ...]
y_train_class1= y_train[y_train == 1, ...]


random_permutation= np.random.permutation(x_train_class0.shape[0])
half_length= int(np.rint(len(random_permutation)/2))
x_train_class0= x_train_class0[random_permutation[0:half_length]]
y_train_class0=y_train_class0[random_permutation[0:half_length]]

x_train= np.concatenate((x_train_class0, x_train_class1))
y_train= np.concatenate((y_train_class0, y_train_class1))


print('Class 0 Normal  shape', x_train[y_train == 0, ...].shape)
print('Class 1 Lung Opacity  shape', x_train[y_train == 1, ...].shape)
print('Class 2 No Lung Opacity/Not Normal shape', x_train[y_train == 2, ...].shape)

print('Test Class 0 Normal  shape', x_test[ y_test == 0, ...].shape)
print('Test Class 1 Lung Opacity  shape', x_test[ y_test == 1, ...].shape)
print('Test Class 2 No Lung Opacity/Not Normal shape', x_test[ y_test == 2, ...].shape)


#Split entre Validation i Test del total
x_train2, x_val, y_train2, y_val = train_test_split(x_train, y_train, test_size=0.15, shuffle=True)

 
x_train2_class1= x_train2[y_train2 == 1, ...]
print('x_train class 1 shape', x_train_class1.shape)

x_train2_class0= x_train2[y_train2 == 0, ...]
y_train2_class0= y_train2[y_train2 == 0, ...]
print('x_train class 0 shape', x_train_class0.shape)






#Tractament a numpy array normalitzat
y_test_no_categorical= y_test
y_test = np_utils.to_categorical(y_test, num_classes)
y_val = np_utils.to_categorical(y_val, num_classes)
y_train2= np_utils.to_categorical(y_train2, num_classes)

#x_train = np.array(x_train)
x_train2 = x_train2.reshape([-1,trainshape, trainshape,1])

#x_val = np.array(x_val)
x_val = x_val.reshape([-1,trainshape, trainshape,1])

#x_test = np.array(x_test)
x_test = x_test.reshape([-1,trainshape, trainshape,1])



print('Train_shape: ',x_train2.shape, y_train2.shape)
print('Validation_shape: ',x_val.shape, y_val.shape)
print('Test_shape: ',x_test.shape, y_test.shape)


def create_downsample(channels, inputs):
    x = keras.layers.BatchNormalization(momentum=0.9)(inputs)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, (1,1), padding='same', use_bias=False)(x)
    x = keras.layers.MaxPool2D(2)(x)
    return x

def create_resblock(channels, inputs):
    x = keras.layers.BatchNormalization(momentum=0.9)(inputs)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, (3,3), padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, (3,3), padding='same', use_bias=False)(x)
    return keras.layers.add([x, inputs])

def create_network(input_size, channels, n_blocks=2, depth=4):
    # input
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = keras.layers.Conv2D(channels, (3,3), padding='same', use_bias=False)(inputs)
    # residual blocks
    for d in range(depth):
        channels = channels * 2
        x = create_downsample(channels, x)
        for b in range(n_blocks):
            x = create_resblock(channels, x)
    # output
    x = keras.layers.BatchNormalization(momentum=0.99)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(1, (1,1), activation='relu')(x)
    x = keras.layers.UpSampling2D(2**depth)(x)

    x = Dense(256, activation = 'linear')(x)     
    x = keras.layers.BatchNormalization(momentum=0.99)(x)
    x = Dense(256, activation = 'relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation = 'linear')(x)     
    x = keras.layers.BatchNormalization(momentum=0.99)(x)
    x = Dense(128, activation = 'relu')(x)
    x = Dropout(0.4)(x)

    x = Dense(64, activation = 'linear')(x)     
    x = keras.layers.BatchNormalization(momentum=0.99)(x)
    x = Dense(64, activation = 'relu')(x)
    x = Dropout(0.4)(x)

    x = Dense(32, activation = 'linear')(x)     
    x = keras.layers.BatchNormalization(momentum=0.99)(x)
    x = Dense(32, activation = 'relu')(x)        
    outputs = Dense(2, activation = 'softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model



model = create_network(input_size=224, channels=32, n_blocks=2, depth=4)

model.summary()

model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format(ROOT_DIR + 'dataPretrainedModels/classificacio_pneumonia_or_not_Model3')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.6, 
                                   patience=4, verbose=1, mode='auto', 
                                   epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min",
                      patience=5) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]

history= model.fit(x_train2, y_train2, validation_data=(x_val, y_val), epochs=25, verbose=1, callbacks=callbacks_list, 
                                                    batch_size=32)

model.load_weights(weight_path)
model.save(ROOT_DIR + 'dataPretrainedModels/classificacio_pneumonia_or_not_Model3.h5')

test_accu = model.evaluate(x_test, y_test, verbose=1)
print('The testing accuracy is :',test_accu[1]*100, '%')



print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')


