import pydicom
import os
import sys
import glob
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input
import csv
import numpy as np # forlinear algebra
import matplotlib.pyplot as plt #for plotting things
import os
from PIL import Image
import sys
from skimage import morphology
from skimage import feature
from skimage import measure
from skimage import util
from skimage import transform
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import pickle

ROOT_DIR= '/veu4/usuaris27/pae2018/GitHub/'
ROOT_DIR_DCM= '/veu4/usuaris27/pae2018/GitHub/DCM/'
ROOT_DIR_CSV= '/veu4/usuaris27/pae2018/GitHub/dataCSV/'

x = []
y = []
classe = 0
images_done=0
trainshape = 224

#Funcions data preprocessing



def processament_imatge(imatge):
    imatge2 = imatge.astype(float)
    imatge3_resize = resize(imatge2, (trainshape, trainshape), mode='reflect')
    return imatge3_resize

def mostrar_imatge(imatge_a_mostrar):
    plt.imshow(imatge_a_mostrar, cmap=plt.cm.bone)
    plt.show()  


for imagedcm in os.listdir(ROOT_DIR + "./DCM"):
    extension = os.path.splitext(imagedcm)[1]
    name=os.path.splitext(imagedcm)[0]
    if extension == ".dcm":
        filedcm=ROOT_DIR_DCM + imagedcm
        ds = pydicom.dcmread(filedcm)
        image_2d = ds.pixel_array
        
        with open(ROOT_DIR_CSV+'stage_1_detailed_class_info.csv') as File:
            filecsv = csv.reader(File)
            exit = 0
            for row in filecsv:
                if row[0] == name and exit == 0:

                    if row[1] == 'No Lung Opacity / Not Normal':
                        classe = 2
                    
                    elif row[1] == 'Normal':
                        classe = 0

                    elif row[1] == 'Lung Opacity':
                        classe = 1
                    
                    mean_black_pixels= np.mean(image_2d == 0)

                    if mean_black_pixels < 0.28:
                        image_2d_processed=processament_imatge(image_2d)
                        x.append(image_2d_processed)
                        y.append(classe)
                        images_done+=1
                        print(images_done)

                    exit = 1

y=np.array(y)
x=np.array(x)



print('Class 0 Normal  shape', x[ y == 0, ...].shape)
print('Class 1 Lung Opacity  shape', x[ y == 1, ...].shape)
print('Class 2 No Lung Opacity/Not Normal shape', x[ y == 2, ...].shape)

#Separem cada classe 
x_class0=x[y==0, ...]

x_class1=x[y==1, ...]
y_class1=y[y==1, ...]

x_class2=x[y==2, ...]


#Permutacions dels elements de la classe2 de forma random
random_permutation2 = np.random.permutation(x_class2.shape[0])
random_permutation3 = np.random.permutation(x_class0.shape[0])


#Agafem la meitat dels elements  classe 2 de forma aleatoria (Meitat dels indexs randoms)
x_class2_half_rand= x_class2[random_permutation2[0:int(np.rint(len(random_permutation2)/2))]]
x_class0_half_rand= x_class0[random_permutation3[0:(1000+int(np.rint(len(random_permutation3)/2)))]]

print('x_class2_half_rand', x_class2_half_rand.shape)
print('x_class0_half_rand', x_class0_half_rand.shape)


#Ajuntem en una sola classe 1 i meitat random classe 2
x_class0_2=np.concatenate([x_class0_half_rand, x_class2_half_rand], axis=0)



#Assignem a aquests nous valors la classe 0
y_class0_2=np.full(x_class0_2.shape[0], 0)
print('y_class0_2 shape', y_class0_2.shape)

#Finalment tornem a restablir els valors x i y ara pero amb 2 classes enlloc de 3
x=np.concatenate([x_class0_2, x_class1], axis=0)
y=np.concatenate([y_class0_2, y_class1], axis=0)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, shuffle=True)

print('Class 0 Normal  shape', x_train[y_train == 0, ...].shape)
print('Class 1 Lung Opacity  shape', x_train[y_train == 1, ...].shape)
print('Class 2 No Lung Opacity/Not Normal shape', x_train[y_train == 2, ...].shape)


output = open(ROOT_DIR+ 'created_Data1/dataX_Train_No_Mtdt_All_224.pkl', 'wb')
pickle.dump(x, output,  protocol=4)
output.close()

output = open(ROOT_DIR+ 'created_Data1/dataY_Train_No_Mtdt_All_224.pkl', 'wb')
pickle.dump(y, output,  protocol=4)
output.close()

output = open(ROOT_DIR+ 'created_Data1/dataX_Test_No_Mtdt_All_224.pkl', 'wb')
pickle.dump(x_test, output,  protocol=4)
output.close()

output = open(ROOT_DIR+ 'created_Data1/dataY_Test_No_Mtdt_All_224.pkl', 'wb')
pickle.dump(y_test, output,  protocol=4)
output.close()

