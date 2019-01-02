import shutil, os
import numpy as np
import os
import csv
import PIL
from PIL import Image
import pickle
from skimage.transform import resize
import shutil, os 

trainshape = 224
classe = 0
x = []
y = []

ROOT_DIR= '/veu4/usuaris27/pae2018/GitHub/'
ROOT_DIR_IMAGES= '/veu4/usuaris27/pae2018/GitHub/images/'

cont0 = 0
cont1 = 0
num = 0
im_files =  os.listdir(ROOT_DIR_IMAGES)

imatges_healthy_train= 0
imatges_No_Healthy_train= 0

healthy_train_images= 0
helthy_test_images= 0
no_healthy_train_images= 0
no_healthy_test_images= 0

for f in im_files:
    if f.endswith("png"):
        with open(ROOT_DIR+ 'dataCSV/Data_Entry_2017.csv') as File:
            filecsv = csv.reader(File)
            filepng = ROOT_DIR_IMAGES + os.path.splitext(f)[0] + os.path.splitext(f)[1]
            flag = 0
            exit = 0
            for row in filecsv:
                if row[0] == f and exit == 0:
                    if row[1] == 'No Finding':
                        if imatges_healthy_train < 32810:
                            shutil.copy(filepng, ROOT_DIR+'DataBase/Train/Healthy2')                            
                            imatges_healthy_train +=1
                            healthy_train_images+= 1
                            print('healthy_train_images',healthy_train_images)

                        if (imatges_healthy_train >= 32810) and (imatges_healthy_train < 37810):
                            shutil.copy(filepng, ROOT_DIR+'DataBase/Test/Healthy2')
                            imatges_healthy_train +=1
                            helthy_test_images +=1
                            print('helthy_test_images', helthy_test_images)


                    else:
                        if imatges_No_Healthy_train < 32810:
                            shutil.copy(filepng, ROOT_DIR+'DataBase/Train/Not_Healthy2' )
                            imatges_No_Healthy_train +=1
                            no_healthy_train_images+= 1
                            print('no_healthy_train_images', no_healthy_train_images)
                        
                        if (imatges_No_Healthy_train >= 32810) and (imatges_No_Healthy_train < 37810):
                            shutil.copy(filepng, ROOT_DIR+'DataBase/Test/Not_Healthy2' )
                            imatges_No_Healthy_train +=1
                            no_healthy_test_images +=1
                            print('no_healthy_test_images', no_healthy_test_images)

                    
