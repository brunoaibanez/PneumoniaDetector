import pandas as pd
import numpy as np
import os
import csv
import PIL
from PIL import Image
import pickle
from skimage.transform import resize
from scipy.spatial import distance
from keras.models import load_model
from keras.models import Model
import pydicom
from skimage import data, img_as_float
from skimage.measure import compare_ssim as compare_sim
import imageio
import scipy
from keras.applications.densenet import preprocess_input



ROOT_DIR = '/veu4/usuaris27/pae2018/GitHub/'

df= pd.read_csv(ROOT_DIR + 'dataCSV/Data_Entry_2017.csv')

rows_done=0
vector_distancies=[]
images_path = []
classes = []
genders = []
predictions= []
trainshape= 224
not_guradar=1
finding_array= []

base_model2 = load_model(ROOT_DIR + 'dataPretrainedModels/DenseNet_Definitiu_Model.h5')
base_model2.load_weights(ROOT_DIR + 'dataPretrainedModels/DenseNet_Definitiu_Model_weights.best.hdf5')
loaded_model2 = Model(input=base_model2.input, output=base_model2.get_layer('dense_4').output)

def obrir_imatge(path_imatge):
    path_imatge= ROOT_DIR +  path_imatge
    similar_image= scipy.misc.imread(path_imatge, mode= 'L')
    similar_image= similar_image.astype(float)
    similar_imag2= similar_image/255
    return(similar_imag2)


def processament_imatge(file_png):
    similar_image= scipy.misc.imread(file_png, mode= 'L')
    img_similar= resize(similar_image, (trainshape, trainshape), mode = 'reflect')
    image = np.stack((img_similar,) * 3, -1)
    img_train= image.reshape([1, trainshape, trainshape, 3])
    img_train= preprocess_input(img_train)
    prediction= loaded_model2.predict(img_train)
    prediction= np.array(prediction)
    return prediction

if not_guradar==0:
    for index, row in df.iterrows():

        image_path= 'images/'+ str(row["Image Index"]) #Folder where the x_ray 2nd DB are llocated  must be images
        finding= str(row["Finding Labels"])
        
        if finding == 'No Finding':
            classe= 0
        else:
            classe= 1

        if rows_done== 0:
            prediction= processament_imatge(image_path)
            prediction= prediction.reshape([1, 64])
            predictions= prediction
        
        else:
            prediction= processament_imatge(image_path)
            prediction= prediction.reshape([1, 64])
            predictions= np.vstack((predictions, prediction))

        gender=str(row["Patient Gender"])
        images_path.append(image_path)
        classes.append(classe)
        genders.append(gender)
        finding_array.append(finding)

        rows_done+=1
        print('rows done', rows_done)
        print('predictions shape', predictions.shape)

    images_path= np.array(images_path)
    classes= np.array(classes)
    genders= np.array(genders)
    predictions= np.array(predictions)
    finding_array=np.array(finding_array)

    output1 = open(ROOT_DIR +'dataSimilarImages/images_path_CNN2.pkl', 'wb')
    pickle.dump(images_path, output1,  protocol=4)
    output1.close()

    output2 = open(ROOT_DIR +'dataSimilarImages/classes_CNN2.pkl', 'wb')
    pickle.dump(classes, output2,  protocol=4)
    output2.close()

    output3 = open(ROOT_DIR +'dataSimilarImages/genders_CNN2.pkl', 'wb')
    pickle.dump(genders, output3,  protocol=4)
    output3.close()

    output4 = open(ROOT_DIR +'dataSimilarImages/predictions_CNN2.pkl', 'wb')
    pickle.dump(predictions, output4,  protocol=4)
    output4.close()

    output5 = open(ROOT_DIR +'dataSimilarImages/finding_array_CNN2.pkl', 'wb')
    pickle.dump(finding_array, output5,  protocol=4)
    output5.close()



    print('images_path', images_path.shape)
    print('classes', classes.shape)
    print('genders', genders.shape)
    print('predictions', predictions.shape)
    print('finding_array', finding_array.shape)


def similar_images(imatge_test, classe_test, sexe_test): #Imatge format igual obrir imatge(array a secas)

    output1 = open(ROOT_DIR +'dataSimilarImages/images_path_CNN2.pkl', 'rb')
    images_path = pickle.load(output1)
    output1.close()

    output2 = open(ROOT_DIR +'dataSimilarImages/classes_CNN2.pkl', 'rb')
    classes = pickle.load(output2)
    output2.close()

    output3 = open(ROOT_DIR +'dataSimilarImages/genders_CNN2.pkl', 'rb')
    genders = pickle.load(output3)
    output3.close()

    output4 = open(ROOT_DIR +'dataSimilarImages/predictions_CNN2.pkl', 'rb')
    predictions = pickle.load(output4)
    output4.close()

    output5 = open(ROOT_DIR +'dataSimilarImages/finding_array_CNN2.pkl', 'rb')
    finding_array = pickle.load(output5)
    output5.close()
    
    print('Predictions shape:', predictions.shape)

    imatge_grayscale= imatge_test.astype(float)
    imatge_grayscale= imatge_grayscale/255

    img_test= resize(imatge_test, (trainshape, trainshape), mode = 'reflect')
    image = np.stack((img_test,) * 3, -1)
    img_train= image.reshape([1, trainshape, trainshape, 3])
    img_train= preprocess_input(img_train)
    prediction_test= loaded_model2.predict(img_train)


    same_class = np.equal(classe_test, classes)
    same_class= np.array(same_class)
    print('same class shape', same_class.shape)
    print('same class array', same_class[0:10])


    genders=np.array(genders)
    same_gender= np.core.defchararray.equal(sexe_test, genders)
    same_gender= np.array(same_gender)

    print('same_gender shape', same_gender.shape)
    print('same_gender array', same_gender[0:10])

    same_class_same_gender= np.logical_and(same_class, same_gender)
    same_class_same_gender= np.array(same_class_same_gender)
    
    print(type(same_class_same_gender))
    print('same_class_same_gender shape', same_class_same_gender.shape)
    print('same_class_same_gender', same_class_same_gender[0:10])
    #print('Total imatges a comparar:', np.sum(same_class_same_gender))


    images_path_similars= images_path[same_class_same_gender]
    predictions_similars= predictions[same_class_same_gender]
    finding_array_similars= finding_array[same_class_same_gender]
    print('images_path_similars', images_path_similars.shape)
    print('predictions_similars', predictions_similars.shape )
    print('finding_array_similars', finding_array_similars.shape)

    vector_distancies=[]

    for i in range(predictions_similars.shape[0]):
        distancia= distance.canberra(predictions_similars[i,:], prediction_test)
        vector_distancies.append(distancia)

    vector_distancies= np.array(vector_distancies)
    vector_distancies=vector_distancies.reshape([-1,1])
    print('Vector_distancies shape', vector_distancies.shape)
    

    similar_image_path_list=[]
    finding_array_similars_list=[]
    ssim_values_array=[]    
    images_done=0

    while images_done < 3 :

        minimum_position= np.argmin(vector_distancies)

        if vector_distancies[minimum_position] <= 0.03:
            vector_distancies= np.delete(vector_distancies, [minimum_position], axis= 0)
            images_path_similars= np.delete(images_path_similars, [minimum_position])
            finding_array_similars= np.delete(finding_array_similars, [minimum_position])

        else:            
            print('Min distance value position', minimum_position , ': ' ,vector_distancies[minimum_position])
            
            similar_image_path = images_path_similars[minimum_position]
            finding_similar = finding_array_similars[minimum_position]
            temp= obrir_imatge(similar_image_path)
            print('similar_image_path', type(similar_image_path))
            print('obrir_imatge(similar_image_path)', obrir_imatge(similar_image_path).shape)
            print('imatge_test', type(imatge_test))

            ssim_value= compare_sim(imatge_grayscale, temp) #float /255
            
            if (images_done != 0 and similar_image_path_list[-1] == similar_image_path) or ssim_value < 0.03:
                vector_distancies= np.delete(vector_distancies, [minimum_position], axis= 0)
                images_path_similars= np.delete(images_path_similars, [minimum_position])
                finding_array_similars= np.delete(finding_array_similars, [minimum_position])
            
            else:
                ssim_values_array.append(ssim_value)             
                similar_image_path_list.append(similar_image_path)
                finding_array_similars_list.append(finding_similar)
                vector_distancies= np.delete(vector_distancies, [minimum_position], axis= 0)
                images_path_similars= np.delete(images_path_similars, [minimum_position] )
                finding_array_similars= np.delete(finding_array_similars, [minimum_position])

                images_done+=1
    
    
    max_ssim_position= np.argmax(ssim_values_array)  #retorna index max_ssim i ja ho tindriem aixo                      
    print(similar_image_path_list[0], 'Finding:', finding_array_similars_list[0])
    print(similar_image_path_list[1], 'Finding:', finding_array_similars_list[1])
    print(similar_image_path_list[2], 'Finding:', finding_array_similars_list[2])
    most_similar_image= obrir_imatge(similar_image_path_list[max_ssim_position])
    finding_most_similar= finding_array_similars_list[max_ssim_position]
    print('Most similar', similar_image_path_list[max_ssim_position], 'Finding:', finding_array_similars_list[max_ssim_position])
    #similar_image1= obrir_imatge(similar_image_path_list[0])
    #similar_image2= obrir_imatge(similar_image_path_list[1])
    #similar_image3= obrir_imatge(similar_image_path_list[2])

    return (most_similar_image, finding_most_similar)