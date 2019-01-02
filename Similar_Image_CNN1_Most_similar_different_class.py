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

ROOT_DIR = '/veu4/usuaris27/pae2018/GitHub/'


df= pd.read_csv(ROOT_DIR + 'dataCSV/image_bbox_full.csv')

rows_done=0
vector_distancies=[]
number_boxes_array=[]
images_path = []
classes = []
genders = []
predictions= []
bounding_box=[]
bounding_box_array=[]
bounding_box_array=np.array(bounding_box_array)
trainshape= 224
not_guradar=1




print("Loading pre-trained model...")
base_model1 = load_model(ROOT_DIR + 'dataPretrainedModels/classificacio_pneumonia_or_not_Model3.h5')
base_model1.load_weights(ROOT_DIR + 'dataPretrainedModels/classificacio_pneumonia_or_not_Model3_weights.best.hdf5')
loaded_model1 = Model(input=base_model1.input, output=base_model1.get_layer('dense_6').output)

def tractament_path_imatge_individual(path_imatge_individual): #path_imatge_individaul expressed as DCM/x-ray.dcm
    similar_image = pydicom.dcmread(path_imatge_individual)
    similar_imag2 = similar_image.pixel_array
    similar_image3= similar_imag2.astype(float)
    return similar_image3
    
def obrir_imatge(path_imatge_individual):
    path_imatge_individual= ROOT_DIR + path_imatge_individual
    similar_image = pydicom.dcmread(path_imatge_individual)
    similar_imag2 = similar_image.pixel_array
    similar_image3= similar_imag2.astype(float)
    return similar_image3




def processament_imatge(file_png):
    ds = pydicom.dcmread(file_png)
    image_2d = (ds.pixel_array).astype(float)
    image_2d = image_2d/255
    img_train = resize(image_2d, (trainshape, trainshape), mode = 'reflect')
    img_train=img_train.reshape([-1, trainshape, trainshape, 1])
    prediction= loaded_model1.predict(img_train)
    return prediction

if not_guradar==0:
    for index, row in df.iterrows():

        bounding_box= np.append(bounding_box, row["x"])
        bounding_box= np.append(bounding_box, row["y"])
        bounding_box= np.append(bounding_box, row["width"])
        bounding_box= np.append(bounding_box, row["height"])
        bounding_box=np.array(bounding_box)
        bounding_box= bounding_box.reshape([1,4])

        if rows_done == 0:
            bounding_box_array= bounding_box
        else:
           bounding_box_array= np.vstack((bounding_box_array, bounding_box)) 

        bounding_box=[]

        number_boxes=row["boxes"]

        image_path=  str(row["path"])
        classe= row["Target"]
        gender=str(row["PatientSex"])


        images_path.append(image_path)
        classes.append(classe)
        genders.append(gender)
        number_boxes_array.append(number_boxes)
        predictions.append(processament_imatge(image_path))
        rows_done+=1
        print('rows done', rows_done)

    images_path= np.array(images_path)
    classes= np.array(classes)
    genders= np.array(genders)
    predictions= np.array(predictions)
    bounding_box_array=np.array(bounding_box_array)
    number_boxes_array=np.array(number_boxes_array)

    output1 = open(ROOT_DIR + 'dataSimilarImages/images_path.pkl', 'wb')
    pickle.dump(images_path, output1,  protocol=4)
    output1.close()

    output2 = open(ROOT_DIR + 'dataSimilarImages/classes.pkl', 'wb')
    pickle.dump(classes, output2,  protocol=4)
    output2.close()

    output3 = open(ROOT_DIR + 'dataSimilarImages/genders.pkl', 'wb')
    pickle.dump(genders, output3,  protocol=4)
    output3.close()

    output4 = open(ROOT_DIR + 'dataSimilarImages/predictions.pkl', 'wb')
    pickle.dump(predictions, output4,  protocol=4)
    output4.close()

    output5 = open(ROOT_DIR + 'dataSimilarImages/bounding_box_array.pkl', 'wb')
    pickle.dump(bounding_box_array, output5,  protocol=4)
    output5.close()

    output6 = open(ROOT_DIR + 'dataSimilarImages/number_boxes_array.pkl', 'wb')
    pickle.dump(number_boxes_array, output6,  protocol=4)
    output6.close()

    print('images_path', images_path.shape)
    print('classes', classes.shape)
    print('genders', genders.shape)
    print('predictions', predictions.shape)
    print('bounding_box_array', bounding_box_array.shape)
    print('number_boxes_array', number_boxes_array.shape)

def similar_images(imatge_test, num_bbxes, x_bbx_test,  classe_test, sexe_test):

    output1 = open(ROOT_DIR + 'dataSimilarImages/images_path.pkl', 'rb')
    images_path = pickle.load(output1)
    output1.close()

    output2 = open(ROOT_DIR + 'dataSimilarImages/classes.pkl', 'rb')
    classes = pickle.load(output2)
    output2.close()

    output3 = open(ROOT_DIR + 'dataSimilarImages/genders.pkl', 'rb')
    genders = pickle.load(output3)
    output3.close()

    output4 = open(ROOT_DIR + 'dataSimilarImages/predictions.pkl', 'rb')
    predictions = pickle.load(output4)
    output4.close()

    output5 = open(ROOT_DIR + 'dataSimilarImages/bounding_box_array.pkl', 'rb')
    bounding_box_array = pickle.load(output5)
    output5.close()

    output6 = open(ROOT_DIR + 'dataSimilarImages/number_boxes_array.pkl', 'rb')
    number_boxes_array = pickle.load(output6)
    output6.close()

    print('Predictions shape:', predictions.shape)
    predictions= predictions.reshape([-1, 64])
    print('Predictions shape:', predictions.shape)


    image_2d_test = imatge_test/255
    img_test = resize(image_2d_test, (trainshape, trainshape), mode = 'reflect')
    img_test= img_test.reshape([1, trainshape, trainshape, 1])
    prediction_test= loaded_model1.predict(img_test)


    same_class = np.equal(classe_test, classes)
    same_class= np.array(same_class)
    different_class = np.equal(same_class, False)
    different_class= np.array(different_class)


    print('same class shape', np.count_nonzero(same_class))
    print('same class array', same_class[0:10])
    
    print('different_class shape', np.count_nonzero(different_class))
    print('different_class array', different_class[0:10])


    genders=np.array(genders)
    same_gender= np.core.defchararray.equal(sexe_test, genders)
    same_gender= np.array(same_gender)


    print('same_gender shape', same_gender.shape)
    print('same_gender array', same_gender[0:10])


    same_class_same_gender= np.logical_and(same_class, same_gender)
    same_class_same_gender= np.array(same_class_same_gender)

    different_class_same_gender= np.logical_and(different_class, same_gender)
    different_class_same_gender= np.array(different_class_same_gender)

    
    print(type(same_class_same_gender))
    print('same_class_same_gender shape', same_class_same_gender.shape)
    print('same_class_same_gender', same_class_same_gender[0:10])

    #Agfem radiografies amb mateix numero de bbx nomes tambe
    same_num_bbxes= np.equal(num_bbxes, number_boxes_array)
    same_num_bbxes= np.array(same_num_bbxes)
    print('same_num_bbxes: ', np.count_nonzero(same_num_bbxes))
    print('same_num_bbxes', same_num_bbxes[0:10])

    #Comparacio final mateixa classe sexe numero bbx
    same_class_gender_bbxes= np.logical_and(same_class_same_gender, same_num_bbxes )
    same_class_gender_bbxes=np.array(same_class_gender_bbxes)
    print('Numero imatges a comprar similars: ',  np.count_nonzero(same_class_gender_bbxes))
    print('Numero imatges a comprar diferents: ',  np.count_nonzero(different_class_same_gender))

    images_path_similars= images_path[same_class_gender_bbxes]
    images_path_different= images_path[different_class_same_gender]

    predictions_similars= predictions[same_class_gender_bbxes]
    predictions_different= predictions[different_class_same_gender]

    bounding_box_array_similars= bounding_box_array[same_class_gender_bbxes, :]
    number_boxes_array_similars= number_boxes_array[same_class_gender_bbxes]

    print('images_path_similars', images_path_similars.shape)
    print('predictions_similars', predictions_similars.shape )
    print('prediccions diferents', predictions_different.shape )
    print('bounding_box_array_similars', bounding_box_array_similars.shape )
    print('number_boxes_array_similars', number_boxes_array_similars.shape)

    vector_distancies_similar=[]
    vector_distancies_different=[]

    for i in range(predictions_different.shape[0]):
        distancia_different= distance.canberra(predictions_different[i,:], prediction_test )
        vector_distancies_different.append(distancia_different)

    for i in range(predictions_similars.shape[0]):
        distancia_similar= distance.canberra(predictions_similars[i,:], prediction_test)
        vector_distancies_similar.append(distancia_similar)

    vector_distancies_similar= np.array(vector_distancies_similar)
    vector_distancies_similar=vector_distancies_similar.reshape([-1,1])

    vector_distancies_different= np.array(vector_distancies_different)
    vector_distancies_different=vector_distancies_different.reshape([-1,1])
    print('vector_distancies_similar shape', vector_distancies_similar.shape)
    print('vector_distancies_different shape', vector_distancies_different.shape)

    
    bounding_box_img1=[]
    bounding_box_img2=[]
    bounding_box_img3=[]
    ssim_values_array=[]
    similar_image_path_list=[]
    
    images_done=0
    while images_done < 3 :

        minimum_position= np.argmin(vector_distancies_similar)

        if vector_distancies_similar[minimum_position] == 0:
            vector_distancies_similar= np.delete(vector_distancies_similar, [minimum_position], axis= 0)
            images_path_similars= np.delete(images_path_similars, [minimum_position])
            bounding_box_array_similars= np.delete(bounding_box_array_similars, minimum_position, 0)
            number_boxes_array_similars= np.delete(number_boxes_array_similars, [minimum_position])
        else:            
            print('Min distance value position', minimum_position , ': ' ,vector_distancies_similar[minimum_position])
            
            similar_image_path = images_path_similars[minimum_position]
            ssim_value= compare_sim(imatge_test, obrir_imatge(similar_image_path))

            same_bbx_position= False
            pos_right_similar=[]
            pos_left_similar=[]
            pos_left= False
            pos_right= False
            left_bbx_match= False
            right_bbx_match= False
            must_left= False
            must_right= False

            for i in range(num_bbxes):
                if x_bbx_test[i] < 512:
                    pos_left= True
                    must_left= True
                else:
                    pos_right= True
                    must_right= True
                
                print('pos_left: ', pos_left, 'must_left:', must_left)
                print('pos_right: ', pos_right, 'must_right:', must_right)
                for n in range(number_boxes_array_similars[minimum_position]):
                    if pos_left:
                        pos_left_similar.append(bounding_box_array_similars[minimum_position + n, 0] < 512)
                        if np.any(pos_left_similar):
                            left_bbx_match= True

                    else:
                        pos_right_similar.append(bounding_box_array_similars[minimum_position + n, 0] >= 512)
                        if np.any(pos_right_similar):
                            right_bbx_match= True
                    
                
                pos_left= False
                pos_right= False
                
            print('pos_left_similar: ', pos_left_similar, ' left_bbx_match: ', left_bbx_match)
            print('pos_right_similar: ', pos_right_similar, ' right_bbx_match: ', right_bbx_match )            
            left_test_similar = must_left == left_bbx_match
            right_test_similar = must_right == right_bbx_match
            same_bbx_position = left_test_similar == right_test_similar 
            print('same_bbx_position', same_bbx_position)

            
            if ((images_done != 0 and similar_image_path_list[-1] == similar_image_path) or (ssim_value < 0.03 or not(same_bbx_position) )):
                vector_distancies_similar= np.delete(vector_distancies_similar, [minimum_position], axis= 0)
                images_path_similars= np.delete(images_path_similars, [minimum_position] )
                bounding_box_array_similars= np.delete(bounding_box_array_similars, minimum_position, 0)
                number_boxes_array_similars= np.delete(number_boxes_array_similars, [minimum_position])
            
            else:
                if images_done == 0 and classe_test == 1:
                    for i in range(number_boxes_array_similars[minimum_position]):
                        bounding_box_img1.append(bounding_box_array_similars[minimum_position + i, :])

                if images_done == 1 and classe_test == 1:
                    for i in range(number_boxes_array_similars[minimum_position]):
                        bounding_box_img2.append(bounding_box_array_similars[minimum_position + i, :])

                if images_done == 2 and classe_test == 1:
                    for i in range(number_boxes_array_similars[minimum_position]):
                        bounding_box_img3.append(bounding_box_array_similars[minimum_position + i, :])

                ssim_values_array.append(ssim_value)             
                similar_image_path_list.append(similar_image_path)
                vector_distancies_similar= np.delete(vector_distancies_similar, [minimum_position], axis= 0)
                images_path_similars= np.delete(images_path_similars, [minimum_position] )
                bounding_box_array_similars= np.delete(bounding_box_array_similars, minimum_position, 0)
                number_boxes_array_similars= np.delete(number_boxes_array_similars, [minimum_position])      

                images_done+=1

    max_ssim_position= np.argmax(ssim_values_array)            
    print(similar_image_path_list[0], 'Bunding box:', bounding_box_img1)
    print(similar_image_path_list[1], 'Bounding box:', bounding_box_img2)
    print(similar_image_path_list[2], 'Bounding box:', bounding_box_img3)
    most_similar_image= obrir_imatge(similar_image_path_list[max_ssim_position])
    if max_ssim_position == 0:
        bounding_box_most_similar= bounding_box_img1
    if max_ssim_position == 1:
        bounding_box_most_similar= bounding_box_img2
    if max_ssim_position == 2:
        bounding_box_most_similar= bounding_box_img3

    print('Most similar Image Path', similar_image_path_list[max_ssim_position])
    print('Bounding Box Most Similar Image', bounding_box_most_similar )

    images_done= 0
    similar_image_path_list=[]
    ssim_values_array=[]
    while images_done < 3 :

        minimum_position= np.argmin(vector_distancies_different)

        if vector_distancies_different[minimum_position] == 0:
            vector_distancies_different= np.delete(vector_distancies_different, [minimum_position], axis= 0)
            images_path_different= np.delete(images_path_different, [minimum_position])
        else:            
            print('Min distance value position', minimum_position , ': ' ,vector_distancies_different[minimum_position])
            
            similar_image_path = images_path_different[minimum_position]
            ssim_value= compare_sim(imatge_test, obrir_imatge(similar_image_path))
            
            if (images_done != 0 and similar_image_path_list[-1] == similar_image_path) or ssim_value < 0.03:
                vector_distancies_different= np.delete(vector_distancies_different, [minimum_position], axis= 0)
                images_path_different= np.delete(images_path_different, [minimum_position] )
            
            else:

                ssim_values_array.append(ssim_value)             
                similar_image_path_list.append(similar_image_path)
                vector_distancies_different= np.delete(vector_distancies_different, [minimum_position], axis= 0)
                images_path_different= np.delete(images_path_different, [minimum_position] )

                images_done+=1

    max_ssim_position= np.argmax(ssim_values_array)            
    most_different_image= obrir_imatge(similar_image_path_list[max_ssim_position])



    return (most_similar_image, bounding_box_most_similar, most_different_image )