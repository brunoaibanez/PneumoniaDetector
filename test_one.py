import argparse
import pandas as pd
import returnImages as ret
import Similar_Image_CNN1_Most_similar_different_class as SI
import Similar_Image_CNN2 as SI2
from mrcnn.model import log
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn import utils
from mrcnn.config import Config
from sklearn.model_selection import KFold
import glob
from tqdm import tqdm
from imgaug import augmenters as iaa
import pydicom
import json
import matplotlib.pyplot as plt
import os
import returnImages as RI
import sys
import random
import math
import numpy as np
import cv2
import matplotlib
from keras.models import model_from_json
from keras.applications.inception_resnet_v2 import preprocess_input
from skimage.transform import resize
from keras_preprocessing import image
from skimage.io import imsave
from keras.models import load_model
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet169 as DenseNetModel


class0 = "Healthy"
class1 = "Pneumonia"
class2 = "Not Healthy"

# Directori d'on seran les imatges
DATA_DIR = '/veu4/usuaris27/pae2018/GitHub/DCM/'
# Directori d'on sera el model de localitzacio
ROOT_DIR = '/veu4/usuaris27/pae2018/GitHub/'
# Directori d'on seran els pesos de localitzacio
model_locate_path = ROOT_DIR + "dataPretrainedModels/locateweigthsfinal.h5"
# Directori d'on sera la imatge a analitzar
dcm_dir = DATA_DIR + "06e09ebb-cb5b-4c30-8d3a-67a6ec34692b.dcm"

# dcm_dir = DATA_DIR + sys.argv[1] Per posar-lo al executar

# Nom de l'arxiu de test_final
name_of_submission = "testprova.txt"
# Path temporal
path_temporal = ROOT_DIR + "/temporal/"
path_temporal2 = ROOT_DIR + "/temporal/xarxa2/"


# Nom dels pesos i model de Classificacio 1
model1_class_path = ROOT_DIR + 'dataPretrainedModels/classificacio_pneumonia_or_not_Model3.h5'
weights1_path = ROOT_DIR + 'dataPretrainedModels/classificacio_pneumonia_or_not_Model3_weights.best.hdf5'

# Nom dels pesos i model de Classificacio 2
model2_class_path = ROOT_DIR + 'dataPretrainedModels/InceptionResNetV2.h5'
weights2_path = ROOT_DIR + 'dataPretrainedModels/InceptionResNetV2_weights.best.hdf5'


class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """

    # Give the configuration a recognizable name
    NAME = 'pneumonia'

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4  # 256:8
    BACKBONE = 'resnet101'
    NUM_CLASSES = 2  # background + 1 pneumonia classes
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    RPN_ANCHOR_SCALES = (64, 128, 192)
    RPN_NMS_THRESHOLD = 0.9
    TRAIN_ROIS_PER_IMAGE = 16
    MAX_GT_INSTANCES = 3
    DETECTION_MAX_INSTANCES = 2  # ytt
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3
    STEPS_PER_EPOCH = 500
    TRAIN_BN = True


class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


ORIG_SIZE = 1024
config = DetectorConfig()


def predict_where(model, ds, image_id, filepath='submission.txt', min_conf=0.9):
    # assume square image
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
    # resize_factor = ORIG_SIZE
    image = ds.pixel_array
    # If grayscale. Convert to RGB for consistency.
    image = np.stack((image,) * 3, -1)
    pix = image
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)

    patient_id = os.path.splitext(os.path.basename(image_id))[0]

    results = model.detect([image])
    r = results[0]

    out_str = ""
    out_str += patient_id
    out_str += ","
    assert(len(r['rois']) == len(r['class_ids']) == len(r['scores']))
    num_instances = len(r['rois'])
    numbb = 0
    bb = []

    for i in range(num_instances):
        if r['scores'][i] > min_conf:
            out_str += ' '
            out_str += str(round(r['scores'][i], 2))
            out_str += ' '

            # x1, y1, width, height
            x1 = r['rois'][i][1]
            y1 = r['rois'][i][0]
            width = r['rois'][i][3] - x1
            height = r['rois'][i][2] - y1
            bboxes_str = "{} {} {} {}".format(x1*resize_factor, y1*resize_factor,
                                              width*resize_factor, height*resize_factor)

            out_str += bboxes_str
            y1 = int(r['rois'][i][0]*resize_factor)
            x1 = int(r['rois'][i][1]*resize_factor)
            x2 = int(r['rois'][i][3]*resize_factor)
            y2 = int(r['rois'][i][2]*resize_factor)
            print(x1, y1, x2, y2)
            bb.append(x1)
            numbb = numbb + 1
            cv2.rectangle(pix, (x1, y1), (x2, y2), (255, 0, 0), 4)

    print(pix)
    name = path_temporal + "output.jpg"
    imsave(name, pix)

    # Codi SI1
    pix = ds.pixel_array.astype(float)

    # Llegir gènere
    genere = ds.PatientSex
    imatgeSI1, bounding_box, imatgeSI2 = SI.similar_images(
        pix, numbb, bb, 1, genere)
    imatgeSI2 = imatgeSI2/255
    imatge_similar1 = np.stack((imatgeSI1,) * 3, -1)

    for bound in bounding_box:
        print(bound)
        x1 = int(bound[0])
        x2 = int(x1 + bound[2])
        y1 = int(bound[1])
        y2 = int(y1 + bound[3])
        cv2.rectangle(imatge_similar1, (x1, y1), (x2, y2), (255, 0, 0), 4)
        print(x1, y1, x2, y2)

    imatgeSI1 = imatge_similar1/255
    name = path_temporal + "outputSI1.jpg"
    print(imatge_similar1)
    imsave(name, imatgeSI1)
    # Codi SI2
    name = path_temporal + "outputSI2.jpg"
    imsave(name, imatgeSI2)
    RI.returnImages(class1, class1 + " detected",
                    "Not Pneumonia", opts.out_path)


def main(opts):
    loaded_model = load_model(model1_class_path)
    loaded_model.load_weights(weights1_path)
    # Loading the CNN weights
    print("Loaded model")

    # Making a prediction
    trainshape = 224
    dcm_dir = opts.db_path
    ds = pydicom.dcmread(dcm_dir)
    pix = ds.pixel_array.astype(float)
    imatge3_resize = resize(pix, (trainshape, trainshape), mode='reflect')
    imatge3_resize = imatge3_resize/255
    fim = imatge3_resize.reshape([-1, trainshape, trainshape, 1])

    # Giving both ponderations based on 0/1
    pt_prediction = loaded_model.predict(fim)
    pt_prediction = pt_prediction * 100
    pt_prediction = np.round(pt_prediction, 2)

    for values in pt_prediction:
        print('Each class percentage is:', values, '%')
    # training_set.class_indices
    # Returning the mostly weighted class
    prediction = np.argmax(pt_prediction, axis=1)
    print('So, the final prediction is class',
          prediction, ', which is defined as:')

    if prediction == 1:
        print(class1)
        maskdir = ROOT_DIR + 'Mask_RCNN-master'
        os.chdir(maskdir)
        # To find local version of the library
        sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN-master'))

        inference_config = InferenceConfig()

        # Recreate the model in inference mode
        model = modellib.MaskRCNN(mode='inference',
                                  config=inference_config,
                                  model_dir=ROOT_DIR)
        # Load trained weights (fill in path to trained weights here)
        print("Loading weights from ", model_locate_path)

        model.load_weights(model_locate_path, by_name=True)
        submission_fp = os.path.join(opts.out_path, name_of_submission)
        predict_where(model, ds, dcm_dir, submission_fp, 0.9)
        print(submission_fp)

    elif prediction == 0:

        print('Not Pneumonia')

        ds = pydicom.dcmread(dcm_dir)
        pix = ds.pixel_array.astype(float)/255

        name = path_temporal2 + "h/output.png"
        imsave(name, pix)

        trainshape= 224
        test_datagen=ImageDataGenerator(preprocessing_function= preprocess_input)
        x_test= test_datagen.flow_from_directory(
            directory=path_temporal2,
            batch_size=1,
            color_mode= 'rgb',
            class_mode = None,
            target_size=(224,224),
            shuffle=False)

        print("Loading pre-trained model...")
        base_model1 = load_model(model2_class_path)
        base_model1.load_weights(weights2_path)


        x_test.reset()
        pt_prediction = base_model1.predict_generator(x_test, verbose = 1)
        print(pt_prediction)

        pt_prediction = pt_prediction * 100
        pt_prediction = np.round(pt_prediction, 2)

        for values in pt_prediction:
            print('Each class percentage is:', values, '%')
        # training_set.class_indices
        # Returning the mostly weighted class
        prediction = np.argmax(pt_prediction, axis=1)

        print('So, the final prediction is class',
              prediction, ', which is defined as:')

        if prediction == 1:

            print('Not Healthy')
            pred = 'Prediction = Not Healthy'

        elif prediction == 0:

            print('Healthy')
            pred = 'Prediction = Healthy'

        genere = ds.PatientSex
        imatgeSI_2, finding = SI2.similar_images(
            pix, prediction, genere)  # Retorna imatge float 0-1

        name = path_temporal + "output.jpg"
        imsave(name, pix)

        name = path_temporal + "outputSI1.jpg"
        imsave(name, imatgeSI_2)

        RI.returnImages2(pred, finding, opts.out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply trained MLP to classify')

    parser.add_argument('--db_path', type=str, default= ROOT_DIR + 'DCM/06e09ebb-cb5b-4c30-8d3a-67a6ec34692b.dcm', #One random image selected for testing 
                        help='path to input files')
    parser.add_argument('--out_path', type=str, default=ROOT_DIR + 'output/',
                        help='path to output folder')

    opts = parser.parse_args()
    main(opts)
