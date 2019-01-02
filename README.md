{\rtf1\ansi\ansicpg1252\cocoartf1561\cocoasubrtf600
{\fonttbl\f0\fswiss\fcharset0 ArialMT;\f1\froman\fcharset0 Times-Roman;\f2\fnil\fcharset0 Tahoma;
\f3\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;\red5\green99\blue193;}
{\*\expandedcolortbl;;\csgenericrgb\c1961\c38824\c75686;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11340\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\ri-1780\sl340\sa240\partightenfactor0

\f0\b\fs36 \cf0 Introduction 
\f1\b0 \

\f0\fs28 This project consists on diagnosing pneumonia from x-rays using a deep learning algorithm.
\f2 \uc0\u8232 
\f0 The application\'92s functionality is, given a radiograph in DCM format, predict whether the image has pneumonia or not. Apart from this, with the prediction it also gives a couple of \'93similar images\'94, having and not the illness respectively, so that doctors can compare and confirm the result. Nevertheless, the diagnosis is not done in such a basic way. \
For you to check the performance of our system, the way to execute the test for one input image is:\
	python3 test_one.py \'97db_path dir/dir/path_to_image/image.dcm\

\b But, firstly, some paths must be changed, ROOT_DIR (where this GitHub folder will be located) and DATA_DIR (where data will be located, even though you are only using the test_one, so similar images will be searched on that folder too)\

\b0 The approach followed when determining the prediction has been the following: 
\f1 \

\f0 1) The program distinguishes if the input radiograph is most likely to have:\

\f1 \'95 
\f0 Pneumonia
\f2 \uc0\u8232 
\f1 \'95 
\f0 Not pneumonia 
\f1 \

\f0 2) After the previous classification, if the input image has been classified as Not pneumonia, the application differs between:\

\f1 \'95 
\f0 Healthy
\f2 \uc0\u8232 
\f1 \'95 
\f0 Other diseases 
\f1 \

\f0 3) If the image has been predicted as pneumonia, the software locates the areas where the illness has been detected. 
\f1 \

\f0 4) Afterwards, the \'93similar images\'94 algorithm is applied for both cases, Pneumonia and Not Pneumonia:\
	- For case Pneumonia, test_one.py will show 2 similar images, one of them with Pneumonia (left) and another without (right), with the predicted bounding boxes printed on the test image (middle).   \
	- For case Healthy, test_one.py will show 1 similar image, diagnosed as healthy too
\f3\fs24 	
\f0\fs28 \
	- For case Other diseases, test_one.py will show 1 similar image, diagnosed as other diseases too, and the diagnosis made by a RSNA doctor will be printed as well. So that, even though we can\'92t create a model of every single Other Disease, what we could say to the user is that the Most Similar image from the Database is this one, and it has X disease.
\f1 \
\pard\pardeftab720\ri-1780\partightenfactor0

\f3 \cf0 	\
\pard\pardeftab720\ri-1780\sl340\sa240\partightenfactor0
\cf0 \
\pard\pardeftab720\ri-1780\partightenfactor0
\cf0 \
\pard\pardeftab720\ri-1780\sl340\sa240\partightenfactor0

\f0\b\fs36 \cf0 Data to be downloaded \
\pard\pardeftab720\ri-1780\partightenfactor0

\f3\b0\fs28 \cf0 Go to https://www.kaggle.com/c/rsna-pneumonia-detection-challenge and download RSNA database. Change DATA paths! The folder must be named as \'91DCM\'92.\
You also need to download the data from {\field{\*\fldinst{HYPERLINK "%22"}}{\fldrslt \cf2 \ul \ulc2 https://www.kaggle.com/nih-chest-xrays/data}}. The folder must be named \'91images\'92.\
\
\pard\pardeftab720\ri-1780\partightenfactor0

\b\fs36 \cf0 Pretrained models:
\b0\fs28 \
\
- First classification: custom Resnet (see classification1 for more information). Accuracy = 0.8\
\
- Second classification: Inception ResNet V2. Accuracy = 0.7\
\
- Localization: Mask-RCNN. IoU = 0.6\
\
All python scripts for training for a new database are attached, for you to use them and see the performance for your own database.\
\
\

\b\fs36 Way to train your model:\
\pard\pardeftab720\ri-1780\partightenfactor0

\b0\fs28 \cf0 \
- Classification1: First, you need to run the code CNN1_Split_Data_Creation_2Classes.py to save you\'92re data from DCM into .pkl files. You will need to modify the paths. You shoud have now your\'92re .pkl files saved into the \'91created_Data1\'92 folder. If so, you can run CNN1_Classification_Pneumonia_or_Not.py (You will also have to change paths) to create and save the model into the folder dataPretainedModels used for the first classification stage. The data must be stored on the \'91DCM\'92 folder from RSNA.\
\
-Classification2: You need first to run CNN2_Separate_Images.py to separate all the x-ray stored on the folder \'91images\'92 into the different classes according to healthy or not and train or test. These directories are specified on the code. Then you can run CNN2_Classification_Healthy_or_Not.py to train the model that will be stored on dataPretrainedModels and we will use for classification stage 2. As in the previous stage, consider modifying the paths. The data must be stored on the \'91images\'92 folder and it\'92s the \'91nih-chest-xrays\'92 one. \
\
- Localization: change ROOT_DIR and DATA_DIR. Download coco weights as .h5 file and place it on dataPretrainedModels. You must place a csv document with only images labelled as Pneumonia. The information for those imatges must be: patient_id, class (must be 1 == Pneumonia), bounding boxes\
\
Where bounding boxes will be separated by spaces, such as: x1 y1 height width\
\
A Pneumonia random folder will be created, so terminal will tell you which is best epoch, to get weights from that epoch for the testing. Change the name of the file to locateweightsfinal.h5 and place it on dataPretrainedModels.\
\
\pard\pardeftab720\partightenfactor0
\cf0 \expnd0\expndtw0\kerning0
-Similar Images CNN1 and CNN2: change ROOT_DIR. For each one, 1 csv is required. The first time you run the code, many variables will be stored for each classification on a folder called dataSimilarImages. You can switch the value of \'91not_guardar\'92 if you want to save or load these variables. The model used is loaded on the top of the code. We use the model created on Classification1 for the similar image of stage 1, so you might first run the Classification 1 code to get it. For the similar image of the 2\super nd\nosupersub  stage, we provide the model used on the folder dataPretainedModels, which is a trained DenseNet.\'a0\kerning1\expnd0\expndtw0 \
\
\pard\pardeftab720\ri-1780\partightenfactor0
\cf0 \
\pard\pardeftab720\ri-1780\partightenfactor0

\b \cf0 All h5 and hdf5 files must be placed on dataPretrainedModels!!
\b0 \
\
\
}