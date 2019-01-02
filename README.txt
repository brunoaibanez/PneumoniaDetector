Introduction 
This project consists on diagnosing pneumonia from x-rays using a deep learning algorithm. The application’s functionality is, given a radiograph in DCM format, predict whether the image has pneumonia or not. Apart from this, with the prediction it also gives a couple of “similar images”, having and not the illness respectively, so that doctors can compare and confirm the result. Nevertheless, the diagnosis is not done in such a basic way. 
For you to check the performance of our system, the way to execute the test for one input image is:
	python3 test_one.py —db_path dir/dir/path_to_image/image.dcm
But, firstly, some paths must be changed, ROOT_DIR (where this GitHub folder will be located) and DATA_DIR (where data will be located, even though you are only using the test_one, so similar images will be searched on that folder too)
The approach followed when determining the prediction has been the following: 
1) The program distinguishes if the input radiograph is most likely to have:
• Pneumonia • Not pneumonia 
2) After the previous classification, if the input image has been classified as Not pneumonia, the application differs between:
• Healthy • Other diseases 
3) If the image has been predicted as pneumonia, the software locates the areas where the illness has been detected. 
4) Afterwards, the “similar images” algorithm is applied for both cases, Pneumonia and Not Pneumonia:
	- For case Pneumonia, test_one.py will show 2 similar images, one of them with Pneumonia (left) and another without (right), with the predicted bounding boxes printed on the test image (middle).   
	- For case Healthy, test_one.py will show 1 similar image, diagnosed as healthy too	
	- For case Other diseases, test_one.py will show 1 similar image, diagnosed as other diseases too, and the diagnosis made by a RSNA doctor will be printed as well. So that, even though we can’t create a model of every single Other Disease, what we could say to the user is that the Most Similar image from the Database is this one, and it has X disease.
	


Data to be downloaded 
Go to https://www.kaggle.com/c/rsna-pneumonia-detection-challenge and download RSNA database. Change DATA paths! The folder must be named as ‘DCM’.
You also need to download the data from https://www.kaggle.com/nih-chest-xrays/data. The folder must be named ‘images’.

Pretrained models:

- First classification: custom Resnet (see classification1 for more information). Accuracy = 0.8

- Second classification: Inception ResNet V2. Accuracy = 0.7

- Localization: Mask-RCNN. IoU = 0.6

All python scripts for training for a new database are attached, for you to use them and see the performance for your own database.


Way to train your model:

- Classification1: First, you need to run the code CNN1_Split_Data_Creation_2Classes.py to save you’re data from DCM into .pkl files. You will need to modify the paths. You shoud have now your’re .pkl files saved into the ‘created_Data1’ folder. If so, you can run CNN1_Classification_Pneumonia_or_Not.py (You will also have to change paths) to create and save the model into the folder dataPretainedModels used for the first classification stage. The data must be stored on the ‘DCM’ folder from RSNA.

-Classification2: You need first to run CNN2_Separate_Images.py to separate all the x-ray stored on the folder ‘images’ into the different classes according to healthy or not and train or test. These directories are specified on the code. Then you can run CNN2_Classification_Healthy_or_Not.py to train the model that will be stored on dataPretrainedModels and we will use for classification stage 2. As in the previous stage, consider modifying the paths. The data must be stored on the ‘images’ folder and it’s the ‘nih-chest-xrays’ one. 

- Localization: change ROOT_DIR and DATA_DIR. Download coco weights as .h5 file and place it on dataPretrainedModels. You must place a csv document with only images labelled as Pneumonia. The information for those imatges must be: patient_id, class (must be 1 == Pneumonia), bounding boxes

Where bounding boxes will be separated by spaces, such as: x1 y1 height width

A Pneumonia random folder will be created, so terminal will tell you which is best epoch, to get weights from that epoch for the testing. Change the name of the file to locateweightsfinal.h5 and place it on dataPretrainedModels.

-Similar Images CNN1 and CNN2: change ROOT_DIR. For each one, 1 csv is required. The first time you run the code, many variables will be stored for each classification on a folder called dataSimilarImages. You can switch the value of ‘not_guardar’ if you want to save or load these variables. The model used is loaded on the top of the code. We use the model created on Classification1 for the similar image of stage 1, so you might first run the Classification 1 code to get it. For the similar image of the 2nd stage, we provide the model used on the folder dataPretainedModels, which is a trained DenseNet. 


All h5 and hdf5 files must be placed on dataPretrainedModels!!
