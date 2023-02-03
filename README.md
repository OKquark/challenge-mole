# challenge-mole
SkinCare image recognition

The aim of the project is to create an AI that can recognise skin images and detect when the mole is dangerous. To prove the concept a simple web page could be developed where the user could upload a picture of the mole and see the result.

For the 1st MVP was built a Convolutional Neural Network using Keras Tensorflow and packed into the .h5 format to be uploaded in the Streamlit application. 

## Getting Started

The **dataset** is in unlabeled .jpg format. The dataset includes 10 000  images in two folders and the matadata with images classification in .csv. 

## Prerequisites

The packages and libraries you need to have a copy of the project up and running on your local machine for development and testing purposes.

* Python 3.x 
* pandas
* numpy
* os
* glob

* keras.preprocessing
* keras.utils
* tensorflow.keras.layers
* tensorflow.keras.models
* keras.utils

* scikit-learn 
* matplotlib
* seaborn
* cv2

## Installation

This step by step guide will get you to have the development environment up and running.

1. Create and activate your virtual environment
2. Install additional packages and libraries
3. Open the notebooks in a code editor of your choice running on the virtual environment you just created

## Usage

1. To devide images into malignant and not malignant based on the metadata run balance_dataset.ipynb
2. To create the labeled dataframe and the model run melanoma_create_train_save_model_recall.ipynb 
3. To run the Streamlit app: python -m streamlit run streamlit_app.py

## Pipeline

1. **Create labeled dataframe**
> All images were divided into two folders: "Malignant" and "Normal" based on the metadata. Then the labeld pd.Dataframes  were created for training, validation and testing the model.  

2. **Preprocessing**
>  The images were resized (150,150). To improve the image recognition was used ImageDataGenerator. 

3. **Building model**
> The 3-layers convolutional model was built. The model was compiled with Adam optimizer and trained within 10 epochs. 

4. **Model evaluation**
> The model is evaluated with the confusion matrix, accuracy, precision, recall, f1-score.   

5. **Save the model and weights of the model**
> The model was saved into .json and .h5 formats to be uploaded into the streamlit app. 

6. **Build up the Streamlit app**
> There are 2 files for web UI and with the model classification.  

## Contributors

1. [Olga Kuznetsova](https://github.com/OKquark) 


## Timeline

Project Start ---> `27 January 2023`  

Project End ---> `03 February 2023`