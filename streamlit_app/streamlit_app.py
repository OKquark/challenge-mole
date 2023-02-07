import streamlit as st
import pandas as pd
from img_classification import melanoma_classification
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.utils import load_img

section = st.sidebar.radio('ScinCare app', ('Melanoma recognition', 'About approach', 'Next steps','Scientific research'))
if section == 'Melanoma recognition':
    st.title("Early Melanoma Detection")
    st.header("Image recognition with Keras Tensorflow")
    #st.text("Upload a melanoma Image for image classification")

    uploaded_file = st.file_uploader("Upload a mole image for classification", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = melanoma_classification(image, 'melanoma_model.h5')
        if label == 1:
            st.write("The mole is not malignant.")
        else:
            st.write("It is malignant. There is a melanoma. Please contact a doctor.")
elif section == 'Next steps':
    st.title('Early detection makes a difference 99% \n 5-year survival rate for patients in the U.S. \n whose melanoma is detected early.(skincancer.org)')
    st.header('Steps after a melanoma diagnosis')
    st.text('1. Skin exam and physical')
    st.text('2. Staging')
    st.text('3. Testing')
    st.text('4. Treatment and possible restaging')
    st.text('5. Observation')
    st.text('6. Lifelong follow-up')
    st.header('Make an appointment with a doctor')
    text = st.text_area('Write a message to the doctor')
    coordinates = pd.DataFrame({
        'lat': [50.50439, 51.05, 50.63373],
        'lon': [4.34878, 3.71667, 5.56749]
    })
    #st.write(coordinates)
    st.map(coordinates)
elif section == 'Scientific research':
    st.title('Further application development')
    st.header('Dataset improvement')
    st.text('Balance dataset')
    st.text('Precluster images before feeding to the model')
    st.header('Model improvement')
    st.text('More complex model(more layers, recall metrics)')
    st.text('Transfer learning')
else:
    st.title('Classification model characteristics')
    st.text('The melanoma recognition is made using a convolutional neural network and enables \n to detetct if the mole is malignant or not.')
    st.text('The neural network is build with Keras Tensorflow and has 89% accuracy score.')
    st.text('The network is trained on the 10 000 images.')
    image3 = tf.keras.utils.load_img('model_score.png')
    st.image(image3, caption= 'Score of the CNN Melanoma Classification Model')
  