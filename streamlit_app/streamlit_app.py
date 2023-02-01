import streamlit as st
import pandas as pd
from img_classification import melanoma_classification
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.utils import load_img

section = st.sidebar.radio('Now what?', ('Melanoma recognition', 'About approach', 'Next steps',))
if section == 'Melanoma recognition':
    st.title("Melanoma Classification")
    st.header("Image Classification with Keras Tensorflow")
    #st.text("Upload a melanoma Image for image classification")

    uploaded_file = st.file_uploader("Upload a melanoma Image for classification", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = melanoma_classification(image, 'melanoma_model.h5')
        if label == 1:
            st.write("The MRI scan is not malignant.")
        else:
            st.write("The MRI scan is malignant.")
elif section == 'Next steps':
    st.title('Steps after a melanoma diagnosis')
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
else:
    st.title('Classification model characteristics')
    st.text('The melanoma recognition is made using a convolutional neural network Tensorflow.')
    st.text('The network is trained on the 10 000 images.')
    image3 = tf.keras.utils.load_img('model_score.png')
    st.image(image3, caption= 'Score of the CNN Melanoma Classification Model')