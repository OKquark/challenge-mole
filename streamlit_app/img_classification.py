import tensorflow as tf
from keras.preprocessing import image
from keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
from keras.utils import np_utils
from PIL import Image, ImageOps

def melanoma_classification(img, weights_file):
    # Load the model
    json_file = open('melanoma_model.json', 'r')
    json_model = json_file.read()
    json_file.close()
    loaded_model = model_from_json(json_model)
    loaded_model.load_weights('melanoma_model.h5')
    loaded_model.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=["accuracy"])
    # Create the array of the right shape to feed into the keras model
    reshape_img = tf.image.resize(img, [150,150]) 
    input_arr = tf.keras.utils.img_to_array(reshape_img)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    result = loaded_model.predict(input_arr)
    return np.argmax(result) # return position of the highest probability