import tensorflow as tf
from tensorflow.keras.models import load_model
import os

try:
    os.mkdir("/mnt/model")
except Exception as e:
    print(e)

try:
    os.mkdir("/mnt/TRT_output")
except Exception as e:
    print(e)


# path_to_.h5_file
keras_model = load_model('path_to_.h5_file',compile=False)

tf.keras.models.save_model(keras_model, '/mnt/model')