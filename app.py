from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
import cv2

from PIL import Image

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

loaded_model = tf.keras.models.load_model('alzheimer_CNN.h5', compile=False)

print('Model loaded. Check http://127.0.0.1:5000/')

@app.route('/', methods=['GET'])

def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction

        # img = image.load_img(path, target_size=(128,128))
        # image_array = img_to_array(img) / 255.0
        # image_array = np.expand_dims(image_array, axis=0)

        # img_array = image_array.reshape((128, 128, 3))
        # predictions = loaded_model.predict(image_array)

        image = cv2.imread(file_path)
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((128, 128))
        expand_input = np.expand_dims(resize_image,axis=0)
        input_data = np.array(expand_input)
        input_data = input_data/255
         
        predicted_label=loaded_model.predict(input_data)
        # predicted_label = train_labels[np.argmax(pred)]
        print(predicted_label)

        def name(predicted_label):
            if(predicted_label[0][0] == 1):
                print('Non Demented Image')
                message = 'Non Demented Image'
                return message
            if(predicted_label[0][1] == 1):
                print('very Mild Image')
                message = 'very Mild Image'
                return message
            if(predicted_label[0][2] == 1):
                print('Mild Image')
                message = 'Mild Image'
                return message
            if(predicted_label[0][3] == 1):
                print('Moderate Image')
                message = 'Moderate Image'
                return message
        s = name(predicted_label)     
        return s
    return None 

if __name__ == '__main__':
    app.run(debug=True)

