from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import os
import shutil
from keras.models import load_model

upload_folder = 'uploaded/image'
if os.path.exists(upload_folder):
    shutil.rmtree(upload_folder)
os.makedirs(upload_folder)

model = keras.models.load_model('Pneumonia_Identifier.h5')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = upload_folder

@app.route('/')
def upload_f():
    return render_template('upload.html')

def finds():
    test_datagen = ImageDataGenerator(rescale=1./255)
    vals = ['Cat', 'Dog']
    test_dir = 'uploaded'
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        color_mode="rgb",
        shuffle=False,
        class_mode=None,
        batch_size=1
    )

    pred = model.predict(test_generator)
    print(pred)
    return str(vals[np.argmax(pred)])

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        val = finds()
        return render_template('pred.html', ss=val)

if __name__ == '__main__':
    app.run()