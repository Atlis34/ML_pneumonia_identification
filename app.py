from flask import Flask, request, jsonify, redirect, url_for, session
import os
import joblib
import numpy as np
import requests

# initialize loaded_model
loaded_model = None

app = Flask(__name__)
app.secret_key = 'absynthesis'

# home route with image upload form
@app.route('/', methods=['GET', 'POST'])
def home():

    global loaded_model  # declare loaded_model as global variable

    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']
        if file.filename == '':
            return 'No selected file'

        if file:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            files = {'file': open(image_path, 'rb')}

            # load model if not already loaded
            if loaded_model is None:
                model_path = 'model.joblib'
                loaded_model = joblib.load(model_path)
                if loaded_model is None:
                    return 'Failed to load the model.'
            
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

# set upload folder where images will be loaded from
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# define route for uploading images
@app.route('/upload', methods=['POST'])
def upload_image():
    # check if POST request has file part
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    # if user does not select file, browser submits empty file without filename
    if file.filename == '':
        return 'No selected file'

    # save uploaded file to upload folder
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    return 'File uploaded successfully'

# lighten image function
def lighten_image(image, factor=1.5):
    return np.clip(image * factor, 0, 255).astype(np.uint8)

# darken image function
def darken_image(image, factor=0.5):
    return np.clip(image * factor, 0, 255).astype(np.uint8)

# augment data function
def augment_dataset(X_train, y_train):
    augmented_images = []
    augmented_labels = []
    
    for image, label in zip(X_train, y_train):
        original_image = image
        lightened_image = lighten_image(image)
        darkened_image = darken_image(image)
        
        augmented_images.extend([original_image, lightened_image, darkened_image])
        augmented_labels.extend([label, label, label])
    
    X_train_augmented = np.array(augmented_images)
    y_train_augmented = np.array(augmented_labels)
    
    return X_train_augmented, y_train_augmented

# route for preprocessing function
@app.route('/preprocess_image', methods=['POST'])
def preprocess_image():
    data = request.json
    
    # image data 
    image_data = data.get('image_data')
    
    # preprocess image by augmenting it
    augmented_image_data = augment_dataset([image_data], [1])  # Assuming label 1 for simplicity
    
    return jsonify({'augmented_image_data': augmented_image_data.tolist()})

# route to load model from joblib
@app.route('/load_model', methods=['GET'])
def load_model():
    global loaded_model
    model_path = 'model.joblib'
    loaded_model = joblib.load(model_path)
    return 'Model loaded successfully'

# route to make a prediction using loaded model
@app.route('/predict', methods=['GET', 'POST'])
def make_prediction():
    global loaded_model
    if loaded_model is None:
        return 'Model not loaded. Please load the model first.'

    image_data = request.json.get('image_data')
    augmented_image_data = augment_dataset([image_data], [1])
    prediction = loaded_model.predict(augmented_image_data)
    prediction_labels = ['Healthy', 'Pneumonia']
    predicted_classes = [prediction_labels[p] for p in prediction]

    # Store prediction data in session
    session['prediction'] = predicted_classes

    return 'Prediction Result: ' + ', '.join(predicted_classes)

if __name__ == '__main__':
    app.run(debug=True)
