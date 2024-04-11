from flask import Flask, render_template, request, flash, redirect
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the saved model
model = load_model('mnist_model.h5')

# Function to preprocess an image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (28, 28))
    preprocessed_image = cv2.bitwise_not(resized_image)
    preprocessed_image = preprocessed_image / 255.0
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=-1)
    return preprocessed_image

# Function to predict the digit from an image
def predict_digit(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_digit = np.argmax(prediction)
    return predicted_digit

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            predicted_digit = predict_digit(file_path)
            return render_template('result.html', filename=filename, predicted_digit=predicted_digit)
        else:
            flash('File type not allowed. Allowed file types are png, jpg, and jpeg.')
            return redirect(request.url)
    return render_template('index.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
