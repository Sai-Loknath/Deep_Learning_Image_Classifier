from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__, template_folder='templates')
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
IMAGE_SIZE = (224, 224)
MODEL_PATH = r'C:\Users\Lenovo\Desktop\Deep_Learning_Image_Classification\models\model.h5'
DATA_DIR = r'C:\Users\Lenovo\Desktop\Deep_Learning_Image_Classification\datasets'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model once when the application starts
model = load_model(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, IMAGE_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure RGB format
    image = image / 255.0  # Normalize pixel values
    return image

def get_class_names():
    return os.listdir(DATA_DIR)  # Assuming class names are stored in a directory

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['image']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Preprocess the uploaded image
            image_data = preprocess_image(filepath)
            input_image = np.expand_dims(image_data, axis=0)
            
            # Make predictions using the loaded model
            predictions = model.predict(input_image)
            predicted_class = np.argmax(predictions[0])
            
            # Get the class name from the list of class_names
            class_names = get_class_names()
            predicted_class_name = class_names[predicted_class]
            
            # Flash a success message with the predicted class name
            flash(f'Prediction: {predicted_class_name}')
            
            # Redirect to the result page with the predicted class name and filename as query parameters
            return redirect(url_for('result', prediction=predicted_class_name, filename=filename))
        
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(request.url)
    
    else:
        flash('Invalid file format. Allowed formats: png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    filename = request.args.get('filename')
    
    if filename is None:
        filename = 'default.jpg'  # Provide a default or handle missing file
    
    return render_template('result.html', prediction=prediction, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
