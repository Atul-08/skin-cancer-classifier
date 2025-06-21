from flask import Flask, render_template, request
import os
import uuid
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load your skin cancer classifier
model = load_model('model.h5')

# Class labels
class_labels = {
    0: "Benign",
    1: "Malignant"
}

@app.route('/')
def index():
    return render_template('index.html', predictions=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'img' not in request.files:
        return render_template('index.html', predictions=None)

    file = request.files['img']
    if file.filename == '':
        return render_template('index.html', predictions=None)

    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess image
    img = Image.open(filepath).convert('RGB')
    img = img.resize((224, 224))  # Match model input
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    # Predict
    prediction = model.predict(img_array)[0]
    top_idx = prediction.argmax()
    label = class_labels[top_idx]
    confidence = round(prediction[top_idx] * 100, 2)

    return render_template('index.html', predictions=(label, confidence), img_path=filepath)

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
