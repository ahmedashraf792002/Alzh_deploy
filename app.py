from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing import image
import io
from keras.applications.vgg16 import preprocess_input
app = Flask(__name__)
model = load_model('model_VGG16.h5')

def preprocess_image(img_path):
    # Preprocessing the image
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    image = file.read()
    '''processed_image = preprocess_image(image)
    preds = model.predict(processed_image)

    # Convert prediction probabilities to class labels
    dic = {0: "Alzheimer's disease", 1: "Cognitively normal", 2: "Early mild cognitive impairment", 3: "Late mild cognitive impairment"}

    pred_class = dic[np.argmax(preds)]   # Get the class label with maximum probability
    pred_proba = str(round(float(preds[0, np.argmax(preds)]),2))+"%"  # Get the probability of the predicted class'''
    return jsonify({'Prediction':pred_class ,'Probability':pred_proba})

if __name__ == '__main__':
    app.run(debug=True)
