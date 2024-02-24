from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
from keras.preprocessing import image
app = Flask(__name__)
model = load_model('model_VGG16.h5')

from keras.utils import img_to_array

def preprocess_image(img_path):
    op_img = Image.open(img_path)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            img = preprocess_image(request.files['file'].stream)
            preds = model.predict(img)
        
            # Convert prediction probabilities to class labels
            dic = {0: "Alzheimer's disease", 1: "Cognitively normal", 2: "Early mild cognitive impairment", 3: "Late mild cognitive impairment"}
        
            pred_class = dic[np.argmax(preds)]   # Get the class label with maximum probability
            pred_proba = str(round(float(preds[0, np.argmax(preds)]),2))+"%"  # Get the probability of the predicted class
            return  pred_class+" "+pred_proba

    except:
        error = "File cannot be processed."
        return render_template("result.html", err=error)






if __name__ == '__main__':
    app.run(debug=True)
