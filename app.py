import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from flask import Flask, url_for, render_template, redirect, request
import flask
import numpy as np
import pandas as pd
import cv2
from PIL import Image

app = Flask(__name__, template_folder='templates')


# Loading main page
@app.route('/')
def main():
    return render_template('brain_tumor.html')


def predict_Btumor(img_path):
    model_load = load_model("Trained Model/brain_tumor.h5")

    img = cv2.imread(img_path)
    img = Image.fromarray(img)
    img = img.resize((64, 64))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)

    preds = model_load.predict(img)
    return preds[0]


@app.route('/predictBTumor', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))

        preds = predict_Btumor(file_path)
        # print(preds)

        if int(preds[0]) == 0:
            result = "No worry! No Brain Tumor"
        else:
            result = "Patient has Brain Tumor"

        print(f'prdicted: {result}')

        return result

    return None


if __name__ == '__main__':
    app.config['Debug'] = True
    app.run()
