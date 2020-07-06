# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np

UPLOAD_FOLDER = './flask app/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesnt exist

app = Flask(__name__,static_url_path='/assets',
            static_folder='./flask app/assets', 
            template_folder='./flask app')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def root():
   return render_template('index.html')

@app.route('/upload.html')
def upload():
   return render_template('index.html')

@app.route('/upload_covidAndPneumonia.html')
def upload_chest():
   return render_template('upload_covid.html')

@app.route('/upload_ct.html')
def upload_ct():
   return render_template('upload_ct.html')


@app.route('/uploaded_covidAndPneumonia', methods = ['POST', 'GET'])
def uploaded_chest():
   if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_chest.jpg'))

   # resnet_chest = load_model('')
   vgg_chest = load_model('models/VGG19_CovidAndPneumonia.h5')

   image = cv2.imread('./flask app/assets/images/upload_chest.jpg') # read file 
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
   image = cv2.resize(image,(224,224))
   image = np.array(image) / 255
   image = np.expand_dims(image, axis=0)
   
   vgg_pred = vgg_chest.predict(image)
   probability = vgg_pred[0]
   print("VGG Predictions:")
   colour = ""
   status = ""
   message= ""
   result = ""
   if probability[0] > 0.5:
      vgg_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
      colour = "danger"
      status = "COVID-19"
      message="SudoSweep found it "
      result = "COVID19 Positive "
   else:
      vgg_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% Pneumonia')
      colour = "warning"
      status = "Pneumonia"
      message="SudoSweep found it "
      result = "Pneumonia Positive "
   print(vgg_chest_pred)

   return render_template('results_covid.html',vgg_chest_pred=vgg_chest_pred, colour=colour, status=status, message=message, result=result)



@app.route('/uploaded_ct', methods = ['POST', 'GET'])
def uploaded_ct():
   if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_ct.jpg'))

   resnet_ct = load_model('models/resnet_ct.h5')
   vgg_ct = load_model('models/vgg_ct.h5')

   image = cv2.imread('./flask app/assets/images/upload_ct.jpg') # read file 
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
   image = cv2.resize(image,(224,224))
   image = np.array(image) / 255
   image = np.expand_dims(image, axis=0)
   
   # resnet_pred = resnet_ct.predict(image)
   # probability = resnet_pred[0]
   # print("Resnet Predictions:")
   # if probability[0] > 0.5:
   #    resnet_ct_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
   # else:
   #    resnet_ct_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   # print(resnet_ct_pred)

   vgg_pred = vgg_ct.predict(image)
   probability = vgg_pred[0]
   print("VGG Predictions:")
   if probability[0] > 0.5:
      vgg_ct_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
   else:
      vgg_ct_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   print(vgg_ct_pred)


   return render_template('results_ct.html', vgg_ct_pred=vgg_ct_pred)

if __name__ == '__main__':
   app.secret_key = ".."
   app.run()