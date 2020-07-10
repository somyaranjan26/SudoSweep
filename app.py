# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np

CURRENT_DIRECTORY    = os.path.dirname( os.path.dirname( os.path.realpath( __file__ ) ) )
CURRENT_PACKAGE_NAME = os.path.basename( CURRENT_DIRECTORY ).rsplit('.', 1)[0]

UPLOAD_FOLDER = './flaskapp/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesnt exist

app = Flask(__name__,static_url_path='/assets',
            static_folder='./flaskapp/assets', 
            template_folder='./flaskapp')
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

@app.route('/upload_covidAndNormal.html')
def upload_covid():
   return render_template('upload_covid.html')

@app.route('/upload_PneumoniaAndNormal.html')
def upload_pneumonia():
   return render_template('upload_covid.html')


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

   image = cv2.imread('./flaskapp/assets/images/upload_chest.jpg') # read file 
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


@app.route('/uploaded_covidAndNormal', methods = ['POST', 'GET'])
def uploaded_covid():
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
   vgg_chest = load_model('models/vgg_covid100.h5')

   image = cv2.imread('./flaskapp/assets/images/upload_chest.jpg') # read file 
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
      vgg_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% Non-COVID')
      colour = "success"
      status = "Normal"
      message="SudoSweep found it "
      result = "Normal "
   print(vgg_chest_pred)

   return render_template('results_covid.html',vgg_chest_pred=vgg_chest_pred, colour=colour, status=status, message=message, result=result)


@app.route('/uploaded_PneumoniaAndNormal', methods = ['POST', 'GET'])
def uploaded_pneumonia():
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
   vgg_chest = load_model('models/vgg_pneumonia.h5')

   image = cv2.imread('./flaskapp/assets/images/upload_chest.jpg') # read file 
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
      vgg_chest_pred = str('%.2f' % (probability[0]*100) + '% PNEUMONIA') 
      colour = "danger"
      status = "Pneumonia"
      message="SudoSweep found it "
      result = "Pneumonia Positive "
   else:
      vgg_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NON-PNEUMONIA')
      colour = "success"
      status = "Normal"
      message="SudoSweep found it "
      result = "Normal "
   print(vgg_chest_pred)

   return render_template('results_covid.html',vgg_chest_pred=vgg_chest_pred, colour=colour, status=status, message=message, result=result)

if __name__ == '__main__':
   app.secret_key = ".."
   app.run(host="0.0.0.0", port=80)