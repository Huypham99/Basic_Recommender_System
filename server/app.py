from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import json
import tensorflow as tf
from flask import Flask, jsonify
from flask_pymongo import pymongo
from pymongo import MongoClient
from bson.json_util import dumps
from bson import ObjectId
import io
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import csv
import pandas as pd
import pickle
from cloudinary.uploader import upload
from cloudinary.utils import cloudinary_url

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)

#get envỉronment variable
MONGO_URI = os.environ.get('MONGO_URI')

# Define a flask app
app = Flask(__name__)
app.config.from_mapping( CLOUDINARY_URL=os.environ.get('CLOUDINARY_URL'))
client = MongoClient(MONGO_URI)
db = client['car-prediction']
car_collection = db['cars']

# Read CSV file contain class name
data_class_index = pd.read_csv('uploads/names.csv')

# Load ResNet34 Model 
model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 216)
model_state = torch.load('models/car_classifier2.pth', map_location=torch.device('cpu'))
model.load_state_dict(model_state)
model.eval()


# Takes image data in bytes, applies the series of transforms and returns a tensor
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize((320, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).float().unsqueeze(0)

# Prediction
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model(tensor)
    _, predicted = torch.max(outputs.data, 1)
    print(int(predicted))
    return outputs

# Using L2 norm to calculate similarity between 2 feature vectors
def calculate_similarity(target_feature, input_feature):
    return np.linalg.norm(
        pickle.loads(target_feature)-np.array(input_feature)[0]
    )

# Api route for prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    if request.method == 'POST':
        file = request.files['file']

        # Read upload image as bytes
        img_bytes = file.read()

        # Return predicted class_name
        output = get_prediction(image_bytes=img_bytes)

        #car_collection.update_one({'name': 'Bugatti Veyron 16.4 Convertible 2009'}, {"$set": {"feature": Binary( pickle.dumps( np.array(output.data)[0]) )}})

        cars = car_collection.find()

        list_cars = list(cars)

        car_sorted = list()

        result = list()

        for car in list_cars:
          score = calculate_similarity(car["feature"], output.data)
          car_sorted.append({'infor': car, 'score': score})

          def get_score(img):
            return img.get('score')

          car_sorted.sort(key=get_score)

        for car in car_sorted[:4]:
          result.append({
              'name': car['infor']['name'],
              'price': car['infor']['price'],
              'imageId': car['infor']['imageId']
          })
        
        print(json.dumps(result))
        
        return json.dumps(result)
        
    return None


@app.route('/cars', methods=['GET'])
def cars():

    result = list()
    cars = list(car_collection.find())
    for car in cars:
          result.append({
              'name': car['name'],
              'price': car['price'],
              'imageId': car['imageId']
          })
    return json.dumps(result)

if __name__ == '__main__':
    app.run(debug=True)

