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
from bson.binary import Binary
import cv2 as cv2
import matplotlib.pyplot as plt
import random
from skimage.feature import hog
from sklearn.svm import LinearSVC
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label
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

# Load LinearSVC Model
classifier1 = pickle.load(open('models/model.pkl', 'rb'))

def SpatialBinningFeatures(image,size):
    image= cv2.resize(image,size)
    return image.ravel()

def GetFeaturesFromHog(image,orient,cellsPerBlock,pixelsPerCell, visualise=False, feature_vector_flag=True):
    if(visualise==True):
        hog_features, hog_image = hog(image, orientations=orient,
                          pixels_per_cell=(pixelsPerCell, pixelsPerCell), 
                          cells_per_block=(cellsPerBlock, cellsPerBlock), 
                          visualize=True, feature_vector=feature_vector_flag)
        return hog_features, hog_image
    else:
        hog_features = hog(image, orientations=orient,
                          pixels_per_cell=(pixelsPerCell, pixelsPerCell), 
                          cells_per_block=(cellsPerBlock, cellsPerBlock), 
                          visualize=False, feature_vector=feature_vector_flag)
        return hog_features

def ExtractFeatures(images,orientation,cellsPerBlock,pixelsPerCell, convertColorspace=False):
    featureList=[]
    imageList=[]
    for image in images:
        if(convertColorspace==True):
            image= cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        local_features_1=GetFeaturesFromHog(image[:,:,0],orientation,cellsPerBlock,pixelsPerCell, False, True)
        local_features_2=GetFeaturesFromHog(image[:,:,1],orientation,cellsPerBlock,pixelsPerCell, False, True)
        local_features_3=GetFeaturesFromHog(image[:,:,2],orientation,cellsPerBlock,pixelsPerCell, False, True)
        x=np.hstack((local_features_1,local_features_2,local_features_3))
        featureList.append(x)
    return featureList

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    
    for bbox in bboxes:
        r=random.randint(0,255)
        g=random.randint(0,255)
        b=random.randint(0,255)
        color=(r, g, b)
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# slide window
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.9, 0.9)):
   
    if x_start_stop[0] == None:
        x_start_stop[0]=0
    if x_start_stop[1] == None:
        x_start_stop[1]=img.shape[1]
    if y_start_stop[0] ==  None:
        y_start_stop[0]= 0
    if y_start_stop[1] ==  None:
        y_start_stop[1]=img.shape[0]
    
    
    window_list = []
    image_width_x= x_start_stop[1] - x_start_stop[0]
    image_width_y= y_start_stop[1] - y_start_stop[0]
     
    windows_x = np.int( 1 + (image_width_x - xy_window[0])/(xy_window[0] * xy_overlap[0]))
    windows_y = np.int( 1 + (image_width_y - xy_window[1])/(xy_window[1] * xy_overlap[1]))
    
    modified_window_size= xy_window
    for i in range(0,windows_y):
        y_start = y_start_stop[0] + np.int( i * modified_window_size[1] * xy_overlap[1])
        for j in range(0,windows_x):
            x_start = x_start_stop[0] + np.int( j * modified_window_size[0] * xy_overlap[0])
            
            x1 = np.int( x_start +  modified_window_size[0])
            y1= np.int( y_start + modified_window_size[1])
            window_list.append(((x_start,y_start),(x1,y1)))
    return window_list 


# draw car
def DrawCars(image,windows, converColorspace=False):
    refinedWindows=[]
    for window in windows:
        
        start= window[0]
        end= window[1]
        clippedImage=image[start[1]:end[1], start[0]:end[0]]
        
        if(clippedImage.shape[1] == clippedImage.shape[0] and clippedImage.shape[1]!=0):
            
            clippedImage=cv2.resize(clippedImage, (64,64))
            
            f1=ExtractFeatures([clippedImage], 9 , 2 , 16,converColorspace)
        
            predictedOutput=classifier1.predict([f1[0]])
            if(predictedOutput==1):
                refinedWindows.append(window)
        
    return refinedWindows

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

# applying a threshold value to the image to filter out low pixel cells
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap



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


@app.route('/detection', methods=['GET'])
def detection():   
    image = mpimg.imread('static/images/car7.jpg')

    windows1 = slide_window(image, x_start_stop=[0, 1280], y_start_stop=[400,464], 
                        xy_window=(64, 64), xy_overlap=(0.15, 0.15))
    windows4 = slide_window(image, x_start_stop=[0, 1280], y_start_stop=[400,480], 
                        xy_window=(80, 80), xy_overlap=(0.2, 0.2))
    windows2 = slide_window(image, x_start_stop=[0, 1280], y_start_stop=[400,612], 
                        xy_window=(96, 96), xy_overlap=(0.3, 0.3))
    windows3 = slide_window(image, x_start_stop=[0, 1280], y_start_stop=[400,660], 
                        xy_window=(128,128), xy_overlap=(0.5, 0.5))
    
    
    windows = windows1 + windows2 +  windows3 + windows4

    refinedWindows=DrawCars(image,windows, True)
    
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    heat = add_heat(heat,refinedWindows)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,3)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    heat_image=heatmap

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    print(" Number of Cars found - ",labels[1])
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    imgplot = plt.imshow(draw_img)
    plt.show()
    return 'result'


if __name__ == '__main__':
    app.run(debug=True)

