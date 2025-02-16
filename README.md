# YOLO_Workshop
YOLO workshop that was operated by ISS Egypt @ Universiti Teknologi Malaysia, Johor Bahru

# INTRODUCTION

You Only Look Once is an effiecient deep learning based object detection model. It predicts bounding boxes and and class propabilities in a single forward pass of the network
This makes it incredibly fast and suitable for real-time applications.

In this workshop, you will:

Collect and preprocess data

Train a YOLO model

Evaluate and visualize predictions

Run YOLO for real-time object detection

# Requirements

To follow along with this workshop, you need:

Python 3.7+

Jupyter Notebook or Google Colab

OpenCV

TensorFlow / PyTorch

YOLOv5 or YOLOv8 framework

You can intall dependencies using:
``` pip install -r requirements.txt ```

# Workshop Steps

## Step 1: Gathering Data

Capture images using a webcam or use an existing dataset.

Save images into a folder for annotation.

Example function to capture images in Google Colab:
```
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import os

// Function to access webcam and capture images
def take_photos(num_photos=3, folder_name='captured_photos'):
    # Implementation...
```

## Step 2: Preparing the Dataset

Annotate images using LabelImg or Roboflow.

Convert annotations to YOLO format.

Split the dataset into training and validation sets.

## Step 3: Training the YOLO Model
Download and configure YOLOv5 or YOLOv8.

Train the model on annotated data.

Example for YOLO5 training
```
python train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --weights yolov5s.pt
```

## Step 4: Evaluating the Model

Analyze training metrics (mAP, loss, precision, recall).

Visualize predictions on test images.

## Step 5: Running Object detection

Perform inference using the trained YOLO model.
```
python detect.py --weights runs/train/exp/weights/best.pt --source test_images/
```
