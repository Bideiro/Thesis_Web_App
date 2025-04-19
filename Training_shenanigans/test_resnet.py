import asyncio, json, subprocess
import base64
import threading
import cv2
import numpy as np
from datetime import datetime
import websockets
import time
import queue
from collections import deque


from tensorflow.keras.preprocessing import image
# ignored cause its just a bug
from tensorflow.keras.applications.resnet_v2 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from tensorflow.keras.models import load_model # type: ignore

# Class names for ResNet classification
class_Name = [
    "All_traffic_must_turn_left",
    "All_traffic_must_turn_right",
    "Be_Aware_of_Pedestrian_Crossing",
    "Be_Aware_of_School_Children_Crossing",
    "Bike_lane_ahead",
    "Give_Way",
    "Keep_Left",
    "Keep_Right",
    "No_Entry",
    "No_Left_Turn",
    "No_Overtaking",
    "No_Parking",
    "No_Right_Turn",
    "No_U-Turn",
    "Pass_Either_Side",
    "Speed_Limit_20_KMPh",
    "Speed_Limit_30_KMPh",
    "Speed_Limit_40_KMPh",
    "Speed_Limit_50_KMPh",
    "Speed_Limit_60_KMPh",
    "Speed_Limit_70_KMPh",
    "Speed_Limit_80_KMPh",
    "Speed_Limit_90_KMPh",
    "Speed_Limit_100_KMPh",
    "Speed_Limit_110_KMPh",
    "Speed_Limit_120_KMPh",
    "Speed_Limit_Derestriction",
    "Stop"
]

ResNet_model = load_model('models/Resnet50V2(newgen_2025-04-07)_2e.keras')


img_path = "c:/Users/dei/Downloads/idk when dis is.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

predictions = ResNet_model(img_array, training=False).numpy()
class_id = predictions.argmax()
confidence = round((predictions.max() * 100), 2)
print(predictions)
print(f"\nClass Name: {class_Name[class_id]} @ {confidence}")