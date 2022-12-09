###############
### IMPORTS ###
###############

import ast
import requests
import os
import pandas as pd
from datetime import datetime as dt
import datetime 
from datetime import timedelta
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
import itertools
from tqdm import tqdm

import time
import schedule

from production_vehicle_detector import vehicle_count


#################
### CONSTANTS ###
#################

# database locations
DATABASE_PATH_ROOT = '../production/database/'
LINKS_DB_FILENAME = 'links_db.csv'
IMG_PATH_DB_FILENAME = 'img_path_db.csv'
VEHICLE_COUNT_DB_FILENAME = 'vehicle_count_db.csv'
PREDICTIONS_LINK_DB_FILENAME = 'prediction_link_db.csv'

# image download variables
IMG_LINK_PREFIX ='https://images.data.gov.sg/api/traffic-images/'
IMAGES_PATH_ROOT =  '../production/images/'

# image processing variables
YOLO_DNN_WEIGHTS_PATH = "../dnn_model/yolov7.weights"
YOLO_DNN_CFG_PATH = "../dnn_model/yolov7.cfg"
IMAGE_MASK_PATH_ROOT = '../production/image_masks/'
OUTPUT_IMAGES_PATH_ROOT =  '../production/processed_images/'

# list of cameras
cam_id_list = [1702,2706,4708,4702,6710,6714,7793]


##############################
### CONSTANT UPDATE MODULE ###
##############################

def get_cam_now():
    '''
    This function will do a vehicle_count for all the cameras in cam_id_list at the current datetime
    '''
    dt_list = [dt.now()]
    
    print(dt.now().strftime('Getting data at %H:%M'))

    combo_list = list(itertools.product(cam_id_list, dt_list))
    combo_list_pbar = tqdm(combo_list)

    for cam_id, dt_call in combo_list_pbar:
        try: 
            vehicle_count(cam_id_call=cam_id, datetime_call=dt_call)
            
        except Exception as e: 
            print(f'error in processing {cam_id} at {dt_call.strftime("%Y-%m-%d %H:%M")} | {e}')



########################
### USER INTERACTION ###
########################

# scheduling the get_cam_now function to run every 10 minutes

schedule.clear()
get_cam_now()
schedule.every(10).minutes.do(get_cam_now)

while True:
    schedule.run_pending()
    time.sleep(1)