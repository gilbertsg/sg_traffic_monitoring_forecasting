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


#######################
### CATCH UP MODULE ###
#######################

def catch_up_count(dt_start=dt.now().replace(hour=0,minute=0), dt_end=dt.now(), dt_resolution_mins=10):
    '''
    This function is used to catch up the vehicle counts from the staring to end datetime, and the predetermined resolution
    '''
    # getting the datetime list from the datetime start and end
    num_of_observations = round((dt_end-dt_start)/timedelta(minutes=dt_resolution_mins)) # getting the number of observations
    dt_list = [dt_end - timedelta(minutes=x*dt_resolution_mins) for x in range(num_of_observations)][::-1] # making the datetime list

    # creating all the combos of datetime and camera_id
    combo_list = list(itertools.product(dt_list,cam_id_list))
    combo_list_pbar = tqdm(combo_list) # converting to a tqdm to display progress bar

    # iterating through all the combo of datetime and camera_id, and doing the vehicle count on those combos
    for dt_call, cam_id in combo_list_pbar:
        try: 
            vehicle_count(cam_id_call=cam_id, datetime_call=dt_call)
        except Exception as e: 
            print(f'error in processing {cam_id} at {dt_call.strftime("%Y-%m-%d %H-%M")} | {e}')

########################
### USER INTERACTION ###
########################

catch_up_option = input('''Do you want to use the catch up module with the default setting?
Default settings: downloads the entirety of today from midnight to current time
Please select [y] for default settings or [n] to input your own start and end time
''')

if catch_up_option.lower() == 'y':
    catch_up_count()
elif catch_up_option.lower() == 'n':
    dt_start_str = input('Please input the start time in the following format: "YYYY MM DD HH MM"\n')
    dt_end_str = input('Please input the end time in the following format: "YYYY MM DD HH MM"\n')
    dt_resolution_mins = input('Please input the required time resolution in minutes (default = 10)\n')
    
    dt_start = dt.strptime(dt_start_str,'%Y %m %d %H %M')  
    dt_end = dt.strptime(dt_end_str,'%Y %m %d %H %M')
    dt_resolution_mins = int(dt_resolution_mins)
    
    try:
        catch_up_count(dt_start=dt_start, dt_end=dt_end, dt_resolution_mins=dt_resolution_mins)
    except Exception as e:
        print(f'Error in requesting catch_up_count module, please check your inputs | {e}')