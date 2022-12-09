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

from production_links_downloader import download_links


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


########################
### IMAGES DOWNLOADS ###
########################

def get_image_link(cam_id_call,datetime_call):
    '''
    This function takes the camera_id and date_time to be called and outputs the download link and download_path for the image file
    The download link is obtained from the links_db, if no link is present in the links_db (because it's not been downloaded yet), it will attempt to download the links
    
    ### NOTES:
    This function loads up the entire links_db during its function call, a more efficient system would involve a SQL database, which will be implemented in the future
    '''
    # LOADING DATABASE
    # loads the links database from csv
    links_db_df = pd.read_csv(DATABASE_PATH_ROOT+LINKS_DB_FILENAME,index_col=0)
    links_db_df.index = pd.to_datetime(links_db_df.index) # converting the index to datetime
    
    cam_id_call = str(cam_id_call)
    
    # CHECKING (AND DOWNLOADING) LINKS
    # checks the availability of the link
    link_is_empty = links_db_df.loc[datetime_call-timedelta(minutes=8):datetime_call,cam_id_call].empty

    # downloads the links if the link is empty
    if link_is_empty:
        links_db_df = download_links(datetime_call=datetime_call)
    
    # GETTING THE IMAGE LINK
    # obtain the image link from the links_db
    img_link = links_db_df.loc[datetime_call-timedelta(minutes=8):datetime_call,cam_id_call][0]

    # ERROR CATCHING IF NO DOWNLOAD LINK AVAILABLE
    move_back_counter = 1
    while type(img_link)!=str:
        datetime_call_new = datetime_call-timedelta(minutes=5*move_back_counter)
        img_link = links_db_df.loc[datetime_call_new-timedelta(minutes=8):datetime_call_new,cam_id_call][0]
        move_back_counter += 1
    
    # GETTING THE IMAGE FILENAME AND PATH
    # obtain the image timestamp from the links_db
    img_timestamp = links_db_df.loc[datetime_call-timedelta(minutes=8):datetime_call,cam_id_call].index[0]
    
    # decide the name of the image file based on the camera_id and time
    img_filename = ("-".join([str(cam_id_call), # get the camera_id
                              img_timestamp.strftime("%Y_%m_%d_%H_%M"), # get the timestamp
                             ]) # combine the cam_id and timestamp with a dash '-'
                    + '.jpg') # add .jpg as filetype
    
    # decide the path of the image file based on the camera_id and time
    img_path = ("/".join([img_timestamp.strftime("%Y_%m_%d"), # get the date
                          str(cam_id_call), # get the camera_id
                             ]) # combine the cam_id and timestamp with a slash '/' (indicating folder structure)
               +'/') # add / for final folder path
    
    return img_link, img_filename, img_path



def download_image_from_link(img_link, img_filename, img_path):
    '''
    This function downloads the image from the data.gov api based on its link and puts it in the proper filepath and filename
    '''
    
    # getting full image link
    img_link = IMG_LINK_PREFIX+img_link
    
    # getting full image download path
    img_path = IMAGES_PATH_ROOT+img_path
    
    # getting the file from the url
    r = requests.get(img_link, allow_redirects=True)
    
    # create folder if it doesn't exist
    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    # combining the path and filename to get the full path
    full_path = img_path + '/' + img_filename 
    
    # writing the file to the path
    with open(full_path, 'wb') as f: 
        f.write(r.content)
        
        
def image_downloader(cam_id_call,datetime_call):
    '''
    This function takes a the cam_id and timestamp and attempts to download the traffic images using links from links_db and save the path to img_path_db
    The function will first check if the called datetime and cam_id is already available in the img_path_db, if so, it will skip the download
    
    
    ### NOTES:
    This function loads up the entire img_path_db during its function call, a more efficient system would involve a SQL database, which will be implemented in the future
    
    Ideally, this function will only be called sequentially (i.e.: only called once every 5 minutes, and no historical calls), this is to make sure that the img_path_db is always sorted
    However, for simplicity purposes, the img_path_db dataframe will be sorted at the end of the function, this is highly inefficient as links_db gets larger
    When deployed using the scheduler, this sorting step will be skipped
    '''
    # LOADING DATABASE
    # loads the img_path database from csv
    img_path_db_df = pd.read_csv(DATABASE_PATH_ROOT+IMG_PATH_DB_FILENAME,index_col=0)
    img_path_db_df.index = pd.to_datetime(img_path_db_df.index) # converting the index to datetime
    
    
    # CHECKING CAMERA ID
    # converts the cam_id_call to a string for indexing
    cam_id_call = str(cam_id_call)
    
    # checks if cam_id_call is part of the available camera
    if cam_id_call not in (img_path_db_df.columns):
        raise Exception("No such camera ID") # throws error if there is no such camera ID
    
    
    # CHECKING IF IMAGE IS ALREADY PRESENT FROM THE img_path_db
    is_img_path_absent = img_path_db_df.loc[datetime_call-timedelta(minutes=8):datetime_call,str(cam_id_call)].dropna().empty
    
    # skipping out of this function if the img_path is NOT absent (i.e.: if image is already downloaded)
    if not is_img_path_absent:
        return None
    
    # GETTING THE IMAGE FROM THE API AND SAVING THE PATH TO img_path_db
    # getting the img link, filename, and path from the get_image link function
    img_link, img_filename, img_path = get_image_link(cam_id_call=cam_id_call,datetime_call=datetime_call)
    
    # downloading the image
    download_image_from_link(img_link, img_filename, img_path)
    
    # getting the image timestamp from the filename
    yr,mo,dy,hr,mn = img_filename[5:9], img_filename[10:12], img_filename[13:15], img_filename[16:18], img_filename[19:21] # getting the datetime stamp
    yr,mo,dy,hr,mn = [int(x) for x in [yr,mo,dy,hr,mn]] # converting the datetime stamp to integers
    img_timestamp = dt(yr,mo,dy,hr,mn)
    
    # getting the image full path
    img_full_path = img_path + img_filename
    
    # adding the full path to the img_path_db
    img_path_db_df.loc[img_timestamp,str(cam_id_call)] = img_full_path
    
    # sorting the dataframe
    ## WILL BE SKIPPED FOR WHEN USING THE SCHEDULER
    img_path_db_df = img_path_db_df.sort_index()
    
    # saving the img_path_db
    img_path_db_df.to_csv(DATABASE_PATH_ROOT+IMG_PATH_DB_FILENAME)
    
    # additionally returns the updated img_path_db_df (for when this function is called to update the database)
    return img_path_db_df