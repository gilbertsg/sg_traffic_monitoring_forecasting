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


#######################
### LINKS DOWNLOADS ###
#######################

def call_lta_api(datetime_call):
    '''
    This function calls the LTA traffic images API based on a certain datetime and returns a datafrane row with the time as index and camera_ids as column
    '''

    # getting the api call
    api = 'https://api.data.gov.sg/v1/transport/traffic-images?date_time='+ \
    datetime_call.strftime("%Y-%m-%d") + "T" + datetime_call.strftime("%H") + "%3A" + datetime_call.strftime("%M") + "%3A00"
    
    # reading the camera data from data.gov.sg
    list_of_camera_info = ast.literal_eval(requests.get(api).content.decode("utf-8"))["items"][0]["cameras"]

    # instantiating a dataframe to contain the output
    output = pd.DataFrame()
    
    for item in list_of_camera_info: # iterating through each item in the list
        item_series = pd.Series(item['image'].replace(IMG_LINK_PREFIX,''), # getting the image names and removing the IMG_LINK_PREFIX to save space
                                index=[(pd.to_datetime(item['timestamp']) # setting the index as the timestamp (all series will be concatenated using this index)
                                       .replace(tzinfo=None))], # removing the timezone information
                                name=item['camera_id']) # setting the name/column_name as the camera ID for storage in database
        output = pd.concat([output, item_series],axis=1) # concatenating
    
    # dropping these cameras as they have been shown to have problems with having different polling time compared to the other cams
    # output = output.drop(['1001','1002','1003','1004','1005','1006'],axis=1)
    
    # checking if there are any asynchronous camera links (i.e.: cameras link occur at more than one timestamp, resulting in multiple rows/timestamps for one call)
    is_asynchronous = output.isna().sum().sum() > 0
    
    if is_asynchronous:
        output = (output.fillna(method='bfill').fillna(method='ffill'). # fill all rows (timecode) in each column (camera) with the non-null_value in that column
                  sort_index(ascending=False).iloc[[0]]) # then condense the whole dataframe to one row by selecting the latest timecode
    
    # returning the output
    return output



def download_links(datetime_call):
    '''
    This function takes a datetime object and obtains the links from the LTA API based on the datetime
    It will then save the links in the links_db
    The function will first check if the called datetime is already available in the links_db, if so, it will skip the download
    
    
    ### NOTES:
    This function loads up the entire links_db during its function call, a more efficient system would involve a SQL database, which will be implemented in the future
    
    Ideally, this function will only be called sequentially (i.e.: only called once every 5 minutes, and no historical calls), this is to make sure that the links_db is always sorted
    However, for simplicity purposes, the links_db dataframe will be sorted at the end of the function, this is highly inefficient as links_db gets larger
    When deployed using the scheduler, this sorting step will be skipped
    '''
    
    # loads the links database from csv
    links_db_df = pd.read_csv(DATABASE_PATH_ROOT+LINKS_DB_FILENAME,index_col=0)
    links_db_df.index = pd.to_datetime(links_db_df.index) # converting the index to datetime
    
    # checking if timestamp is already available in the links_db
    df_is_empty = links_db_df.loc[datetime_call-timedelta(minutes=8):datetime_call].empty
    
    # IF NOT AVAIL
    if df_is_empty:

        # downloads the list of links from the API and adding it to the end of the dataframe
        new_links_df_row = call_lta_api(datetime_call) # downloads links using the function to generate a new row
        links_db_df = pd.concat([links_db_df,new_links_df_row],axis=0) # adding the row to the bottom of the links_db dataframe

        # sorting the dataframe
        ## WILL BE SKIPPED FOR WHEN USING THE SCHEDULER
        links_db_df = links_db_df.sort_index()

        # saves the dataframe to the links_db.csv
        links_db_df.to_csv(DATABASE_PATH_ROOT+LINKS_DB_FILENAME)
    
    # additionally returns the updated links_db_df (for when this function is called to update the database)
    return links_db_df

