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

from production_images_downloader import image_downloader


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
### VEHICLE COUNTER ###
#######################

class VehicleDetector:
    '''
    This class is used to contain the vehicle detector using the pretrained YOLOv7
    Using self.class_allowed, the user can filter which types of objects (or vehicles) that is detected
    '''

    def __init__(self):
        # initialize the class by loading the pre-trained model and setting the allowable classes
        
        # Load DNN from pre-trained model
        net = cv2.dnn.readNet(YOLO_DNN_WEIGHTS_PATH, YOLO_DNN_CFG_PATH)
        
        # setup model and parameters
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setNmsAcrossClasses(True) # setting so that the NMS applies across different classes
        self.model.setInputParams(size=(832, 832), scale=1 / 255)

        # Allow classes containing Vehicles only
        self.classes_allowed = [1, 2, 3, 5, 7] # classes are same as COCO class, but SUBTRACT BY ONE, 
        # i.e.: {1:'bicycle', 2:'car',3:'motorcycle', 5:'bus', 7:'truck'}

    def get_bounding_box(self, img):
        '''
        This function takes an image and returns the bounding boxes of vehicles detected inside
        '''
        
        # Create a list to contain all detected instance of vehicles
        vehicle_boxes = []
        
        # detect if a none-type image is loaded (could be because of image error) and returns an error, this will be caught later in the main detection function
        if img is None:
            vehicle_boxes = ['image_error!']
            return vehicle_boxes
        
        # Detect Objects
        class_ids, scores, boxes = self.model.detect(img, 
                                                     nmsThreshold=0.5, # NMS threshold --> higher = more close boxes together
                                                     confThreshold=0.15)
        
        # looping through each object detected
        for class_id, score, box in zip(class_ids, scores, boxes):
            # if the object is within the allowed class, then add the item in the vehicle_boxes list
            if class_id in self.classes_allowed:
                vehicle_boxes.append({'class_id':class_id+1,
                                      'score':score,
                                      'box':box})
                
        return vehicle_boxes
    
    
    def preprocess(self, img, mask_path=None): 
        '''
        This is a helper function to preprocess the image given a mask
        In this particular instance, no further preprocessing was implemented,
        but in theory, sharpening or contrast correction could be added here to help the image detection algorithm
        '''
        # load mask from directory
        if mask_path==None: # if no maks is specified, then generate a white mask (i.e.: everything will pass)
            mask = np.zeros((1080,1920),dtype=np.uint8)
            mask[:] = 255

        else: # if a mask is specified, then use the pre-defined mask
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)

        # masking image using the pre-defined mask
        img = cv2.bitwise_or(img,img,mask=mask)

        return img
    
    
    def process_image(self, img, mask_path=None):
        '''
        This function returns the processed image and total vehicle count given an image and a mask_path (used for masking the camera to only the ROI)
        There are various error catching function here which will raise a warning if the function is unable to conduct the preprocessing or detection
        '''
        # INITIALIZATION
        # define vehicle dictionary
        object_dictionary = {2:'bicycle',3:'car',4:'motorcycle',6:'bus',8:'truck'}

        # print error if image failed to load
        if img is None:
            print(f'error in loading image {img_filename}')

        # create a clean copy (without masking or preprocessing) to be outputed later with the bounding boxes
        output_img = img.copy() 

        # PREPROCESSING
        # attempt to preprocess and mask the image
        try:
            img = self.preprocess(img=img,
                                  mask_path=mask_path)
        # if masking fails (due to absence of mask or other things) use the original image
        except:
            img = output_img
            warnings.warn("Warning: Image Preprocessing Error")
            
        # DETECTING VEHICLES
        # use the get_bounding_box function to return the vehicle boxes
        vehicle_boxes = self.get_bounding_box(img)

        # error catching for detection error
        if vehicle_boxes == ['image_error!']:
            warnings.warn("Warning: Image Detection Error")

        # counting number of vehicles
        vehicle_count = len(vehicle_boxes)

        # DRAWING BOUNDING BOXES
        for vehicle_box in vehicle_boxes:
            x, y, w, h = vehicle_box['box']

            cv2.rectangle(output_img, (x, y), (x + w, y + h), (25, 0, 180), 3)
            cv2.putText(output_img, f"{object_dictionary[vehicle_box['class_id']]} | {vehicle_box['score']:.2f}", (x, y + h), 0, 1, (255, 255, 255), 1)

        # ADDING TEXT WITH VEHICLE COUNT
        cv2.putText(output_img, "Vehicle count: " + str(vehicle_count), (20, 50), 0, 2, (100, 200, 0), 3)

        return output_img, vehicle_count
    

    
def vehicle_count(cam_id_call, datetime_call):
    '''
    This function takes in the camera_id and datetime and runs a vehicle detection on the corresponding image
    If the image is not yet downloaded, the function will attempt to download the image
    '''
    # LOADING DATABASE
    # loads the img_path database from csv
    img_path_db_df = pd.read_csv(DATABASE_PATH_ROOT+IMG_PATH_DB_FILENAME,index_col=0)
    img_path_db_df.index = pd.to_datetime(img_path_db_df.index) # converting the index to datetime
    
    # loads the img_path database from csv
    vehicle_count_db_df = pd.read_csv(DATABASE_PATH_ROOT+VEHICLE_COUNT_DB_FILENAME,index_col=0)
    vehicle_count_db_df.index = pd.to_datetime(vehicle_count_db_df.index) # converting the index to datetime
    
    
    # CHECKING CAMERA ID
    # converts the cam_id_call to a string for indexing
    cam_id_call = str(cam_id_call)
    
    # checks if cam_id_call is part of the available camera
    if cam_id_call not in (img_path_db_df.columns):
        raise Exception("No such camera ID") # throws error if there is no such camera ID
    
    
    # CHECKING IF IMAGE HAS BEEN DOWNLOADED FROM THE img_path_db
    is_img_path_absent = img_path_db_df.loc[datetime_call-timedelta(minutes=8):datetime_call,str(cam_id_call)].dropna().empty
    
    # downloads the image if the image is absent
    if is_img_path_absent: 
        img_path_db_df = image_downloader(cam_id_call=cam_id_call, datetime_call=datetime_call)
        
        
    # CHECKING IF IMAGE HAS BEEN PROCESSED FROM THE vehicle_count_db
    is_vehicle_count_absent = vehicle_count_db_df.loc[datetime_call-timedelta(minutes=8):datetime_call,str(cam_id_call)].dropna().empty
    
    # skips the vehicle count if image has already been processed (i.e.: NOT absent)
    if not is_vehicle_count_absent: 
        return None
    
    
    # GETTING THE IMAGE PATH FROM img_path_db
    img_path = img_path_db_df.loc[datetime_call-timedelta(minutes=8):datetime_call,str(cam_id_call)][0]
    
    # getting the image timestamp from the filename
    img_filename = os.path.basename(img_path)
    yr,mo,dy,hr,mn = img_filename[5:9], img_filename[10:12], img_filename[13:15], img_filename[16:18], img_filename[19:21] # getting the datetime stamp
    yr,mo,dy,hr,mn = [int(x) for x in [yr,mo,dy,hr,mn]] # converting the datetime stamp to integers
    img_timestamp = dt(yr,mo,dy,hr,mn)
    
    # VEHICLE DETECTION
    # Load Veichle Detector class
    vd = VehicleDetector()

    # read the image from path
    img = cv2.imread(IMAGES_PATH_ROOT+img_path)
    
    # obtain mask_path from cam_id
    mask_path = IMAGE_MASK_PATH_ROOT+str(cam_id_call)+'.jpg'
    
    # getting the processed image and vehicle count from the process_image function
    output_img, vehicle_count = vd.process_image(img=img,mask_path=mask_path)
    
    # saving processed image
    img_path_out = OUTPUT_IMAGES_PATH_ROOT + img_path # getting the output image path
    os.makedirs(os.path.dirname(img_path_out), exist_ok=True) # create folder if doesn't exist
    cv2.imwrite(img_path_out, output_img) # writing the image to the output folder
    
    # SAVING VEHICLE_COUNT TO DATABASE
    # adding the vehicle count to the img_path_db
    vehicle_count_db_df.loc[img_timestamp,str(cam_id_call)] = int(vehicle_count)
    
    # sorting the dataframe
    ## WILL BE SKIPPED FOR WHEN USING THE SCHEDULER
    vehicle_count_db_df = vehicle_count_db_df.sort_index()
    
    # saving the img_path_db
    vehicle_count_db_df.to_csv(DATABASE_PATH_ROOT+VEHICLE_COUNT_DB_FILENAME)