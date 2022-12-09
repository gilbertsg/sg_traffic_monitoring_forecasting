###############
### IMPORTS ###
###############
import os

import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image
from datetime import datetime as dt
from datetime import time
from datetime import timedelta

import plotly.graph_objects as go



#################
### CONSTANTS ###
#################
## This segment contains all the constants used in the deployment of the streamlit app

# database locations
DATABASE_PATH_ROOT = '../production/database/'
LINKS_DB_FILENAME = 'links_db.csv'
IMG_PATH_DB_FILENAME = 'img_path_db.csv'
VEHICLE_COUNT_DB_FILENAME = 'vehicle_count_db.csv'
CAMERA_STATS_FILENAME = 'camera_stats.csv'
PREDICTIONS_LINK_DB_FILENAME = 'prediction_link_db.csv'

# image download variables
IMAGES_PATH_ROOT =  '../production/images/others/'

# image processing variables
OUTPUT_IMAGES_PATH_ROOT =  '../production/processed_images/'
MISSING_IMAGE_PATH = '../production/images/others/notfound.png'

# camera selection
cam_ids = [1702,2706,4708,4702,6710,6714,7793]



#################
### FUNCTIONS ###
#################
## This segment contains all the functions used in the streamlit app
def st_get_image_link(cam_id_call, datetime_call):
    """
    This function returns the img path given a particular camera ID and datetime
    If the function cannot find the image of a camera ID of the past 10 minutes, it will return the previous available image
    """
    # LOADING DATABASE
    # loads the img_path database from csv
    img_path_db_df = pd.read_csv(DATABASE_PATH_ROOT+IMG_PATH_DB_FILENAME,index_col=0)
    img_path_db_df.index = pd.to_datetime(img_path_db_df.index) # converting the index to datetime
    
    # CHECKING IF NO IMAGE FOR THE DAY
    is_no_image_for_the_day = img_path_db_df.loc[datetime_call.replace(hour=0,minute=0):datetime_call,str(cam_id_call)].dropna().empty
    
    # sets image path as the error image path
    if is_no_image_for_the_day:
        img_path = MISSING_IMAGE_PATH
        return img_path
    
    # CHECKING IF IMAGE HAS BEEN DOWNLOADED FROM THE img_path_db
    is_img_path_absent = img_path_db_df.loc[datetime_call-timedelta(minutes=12):datetime_call,str(cam_id_call)].dropna().empty
    
    # returns an error if there is no image
    if is_img_path_absent:
        # finds the previous available image
        move_back_by_x = 1 # instantiating a counter to move back the timedelta by 
        while is_img_path_absent: # iterating while the img_path is still absent
            is_img_path_absent = img_path_db_df.loc[datetime_call-timedelta(minutes=10*move_back_by_x):datetime_call,str(cam_id_call)].dropna().empty
            move_back_by_x += 1
        img_path = img_path_db_df.loc[datetime_call-timedelta(minutes=10*move_back_by_x):datetime_call,str(cam_id_call)][-1] # getting the previous available img path
    else:
        # if available, use the latest image
        img_path = img_path_db_df.loc[datetime_call-timedelta(minutes=10):datetime_call,str(cam_id_call)][-1]
    
    return img_path


def st_get_day_traffic_count(cam_id_call, datetime_call):
    """
    This function returns a pd Series which contains the traffic count of the requested cameraID on that particular day
    """
    # LOADING DATABASE
    # loads the vehicle_count database from csv
    vehicle_count_db_df = pd.read_csv(DATABASE_PATH_ROOT+VEHICLE_COUNT_DB_FILENAME,index_col=0)
    vehicle_count_db_df.index = pd.to_datetime(vehicle_count_db_df.index) # converting the index to datetime
    
    # FILTERING THE DATABASE
    # converts the cam_id_call to a string for indexing
    cam_id_call = str(cam_id_call)
    
    # getting the start and end datetime for filtering
    datetime_start = datetime_call.date() # start date is based on only the day
    datetime_end = datetime_call.date() + timedelta(days=1) # getting the whole day by setting end date as the next day
    
    # filtering the database
    vehicle_count_db_df = vehicle_count_db_df.loc[datetime_start:datetime_end, # filtering the datetime
                                                  cam_id_call] # and the cam_id
    return vehicle_count_db_df


def st_get_timestamp_from_image_link(image_link):
    """
    This function returns the timecode from a particular image link, based on the image link's filename
    """
    # getting the image timestamp from the filename
    img_filename = os.path.basename(image_link)
    yr,mo,dy,hr,mn = img_filename[5:9], img_filename[10:12], img_filename[13:15], img_filename[16:18], img_filename[19:21] # getting the datetime stamp
    yr,mo,dy,hr,mn = [int(x) for x in [yr,mo,dy,hr,mn]] # converting the datetime stamp to integers
    img_timestamp = dt(yr,mo,dy,hr,mn)
    
    return img_timestamp


def st_interpret_predictions(cam_id_call, datetime_call):
    '''
    This function loads up the latest prediction from the prediction_links database.
    The function then assigns the correct datetime index to the prediction list, and then truncate the prediction to only include the current date
    '''
    # LOADING DATABASE
    # loads the predictions link database from csv
    predictions_link_db_df = pd.read_csv(DATABASE_PATH_ROOT+PREDICTIONS_LINK_DB_FILENAME,index_col=0,parse_dates=True)

    # getting the latest prediction file path
    latest_prediction_file_path = predictions_link_db_df['path'].iloc[-1]

    # loads the latest predictions database from csv
    weekly_predictions_df = pd.read_csv(DATABASE_PATH_ROOT+latest_prediction_file_path,index_col=0)


    # REINDEXING THE PREDICTION
    # the prediction starts from midnight on monday and has a resolution of 30 minutes

    # getting the datetime of the monday midnight of this week
    today_midnight = datetime_call.replace(microsecond=0,hour=0,minute=0,second=0) # getting the datetime of today's midnight
    monday_midnight = today_midnight - timedelta(days = today_midnight.weekday()) # getting the datetime of this monday's midnight

    # getting a datetime list from monday midnight with 30 minute interval for 1 week (7*48)
    new_dt_index = [monday_midnight + timedelta(minutes=30*x) for x in range(7*48)] 

    # reindexing the weekly_predictions_df
    weekly_predictions_df.index = new_dt_index


    # FILTERING THE PREDICTION
    # filtering based on camera_id
    prediction_df = weekly_predictions_df[str(cam_id_call)]

    # filtering based on time (from today midnight to tomorrow midnight)
    prediction_df = prediction_df.loc[today_midnight:today_midnight+timedelta(days=1)]
    
    return prediction_df


def st_generate_map_plot_df(cam_id_call,img_timestamp):
    '''
    This function returns a dataframe for plotting purposes based on the current image timestamp
    The function will refer to the vehicle_count database and filter it based on the current image timestamp
    The function will also provide formatting options for the plotly graph objects
    '''
    # LOADING DATABASE
    # loads the vehicle_count database from csv
    vehicle_count_db_df = pd.read_csv(DATABASE_PATH_ROOT+VEHICLE_COUNT_DB_FILENAME,index_col=0,parse_dates=True)
    vehicle_count_db_df.index = pd.to_datetime(vehicle_count_db_df.index) # converting the index to datetime

    # Loads the camera_stats database from csv
    camera_stats_df = pd.read_csv(DATABASE_PATH_ROOT+CAMERA_STATS_FILENAME,index_col=0)
    camera_stats_df.index = camera_stats_df.index.astype(str)

    # FILTERING DATABASE BASED ON CURRENT TIME
    vehicle_count_db_df = vehicle_count_db_df.loc[img_timestamp].dropna()
    vehicle_count_db_df = pd.DataFrame(vehicle_count_db_df)
    vehicle_count_db_df.columns = ['current_count']

    # SETTING UP THE MAP_DATA_DF
    # combining the camera stats and vehicle count database
    map_data_df = vehicle_count_db_df.join(camera_stats_df[['latitude','longitude','med_thres','hi_thres']])

    # extracting traffic condition from map_data_df
    # setting up the criteria for the various traffic_conditions
    criteria = [map_data_df['current_count'].le(map_data_df['med_thres']),
                map_data_df['current_count'].le(map_data_df['hi_thres']),
                map_data_df['current_count'].gt(map_data_df['hi_thres'])]

    # setting up the categories of traffic_conditions based on the criteria above
    categories = ['low','med','high']

    # applying the criteria to infer the traffic_condition category
    map_data_df['traffic_cond'] = np.select(criteria, categories, default=0)

    # SETTING UP FORMATTING OPTIONS
    # setting up dictionary to convert the colors based on the trafic_condition category
    traffic_cond_to_colors = {'high':'red','med':'yellow','low':'green'}

    # setting the color based on the dictionary above
    map_data_df['color'] = map_data_df['traffic_cond'].map(traffic_cond_to_colors)

    # setting the color based on the current camera selected
    map_data_df['size'] = 10
    map_data_df.loc[str(cam_id_call),'size'] = 20
    
    return map_data_df



######################################
### SETTING DEFAULT SESSION STATES ###
######################################
## This section defines the default session states when the streamlit app has just been started

default_cam_id_call = cam_ids[0]
default_datetime_call = dt.now() - timedelta(minutes=5)

if 'cam_id_call' not in st.session_state:
    st.session_state.cam_id_call = default_cam_id_call
    
if 'datetime_call' not in st.session_state:
    st.session_state.datetime_call = default_datetime_call
    

    
########################################
### HANDLING STREAMLIT INPUT CHANGES ###
########################################

def handle_change():
    """
    This function handles changes in inputs of streamlit elements
    The function will update the corresponding streamlit session states depending on the input
    """
    # updates the datetime call if the slider is changed
    if st.session_state.slider_datetime_update: 
        st.session_state.datetime_call = st.session_state.slider_datetime_update
    
    # updates the datetime call with the current date if the calendar is changed
    if st.session_state.dateinput_date_update: 
        new_date = st.session_state.dateinput_date_update
        st.session_state.datetime_call = st.session_state.datetime_call.replace(year=new_date.year,
                                                                                month=new_date.month,
                                                                                day=new_date.day,)
        
    # updates the cam_id call if the selectbox is changed
    if st.session_state.selectbox_cam_id_update:
        st.session_state.cam_id_call = st.session_state.selectbox_cam_id_update
        
    if st.session_state.button_go_to_latest_click:
        st.session_state.datetime_call = default_datetime_call

    

######################################
### GETTING THE DATA FROM DATABASE ###
######################################

## Getting the traffic images
detected_traffic_image_link = st_get_image_link(cam_id_call=st.session_state.cam_id_call, 
                                                datetime_call=st.session_state.datetime_call)
try: detected_traffic_image = Image.open(OUTPUT_IMAGES_PATH_ROOT+detected_traffic_image_link)
except: detected_traffic_image = Image.open(MISSING_IMAGE_PATH) # error catching if image fails to load

## Getting the traffic counts
day_traffic_count = st_get_day_traffic_count(cam_id_call=st.session_state.cam_id_call,
                                             datetime_call=st.session_state.datetime_call)

# Getting the aggregated traffic count
day_traffic_count_agg = day_traffic_count.groupby(pd.Grouper(freq='30Min')).aggregate(np.mean).round(0) 

# Getting the traffic count corresponding to the image
try:
    img_timestamp = st_get_timestamp_from_image_link(detected_traffic_image_link) # Getting the timestamp from of the traffic image
    img_timestamp_string = img_timestamp.strftime("%H:%M %d/%m/%y")
    detected_traffic_image_traffic_count = int(day_traffic_count.loc[img_timestamp-timedelta(minutes=15):img_timestamp][-1])
except: 
    img_timestamp = st.session_state.datetime_call
    img_timestamp_string = 'Error! No image found'
    detected_traffic_image_traffic_count = 0

# Getting the days with available traffic counts 
earliest_date = pd.read_csv(DATABASE_PATH_ROOT+VEHICLE_COUNT_DB_FILENAME,index_col=0,parse_dates=True)[str(st.session_state.cam_id_call)].dropna().index.min()
latest_date = pd.read_csv(DATABASE_PATH_ROOT+VEHICLE_COUNT_DB_FILENAME,index_col=0,parse_dates=True)[str(st.session_state.cam_id_call)].dropna().index.max()



##########################
### TRAFFIC STATISTICS ###
##########################

# reading the (static) traffic stats database
camera_stats_df = pd.read_csv(DATABASE_PATH_ROOT+CAMERA_STATS_FILENAME,index_col=0)

# getting traffic count thresholds
med_traffic_thres, high_traffic_thres = camera_stats_df.loc[st.session_state.cam_id_call,['med_thres','hi_thres']].to_list()

# getting current traffic condition based on current count
if detected_traffic_image_traffic_count >= high_traffic_thres: traffic_condition, text_color = 'High Traffic', 'Red'
elif detected_traffic_image_traffic_count >= med_traffic_thres: traffic_condition, text_color='Medium Traffic', 'Yellow'
else: traffic_condition, text_color = 'Low Traffic', 'Green'



###################################
### GETTING CAMERA DESCRIPTIONS ###
###################################
# camera description for selectbox and display
options = cam_ids # options for selectbox are the same as cam_id
display_names = {1702:'CTE | 1702 | Braddell Flyover', # the display names are obtained from the LTA website
                 2706:'BKE | 2706 | after KJE Exit',
                 4708:'AYE | 4708 | Near Dover Drive',
                 4702:'AYE | 4702 | Keppel Viaduct',
                 6710:'PIE | 6710 | Entrance from Jalan Anak Bukit',
                 6714:'PIE | 6714 | Exit 35 to KJE',
                 7793:'YPE | 7793 | Tampines Ave 10 Entrance'}



###################################
### LOADING TRAFFIC PREDICTIONS ###
###################################
try: prediction_df = st_interpret_predictions(st.session_state.cam_id_call, st.session_state.datetime_call).apply(round)
except: prediction_df = pd.Series()
prediction_df.name = 'Prediction'



###############################
### PLOTTING: TRAFFIC GRAPH ###
###############################

### PLOTTING SETUP
plotting_df = pd.DataFrame(day_traffic_count_agg) # instantiating the plotting dataframe by using the aggregated traffic count
plotting_df.columns = ["Today's traffic"] # changing the name of the column
plotting_df.index.name = 'Time' # renaming the index
plotting_df.loc[img_timestamp,"Current Traffic Count"] = detected_traffic_image_traffic_count # adding the current detected traffic count
plotting_df = plotting_df.join(other=prediction_df,how='outer') # adding the predictions (using join so that we can display the entirety of the prediction)


### PLOTTING USING PLOTLY
fig = go.Figure() # instantiating plotly figure

fig.add_hrect(y0=0,y1=med_traffic_thres,# add rectangles showing low traffic
              fillcolor="green",opacity=0.15,
              annotation_text="low traffic",annotation_position="top left") 

fig.add_hrect(y0=med_traffic_thres, y1=high_traffic_thres, # add rectangles showing medium traffic (above 50th precentile of historical data)
              fillcolor="yellow",opacity=0.15,
              annotation_text="medium traffic",annotation_position="top left") 

# plotting the high traffic rect
fig_high_point = max(plotting_df["Today's traffic"].max(),plotting_df["Current Traffic Count"].max(),plotting_df["Prediction"].max()) # getting the highest point

if fig_high_point >= high_traffic_thres: # plot the rectangle only if the highest point is larger than the high traffic threshold
    fig.add_hrect(y0=high_traffic_thres, y1=fig_high_point+1, # add rectangles showing high traffic (above 80th precentile of historical data)
                  fillcolor="red",opacity=0.15,
                  annotation_text="high traffic",annotation_position="top left") 

fig.add_trace(go.Scatter(name="Today's traffic (30 min agg)", # plotting the daily traffic
                         x=plotting_df.index,
                         y=plotting_df["Today's traffic"],
                         hovertemplate = 'Today: %{y:0}<extra></extra>',
                         mode='lines+markers',connectgaps=True))

fig.add_trace(go.Scatter(name='Current Traffic', # plotting the current traffic count
                         x=plotting_df.index,
                         y=plotting_df["Current Traffic Count"],
                         hovertemplate = 'Current: %{y:0}<extra></extra>',
                         mode='markers',marker={'size':12,'symbol':'x'}))

fig.add_trace(go.Scatter(name='Predicted Traffic', # plotting the prediction
                         x=plotting_df.index,
                         y=plotting_df["Prediction"],
                         hovertemplate = 'Prediction: %{y:0}<extra></extra>',
                         mode='lines',line_shape='spline',connectgaps=True))

fig.update_layout(hovermode='x unified',
                  yaxis_title="Traffic Count");



#####################
### PLOTTING: MAP ###
#####################

### PLOTTING SETUP
map_data_df = st_generate_map_plot_df(cam_id_call=st.session_state.cam_id_call,
                                      img_timestamp=img_timestamp)

### PLOTTING USING PLOTLY
map_plot = go.Figure() # instantiating plotly figure


map_plot.add_trace(go.Scattermapbox(mode = "markers", # generating the scatter mapbox
                                    marker = {'size':15,
                                              # 'size':map_data_df['size'],
                                              'color':map_data_df['color']}, # plotting the color based on the traffic condition
                                    lon = map_data_df['longitude'],
                                lat = map_data_df['latitude'],
                                    customdata=np.dstack((map_data_df.index, map_data_df['current_count'], map_data_df['traffic_cond']))[0], # extracting the current_count and traffic_cond for use in hovertemplate
                                    hovertemplate = '''Camera ID: %{customdata[0]}<br>Traffic Count: %{customdata[1]}<br>Traffic Condition: %{customdata[2]}<extra></extra>''',
                                    uirevision=map_data_df.loc[:,'current_count']
                                   ))



map_plot.update_layout(margin ={'l':0,'t':0,'b':0,'r':0}, # setting up the layout of the map
                       mapbox_style="carto-darkmatter",
                       mapbox = {
                           'center':{'lon':103.852, 'lat':1.35},
                           'zoom':10,},
                       uirevision=map_data_df.loc[:,'current_count']
                      )



#################
### FRONT END ###
#################

# STREAMLIT SETUP
st.set_page_config(layout="wide")


# TITLE and SELECTION
st.title('🚗🚙🚌SG Traffic Camera Counting App🚐🚕🚜')
# st.write('Description here')
st.write('')
st.write('')


# SIDEBAR
with st.sidebar:
    # options = cam_ids
    # value = st.selectbox("gender", options, format_func=lambda x: display_names[x])
    
    st.write('### 📸 Camera ID')
    st.selectbox(label='Camera ID', label_visibility ='collapsed',
                 options=options, index=0,
                 format_func=lambda x: display_names[x],
                 on_change=handle_change, key='selectbox_cam_id_update')
    st.write('#') # add space

    # date_input for selecting date
    st.write('### 📅 Date')
    st.date_input(label='Date', label_visibility ='collapsed',
                  value=latest_date,
                  min_value=earliest_date,
                  max_value=latest_date,
                  on_change=handle_change, key='dateinput_date_update'
                 )
    st.write('#') # add space
    
    # Slider for selecting Time
    st.write('### ⏲️ Time')
    st.slider(label='Time', label_visibility ='collapsed',
              value=st.session_state.datetime_call,
              min_value=st.session_state.datetime_call.replace(hour=0, minute=0,second=0), 
              max_value=min(dt.now(),st.session_state.datetime_call.replace(hour=23, minute=59,second=59)), # maximum value is either the last minute of the day (for historical) or current time
              step=timedelta(minutes=10),
              format="HH:mm",
              # disabled=True,
              on_change=handle_change, key='slider_datetime_update'
             )
    st.write('#') # add space
    

    st.button(label='Go to latest', on_click =handle_change, key='button_go_to_latest_click')
    
    
# MAIN PAGE
# splitting into two columns
col1, col2 = st.columns(2)

# left column - traffic image and metadata
with col1:
    st.write('### Traffic Image')
    st.image(detected_traffic_image)
    st.write(f'### Camera ID: {st.session_state.cam_id_call}')
    st.write(f'##### Camera Location: {" | ".join(display_names[st.session_state.cam_id_call].split("|")[::2])}') # getting just the camera location thru formatting
    st.write(f'##### Detection Time: {img_timestamp_string}')
    
# right column - traffic count plot
with col2:
    st.plotly_chart(fig)
    st.write(f'#### Vehicle Count: {detected_traffic_image_traffic_count}')
    st.markdown(f'<h4 style="color:{text_color}">{traffic_condition}</h5>', unsafe_allow_html=True) # display text and color according to traffic condition

    
# bottom (outside of columns)
st.write('#')
st.write('### Map View (current traffic conditions)')
st.plotly_chart(map_plot)

# st.write(map_plot['data'])