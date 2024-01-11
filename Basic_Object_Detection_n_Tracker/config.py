import os
import pickle

CONFIG_PATH = 'tarsyer-config.pkl'

with open(CONFIG_PATH, 'rb') as file_:
    config_dict = pickle.load(file_)

CAMERA_NO = 1
VIDEO_INPUT = 'atm1_20220727054459.avi'
VIDEO_INPUT = 'rtsp://admin:Tarsyer123@192.168.1.64'

CAMERA_DETAIL = config_dict['camera_detail'] ### unique name

file_fps = 15

SKIP_FRAME = 5

CROPPING_COORD_PRI = config_dict['crop_coord'] #x1, y1, x2, y2
PADDING = False


MIN_CONTOUR_AREA = 3000
CONFIDENCE_THRESH = 0.6

MODEL_TYPE = 'TFLITE'

HEIGHT, WIDTH = config_dict['model_dim']
MODEL_PATH = 'models/person_detect_mobilenet_v1.tflite'
    
INPUT_SHAPE = [HEIGHT, WIDTH]
print(INPUT_SHAPE)

# Counting parameter

LINE = config_dict['line']
ENTRY_POINT = config_dict['entry_point']

DRAW = True
VIDEO_WRITE = False


if VIDEO_WRITE:
    if VIDEO_INPUT == 0:
        VIDEO_INPUT = 'webcam'
    output_filename = 'processed_video/'+VIDEO_INPUT.split('/')[-1].split('.')[0]+'_'+MODEL_TYPE+ '_' + str(CONFIDENCE_THRESH*100) + '_result2dev.mp4'

    saving_fps = file_fps/SKIP_FRAME
