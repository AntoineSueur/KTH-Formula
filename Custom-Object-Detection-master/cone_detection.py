'''
Change the fields below for your code to make it more presentable:

@author :       Ajinkya Khoche
email:          khoche@kth.se
Date:           2018/09/03   
Description:    This program
                - Reads a video from 'test_videos' folder and reads every frame 
                till end of video. It stores every frame in variable of same name.
                - Your algorithm should process 'frame' variable (or frame_cloned. 
                its good to clone the frame to preserve original data)
                - The result of your algorithm should be lists of 'bounding_boxes'
                and 'labels'. 
                - The helper code takes 'bounding_boxes' to draw rectangles on the
                positions where you found the cones. It uses corresponding 'labels'
                to name which type of cone was found within 'bounding_boxes'.  

                Color Convention that we follow:
                ---------------- 
                    0-  YELLOW
                    1-  BLUE
                    2-  ORANGE
                    3-  WHITE
                    4-  BLACK

                This basically means that if labels[i] = 0, then you can set the i_th
                bounding_box as 'yellow cone'    
'''
import cv2 
import numpy as np

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import time
import glob

from io import StringIO
from PIL import Image

import matplotlib.pyplot as plt

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util

from multiprocessing.dummy import Pool as ThreadPool

MAX_NUMBER_OF_BOXES = 10
MINIMUM_CONFIDENCE = 0.01

PATH_TO_LABELS = './annotations/label_map.pbtxt'

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=sys.maxsize, use_display_name=True)
CATEGORY_INDEX = label_map_util.create_category_index(categories)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'output_inference_graph'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'


def main():

    print('Loading model...')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
			
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Read video from disk and count frames
            cap = cv2.VideoCapture('./20180626_102839414.mp4')
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            count = 0
    
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Read every frame till the end of video
            while count < frameCount:
                ret, frame = cap.read()
                if ret == True:
                    count = count + 1

                    frame_cloned = np.copy(frame)
                    image_np_expanded = np.expand_dims(frame_cloned, axis=0)

                    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})

                    bounding_box = [list(x) for x in boxes[0]]
                    labels = [0]*len(bounding_box)
                    for i, (box, score) in enumerate(zip(bounding_box,scores[0])):#box, i in zip(bounding_box, range(len(bounding_box))):
                        if(score < MINIMUM_CONFIDENCE):
                            continue
                        xmin = int(box[0]*width)
                        ymin = int(box[1]*height)
                        xmax = int(box[2]*width)   # w = box[2]
                        ymax = int(box[3]*height)   # h = box[3]

                        if labels[i] == 0:			
                            cv2.rectangle(frame_cloned ,(xmin,ymin), (xmax,ymax), (0, 255, 255), 5)     #cv2.rectangle(frame_cloned ,(xmin,ymin), (xmin + w,ymin + h), (0,255,0), 5)
                            cv2.putText(frame_cloned, 'cone', (int(xmin),int(ymin)-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                    cv2.imshow('Original frame', frame)
                    cv2.waitKey(10)
                    cv2.imshow('Result of cone detection', frame_cloned)
                    cv2.waitKey(10)
if __name__ == '__main__':
    main()