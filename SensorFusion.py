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
import cv2
from datetime import datetime
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import colorsys
from PIL import Image
import _tkinter
import tkMessageBox

from io import StringIO
from PIL import Image
import matplotlib.pyplot as plt

from utils import visualization_utils as vis_util
from utils import label_map_util

from multiprocessing.dummy import Pool as ThreadPool

MAX_NUMBER_OF_BOXES = 10
MINIMUM_CONFIDENCE = 0.9

PATH_TO_LABELS = 'annotations/label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'object_detection'

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=sys.maxsize, use_display_name=True)
CATEGORY_INDEX = label_map_util.create_category_index(categories)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'output_inference_graph'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def detect_objects(image_path):
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
    img_file = Image.open(open("object_detection/p.jpeg"))
    img = img_file.load()

# (2) Get image width & height in pixels
    [xs, ys] = img_file.size
    max_intensity = 100
    hues = {}

   # (3) Examine each pixel in the image file
    for x in xrange(0, xs):
      for y in xrange(0, ys):
    # (4)  Get the RGB color of the pixel
        [r, g, b] = img[x, y]
	if(r > 245 and b>245):  
          uv = 100
	  uvdeger='cop' 
	  break
      if(r>225 and b>225):
	uv = 80
	uvdeger='cop'
	break
    if(r<225 and b<225):
      uv=90
      uvdeger ='copdegil'
    if vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              CATEGORY_INDEX,
              min_score_thresh=MINIMUM_CONFIDENCE,
              use_normalized_coordinates=True,
              line_thickness=8) == 'copdegil' and uvdeger == 'copdegil': 
      print('copdegil')
      deepdeger = vis_util.visualize_boxes_and_labels_on_image_array2(
	      image_np,
	      np.squeeze(boxes),
	      np.squeeze(classes).astype(np.int32),
	      np.squeeze(scores),
	      CATEGORY_INDEX,min_score_thresh=MINIMUM_CONFIDENCE,
	      use_normalized_coordinates=True,
	      line_thickness=8) * 100
      deger = (deepdeger + uv) / 2
      print(deger)
    if vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              CATEGORY_INDEX,
              min_score_thresh=MINIMUM_CONFIDENCE,
              use_normalized_coordinates=True,
              line_thickness=8) != 'copdegil' and uvdeger == 'cop': 
      print('cop')
      deepdeger1 = vis_util.visualize_boxes_and_labels_on_image_array2(
	      image_np,
	      np.squeeze(boxes),
	      np.squeeze(classes).astype(np.int32),
	      np.squeeze(scores),
	      CATEGORY_INDEX,min_score_thresh=MINIMUM_CONFIDENCE,
	      use_normalized_coordinates=True,
	      line_thickness=8) * 100
      deger1 = (deepdeger1 + uv) / 2
      print(deger1)
      print(uv)
      print(deepdeger1)
 
        
    fig = plt.figure()
    fig.set_size_inches(16, 9)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    

    plt.imshow(image_np, aspect = 'auto')
    plt.savefig('output/{}'.format(image_path), dpi = 62)
    plt.close(fig)
    


#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'p.jpg'.format(i)) for i in range(0,1) ]
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'p.jpeg')]
#TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg'))

# Load model into memory
print('Loading model...')
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

print('detecting...')
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        for image_path in TEST_IMAGE_PATHS:
            detect_objects(image_path)



            


 



