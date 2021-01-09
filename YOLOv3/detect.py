''' 
detect.py
---------
this is the main file that utilizes the other files in this project. as of writing this,
running it results in applying the yolo network to all the images in the images folder.
however, i've updated all the dependencies such that, with some slight modifications, 
code simmilar to this can be used to train the network.

last updated 01/09/2021
'''

# +++ imports
import time
import torch 
import numpy as np
import cv2 
from util import *
import os 
import copy
from yolo_model import yolo_net
import random
import pickle as pkl

# +++ preferences
CUDA = False # should cuda be used?
print_detections = False # should detections be printed to console?

# +++ file locations
image_dir = 'images' # folder that images are in
results_dir = 'result' # folder for the results to go to
classes_file = 'data/coco.names' # file of classes
config_file = 'cfg/yolov3-320.cfg' # config file for the network
weights_file = 'weights/yolov3-320.weights' # weights file for the network
color_file = 'misc/pallete'

# +++ detection hyperparameters
input_res = 256         # resolution of the network input
batch_size = 1          # size of batches (should be a factor of number of images)
confidence_thresh = 0.4 # confidence threshold
nms_thresh = 0.4        # non-max supression threshold

# +++ load the classes
classes = load_classes(classes_file) # load the classes from the file
num_classes = len(classes) # number of classes

# +++ setup the model
model = yolo_net(config_file) # setup the network using the config file
model.load_weights(weights_file) # load the weights from the weights file
model.set_input_dims(input_res, input_res) # set the model's input resolution
model.eval() # put into evaluation model

# +++ find all the images in the folder
image_paths = get_image_paths(image_dir)

# +++ setup the batches
image_batches, orig_images, image_dim_list = \
    preprocess_images(image_paths, (input_res, input_res), batch_size)

# +++ setup some shit for main loop
output = None # output variable (will store tensor eventually)
objs = {} # will store detected objects

for batch_i in range(len(image_batches)):
    batch = image_batches[batch_i] # get the batch

    # +++ make predicion using network
    with torch.no_grad(): # don't track gradients
        prediction = model(batch, CUDA) # make prediction w/ model

    # +++ the get_results function takes in the outputs from the network 
        # and applies some shit like NMS and IoU to sort out the actual 
        # bounding boxes that the network is predicting returns that info 
        # in a tensor with shape (num_detections, 8) where the eight values
        # in the first dimension are as follows:
        # (image_num, x_min, y_min, x_max, y_max, box_conf, class_conf, class_index)
        # where image_num is the image's number within it's batch
    
    prediction = get_results(prediction, confidence_thresh, \
        num_classes, nms=True, nms_thresh=nms_thresh)
    
    # +++ real quick, deal with the case if prediction is None, meaning
        # no objects were detected with reasonable confidence
    if prediction == None:
        continue # loop back and do the next batch

    # +++ now we want to get the first element to store the absolute image number,
        # not just it's position in the batch, so we add the number of images that
        # came before it
    prediction[:, 0] += batch_i * batch_size

    # +++ and lastly we can write the output to the output tensor
    if output == None: # if no ouptput has been written yet
        output = prediction.clone() # just use the prediction tensor
    else: # if output already has been written
        output = torch.cat((output, prediction), 0)

'''
NOTE
this is the end of the yolov3 code! the network has run and made it's predictions,
the rest of the code here is just sorting out what, exactly, those predictions are,
and displaying them so that we can see what the network "sees".

so now we have this output tensor and it's storing the data for our detections
like so:
[[image_number, x_min, x_max, y_min, y_max, box_conf, class_conf, class_num],
 ...]

but the x_min, x_max, etc. variables are in relation to the input_dim x input_dim
image that we fed into the network, so we need to somehow extract from that the
box dimensions for the box in the original image, essentially unscaling the boxes.
that's what this next section of code does.
'''

# +++ show detections (if wanted)
if print_detections:
    for img_num, img in enumerate(image_paths): # go through images
        obj_nums = [int(x[-1]) for x in output if int(x[0]) == img_num]
        objs = [classes[n] for n in obj_nums]
        print(f'image {img_num} : {img}')
        print(f'objects detected : {objs}')

# +++ this sets up the image dimension list to deal with unscaling the boxes, it
    # makes a copy of the dimensions of the image for each detection made in
    # each image (using the output tensor's 0th column as indecies)
img_dim_tensor = image_dim_list[output[:,0].long()]

# +++ now that we have a copy of each image's dimensions in the list, we need
    # to find the scaling factor that was used to scale each image down to fit the
    # network's input. the factor is input resolution / max(image's dimensions) or
    # min(input res / image's dimensions) which we can get like so
scaling_factors = torch.min(input_res/img_dim_tensor, 1)[0] # get the factors
scaling_factors = scaling_factors.view(-1,1) # view it as a vector

# +++ now that we have the scaling factors we need to re-scale the bottom and top
    # corners of the bounding boxes (as we have in out output images) to be those
    # same coordinates in the original image
output[:,[1,3]] -= (input_res - scaling_factors * img_dim_tensor[:,0].view(-1,1))/2
output[:,[2,4]] -= (input_res - scaling_factors * img_dim_tensor[:,1].view(-1,1))/2
output[:,1:5] /= scaling_factors

# +++ this part makes sure that none of the bounding boxes overstep their bounds...
    # literally
for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, img_dim_tensor[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, img_dim_tensor[i,1])

# +++ now we setup this write function which takes in the outputs and draws
    # bounding boxes and shit on the images, nice!
colors = pkl.load(open(color_file, 'rb'))
def write(x, originals):
    c1 = tuple(x[1:3].int()) # top left corner of box
    c2 = tuple(x[3:5].int()) # bottom right corner of box
    img = originals[int(x[0])] # get the original image
    classidx = int(x[-1]) # get the class index of the object
    label = f'{classes[classidx]} ({x[-2]:.2f},{x[-3]:.2f})' # make the label for the thing
    color = random.choice(colors) # pick a random color

    ydim = image_dim_list[x[0].long()][1] # get the y dimension of the image
    txt_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
        # get the size of the of the text
    font_size = ydim/50
    font_scale = float(font_size/txt_size[1]) # calculate font scale
    thickness = max(1, int(font_size / 10))
    txt_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        # get the size of the of the text witht the right scale
    
    ct1 = (c1[0], c1[1] - txt_size[1] - 4) # top left corner of text box
    ct2 = (c1[0] + txt_size[0] + 3, c1[1]) # bottom right corner of txt box

    # +++ now we add some shit to the image
    cv2.rectangle(img, c1, c2, color, thickness*2) # add the bounding box rectangle
    cv2.rectangle(img, ct1, ct2, color, -1) # add the text background box
    cv2.putText(img, label, c1, 
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, [255,255,255], thickness)
        # add the text to the box

# +++ now we apply it to all the images
det_images = copy.deepcopy(orig_images) # make a copy of the original images
for x in output: # go through the output
    write(x, det_images) # write the output onto the detection images

# +++ create the results folder (if it doesn't exist yet)
results_path = os.path.realpath(results_dir) # full path to the results directory
if not os.path.exists(results_path): # if it doesn't exist
    os.mkdir(results_path) # make it

# +++ now make a path for each detection image
detection_paths = [] # path for each detection image
for impath in image_paths: # loop to make paths
    imname = impath.split('/')[-1] # get the image's name
    detpath = os.path.join(results_path, f'det_{imname}') # detection image path
    detection_paths.append(detpath) # add to list

# +++ finally, write the detections to the output
for detpath, img in zip(detection_paths, det_images):
    cv2.imwrite(detpath, img)
