''' detector.py
this file contains the Detector class, which kind of works as a wrapper to a
pre-trained and saved yolo network. it can do things like make detections on
images, save copies of the images with the detections shown, etc.
'''
import torch
import cv2
import copy
import random
import os
from util import unique, bbox_IoU
from datamanager import DataManager


class Detector(object):
    ''' Detector class
    this class takes in a yolo network and uses it to make detections on images.

    --- args ---
    yolo_net_path : str
        the path to the yolo network to use to make detections.
    class_file : str
        the path to the file containing the names of classes that the network is
        detecting.
    confidence_thresh : float between 0 and 1
        confidence threshold; detections with a confidence value lower than this
        threshold will be ignored.
    nms_thresh : float between 0 and 1
        non-max suppresion threshold; detections of the same class with IoU 
        greater than this threshold will be ignored.
    CUDA : bool, optional (default=False)
        if true, detections will use the GPU instead of the CPU (default).
    '''
    def __init__(self, yolo_net_path, classnames_path, confidence_thresh=0.5, nms_thresh=0.3, CUDA=False):
        # +++ set device variable
        self.device = 'cuda:0' if CUDA else 'cpu' # set device var

        # +++ save classnames path
        self.classnames_path = classnames_path

        # +++ load and configure the yolo network
        f = open(yolo_net_path, 'rb') # open the file
        self.model = torch.load(f) # load the network
        f.close() # close the file
        self.model.to(self.device) # send the network to the device
        self.model.eval() # put in evaluation mode

        # +++ save important attributes
        self.input_dim = self.model.input_dim # input dimension
        self.num_classes = self.model.num_classes # number of classes
        self.conf_thresh = confidence_thresh # confidence threshold
        self.nms_thresh = nms_thresh # nms threshold
    
    # +++ internal methods

    def _NMS_prediction(self, predictions):
        ''' +++ INTERNAL FUNCTION +++ 
        takes in one set of predictions and applies non-max suppression to weed
        out multiple predictions.

        --- args ---
        predictions : torch.FloatTensor with size (_, 7)
            the predictions, where the first dimension stores the prediction
            attributes (xmin, xmax, ymin, ymax, conf, cls conf, cls)
        
        --- returns ---
        torch.FloatTensor with size (_, 7) : the refined predicticions
        '''
        # +++ first, we sort the predictions by confidence for good NMS
        conf_sorted_idxs = torch.sort(
            predictions[:,4], descending=True)[1]
        predictions = predictions[conf_sorted_idxs] # sort in place

        # +++ loop for NMS
        i = 0 # index counter
        while i < predictions.size(0):
            # get the ious of this box with all the boxes after it
            ious = bbox_IoU(predictions[i].unsqueeze(0), predictions[i+1:])

            # get the indexes for detections with IoU < nms_thresh
            iou_idxs = torch.where(ious < self.nms_thresh)[0]
            iou_idxs = iou_idxs + i+1 # add i+1 to get absolute indecies of these

            # eliminate the predictions that have ious > nms_thresh
            predictions = torch.cat((
                predictions[:i+1], # all predictions up to and including this one
                predictions[iou_idxs] # all other predictions with iou < thresh
            ), dim=0)

            i += 1 # increment index
        
        return predictions
    
    # +++ methods
    def get_detections(self, x, targets=None, classwise_nms=True):
        ''' get detections method
        this method is essentially just a forward pass through the yolo network
        where algorithms line NMS get applied to the output to figure out what
        *actual* detections the network is making.
        
        --- args ---
        x : torch.FloatTensor, with size (batch_size, 3, input_dim, input_dim)
            the input to the network.
        targets : torch.FloatTensor, optional (defualt=None)
            the targets bounding boxes that should be detected.
        classwise_nms : bool, optional (defualt=True)
            if True, non-max suppression will be applied to the detections by
            class (i.e. an object may be given multiple labels), if False, NMS
            will be applied to the detections regardless of class, and if None,
            NMS won't be applied to the detections.
        
        --- returns ---
        torch.FloatTensor : the predicitons that the network is making, in the
            shape of a tensor with size (num_detections, 8) like so:
            [...
             0          1     2     3     4     5         6           7
            [image num, xmin, ymin, xmax, ymax, box conf, class conf, class num]
            ...]
        float (if targets != None) : loss of the network on the given batch of 
            images.
        '''
        # +++ get ready
        x = x.to(self.device) # send x to right device
        targets = targets.to(self.device) if targets != None else None

        # +++ make the prediction using the model
        with torch.no_grad(): # don't track gradients
            # use the model to make the prediction
            if targets != None:
                predictions, loss = self.model(x, targets=targets)
            else:
                predictions = self.model(x)
                loss = 0.
        
        # +++ first, we zero out all predictions with a confidence value
            # lower than the threshold
        conf_mask = (predictions[:,:,4] > self.conf_thresh)
        conf_mask = conf_mask.float().unsqueeze(2)
        predictions *= conf_mask

        # +++ check if there are any predictions
        if torch.nonzero(predictions, as_tuple=False).numel() == 0:
            # return None
            return None, loss

        # +++ now we want to convert the networks ouput which is in the form
            # (centerx, centery, width, height, ...) into the more managable
            # form of (x_min, x_max, y_min, y_max)
        
        x_center = predictions[:,:,0].clone()    # x center coordinates
        y_center = predictions[:,:,1].clone()    # y center coords
        width = predictions[:,:,2].clone()       # box widths
        height = predictions[:,:,3].clone()      # box heights
        
        predictions[:,:,0] = x_center - width/2  # x min
        predictions[:,:,1] = y_center - height/2 # y min
        predictions[:,:,2] = x_center + width/2  # x max
        predictions[:,:,3] = y_center + height/2 # y max

        # +++ now we iterate over each image in the batch to do more miniscule
            # things like get the predicted class and non max suppression

        batch_size = predictions.size(0) # get the batch size
        detections = None # detections variable (will be tensor)

        for img_i in range(batch_size): # go through images
            image_pred = predictions[img_i] # get the predictions for the image

            # +++ now we get the predicted class and confidence for each
            cls_pred_conf, cls_pred = torch.max(image_pred[:,5:], dim=1)
                # get the confidence and predictions
            cls_pred_conf = cls_pred_conf.float().unsqueeze(1) # make float vector
            cls_pred = cls_pred.float().unsqueeze(1) # make float vector

            # +++ and now we replace the last num_classes dimensions in image_pred
                # with those values
            image_pred = torch.cat(
                (image_pred[:,:5], cls_pred_conf, cls_pred), dim=1)
            
            # +++ first, we can eliminate all predictions with a confidence value of 0
                # (i.e. all predictions w/ conf < threshold)
            nz_idxs = torch.nonzero(image_pred[:,4], as_tuple=False).squeeze()
            image_pred = image_pred[nz_idxs, :].view(-1,7)
            
            # +++ do NMS in the way perscribed
            if classwise_nms == True: # classwise NMS
                img_classes = unique(image_pred[:,-1]) # get all the classes detected
                for cls_n in img_classes: # loop through em
                    # find indexes of predictions of this class
                    cls_idxs = torch.where(image_pred[:,-1] == cls_n)[0]
                    image_pred_cls = image_pred[cls_idxs] # get only those predictions

                    # do NMS on them (using the NMS prediction method)
                    image_pred_cls = self._NMS_prediction(image_pred_cls)
                    
                    # now we just need to add the image index as the first element of
                    # dimension 1, then we can add this to the output
                    img_i_col = torch.FloatTensor(
                        image_pred_cls.size(0), 1).fill_(img_i).to(self.device)
                    image_pred_cls = torch.cat((img_i_col, image_pred_cls), dim=1)

                    # now, we can write these to the output
                    if detections == None: # if no output yet
                        detections = image_pred_cls.clone() # just replace it
                    else: # otherwise
                        # just tack it on to the end
                        detections = torch.cat((detections, image_pred_cls), dim=0)
            elif classwise_nms == False: # class independent NMS
                # just do NMS using the method
                image_pred = self._NMS_prediction(image_pred)

                # add the image index column
                img_i_col = torch.FloatTensor(
                    image_pred.size(0), 1).fill_(img_i).to(self.device)
                image_pred = torch.cat((img_i_col, image_pred), dim=1)

                # and then write the output
                if detections == None: # if no output yet
                    detections = image_pred.clone() # copy the image predictions
                else: # if there is output
                    # add the current detections to the end
                    detections = torch.cat((detections, image_pred), dim=0)
            elif classwise_nms == None: # no NMS at all
                # just add the image index column
                img_i_col = torch.FloatTensor(
                    image_pred.size(0), 1).fill_(img_i).to(self.device)
                image_pred = torch.cat((img_i_col, image_pred), dim=1)

                # and then write the output
                if detections == None: # if no output yet
                    detections = image_pred.clone() # copy the image predictions
                else: # if there is output
                    # add the current detections to the end
                    detections = torch.cat((detections, image_pred), dim=0)
        
        # +++ return the detections and maybe loss
        if targets != None: # if targets given
            return detections, loss # return detections and loss
        else: # otherwise
            return detections # just return detections
    
    def log_detections(self, detections):
        ''' log detections method
        essentially just helps you read the detections that come out of the get
        detections method. prints the detections to the console.
        
        --- args ---
        detections : torch.FloatTensor
            output from get_detections method
        '''
        img_idxs = unique(detections[:,0]) # get the images where detections were made

        for img_i in img_idxs:
            # get all detections in that image
            obj_nums = [int(x[-1]) for x in detections if int(x[0]) == img_i]
            # get the object names
            objs = [self.classes[i] for i in obj_nums]
            # print the objects detected
            print(f'image #{img_i} : {objs}')
    
    def detect(self, path, batch_size=1, classwise_nms=True, colorfile='colors.pt', results_dir='results', shuffle_colors=False):
        '''
        
        '''
        # +++ load up the data using the datamanager
        data = DataManager(path, self.input_dim, self.classnames_path, mosaics=1.)
        batches = data.batches(batch_size=batch_size, shuffle=False) # get batches
        
        # +++ setup for loop to get detections
        all_detections = None # all detections (will be tensor)
        batch_losses = [] # batch losses list

        # +++ loop to get the detections
        for batch_i in range(len(batches)): # go through batch idxs
            batch, label = batches[batch_i] # get the batch

            # get detections for the batch
            detections, loss = self.get_detections(batch, targets=label,
                classwise_nms=classwise_nms)
            # add loss to the list
            batch_losses.append(loss.cpu().item())
            
            # check if there are detections
            if detections == None: # no detections
                continue # loop back
            else: # otherwise
                detections = detections.cpu() # bring detections to CPU
            
            # add number of images before this batch to image index
            detections[:, 0] += batch_size * batch_i

            # and now we can write the output
            if all_detections == None: # if none yet
                all_detections = detections # just use detections
            else: # otherwise
                # tack these detections onto the end
                all_detections = torch.cat((all_detections, detections), dim=0)
        # +++ use the datamanager to scale the detections onto the orig images
        detections = data.scale_detections(all_detections)

        # +++ get a copy of the original images to write the detections to
        det_images = data.orig_images
        
        # +++ make the set of loss labels (if applicable)
        if batch_size == 1:
            loss_lbls = [f'loss = {l:.5f}' for l in batch_losses]
        else:
            loss_lbls = None

        
        # +++ write the detections to the images
        data.write_boxes(det_images, detections, img_labels=loss_lbls, boxes_format=1)
        
        # +++ make the results directory if it is't around
        results_dir = os.path.realpath(results_dir) # get the real path
        if not os.path.exists(results_dir): # if it doesn't exist
            os.mkdir(results_dir) # make it
        
        # +++ get the path for each detection image
        det_paths = [os.path.join(results_dir, f'det_{imname}.jpg') \
            for imname in data.img_names] # list for that
        
        # +++ save the images !!!
        for path, img in zip(det_paths, det_images):
            cv2.imwrite(path, img)
    
    def _write_boxes(self, dets, images, colors, rel_txt_height=0.02):

        font = cv2.FONT_HERSHEY_SIMPLEX # save the font real quick

        for det in dets: # go through detections
            # +++ unpack the detection
            img = images[int(det[0])] # image the detection is on
            cidx = int(det[7]) # class index
            color = colors[cidx] # get the color to use for this class
            c1 = tuple(det[1:3].int().tolist()) # top left corner of box
            c2 = tuple(det[3:5].int().tolist()) # bottom right corner of box
            label = f'{self.classes[cidx]}; {det[5]:.2f}, {det[6]:.2f}' # make label text
            
            # +++ setup some shit for the text making and whatnot
            ydim = img.shape[0] # get the y dimension of image
            txt_h = rel_txt_height * ydim # get text height in pixels
            real_txt_height = cv2.getTextSize(label, font, 1, 1)[0][1] # get text height
            font_scale = float(txt_h / real_txt_height) # calc font scale
            thickness = max(1, int(txt_h / 10)) # calc font thickness
            txt_size = cv2.getTextSize(label, font, font_scale, thickness)[0]   
                # get the *actual* text size with the right scale and everything
            
            ct1 = (c1[0], c1[1] - txt_size[1] - 4) # top left of text box
            ct2 = (c1[0] + txt_size[0] + 4, c1[1]) # bottom right of text box

            # +++ now we add shit to the image
            cv2.rectangle(img, c1, c2, color, thickness*2)
            cv2.rectangle(img, ct1, ct2, color, -1)
            cv2.putText(img, label, c1, font, font_scale, [255,255,255], thickness)

    def _write_losses(self, losses, images, rel_txt_height=0.02):
        '''
        write losses to some images
        '''

        font = cv2.FONT_HERSHEY_SIMPLEX # save the font
        black = [0, 0, 0] # black color
        white = [255, 255, 255] # white color

        for loss, img in zip(losses, images):
            lbl = f'image loss = {loss:.5f}' # make the loss label to write

            # +++ setup some shit to get the right size of the text
            ydim = img.shape[0] # get the y dim of the img
            txt_h = rel_txt_height * ydim # get the target text height
            curr_txt_height = cv2.getTextSize(lbl, font, 1, 1)[0][1] # get actual height
            font_scale = float(txt_h / curr_txt_height) # calculate font scale to use
            thickness = max(1, int(txt_h / 10)) # calc thickness to use
            txt_size = cv2.getTextSize(lbl, font, font_scale, thickness)[0] # actual

            c1 = (0,0) # top left corner of box
            c2 = (txt_size[0] + 5, txt_size[1] + 5) # bottom right corner of box
            txt_pos = (0, txt_size[1]) # txt position

            cv2.rectangle(img, c1, c2, black, -1) # write the rectangle
            cv2.putText(img, lbl, txt_pos, font, font_scale, white, thickness) # put txt






det = Detector('chess-yolov3-tiny.pt', 'data/class-names.labels', CUDA=True)

# ds = DataSet('data/test', 256, shuffle=False)
# bs = ds.batches()
# x, y = bs[0]
# x = x.cuda()
# y = y.cuda()

# data = DataManager('data/test', 256, det.classnames_path)
# bs = data.batches(shuffle=False)
# x, y = bs[5]
# x = x.cuda()
# y = y.cuda()
# with torch.no_grad():
#     o = det.model(x)


import time

tstart = time.time()
det.detect('data/test', batch_size=1, shuffle_colors=True, results_dir='yolov3-tiny-results')
tend = time.time()

print(f'detection took {tend-tstart}')

# img_names, orig_images, image_dims, image_batches, labels = \
#     ds.all_data(batch_size=1)
# o, l = det.get_detections(x, targets=y)

