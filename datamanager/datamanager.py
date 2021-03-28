''' datamanager.py
contains the DataManager class
'''
import os
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
from .util import *
import cv2

class DataManager(object):
    ''' data manager class
    this class is used to load image and label data for the neural networks. it
    loads data from a given directory that has two sub directories: 'images' 
    (containing files that look like '[filename].jpg'), and 'labels' (with files
    that look like '[filename].txt') with the labels for each image in the 
    dataset. additionally, if not otherwise specified, there should be a .labels
    file in the directory that has the class labels. see documentation for more 
    information about formatting.
    
    --- args ---
    path : str
        path to the directory to load data from.
    input_dim : int
        the square size of images that should be when they get batched.
    class_file : str, optional (default=None)
        if the .txt file containing class data is not in the given directory,
        provide it here.
    **data_aug
        keyword arguments for data augmentation. each argument should be the
        type of data augmentation that should be done, associated with a float
        indicating the amount of that type of augmented data, relative to the
        whole dataset.
        valid keywords are : mosaics
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    _columns = ['names','originals','dims','images','labels']
    # +++ built in methods
    def __init__(self, path, input_dim, class_file=None, **data_aug):
        # save the inputs 
        self.path = os.path.realpath(path)
        self.input_dim = input_dim
        # read all the shit from the path
        self._classes, img_dir, lbl_dir, names = read_dir(self.path)
        # sort out the classes
        if self._classes == None and class_file == None:
            raise AttributeError(f'no class file found and none provided')
        elif self._classes == None and class_file != None:
            self._classes = load_classes(os.path.realpath(class_file))
        # now we make the whole data array
        self._data = np.empty((len(names), 5), dtype=object)
        # loading stuff into the data array
        self._data[:,0] = np.array(names)
        self._data[:,1:4] = load_images(img_dir, names, self.input_dim)
        self._data[:,4] = load_labels(lbl_dir, names)
        # save the info about data augmentation
        self.data_aug = data_aug
        self._aug_imgs = None
        self._aug_lbls = None
        # end of __init__
    
    # +++ built in functions
    def __repr__(self):
        datadir = os.path.relpath(self.path)
        return f'DataManager(dir=\'{datadir}\', dim={self.input_dim})'

    def __getitem__(self, index):
        if isinstance(index, int):
            assert index in range(5), f'invalid data manager index ({index})'
            return list(self._data[:, index])
        elif isinstance(index, str):
            assert index in self._columns, f'got invalid column \'{index}\''
            i = self._columns.index(index)
            return list(self._data[:,i])
        
    # +++ get methods
    def get_dims(self):
        return torch.stack(self[2], dim=0)

    def get_imgs(self, augmented=False):
        imgs = torch.stack(self[3], dim=0)
        if augmented and self._aug_imgs != None:
            imgs = torch.cat((imgs, self._aug_imgs), dim=0)
        return imgs
    
    def get_lbls(self, augmented=False):
        lbls = self[4]
        if augmented and self._aug_lbls != None:
            lbls += self._aug_lbls
        return lbls
    
    def get_len(self, augmented=False):
        l = self._data.shape[0]
        if augmented and self._aug_imgs != None:
            l += self._aug_imgs.size(0)
        return l

    # +++ data manager methods
    def augment_data(self):
        # reset the current data
        self._aug_imgs = torch.FloatTensor(0,3,self.input_dim,self.input_dim)
        self._aug_lbls = []
        # make the augmented data
        if 'mosaics' in self.data_aug and self.data_aug['mosaics'] != 0.0:
            n = int(self.get_len() * self.data_aug['mosaics'])
            imgs, lbls = make_mosaics(self.get_imgs(), self.get_lbls(), n)
            self._aug_imgs = torch.cat((self._aug_imgs, imgs), dim=0)
            self._aug_lbls += lbls

    def make_batch(self, idxs):
        ''' make batch method
        this method takes in a list of indexes to batch together and outputs a
        tuple with the (batch_data, batch_labels) for that set of indices.

        --- args ---
        idxs : list[int(s)]
            the indexes of images in this batch.
        
        --- returns ---
        torch.tensor : the batch_data, with shape (batch_size, 3, input_dim,
            input_dim)
        torch.tensor : the batch_labels, with shape (num_lbls, 6)
        '''
        # get images for the batch
        imgs = self.get_imgs()[idxs]
        # get labels for the batch
        lbls = torch.FloatTensor(0,6)
        for img_n, i in enumerate(idxs):
            lbl = self._data[i, 4].clone()
            img_n_col = torch.FloatTensor(lbl.size(0), 1).fill_(img_n)
            lbl = torch.cat((img_n_col, lbl), dim=1)
            lbls = torch.cat((lbls, lbl), dim=0)
        # return the batch tuple
        return (imgs, lbls)
    
    def batches(self, batch_size=None, shuffle=True, augmented=True):
        ''' batches method
        this method makes batches with the perscribed hyper parameters. note
        that if mini_batch_size is NOT set, this will just return a list of 
        batches, where as is it IS set, it will return a list of lists of mini
        batches.

        --- args ---
        batch_size : int, optional (default=None)
            the size that the returned batches should be. if None, all the data
            will be in a single batch.
        mosaics : float, optional (default=None)
            if set, the dataset will be augmented with this percent of mosaic 
            data. these mosaics are generated fresh each time it's called, and
            will never be the same.
        shuffle : bool, optional (defualt=True)
            if true, the data will be shuffled prior to batching.
        augmented : bool, optional (defualt=True)
            if true, augmented data will be mixed in amoung the batches.
        
        --- returns ---
        list[tuple[torch.tensor, torch.tensor]] : a list of the batch tuples,
            containing batch_data and batch_labels
        '''
        # generate augmented data
        if augmented: self.augment_data()
        l = self.get_len(augmented)
        # check batch size
        if batch_size == None:
            batch_size = l
        # get the images and labels
        images = self.get_imgs(augmented)
        labels = self.get_lbls(augmented)
        # setup for the main loop
        num_batches = l//batch_size
        img_idxs = list(range(l))
        if shuffle: random.shuffle(img_idxs)
        batches = [] # output
        # loop through batch indexes
        for batch_i in range(num_batches):
            # get the image indexes for this batch
            b_idxs = img_idxs[batch_i * batch_size : (batch_i + 1) * batch_size]
            # make the batch
            bimgs = images[b_idxs]
            blbls = batch_labels(labels, b_idxs)
            batches.append((bimgs, blbls))
        # return the batches
        return batches
    
    def get_class(self, class_num):
        ''' get class method
        used to get the class label from the class number.
        
        --- args ---
        class_num : int
            the index of the class.
        
        --- returns ---
        str : the name of the class.
        '''
        return self._classes[int(class_num)]

    def write_boxes(self, imgs, boxes, img_labels=None, boxes_format=0, \
                    colorfile='colors.pt', rel_txt_h=0.02):
        ''' write boxes method
        --- args ---
        imgs : list[numpy.ndarray]
            the set of images that are having boxes written onto them. the 
            images should be formatted exactly how cv2 reads them.
        boxes : see format codes
            the data for boxes being written onto images. see format codes below
            for more information.
        boxes_format : int, optional (default=0)
            the code indicating the formating of the boxes variable. see format
            codes below for more information.
        colorfile : str, optional (default='colors.pt')
            the file containing colors for the boxes.

        --- format codes ---
        0 : boxes = list[torch.tensor]
            boxes is a list of tensors, where each tensor corresponds to an
            image in imgs, and the first dimension of each tensor should contain
            [class #, xcenter, ycenter, width, height]
            where all the measurements are normalized with respect to the 
            image's dimensions. boxes will be labeled and colored corresponding
            to class numbers.
        1 : boxes = torch.tensor
            boxes is a tensor who's first dimension should contain
            [img #, xmin, ymin, xmax, ymax, box conf, class conf, class num]
            where all the measurements are in units of pixels. boxes will be
            colored based on class, and labeled with "[class]; [box conf], 
            [class conf]".
        
        --- returns ---
        list[numpy.ndarray] : list of images with labeled boxes shown
        '''
        # read the color file
        f = open(colorfile, 'rb')
        color_list = torch.load(f)
        f.close()

        # list of all boxes and labels to display
        all_boxes = []

        # +++ populate the all_boxes list
        if boxes_format == 0: # format code 0
            # loop through images and the boxes for them
            for img, img_boxes in zip(imgs, boxes):
                img_boxes = img_boxes.clone() # make a copy of the boxes tensor

                # get image dimensions
                imy = img.shape[0] # img y dim
                imx = img.shape[1] # img x dim

                # calculate xmin and xmaxes
                xmin = (img_boxes[:,1] - img_boxes[:,3]/2) * imx
                ymin = (img_boxes[:,2] - img_boxes[:,4]/2) * imy
                xmax = (img_boxes[:,1] + img_boxes[:,3]/2) * imx
                ymax = (img_boxes[:,2] + img_boxes[:,4]/2) * imy
                
                # get the box corners
                c1s = torch.stack((xmin, ymin), dim=1).int().tolist()
                c2s = torch.stack((xmax, ymax), dim=1).int().tolist()
                c1s = [tuple(c1) for c1 in c1s]
                c2s = [tuple(c2) for c2 in c2s]

                # get the labels
                lbls = [self.get_class(i) for i in img_boxes[:,0]]

                # get colors
                colors = [color_list[int(i)] for i in img_boxes[:,0]]

                # make the final list of formatted image boxes
                out_img_boxes = list(zip(lbls, c1s, c2s, colors))
                all_boxes.append(out_img_boxes) # add it to the list
        elif boxes_format == 1: # format code 1
            for i in range(len(imgs)): # go through all img idxs
                img_box_idxs = torch.where(boxes[:,0] == i) # get all box indexes
                img_boxes = boxes[img_box_idxs] # get the boxes for this img
                
                # get the corners of the boxes
                c1s = img_boxes[:,1:3].int().tolist()
                c2s = img_boxes[:,3:5].int().tolist()
                c1s = [tuple(c1) for c1 in c1s]
                c2s = [tuple(c2) for c2 in c2s]

                # make the labels and colors for the boxes
                lbls = [] # labels list
                colors = [] # colors list
                for det in img_boxes: # go through detections
                    lbl = f'{self.get_class(det[-1])};{det[5]:.3f},{det[6]:.3f}'
                    lbls.append(lbl)
                    colors.append(color_list[int(det[-1])])
                
                # make the formatted list of boxes for this img
                out_img_boxes = list(zip(lbls, c1s, c2s, colors))
                all_boxes.append(out_img_boxes)

        # +++ write the boxes and labels onto the images
        for i in range(len(imgs)): # go through indices
            # get all the things for this image
            img = imgs[i] # image
            img_boxes = all_boxes[i] # boxes in this image

            # +++ check if there are any boxes
            if len(img_boxes) == 0: # no boxes
                continue # loop back

            # +++ first, calculate the font_scale and thickness to use for this img
            imy = img.shape[0] # get the images y dimension
            txt_h = rel_txt_h * imy # get the text height (pixels)
            real_txt_h = [cv2.getTextSize(lbl, self.font, 1, 1)[0][1] \
                          for lbl, _, _, _ in img_boxes]
                # get the actual text height of labels with no scaling
            font_scale = float(txt_h / max(real_txt_h)) # get the scaling to be used
            thickness = max(1, int(txt_h/10)) # get the thickness to use
            bxthickness = int(thickness * 2) # box thickness

            # +++ now loop through to write the boxes
            for lbl, c1, c2, color in img_boxes:
                # get the actual text size to make the text box corners with
                txt_size = cv2.getTextSize(lbl, self.font, font_scale, thickness)[0]

                # make the textbox corners
                ct1 = (c1[0], c1[1] - txt_size[1] - 4) # top left of txt box
                ct2 = (c1[0] + txt_size[0] + 4, c1[1]) # bottom right of txt box

                # now write shit to the image
                cv2.rectangle(img, c1, c2, color, bxthickness) # bounding box
                cv2.rectangle(img, ct1, ct2, color, -1) # text box
                cv2.putText(img, lbl, c1, self.font, font_scale, [255,255,255], thickness) 
                    # text
            
            # +++ now write the label to the top left, if there is one
            if img_labels != None:
                lbl = img_labels[i] # get the label

                # get the text size
                txt_size = cv2.getTextSize(lbl, self.font, font_scale, thickness)[0]

                # get corners
                c1 = (0,0)
                c2 = (txt_size[0]+4, txt_size[1]+4)
                org = (2, txt_size[1]+2)

                # write that shit
                cv2.rectangle(img, c1, c2, [0,0,0], -1) # text box
                cv2.putText(img, lbl, org, self.font, font_scale, [255,255,255], thickness)
    
    def scale_detections(self, detections):
        ''' scale detections method
        this method takes in some detections that have been made on the resized
        images in this data set, and scales them so that they are the detections
        on the original images of this dataset.
        
        --- args ---
        detections : torch.tensor with size (num_detections, >=5)
            the tensor should store the detections as
            [[img index, xmin, ymin, xmax, ymax, ...], ...]
            where the image index is absolute in terms of the dataset. and the
            measurements are in units of pixels of the resized images.
        
        --- returns ---
        torch.tensor with same size : the scaled detections that can be mapped
            onto the original images of this set.
        '''
        detections = detections.clone() # make a clone of the tensor
        # get the info about how the detections need to be scaled
        img_dims = self.get_dims()[detections[:,0].long()]
        scl_facs = img_dims/self.input_dim
        # scale the detections
        detections[:,1:5] *= scl_facs.repeat(1,2)
        # make sure they are all in the valid range
        detections[:,1:5] = torch.clamp(detections[:,1:5], min=0)
        detections[:,1:5] = torch.min(detections[:,1:5], img_dims.repeat(1,2))

        return detections

    # +++ end of DataManager class
