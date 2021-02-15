''' datamanager.py
contains the DataManager class
'''
import os
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
from config import parse_value
from util import *
import cv2

class DataManager(object):
    ''' data manager class
    this class is used to load data from a directory. the directory given should
    have at least two sub directories, 'images' (containing files that look like
    '[filename].jpg'), and 'labels' (with files that look like '[filename].txt')
    with the labels for each image in the dataset. the labels should be text
    files with a line for each object thats present. each line should contain 5
    numbers seperated by spaces: class number of the object, x center, y center,
    width, and height of the bounding box. the latter four of these values 
    should be NORMALIZED W/R/T THE IMAGE DIMENSIONS. in addition, there should
    be a file in the directory called '[filename].labels', containing the class
    labels that the dataset labels correspond to, with each label on it's own 
    line indicating it's number (i.e. the label 0 should be on the first line,
    and so on).
    
    --- args ---
    path : str
        path to the directory to load data from.
    input_dim : int
        the square size of images that should be when they get batched.
    class_file_path : str, optional (default=None)
        if the file storing the class data is not in the given path, provide the
        path to it with this parameter.
    mosaics : float, optional (default=None)
        if set to a float, the dataset will be augmented with that percent of
        mosaic data. note that this is different from mosaics that are generated
        using the batches method, these mosaics persist in the data set 
        throughout training.
    '''
    # +++ global class variables
    font = cv2.FONT_HERSHEY_SIMPLEX

    # +++ static methods
    @staticmethod
    def load_classes(path):
        ''' +++ INTERNAL FUNCTION +++
        reads class names from a file.

        --- args ---
        path : str
            path to the file containing the class names.
        
        --- returns ---
        list[str] : list of class names in order, as read from the file.
        '''
        f = open(path, 'r')
        names = f.read().split('\n')
        names = [name.strip() for name in names] # strip spaces
        names = [name for name in names if name != ''] # ignore empty lines
        return names
    
    @staticmethod
    def _resize_image(image, new_dims, default=128):
        ''' +++ INTERNAL METHOD +++
        resizes an image (numpy array, as read by CV2) to have size new_dims 
        WITHOUT changing the aspect ratio. all other pixels values will be set
        to the defualt value given.

        --- args ---
        image : numpy.ndarray
            array of integers that represents the image.
        new_dims : tuple[int,int]
            the new dimensions of the resized image (x,y).
        default : int between 0 and 255
            the default pixel value given to all pixels not covered by the 
            image.
        
        --- returns ---
        torch.tensor : image tensor with size (new_dims[1], new_dims[0], 3)
        float : scaling factor used to scale the image
        '''
        imy, imx = image.shape[0:2] # get the dimensions of the image
        outx, outy = new_dims # unpack the resized image dimensions
        scale_factor = min(outy/imy, outx/imx) # get the scale factor to use
        newx = int(scale_factor * imx) # calculate the new x dimension of the image
        newy = int(scale_factor * imy) # calculate the new y
        resized_image = cv2.resize(image, (newx, newy), interpolation=cv2.INTER_CUBIC)
            # make the resized image
        xbuff = (outx - newx) // 2 # x dimension buffer
        ybuff = (outy - newy) // 2 # y dimension buffer
        
        output = torch.full((outy, outx, 3), default, dtype=int) # make the output
            # tensor, full of default value
        output[ybuff:ybuff+newy, xbuff:xbuff+newx, :] = \
            torch.from_numpy(resized_image) # inject the resized image into output
        
        return output, scale_factor, xbuff, ybuff
    
    @staticmethod
    def show_img(img, show_plot=True):
        ''' show image method
        shows some image onto the current matplotlib.pyplot plot.
        
        --- args ---
        img : torch.tensor or np.ndarray
            the array with the image data in it.
        show_plot : bool, optional (default=True)
            should this method implicitly call plt.show?
        '''
        # make a copy of the image
        img = copy.deepcopy(img)

        # formatting stuff
        if isinstance(img, torch.Tensor): # if it's a tensor
            img = img.cpu() # make sure it's on the cpu
            img = img.permute(1,2,0) # permute to have 2nd dim channels
        elif isinstance(img, np.ndarray): # if it's a numpy array
            img = torch.FloatTensor(img) # make it float tensor
            img = img.flip(2) # flip second dimension
            img = img.div(255.) # normalize pixel values

        # show the image on the plot
        plt.imshow(img)

        # show the plot?
        if show_plot:
            plt.show()
    
    @staticmethod
    def _iou_corners(boxes1, boxes2):
        ''' +++ INTERNAL METHOD +++
        gets the iou (intersect over union) of two sets of boxes with eachother,
        pairwise, where each box is described by it's two corners.
        
        --- args ---
        boxes1 : torch.tensor of size (num_boxes, 4)
            the first set of boxes, with the first dimension containing the 
            xmin, ymin, xmax, ymax values for the box.
        boxes2 : torch.tensor of size (num_boxes, 4)
            the second set of boxes, with the first dimension containing the
            xmin, ymin, xmax, ymax values for the box.
        
        --- returns ---
        torch.tensor of size (num_boxes) : the intersect over union of the sets
            of boxes that are given.
        '''
        b1_c1 = boxes1[:,0:2] # top left of boxes1
        b1_c2 = boxes1[:,2:4] # bottom right of boxes1
        b2_c1 = boxes2[:,0:2] # top left of boxes2
        b2_c2 = boxes2[:,2:4] # bottom right of boxes2

        # figure out which boxes actually intersect at all
        intersects = (b1_c2 > b2_c1).all(1) * (b2_c2 > b1_c1).all(1)
        intersects = intersects.float() # make it float

        int_c1 = torch.max(b1_c1, b2_c1) # get intersect box c1
        int_c2 = torch.min(b1_c2, b2_c2) # get intersect box c2

        int_area = (int_c2 - int_c1).prod(1) # get width/height and multiply

        b1_area = (b1_c2 - b1_c1).prod(1) # box1 area
        b2_area = (b2_c2 - b2_c1).prod(1) # box2 area

        union_area = b1_area + b2_area - int_area

        iou = int_area/union_area
        iou *= intersects # zero out ones that don't intersect
        return iou
    
    @staticmethod
    def _trim_square(new_dim, img, lbl, iou_thresh=0.3):
        ''' +++ INTERNAL METHOD +++
        useful method that takes an image and trims out a square with side 
        lengths new_dim, and also trims the label values.
        
        --- args ---
        new_dim : int
            the new (square) dimension of the image
        img : np.ndarray
            the image to trim, as read by cv2.
        lbl : torch.tensor
            the label for the image, as read by DataManager.
        iou_thresh : float
            all labels that are trimmed such that IOU of new label and old label
            is less than this threshold will be ignored.
        
        --- returns ---
        np.ndarray : the trimmed image.
        torch.tensor : the trimmed label.
        '''
        # +++ make a copy of all inputs
        img = copy.deepcopy(img)
        lbl = copy.deepcopy(lbl)

        # +++ first, we convert the label to (xmin,ymin,xmax,ymax)
        xc, yc, w, h = lbl[:,1:].t().clone() # get the label values
        lbl[:,1] = xc - w/2 # x min
        lbl[:,2] = yc - h/2 # y min
        lbl[:,3] = xc + w/2 # x max
        lbl[:,4] = yc + h/2 # y max
        lblcpy = copy.deepcopy(lbl) # make a copy of the label (for comparison later)

        # +++ now we trim both the x and y dimensions of the image (if neccesary)
        imy, imx = img.shape[:2] # get image x and y dimensions
        img_dims = torch.tensor([imx, imy]).float() # img dims tensor

        # +++ now we get the corners of what will be the new bounding box
        c1i, c2i = torch.zeros((2,2)).long() # two two dimensional indices (x,y) in PIXELS

        if imy > new_dim: # if y dim needs to be trimmed
            trim_from = random.randint(0, imy-new_dim) # where to start the trim
            c1i[1] = trim_from # set min y
            c2i[1] = trim_from + new_dim # set max y
        else: # no trimming
            # leave miny 0
            c2i[1] = imy # maxy = image dim

        if imx > new_dim: # if x needs to be trimmed
            trim_from = random.randint(0, imx-new_dim) # start the trim
            c1i[0] = trim_from # set min x
            c2i[0] = trim_from + new_dim # set max x
        else: # no trimming
            # leave miny 0
            c2i[0] = imy # max x = image dim
        
        # lastly, convert pixels to relative measurements
        c1 = c1i/img_dims
        c2 = c2i/img_dims

        # +++ now, make sure that the bounding boxes stay within the bounds

        # make sure the all box x/y are >= min of new box
        lbl[:,1:3] = torch.max(lbl[:,1:3], c1)
        lbl[:,3:5] = torch.max(lbl[:,3:5], c1)
        # make sure all x/y are <= max of new box
        lbl[:,1:3] = torch.min(lbl[:,1:3], c2)
        lbl[:,3:5] = torch.min(lbl[:,3:5], c2)

        # +++ zero out boxes based on iou thresh
        ious = DataManager._iou_corners(lbl[:,1:5], lblcpy[:,1:5])
        iou_mask_idxs = torch.where(ious > iou_thresh)
        lbl = lbl[iou_mask_idxs] # keep only those above the thresh
        
        # +++ scale the labels to fit the new image
        lbl[:,1:3] = (lbl[:,1:3] - c1)*(img_dims/new_dim) # scale first corner
        lbl[:,3:5] = (lbl[:,3:5] - c1)*(img_dims/new_dim) # scale second corner
        
        # +++ trim the image array
        img = img[c1i[1]:c2i[1],c1i[0]:c2i[0],:]
        
        # +++ now we need to convert label back to xc, yc, w, h
        xmin, ymin, xmax, ymax = lbl[:,1:].t().clone() # get current vals
        lbl[:,1] = (xmin + xmax)/2 # x center
        lbl[:,2] = (ymin + ymax)/2 # y center
        lbl[:,3] = xmax - xmin # width
        lbl[:,4] = ymax - ymin # height
        
        return img, lbl
    
    # +++ built in methods
    def __init__(self, path, input_dim, class_file_path=None, mosaics=None):
        # +++ save the inputs 
        self.path = os.path.realpath(path)
        self.input_dim = input_dim

        # +++ read all the shit from the path
        self._read_path(class_file_path)

        # +++ now we get ready to store all the data we will collect from the images
            # these will eventually be numpy arrays, but for now they're lists
        self._names = [] # image names
        self._orig_imgs = [] # original images
        self._img_dims = [] # original image dimensions
        self._scale_factors = [] # scaling factors
        self._xbuffers = [] # x dimension buffers
        self._ybuffers = [] # y dimension buffers
        self._imgs = [] # resized images
        self._lbls = [] # labels for each image
        
        # +++ now we need to get the files to sort through
        img_files = os.listdir(self.img_dir) # all image files
        lbl_files = os.listdir(self.lbl_dir) # all label files
        self._names, img_files, lbl_files = self._get_valid_data() # get the matching data points
        
        for imgf, lblf in zip(img_files, lbl_files): # go through the files
            # get all the info about the image and resize it
            orig_img, img_dims, scale_fac, xbuff, ybuff, img = self._get_image(imgf)
            # get the label for the image
            lbl = self._get_label(lblf)

            # add all that shit to the lists
            self._orig_imgs.append(orig_img)
            self._img_dims.append(img_dims)
            self._scale_factors.append(scale_fac)
            self._xbuffers.append(xbuff)
            self._ybuffers.append(ybuff)
            self._imgs.append(img)
            self._lbls.append(lbl)
        
        # +++ save the number of images
        self._len = len(self._imgs)

        # +++ build some mosaic images! if asked for
        if mosaics != None:
            # get the mosaic images and whatnot
            m_oimgs, m_timgs, m_lbls = self.get_mosaics(mosaics)

            # loop to add mosaics to class lists
            for i, (moimg, mtimg, mlbl) in enumerate(zip(m_oimgs, m_timgs, m_lbls)):
                self._names.append(f'mosaic{i}.jpg')
                self._orig_imgs.append(moimg) # add the og image
                self._img_dims.append([self.input_dim, self.input_dim])
                self._scale_factors.append(1.)
                self._xbuffers.append(0)
                self._ybuffers.append(0)
                self._imgs.append(mtimg)
                self._lbls.append(mlbl)
            
            # update length
            self._len = len(self._imgs)
        
        # +++ format some of the data (make some shit tensors)
        self._img_dims = torch.tensor(self._img_dims).float() # make it a tensor
        self._imgs = torch.stack(self._imgs) # make the images a tensor
        self._scale_factors = torch.tensor(self._scale_factors) # make it a tensor
        self._scale_factors = self._scale_factors.view(-1,1) # view as a vector
        self._xbuffers = torch.tensor(self._xbuffers).view(-1,1) # make vector
        self._ybuffers = torch.tensor(self._ybuffers).view(-1,1) # make vector

        # +++ end of __init__
    
    def __repr__(self):
        return f'DataManager({self.path}, dim={self.input_dim})'

    def __len__(self):
        ''' returns the number of images '''
        return self._len

    # +++ internal methods
    def _read_path(self, class_file_path=None):
        ''' +++ INTERNAL METHOD +++
        gets pertinent paths from self.path, specifically, gets self.img_dir,
        self.lbl_dir, and reads the class labels file.
        '''
        # +++ first, find and read the class labels file
        if class_file_path: # if class file path was given
            class_file_path = os.path.realpath(class_file_path) # get the real path
            self._classes = self.load_classes(class_file_path) # load the classes
        else: # if class file path wasn't given
            # loop to find class labels
            for fname in os.listdir(self.path):
                if '.' in fname and fname.split('.')[-1] == 'labels': # if it's the class file
                    class_file_path = os.path.join(self.path, fname) # get the real path to the file
                    self._classes = self.load_classes(class_file_path) # load the classes

        # +++ make sure class file was found
        if not '_classes' in self.__dict__: # no classes found
            raise FileNotFoundError(f'could not find a .labels file in {self.path}')

        # +++ get the image and label sub directories
        self.img_dir = os.path.join(self.path, 'images')
        self.lbl_dir = os.path.join(self.path, 'labels')

        # +++ make sure the image and label directories exist
        if not os.path.exists(self.img_dir): # no image dir found
            raise FileNotFoundError(f'could not find \'images\' subdirectory in {self.path}')
        if not os.path.exists(self.lbl_dir): # no label dir found
            raise FileNotFoundError(f'could not find \'labels\' subdirectory in {self.path}')

    def _get_valid_data(self):
        ''' +++ INTERNAL METHOD +++
        this sorts out only the data for which we have both an image and a label
        file, and prints warnings if any files are missing.
        '''
        out_fnames = [] # output file names
        out_img_files = [] # output img files list
        out_lbl_files = [] # output label files list

        img_files = os.listdir(self.img_dir) # get the image files
        lbl_files = os.listdir(self.lbl_dir) # get the label files
        img_files.sort() # sort the image files
        lbl_files.sort() # sort the label files

        # +++ main loop
        while len(img_files) + len(lbl_files) > 0: # while there are files to go to
            # +++ CASES CASES CASES
            if len(img_files) == 0: # no more images
                lblf = lbl_files[0] # get the label file
                print(f'warning: label file ({lblf}) has no matching image, skipping it')
                continue # loop back
            elif len(lbl_files) == 0:
                imgf = img_files[0] # get the img file
                print(f'warning: image file ({imgf}) has no matching label, skipping it')
                continue # loop back
            else: # otherwise
                imgf = img_files[0] # get the next img file
                lblf = lbl_files[0] # get the next lbl file

                # get info about the two files
                (imgname, imgext), (lblname, lblext) = self._get_file_info(imgf, lblf)

            # +++ MORE CASES
            if imgext != 'jpg': # not an image
                print(f'warning: image file ({imgf}) has unknown extension ({imgext}), skipping it')
                img_files.pop(0) # remove it
                continue # loop back
            elif lblext != 'txt': # not a label file
                print(f'warning: label file ({lblf}) has unknown extension ({lblext}), skipping it')
                lbl_files.pop(0) # remove it
                continue # loop back
            elif imgname < lblname: # image is less than label --> no label for image
                print(f'warning: image file ({imgf}) has no matching label, skipping it')
                img_files.pop(0) # remove it
                continue # loop back
            elif lblname < imgname: # label is less than image --> no image for label
                print(f'warning: label file ({lblf}) has no matching image, skipping it')
                lbl_files.pop(0) # remove it
                continue # loop back
            elif lblname == imgname: # matching files!
                out_fnames.append(imgname) # add the name to the output
                out_img_files.append(img_files.pop(0)) # add image file (and remove)
                out_lbl_files.append(lbl_files.pop(0)) # add the label file (and remove)

        # return all the matching datapoints        
        return out_fnames, out_img_files, out_lbl_files
    
    def _get_file_info(self, *filenames):
        ''' +++ INTERNAL METHOD +++ 
        this takes in a bunch of filenames and splits the strings into tuples
        with each being (filename, extension).
        '''
        out = []

        for f in filenames:
            ext = f.split('.')[-1] # get the extension
            name = f[:-1-len(ext)] # get the filename
            out.append((name, ext)) # add info to output
        
        if len(out) == 1: # if theres only one filename
            out = out[0] # make the output it's info
        
        return out
    
    def _get_image(self, imgf):
        ''' +++ INTERNAL METHOD +++ 
        takes in some image filename ([filename].jpg) and outputs the original
        image, the original image dimensions, scaling factor used for resizing, 
        and the resized image.
        '''
        # get the actual path of the image
        img_path = os.path.join(self.img_dir, imgf)
        
        # get the image and do some shit with it
        orig_img = cv2.imread(img_path) # read the image
        img_dims = [orig_img.shape[1], orig_img.shape[0]] # x,y img dimensions

        ### make the resized image
        img, scl_fac, xbuff, ybuff = self._resize_image(orig_img, 
            (self.input_dim, self.input_dim)) # resize the img
        img = img.flip(2).permute(2,0,1).float() # flip the channels and
            # change it to be (channels, y, x) (from (y,x,channels))
        img = img.div(255.) # divide by 255. to normalize values

        # return all that shit
        return orig_img, img_dims, scl_fac, xbuff, ybuff, img
    
    def _get_label(self, lblf):
        ''' +++ INTERNAL METHOD +++
        reads a label file and returns a numpy array of the label.
        '''
        lbl_path = os.path.join(self.lbl_dir, lblf)
        lbl_f = open(lbl_path, 'r') # open the file
        lbl_f_lines = lbl_f.read().split('\n') # read the file's lines
        lbl_f.close() # close the file
        img_lbls = [] # image labels list (will be tensor)

        # +++ loop to build lbls
        for line in lbl_f_lines:
            lbl = parse_value(line) # read the line
            if lbl != None: # if there is a label on this line
                img_lbls.append(lbl) # add the label to the list
        
        img_lbls = torch.tensor(img_lbls).reshape(-1, 5) # make lbls a tensor
        img_lbls = img_lbls.float() # make it a float tensor
        
        return img_lbls
    
    # +++ properties
    @property
    def orig_images(self):
        # return a detached clone of the original images
        return copy.deepcopy(self._orig_imgs)
    
    @property
    def img_names(self):
        # return a detached copy of the image names
        return copy.deepcopy(self._names)

    # +++ data manager methods
    def get_mosaics(self, pct):
        ''' get mosaics method
        this method generates as many mosaics as are requested, which is useful
        for augmenting training data every epoch so the model doesn't start 
        memorizing the training data.

        --- args ---
        pct : float
            the percent of mosaic data that should be added to the dataset.
        
        --- returns ---
        list[numpy.ndarray] : the mosaic images in their original form
        list[torch.tensor] : the mosaic images in processed form
        list[torch.tensor] : the labels for the images
        '''
        num_mosaics = int(pct * self._len) # get total number of mosaics

        orig_imgs = [] # original mosaics
        imgs = [] # processed mosaics
        lbls = [] # labels
        img_idxs = range(self._len) # all image indexes

        # +++ loop to make mosaics
        for _ in range(num_mosaics):
            midxs = random.choices(img_idxs, k=4) # get the indexes for this one
            
            mimgs = [self._orig_imgs[i] for i in midxs] # imgs for this mosaic
            mlbls = [self._lbls[i] for i in midxs] # labels

            oimg, lbl = self.mosaic_stitch(mimgs, mlbls) # make the mosaic

            # make a tensor version and process that shit
            timg = torch.from_numpy(oimg).float().flip(2).permute(2,0,1).div(255.)

            # add everything to lists
            orig_imgs.append(oimg)
            imgs.append(timg)
            lbls.append(lbl)
        
        imgs = torch.stack(imgs, dim=0)

        return orig_imgs, imgs, lbls

    def mosaic_stitch(self, imgs, lbls):
        ''' mosaic_stitch method
        this method performs mosaic data augmentation on 4 images and 4 labels,
        generating a new, stitched together version of the image for the network
        to get better at recognizing portions of classes, instead of relying too
        heavily on specific features of objects.

        --- args ---
        imgs : list[np.ndarray] of length 4
            the list of images (as read by cv2) to stitch together.
        lbls : list[torch.tensor] of length 4
            the list of labels (as read by the DataManager) that correspond to
            the image.
        
        --- returns ---
        np.ndarray : the new image array
        torch.tensor : the new image label
        '''
        # +++ make a deep copy of inputs
        imgs = copy.deepcopy(imgs)
        lbls = copy.deepcopy(lbls)
        
        # +++ get the image dimensions
        img_dims = torch.FloatTensor([img.shape[:2] for img in imgs])
        img_dims = img_dims.flip(1) # flip dims to be (x,y)
        new_dim = img_dims.min().int().item() # new square dim for all images

        # +++ now we rescale the images so their smallest dimension matches new_dim
        scl_facs = torch.max(new_dim / img_dims, dim=1, keepdim=True)[0] 
            # calculates factors to scale by
        img_dims = (img_dims * scl_facs).int() # scale the image dimensions
        
        # loop to scale each image
        for i in range(len(imgs)):
            new_dims = tuple(img_dims[i].tolist()) # new image dims tuple
            imgs[i] = cv2.resize(imgs[i], new_dims, interpolation=cv2.INTER_CUBIC)

        # +++ trim all images to be square with new_dim
        sqimgs = []
        sqlbls = []
        for img, lbl in zip(imgs, lbls):
            img, lbl = self._trim_square(new_dim, img, lbl)
            lbl[:,1:] = lbl[:,1:] / 2 # divide lbl (x,y,w,h) by 2 (scaling factor of stitching)
            sqimgs.append(img)
            sqlbls.append(lbl)
        
        # +++ now we gotta stitch em all together
        # here we stitch together the images
        top = np.concatenate(sqimgs[:2], axis=1)
        bottom = np.concatenate(sqimgs[2:], axis=1)
        stitched = np.concatenate((top,bottom), axis=0)

        # and now the labels
        sqlbls[1][:,1] += .5 # 1st image (top right) gets xcenter + .5
        sqlbls[2][:,2] += .5 # 2nd image (bottom left) gets ycenter + .5
        sqlbls[3][:,1:3] += .5 # 3rd img (bottom right) gets both centers + .5
        stitched_lbls = torch.cat(sqlbls, dim=0) # concatenate all labels together
        
        # +++ now we need to pick a square out of the imagg
        # first resize it so each sub image is the input dim size
        stitched = cv2.resize(stitched, 
            (self.input_dim*2, self.input_dim*2), 
            interpolation=cv2.INTER_CUBIC)
        
        # then cut a input dim sized square out
        final_img, final_lbl = self._trim_square(
           self.input_dim, stitched, stitched_lbls)

        return final_img, final_lbl

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
        # +++ get the images in the batch
        imgs = self._imgs[idxs]

        # +++ get the labels
        lbls = torch.FloatTensor(0,6) # labels float tensor
        for img_n, i in enumerate(idxs): # go through the indicies
            lbl = self._lbls[i] # get the label tensor
            img_n_col = torch.FloatTensor(lbl.size(0), 1).fill_(img_n) # img idx column
            lbl = torch.cat((img_n_col, lbl), dim=1) # add it to the front
            lbls = torch.cat((lbls, lbl), dim=0) # add the label to the tensor
        
        # +++ return the batch tuple
        return (imgs, lbls)
    
    def num_batches(self, batch_size):
        ''' num batches method
        returns the number of batches that can be made with a given batch size.
        
        --- args ---
        batch_size : int
            the batch size being used.
        
        --- returns ---
        int : the number of batches that will be made.

        '''
        return self._len // batch_size

    def batches(self, batch_size=None, mini_batch_size=None, mosaics=None, shuffle=True):
        ''' batches method
        this method makes batches with the perscribed hyper parameters. note
        that if mini_batch_size is NOT set, this will just return a list of 
        batches, where as is it IS set, it will return a list of lists of mini
        batches.

        --- args ---
        batch_size : int, optional (default=None)
            the size that the returned batches should be. if None, all the data
            will be in a single batch.
        mini_batch_size : int, optional (default=None)
            if none, no minibatching will be done. otherwise batches will be 
            split into minibatches of the perscribed size. note that if it is
            not a factor of batch_size, the actual batch_size will be cut to the
            nearest multiple of minibatchsize.
        mosaics : float, optional (default=None)
            if set, the dataset will be augmented with this percent of mosaic 
            data. these mosaics are generated fresh each time it's called, and
            will never be the same.
        shuffle : bool, optional (defualt=True)
            if true, the data will be shuffled prior to batching.
        
        --- returns ---
        IF mini_batch_size == None:
        list[tuple[torch.tensor, torch.tensor]] : a list of the batch tuples,
            containing batch_data and batch_labels
        OTHERWISE:
        list[list[tuple[torch.tensor, torch.tensor]]] : a list of lists of mini
            batch data and mini batch labels.
        '''
        # +++ generate mosaics, if asked for
        if mosaics != None:
            old_len = self._len # save the old length to revert to
            _, mimgs, mlbls = self.get_mosaics(mosaics) # get the mosaics
            self._imgs = torch.cat((self._imgs, mimgs), dim=0) # add images
            self._lbls += mlbls # add labels
            self._len = self._imgs.size(0) # get the new length

        # +++ check batch size
        if batch_size == None: # if it's none
            batch_size = self._len # use all the data
            
        # +++ setup for the main loop
        num_batches = self.num_batches(batch_size) # number of batches
        
        img_idxs = list(range(self._len)) # list of image indecies
        if shuffle: random.shuffle(img_idxs) # shuffle the img indecies
        
        batches = [] # list of batches (output)

        # +++ loop through batches
        for batch_i in range(num_batches):
            # get the image indexes for this batch
            b_idxs = img_idxs[batch_i * batch_size : (batch_i + 1) * batch_size]
            
            if mini_batch_size == None: # if there are no mini batches
                batch = self.make_batch(b_idxs) # make the batch from the indexes
            else: # if there are minibatches
                batch = [] # batch (list of minibatches)
                # loop through minibatches
                for mb_i in range(0, batch_size, mini_batch_size):
                    # get the indexes for this minibatch
                    mb_idxs = b_idxs[mb_i : mb_i + mini_batch_size]
                    mini_batch = self.make_batch(mb_idxs) # make the minibatch
                    batch.append(mini_batch) # add it to the batch list
            
            # add the batch to the batches
            batches.append(batch)
        
        # +++ cut out the mosaics and shit from the dataset
        if mosaics != None:
            self._len = old_len # reset length
            self._imgs = self._imgs[:old_len] # only keep the old images
            self._lbls = self._lbls[:old_len] # only keep old labels

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

    def all_data(self, batch_size=1, shuffle=False):
        ''' all data method
        this method is used to get all of the data from the datamanager. in
        particular, it is very useful for the detector class to get all of the
        information about the data at once.
        '''
        # +++ setup for batch building
        image_batches = [] # list for image batches
        label_batches = [] # list for label batches
        img_idxs = list(range(self._data.shape[0])) # get all indecies of data
        if shuffle: # if they should be shuffled
            random.shuffle(img_idxs) # shuffle em
        num_batches = len(img_idxs) // batch_size # calculate number of batches
        img_idxs = img_idxs[:batch_size * num_batches] # only keep ones that get batched
        
        # +++ loop to build batches!
        for batch_i in range(len(self._data) // batch_size):
            # get indecies of images for this batch
            batch_idxs = img_idxs[batch_i * batch_size : (batch_i + 1) * batch_size]
            
            # get the images and labels for the batch
            imgs, lbls = self.make_batch(batch_idxs)

            # add these things to the batches
            image_batches.append(imgs)
            label_batches.append(lbls)
        
        # +++ get the original images
        orig_images = list(self._data[img_idxs, 1])

        # +++ make the image dimensions tensor
        image_dims = list(self._data[img_idxs, 2]) # get dims
        image_dims = [torch.FloatTensor(dims) for dims in image_dims] # make tensors
        image_dims = torch.stack(image_dims, dim=0) # stack em into one tensor

        # +++ get image names
        image_names = list(self._data[img_idxs,0])

        return image_names, orig_images, image_dims, image_batches, label_batches

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
            the detections that have been made. note that the first dimension of
            the tensor should store the data as indicated below:
            [img index, xmin, ymin, xmax, ymax, ...]
            where the image index is absolute in terms of the dataset.
        
        --- returns ---
        torch.tensor with same size : the scaled detections that can be mapped
            onto the original images of this set.
        '''
        detections = detections.clone() # make a clone of the tensor

        # +++ get versions of the xbuffs, ybuffs, scale factors, and image dimensions
            # that align with the images for each detection
        img_idxs = detections[:,0].long() # get the image indexes
        xbuffs = self._xbuffers[img_idxs]
        ybuffs = self._ybuffers[img_idxs]
        scl_facs = self._scale_factors[img_idxs]
        img_dims = self._img_dims[img_idxs]

        # +++ scale the detections to match the original images
        detections[:,[1,3]] -= xbuffs # subtract off any x buffer
        detections[:,[2,4]] -= ybuffs # subtract any y buffer
        detections[:,1:5] /= scl_facs # divide by the factor used to scale the images
        
        # +++ make sure the bounding boxes stay in bounds
        detections[:,1:5] = torch.clamp(detections[:,1:5], min=0)
            # make sure there are no negatives
        detections[:,[1,3]] = torch.min(detections[:,[1,3]], img_dims[:,0:1])
            # make sure the x dimensions don't go over
        detections[:,[2,4]] = torch.min(detections[:,[2,4]], img_dims[:,1:2])
            # make sure the y dimensions don't go over
        
        return detections

    # +++ end of DataSet class


dm = DataManager('data/test', 256, class_file_path='data/class-names.labels', mosaics=None)

# imgs = dm._imgs[-10:]
# idxs = random.choices(range(dm._len), k=4)
# imgs = [dm._orig_imgs[i] for i in idxs]
# lbls = [dm._lbls[i] for i in idxs]
# img, lbl, imgt, lblt = dm.mosaic_stitch(imgs, lbls)
# dm.write_boxes([img, imgt], [lbl, lblt])
# dm.show_img(img)
# dm.show_img(imgt)
