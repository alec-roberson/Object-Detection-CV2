''' util.py
this file contain utility functions that read things from files, etc.

'''

# +++ imports
import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

# +++ FUNCTIONS
# cuda function
CUDA_DEVICE = 'cuda:0'
def get_device(CUDA):
    if CUDA:
        return CUDA_DEVICE
    else:
        return 'cpu'

def norm_params(img_tensor):
    means = [img_tensor[i].mean() for i in range(img_tensor.size(0))]
    stds = [img_tensor[i].std() for i in range(img_tensor.size(0))]
    return means, stds

# file accessing functions
def get_file_info(*filenames):
    ''' get file info function
    this takes in one or more filenames and breaks them up by 'name' and 
    extension.

    --- args ---
    *filenames : str
        the filenames to read.
    
    --- returns ---
    tuple[str, str] : filename, extension, if one file was given.
    list[tuple[str, str]] : list of filenames and extensions, if multiple files
        were given.
    '''
    out = []
    for f in filenames:
        fname = f.split('/')[-1]
        spf = fname.split('.')
        if len(spf) == 1:
            out.append(tuple(spf + ['']))
        else:
            ext = spf[-1] # extension
            name = fname[:-1-len(ext)] # filename
            out.append((name, ext))
    
    if len(out) == 1: # only one input
        out = out[0]
    
    return out

def read_dir(path):
    ''' read directory method
    reads a given data directory, looking for dub directories images and labels
    and a .labels file. returns the classes and the names of files that have 
    both images and labels.

    --- args ---
    path : str
        the path to the directory to load data from.

    --- returns ---
    list[str] or None : class labels, if found
    str : the directory images are in
    str : the directory labels are in
    list[str] : names of the datapoints (filenames sans extensions)
    '''
    # read contents of directory
    contents = os.listdir(path)
    assert 'images' in contents, f'images folder not found in data directory {path}'
    assert 'labels' in contents, f'labels folder not found in data directory {path}'
    img_dir = os.path.join(path, 'images')
    lbl_dir = os.path.join(path, 'labels')
    # read classes
    classes = None
    for f in contents:
        if get_file_info(f)[1] == 'txt': # labels file
            classes = load_classes(os.path.join(path, f))
    # get all valid file names
    img_fs = os.listdir(img_dir) # get all files
    lbl_fs = os.listdir(lbl_dir)
    img_fs = [f for f in img_fs if get_file_info(f)[1] == 'jpg'] # only valid
    lbl_fs = [f for f in lbl_fs if get_file_info(f)[1] == 'txt'] # extensions
    names = set([fi[0] for fi in get_file_info(*img_fs)]) # get image names
    names.intersection_update(
       set([fi[0] for fi in get_file_info(*lbl_fs)])) # keep ones with lbls
    names = list(names)
    names.sort()
    return classes, img_dir, lbl_dir, names

def load_classes(path):
    ''' load classes function
    loads classes from a .txt file into a list.

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

def load_labels(lbl_dir, names):
    ''' get labels function
    reads specified label files from a label directory and returns a numpy array
    of the label tensors in the order of files given.

    --- args ---
    lbl_dir : str
        the directory the label files are in.
    lbl_files : list[str]
        the label files to read from the directory.
    
    --- returns ---
    np.ndarray[torch.FloatTensor] : a numpy array containing the label tensors.
    '''
    lbl_files = [f'{n}.txt' for n in names]
    labels = np.empty((len(lbl_files),), dtype=object)
    for i, f in enumerate(lbl_files):
        lbl_path = os.path.join(lbl_dir, f)
        lbl_f = open(lbl_path, 'r')
        lbl_lines = lbl_f.read().split('\n') # lines of the label file
        lbl_f.close()
        lbl_lines = [l.strip() for l in lbl_lines] # strip
        lbl_lines = [l for l in lbl_lines if l != ''] # remove empties
        # read the lines as lists of numbers
        lbl = [ [float(n) for n in line.strip().split(' ') if n != '']
                for line in lbl_lines]
        lbl = torch.FloatTensor(lbl).reshape(-1, 5)
        labels[i] = lbl
    return labels

def load_images(img_dir, names, resize_to):
    ''' load images function
    loads images from a specified label directory and returns a numpy array
    containing the original image as well as a rescaled copy, alongside the
    original dimensions of the loaded images and the factor they were scaled 
    down by.

    --- args ---
    img_dir : str
        the directory the image files are in.
    img_files : list[str]
        the image files to read from the directory.
    resize_to : int
        the square size images should be resized to.
    
    --- returns ---
    np.ndarray[object] : the data array with size (num_images, 5), where the 5
        values in the first dimension are: original image, original dimensions,
        scale factor, padding, resized image.
    '''
    # turn the file names into image files
    img_files = [f'{n}.jpg' for n in names]

    # ready the transformations to apply to the image
    tfs = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((resize_to, resize_to)),
        transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5),(.5,.5,.5))])

    # loop to format images and save data
    out = np.empty((len(img_files), 3), dtype=object)
    for i, f in enumerate(img_files):
        # get the original image
        img_path = os.path.join(img_dir, f)
        orig_img = cv2.imread(img_path)
        orig_dims = torch.FloatTensor(orig_img.shape[:2]).flip(0)
        # get the new image
        img = tfs(np.flip(orig_img, axis=2))
        # write output
        out[i,0] = orig_img
        out[i,1] = orig_dims
        out[i,2] = img
    return out

# bounding box functions
def bb_xywh_to_cs(boxes):
    '''
    '''
    out = torch.zeros_like(boxes)
    xc, yc, w, h = boxes.t()
    out[:,0] = xc - w/2 # x min
    out[:,1] = yc - h/2 # y min
    out[:,2] = xc + w/2 # x max
    out[:,3] = yc + h/2 # y max
    return out

def bb_cs_to_xywh(boxes):
    '''
    '''
    out = torch.zeros_like(boxes)
    xmin, ymin, xmax, ymax = boxes.t()
    out[:,0] = (xmin + xmax) / 2
    out[:,1] = (ymin + ymax) / 2
    out[:,2] = (xmax - xmin)
    out[:,3] = (ymax - ymin)
    return out

def bb_cs_iou(boxes1, boxes2):
    '''
    '''
    # unpack corners
    b1c1s = boxes1[:, :2]
    b1c2s = boxes1[:, 2:]
    b2c1s = boxes2[:, :2]
    b2c2s = boxes2[:, 2:]
    # get the intersection corners
    intc1s = torch.max(b1c1s, b2c1s)
    intc2s = torch.min(b1c2s, b2c2s)
    # get the box areas
    b1As = (b1c2s - b1c1s).prod(1)
    b2As = (b2c2s - b2c1s).prod(1)
    # get intersection areas
    intwh = (intc2s - intc1s)
    intwh *= (intwh > 0).all(1).unsqueeze(1) # zero out dims < 0
    intAs = intwh.prod(1)
    # get union areas
    uniAs = b1As + b2As - intAs
    return intAs/uniAs

# image manipulation functions
def square_crop(img, lbl, crop_dim, iou_thresh=0.3, normalize=True):
    ''' square crop function
    this function crops a square section out of a given image and manupulates
    the label to match the cropped image.

    --- args ---
    image : torch.FloatTensor
        the array of the image that's being cropped.
    lbl : torch.FloatTensor
        the label associated with this image.
    crop_dim : int
        the dimension of the cropped image.
    iou_thresh : float, optional (default=0.3)
        the IOU threshold below which bounding boxes will be ignored.
    normalize : bool, optional (default=True)
        if true, the output image will be normalized before being returned.
    '''
    # get the device
    device = img.device
    # get the image dimensions
    dims = torch.tensor(img.shape[1:]).long()
    # get the corners of the cropping (y, x)
    c1 = ((dims - crop_dim) * torch.rand((2,))).long()
    c2 = c1 + crop_dim
    # get the corners as bounding boxes (pcts of images; x, y)
    c1bb, c2bb = c1.flip(0) / dims, c2.flip(0) / dims
    c1bb, c2bb = c1bb.to(device), c2bb.to(device)
    # make copies of the labels
    nlbl = lbl.clone()
    nlbl[:, 1:] = bb_xywh_to_cs(lbl[:, 1:]) # turn bbs into corners
    orig_bbs = nlbl[:, 1:].clone()
    # trim the labels inside the box
    nlbl[:,1:3] = torch.min(torch.max(nlbl[:,1:3], c1bb), c2bb)
    nlbl[:,3:5] = torch.min(torch.max(nlbl[:,3:5], c1bb), c2bb)
    # zero out boxes below iou_thresh
    ious = bb_cs_iou(orig_bbs, nlbl[:,1:])
    valid_bbs = torch.where(ious > iou_thresh)
    nlbl = nlbl[valid_bbs]
    # rescale the labels to the cropped image
    nlbl[:,1:5] -= c1bb.repeat(2)
    nlbl[:,1:5] /= (c2bb - c1bb).repeat(2)
    # return lbl to xywh
    nlbl[:, 1:] = bb_cs_to_xywh(nlbl[:, 1:])
    # crop image
    nimg = img[:, c1[0]:c2[0], c1[1]:c2[1]]
    # normalize image
    if normalize:
        nimg = F.normalize(nimg, (.5,.5,.5), (.5,.5,.5))
    return nimg, nlbl

# data augmentation functions
def make_mosaics(imgs, lbls, n):
    ''' make mosaics function
    makes n mosaics using the images and labels from a dataset.

    --- args ---
    imgs : torch.FloatTensor
        the stack of images to be made into mosaics
    lbls : list[torch.FloatTensor]
        the list of labels that are associated with the images.
    n : int
        the number of mosaic images to return.
    
    --- returns ---
    torch.FloatTensor : the mosaic images
    list[torch.FloatTensor] : the labels for the images
    '''
    # get basic info
    dim = imgs.size(-1)
    length = imgs.size(0)
    # make grids of image indices to use
    midxs = torch.randint(0, length, (n, 2, 2))
    # make the mosaics of images
    mosaics = imgs[midxs]
    mosaics = torch.cat((mosaics[:,0,...], mosaics[:,1,...]), dim=3)
    mosaics = torch.cat((mosaics[:,0,...], mosaics[:,1,...]), dim=3)
    # now make the labels for the mosaics
    mosaic_lbls = []
    # scales and buffs to apply to the labels
    scale, right_buff, down_buff = torch.zeros((3,1,5))
    scale[0,0], scale[0,1:] = 1.0, 0.5
    right_buff[0,1] = 0.5
    down_buff[0,2] = 0.5
    # loop to create and correctly scale labels
    for idxs in midxs:
        (i1, i2), (i3, i4) = idxs
        mosaic_lbls.append(torch.cat((
            lbls[i1]*scale,
            lbls[i2]*scale + right_buff,
            lbls[i3]*scale + down_buff,
            lbls[i4]*scale + right_buff + down_buff), dim=0))
    # crop and normalize the images and labels
    out = [[], []]
    for img, lbl in zip(mosaics, mosaic_lbls):
        img, lbl = square_crop(img, lbl, dim)
        img = F.normalize(img, *norm_params(img))
        out[0].append(img)
        out[1].append(lbl)
    # and make the images back into a tensor
    out[0] = torch.stack(out[0], dim=0)
    return out

# misc functions
def batch_labels(labels, idxs):
    ''''''
    lbls = []
    for n, i in enumerate(idxs):
        lbl = labels[i]
        img_i_col = torch.FloatTensor(lbl.size(0), 1).fill_(n)
        lbls.append(torch.cat((
            img_i_col, lbl), dim=1))
    lbls = torch.cat(lbls, dim=0)
    return lbls

def show_img(imgarr):
    if isinstance(imgarr, torch.Tensor):
        img = imgarr.clone().permute(1,2,0)
        img -= img.min() # min = 0
        img /= img.max() # max = 1
        img = img.numpy()
    elif isinstance(imgarr, np.ndarray):
        img = np.flip(imgarr, axis=2) # make RGB
    plt.imshow(img)
    plt.show()

def to_cvUMat(tensor):
    arr = tensor.cpu().clone()
    arr = arr.flip(0).permute(1,2,0)
    arr -= arr.min()
    arr /= arr.max()
    arr *= 255
    arr = np.array(arr, dtype=np.uint8)
    arr = cv2.UMat(arr).get()
    return arr
