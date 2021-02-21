''' load_data.py
this file is used to load and split the data into training and testing 
directories. this file only needs to be run once, when you first upload the 
dataset you wish to use for the network.

dataset formatting: the dataset should come in one directory with a few key 
things to note. first, there should be a file in the dataset directory with the
extension '.labels', these should have the class labels for detection, with each
label printed on it's own line. there should also be image files in the 
directory with the form '[filename].jpg', and corresponding label files with 
names '[filename].txt'. label files should have a line for each object present
in the image, with five numbers on each line seperated by spaces. the first
number should be the class number of the object, and the following four should
be the x center, y center, width, and height of the bounding box, ALL NORMALIZED
BY THE IMAGE DIMENSIONS.
'''
################################################################################
############################## SET THESE VALUES ################################
################################################################################
data_in_dir = 'data-raw' # directory to get the raw data from
data_out_dir = 'data' # directory to save the split data to
class_filename = 'class_names.labels' # file to save classes to
test_pct = .10 # percent of data that should be allocated for testing
shuffle = True # should the data be shuffled?
################################################################################
################################################################################
################################################################################

# +++ imports
import os
import random
from math import log10
from shutil import copyfile

# +++ open the in data directory and get the files
data_in_dir = os.path.realpath(data_in_dir) # data directory path
files = os.listdir(data_in_dir) # get the file names in the path
file_paths = [os.path.join(data_in_dir, f) for f in files] # turn them into legit paths

# +++ function to make directories
def mkdir(*paths):
    # function to make directories and sub directories
    path = os.path.join(*paths) # join the path
    path = os.path.realpath(path) # get the *real* path
    if not os.path.isdir(path): # if it doesn't exist
        os.mkdir(path) # make it
    return path # return the path

# +++ make some directories and paths and shit
data_out_dir = mkdir(data_out_dir) # make data out directory
class_filepath = os.path.join(data_out_dir, class_filename) # class filepath

# make the training directory
train_dir = mkdir(data_out_dir, 'train') # make train sub folder
train_img_dir = mkdir(train_dir, 'images') # make images folder
train_lbl_dir = mkdir(train_dir, 'labels') # and labels folder

# make the test directory
if test_pct != 0.: # only if there IS test data
    test_dir = mkdir(data_out_dir, 'test') # make test sub folder
    test_img_dir = mkdir(test_dir, 'images') # make images folder
    test_lbl_dir = mkdir(test_dir, 'labels') # and labels folder


# +++ now go through the files to build the data and shit
all_files = [] # list of files, in tuples (img_path, lbl_path)

# +++ loop to get the files for training and testing
while len(file_paths) > 0: # while there are still paths to go through
    # +++ get the file and shit
    fp = file_paths.pop(0) # get the first file path off the list
    fname = fp.split('/')[-1] # get the file name
    fext = fname.split('.')[-1] # get the extension of the file

    # +++ cases for file extensions
    if fext == 'labels': # if it's the label file!
        copyfile(fp, class_filepath) # copy the file to where it needs to go
        continue # loop back to top
    elif fext == 'txt': # if it's a text file
        continue # skip over it (we'll get those with the images)
    elif fext != 'jpg': # if it isn't jpg
        # that's fine, just print a warning and skip it
        print(f'warning: skipping file with unknown extension \'{fname}\'')
        continue
    
    # +++ get the label filepath and make sure it exists
    img_fp = fp # filepath is now image filepath
    lbl_fp = img_fp[:-len(fext)] + 'txt' # get the label filepath
    if not os.path.exists(lbl_fp): # if it doesn't exist
        # print a warning and skip it
        print(f'warning: could not find label for image \'{fname}\', skipping it')
        continue
    
    # +++ add the files to the list of files to copy
    all_files.append((img_fp, lbl_fp))


# +++ split the files into test files and train files
if shuffle: # if they should be shuffled
    random.shuffle(all_files) # shuffle em first
n_test = int(test_pct * len(all_files)) # calculate number of files for testing
test_files = all_files[:n_test] # allocate test files
train_files = all_files[n_test:] # allocate training files

# +++ loop to save the test images
if n_test != 0: # if there were testing images
    num_len = int(log10(len(test_files)-1)) + 1 # length (characters) of the max number for this set
    for i, (img_path, lbl_path) in enumerate(test_files):
        img_to_path = os.path.join(test_img_dir, f'img{i:0>{num_len}}.jpg') # path for image
        lbl_to_path = os.path.join(test_lbl_dir, f'img{i:0>{num_len}}.txt') # path for label
        copyfile(img_path, img_to_path) # copy image file
        copyfile(lbl_path, lbl_to_path) # copy label file

# +++ loop to save the training images
num_len = int(log10(len(train_files)-1)) + 1 # length (characters) of the max number for this set
for i, (img_path, lbl_path) in enumerate(train_files):
    img_to_path = os.path.join(train_img_dir, f'img{i:0>{num_len}}.jpg') # path for image
    lbl_to_path = os.path.join(train_lbl_dir, f'img{i:0>{num_len}}.txt') # path for label
    copyfile(img_path, img_to_path) # copy image file
    copyfile(lbl_path, lbl_to_path) # copy label file

