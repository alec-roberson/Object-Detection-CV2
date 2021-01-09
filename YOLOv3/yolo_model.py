''' yolo_model.py
this file contains an arbitrary model of a yolo network (yolo_net) that can be 
initialized with any yolo.cfg file and yolo.weights file. this allows for the
network's hyperparameters - even down to the layer by layer construction - to
be changed on the fly.

last updated 01/09/2021
'''

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import cv2 
from util import *

def parse_value(value):
    ''' +++ INTERNAL FUNCTION +++ 
    parses values from a config file.
    '''
    # before anything, strip the value
    value = value.strip()
    # first, check if it's NOT a list of any sort
    if not ' ' in value and not ',' in value:
        # so it must be a number or a string
        try:
            out = float(value) # make it a number
            if out % 1 == 0: # it's an integer
                out = int(value)
            return out
        except ValueError:
            # if that throws a value error, it must be a string
            return str(value)
    elif ' ' in value: # then it must be a list
        out = value.split(' ') # split by spaces
        out = [val for val in out if val != ''] # remove empty strings
        out = [parse_value(val) for val in out] # parse arguments individually
    elif ',' in value: # then it must be a list
        out = value.split(',') # split by spaces
        out = [val for val in out if val != ''] # remove empty strings
        out = [parse_value(val) for val in out] # parse arguments individually
    
    # so we had a list, now we make sure it's not a stupid list like [1]
    if len(out) == 1:
        return out[0]
    else:
        return out
    
def read_cfg(configFile):
    '''
    this function takes in a .cfg file and outputs a list of "network blocks"
    (essentially layers) that the network should have in it. the network blocks
    are dictionaries with attributes saved as keys.

    ---args---
    configFile: str
        the path to the configuration file that should be used to build this
        network.
    
    ---returns---
    list[dict]: the list of dictionaries describing each block of the network
    '''
    # +++ read config file and pre-process a bit
    f = open(configFile,'r') # open the config file
    ftxt = f.read() # read the raw text of the file
    flines = ftxt.split('\n') # break it into a list of lines
    
    # +++ prepare some stuff for the loop
    networkBlocks = [] # list to hold all the blocks (~layers~(kinda)~)
    networkBlock = {} # the current network block we are working on

    # +++ main loop
    for x in flines: # iterate through the lines of the file
        if len(x) == 0: # blank line
            continue # skip it
        
        elif x[0] == '#': # comment
            continue # skip it
        
        elif x[0] == '[': # start of a new block
            if len(networkBlock) != 0: # if we are already working on a network block
                # add the old network block to the list of blocks
                networkBlocks.append(networkBlock)
                # and start a new block with the right layer index
                networkBlock = {'layer_num': len(networkBlocks) - 1}
            
            # set the type for the current network block to be this layer's type
            networkBlock['type'] = x.strip(' []')

        else: # if this is *not* a new network block
            entity , value = x.split('=') # get the attribute/value pair
            entity = entity.strip() # strip the attribute
            value = parse_value(value) # parse the value
            networkBlock[entity] = value # add them to the current block
    
    # add the last block we were working on to the list
    networkBlocks.append(networkBlock)
    
    # return the list of network blocks
    return networkBlocks

class MaxPoolStride1(nn.Module):
    ''' +++ INTERNAL CLASS +++
    this represents a max-pooling layer with stride 1, since dark-net (the 
    platform yolo was originally written in) handles those quite differently
    than pytorch.
    '''
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        super(MaxPoolStride1, self).__init__()
    
    def forward(self, x):
        # feeding a batch x through this module
        return F.max_pool2d(
            F.pad(x, (0,1,0,1), mode='replicate'),
            kernel_size=self.kernel_size, stride=1)

class yolo_net(nn.Module):
    ''' net class
    this class reads a .cfg file and builds itself according to that
    configuration file.
    
    ---args---
    cfgfile: str
        the path to the configuration file to build this network
    '''
    def __init__(self, cfgfile):
        super(yolo_net, self).__init__()
        self.netblocks = read_cfg(cfgfile) # read config file
        self._build_network() # build the network from the config file
        self.header = torch.IntTensor([0,0,0,0]) # initialize these as zeros
        self.seen = 0 # they will be set in load_weights
    
    def _build_network(self):
        ''' +++ INTERNAL METHOD +++
        gets called by __init__ to build the network'''
        # +++ first, some setup
        self.net_info = self.netblocks[0] # get the info about the network
        self.netblocks.pop(0) # get rid of the info from the netblocks list
        self.numblocks = len(self.netblocks) # set number of blocks
        self.layers = nn.ModuleDict({}) # the layers of the network
            # stored in a dict for easy indexing since some layers, like shortcuts
            # and routing layers won't need any representation
        
        channels = 3 # channels (starts as 3, for RGB data)
        filter_tracker = [] # list of filters used in each layer

        # +++ main loop to build network
        for x in self.netblocks: # loop through network blocks
            i = x['layer_num'] # get the layer number
            block_module = nn.Sequential() # this block's sequential module

            # +++ convolutional block
            if x['type'] == 'convolutional':
                ### get some variables we want to keep
                ker_size = x['size']
                stride = x['stride']
                filters = x['filters']

                ### figure out how much padding to use
                if x['pad']: 
                    # if it should have padding, calculate it
                    padding = (ker_size - 1) // 2
                else:
                    # otherwise set padding to zero
                    padding = 0
                
                ### figure out if we should do batch normalization
                try:
                    # try to get the batch_normalize variable from the block
                    bn = x['batch_normalize']
                    bias = False # if we have it, no bias on the convolution
                except:
                    # if we don't have batch_normalize
                    bn = False # no batch normalization
                    bias = True # yes bias on convolution
                
                ### put together the convolutional part of the layer
                conv = nn.Conv2d(channels, filters, ker_size, stride, 
                    padding=padding, bias=bias)
                block_module.add_module(f'conv_{i}', conv) # add to block module

                ### add batch_normalization (maybe)
                if bn:
                    batch_norm = nn.BatchNorm2d(filters)
                    block_module.add_module(f'batch_norm_{i}', batch_norm)
                
                ### add activation function
                if x['activation'] == 'leaky':
                    activation = nn.LeakyReLU(0.1, inplace=True) # make the activation
                    block_module.add_module(f'leaky_{i}', activation) # add to block
            
            # +++ max pool block
            elif x['type'] == 'maxpool':
                ### make the layer
                if x['stride'] == 1: # if it has stride 1
                    maxpool = MaxPoolStride1(x['size'])
                else: # if it has a stride other than 1
                    maxpool = nn.MaxPool2d(kernel_size=x['size'], stride=x['stride'],
                        padding=0)
                ### add to the block's module
                block_module.add_module(f'maxpool_{i}', maxpool)
            
            # +++ upsample block
            elif x['type'] == 'upsample':
                ### make the layer
                scale_factor = x['stride']
                upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', 
                    align_corners=False)
                ### add to module
                block_module.add_module(f'upsample_{i}', upsample)
            
            # +++ routing block
            elif x['type'] == 'route':
                # NOTE route layers essentially concatenate two previous layers
                # together along the first dimension, so if two input tensors with
                # shape (N, c1, y, x) and (N, c2, y, x) get routed together, the
                # output will have shape (N, c1+c2, y, x)

                ### for standardization, if it only has one layer, we make it a list
                if type(x['layers']) == int: # only one layer
                    x['layers'] = [x['layers']] # make it a list
                
                ### now, make all the layer numbers RELATIVE to this one, meaning we just
                    # subtract the current layer number (i) from the absolute layer 
                    # number (l > 0) to get that the layer number it l-i 
                x['layers'] = [l - i if l > 0 else l for l in x['layers']]

                ### lastly, we calculate the number of filters (channels) that the
                    # output of this layer has, so we can just sum all of the
                    # layers getting routed together
                filters = sum([filter_tracker[i + l] for l in x['layers']])
            
            # +++ shortcut layer
            elif x['type'] == 'shortcut':
                # NOTE like route, shortcut layers don't need any representation in
                # module dictionary, so we'll just focus on cleaning up the layer number
                # which needs to be relative
                if x['from'] > 0:
                    x['from'] = i - x['from']
            
            # +++ yolo layer!
            elif x['type'] == 'yolo':
                # NOTE yolo layers are also special, in that they are more formatting
                # the output then actually computing anything special, so they don't
                # need to be represented in module dictionary, instead here we'll apply
                # the mask to the anchors so it won't have to be done later
                x['anchors'] = [x['anchors'][j] for j in x['mask']]
            
            # +++ exception
            else:
                raise ValueError(f'yolo_net encountered unknown layer type {x["type"]} in yolo_net._build_network')
        
            # +++ now, we just need to put everything where it belongs
            if len(block_module) > 0: # if there are things in the block module
                # then add it with the key being the number
                self.layers.add_module(str(i), block_module)
            
            channels = filters # set the channels (for the input to the next layer) to
                # be the filters that were used in this layer
            
            filter_tracker.append(filters) # add the filters used to the filter tracker

    def load_weights(self, weightfile):
        ### open and start to unpack the weight file
        fp = open(weightfile, "rb") # read the weights from the file

        ### the first 4 values stored in the file are header information 
        # 1. major version number
        # 2. minor Version Number
        # 3. subversion number 
        # 4. # of images this network was trained on 
        header = np.fromfile(fp, dtype = np.int32, count = 5) # get the header info
        self.header = torch.from_numpy(header) # set self.header to that
        self.seen = self.header[3] # set self.seen using the header
        
        ### read the rest of the file as a list of weights
        weights = np.fromfile(fp, dtype = np.float32)
        
        ### loop to set weights
        tracker = 0 # tracks # of weights we've seen
        for i, block in enumerate(self.netblocks): # loop through the blocks
            module_type = block['type'] # get this block's type
            
            ### if it's a convolutional block
            if module_type == "convolutional":
                # the weights file stores info about this layer in this order:
                # if batch norm is being done:
                #   => batch norm bias 
                #   => batch norm weights 
                #   => running mean 
                #   => running variation
                # if batch norm is not being done:
                #   => convolution bias
                # => convolution weights
                model = self.layers[str(i)] # get the model from the module list

                ### find out if this block has batch_normalize or not
                try:
                    batch_normalize = block["batch_normalize"]
                except:
                    batch_normalize = 0
                
                convPart = model[0] # get the convolutional part of this block
                
                # if there is batch normalization for this block
                if batch_normalize:
                    bnPart = model[1] # get the batch norm layer from the model
                    
                    # get the number of biases that the model uses
                    biasCount = bnPart.bias.numel()
                        # (which is the same as number of weights, running mean, and
                        # running var) (for batch norm layers)
                    
                    # get the biases from the weights file
                    bnBias = torch.from_numpy(weights[tracker:tracker + biasCount])
                    tracker += biasCount # increment tracker
                    
                    # get the weights from the weights file
                    bnPart_weights = torch.from_numpy(weights[tracker: tracker + biasCount])
                    tracker += biasCount
                    
                    # get the running means from the weights file
                    bnPart_running_mean = torch.from_numpy(weights[tracker: tracker + biasCount])
                    tracker += biasCount
                    
                    # get the running var from the weights file
                    bnPart_running_var = torch.from_numpy(weights[tracker: tracker + biasCount])
                    tracker  += biasCount
                    
                    # reshape all the above tensors to match the size of the batch norm model
                    bnBias = bnBias.view_as(bnPart.bias.data)
                    bnPart_weights = bnPart_weights.view_as(bnPart.weight.data)
                    bnPart_running_mean = bnPart_running_mean.view_as(bnPart.running_mean)
                    bnPart_running_var = bnPart_running_var.view_as(bnPart.running_var)

                    # copy all the above data into the model
                    bnPart.bias.data.copy_(bnBias)
                    bnPart.weight.data.copy_(bnPart_weights)
                    bnPart.running_mean.copy_(bnPart_running_mean)
                    bnPart.running_var.copy_(bnPart_running_var)
                # if we aren't using batch norm, then this convolution has bias
                else:
                    biasCount = convPart.bias.numel() # get number of biases

                    # get the biases from the weight file
                    convBias = torch.from_numpy(weights[tracker: tracker + biasCount])
                    tracker = tracker + biasCount
                    
                    # shape the biases to match the model
                    convBias = convBias.view_as(convPart.bias.data)
                    
                    # add the biases to the convolutional layer
                    convPart.bias.data.copy_(convBias)
                    
                # get the number of weights for this convolutional layer
                weightCount = convPart.weight.numel()
                
                # get the weights from the weight file
                convWeight = torch.from_numpy(weights[tracker:tracker+weightCount])
                tracker = tracker + weightCount
                
                # reshape weights and set the weights of the actual model
                convWeight = convWeight.view_as(convPart.weight.data)
                convPart.weight.data.copy_(convWeight)
            
            # no need to account for non-convolutional blocks since maxpool, 
            # route, shortcut and yolo layers don't use weights or biases
    
    def set_input_dims(self, width, height=None):
        '''
        set the dimensions of the image tensors that this network takes as 
        input.

        --- args ---
        width : int
            width of input images (pixels).
        height : int, optional (defualt=None)
            height of input images (pixels). if none, will be set to width.
        '''
        self.net_info['width'] = int(width)
        if height == None:
            height = width
        self.net_info['height'] = int(height)

    def forward(self, x, CUDA):
        '''
        forward function to feed a minibatch through the network
        
        ---args---
        x: torch.Tensor
            the minibatch; should have shape (batch_size, h, w, channels)
        '''
        detections = [] # list of detections made
        layer_outputs = {} # outputs of each network layer (need to save each
            # since routing/shortcut layers will refer back to them)
        
        output = None # output variable (will store a tensor)
        # iterate throught each module 
        for i in range(self.numblocks):
            module_type = self.netblocks[i]["type"] # get the current module's type

            # if this is a straight forward module that we can feed through
            if module_type in ['convolutional', 'upsample', 'maxpool']:
                x = self.layers[str(i)](x) # feed the input through the module
                layer_outputs[i] = x # store output in layer outputs

            # if this is a route layer
            elif module_type == 'route':
                layer_is = self.netblocks[i]['layers'] # get the layer idxs to route
                layers = [layer_outputs[i+j] for j in layer_is] # get the layer values
                x = torch.cat(layers, 1) # concatenate along channels
                layer_outputs[i] = x # set the layer outputs
            
            # shortcut layer
            elif module_type == 'shortcut':
                j = self.netblocks[i]['from'] # index of layer taking the shortcut
                # set the output to be the output of the previous one plus the output
                # of the shortcutting layer
                x = layer_outputs[i-1] + layer_outputs[i+j]
                layer_outputs[i] = x # save this layer's output
            
            # yolo ("detection") layer
            elif module_type == 'yolo':
                # this is the "detection layer" that performs the yolo transformation
                # to figure out what the objects being detected here are
                anchors = self.netblocks[i]['anchors'] # get the anchors to use
                inp_dim = self.net_info['height'] # get the input dimensions
                num_classes = self.netblocks[i]['classes'] # get the number of classes
                ###x = x.data # get the data from the current output tensor
                
                # this filters the data in the output tensor to parse out what predictions
                # are being made, in the form of a tensor of size
                # (batch_size, grids, bounding box attributes
                x = yolo_transform(x, inp_dim, anchors, num_classes, CUDA)

                ### REMOVE THIS
                ###print("Size after transform => " ,x.size())

                '''
                ### REMOVE THIS
                # if no detections were made (x = 0)
                if type(x) == int:
                    continue # continue on with the rest of the network
                '''
                # if no output has been written yet
                if output == None:
                    output = x # set to the output of the yolo transformation
                else: # if decetions have been made already
                    # add the detections we just made to the list
                    output = torch.cat((output, x), 1)
                
                # set the outputs of this layer to be the same as the previous one
                layer_outputs[i] = layer_outputs[i-1]

        return output

yolo = yolo_net('cfg/yolov3-320.cfg')
yolo.load_weights('weights/yolov3-320.weights')
###