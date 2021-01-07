# DNModel.py
# 
# NOTE: this file was borrowed from the following github link
# https://github.com/AyushExel/Detectx-Yolo-V3
# 
# however, the file may have been modified since
# 
# 6/1/21

from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import cv2 
from util import *

class dummyLayer(nn.Module):
    def __init__(self):
        super(dummyLayer, self).__init__()

class detector(nn.Module):
    def __init__(self, anchors):
        super(detector, self).__init__()
        self.anchors = anchors

def construct_cfg(configFile):
    '''
    this function takes in a .cfg file and outputs a list of "network blocks"
    (essentially layers) that the network should have in it. the network blocks
    are dictionaries with attributes saved as keys.

    ---args---
    configFile: str
        the path to the configuration file that should be used to create this
        network
    
    ---returns---
    list[dict(s)]: the list of dictionaries describing each block of the network
    ''' 
    
    ### read and pre-process the configuration file
    
    config = open(configFile,'r') # open the config file
    file = config.read().split('\n') # break it into a list of lines
    
    file = [line for line in file if len(line) > 0 and line[0]!= '#'] # ignore lines that start with a comment
    file = [line.lstrip().rstrip() for line in file] # strip spaces from front and back of lines
    
    
    # separate network blocks in a list 
    
    networkBlocks = [] # list to hold all the blocks (~layers~(kinda)~)
    networkBlock = {} 

    for x in file: # iterate through the lines of the file
        if x[0] == '[': # when we find an open bracket (indicating a new layer)
            if len(networkBlock) != 0: # if we are already working on a network block
                networkBlocks.append(networkBlock) # then add the old network block to the list of blocks
                networkBlock = {} # and start a new block for this one
            networkBlock["type"] = x[1:-1].rstrip() # set the type for the current network block to be this layer's type
        else: # if this is *not* a new network block
            entity , value = x.split('=') # then split the line to get the attribute/value pair
            networkBlock[entity.rstrip()] = value.lstrip() # strip spaces off those and add them to the current block
    networkBlocks.append(networkBlock) # add the last block we were working on to the list
    
    # return the list of network blocks
    return networkBlocks

def buildNetwork(networkBlocks):
    '''
    contrary to it's name, this does NOT build the network from the blocks. it
    reads the blocks, and puts together a *rough* outline of the network that
    can then be fed into the network class to finally be read and assembled
    into a fully-functioning network.
    
    ---args---
    networkBlocks: list[dict(s)]
        the output of construct_cfg that should be used to put together the 
        network
    
    ---returns---
    dict: basic info about this network as a whole
    torch.nn.ModuleList([torch.nn.Sequential([torch.nn.Module(s)])]): the 
        modules that will be used to put together the network
    '''
    ### setup
    DNInfo = networkBlocks[0] # get info about the network (start of cfg file)
    modules = nn.ModuleList([]) # list of modules that will become the network
    channels = 3 # for RGB image data
    filterTracker = [] # list to keep track of the filters we use

    ### main loop to build network
    for i,x in enumerate(networkBlocks[1:]): # loop through all the network blocks
        seqModule  = nn.Sequential() # this block's sequential module

        # convolutional block
        if (x["type"] == "convolutional"):
            
            # get all the info about it
            filters= int(x["filters"])
            pad = int(x["pad"])
            kernelSize = int(x["size"])
            stride = int(x["stride"])
            activation = x["activation"]

            if pad: # if this conv layer should be padded
                # calculate the padding
                padding = (kernelSize - 1) // 2
            else:
                # otherwise set padding to zero
                padding = 0
            
            try:
                # try to get if we should do batch normalization
                bn = int(x["batch_normalize"]) # if this works, then we should, so
                bias = False # don't use bias on this layer
            except: # however, if the block has no "batch_normalize" key
                bn = 0 # set bn to zero
                bias = True # and yes we should use bias

            ### use all the info we gathered to put together a convolutional
                # layer and add it to this block's sequential module
            conv = nn.Conv2d(channels, filters, kernelSize, stride, padding, bias = bias)
            seqModule.add_module(f'conv_{i}', conv)

            ### add batch normalization (maybe)
            if bn:
                bn = nn.BatchNorm2d(filters)
                seqModule.add_module(f'batch_norm_{i}', bn)

            ### add the activation function
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                seqModule.add_module(f'leaky_{i}', activn)

        # upsample block
        elif (x["type"] == "upsample"):
            ### make the layer and add it to the sequential module
            scale_factor = int(x["stride"]) # get the scale factor to use
            upsample = nn.Upsample(scale_factor = scale_factor, mode = "bilinear", align_corners=False)
            seqModule.add_module(f'upsample_{i}', upsample)
        
        # route block
        elif (x["type"] == "route"):
            # brief note: a route layer essentially concatenates some layers together,
            # the "layers" argument tells what layers to concatenate so for instance if
            # "layers" = "-2, 12" it means to concatenate this layer with the one two 
            # behind it and 12 in front of it. it concatenates layers with their channels
            # so if two layers of shape (N, h, w, c1) and (N, h, w, c2) get routed together
            # the output is of shape (N, h, w, c1+c2)

            x['layers'] = x["layers"].split(',') # extract the layers to concatenate
            start = int(x['layers'][0]) # start is the first layer to be concatenated

            try:
                # try to get end, the second layer to be concatenated
                end = int(x['layers'][1])
            except:
                # if that doesn't work, there is no end, so set end to zero
                end =0
            
            if start > 0: # if start is the positive layer number
                start = start - i # get the relative layer number
            # same goes for end
            if end > 0:
                end = end - i
            
            route = dummyLayer() # set the route layer to just be a dummy layer
            # add the place holder to the sequential module for this block
            seqModule.add_module(f'route_{i}', route)

            if end < 0: # if this route layer manages two layers
                # then number of filters (channels) for this layer will be the 
                # number of filters for the start and end layer added together 
                filters = filterTracker[i+start] + filterTracker[i+end]
            else: # if there's just one layer routed here
                # the number of filters will be the number of filters in that layer
                filters = filterTracker[i+start]
        
        # shortcut layer
        elif (x["type"] == "shortcut"):
            # like the route layer, the shortcut layer won't be implemented here 
            # quite yet, instead we'll just leave in a dummy layer as a reminder
            shortcut = dummyLayer() # set the shortcut to just be a dummy layer
            seqModule.add_module(f'shortcut_{i}', shortcut) # add to block
        
        # yolo layer
        elif (x["type"] == "yolo"):
            # a yolo layer is a detector layer, it's the core of the yolo algorithm
            anchors = x["anchors"].split(',') # extract the anchors 
            anchors = [int(a) for a in anchors] # convert anchors to integers
            anchors = [(anchors[j],anchors[j+1]) for j in range(0,len(anchors),2)] 
                # reorganize the anchors back into tuples (y,x) 
            masks = x["mask"].split(',') # get the mask (which anchors that this layer should use)
            masks = [int(a) for a in masks] # convert mask nums to integers
            anchors = [anchors[j] for j in masks] # filter for only the anchors that the mask permits
            detectorLayer = detector(anchors) # make the detector layer

            seqModule.add_module(f'detection_{i}',detectorLayer) # add to the module
        
        modules.append(seqModule) # add the sequential module for this block 
            # to the list of modules
        channels = filters # set the channels (for the next layer) to be the 
            # number of filters used in this layer
        filterTracker.append(filters) # add the number of filters to the
            # filter tracker list

    # return the info about the network and the module list
    return DNInfo, modules

class net(nn.Module):
    ''' net class
    this class reads a .cfg file and builds itself according to that
    configuration file.
    
    ---args---
    cfgfile: str
        the path to the configuration file to build this network
    '''
    def __init__(self, cfgfile):
        super(net, self).__init__()
        self.netBlocks = construct_cfg(cfgfile) # get the network block info from file
        self.DNInfo, self.moduleList = buildNetwork(self.netBlocks) # build the modules from the block info
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
    
    def forward(self, x, CUDA):
        '''
        forward function to feed a minibatch through the network
        
        ---args---
        x: torch.Tensor
            the minibatch; should have shape (batch_size, h, w, channels)
        '''
        detections = [] # list of detections made
        modules = self.netBlocks[1:] # the blocks to use (skip the info block)
        layerOutputs = {} # outputs of each network layer (need to save each
            # since routing/shortcut layers will refer back to them)
        
        
        written_output = 0
        # iterate throught each module 
        for i in range(len(modules)):    
            
            module_type = (modules[i]["type"]) # get the current module's type

            # if this is a convolutional module
                # (upsampling is basically a form of convolution)
            if module_type == "convolutional" or module_type == "upsample" :
                # then the module in the modulelist will be able to handle it
                x = self.moduleList[i](x) # feed the input through the module
                layerOutputs[i] = x # store output in layer outputs

            # if this is a route layer
            elif module_type == "route":
                layers = modules[i]["layers"] # get the layers from the modules dict
                layers = [int(a) for a in layers] # convert to ints
                
                if (layers[0]) > 0: # if the layer number is absolute
                    layers[0] = layers[0] - i # convert to relative

                if len(layers) == 1: # if there is only one layer being routed together
                    # then set the output to just be that layer's outputs
                    x = layerOutputs[i + (layers[0])]

                else: # if there are two layers being added
                    if (layers[1]) > 0: # if the second one's index is absolute
                        layers[1] = layers[1] - i # convert to relative
                        
                    map1 = layerOutputs[i + layers[0]] # get outputs from the first
                    map2 = layerOutputs[i + layers[1]] # get outputs from the second
                    
                    
                    x = torch.cat((map1, map2), 1) # concatenate the two along channels
                
                layerOutputs[i] = x # set the output of this layer in layerOutputs
            
            # shortcut layer
            elif module_type == "shortcut":
                from_ = int(modules[i]["from"]) # get the layer taking the shortcut
                # set the output to be the output of the previous one plus the output
                # of the shortcutting layer
                x = layerOutputs[i-1] + layerOutputs[i+from_]
                layerOutputs[i] = x # save this layer's output
            
            # yolo layer
            elif module_type == 'yolo':        
                # this is the "detection layer" that does object detection

                anchors = self.moduleList[i][0].anchors # get the anchors for this layer
                    # (these are coming from moduleList, so they've been filtered by mask)
                
                inp_dim = int (self.DNInfo["height"]) # get the input dimensions
                num_classes = int (modules[i]["classes"]) # get the number of classes
                x = x.data # get the data from the current output tensor
                print("Size before transform => " ,x.size())
                
                # this filters the data in the output tensor to parse out what predictions
                # are being made, in the form of a tensor of size
                # (batch_size, grids, bounding box attributes)
                # also (supposedly) if none of the bounding boxes find a class then this
                # should return 0... but i don't really see how...
                x = transformOutput(x, inp_dim, anchors, num_classes, CUDA)
                print("Size after transform => " ,x.size())

                
                # if no detections were made (x = 0)
                if type(x) == int:
                    continue # continue on with the rest of the network

                # if no output has been written yet
                if not written_output:
                    detections = x # set the detections to the detections found
                    written_output = 1 # set written output to 1
                else: # if decetions have been made so far
                    # add these detections to the list
                    detections = torch.cat((detections, x), 1)
                
                # set the outputs of this layer to be the same as the previous one
                layerOutputs[i] = layerOutputs[i-1]
        
        try:
            # return the detections made
            return detections
        except:
            # if no detections have been made, return 0
            return 0
    
    def load_weights(self, weightfile):
        ### open and start to unpack the weight file
        fp = open(weightfile, "rb") # read the weights from the file

        # the first 4 values stored in the file are header information 
        # 1. major version number
        # 2. minor Version Number
        # 3. subversion number 
        # 4. # of images this network was trained on 
        header = np.fromfile(fp, dtype = np.int32, count = 5) # get the header info
        self.header = torch.from_numpy(header) # set self.header to that
        self.seen = self.header[3] # set self.seen using the header
        
        weights = np.fromfile(fp, dtype = np.float32) # read the rest of the file as weights
        
        ### loop to set weights
        tracker = 0 # tracks # of weights we've seen
        for i in range(len(self.moduleList)):
            module_type = self.netBlocks[i + 1]["type"] # get the module type from the list
            
            # convolutional block
            if module_type == "convolutional":
                model = self.moduleList[i] # get the model from the module list

                # find out if this block has batch_normalize or not
                try:
                    batch_normalize = int(self.netBlocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                
                convPart = model[0] # get the convolutional part of this block
                
                # if there is batch normalization for this block
                if (batch_normalize):
                    # the weights file stores info about this layer in this order:
                    # => batch norm bias 
                    # => batch norm weights 
                    # => running mean 
                    # => running variation
                    
                    #Weights file Configuration=> bn bais->bn weights-> running mean-> running var
                    #The weights are arranged in the above mentioned order
                    
                    bnPart = model[1] # get the batch norm layer from the model
                    
                    # get the number of biases that the model uses
                    biasCount = bnPart.bias.numel()
                        # this is actually the same as number of weights, running mean, and
                        # running var (for batch norm layers)
                    
                    # get the biases from the weights file
                    bnBias = torch.from_numpy(weights[tracker:tracker + biasCount])
                    tracker += biasCount # increment tracker
                    
                    # get the weights from the weights file
                    bnPart_weights = torch.from_numpy(weights[tracker: tracker + biasCount])
                    tracker  += biasCount
                    
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
            
            # no need to account for non-convolutional blocks since route, shortcut
            # and yolo layers don't need weights or biases since they don't
            # manipulate the input data at all, just rearrange it
    
    def num_weights(self):
        block_weights = {}# number of weights per block
        for i in range(len(self.moduleList)):
            model = self.moduleList[i]
            block_info = self.netBlocks[i+1]

            module_type = block_info['type'] # get the module's type
            weights = 0 # weight counter (starts at 0)

            if module_type == 'convolutional':
                convPart = model[0] # get the part of the model that does convolution

                ### figure out if there is batch normalization
                try:
                    # if this throws an error, then no batch normalization
                    batch_norm = block_info['batch_normalize']
                except:
                    batch_norm = 0
                
                ### cases for batch norm
                if batch_norm:
                    bnPart = model[1] # get the part of the model that does batch norm
                    weights += bnPart.bias.data.numel() # add number of biases
                    weights += bnPart.weight.data.numel() # add num weights
                    weights += bnPart.running_mean.numel() # add running mean
                    weights += bnPart.running_var.numel() # add running variation
                else:
                    # if no batch norm, then convolution has bias
                    weights += convPart.bias.numel()
                
                ### convolutional part
                weights += convPart.weight.data.numel()
            
            block_weights[i] = weights
        return block_weights


                
'''
#Test CFG:
construct = construct_cfg('cfg/yolov3.cfg')
print(construct,"/n constructed from cfg file")
'''
'''
#TestMOdel:

num_classes = 80
classes = load_classes('data/coco.names') 

model = net('cfg/yolov3.cfg') # make the model from the config file
model.load_weights("yolov3.weights") # load the weights into the network
print("Network loaded")

test_data = torch.randn(1,3,256,256,dtype = torch.float)
test_output = model(test_data,False)

print(test_output.size())
'''
'''
### CONSTRUCT_CFG TESTER
cfg = construct_cfg('./cfg/yolov3.cfg')
for i, block in enumerate(cfg):
    if block['type'] == 'route':
        print(f'{i} route layer: {block["layers"]}')
    else:
        print(f'{i} {block["type"]} layer')
'''
