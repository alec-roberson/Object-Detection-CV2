'''
this file holds network block classes that can be constructed from dictionaries
assembled from the config files.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import make_tup, bbox_wh_iou, bbox_xywh_ious

# +++ custom activation functions


# +++ global functions
def get_activ(activation):
    ''' gets the activation function from the string provided.
    
    --- args ---
    activation : str
        the string name of the activation function.
    
    --- returns ---
    torch.nn.Module : the activation function requested.
    OR
    None : if activation = 'linear'
    '''
    if activation == 'relu':
        act = nn.ReLU() # relu activation
    elif activation == 'leaky':
        act = nn.LeakyReLU(0.1)
    elif activation == 'sigmoid':
        act = nn.Sigmoid()
    elif activation == 'linear':
        act = nn.Identity()
    return act

# +++ network block classes

class DetBlock(nn.Module):
    ''' detection block
    a detection block is at the core of both YOLO and SqueezeDet architectures.
    it is essentially just a bunch of transformations that get applied to the
    previous output to interpret it as detections of bounding boxes for the
    network. this block is also set up to compute the loss of the detections
    alongside the transformations which helps a LOT with network design.

    --- args ---
    block_dict : dict
        the dictionary that describes this yolo block. should contain keys like
        'input_dim', 'anchors', 'classes', and 'ignore_thresh'.
    CUDA : bool, optional (default=False)
    '''
    def __init__(self, block_dict, CUDA=False):
        super(DetBlock, self).__init__() # run super init
        
        # +++ set the device variable
        if CUDA:
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        
        # +++ save the block dict and attributes
        self.block_dict = block_dict # save block dictionary
        self.n = block_dict['layer_num'] # save block number
        self.type = block_dict['type'] # save block type

        # +++ save the inputs
        self.img_dim = block_dict['input_dim'] # input image dimensions
        self.anchors = block_dict['anchors'] # anchor boxes
        self.num_anchors = len(self.anchors) # how many anchor boxes
        self.num_classes = block_dict['classes'] # number of classes
        self.ignore_thresh = block_dict['ignore_thresh'] # confidence threshold
        self.CUDA = CUDA
        
        # +++ initialize some variables
        # loss functions
        # FIXME: add the ability to choose between MSE and SSE using kwarg
        # FIXME: reduction='sum' (or 'mean'?)
        self.mse_loss = nn.MSELoss(reduction=block_dict['loss_reduction']) # SUM/MEAN squared error loss function
        self.bce_loss = nn.BCELoss() # binary cross entropy loss function
        # loss scalings
        self.lambda_obj = block_dict['lambda_obj'] # loss scale for obj
        self.lambda_noobj = block_dict['lambda_noobj'] # loss scale for no obj
        self.lambda_bbox = block_dict['lambda_bbox'] # loss scale for bbox
        self.lambda_cls = block_dict['lambda_cls'] # loss scale for class
        # other things
        self.metrics = {}
        self.grid_size = 0 # the grid size of this detection layer's input
        self.stride = 0 # the size of each grid cell in pixels
        # (these get initialized to zero but will be set whenever forward is called)

        # +++ lastly, send the whole layer to the right place
        self.to(self.device)
    
    # +++ compute grid offsets method
    def compute_grid_offsets(self, grid_size):
        ''' +++ INTERNAL METHOD +++
        computes the grid offsets for each cell in a given grid_size. these get
        stored as the grid_x and grid_y variables.
        '''
        self.grid_size = grid_size # update self.grid_size
        self.stride = self.img_dim / self.grid_size # calculate stride
        g = grid_size # save grid_size as g

        # +++ calculate offsets for each cell in the grid
        grid_range = torch.arange(g).to(self.device) # range of the grid
        self.grid_y, self.grid_x = torch.meshgrid(grid_range, grid_range) # y/x offsets
        
        # unsqueeze and make float tensors
        self.grid_x = self.grid_x.view(1,1,g,g).float()
        self.grid_y = self.grid_y.view(1,1,g,g).float()

        # +++ calculate anchors normalized to the stride
        self.scaled_anchors = torch.FloatTensor(self.anchors).to(self.device)
        self.scaled_anchors /= self.stride

        # +++ get the anchor w and anchor heights
        self.anchor_w = self.scaled_anchors[:,0:1].view(1, self.num_anchors, 1, 1)
        self.anchor_h = self.scaled_anchors[:,1:2].view(1, self.num_anchors, 1, 1)
    
    # +++ forward (implicit __call__) method
    def forward(self, x, targets=None):
        ''' feed forward method
        takes in the output of a yolo network thus far, some tensor with size 
        (batch_size, (num_classes+5) * num_anchors, grid_size, grid_size), and
        outputs a tensor of size ([OUTPUT SIZE]).

        additionally, if targets are passed, the loss of the network will also
        be calculated as it feeds through. how neat!

        --- args ---
        x : torch.Tensor
            the output of the previous network layer (described above).
        targets : IDK
        '''
        # +++ take a looksie at x's dimensions
        self.batch_size = x.size(0) # get the batch size
        grid_size = x.size(2) # get the grid size
        
        # +++ reformat x to get predictions
            # essentially all we do here is breakup x's first dimension, which
            # is (num_classes+5) * num_anchors, into two dimensions of 
            # num_anchors and num_classes+5
        prediction = x.view(self.batch_size, self.num_anchors, self.num_classes+5, grid_size, grid_size)
        # then we want to move the num_classes+5 dimension to the back, so
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()

        # +++ get the outputs from the prediction
        x = torch.sigmoid(prediction[...,0]) # dim 0 => center x
        y = torch.sigmoid(prediction[...,1]) # dim 1 => center y
        w = prediction[...,2] # dim 2 => width
        h = prediction[...,3] # dim 3 => height
        box_conf = torch.sigmoid(prediction[...,4]) # dim 4 => box conf
        cls_conf = torch.sigmoid(prediction[...,5:]) # dim 5: => class predictions

        # +++ if the grid size changed, compute the new offsets and shit
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size)
        
        # +++ adjust the anchors to get the actual bounding box predictions
        pred_boxes = torch.zeros(prediction[...,:4].size()).to(self.device) 
            # a blank tensor with the same size as predictions but with 
            # the last dimension being size 4
        pred_boxes[...,0] = x.data + self.grid_x # add the grid offsets to the
        pred_boxes[...,1] = y.data + self.grid_y # x and y center coordinates
        pred_boxes[...,2] = torch.exp(w)*self.anchor_w # calculate the width and height
        pred_boxes[...,3] = torch.exp(h)*self.anchor_h # using the exponential of output

        # +++ make the output tensor
        output = torch.cat(
            (
                pred_boxes.view(self.batch_size, -1, 4) * self.stride, 
                    # box pred w/ flattened grid, multiplied by the stride
                box_conf.view(self.batch_size, -1, 1), # box conf w/ flat grid
                cls_conf.view(self.batch_size, -1, self.num_classes) # flatten grid
            ), dim = -1)
        
        # +++ should loss be calculated or no?
        if targets == None: # no targets given => no loss
            return output, 0
        else: # targets given => calculate loss
            # +++ use the build targets method to build the targets for this input
            iou_scores, cls_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = \
                self.build_targets(targets, pred_boxes, cls_conf)

            # +++ calculate some losses, using the masks to ignore certain outputs
                # that we don't care about/don't want to affect our loss
            
            ### bounding box loss
            loss_x = self.mse_loss(x * obj_mask, tx * obj_mask) * self.lambda_bbox
            loss_y = self.mse_loss(y * obj_mask, ty * obj_mask) * self.lambda_bbox
            loss_w = self.mse_loss(w * obj_mask, tw * obj_mask) * self.lambda_bbox
            loss_h = self.mse_loss(h * obj_mask, th * obj_mask) * self.lambda_bbox
            loss_bbox = loss_x + loss_y + loss_w + loss_h # total bounding box loss

            ### confidence loss
            # loss from objectness confidence where there IS an object
            loss_conf_obj = self.mse_loss(box_conf * obj_mask, tconf * obj_mask)
            loss_conf_obj *= self.lambda_obj
            # loss from objectness confidence where there IS NOT an object
            loss_conf_noobj = self.mse_loss(box_conf * noobj_mask, tconf * noobj_mask)
            loss_conf_noobj *= self.lambda_noobj
            # total confidence loss
            loss_conf = loss_conf_obj + loss_conf_noobj
            
            ### class loss
            # loss from class predictions where an object was present
            
            loss_cls = self.bce_loss(cls_conf * obj_mask.unsqueeze(-1), 
                                     tcls * obj_mask.unsqueeze(-1))
            loss_cls *= self.lambda_cls
            # sum above to get total loss
            total_loss = loss_bbox + loss_conf + loss_cls
            
            # +++ calculate some METRICS !!!
            cls_acc = (cls_mask * obj_mask).sum()/obj_mask.sum() # percent of correct
                # class predictions (where there was a target object present)
            conf_obj = (box_conf * obj_mask).sum()/obj_mask.sum()
                # average confidence where there was an object present
            conf_noobj = (box_conf * noobj_mask).sum()/noobj_mask.sum() 
                # average confidence where there was no object present
            
            conf50 = (box_conf > 0.5).float() # a 1 for all box confidence values 
                # that were above 50%
            iou50 = (iou_scores > 0.5).float() # a 1 for all the grid cells that 
                # correctly predicted a bounding box with iou > 50%
            iou75 = (iou_scores > 0.75).float() # same as iou50 but > 75%
            
            detected_mask = conf50 * cls_mask * obj_mask # a 1 for all of the cells
                # that detected a box with >50 % confidence, and the right class
            
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
                # precision as (the number of boxes that were detected correctly with
                # iou >50%) / (the number of boxes that the network was >50% confident
                # on)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
                # this is the percent of objects that were detected and given
                # bouding boxes with iou > 50%
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)
                # this is the percent of objects that were detected and given
                # bouding boxes with iou > 75%
            
            # +++ save the metrics
            self.metrics = {
                'loss': total_loss.item(),
                'bbox-loss': loss_bbox.item(),
                'conf-loss': loss_conf.item(),
                'cls-loss': loss_cls.item(),
                'cls-accuracy': cls_acc.item(),
                'recall50': recall50.item(),
                'recall75': recall75.item(),
                'precision': precision.item(),
                'conf-obj': conf_obj.item(),
                'conf-noobj': conf_noobj.item(),
                'grid-size': self.grid_size}
            
            # return the output and the loss
            return output, total_loss
    
    # +++ build targets method
    def build_targets(self, targets, pred_boxes, pred_cls):
        ''' +++ INTERNAL METHOD +++ 
        this function takes in some targets in the form that datasets usually
        give them, i.e.
        [[image_num, class_num, center_x, center_y, width, height],...]
        and creates the masks that can be used to...
        '''
        # +++ set the tensor variables
        LongTensor = torch.cuda.LongTensor if self.CUDA else torch.LongTensor
        FloatTensor = torch.cuda.FloatTensor if self.CUDA else torch.FloatTensor

        # +++ setup the output tensors
        # first set the mask size variable, since many of the masks have the same shape
        mask_size = (self.batch_size, self.num_anchors, self.grid_size, self.grid_size)
        obj_mask = LongTensor(*mask_size).fill_(0) # object mask
        noobj_mask = LongTensor(*mask_size).fill_(1) # no object mask
        cls_mask = FloatTensor(*mask_size).fill_(0) # class mask
        iou_scores = FloatTensor(*mask_size).fill_(0) # iou scores
        tx = FloatTensor(*mask_size).fill_(0) # target x
        ty = FloatTensor(*mask_size).fill_(0) # target y
        tw = FloatTensor(*mask_size).fill_(0) # target width
        th = FloatTensor(*mask_size).fill_(0) # target height
        tcls = FloatTensor(*mask_size, self.num_classes).fill_(0) # target classes

        # +++ check if there are NO targets for this batch
        if targets.numel() == 0: # if there are no targets
            # then we should just return everything as is
            tconf = obj_mask.float() # make the target confidence mask
            return iou_scores, cls_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf 

        # +++ convert the targets coords to the grid scale
        target_boxes = targets[:,2:6] * self.grid_size
        gxy = target_boxes[:,:2] # target box center x and ys
        gwh = target_boxes[:,2:] # target box widths and heights

        # +++ get anchors that have the best IOUs
        ious = bbox_wh_iou(self.scaled_anchors, gwh) # get the IOUs
        best_ious, best_anchs = ious.max(0) # get best IOUs for each target and the best
            # anchor box that results in that IOU, for each target
        
        # +++ seperate the target values
        img_i, target_lbls = targets[:,:2].long().t() # get the img index and class
            # labels for each of the target boxes
        gx, gy = gxy.t() # get target x and y coordinates
        gw, gh = gwh.t() # get target widths and heights
        gi, gj = gxy.long().t() # get target grid indicies responsible for detecting
            # each target box (gi and gj are type long ==> INTEGERS)
        
        # +++ set some values in the masks
        obj_mask[img_i, best_anchs, gj, gi] = 1 # set the grid cells that have the best
            # anchor box and nearest to center of target boxes to one, indicating that
            # this gridcell/anchor box pairing is responsible for detection
        noobj_mask[img_i, best_anchs, gj, gi] = 0 # set the grid cells that are 
            # responsible for detection to 0 in the no target mask, so as to not penalize
            # their predictions
        
        # +++ set noobj_mask to zero when iou exceeds threshold
        for i, anchor_ious in enumerate(ious.t()):
            # so we are looping through each target box given, and we have it's index
            # (in the targets tensor) and the IOUs it has with each of the anchor boxes
            # and we want to say that if the IOU of the target box with an anchor box
            # exceeds a certain threshold, then it's okay for the cell near that target
            # to make a prediction
            # update the mask by using anchor_ious > thresh as index for anchor boxes
            noobj_mask[img_i[i], anchor_ious > self.ignore_thresh, gj[i], gi[i]] = 0
        
        # +++ set the target coordinates for the network
        tx[img_i, best_anchs, gj, gi] = gx - gx.floor() # set center x targets
        ty[img_i, best_anchs, gj, gi] = gy - gy.floor() # set center y targets

        # +++ set width and height targets for the network
        tw[img_i, best_anchs, gj, gi] = \
            torch.log(gw / self.scaled_anchors[best_anchs][:,0] + 1e-16)
            # target widths (as the inverse of the exponential)
        th[img_i, best_anchs, gj, gi] = \
            torch.log(gh / self.scaled_anchors[best_anchs][:,1] + 1e-16)
            # target heights (as the inverse of the exponential)
        
        # +++ one-hot encoding of class labels
        tcls[img_i, best_anchs, gj, gi, target_lbls] = 1

        # +++ compute label corectness and IOU at best grid cell
        cls_mask[img_i, best_anchs, gj, gi] = \
            (pred_cls[img_i, best_anchs, gj, gi].argmax(-1) == target_lbls).float()
            # this one does the class mask, so it holds a 0 or 1 for each grid cell
            # anchor box pairing, a 0 if the cell either did not need to make a 
            # prediction or made the wrong prediction and a 1 if it got it right
        iou_scores[img_i, best_anchs, gj, gi] = \
            bbox_xywh_ious(
            pred_boxes[img_i, best_anchs, gj, gi], target_boxes)
            # set the iou scores for the cells/anchor boxes responsible for predicting
            # the targets to be the IOU of the predicted box with the target boxes
        
        tconf = obj_mask.float() # make obj mask a float which we will use as the
            # target objectness confidence of the network, i.e. it should be 100%
            # certain an object is present where it is, and 0% when it's not
        return iou_scores, cls_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

    # +++ end of YoloBlock class

class ConvolutionalBlock(nn.Sequential):
    ''' convolutional network block
    a convolutional block applies convolution (obviously), along with an 
    activation function and batch normalization, as those are provided.

    --- args ---
    block_dict : dict
        the dictionary that describes this convolutional block.
    '''
    def __init__(self, block_dict):
        super(ConvolutionalBlock, self).__init__() # run super init

        # save the block dict and attributes
        self.block_dict = block_dict # save block dictionary
        self.n = block_dict['layer_num'] # save block number
        self.type = block_dict['type'] # save block type

        # make the convolutional part
        conv = nn.Conv2d(
            block_dict['in_channels'], 
            block_dict['filters'], 
            kernel_size = block_dict['kernel_size'],
            stride = block_dict['stride'],
            padding = block_dict['padding'],
            bias = block_dict['bias'])

        # add conv part to the module
        self.add_module('convolution', conv)

        # maybe make and add batch normalization
        if block_dict['batch_normalize']:
            batch_norm = nn.BatchNorm2d(block_dict['filters']) # make the part
            self.add_module('batch norm', batch_norm) # add it
        
        # add the activation function
        activation = get_activ(block_dict['activation'])
        self.add_module(block_dict['activation'], activation)

class FireBlock(nn.Module):
    ''' fire block
    this is a fire block as described in the paper for sqeeze net.
    
    --- args ---
    block_dict : dict
        the config block dictionary.
    '''
    def __init__(self, block_dict):
        super(FireBlock, self).__init__() # run super init
        # unpack input
        self.n = block_dict['layer_num'] # save layer number
        in_chan = block_dict['in_channels']
        fs = block_dict['fsqueeze']
        fe = block_dict['fexpand']
        
        # make the layers of the module
        self.squeeze = nn.Conv2d(in_chan, fs, kernel_size=1)
        self.expand1 = nn.Conv2d(fs, fe, 1)
        self.expand3 = nn.Conv2d(fs, fe, 3, padding=1)

        # batch normalization
        if block_dict['batch_normalize']:
            self.batch_norm = nn.BatchNorm2d(fe * 2)
        else:
            self.batch_norm = None
        
        # get the activation function to use
        self.activ = get_activ(block_dict['activation'])
    
    def forward(self, x):
        s = self.activ(self.squeeze(x)) # squeeze x
        e1 = self.activ(self.expand1(s)) # expand 1
        e3 = self.activ(self.expand3(s)) # expand 3
        output = torch.cat((e1, e3), dim=1) # concatenate expand layers
        if self.batch_norm != None:
            output = self.batch_norm(output)
        return output

class MaxPoolBlock(nn.Module):
    ''' max pooling block
    a block of the network that performs max pooling.

    --- args ---
    block_dict : dict
        the dictionary that describes this max pooling block. should contain 
        keys like 'kernel_size', 'stride', and 'padding'.
    '''
    def __init__(self, block_dict):
        super(MaxPoolBlock, self).__init__() # super init

        # save the block dict and attributes
        self.block_dict = block_dict # save block dictionary
        self.n = block_dict['layer_num'] # save block number
        self.type = block_dict['type'] # save block type

        # save the given variables
        self.kernel_size = block_dict['kernel_size']
        self.stride = block_dict['stride']
        self.padding = make_tup(block_dict['padding'], tup_len=4, expand_method=1)

    def forward(self, x):
        return F.max_pool2d(
            F.pad(
                x, 
                pad = self.padding,
                mode = 'replicate'),
            kernel_size = self.kernel_size,
            stride = self.stride)
    
class UpsampleBlock(nn.Upsample):
    ''' upsample block
    a network block that performs upsampling.
    
    --- args ---
    block_dict : dict
        the dictionary that describes this block. should have key 
        'scale_factor'.
    '''
    def __init__(self, block_dict):
        # initialize urself
        super(UpsampleBlock, self).__init__(
            scale_factor = block_dict['scale_factor'],
            mode = 'bilinear',
            align_corners = False)
        
        # save the block dict and attributes
        self.block_dict = block_dict # save block dictionary
        self.n = block_dict['layer_num'] # save block number
        self.type = block_dict['type'] # save block type
    
class RouteBlock(nn.Module):
    ''' route block
    a block of the network that performs routing.

    --- args ---
    block_dict : dict
        the dictionary that describes this routing block. should have key 
        'layers'.
    '''
    def __init__(self, block_dict):
        super(RouteBlock, self).__init__() # super init

        # save the block dict and attributes
        self.block_dict = block_dict # save block dictionary
        self.n = block_dict['layer_num'] # save block number
        self.type = block_dict['type'] # save block type

        # save routing layer numbers
        self.layers = block_dict['layers']
    
    def forward(self, layer_activations):
        # keep only the ones we want to route together
        to_route = [layer_activations[i] for i in self.layers]
        return torch.cat(to_route, dim=1)

class ShortcutBlock(nn.Module):
    ''' shortcut block
    a block that describes a shortcut in the network.
    
    --- args ---
    block_dict : dict
        the dictionary that describes this block of the network. should have key
        'from'.
    '''
    def __init__(self, block_dict):
        super(ShortcutBlock, self).__init__() # super init
        
        # save the block dict and attributes
        self.block_dict = block_dict # save block dictionary
        self.n = block_dict['layer_num'] # save block number
        self.type = block_dict['type'] # save block type

        # save layer to shortcut from
        self.layer = block_dict['from']
    
    def forward(self, *layer_activations):
        return sum(layer_activations)

class DropBlock2D(nn.Module):
    ''' dropblock layer 
    a dropblock layer performs dropblock on the incoming data, if and only if
    the model is in training mode.

    --- args ---
    block_size : int
        the (square) size of blocks that will be dropped.
    keep_prob : float between 0 and 1
        the probablility that a certain cell will be kept after going through
        this layer. this can be modified later by setting keep_prob.
    '''
    def __init__(self, block_dict):
        super(DropBlock2D, self).__init__() # run super init

        # save the input variables
        self.n = block_dict['layer_num']
        self.block_size = block_dict['block_size']
        self.target_keep_prob = block_dict['target_keep_prob']
        self.init_keep_prob = block_dict['init_keep_prob']
        self.keep_prob = block_dict['init_keep_prob']

    def set_kp(self, pct):
        ''' set keep prob method

        --- args ---
        pct : float between 0 and 1, optional (defualt=0)
            the percent through training that we are currently
        '''
        self.keep_prob = self.init_keep_prob + \
            pct * (self.target_keep_prob - self.init_keep_prob)
    
    def forward(self, x):
        ''' feed a batch x through the module 
        note that this module will only affect the output if it's in training 
        mode. if its in evaluation mode, it will just return the input.
        
        --- args ---
        x : torch.tensor
            the input to the layer.
        '''
        # +++ first, make sure we're training
        if self.training == False: # if we're not training DO NOTHING
            return x

        # +++ get info about the x tensor
        device = x.device
        batch_size, n_feats, height, width = x.size()

        # +++ since we are training, we need to make the mask and shit
        gamma = (1 - self.keep_prob) / self.block_size ** 2 # calculate the gamma value

        mask = torch.rand((batch_size, 1, height, width)) # random array
        mask = (mask < gamma).float() # keep only ones < gamma
        mask = mask.to(device) # send to same device as x

        # +++ now we need to make the mask into a block mask
        bmask = F.max_pool2d(
            input=mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size//2)
        
        if self.block_size % 2 == 0: # if it's an even block size
            bmask = bmask[:,:,:-1,:-1] # sluff off a bit of the edges
        
        bmask = 1 - bmask # flip 1s and 0s

        # +++ apply the block mask to the input
        out = x * bmask

        # +++ scale the output
        out = out * (bmask.numel() / bmask.sum())

        return out

# +++ net blocks dictionary
NET_BLOCKS = {
    'convolutional': ConvolutionalBlock,
    'fire': FireBlock,
    'maxpool': MaxPoolBlock,
    'upsample': UpsampleBlock,
    'route': RouteBlock,
    'shortcut': ShortcutBlock,
    'dropblock': DropBlock2D}
