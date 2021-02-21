''' train.py
this file trains the network.
'''

# +++ imports
import os
import random
from tqdm import tqdm
import torch 
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import NetworkModel
from datamanager import DataManager

# +++ training prefrences
CUDA = True # should cuda be used?
NUM_EPOCHS = 1000 # number of epochs to run for
BATCH_SIZE = 64 # batch size
MINI_BATCH_SIZE = 32 # mini batch size
MOSAICS = None # amount of mosaic augmented data to train on
LEARNING_RATE = 0.01 # learning rate
WEIGHT_DECAY = 0.001 # learning rate decay
BETAS = (0.9, 0.999) # betas for Adam training
WRITE_EVERY = 1 # tensorboard data will be written every ___ epochs
START_SAVING_BEST = None # how far into training saving should begin

# +++ file locations
load_net = None # if you'd like to load a network, set this to the path of the network
net_cfg = 'cfg/squeezedet.cfg' # network configuration file
classes_file = 'data/class-names.labels' # class labels file
train_directory = 'data/train' # training images/labels directory
test_directory = 'data/test' # testing images/labels directory
net_name = 'chess-squeezedet' # where to save the network after training
tb_logdir = 'runs/' + net_name # log for tensorboard

# +++ setup files
save_net = net_name + '.pt' # file to save net to
### save_best_net = net_name + '-best.pt' # file to save "best" net to


# +++ set the device variable
device = 'cuda:0' if CUDA else 'cpu'

# +++ setup the network
if load_net == None: # if we're not loading a network
    # initialize a new network!
    model = NetworkModel(net_cfg, CUDA=CUDA)
else: # if we are loading a network
    netf = open(load_net, 'rb') # open the network file
    model = torch.load(netf) # load the model
    netf.close() # close the network file

model.to(device) # send network to the right device
model.train() # put in training mode

# +++ load up them datas
train_data = DataManager(
    path = train_directory, 
    class_file_path = classes_file,
    input_dim = model.input_dim)
test_data = DataManager(
    path = test_directory, 
    class_file_path = classes_file,
    input_dim = model.input_dim)

# +++ setup the optimizer and shit
# optimizer = optim.SGD(
#     model.parameters(),
#     lr=LEARNING_RATE,
#     momentum=0.9,
#     weight_decay=WEIGHT_DECAY)
optimizer = optim.Adam(
    model.parameters(), # filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=BETAS)

# +++ tensorboard setup 
writer = SummaryWriter(log_dir=tb_logdir)

# +++ log metrics function
def log_metrics(step):
    # +++ first, plotting the metrics collected by the model
    plots = [
        ('loss', 'loss'),
        ('loss breakdown', ('bbox-loss','conf-loss','cls-loss')),
        ('confidence', ('conf-obj', 'conf-noobj')),
        ('class accuracy', 'cls-accuracy'),
        ('percent correct detections', ('recall50', 'recall75')),
        ('precision', 'precision')]
    
    metrics = model.metrics # get the metrics
    for plot_name, plot_keys in plots:
        if isinstance(plot_keys, str): 
            # single value plot
            writer.add_scalar(plot_name, metrics[plot_keys], global_step=step)
        else:
            # multivalue plot
            plot_dict = dict([(k, metrics[k]) for k in plot_keys])
            writer.add_scalars(plot_name, plot_dict, global_step=step)
    model.reset_metrics() # reset the model's metrics
    
    # +++ now, we test the model on the test data
    data, targets = test_data.batches()[0] # get the test batch
    data, targets = data.to(device), targets.to(device)

    with torch.no_grad(): # don't track gradients
        model.eval() # put in evaluation mode
        _, loss =  model(data, targets=targets) # get the loss of the model
        model.train() # put back in train mode
    loss = loss.cpu().item() # get the float loss

    writer.add_scalar('test loss', loss, global_step=step) # add it to tensorboard

# +++ main training loop
for epoch in tqdm(range(1, NUM_EPOCHS+1), 'training'):

    # +++ loop through batches in this epoch
    for batch in train_data.batches(BATCH_SIZE, MINI_BATCH_SIZE, MOSAICS):

        model.zero_grad() # zero the model's gradients
        batch_loss = 0. # batch loss

        # +++ loop through minibatches in the batch
        for x, targets in batch:
            x = x.to(device) # send x to right device
            targets = targets.to(device) # send targets to the right device
            _, loss = model(x, targets=targets) # feed through the network
            batch_loss += loss # add the loss to the batch loss

        batch_loss.backward() # backpropogate all batch loss

        optimizer.step() # take a step with the optimizer
    
    # +++ check if we should write data now!
    if epoch % WRITE_EVERY == 0:
        log_metrics(epoch) # log the metrics for this epoch

    # +++ update the dropblocks
    model.set_db_kp(epoch / (NUM_EPOCHS - 1))

f = open(save_net, 'wb')
torch.save(model, f)
f.close()
