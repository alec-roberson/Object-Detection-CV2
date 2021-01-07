import torch
import torchvision
from torchvision import transforms,datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
#big ol variables
BATCHSIZE = 50
EPOCHS = 10
GPU = True

if GPU:
    device = 'cuda:0'
else:
    device = 'cpu'
#loading data
train = datasets.MNIST('', train=True, download=True,
                    transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST('', train=False, download=True,
                    transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train,batch_size = BATCHSIZE, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size = BATCHSIZE, shuffle=True)

#main network class
class Net(nn.Module):
    def __init__(self):
        super().__init__()#superinit!
        #convolutional layers
        self.conv1 = nn.Conv2d(1, 64, 5)
        #self.conv2 = nn.Conv2d(32,64, 3)
        #fully connected
        self.fc3 = nn.Linear(12*12*64, 128)
        self.fc4 = nn.Linear(128, 10)

    def convs(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,(2,2))
        #x = F.relu(self.conv2(x))
        #x = F.max_pool2d(x,(2,2))
        return x

    def forward(self,x):
        #convs
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,(2,2))
        #x = F.relu(self.conv2(x))
        #x = F.max_pool2d(x,(2,2))
        x = x.view(-1,1,12*12*64)
        #fcs
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x),dim=0)
        #x = F.relu(self.fc4(x))
        return x

net = Net()
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=0.002)
lossfunction = nn.NLLLoss()

'''
tl = list(trainset)
x = tl[0][0][0]
y = tl[0][1][0]
o = net.convs(x.view(-1,1,28,28))
print(o)
print(x.shape)
print(o.shape)

'''
from tqdm import tqdm
training_losses = []
batch_losses = []
for epoch in range(1,1+EPOCHS):
    print('EPOCH = ',epoch,' / ',EPOCHS)
    losses = []
    for data in tqdm(trainset):
        x,y = data
        x = x.view(-1,1,28,28).to(device)
        y = y.to(device)
        net.zero_grad() # zeros gradient
        output = net(x) # 
        loss = lossfunction(output.view(-1,10),y.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss)
    training_losses += losses
    avgloss = sum(losses).item()/len(losses)
    print('average loss = ', avgloss)
    batch_losses.append(avgloss)

#TESTING
correct = 0
total = 0
with torch.no_grad():
    for data in tqdm(testset):
        x,y = data
        x = x.view(-1,1,28,28).to(device)
        y.to(device)
        out = net(x.view(-1,1,28,28))
        for idx, o in enumerate(out):
            if torch.argmax(o) == y[idx]:
                correct += 1
            total += 1
accuracy = round(100*correct/total,4)
print('accuracy = ',accuracy, ' %')























####
