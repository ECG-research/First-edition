import torch
import torch.nn as nn
import torch.nn.functional as F

class Block_STCNN1(nn.Module):
    def __init__(self, ins, outs, kernel, pool = None, stride = 1, avg = False):
        super().__init__()
        self.conv = nn.Conv2d(ins,outs,kernel,stride = stride,padding = "same")
        self.batchnorm = nn.BatchNorm2d(outs)
        self.relu = nn.ReLU()
        if avg:
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = nn.MaxPool2d((pool,1), stride = 1, padding = (pool//2,0))
    def forward(self,x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return self.pool(x)
    
class Block_STCNN2(nn.Module):
    def __init__(self, ins, outs, kernel, pool = None, stride = 1, avg = False):
        super().__init__()
        self.conv = nn.Conv2d(ins,outs,kernel,stride = stride,padding = "same")
        self.batchnorm = nn.BatchNorm2d(outs)
        self.relu = nn.ReLU()
        if avg:
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = nn.MaxPool2d((pool,1), stride = 1, padding = (pool//2,0))
    def forward(self,x,skip):
        x = self.conv(x)
        x = self.batchnorm(x)
        # this for skip connection
        x = x + skip
        x = self.relu(x)
        return self.pool(x)
    
class Dense(nn.Module):
    def __init__(self,ins,outs,regularizer,dropout):
        super().__init__()
        dense = []
        dense.append(nn.Flatten())
        dense.append(nn.Linear(ins,outs))
        dense.append(nn.BatchNorm1d(outs,eps = regularizer))
        dense.append(nn.Dropout(dropout))
        self.dense = nn.Sequential(*dense)
    def forward(self,x):
        return self.dense(x)

class ST_CNN5(nn.Module):
    def __init__(self,in_channel = 12):
        super().__init__()

        self.block1 = Block_STCNN1(in_channel,32,(5,1),3) # I changed the pool's kernels (2->3 and 4->5) so that model could run with pytorch
        self.block2 = Block_STCNN1(32,32,(5,1),5)  # the convolutional's kernels could not be (1,5) but (5,1) because the input was (1000,1)
        self.block3 = Block_STCNN2(32,64,(5,1),3)
        self.block4 = Block_STCNN1(64,64,(3,1),5)
        self.block5 = Block_STCNN2(64,64,(3,1),3)
        self.block6 = Block_STCNN1(64,64,(2,1),avg = True) # (2,1) on github but (12,1) in the paper

        self.convC1 = nn.Conv2d(32,64,(7,1), padding = "same")
        self.convC2 = nn.Conv2d(64,64,(6,1), padding = "same")
        self.convE1 = nn.Conv2d(64,32,(4,1), padding = "same")
        self.convE2 = nn.Conv2d(32,64,(5,1), padding = "same")

        self.dense1 = Dense(64,128,0.005,0.2) # the dropout coeffs on github did not match those on the paper!
        self.dense2 = Dense(128,64,0.009,0.25)
        self.classify = nn.Linear(64,5)
        self.sig = nn.Sigmoid()
    def forward(self,x):
        x = self.block1(x)
        out1 = x
        out1 = self.convC1(out1)
        out1 = self.convC2(out1)
        x = self.block2(x)
        x = self.block3(x,out1)
        out2 = x
        out2 = self.convE1(out2)
        out2 = self.convE2(out2)
        x = self.block4(x)
        x = self.block5(x,out2)
        x = self.block6(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.classify(x)
        return self.sig(x)


        