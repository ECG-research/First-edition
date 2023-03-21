import torch
import torch.nn as nn
import torch.nn.functional as F


class maxpool(nn.Module):
    def __init__(self,kernel,stride):
        super().__init__()
        ker_row = kernel[0]
        ker_col = kernel[1]
        self.pad = nn.ConstantPad2d((0,0,ker_col-1,ker_row-1), 0)
        self.pool = nn.MaxPool2d(kernel,stride = stride)
    def forward(self,x):
        x = self.pad(x)
        return self.pool(x)

class Block_STCNN1(nn.Module):
    def __init__(self, ins, outs, kernel, pool = None, stride = 1, avg = False):
        super().__init__()
        self.conv = nn.Conv2d(ins,outs,kernel,stride = stride,padding = "same")
        self.batchnorm = nn.BatchNorm2d(outs)
        self.relu = nn.ReLU()
        if avg:
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = maxpool((pool,1), stride = 1)
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
            self.pool = maxpool((pool,1), stride = 1)
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
        # dense.append(nn.Flatten())
        dense.append(nn.Linear(ins,outs))
        dense.append(nn.BatchNorm1d(outs,eps = regularizer))
        dense.append(nn.ReLU())
        dense.append(nn.Dropout(dropout))
        self.dense = nn.Sequential(*dense)
    def forward(self,x):
        return self.dense(x)

class ST_CNN5(nn.Module):
    def __init__(self,in_channel = 12):
        super().__init__()

        self.block1 = Block_STCNN1(in_channel,32,(5,1),2) 
        self.block2 = Block_STCNN1(32,32,(5,1),4) 
        self.block3 = Block_STCNN2(32,64,(5,1),2)
        self.block4 = Block_STCNN1(64,64,(3,1),4)
        self.block5 = Block_STCNN2(64,64,(3,1),2)
        self.block6 = Block_STCNN1(64,64,(12,1),avg = True)

        self.convC1 = nn.Conv2d(32,64,(7,1), padding = "same")
        self.convC2 = nn.Conv2d(64,64,(6,1), padding = "same")
        self.convE1 = nn.Conv2d(64,32,(4,1), padding = "same")
        self.convE2 = nn.Conv2d(32,64,(5,1), padding = "same")

        self.dense1 = Dense(64,128,0.005,0.1) 
        self.dense2 = Dense(128,64,0.009,0.15)
        self.flat = nn.Flatten()
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
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.classify(x)
        return self.sig(x)


        