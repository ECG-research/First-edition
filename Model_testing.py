import torch
import torch.nn as nn
import torch.nn.functional as F
from third_edition import PositionalEncoding
from Conv_bigru_atten import H_att

from basic_conv1d import AdaptiveConcatPool1d,create_head1d

def maxpool(kernel,stride):
    return nn.MaxPool1d(kernel,stride=stride,padding=(kernel-1)//2)
def conv(in_planes, out_planes, kernel_size=3, stride=1, groups=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                    padding=(kernel_size-1)//2,bias = False,groups=groups)
def avgpool(kernel,stride):
    return nn.AvgPool1d(kernel,stride=stride,padding=(kernel-1)//2)

t = 4
class Model_conv_block(nn.Module):
    def __init__(self,ic,oc_filter,bottle_neck=32,stride=1):
        super().__init__()
        self.s_conv = nn.Conv1d(ic,bottle_neck,1,stride=1)
        #0.931
        self.s_convb = conv(bottle_neck,oc_filter,39,stride=stride)
        self.s_convm = conv(bottle_neck,oc_filter,19,stride=stride)
        self.s_convs = conv(bottle_neck,oc_filter,9,stride=stride)

        self.t_pool = maxpool(3,stride=stride)
        self.t_conv = conv(ic,oc_filter,3)

        self.skip = conv(ic,oc_filter*t,3,stride=stride)

        self.bn_relu = nn.Sequential(nn.BatchNorm1d(oc_filter*t), nn.ReLU())
    def forward(self,x):
        bs = []
        # spatial
        b = self.s_conv(x)
        bs.append(self.s_convb(b))
        bs.append(self.s_convm(b))
        bs.append(self.s_convs(b))
        bs.append(self.t_conv(self.t_pool(x)))
        bs = torch.concatenate(bs,dim = 1)
        return self.bn_relu(bs + self.skip(x))

class MultiAtten(nn.Module):
    def __init__(self,num_head,oc_filter,input_dim=1000):
        super().__init__()
        self.num_head = num_head
        self.attens = nn.ModuleList([H_att(oc_filter*t,input_dim,dropout=0.0) for _ in range(num_head)])
    def forward(self,x):
        lst = []
        for i in range(self.num_head):
            lst.append(self.attens[i](x))
        return torch.concatenate(lst,dim=1)
    
    
class Model_test(nn.Module):
    def __init__(self,num_class=5,ic=12,oc_filter=32,bottle_neck=32,depth=7,stride=1,input_dim=1000):
        super().__init__()
        self.blocks = nn.Sequential(*[Model_conv_block(ic=oc_filter*t*2 if i>=depth-1 else oc_filter*t,
                                                       oc_filter=2*oc_filter if i>=depth-2 else oc_filter,
                                                       bottle_neck=2*bottle_neck if i>=depth-2 else bottle_neck,
                                                       stride=2 if i>=depth-2 else stride) for i in range(depth-1)])
        self.blocks.insert(0,Model_conv_block(ic=ic,oc_filter=oc_filter,bottle_neck=bottle_neck,stride=stride))

        # self.blocks = nn.Sequential(*[Model_conv_block(ic=ic if i==0 else oc_filter*t,
        #                                                oc_filter=oc_filter,
        #                                                bottle_neck=bottle_neck ,
        #                                                stride=stride) for i in range(depth)])
        self.pool = AdaptiveConcatPool1d(1)

        # self.pos = PositionalEncoding(oc_filter*t,input_dim)
        # self.pool = nn.AdaptiveAvgPool1d(1)
        # self.att = H_att(oc_filter*t,input_dim,dropout=0.1)
        # self.att = MultiAtten(2,oc_filter,input_dim=input_dim)

        classify = [nn.Flatten()]
        classify.append(nn.Dropout(0.05))
        classify.append(nn.Linear(4*oc_filter*t,num_class))
        classify.append(nn.Sigmoid())
        self.classify = nn.Sequential(*classify)
    def forward(self,x):
        x = self.blocks(x)
        x = self.pool(x)
        # skip = self.pool(x)
        # x = torch.transpose(x,1,2)
        # x = self.att((x,skip))
        return self.classify(x)