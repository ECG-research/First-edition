import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from fastai.layers import *
from fastai.basics import *

from basic_conv1d import AdaptiveConcatPool1d,create_head1d

def conv(in_planes, out_planes, kernel_size=3, stride=1):
    "convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size-1)//2, bias=False)

def noop(x): return x

class InceptionBlock1d(nn.Module):
    def __init__(self, ni, nb_filters, kss, d, stride=1, act='linear', bottleneck_size=32):
        super().__init__()
        self.bottleneck = conv(ni, bottleneck_size, 1, stride) if (bottleneck_size>0) else noop

        self.convs = nn.ModuleList([conv(bottleneck_size if (bottleneck_size>0) else ni, nb_filters, ks) for ks in kss])
        self.conv_bottle = nn.Sequential(nn.MaxPool1d(3, stride, padding=1), conv(ni, nb_filters, 1))
        self.bn_relu = nn.Sequential(nn.BatchNorm1d(nb_filters*(len(kss)+1)), nn.ReLU())

    def forward(self, x):
        #print("block in",x.size())
        bottled = self.bottleneck(x)
        out = self.bn_relu(torch.cat([c(bottled) for c in self.convs]+[self.conv_bottle(x)], dim=1))
        return out
class Shortcut1d(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.act_fn=nn.ReLU(True)
        self.conv=conv(ni, nf, 1)
        self.bn=nn.BatchNorm1d(nf)

    def forward(self, inp, out): 
        #print("sk",out.size(), inp.size(), self.conv(inp).size(), self.bn(self.conv(inp)).size)
        #input()
        return self.act_fn(out + self.bn(self.conv(inp)))
class InceptionBackbone(nn.Module):
    def __init__(self, input_channels, kss, depth, bottleneck_size, nb_filters, use_residual):
        super().__init__()

        self.depth = depth
        assert((depth % 3) == 0)
        self.use_residual = use_residual

        n_ks = len(kss) + 1
        self.im = nn.ModuleList([InceptionBlock1d(input_channels if d==0 else n_ks*nb_filters,
            nb_filters=nb_filters,kss=kss, d= d, bottleneck_size=bottleneck_size) for d in range(depth)])
        self.sk = nn.ModuleList([Shortcut1d(input_channels if d==0 else n_ks*nb_filters,
            n_ks*nb_filters) for d in range(depth//3)])    
        
    def forward(self, x):
        
        input_res = x
        for d in range(self.depth):
            x = self.im[d](x)
            if self.use_residual and d % 3 == 2:
                x = (self.sk[d//3])(input_res, x)
                input_res = x.clone()
        return x
class Cross_attention(nn.Module):
    def __init__(self,query_dim,num_head=2,dropout = 0.1,batch_first= True):
        super().__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(query_dim,num_heads=num_head,dropout=dropout,
                    batch_first=batch_first) for i in range(2)])
    def forward(self,query0,query1):
        feature0 = self.layers[0](query0,query1,query1)[0]
        feature0 = torch.add(feature0,query0)
        feature1 = self.layers[1](query1,query0,query0)[0]
        feature1 = torch.add(feature1,query1)
        return torch.cat((feature1,feature0),dim = -1)

class Modified_version(nn.Module):
    '''inception time architecture'''
    def __init__(self,num_leads = 3, num_classes=5, input_channels=1, kernel_size=40, depth=3, bottleneck_size=32, nb_filters=32, 
            use_residual=True,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
        super().__init__()
        assert(kernel_size>=40)
        kernel_size = [k-1 if k%2==0 else k for k in [kernel_size,kernel_size//2,kernel_size//4]] #was 39,19,9
        self.num_leads = num_leads
        #hyperpara
        self.n_ks = len(kernel_size) + 1
        self.bottleneck_size = bottleneck_size
        #inceptiontime layers depth = depth
        self.Incep_layers = nn.ModuleList([InceptionBackbone(input_channels=input_channels, kss=kernel_size, depth=depth, bottleneck_size=bottleneck_size, 
                nb_filters=nb_filters, use_residual=use_residual) for i in range(num_leads)])
        self.Incep_layers.append(nn.Linear(32,self.num_leads*self.n_ks*self.bottleneck_size)) # for metadata
        #MHA
        self.MHA = Cross_attention(self.num_leads*self.n_ks*self.bottleneck_size)
        #classify layers
        layers = []
        layers.append(nn.Flatten())
        layers.append(nn.Linear(2*self.num_leads*self.n_ks*self.bottleneck_size,num_classes))
        self.layers = nn.Sequential(*layers)
    def forward(self,x0,x1,x2,metadata):
        x0 = self.Incep_layers[0](x0)
        x1 = self.Incep_layers[1](x1)
        x2 = self.Incep_layers[2](x2)
        metadata = self.Incep_layers[3](metadata) #project to embedding space

        feature = torch.cat([x0,x1,x2],dim= 1)
        feature = nn.AdaptiveAvgPool1d(1)(feature) #match the dim of metadata
        feature = torch.reshape(feature, (-1,1,self.num_leads*self.n_ks*self.bottleneck_size)) #the required shape for MHAttention
        #multiple head attention
        enhanced_feature = self.MHA(feature,metadata)
        #classify
        return self.layers(enhanced_feature)

#create model
def modified_version(**kwargs):
    return Modified_version(**kwargs)