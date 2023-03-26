import torch
import torch.nn as nn
import torch.nn.functional as F
from Inception1d import inception1d
from torchvision import models

def conv2d(inplanes, planes, kernelsize, bias = True, stride = 1):
    ker_row = kernelsize[0]
    ker_col = kernelsize[1]
    return nn.Conv2d(inplanes, planes, kernel_size=kernelsize, padding = ((ker_row-1)//2,(ker_col-1)//2), bias = bias, stride=stride)
def conv1d(inplanes, planes, kernelsize, bias = False, stride = 1):
    return nn.Conv1d(inplanes, planes, kernel_size=kernelsize, padding = (kernelsize-1)//2, bias = bias, stride=stride)

#create seblock
class SEblock(nn.Module):
    def __init__(self, inplanes, planes, kernel = (3,7), reduction=16, downsample=False, stride = (1,4), dropout = 0.1):
        super().__init__()
        self.downsample = downsample
        if downsample:    
            self.downsample = conv2d(inplanes, planes, kernel, stride=stride)
            self.bn_down = nn.BatchNorm2d(planes)
            self.conv1 = conv2d(inplanes, planes, kernel, stride=stride)
        else:
            self.conv1 = conv2d(inplanes, planes, kernel, stride=1)
            self.residual = conv2d(inplanes, planes, kernel, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv2d(planes, planes, kernel, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(dropout)
        # SE
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(
            planes , planes // reduction, kernel_size=1, bias=False)
        self.conv_up = nn.Conv2d(
            planes // reduction, planes , kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        #SE
        score = self.global_pool(out)
        score = self.conv_down(score)
        score = self.relu(score)
        score = self.conv_up(score)
        score = self.sig(score)

        if self.downsample:
            residual = self.downsample(residual)
            residual = self.bn_down(residual)
        else:
            residual = self.residual(residual)

        res = score * out + residual
        res = self.relu(res)

        return res

class SE_ResNet(nn.Module):
    def __init__(self,in_channel,type):
        super().__init__()
        
        stem = []
        if type == "stft":
            stem_kernel = 5
            maxpool_stride = 1
            maxpool_kernel = 3
            se_kernel = 3
        elif type == "wavelet":
            stem_kernel = (5,15)
            maxpool_stride = (2,2)
            maxpool_kernel = (3,7)
            se_kernel = (3,7)
        else:    
            stem.append(nn.MaxPool2d(4,stride = 4))
            stem_kernel = 15
            maxpool_stride = 2
            maxpool_kernel = 7
            se_kernel = 7
        
        stem.append(conv2d(in_channel, 64, stem_kernel, stride=1))
        stem.append(nn.MaxPool2d(maxpool_kernel,stride = maxpool_stride, padding = ((maxpool_kernel[0]-1)//2,(maxpool_kernel[1]-1)//2)))
        stem.append(nn.BatchNorm2d(64))
        stem.append(nn.Dropout(0.2))
        stem.append(nn.ReLU())
        self.stem = nn.Sequential(*stem)

        stage1 = [SEblock(64,64,se_kernel,downsample=False),SEblock(64,64,se_kernel)]
        stage2 = [SEblock(64,128,se_kernel,downsample=True),SEblock(128,128,se_kernel)]
        stage3 = [SEblock(128,256,se_kernel,downsample=True),SEblock(256,256,se_kernel)]
        # stage4 = [SEblock(256,256,se_kernel,downsample=True),SEblock(256,256,se_kernel)]
        self.se_stage = stage1+stage2+stage3
        self.se_stage = nn.Sequential(*
            self.se_stage
        )
    def forward(self,x):
        x = self.stem(x)
        return self.se_stage(x)

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        ) # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001, # value found in tensorflow
            momentum=0.1, # default pytorch value
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

#projetion_module
class Projection_Module(nn.Module):
    def __init__(self, in_channel = 12, type = "stft", d_meta = 64, embed_dims = 128,conv = "seresnet",rgb = False):
        super().__init__()
        assert type == "stft" or type == "wavelet" or type == "wignerville" or type == "raw", "Wrong module type, it should be wavelet, stft, wignerville or raw"
        #se-resnet 
        self.d_meta = d_meta
        if type == "stft" or type =="wavelet" or type == "wignerville":
            self.feature = SE_ResNet(in_channel, type = type)
            self.proj = nn.Linear(256+d_meta,embed_dims)
        else:
            self.feature = inception1d(False)
            self.proj = nn.Linear(256+d_meta,embed_dims)
        self.type = type
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        #metadata
        meta_layers = []
        meta_layers.append(nn.Conv1d(32,48,kernel_size = 1))
        meta_layers.append(nn.ReLU())
        meta_layers.append(nn.Conv1d(48,d_meta,kernel_size = 1))
        self.meta_layers = nn.Sequential(*meta_layers)

    def forward(self,x,metadata):
        
        x = self.feature(x)
        if self.type == "stft" or self.type == "wavelet" or self.type == "wignerville":
            # from (-1,512,1,1) to (-1,1,512)
            x = self.pool(x)
            x = torch.reshape(x, (-1,1,256))
        else:
            x = torch.reshape(x, (-1,1,256))
        # metadata
        metadata = self.meta_layers(metadata)
        metadata = torch.reshape(metadata,(-1,1,self.d_meta))

        feature = torch.cat((x,metadata),dim = 2)
       
        return self.proj(feature)

# class Projection_Module1(nn.Module):
#     def __init__(self, in_channel = 12, type = "wavelet", d_meta = 64,embed_dims = 128,weights = None,conv = "alexnet",rgb = True):
#         super().__init__()
#         assert type == "stft" or type == "wavelet" or type == "wignerville", "Wrong module type, it should be wavelet, stft or wignerville"
#         #se-resnet 
#         self.d_meta = d_meta
#         if conv == "seresnet":
#             self.conv = SE_ResNet(in_channel, type = type)
#             if rgb:
#                 self.conv = SE_ResNet(in_channel, type = type)
#             self.num_ftrs = 512
#             self.pool = nn.AdaptiveAvgPool2d((1,1))
#             self.proj = nn.Linear(512+d_meta,128)
#         elif conv == "densenet": # 161, 201, 121, 169 tuỳ vào trường hợp
#             dense = models.densenet121(weights=weights)
#             if rgb:
#                 dense.features[0] = nn.Conv2d(in_channel, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#             self.num_ftrs = dense.classifier.in_features
#             dense.classifier = Identity()
#             self.conv = dense
#             self.proj = nn.Linear(self.num_ftrs+d_meta,128)
#         elif conv == "resnet": # ở đây thích thì thay bằng resnet34, nhma nhìn chung là doesnt work
#             resnet = models.resnet18(weights=weights)
#             if rgb:
#                 resnet.conv1 = nn.Conv2d(in_channel,64,kernel_size=(7,7),stride = (2,2),padding = (3,3),bias=False)
#             resnet.fc = Identity() #deo on roi
#             resnet.layer1.insert(1,nn.Dropout(0.2))
#             resnet.layer2.insert(1,nn.Dropout(0.2))
#             resnet.layer3.insert(1,nn.Dropout(0.2))
#             # resnet.layer4.insert(1,nn.Dropout(0.2))
#             resnet.layer3 = Identity()
#             resnet.layer4 = Identity()
#             self.conv = resnet
#             self.num_ftrs = 128
#             self.proj = nn.Linear(self.num_ftrs+d_meta,128)
#         elif conv == "alexnet": # thằng này thì bỏ qua cái đã, nó không có đủ sâu,
#             alexnet = models.alexnet(weights=weights)
#             if rgb:
#                 alexnet.features[0] = nn.Conv2d(in_channel,64,kernel_size=(11,11),stride = (4,4),padding = (2,2))
#             alexnet.features.insert(4,nn.Dropout(0.1)).insert(8,nn.Dropout(0.1)).insert(12,nn.Dropout(0.1))
#             self.conv = alexnet
#             self.num_ftrs = 1000
#             self.proj = nn.Linear(1000+64,128)
#         elif conv == "googlenet": # ggnet đang ko có work đâu nha
#             ggnet = models.googlenet(weights=weights)
#             if rgb:
#                 ggnet.conv1 = BasicConv2d(in_channel,64,kernel_size=(7,7),stride=(2,2),padding=(3,3))
#             self.conv = ggnet
#             self.num_ftrs = 1000 
#             self.proj = nn.Linear(1000+64,128)       
#         self.pool = nn.AdaptiveAvgPool2d((1,1))       
#         #project onto embedding space

#         #metadata
#         meta_layers = []
#         meta_layers.append(nn.Conv1d(32,48,kernel_size = 1))
#         meta_layers.append(nn.ReLU(inplace = True))
#         meta_layers.append(nn.Conv1d(48,d_meta,kernel_size = 1))
#         self.meta_layers = nn.Sequential(*meta_layers)

#     def forward(self,x,metadata):
#         x = self.conv(x)
#         # from (-1,512,1,1) to (-1,1,512)
#         if self.conv == "seresnet":
#             x = self.pool(x)
#         x = torch.reshape(x, (-1,1,self.num_ftrs))
#         # metadata
#         metadata = self.meta_layers(metadata)
#         metadata = torch.reshape(metadata,(-1,1,self.d_meta))

#         feature = torch.cat((x,metadata),dim = 2)
#         return self.proj(feature)

class Pseudo_classifier(nn.Module):
    def __init__(self,num_class = 5, embed_dims = 128):
        super().__init__()

        p_classify = []
        p_classify.append(nn.Flatten())
        p_classify.append(nn.Linear(embed_dims,num_class))
        p_classify.append(nn.Sigmoid())
        self.p_classify = nn.Sequential(*p_classify)
        
    def forward(self,spec):
        return self.p_classify(spec)

class Branch(nn.Module):
    def __init__(self, in_channel = 12, type = "stft", embed_dims = 128, classifier = None, num_class = 5):
        super().__init__()

        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = Pseudo_classifier(num_class, embed_dims=embed_dims)
        # feature extracting model
        self.pm = Projection_Module(in_channel, type, embed_dims=embed_dims)
        
    def forward(self,x,metadata):
        self.x = self.pm(x,metadata)
        return self.classifier(self.x)

class Gated_fusion(nn.Module):
    def __init__(self, num_modal, embed_dims = 128, skip = False, first = False, will_classify = False):
        super().__init__()
        if skip == False and first == True:
            raise "'first' only be activated when 'skip' = True"
        self.skip = skip
        self.first = first
        list_conv = []
        self.list_matmul = []
        self.will_classify = will_classify
        self.num_modal = num_modal
        for _ in range(num_modal):
            if skip:
                if first:
                    list_conv.append(conv1d(num_modal,1,kernelsize = 1))
                else:
                    list_conv.append(conv1d(2*num_modal,1,kernelsize = 1))
            else:
                list_conv.append(conv1d(num_modal,1,kernelsize = 1))
            self.list_matmul.append(nn.Linear(embed_dims,embed_dims, bias = False))
        self.list_matmul = nn.ModuleList(self.list_matmul)
        self.list_conv = nn.ModuleList(list_conv)
        self.sig = nn.Sigmoid()
    def forward(self,x):
        if self.skip:
            if self.first:
                conv_weights = torch.cat(x,dim = 1) 
            else:
                x, prevx = x
                for i in range(len(x)):
                    prevx.append(x[i])
                conv_weights = torch.cat(prevx,dim = 1)
        else:
            conv_weights = torch.cat(x,dim = 1)   
        weights = []
        for i, mat in enumerate(self.list_matmul):
            weights.append(self.sig(mat(self.list_conv[i](conv_weights[i]))))
        out = []
        for i in range(self.num_modal):
            out.append(x[i] * weights[i])
        if self.will_classify:
            return torch.sum(torch.cat(out,dim = 1),dim = 1)
        if self.skip:
            return out,x
        else:
            return out

# class Gated_fusion(nn.Module):
#     def __init__(self, num_modal, embed_dims = 128, first = False, will_classify = False):
#         super().__init__()
#         self.first = first
#         list_conv = []
#         self.list_matmul = []
#         self.will_classify = will_classify
#         self.num_modal = num_modal
#         for _ in range(num_modal):
#             if first:
#                 list_conv.append(conv1d(num_modal,1,kernelsize = 1))
#             else:
#                 list_conv.append(conv1d(2*num_modal,1,kernelsize = 1))
#             self.list_matmul.append(nn.Linear(embed_dims,embed_dims, bias = False))
#         self.list_matmul = nn.ModuleList(self.list_matmul)
#         self.list_conv = nn.ModuleList(list_conv)
#         self.sig = nn.Sigmoid()
#     def forward(self,x):
#         if self.first:
#             conv_weights = torch.cat(x,dim = 1) 
#         else:
#             x, prevx = x
#             for i in range(len(x)):
#                 prevx.append(x[i])
#             conv_weights = torch.cat(prevx,dim = 1)
#         weights = []
#         for i, mat in enumerate(self.list_matmul):
#             weights.append(self.sig(mat(self.list_conv[i](conv_weights[i]))))
#         out = []
#         for i in range(self.num_modal):
#             out.append(x[i] * weights[i])
#         if self.will_classify:
#             return torch.sum(torch.cat(out,dim = 1),dim = 1)
#         return out,x

class Gated_block(nn.Module):
    def __init__(self, num_modal, embed_dims = 128, first = False, will_classify = False):
        super().__init__()
        self.will_classify = will_classify
        self.block1 = Gated_fusion(num_modal, embed_dims = embed_dims, skip = True, first = first, will_classify = False)
        self.block2 = Gated_fusion(num_modal, embed_dims = embed_dims, will_classify = False)
        self.block3 = Gated_fusion(num_modal, embed_dims = embed_dims, will_classify = will_classify)
    def forward(self,x):
        self.out_block, prevx = self.block1(x)
        self.out_block = self.block2(self.out_block)
        self.out_block = self.block3(self.out_block)
        return self.out_block, prevx

class Second_edition(nn.Module):
    def __init__(self, projection_module, num_class = 5, embed_dims = 128, num_gate_blocks = 3):
        super().__init__()
        self.num_gate_blocks = num_gate_blocks-1 # this is not a mistake, the first gated_fusion is different
        self.num_branch = len(projection_module)
        self.proj_module = nn.ModuleList(projection_module)
        self.gated_fusion = []
        self.gated_fusion.append(Gated_fusion(self.num_branch, embed_dims=embed_dims))
        self.gated_fusion.append(Gated_block(self.num_branch, embed_dims=embed_dims, first = True))
        for i in range(num_gate_blocks):
            if i == num_gate_blocks-1:
                self.gated_fusion.append(Gated_block(self.num_branch, embed_dims=embed_dims, will_classify=True))
            else:
                self.gated_fusion.append(Gated_block(self.num_branch, embed_dims=embed_dims))
        # self.gated = Gated_fusion(self.num_branch, embed_dims=embed_dims)
        self.gated_fusion = nn.Sequential(*self.gated_fusion)
        classify = []
        classify.append(nn.Flatten())
        classify.append(nn.Linear(embed_dims,num_class))
        classify.append(nn.Sigmoid())
        self.classify = nn.Sequential(*classify)

    def forward(self,x, metadata):
        self.features = []
        for i in range(self.num_branch):
            feature = self.proj_module[i](x[i], metadata)
            self.features.append(feature)

        # out_gate = self.gated(self.features)
        out_gate, prev_gate = self.gated_fusion(self.features)
        # out_gate = out_gate + prev_gate
        return self.classify(out_gate)

# class Second_edition1(nn.Module):
#     def __init__(self, projection_module, num_class = 5, query_dim = 128, num_head=2, dropout = 0.2,batch_first= True, num_atten = 1):
#         super().__init__()
#         self.num_branch = len(projection_module)
#         self.num_atten = num_atten
#         self.proj_module = nn.ModuleList(projection_module)
#         self.atten = nn.ModuleList([nn.MultiheadAttention(query_dim,num_heads=num_head,dropout=dropout,
#                     batch_first=batch_first) for i in range(num_atten)])
#         classify = []
#         classify.append(nn.Flatten())
#         classify.append(nn.Linear(self.num_branch*query_dim,num_class))
#         classify.append(nn.Sigmoid())
#         self.classify = nn.Sequential(*classify)

#     def forward(self,x, metadata):
#         self.features = []
#         for i in range(self.num_branch):
#             feature = self.proj_module[i](x[i], metadata)
#             self.features.append(feature)
#         self.features = torch.cat(self.features,dim = 1)
#         for i in range(self.num_atten):
#             self.features = self.atten[i](self.features,self.features,self.features)[0]
#         return self.classify(self.features)
        
# class Second_edition(nn.Module):
#     def __init__(self, projection_module, num_class = 5, embed_dims = 128, num_gate_blocks = 10):
#         super().__init__()
#         self.num_gate_blocks = num_gate_blocks-1 # this is not a mistake, the first gated_fusion is different
#         self.num_branch = len(projection_module)
#         self.proj_module = nn.ModuleList(projection_module)
#         self.gated_fusion = []
#         for i in range(num_gate_blocks):
#             if i == num_gate_blocks-1:
#                 self.gated_fusion.append(Gated_fusion(self.num_branch, embed_dims=embed_dims, will_classify=True))
#             else:
#                 self.gated_fusion.append(Gated_fusion(self.num_branch, embed_dims=embed_dims))
#         self.gated_fusion = nn.Sequential(*self.gated_fusion)
#         self.gated = Gated_fusion(self.num_branch, embed_dims=embed_dims, first = True)
#         classify = []
#         classify.append(nn.Flatten())
#         # classify.append(nn.Dropout(0.2))
#         classify.append(nn.Linear(embed_dims,num_class))
#         self.classify = nn.Sequential(*classify)

#     def forward(self,x, metadata):
#         self.features = []
#         for i in range(self.num_branch):
#             feature = self.proj_module[i](x[i], metadata)
#             self.features.append(feature)

#         out_gate = self.gated(self.features)
#         out_gate = self.gated_fusion(out_gate)
#         return self.classify(out_gate)

class Merge(nn.Module):
    def __init__(self,num_modal):
        super().__init__()
        # self.weights = nn.ParameterList(nn.Parameter(torch.rand(1)) for i in range(num_modal))
        self.weights = nn.ParameterList()
        self.weights.append(nn.Parameter(torch.Tensor([12])))
        self.weights.append(nn.Parameter(torch.Tensor([10])))
        self.weights.append(nn.Parameter(torch.Tensor([6])))
        self.weights.append(nn.Parameter(torch.Tensor([1])))
        self.weights.append(nn.Parameter(torch.Tensor([2])))

        # self.weights.append(nn.Parameter(torch.Tensor([1])))
        # self.weights.append(nn.Parameter(torch.Tensor([1])))
        # self.weights.append(nn.Parameter(torch.Tensor([1])))
        # self.weights.append(nn.Parameter(torch.Tensor([1])))
        self.num_modal = num_modal
    def forward(self,x):
        out = torch.zeros_like(x[0])
        sum = torch.zeros_like(self.weights[0])
        for i in range(self.num_modal):
            sum += self.weights[i]
            out += x[i]*self.weights[i]
        return out/sum
    
