import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d(inplanes, planes, kernelsize, stride = 1):
    return nn.Conv2d(inplanes, planes, kernel_size=kernelsize, padding = (kernelsize-1)//2, stride=stride)

#create seblock
class SEblock(nn.Module):
    def __init__(self, inplanes, planes, kernel = 3, reduction=16, downsample=False, dropout = 0.2):
        super().__init__()
        self.downsample = downsample
        if downsample:    
            self.downsample = conv2d(inplanes, planes, kernel, stride=2)
            self.bn_down = nn.BatchNorm2d(planes)
            self.conv1 = conv2d(inplanes, planes, kernel, stride=2)
        else:
            self.conv1 = conv2d(inplanes, planes, kernel, stride=1)
            self.residual = conv2d(inplanes, planes, kernel, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
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
            stem_kernel = 15
            maxpool_stride = 2
            maxpool_kernel = 7
            se_kernel = 7
        else:    
            stem.append(nn.MaxPool2d(4,stride = 4))
            stem_kernel = 15
            maxpool_stride = 2
            maxpool_kernel = 7
            se_kernel = 7
        
        stem.append(conv2d(in_channel, 64, stem_kernel, stride=1))
        stem.append(nn.MaxPool2d(maxpool_kernel,stride = maxpool_stride, padding = (maxpool_kernel-1)//2))
        stem.append(nn.BatchNorm2d(64))
        stem.append(nn.Dropout(0.2))
        stem.append(nn.ReLU(inplace = True))
        self.stem = nn.Sequential(*stem)

        stage1 = [SEblock(64,64,se_kernel,downsample=False),SEblock(64,64,se_kernel)]
        stage2 = [SEblock(64,128,se_kernel,downsample=True),SEblock(128,128,se_kernel)]
        stage3 = [SEblock(128,256,se_kernel,downsample=True),SEblock(256,256,se_kernel)]
        stage4 = [SEblock(256,512,se_kernel,downsample=True),SEblock(512,512,se_kernel)]
        self.se_stage = stage1+stage2+stage3+stage4
        self.se_stage = nn.Sequential(*
            self.se_stage
        )
    def forward(self,x):
        x = self.stem(x)
        return self.se_stage(x)

#projetion_module
class Projection_Module(nn.Module):
    def __init__(self, in_channel = 3, type = "stft", d_meta = 64):
        super().__init__()
        assert type == "stft" or type == "wavelet" or type == "wignerville", "Wrong module type, it should be wavelet, stft or wignerville"
        #se-resnet 
        self.d_meta = d_meta
        self.se = SE_ResNet(in_channel, type = type)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        
        #project onto embedding space
        self.proj = nn.Linear(512+d_meta,128)

        #metadata
        meta_layers = []
        meta_layers.append(nn.Conv1d(32,48,kernel_size = 1))
        meta_layers.append(nn.ReLU(inplace = True))
        meta_layers.append(nn.Conv1d(48,d_meta,kernel_size = 1))
        self.meta_layers = nn.Sequential(*meta_layers)

    def forward(self,x,metadata):
        
        x = self.se(x)
        x = self.pool(x)
        # from (-1,512,1,1) to (-1,1,512)
        x = torch.reshape(x, (-1,1,512))
        # metadata
        metadata = self.meta_layers(metadata)
        metadata = torch.reshape(metadata,(-1,1,self.d_meta))

        feature = torch.cat((x,metadata),dim = 2)
       
        return self.proj(feature)
class Pseudo_classifier(nn.Module):
    def __init__(self,num_class = 5):
        super().__init__()

        p_classify = []
        p_classify.append(nn.Flatten())
        p_classify.append(nn.Linear(128,num_class))
        self.p_classify = nn.Sequential(*p_classify)
        
    def forward(self,spec):
        return self.p_classify(spec)

class Branch(nn.Module):
    def __init__(self, in_channel = 3, type = "stft", classifier = None, num_class = 5):
        super().__init__()

        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = Pseudo_classifier(num_class)
        self.pm = Projection_Module(in_channel, type)
    def forward(self,x,metadata):
        x = self.pm(x,metadata)
        return self.classifier(x)


class Second_edition(nn.Module):
    def __init__(self, projection_module, num_class = 5):
        super().__init__()
        
        self.num_branch = len(projection_module)
        self.proj_module = nn.ModuleList(projection_module)
        
        trans = nn.TransformerEncoderLayer(d_model = 128, nhead = 2, batch_first = True, dim_feedforward = 128)
        self.trans = nn.TransformerEncoder(trans, num_layers = 1)

        classify = []
        classify.append(nn.Flatten())
        classify.append(nn.Linear(self.num_branch*128,num_class))
        self.classify = nn.Sequential(*classify)

    def forward(self,x, metadata):
        features = []
        for i in range(self.num_branch):
            feature = self.proj_module[i](x, metadata)
            features.append(feature)
        features = torch.cat(features,dim = 1)
        enhanced_feature = self.trans(features)      
        return self.classify(enhanced_feature)