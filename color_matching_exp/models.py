# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# -----
# The following software may be included in this product: pytorch-unet,
# https://github.com/usuyama/pytorch-unet
# License: https://github.com/usuyama/pytorch-unet/blob/master/LICENSE
# MIT License
#
# Copyright (c) 2018 Naoto Usuyama
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
from layers import \
    SetDropout,ReluSets,Conv2dDeepSym,Conv2dSiamese,\
    SetMaxPool2d,SetUpsample,DeepSetsBlock,DeepSetsBlockSiamese,MLPBlock,Conv2dSAittala,Conv2dSridhar


def double_conv(in_channels, out_channels,model_type,p_drop,use_max=0):
    if model_type=='deepsets' or model_type == 'fullySiamese':
        return nn.Sequential(
            Conv2dSiamese(in_channels, out_channels, 3,padding=1),
            ReluSets(),
            SetDropout(p_drop=p_drop),
            Conv2dSiamese(out_channels, out_channels, 3,padding=1),
            ReluSets(),
            SetDropout(p_drop=p_drop)
        )
    elif model_type == 'Sridhar':
        return nn.Sequential(
            Conv2dSridhar(in_channels, out_channels, 3, padding=1),
            ReluSets(),
            SetDropout(p_drop=p_drop),
            Conv2dSridhar(out_channels, out_channels, 3, padding=1),
            ReluSets(),
            SetDropout(p_drop=p_drop)
        )
    elif model_type == 'Aittala':
        return nn.Sequential(
            Conv2dSAittala(in_channels, out_channels, 3, padding=1),
            ReluSets(),
            SetDropout(p_drop=p_drop),
            Conv2dSAittala(out_channels, out_channels, 3, padding=1),
            ReluSets(),
            SetDropout(p_drop=p_drop)
        )
    elif model_type == 'DeepSymmetricNet':
        return nn.Sequential(
            Conv2dDeepSym(in_channels, out_channels, 3,use_max=use_max,padding=1),
            ReluSets(),
            SetDropout(p_drop=p_drop),
            Conv2dDeepSym(out_channels, out_channels, 3,use_max=use_max,padding=1),
            ReluSets(),
            SetDropout(p_drop=p_drop)
        )

def last_conv(model_type,c,use_max=0):
    if model_type == 'deepsets' or model_type == 'fullySiamese':
        conv_last = Conv2dSiamese(c[2] + c[1], 4, 1,padding=0)
    elif model_type == 'Sridhar':
        conv_last = Conv2dSiamese(c[2] + c[1], 4, 1, padding=0)
    elif model_type == 'Aittala':
        conv_last = Conv2dSiamese(c[2] + c[1], 4, 1, padding=0)
    elif model_type=='DeepSymmetricNet':
        conv_last = Conv2dDeepSym(c[2] + c[1], 4, 1,use_max=use_max,padding=0)
    return conv_last


def get_feature_processing_block(model_type,channels):
    if model_type == 'deepsets' or model_type == 'DeepSymmetricNet'\
            or model_type == 'Sridhar' or model_type == 'Aittala':
        return DeepSetsBlock(channels=(channels[4],channels[4],channels[4]))
    elif model_type == 'fullySiamese':
        return DeepSetsBlockSiamese(channels=(channels[4],3*channels[4],channels[4]))
    elif model_type == 'DeepSymmetricNetMLP':
        return MLPBlock(channels=(channels[4],channels[4],channels[4]))



class UNet(nn.Module):

    def __init__(self,model_type,p_drop,use_max):
        super().__init__()
        self.model_type = model_type

        if self.model_type =='deepsets' or self.model_type == 'Sridhar':
            # c = (4, 8, 16, 32, 64, 128, 200, 300) # initially: (4, 64, 128, 200, 300)
            c = (4, 64, 128, 200, 300, 128, 200, 300) # initially: (4, 64, 128, 200, 300)
            ####################### WAS MESSING WITH THIS LINE TO TRY TO MAKE IT WORK ######################
        elif model_type == 'Aittala':
            c = (4, 150, 200, 300, 320)
        elif model_type == 'DeepSymmetricNet':
            c = (4, 50, 100, 150, 200)

        self.dconv_down1 = double_conv(c[0], c[1],self.model_type,p_drop,use_max=use_max)
        self.dconv_down2 = double_conv(c[1], c[2],self.model_type,p_drop,use_max=use_max)
        self.dconv_down3 = double_conv(c[2], c[3],self.model_type,p_drop,use_max=use_max)
        self.dconv_down4 = double_conv(c[3], c[4],self.model_type,p_drop,use_max=use_max)
        # Below added by me
        self.dconv_down5 = double_conv(c[4], c[5],self.model_type,p_drop,use_max=use_max)
        self.dconv_down6 = double_conv(c[5], c[6],self.model_type,p_drop,use_max=use_max)
        self.dconv_down7 = double_conv(c[6], c[7],self.model_type,p_drop,use_max=use_max)
        #

        self.maxpool2 = SetMaxPool2d(stride=2)
        self.maxpool8 = SetMaxPool2d(stride=8)

        self.feature_processing_block = get_feature_processing_block(model_type=model_type,channels=c)

        self.upsample2 = SetUpsample(scale_factor=2)
        self.upsample8 = SetUpsample(scale_factor=8)

        self.dconv_up3 = double_conv(c[4] + c[4], c[3],self.model_type,p_drop,use_max=use_max)
        self.dconv_up2 = double_conv(c[3] + c[3], c[2],self.model_type,p_drop,use_max=use_max)
        self.dconv_up1 = double_conv(c[2] + c[2], c[2],self.model_type,p_drop,use_max=use_max)
        # Below added by me
        self.dconv_up6 = double_conv(c[7] + c[7], c[6],self.model_type,p_drop,use_max=use_max)
        self.dconv_up5 = double_conv(c[6] + c[6], c[5],self.model_type,p_drop,use_max=use_max)
        self.dconv_up4 = double_conv(c[5] + c[5], c[4],self.model_type,p_drop,use_max=use_max)
        #

        self.last_conv = last_conv(self.model_type,c,use_max=use_max)


    def forward(self, x):
        
        conv1 = self.dconv_down1(x)
        x = self.maxpool2(conv1) # prev 32, now 256
        print("1", x.shape)

        conv2 = self.dconv_down2(x)
        x = self.maxpool2(conv2) # prev 16, now 128
        print("2", x.shape)

        conv3 = self.dconv_down3(x)
        x = self.maxpool2(conv3) # prev 8, now 64
        print("3", x.shape)

        conv4 = self.dconv_down4(x)
        x = self.maxpool2(conv4) # prev 1 (END), now 32
        print("4", x.shape)

        conv5 = self.dconv_down5(x)
        x = self.maxpool2(conv5) # 16
        print("5", x.shape) 
        
        conv6 = self.dconv_down6(x)
        x = self.maxpool2(conv6) # 8
        print("6", x.shape)
        
        conv7 = self.dconv_down7(x)
        x = self.maxpool8(conv7) # 1
        print("7", x.shape)
        
        # deep sets block
        print("BEFORE SQUEEZE", x.size()) # should be (1, 5, x, 1, 1)
        x = x.squeeze(dim=4)
        x = x.squeeze(dim=3)
        print("AFTER SQUEEZE", x.size())
        x = self.feature_processing_block(x)
        x = x.unsqueeze(dim=3).unsqueeze(dim=4)
        x = self.upsample8(x) # 8
        x = torch.cat([x, conv7], dim=2)
        print("8", x.shape)

        x = self.dconv_up6(x)
        x = self.upsample2(x) # 16
        x = torch.cat([x, conv6], dim=2)
        print("9", x.shape)

        x = self.dconv_up5(x)
        x = self.upsample2(x) # 32
        x = torch.cat([x, conv5], dim=2)
        print("10", x.shape)

        x = self.dconv_up4(x)
        x = self.upsample2(x) # 64
        x = torch.cat([x, conv4], dim=2)
        print("11", x.shape)

        x = self.dconv_up3(x)
        x = self.upsample2(x) # prev 16, now 128
        x = torch.cat([x, conv3], dim=2)
        print("12", x.shape)

        x = self.dconv_up2(x)
        x = self.upsample2(x) # prev 32, now 256
        x = torch.cat([x, conv2], dim=2)
        print("13", x.shape)

        x = self.dconv_up1(x)
        x = self.upsample2(x) # prev 64, now 512
        x = torch.cat([x, conv1], dim=2)
        print("14", x.shape)

        out = self.last_conv(x)

        return out

