import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .optim import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *

from .backbones.Res2Net_v1b import res2net50_v1b_26w_4s

class Legacy(nn.Module):
    # res2net based encoder decoder
    def __init__(self, depth=64, pretrained=True, base_size=[576, 384]):
        super(Legacy, self).__init__()
        self.backbone = res2net50_v1b_26w_4s(pretrained=pretrained)

        self.context1 = PAA_e(64, depth)
        self.context2 = PAA_e(256, depth)
        self.context3 = PAA_e(512, depth)
        self.context4 = PAA_e(1024, depth)
        self.context5 = PAA_e(2048, depth)

        self.decoder = PAA_d(depth)

        self.attention  = ASCA(depth    , depth, base_size=base_size, stage=0, lmap_in=True)
        self.attention1 = ASCA(depth * 2, depth, base_size=base_size, stage=1, lmap_in=True)
        self.attention2 = ASCA(depth * 2, depth, base_size=base_size, stage=2              )

        # self.loss_fn = lambda x, y: weighted_tversky_bce_loss(x, y, alpha=0.2, beta=0.8, gamma=2)
        self.loss_fn = bce_loss
        self.pyramidal_consistency_loss_fn = nn.L1Loss()

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
        self.pyr = Pyr(7, 1)

    def cuda(self):
        self.pyr = self.pyr.cuda()
        self = super(Legacy, self).cuda()
        return self
    
    def forward(self, sample):
        if type(sample) == dict:
            x = sample['image']
        else:
            x = sample
            
        B, _, H, W = x.shape # (b, 32H, 32W, 3)
        # base_size = x.shape[-2:]  # (b, 32H, 32W, 3)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x1 = self.backbone.relu(x)  # (b, 16H, 16W, 64)
        x2 = self.backbone.maxpool(x1)

        x2 = self.backbone.layer1(x2)  # (b, 8H, 8W, 256)
        x3 = self.backbone.layer2(x2)  # (b, 4H, 4W, 512)
        x4 = self.backbone.layer3(x3)  # (b, 2H, 2W, 1024)
        x5 = self.backbone.layer4(x4)  # (b, H, W, 2048)

        x1 = self.context1(x1)
        x2 = self.context2(x2)
        x3 = self.context3(x3)
        x4 = self.context4(x4)
        x5 = self.context5(x5)

        f3, d3 = self.decoder(x5, x4, x3) # 2h 2w

        f2, p2 = self.attention2(torch.cat([x2, self.ret(f3, x2)], dim=1), d3.detach()) 
        d2 = self.pyr.rec(d3.detach(), p2) # 4h 4w

        f1, p1 = self.attention1(torch.cat([x1, self.ret(f2, x1)], dim=1), d2.detach(), p2.detach())
        d1 = self.pyr.rec(d2.detach(), p1) # 8h 8w

        _, p = self.attention(self.res(f1, (H, W)), d1.detach(), p1.detach())
        d = self.pyr.rec(d1.detach(), p) # 32H X 32W

        if type(sample) == dict and 'gt' in sample.keys() and sample['gt'] is not None:
            y = sample['gt']
            py1, y1 = self.pyr.dec(y)
            py2, y2 = self.pyr.dec(y1)
            py3, y3 = self.pyr.dec(y2)
            py4, y4 = self.pyr.dec(y3)
            py5, y5 = self.pyr.dec(y4)

            dd3 = self.pyr.down(d2)
            dd2 = self.pyr.down(d1)
            dd1 = self.pyr.down(d)

            d3 = self.des(d3, (H, W))
            d2 = self.des(d2, (H, W))
            d1 = self.des(d1, (H, W))

            dd3 = self.des(dd3, (H, W))
            dd2 = self.des(dd2, (H, W))
            dd1 = self.des(dd1, (H, W))

            ploss1 = self.pyramidal_consistency_loss_fn(d3, dd3.detach()) * 0.0001
            ploss2 = self.pyramidal_consistency_loss_fn(d2, dd2.detach()) * 0.0001
            ploss3 = self.pyramidal_consistency_loss_fn(d1, dd1.detach()) * 0.0001

            y3 = self.des(y3, (H, W))
            y2 = self.des(y2, (H, W))
            y1 = self.des(y1, (H, W))

            closs =  self.loss_fn(d3, y3)
            closs += self.loss_fn(d2, y2)
            closs += self.loss_fn(d1, y1)
            closs += self.loss_fn(d,  y)
            
            loss = ploss1 + ploss2 + ploss3 + closs

        else:
            loss = 0

        if type(sample) == dict:
            return {'pred': d, 
                    'loss': loss, 
                    'gaussian': [d3, d2, d1, d], 
                    'laplacian': [p2, p1, p]}
        
        else:
            return d