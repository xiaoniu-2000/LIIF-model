'''
Author: xiaoniu
Date: 2026-01-06 16:46:34
LastEditors: xiaoniu
LastEditTime: 2026-01-06 16:46:40
Description: model structure of LIIF
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class ResidualBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, res_scale=1):
        super(ResidualBlock,self).__init__()
        layers = []
        for i in range(2):
            layers.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: layers.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                layers.append(nn.ReLU(True))
        self.body = nn.Sequential(*layers)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

#区别于源代码，此EDSR没有self.sub_mean = MeanShift(rgb,range),self.add_mean = MeanShift(rgb,range,sign=1); 
# MeanShift本质上是通过一个nn.Conv2d实现归一化和反归一化
class EDSR(nn.Module):
    def __init__(self, n_resblocks,in_channels,out_channels,res_scale=1,conv=conv):
        super(EDSR, self).__init__()
        kernel_size = 3
        #define the head module
        head = [conv(in_channels, out_channels, kernel_size)]
        #define the body module
        body = [ResidualBlock(conv, out_channels, kernel_size,res_scale=res_scale) for _ in range(n_resblocks)]
        body.append(conv(out_channels, out_channels, kernel_size))
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.out_dim = out_channels

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        return res

class MLP(nn.Module):
    def __init__(self, in_dim,out_dim,hidden_list):
        super().__init__()
        layers = []
        last_dim = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.ReLU())
            last_dim = hidden
        layers.append(nn.Linear(last_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

class LIIF(nn.Module):
    def __init__(self,in_channel, out_channel, hidden_list, local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.in_channel = in_channel
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.encoder = EDSR(n_resblocks=6,in_channels=in_channel,out_channels=64,res_scale=1)
        imnet_in_dim = self.encoder.out_dim
        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2
        if self.cell_decode:
            imnet_in_dim += 2
        self.imnet = MLP(imnet_in_dim, out_channel, hidden_list)
    
    def generate_feat(self,input):
        self.feat = self.encoder(input)
        return self.feat
    
    def query(self, coord, cell=None):
        feat = self.feat
        if self.feat_unfold:
            feat = F.unfold(feat, kernel_size=3, padding=1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        
        if self.local_ensemble:
            vx_list = [-1, 1]
            vy_list = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_list, vy_list, eps_shift = [0], [0], 0
        
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        feat_coord = make_coord(feat.shape[-2:],flatten=False) \
            .permute(2,0,1).unsqueeze(0) \
            .expand(feat.shape[0], 2, *feat.shape[-2:]) 
        
        preds = []
        areas = []
        for vx in vx_list:
            for vy in vy_list:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret
    
    def forward(self, input, coord, cell):
        self.generate_feat(input)
        return self.query(coord, cell)

if __name__ == "__main__":
    hidden_list = [256,256,256,256]
    net = LIIF(in_channel=1, out_channel=1, hidden_list=hidden_list)
    x = torch.randn(16, 1, 64, 64)
    #h,w = output_resolution[0,1]
    scale_factor = 4
    h,w = [x.shape[-2] * scale_factor, x.shape[-1] * scale_factor]
    coord = make_coord((h, w))
    cell = torch.ones_like(coord)
    cell[:,0] *= 2 /h
    cell[:,1] *= 2 /w
    coord = coord.unsqueeze(0).expand(x.shape[0], -1, -1)
    cell = cell.unsqueeze(0).expand(x.shape[0], -1, -1)
    y = net(x, coord, cell)
    y = y.view(h, w, -1).permute(2, 0, 1)
    print(y.shape)


        