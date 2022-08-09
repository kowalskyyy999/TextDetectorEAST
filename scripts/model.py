import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import math

class VGG16BN(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16BN, self).__init__()
        backbone = torchvision.models.vgg16_bn(pretrained=pretrained)
        self.features = backbone.features

    def forward(self, x):
        out = []
        for m in self.features:
            x = m(x)
            if isinstance(m, nn.MaxPool2d):
                out.append(x)

        return out[1:]

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.h2 = self.make_layers(1024, 128)
        self.h3 = self.make_layers(384, 64)
        self.h4 = self.make_layers(192, 32)
        
        self.conv = nn.Conv2d(32, 32, 3, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

    def forward(self, out):
        x = F.interpolate(out[3], size=out[2].size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x, out[2]), 1)
        x = self.h2(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat((x, out[1]), 1)
        x = self.h3(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat((x, out[0]), 1)
        x = self.h4(x)

        x = self.relu(self.bn(self.conv(x)))
        return x


    @staticmethod
    def make_layers(in_channels, out_channels):
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())

class EAST(nn.Module):
    def __init__(self, pretrained = False, scope = 1024, batch_norm=True):
        super(EAST, self).__init__()

        self.encoder = VGG16BN(pretrained=pretrained)
        self.decoder = Decoder()

        self.score_layer = self.score_map()
        self.rot_layer, self.box_layer = self.RBOX_geo()
        
        self.scope = scope

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)

        score = torch.sigmoid(self.score_layer(out))
        angle = (torch.sigmoid(self.rot_layer(out)) - 0.5) * math.pi
        loc = torch.sigmoid(self.box_layer(out)) * self.scope
        geo = torch.cat((loc, angle), 1)
        return score, geo


    @staticmethod
    def score_map():
        return nn.Conv2d(32, 1, 1)
    
    @staticmethod
    def RBOX_geo():
        return nn.Conv2d(32, 1, 1), nn.Conv2d(32, 4, 1)

    @staticmethod
    def QUAD_geo():
        return nn.Conv2d(32, 8, 1)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x = torch.rand(3, 1200, 1600)
    x = x.unsqueeze(0)
    x = x.to(device)

    model = EAST(batch_norm=True).to(device)
    score, geo = model(x)

    print("Geometric Shape:", geo.shape)
    print("Score Shape:", score.shape)



    # if device == 'cuda':   
    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_max_memory_allocated()