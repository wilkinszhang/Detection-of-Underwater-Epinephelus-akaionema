import torch
import torch.nn as nn
import torch.nn.functional as F

class MIPAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(MIPAM, self).__init__()
        # Channel-level information collection
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        # Spatial-level information collection
        self.spatial_conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_bn = nn.BatchNorm2d(1)
        
        # Adaptive feature fusion
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        
        # Channel-level attention
        avg_out = self.channel_avg_pool(x).view(b, c)
        channel_att = self.channel_fc(avg_out).view(b, c, 1, 1)
        channel_att = self.sigmoid(channel_att)
        
        # Spatial-level attention
        spatial_att = self.spatial_conv(x)
        spatial_att = self.spatial_bn(spatial_att)
        spatial_att = self.sigmoid(spatial_att)
        
        # Combine attentions
        out = x * channel_att * spatial_att
        return out

# Testing the MIPAM module
if __name__ == '__main__':
    x = torch.randn(1, 64, 20, 20)
    mipam = MIPAM(in_channels=64)
    y = mipam(x)
    print(y.size())
