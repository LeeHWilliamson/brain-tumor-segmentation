import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
'''
summary of functions
DoubleConv3D: extract features at current scale
MaxPool3D: Downsample (reduce resolution, increase receptive field)
ConvTranspose3D: Upsample to higher resolution
Skip Connections: Preserve Fine Details
Final 1x1x1 conv: map to output classes
'''
class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            #3 x 3convolution on way down
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False), #kernel size = 3, stride = 1, padding = 1, padding of 1 ensures spatial dimensions are unaltered
            # nn.InstanceNorm3d(out_channels, affine=True), 
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            #3 x 3 convolution on way up
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias = False),
            # nn.InstanceNorm3d(out_channels, affine=True),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__( #all modules we need for forward step
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],  #will need to change outchannels for 3d segmentation
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList() #important because we want to do model evaluation
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2) #used in between conv layers in forward method, shrinks volume in 1/2 in each dim

        #Down part of UNET (encoder path)
        for feature in features:
            self.downs.append(DoubleConv3D(in_channels, feature)) #map 3 to 64, increase channels, reduce spatial dimensions, we abstract from pixel values to relationships to categories
            in_channels = feature
        
        #Up part (decoder path)
        #COULD use transpose convolutions may be cheaper and better
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(
                    feature*2, feature, kernel_size=2, stride = 2, #512 * 2, we are resizing (expanding back to original dimensions) image here
                )
            )
            self.ups.append(DoubleConv3D(feature*2, feature)) #up, 2 convs, up, 2 convs... We are appending to feature map we built during pooling

        self.bottleneck = DoubleConv3D(features[-1], features[-1]*2) #access last feature, max channels, min resolution
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1) #map channels to output
    def forward(self, x):
        skip_connections = [] #store outputs from down layers
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x) #bottom part of U
        skip_connections = skip_connections[::-1] #reverse

        for index in range(0, len(self.ups), 2): #up, double conv each step
            x = self.ups[index](x)
            skip_connection = skip_connections[index//2]

            if x.shape != skip_connection.shape:
                #x = TF.resize(x, size=skip_connection.shape[2:])
                x = F.interpolate(x, size = skip_connection.shape[2:], mode = "trilinear", align_corners=False) #resize to match shape before we do torch.cat
            concat_skip = torch.cat((skip_connection, x), dim=1) #reuse the high-res (pre-pooling) data, preserve details that would otherwise be lost due to pooling
            x = self.ups[index+1](concat_skip)
        return self.final_conv(x)

def test():
    x = torch.randn((3, 4, 16, 160, 160))
    model = UNET(in_channels=4, out_channels=4)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    #assert preds.shape == x.shape #make sure the final output truly matches the original input shape so we can compare voxel by voxel

if __name__ == "__main__":
    test()