import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), #kernel size = 3, stride = 1, padding = 1 (same convolution)
            nn.BatchNorm2d(out_channels), #batchNorm cancels any bias
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
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
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #used in between conv layers in forward method

        #Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature)) #map 3 to 64
            in_channels = feature
        
        #Up part
        #COULD use transpose convolutions may be cheaper and better
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride = 2, #512 * 2, we are resizing image here
                )
            )
            self.ups.append(DoubleConv(feature*2, feature)) #up, 2 convs, up, 2 convs...

        self.bottleneck = DoubleConv(features[-1], features[-1]*2) #access last feature
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x) #bottom part of U
        skip_connections = skip_connections[::-1]

        for index in range(0, len(self.ups), 2): #up, double conv each step
            x = self.ups[index](x)
            skip_connection = skip_connections[index//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[index+1](concat_skip)
        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()