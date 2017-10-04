import torch
from torch import nn
from torch.nn import functional as F


class Conv3BN(nn.Module):
    """A module which applies the following actions:
        - convolution with 3x3 kernel;
        - batch normalization (if enabled);
        - ELU.
    Attributes:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        bn: A boolean indicating if Batch Normalization is enabled or not.
    """

    def __init__(self, in_ch: int, out_ch: int, bn=True):
        super(Conv3BN, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class UNetEncoder(nn.Module):
    """UNetEncoder module. Applies
       - MaxPool2d to reduce the input sice twice
       - twice Conv3BN, first with different size of channels and then with the same numbers of channels    
    Attributes:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
    """    
    def __init__(self, in_ch: int, out_ch: int):
        super(UNetEncoder, self).__init__()
        self.encode = nn.Sequential(nn.MaxPool2d(2, 2),
                                    Conv3BN(in_ch, out_ch),
                                    Conv3BN(out_ch, out_ch),                                    
        )
    def forward(self, x):
        x = self.encode(x)
        return x



class UNetDecoder(nn.Module):
    """UNetDecoder module. Applies
       - Upsample with scale_factor = 2
       - concatanation of miror slice with upsampled image along rows as a result the number of chanal increases
       - twice Conv3BN    
    Attributes:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super(UNetDecoder, self).__init__()

        self.decode = nn.Sequential(Conv3BN(in_ch, out_ch),
                                    Conv3BN(out_ch, out_ch),
                                    Conv3BN(out_ch, out_ch),
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    def forward(self, x_copy, x_down):
        #N, C, H, W = x_copy.size()
        x_up = self.upsample(x_down) #F.upsample(x_down, size=(H, W), mode='bilinear')
        x_up = torch.cat([x_copy, x_up], 1)
        x_new = self.decode(x_up)
        return x_new
    

class UNet(nn.Module):
    """A UNet module. Applies
        - once input_layer
        - depth times of
            - UNetEncoder
            - UNetDecoder
        - activation (sigmoid)
        The number of output channels of each UNetEncoder/UNetDecoder is twice larger/less than the previous
        number of input channels;
    Attributes:
        num_classes: Number of output channels.
        input_channels: Number of input image channels.
        filter_base: Number of out channels of the first UNet layer and base size for the each next.
        depth: number of UNet layers UNetEncoder/UNetDecoder on the way down/up.
        filter_base and depthe are connected as filter_base*2**depth = 1024 - the number of channels on the bottom layer
    """

    def __init__(self,
                 num_classes: int=1,
                 input_channels: int=3,
                 filters_base: int=8,
                 depth: int=7):
        super(UNet, self).__init__()

        #filter sizes for down, center and up
        down_filter_sizes = [filters_base * 2**i for i in range(depth+1)] #  32, 64, 128, 256, 512, 1024
        up_filter_sizes = list(reversed(down_filter_sizes))

        # input layer
        self.input_layer = nn.Sequential(Conv3BN(input_channels, filters_base),
                                         Conv3BN(filters_base, filters_base),
        )
        # Going down:        

        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        # depth filters to go down
        for i in range(1, depth+1):
            self.down.append(UNetEncoder(down_filter_sizes[i-1], down_filter_sizes[i]))            
        
        #depth filters to go up
        for i in range(1, depth+1): # the number of channel increseas after concatenation
            self.up.append(UNetDecoder(up_filter_sizes[i-1]+up_filter_sizes[i], up_filter_sizes[i]))

        # Final layer and activation:
        self.output = nn.Conv2d(up_filter_sizes[-1], out_channels=num_classes, kernel_size=1)
        
        self.activation = F.sigmoid
        
    def forward(self, x):
        
        x = self.input_layer(x) 
        xs = [x] # collect slices from down side to copy them to up side
        #go down                                        
        for module in self.down:
            x = module(x)
            xs.append(x)

        xs.reverse()    

        #go up
        x = xs[0]
        for xc, module in zip(xs[1:], self.up):
            x = module(xc, x)

        x = self.output(x)    
        x = self.activation(x)
        return x

