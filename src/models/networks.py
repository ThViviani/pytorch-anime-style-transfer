import torch
import torch.nn as nn

from torch.nn import init


def weights_init(m):
    """Define the initialization function
    Parameters:
        m (torch.nn) -- the torch module like Conv or BatchNorm
    """

    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)

def define_discriminator(input_nc, ndf, discrim_type="patch", n_layers=3):
    """Create a discriminator
    Parameters:
        input_nc (int)      -- the number of channels in input images
        ndf (int)           -- the number of filters in the first conv layer
        discrim_type (str)  -- the arhitecture's name: patch
        n_layers (int)      -- the number of conv layers in the discriminator
    
    Returns a discriminator
    """

    net = None
    if discrim_type == "patch":
        net = PatchDiscriminator(input_nc, ndf, n_layers=3)
    
    net.apply(weights_init)
    return net

class DiscriminatorCNNBlock(nn.Module):
    """Defines a discriminator CNN block"""

    def __init__(self, in_channels, out_channels, norm_layer, stride=2):
        """Construct a convolutional block.
        Parameters:
            in_channels (int)  -- the number of channels in the input 
            out_channels (int) -- the number of channels int the ouput when applyed this block
            norm_layer         -- normalization layer 
            stride (int)       -- the stride of conv layer
        """

        super(DiscriminatorCNNBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, padding_mode="reflect", padding=1),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, x):
        return self.conv(x)
    
class PatchDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, in_channels=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d): # 256 x 256 -> 70x70 receptive field
        """Construct a PatchGAN discriminator
        
        Parameters:
            in_channel (int) -- the number of channels in input image
            ndf (int)        -- the number of filters in the first conv layer
            n_layers (int)   -- the number of conv layers in the discriminator
            norm_layer       -- normalization layer
        """
        
        super().__init__()

        features = [ndf * (2 ** i) for i in range(n_layers + 1)]
            
        layers = [
            nn.Sequential(
                nn.Conv2d(in_channels * 2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
                nn.LeakyReLU(0.2),
            )
        ]

        in_channel = features[0]
        for out_channel in features[1:]:
            layers += [DiscriminatorCNNBlock(in_channel, out_channel, norm_layer, stride=1 if out_channel == features[-1] else 2)]
            in_channel = out_channel
        
        layers += [DiscriminatorCNNBlock(in_channel, 1, norm_layer, stride=1)]
        layers += [nn.Sigmoid()]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, input):
        return self.model(input)