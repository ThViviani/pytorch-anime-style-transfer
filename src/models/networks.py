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
            out_channels (int) -- the number of channels in the ouput when applyed this block
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

class GeneratorCNNBlock(nn.Module):
    """Defines a generator CNN block"""

    def __init__(self, in_channels, out_channels, down=True, activation="relu", use_dropout=False, norm_layer=nn.BatchNorm2d):
        """Construct a generator CNN block
        Parameters:
            in_channels (int)            -- the number of channels in the input 
            out_channels (int)           -- the number of channels in the ouput when applyed this block
            down (bool)                  -- the type of conv layer. if down=True Conv2d(stride=2) else ConvTranspose2d
            activation (str)             -- the name of activation function: relu | another names
            use_dropout (bool)           -- if the flag is True then will add nn.Dropout after conv layer
            norm_layer (torch.nn)        -- normalization layer   
        """

        super(GeneratorCNNBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, padding_mode="reflect", bias=False) if down 
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.2), 
        )
        
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        return self.dropout(self.conv(x)) if self.use_dropout else self.conv(x)
    
