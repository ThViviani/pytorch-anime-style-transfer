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

def define_discriminator(input_nc, ndf, discrim_type="patch", n_layers=3, norm_layer=nn.BatchNorm2d):
    """Create a discriminator
    Parameters:
        input_nc (int)        -- the number of channels in input images
        ndf (int)             -- the number of filters in the first conv layer
        discrim_type (str)    -- the arhitecture's name: patch
        n_layers (int)        -- the number of conv layers in the discriminator
        norm_layer (torch.nn) -- normalization layer 
    Returns a discriminator
    """

    net = None
    if discrim_type == "patch":
        net = PatchDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    
    net.apply(weights_init)
    return net

def define_generator(input_nc, ndf, generator_type="unet", norm_layer=nn.BatchNorm2d):
    """Create a generator
    Parameters:
        input_nc (int)        -- the number of channels in input images
        ndf (int)             -- the number of filters in the first conv layer
        generator_type (str)  -- the arhitecture's name: unet
        norm_layer (torch.nn) -- normalization layer 
    
    Returns a generator
    """

    net = None
    if generator_type == "unet":
        net = UnetGenerator(input_nc, ndf, norm_layer=norm_layer)
    elif generator_type == "residual":
        net = ResidualGenerator()
    
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
                nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
                nn.LeakyReLU(0.2),
            )
        ]

        in_channel = features[0]
        for out_channel in features[1:]:
            layers += [DiscriminatorCNNBlock(in_channel, out_channel, norm_layer, stride=1 if out_channel == features[-1] else 2)]
            in_channel = out_channel
        
        layers += [DiscriminatorCNNBlock(in_channel, 1, norm_layer, stride=1)]
        # layers += [nn.Sigmoid()]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, input):
        return self.model(input)

class GeneratorCNNBlock(nn.Module):
    """Defines a generator CNN block"""

    def __init__(self, in_channels, out_channels, down=True, activation="relu", use_dropout=False, norm_layer=nn.BatchNorm2d, **kwargs):
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
        
        kernel_size    = kwargs.get("kernel_size", 4)
        stride         = kwargs.get("stride", 2)
        padding        = kwargs.get("padding", 1)
        output_padding = kwargs.get("output_padding", 0)
        
        use_bias = True if norm_layer == nn.InstanceNorm2d else False

        activation_module = None
        if activation == "relu":
            activation_module = nn.ReLU(inplace=True)
        elif activation == "identity":
            activation_module = nn.Identity()
        else:
            activation_module = nn.LeakyReLU(0.2)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size, 
                stride, 
                padding, 
                padding_mode="reflect", 
                bias=use_bias
            ) if down 
            else nn.ConvTranspose2d(
                in_channels, 
                out_channels, 
                kernel_size, 
                stride, 
                padding, 
                bias=use_bias, 
                output_padding=output_padding
            ),
            norm_layer(out_channels),
            activation_module, 
        )
        
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        return self.dropout(self.conv(x)) if self.use_dropout else self.conv(x)
    
class UnetGenerator(nn.Module):
    """Defines a Unet generator"""

    def __init__(self, in_channels=3, features=64, norm_layer=nn.BatchNorm2d):
        """Construct Unet generator
        Parameters:
            in_channels (int)     -- the number of channels in the input image
            features (int)        -- the number of filters in the first conv layer
            norm_layer (torch.nn) -- normalization layer
        """
        # TODO: Rewrite creating gen blocks in a loop. This implementation is designed for 256x256 resolution.

        super().__init__()
        
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        ) # 128 x 128
        
        self.down1 = GeneratorCNNBlock(features, features * 2, kernel_size=4, stride=2, padding=1, down=True, activation="leaky", use_dropout=False, norm_layer=norm_layer) # 64 x 64
        self.down2 = GeneratorCNNBlock(features * 2, features * 4, kernel_size=4, stride=2, padding=1, down=True, activation="leaky", use_dropout=False, norm_layer=norm_layer) # 32 x 32
        self.down3 = GeneratorCNNBlock(features * 4, features * 8, kernel_size=4, stride=2, padding=1, down=True, activation="leaky", use_dropout=False, norm_layer=norm_layer) # 16 x 16
        self.down4 = GeneratorCNNBlock(features * 8, features * 8, kernel_size=4, stride=2, padding=1, down=True, activation="leaky", use_dropout=False, norm_layer=norm_layer) # 8 x 8
        self.down5 = GeneratorCNNBlock(features * 8, features * 8, kernel_size=4, stride=2, padding=1, down=True, activation="leaky", use_dropout=False, norm_layer=norm_layer) # 4 x 4
        self.down6 = GeneratorCNNBlock(features * 8, features * 8, kernel_size=4, stride=2, padding=1, down=True, activation="leaky", use_dropout=False, norm_layer=norm_layer) # 2 x 2
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"), # 1 x 1
            nn.ReLU()
        )
        
        self.up1 = GeneratorCNNBlock(features * 8, features * 8, kernel_size=4, stride=2, padding=1, down=False, activation="relu", use_dropout=True, norm_layer=norm_layer)
        self.up2 = GeneratorCNNBlock(features * 8 * 2, features * 8, kernel_size=4, stride=2, padding=1, down=False, activation="relu", use_dropout=True, norm_layer=norm_layer)
        self.up3 = GeneratorCNNBlock(features * 8 * 2, features * 8, kernel_size=4, stride=2, padding=1, down=False, activation="relu", use_dropout=True, norm_layer=norm_layer)
        self.up4 = GeneratorCNNBlock(features * 8 * 2, features * 8, kernel_size=4, stride=2, padding=1, down=False, activation="relu", use_dropout=False, norm_layer=norm_layer)
        self.up5 = GeneratorCNNBlock(features * 8 * 2, features * 4, kernel_size=4, stride=2, padding=1, down=False, activation="relu", use_dropout=False, norm_layer=norm_layer)
        self.up6 = GeneratorCNNBlock(features * 4 * 2, features * 2, kernel_size=4, stride=2, padding=1, down=False, activation="relu", use_dropout=False, norm_layer=norm_layer)
        self.up7 = GeneratorCNNBlock(features * 2 * 2, features, kernel_size=4, stride=2, padding=1, down=False, activation="relu", use_dropout=False, norm_layer=norm_layer)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        x0 = self.initial_down(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x_ = self.bottleneck(x6)
        x_ = self.up1(x_)
        x_ = self.up2(torch.concat([x6, x_], dim=1))
        x_ = self.up3(torch.concat([x5, x_], dim=1))
        x_ = self.up4(torch.concat([x4, x_], dim=1))
        x_ = self.up5(torch.concat([x3, x_], dim=1))
        x_ = self.up6(torch.concat([x2, x_], dim=1))
        x_ = self.up7(torch.concat([x1, x_], dim=1))
        x_ = self.final_up(torch.concat([x0, x_], dim=1))
        return x_ 

class ResidualBlock(nn.Module):
    """Defines a original Residual block"""

    def __init__(self, channels, norm_layer=nn.InstanceNorm2d):
        """Construct a Residual block
        Parameters:
            channels (int)            -- the number of channels in the input 
            norm_layer (torch.nn)     -- normalization layer   
        """
        
        super().__init__()

        self.block = nn.Sequential(
            GeneratorCNNBlock(channels, channels, activation="relu", kernel_size=3, padding=1, stride=1, norm_layer=norm_layer),
            GeneratorCNNBlock(channels, channels, activation="identity", kernel_size=3, padding=1, stride=1, norm_layer=norm_layer)
        )

        # self.relu = nn.ReLU() see ResnetBlock in the https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L316 
        

    def forward(self, x):
        return x + self.block(x)
    
class ResidualGenerator(nn.Module):
    def __init__(self, img_channels=3, ndf=64, num_residuals=9, norm_layer=nn.InstanceNorm2d):
        
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, ndf, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            norm_layer(ndf),
            nn.ReLU(inplace=True)
        )

        self.down_block = nn.Sequential(
            GeneratorCNNBlock(ndf, ndf * 2, kernel_size=3, stride=2, padding=1, norm_layer=norm_layer),
            GeneratorCNNBlock(ndf * 2, ndf * 4, kernel_size=3, stride=2, padding=1, norm_layer=norm_layer)
        )

        self.residuals_blocks = nn.Sequential(
            *[ResidualBlock(ndf * 4) for _ in range(num_residuals)]
        )

        self.up_block = nn.Sequential(
            GeneratorCNNBlock(
                ndf * 4, 
                ndf * 2, 
                kernel_size=3, 
                stride=2, 
                padding=1, 
                down=False, 
                norm_layer=norm_layer, 
                output_padding=1
            ),
            GeneratorCNNBlock(
                ndf * 2, 
                ndf, 
                kernel_size=3, 
                stride=2, 
                padding=1, 
                down=False, 
                norm_layer=norm_layer, 
                output_padding=1
            )
        )

        self.final_block = nn.Conv2d(
            ndf, 
            img_channels, 
            kernel_size=7, 
            stride=1, 
            padding=3, 
            padding_mode="reflect"
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down_block(x)
        x = self.residuals_blocks(x)
        x = self.up_block(x)
        x = self.final_block(x)
        return x