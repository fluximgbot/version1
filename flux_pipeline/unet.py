import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    """
    A basic block used in the UNet architecture for downsampling and upsampling.
    """
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.batch_norm(self.conv1(x)))
        x = F.relu(self.batch_norm(self.conv2(x)))
        return x


class UNetModel(nn.Module):
    """
    A simplified UNet model for image-to-image tasks.
    """
    def __init__(self, in_channels=3, out_channels=3, feature_channels=[64, 128, 256, 512]):
        super(UNetModel, self).__init__()
        
        # Downsampling layers
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for feature in feature_channels:
            self.downs.append(UNetBlock(in_channels, feature))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature

        # Bottleneck
        self.bottleneck = UNetBlock(feature_channels[-1], feature_channels[-1] * 2)

        # Upsampling layers
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for feature in reversed(feature_channels):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.up_convs.append(UNetBlock(feature * 2, feature))

        # Final output layer
        self.final_conv = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downsampling
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse the skip connections for upsampling
        skip_connections = skip_connections[::-1]

        # Upsampling
        for up, up_conv, skip in zip(self.ups, self.up_convs, skip_connections):
            x = up(x)
            x = torch.cat((x, skip), dim=1)  # Concatenate along the channel dimension
            x = up_conv(x)

        # Final output layer
        return self.final_conv(x)


# Example: Creating a UNet instance
def get_unet_model(in_channels=3, out_channels=3):
    """
    Returns an instance of the UNet model.

    Args:
        in_channels (int): Number of input channels (default: 3 for RGB images).
        out_channels (int): Number of output channels.

    Returns:
        UNetModel: The UNet instance.
    """
    return UNetModel(in_channels, out_channels)
