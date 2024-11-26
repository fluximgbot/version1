import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEEncoder(nn.Module):
    """
    Encoder part of the VAE: Maps input image to latent space (mean and variance).
    """
    def __init__(self, in_channels=3, latent_dim=256):
        super(VAEEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc_mean = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar


class VAEDecoder(nn.Module):
    """
    Decoder part of the VAE: Maps latent vector back to the image space.
    """
    def __init__(self, latent_dim=256, out_channels=3):
        super(VAEDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 128 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = z.view(z.size(0), 128, 8, 8)  # Reshape to feature map
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = torch.sigmoid(self.deconv3(z))  # Sigmoid to output in range [0, 1]
        return z


class VAE(nn.Module):
    """
    Variational Autoencoder model combining the encoder and decoder.
    """
    def __init__(self, in_channels=3, latent_dim=256, out_channels=3):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim)
        self.decoder = VAEDecoder(latent_dim, out_channels)

    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick: sample from N(0, 1) and scale by the learned mean and stddev.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, logvar


def vae_loss(reconstructed_x, x, mean, logvar):
    """
    Compute the VAE loss, combining reconstruction loss and KL divergence.
    """
    # Reconstruction loss (binary cross-entropy)
    BCE = F.binary_cross_entropy(reconstructed_x.view(-1, 3 * 256 * 256), x.view(-1, 3 * 256 * 256), reduction='sum')

    # KL divergence loss
    # D_KL(q(z|x) || p(z)) = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # where mu is the mean and sigma is the standard deviation
    # This penalizes large deviations of the learned distribution from the standard normal distribution.
    # Higher KL divergence leads to more regularized latent space.
    # Assuming p(z) ~ N(0, I), where I is the identity covariance matrix.
    KL_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    return BCE + KL_divergence


# Example: Creating a VAE instance
def get_vae_model(in_channels=3, latent_dim=256, out_channels=3):
    """
    Returns an instance of the VAE model.

    Args:
        in_channels (int): Number of input channels (default: 3 for RGB images).
        latent_dim (int): Dimensionality of the latent space.
        out_channels (int): Number of output channels.

    Returns:
        VAE: The VAE instance.
    """
    return VAE(in_channels, latent_dim, out_channels)
