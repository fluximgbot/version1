import torch
import numpy as np

class EmptyLatentImage:
    """
    A class that handles the creation and manipulation of latent images.
    Latent images are the internal representations of the image data in the diffusion process.
    """

    @staticmethod
    def generate(width, height):
        """
        Generates an empty latent image of the given dimensions.

        Args:
            width (int): The width of the latent image.
            height (int): The height of the latent image.

        Returns:
            torch.Tensor: A tensor representing the latent image, with random noise values.
        """
        # Generate a random latent image with values in the range [-1, 1]
        latent_image = torch.randn(1, 3, height, width) * 0.5
        return latent_image

    @staticmethod
    def resize(latent_image, new_width, new_height):
        """
        Resize an existing latent image to the new dimensions.

        Args:
            latent_image (torch.Tensor): The existing latent image tensor.
            new_width (int): The desired width of the new latent image.
            new_height (int): The desired height of the new latent image.

        Returns:
            torch.Tensor: The resized latent image.
        """
        # Resize the latent image using PyTorch's functional API (torch.nn.functional.interpolate)
        latent_image_resized = torch.nn.functional.interpolate(
            latent_image, size=(new_height, new_width), mode='bilinear', align_corners=False
        )
        return latent_image_resized

    @staticmethod
    def denoise(latent_image, denoise_factor=0.1):
        """
        Applies denoising to a latent image. In this case, it's just a simple example of noise reduction.
        This could be replaced with more complex denoising algorithms.

        Args:
            latent_image (torch.Tensor): The latent image tensor to denoise.
            denoise_factor (float): The factor by which to reduce the noise.

        Returns:
            torch.Tensor: The denoised latent image.
        """
        # Apply a simple denoising by blending the image with its smoothed version
        latent_image_denoised = latent_image * (1 - denoise_factor) + torch.randn_like(latent_image) * denoise_factor
        return latent_image_denoised

    @staticmethod
    def to_image(latent_image):
        """
        Converts the latent image tensor to a normal image format (e.g., [0, 255] range for RGB).

        Args:
            latent_image (torch.Tensor): The latent image tensor.

        Returns:
            np.ndarray: The converted image as a numpy array.
        """
        # Convert from latent space (range [-1, 1]) to normal image space (range [0, 255])
        image = (latent_image.squeeze().clamp(-1, 1) + 1) / 2 * 255
        image = image.permute(1, 2, 0).byte().cpu().numpy()  # Convert to HWC format (height, width, channels)
        return image

