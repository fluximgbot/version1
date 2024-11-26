import torch
import torch.nn.functional as F

class VAEDecode:
    """
    A class that decodes latent representations back into image space using a VAE model.
    The VAE (Variational Autoencoder) decoder transforms the latent vector into a reconstructed image.
    """

    @staticmethod
    def decode(vae_model, latent_image):
        """
        Decodes a latent image back to the image space using the VAE model.

        Args:
            vae_model (torch.nn.Module): The VAE model to be used for decoding.
            latent_image (torch.Tensor): The latent image tensor to decode.

        Returns:
            torch.Tensor: The decoded image in the original image space.
        """
        # Pass the latent image through the VAE decoder
        decoded_image = vae_model.decode(latent_image)
        return decoded_image

    @staticmethod
    def reconstruct(vae_model, latent_image):
        """
        Reconstructs the image from the latent vector using the VAE model.

        Args:
            vae_model (torch.nn.Module): The VAE model to be used for reconstruction.
            latent_image (torch.Tensor): The latent vector (latent space representation).

        Returns:
            torch.Tensor: The reconstructed image tensor.
        """
        # Reconstruct the image from the latent representation
        return VAEDecode.decode(vae_model, latent_image)

    @staticmethod
    def to_image(decoded_image):
        """
        Converts the decoded image tensor to a standard image format (e.g., [0, 255] range for RGB).

        Args:
            decoded_image (torch.Tensor): The decoded image tensor.

        Returns:
            np.ndarray: The decoded image as a numpy array.
        """
        # Convert the decoded image from range [-1, 1] to [0, 255] for visualization
        decoded_image = decoded_image.squeeze().clamp(-1, 1)
        image = (decoded_image + 1) / 2 * 255
        image = image.permute(1, 2, 0).byte().cpu().numpy()  # Convert to HWC format (height, width, channels)
        return image
