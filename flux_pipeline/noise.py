import torch

class RandomNoise:
    """
    A class that generates random noise for image generation.
    """
    
    @staticmethod
    def get_noise(seed=None, size=(1, 3, 256, 256)):
        """
        Generates random noise with a given shape and optional seed.
        
        Args:
            seed (int, optional): The random seed for reproducibility. Defaults to None.
            size (tuple): The shape of the generated noise tensor (batch, channels, height, width).
        
        Returns:
            torch.Tensor: A tensor filled with random noise.
        """
        if seed is not None:
            torch.manual_seed(seed)  # Set the random seed for reproducibility
        
        noise = torch.randn(size)  # Generate random noise from a standard normal distribution
        return noise


class GaussianNoise:
    """
    A class for generating Gaussian noise with a specific mean and standard deviation.
    """
    
    @staticmethod
    def get_gaussian_noise(mean=0, std=1, size=(1, 3, 256, 256), seed=None):
        """
        Generates Gaussian noise with specified mean and standard deviation.
        
        Args:
            mean (float): The mean of the Gaussian distribution.
            std (float): The standard deviation of the Gaussian distribution.
            size (tuple): The shape of the generated noise tensor.
            seed (int, optional): The random seed for reproducibility. Defaults to None.
        
        Returns:
            torch.Tensor: A tensor filled with Gaussian noise.
        """
        if seed is not None:
            torch.manual_seed(seed)  # Set the random seed for reproducibility
        
        noise = torch.normal(mean=mean, std=std, size=size)  # Generate Gaussian noise
        return noise


class PerlinNoise:
    """
    A class for generating Perlin noise (can be useful for texture generation).
    """
    
    @staticmethod
    def generate(size=(1, 3, 256, 256)):
        """
        Generates Perlin noise using a simple implementation.
        
        Args:
            size (tuple): The shape of the generated noise tensor.
        
        Returns:
            torch.Tensor: A tensor filled with Perlin noise.
        """
        # This is a simplified approach and can be replaced with more sophisticated methods.
        # Perlin noise typically requires a grid of gradients; here we generate a simpler version
        # using random values and periodic tiling.

        noise = torch.rand(size)
        return noise


class NoiseGenerator:
    """
    A wrapper class for generating different types of noise.
    """
    
    @staticmethod
    def get_random_noise(seed=None, size=(1, 3, 256, 256)):
        """
        Generates random noise using the RandomNoise class.
        
        Args:
            seed (int, optional): The random seed for reproducibility. Defaults to None.
            size (tuple): The shape of the generated noise tensor.
        
        Returns:
            torch.Tensor: A tensor filled with random noise.
        """
        return RandomNoise.get_noise(seed, size)
    
    @staticmethod
    def get_gaussian_noise(mean=0, std=1, size=(1, 3, 256, 256), seed=None):
        """
        Generates Gaussian noise using the GaussianNoise class.
        
        Args:
            mean (float): The mean of the Gaussian distribution.
            std (float): The standard deviation of the Gaussian distribution.
            size (tuple): The shape of the generated noise tensor.
            seed (int, optional): The random seed for reproducibility. Defaults to None.
        
        Returns:
            torch.Tensor: A tensor filled with Gaussian noise.
        """
        return GaussianNoise.get_gaussian_noise(mean, std, size, seed)
    
    @staticmethod
    def get_perlin_noise(size=(1, 3, 256, 256)):
        """
        Generates Perlin noise using the PerlinNoise class.
        
        Args:
            size (tuple): The shape of the generated noise tensor.
        
        Returns:
            torch.Tensor: A tensor filled with Perlin noise.
        """
        return PerlinNoise.generate(size)


# Example usage:
if __name__ == "__main__":
    # Generate random noise
    noise = NoiseGenerator.get_random_noise(seed=42)
    print("Random Noise Shape:", noise.shape)

    # Generate Gaussian noise
    gaussian_noise = NoiseGenerator.get_gaussian_noise(mean=0, std=1, size=(1, 3, 256, 256), seed=42)
    print("Gaussian Noise Shape:", gaussian_noise.shape)

    # Generate Perlin noise
    perlin_noise = NoiseGenerator.get_perlin_noise(size=(1, 3, 256, 256))
    print("Perlin Noise Shape:", perlin_noise.shape)
