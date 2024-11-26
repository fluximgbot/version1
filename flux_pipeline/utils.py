import torch
import numpy as np
from PIL import Image
import os

def save_image(image_tensor, path, normalize=True):
    """
    Saves a tensor image to a specified file path.

    Args:
        image_tensor (torch.Tensor): The image tensor to save.
        path (str): The file path where the image will be saved.
        normalize (bool): Whether to normalize the tensor to the range [0, 255].
                          Default is True.

    Returns:
        None
    """
    if normalize:
        # Normalize the image from range [-1, 1] to [0, 255]
        image_tensor = image_tensor.clamp(-1, 1)
        image_tensor = (image_tensor + 1) / 2 * 255
    image_tensor = image_tensor.squeeze().permute(1, 2, 0).byte().cpu().numpy()
    img = Image.fromarray(image_tensor)
    img.save(path)

def load_image(image_path, size=None):
    """
    Loads an image from a file and optionally resizes it.

    Args:
        image_path (str): The path to the image file.
        size (tuple, optional): The target size to resize the image to (width, height).

    Returns:
        torch.Tensor: The loaded and optionally resized image as a tensor.
    """
    img = Image.open(image_path).convert("RGB")
    if size:
        img = img.resize(size, Image.ANTIALIAS)
    img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor * 2 - 1  # Normalize to range [-1, 1]
    return img_tensor.unsqueeze(0)  # Add batch dimension

def normalize_tensor(tensor, mean, std):
    """
    Normalizes a tensor image using the provided mean and std values.

    Args:
        tensor (torch.Tensor): The image tensor to normalize.
        mean (tuple): The mean values for each channel.
        std (tuple): The standard deviation values for each channel.

    Returns:
        torch.Tensor: The normalized image tensor.
    """
    normalize = torch.nn.functional.normalize(tensor, mean=mean, std=std)
    return normalize

def denormalize_tensor(tensor, mean, std):
    """
    Denormalizes a tensor image using the provided mean and std values.

    Args:
        tensor (torch.Tensor): The image tensor to denormalize.
        mean (tuple): The mean values for each channel.
        std (tuple): The standard deviation values for each channel.

    Returns:
        torch.Tensor: The denormalized image tensor.
    """
    tensor = tensor * std[None, :, None, None] + mean[None, :, None, None]
    return tensor

def ensure_dir_exists(path):
    """
    Ensures that the directory for a given path exists.

    Args:
        path (str): The path to check or create.

    Returns:
        None
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

def set_seed(seed):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to set.

    Returns:
        None
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

def closest_number(x, base=16):
    """
    Rounds a number to the closest multiple of a given base.

    Args:
        x (int): The number to round.
        base (int): The base to round to (default is 16).

    Returns:
        int: The rounded number.
    """
    return base * round(x / base)

def load_model_from_checkpoint(model, checkpoint_path):
    """
    Loads a model's weights from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load weights into.
        checkpoint_path (str): The path to the checkpoint file.

    Returns:
        torch.nn.Module: The model with weights loaded.
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model weights loaded from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
    return model
