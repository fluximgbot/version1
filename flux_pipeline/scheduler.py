import torch
import numpy as np

class BasicScheduler:
    """
    A simple scheduler that controls the noise levels throughout the generation process.
    This can be used for adjusting the noise scale in each step of the diffusion process.
    """
    
    @staticmethod
    def get_sigmas(unet, scheduler_name, steps, beta_start=0.1, beta_end=0.2):
        """
        Generates a set of noise scaling factors (sigmas) used for the denoising process
        in diffusion models.

        Args:
            unet (torch.nn.Module): The UNet model used in the diffusion process.
            scheduler_name (str): The name of the scheduler to be used ('simple', 'cosine', etc.).
            steps (int): The number of steps to generate the schedule for.
            beta_start (float): The starting value for the noise schedule.
            beta_end (float): The ending value for the noise schedule.

        Returns:
            torch.Tensor: The generated sigmas for the given noise schedule.
        """
        if scheduler_name == "simple":
            return BasicScheduler.simple_scheduler(steps, beta_start, beta_end)
        else:
            raise ValueError(f"Scheduler '{scheduler_name}' not recognized. Use 'simple'.")

    @staticmethod
    def simple_scheduler(steps, beta_start, beta_end):
        """
        Generates a simple linear noise schedule for the diffusion process.

        Args:
            steps (int): The number of steps for the schedule.
            beta_start (float): The starting value for the noise schedule.
            beta_end (float): The ending value for the noise schedule.

        Returns:
            torch.Tensor: The noise schedule (sigmas) over the steps.
        """
        # Create a linear schedule from beta_start to beta_end
        beta_values = np.linspace(beta_start, beta_end, steps)
        sigmas = torch.tensor(beta_values, dtype=torch.float32)
        return sigmas

    @staticmethod
    def cosine_scheduler(steps, beta_start, beta_end):
        """
        Generates a cosine noise schedule for the diffusion process.

        Args:
            steps (int): The number of steps for the schedule.
            beta_start (float): The starting value for the noise schedule.
            beta_end (float): The ending value for the noise schedule.

        Returns:
            torch.Tensor: The cosine noise schedule over the steps.
        """
        # Create a cosine schedule from beta_start to beta_end
        t = torch.linspace(0, 1, steps)
        sigmas = beta_start + 0.5 * (beta_end - beta_start) * (1 - torch.cos(np.pi * t))
        return sigmas

    @staticmethod
    def get_timesteps(sigmas, total_steps):
        """
        Adjust the timesteps for the noise schedule.

        Args:
            sigmas (torch.Tensor): The noise schedule.
            total_steps (int): The total number of steps for the diffusion process.

        Returns:
            torch.Tensor: The adjusted timesteps based on the noise schedule.
        """
        return torch.linspace(0, total_steps, len(sigmas))

