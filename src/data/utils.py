import torch


def denorm_tensor(tensor, mean, std):
    """Denormalize a input tensor
    Parameters:
        tensor (torch.Tensor)       -- Normalized tensor with shape [B, C, H, W].
        mean (list or torch.Tensor) -- The mean value for each channel in the input tensor.
        std (list or torch.Tensor)  -- The std value for each channel in the input tensor.
    Return denormalized tensor
    """

    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    
    return tensor * std + mean