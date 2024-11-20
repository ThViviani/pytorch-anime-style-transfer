import torch


def denorm_tensor(tensor, mean, std, device):
    """Denormalize a input tensor
    Parameters:
        tensor (torch.Tensor)       -- Normalized tensor with shape [B, C, H, W].
        mean (list or torch.Tensor) -- The mean value for each channel in the input tensor.
        std (list or torch.Tensor)  -- The std value for each channel in the input tensor.
        device (torch.device).      -- The device of the tensor.
    Return denormalized tensor
    """

    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    if tensor.is_cuda:
        mean = mean.to(device)
        std = std.to(device)

    return tensor * std + mean