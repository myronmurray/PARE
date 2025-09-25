import torch
from typing import Union


def resolve_device(device: Union[str, torch.device]) -> torch.device:
    """Resolve device string to torch.device object."""
    if isinstance(device, torch.device):
        return device

    if device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    return torch.device(device)


def device_to_string(device: torch.device) -> str:
    """Convert torch.device to string representation."""
    return str(device)


def device_to_accelerator(device: torch.device) -> str:
    """Convert torch.device to PyTorch Lightning accelerator string."""
    if device.type == 'cuda':
        return 'gpu'
    else:
        return 'cpu'
