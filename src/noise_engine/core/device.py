import torch
from contextlib import contextmanager
from typing import Optional

# Default device configuration
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_device(device_str: str = DEFAULT_DEVICE) -> torch.device:
    """
    Get and validate a PyTorch device.

    Args:
        device_str: Device string ('cpu', 'cuda', 'cuda:0', etc.)

    Returns:
        Validated torch.device object

    Raises:
        ValueError: If device is not available
    """
    if isinstance(device_str, torch.device):
        return device_str

    device = torch.device(device_str)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            f"CUDA not available. Using 'cpu' instead.\n"
            f"Available devices: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}"
        )

    return device


# Global default device (for backward compatibility)
device = get_device(DEFAULT_DEVICE)


@contextmanager
def use_device(device_str: str):
    """
    Context manager for temporarily switching devices.

    Args:
        device_str: Target device string

    Example:
        with use_device("cpu"):
            tensor = torch.randn(100).to("cpu")
        # Back to default device after context
    """
    global device
    original_device = device
    try:
        device = get_device(device_str)
        yield device
    finally:
        device = original_device
