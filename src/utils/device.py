import torch

def resolve_device(device: str | None = None) -> torch.device:
    if device and device.lower() in {"cpu", "cuda", "mps"}:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
