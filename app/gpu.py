import torch


def get_gpu_info():
    """Return information about the first available CUDA GPU."""

    if not torch.cuda.is_available():
        return {
            "available": False,
            "name": None,
            "vram_gb": 0.0,
            "compute_capability": None,
            "profile": "cpu",
        }

    device_index = 0
    properties = torch.cuda.get_device_properties(device_index)

    vram_gb = properties.total_memory / (1024 ** 3)

    name = properties.name

    if "RTX 2060" in name:
        profile = "rtx2060"

    elif "RTX 4070 Ti" in name:
        profile = "rtx4070ti"

    else:
        profile = "unknown_gpu"

    return {
        "available": True,
        "name": name,
        "vram_gb": round(vram_gb, 2),
        "compute_capability": (
            properties.major,
            properties.minor,
        ),
        "profile": profile,
    }


def print_gpu_info():
    info = get_gpu_info()

    print()
    print("=" * 60)
    print("GPU CONFIGURATION")
    print("=" * 60)

    if not info["available"]:
        print("CUDA:         unavailable")
        print("Profile:      cpu")
        return info

    print(f"GPU:          {info['name']}")
    print(f"VRAM:         {info['vram_gb']:.2f} GB")
    print(
        f"Compute:      "
        f"{info['compute_capability'][0]}."
        f"{info['compute_capability'][1]}"
    )
    print(f"Profile:      {info['profile']}")

    print("=" * 60)

    return info
