import subprocess
import sys

import torch


def print_section(title: str):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def get_ffmpeg_version():
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True,
        )

        first_line = result.stdout.splitlines()[0]
        return first_line

    except Exception as exc:
        return f"ERROR: {exc}"


def main():
    print_section("DUBBING WORKFLOW - GPU TEST")

    print(f"Python:        {sys.version.split()[0]}")
    print(f"PyTorch:       {torch.__version__}")
    print(f"PyTorch CUDA:  {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print()
        print("GPU TEST: FAILED")
        print("PyTorch cannot access CUDA.")
        print()
        print("Check:")
        print("  1. NVIDIA driver")
        print("  2. Docker Desktop GPU support")
        print("  3. WSL2")
        print("  4. Docker Compose GPU configuration")
        sys.exit(1)

    device_count = torch.cuda.device_count()

    print(f"GPU count:     {device_count}")

    for index in range(device_count):
        props = torch.cuda.get_device_properties(index)

        vram_gb = props.total_memory / (1024 ** 3)

        print()
        print(f"GPU {index}:")
        print(f"  Name:        {props.name}")
        print(f"  VRAM:        {vram_gb:.2f} GB")
        print(f"  Compute:     {props.major}.{props.minor}")

    print_section("CUDA COMPUTE TEST")

    device = torch.device("cuda:0")

    a = torch.tensor(
        [1.0, 2.0, 3.0],
        device=device,
    )

    b = torch.tensor(
        [4.0, 5.0, 6.0],
        device=device,
    )

    c = a + b

    torch.cuda.synchronize()

    print(f"Tensor device: {c.device}")
    print(f"Result:        {c.tolist()}")

    expected = [5.0, 7.0, 9.0]

    if c.tolist() != expected:
        print()
        print("CUDA TEST: FAILED")
        sys.exit(1)

    print()
    print("CUDA TEST: PASS")

    print_section("FFMPEG TEST")

    print(get_ffmpeg_version())

    print()
    print("=" * 60)
    print("GPU RUNTIME TEST: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()