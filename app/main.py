from gpu import print_gpu_info


def main():
    print("Starting video dubbing workflow...")

    gpu = print_gpu_info()

    print()
    print(f"Selected GPU profile: {gpu['profile']}")


if __name__ == "__main__":
    main()
