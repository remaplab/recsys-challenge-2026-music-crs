import torch
from llama_cpp import llama_print_system_info

def sanity_check():
    print("Hello from SLURM!")

    # Check PyTorch and CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} CUDA device(s):")
        
        for i in range(num_gpus):
            device_name = torch.cuda.get_device_name(i)
            # Calculate total memory in Gigabytes
            total_memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  Device {i}: {device_name} (VRAM: {total_memory_gb:.2f} GB)")
    else:
        print("No CUDA-compatible GPU found. Please check your SLURM configuration and GPU availability.")
        exit(1)

    # Print system info from llama_cpp
    print("\n--- System Information ---")
    print(llama_print_system_info().decode('utf-8'))