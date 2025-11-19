import torch
import sys

print("="*60)
print("PYTORCH & CUDA INSTALLATION CHECK")
print("="*60)

# PyTorch version
print(f"\nPyTorch Version: {torch.__version__}")

# CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\nCUDA Available: {'YES ✓' if cuda_available else 'NO ✗'}")

if cuda_available:
    # CUDA version
    print(f"CUDA Version: {torch.version.cuda}")
    
    # cuDNN version
    if torch.backends.cudnn.is_available():
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    
    # Number of GPUs
    gpu_count = torch.cuda.device_count()
    print(f"\nNumber of GPUs: {gpu_count}")
    
    # GPU details
    print("\nGPU Details:")
    print("-" * 60)
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  - Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  - Compute Capability: {props.major}.{props.minor}")
        print(f"  - Multi-Processors: {props.multi_processor_count}")
        
        # Current memory usage
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  - Memory Allocated: {allocated:.2f} GB")
        print(f"  - Memory Reserved: {reserved:.2f} GB")
    
    # Test tensor creation on GPU
    print("\n" + "="*60)
    print("GPU FUNCTIONALITY TEST")
    print("="*60)
    try:
        # Create a test tensor on GPU
        test_tensor = torch.randn(1000, 1000).cuda()
        result = test_tensor @ test_tensor.T
        print("✓ Successfully created and multiplied tensors on GPU")
        print(f"✓ Result shape: {result.shape}")
        print(f"✓ Result device: {result.device}")
        del test_tensor, result
        torch.cuda.empty_cache()
        print("✓ GPU memory cleared")
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
    

else:
    print("\n" + "="*60)
    print("WARNING: NO GPU DETECTED")
    print("="*60)
    print("\nTraining will use CPU, which is EXTREMELY slow.")
    print("\nTo enable GPU training:")
    print("1. Check if you have an NVIDIA GPU:")
    print("   - Run: nvidia-smi")
    print("\n2. Install CUDA-enabled PyTorch:")
    print("   - Visit: https://pytorch.org/get-started/locally/")
    print("   - For CUDA 11.8:")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("   - For CUDA 12.1:")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("\n3. Verify installation:")
    print("   - Run this script again")
    
    # Check if nvidia-smi is available
    print("\n" + "="*60)
    print("CHECKING NVIDIA DRIVERS")
    print("="*60)
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ NVIDIA driver is installed")
            print("\nNVIDIA-SMI Output:")
            print(result.stdout)
            print("\n⚠ You have NVIDIA GPU but PyTorch is not using it!")
            print("⚠ Reinstall PyTorch with CUDA support (see instructions above)")
        else:
            print("✗ nvidia-smi command failed")
            print("✗ Make sure NVIDIA drivers are installed")
    except FileNotFoundError:
        print("✗ nvidia-smi not found")
        print("✗ No NVIDIA GPU detected OR drivers not installed")
    except Exception as e:
        print(f"✗ Error checking NVIDIA drivers: {e}")

print("\n" + "="*60)
print("SYSTEM INFORMATION")
print("="*60)
print(f"Python Version: {sys.version}")
print(f"Platform: {sys.platform}")

print("\n" + "="*60 + "\n")
