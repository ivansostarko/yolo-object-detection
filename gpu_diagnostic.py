"""
GPU Diagnostic Script for YOLO/PyTorch
Checks if your GPU is properly configured and working
"""

import sys
import subprocess
import platform

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_system_info():
    """Check basic system information"""
    print_section("SYSTEM INFORMATION")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Architecture: {platform.machine()}")

def check_nvidia_smi():
    """Check if nvidia-smi is available"""
    print_section("NVIDIA GPU DETECTION (nvidia-smi)")
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print("❌ nvidia-smi command failed")
            print("This might mean NVIDIA drivers are not installed")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        print("NVIDIA drivers may not be installed")
        print("Download from: https://www.nvidia.com/download/index.aspx")
        return False
    except Exception as e:
        print(f"❌ Error running nvidia-smi: {e}")
        return False

def check_pytorch():
    """Check PyTorch and CUDA availability"""
    print_section("PYTORCH & CUDA CHECK")
    
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"\nCUDA available: {'✓ YES' if cuda_available else '❌ NO'}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}:")
                print(f"  Name: {torch.cuda.get_device_name(i)}")
                print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
                
                # Memory info
                total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  Total Memory: {total_memory:.2f} GB")
                
            return True
        else:
            print("\n⚠️  PyTorch is installed but CUDA is not available")
            print("Possible reasons:")
            print("  1. No NVIDIA GPU in system")
            print("  2. NVIDIA drivers not installed")
            print("  3. PyTorch installed without CUDA support")
            print("\nTo install PyTorch with CUDA support:")
            print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        print("Install with: pip install torch torchvision")
        return False
    except Exception as e:
        print(f"❌ Error checking PyTorch: {e}")
        return False

def check_ultralytics():
    """Check Ultralytics YOLO installation"""
    print_section("ULTRALYTICS YOLO CHECK")
    
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"✓ Ultralytics installed: {ultralytics.__version__}")
        
        # Try to load a model
        try:
            print("\nLoading YOLOv8n model...")
            model = YOLO('yolov8n.pt')
            device = model.device
            print(f"✓ Model loaded successfully")
            print(f"Default device: {device}")
            
            # Try to move to CUDA if available
            import torch
            if torch.cuda.is_available():
                print("\nAttempting to move model to GPU...")
                model.to('cuda')
                print(f"✓ Model moved to GPU: {model.device}")
                return True
            else:
                print("⚠️  CUDA not available, model will use CPU")
                return False
                
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            return False
            
    except ImportError:
        print("❌ Ultralytics not installed")
        print("Install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ Error checking Ultralytics: {e}")
        return False

def run_performance_test():
    """Run GPU performance benchmark"""
    print_section("GPU PERFORMANCE BENCHMARK")
    
    try:
        import torch
        import time
        
        if not torch.cuda.is_available():
            print("⚠️  Skipping GPU benchmark - CUDA not available")
            print("Running CPU benchmark instead...\n")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
            print(f"Testing on: {torch.cuda.get_device_name(0)}\n")
        
        # Test 1: Matrix Multiplication
        print("Test 1: Large Matrix Multiplication (10000x10000)")
        size = 10000
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        start = time.time()
        c = torch.matmul(a, b)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
        
        gpu_time = end - start
        print(f"  Time: {gpu_time:.4f} seconds")
        
        if device.type == 'cuda':
            print(f"  GPU Memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            
            # Performance interpretation
            if gpu_time < 1.0:
                print("  ✓ Excellent GPU performance!")
            elif gpu_time < 3.0:
                print("  ✓ Good GPU performance")
            else:
                print("  ⚠️  Slower than expected - check GPU drivers")
        else:
            print(f"  (CPU baseline - GPU should be 10-50x faster)")
        
        # Test 2: YOLO Inference
        print("\nTest 2: YOLO Inference Speed")
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            
            if device.type == 'cuda':
                model.to('cuda')
            
            # Warmup
            print("  Warming up...")
            for _ in range(3):
                results = model('https://ultralytics.com/images/bus.jpg', verbose=False)
            
            # Actual test
            print("  Running inference...")
            start = time.time()
            num_runs = 10
            for _ in range(num_runs):
                results = model('https://ultralytics.com/images/bus.jpg', verbose=False)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.time()
            avg_time = (end - start) / num_runs
            fps = 1 / avg_time
            
            print(f"  Average inference time: {avg_time*1000:.2f} ms")
            print(f"  FPS: {fps:.1f}")
            
            if device.type == 'cuda':
                if avg_time < 0.01:
                    print("  ✓ Excellent GPU inference speed!")
                elif avg_time < 0.05:
                    print("  ✓ Good GPU inference speed")
                else:
                    print("  ⚠️  Slower than expected")
            
        except Exception as e:
            print(f"  ⚠️  Could not run YOLO test: {e}")
        
        return True
        
    except ImportError:
        print("❌ PyTorch not available for benchmarking")
        return False
    except Exception as e:
        print(f"❌ Error during performance test: {e}")
        return False

def main():
    """Run all diagnostic checks"""
    print("\n" + "🔍 GPU DIAGNOSTIC TOOL 🔍".center(60))
    print("Checking your GPU setup for YOLO/PyTorch\n")
    
    # Run all checks
    check_system_info()
    nvidia_ok = check_nvidia_smi()
    pytorch_ok = check_pytorch()
    yolo_ok = check_ultralytics()
    
    if pytorch_ok:
        run_performance_test()
    
    # Summary
    print_section("SUMMARY")
    
    if nvidia_ok and pytorch_ok and yolo_ok:
        print("✓ SUCCESS! Your GPU is properly configured for YOLO")
        print("\nYou can now run YOLO models with GPU acceleration!")
        print("\nExample usage:")
        print("  from ultralytics import YOLO")
        print("  model = YOLO('yolov8n.pt')")
        print("  results = model('image.jpg')")
    else:
        print("⚠️  ISSUES DETECTED\n")
        
        if not nvidia_ok:
            print("❌ NVIDIA drivers not detected")
            print("   → Install from: https://www.nvidia.com/download/index.aspx\n")
        
        if not pytorch_ok:
            print("❌ PyTorch CUDA not available")
            print("   → Install PyTorch with CUDA:")
            print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n")
        
        if not yolo_ok:
            print("❌ YOLO not properly installed")
            print("   → Install Ultralytics:")
            print("     pip install ultralytics\n")
    
    print("\n" + "="*60)
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDiagnostic cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
