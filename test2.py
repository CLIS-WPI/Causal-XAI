#!/usr/bin/env python3
import re
import subprocess
import drjit as dr
import os
import sys

def get_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode()
        match = re.search(r"release\s+([\d.]+)", output)
        return match.group(1) if match else "Unknown CUDA version"
    except Exception as e:
        return f"Error checking CUDA version: {e}"

def get_optix_version():
    try:
        # First check if OptiX is in a common location
        optix_paths = ["/usr/local/optix", "/opt/optix"]
        
        for path in optix_paths:
            if os.path.exists(path):
                # Try to find version information in include files
                include_dir = os.path.join(path, "include")
                if os.path.exists(include_dir):
                    for filename in os.listdir(include_dir):
                        if filename.startswith("optix") and filename.endswith(".h"):
                            file_path = os.path.join(include_dir, filename)
                            with open(file_path, 'r') as f:
                                content = f.read()
                                version_match = re.search(r'#define\s+OPTIX_VERSION\s+(\d+)', content)
                                if version_match:
                                    version_num = int(version_match.group(1))
                                    major = version_num // 10000
                                    minor = (version_num % 10000) // 100
                                    patch = version_num % 100
                                    return f"{major}.{minor}.{patch}"
                return "OptiX installed (version unknown)"
        
        # Try to find OptiX through system paths
        try:
            output = subprocess.check_output(["find", "/usr", "-name", "optix*.h"], stderr=subprocess.DEVNULL).decode()
            if output.strip():
                return "OptiX installed (version unknown)"
        except:
            pass
            
        return "OptiX not found"
    except Exception as e:
        return f"Error checking OptiX: {e}"

def get_tensorrt_version():
    try:
        # Check if TensorRT is installed in Python
        try:
            import tensorrt
            return getattr(tensorrt, "__version__", "TensorRT installed (version unknown)")
        except ImportError:
            pass
            
        # Check if TensorRT is installed via apt
        try:
            output = subprocess.check_output(["dpkg", "-l", "*tensorrt*"]).decode()
            for line in output.splitlines():
                if "libnvinfer" in line:
                    parts = line.split()
                    if len(parts) > 2:
                        return parts[2]  # Extract version
            
            # Check in library paths
            lib_paths = ["/usr/lib/x86_64-linux-gnu", "/usr/local/lib"]
            for path in lib_paths:
                if os.path.exists(path):
                    for file in os.listdir(path):
                        if file.startswith("libnvinfer.so."):
                            # Extract version from file name
                            version_match = re.search(r'libnvinfer\.so\.(\d+)\.(\d+)\.(\d+)', file)
                            if version_match:
                                return f"{version_match.group(1)}.{version_match.group(2)}.{version_match.group(3)}"
                            return "TensorRT installed (version unknown)"
        except:
            pass
            
        # Check CUDA paths for TensorRT
        try:
            cuda_paths = ["/usr/local/cuda", "/usr/cuda"]
            for cuda_path in cuda_paths:
                if os.path.exists(os.path.join(cuda_path, "include/NvInfer.h")):
                    return "TensorRT installed (version unknown)"
        except:
            pass
        
        return "TensorRT not found"
    except Exception as e:
        return f"Error checking TensorRT: {e}"

def get_cudnn_version_tf():
    try:
        import tensorflow as tf
        info = tf.sysconfig.get_build_info()
        return str(info.get('cudnn_version', "Not found"))
    except Exception as e:
        return f"Error reading cuDNN version from TensorFlow: {e}"

def check_gpu_tf():
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        return len(gpus) > 0
    except Exception:
        return False

def get_lib_tinfo_version():
    try:
        output = subprocess.check_output(["dpkg", "-s", "libtinfo6"]).decode()
        for line in output.splitlines():
            if line.startswith("Version: "):
                return line.split("Version: ")[1].strip()
        return "libtinfo6 installed, but version not found"
    except Exception as e:
        return f"Error checking libtinfo6 version: {e}"

def get_nvidia_driver_version():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).decode()
        return output.strip().splitlines()[0] if output else "No driver info found"
    except Exception as e:
        return f"Error checking NVIDIA driver version: {e}"

def get_tf_build_info():
    try:
        import tensorflow as tf
        build_info = tf.sysconfig.get_build_info()
        return build_info
    except Exception:
        return {}

def main():
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        tf_file = tf.__file__
    except ImportError:
        tf_version = "TensorFlow not installed"
        tf_file = "N/A"

    try:
        import sionna
        sionna_version = sionna.__version__
    except ImportError:
        sionna_version = "Sionna not installed"

    try:
        import mitsuba
        mitsuba_version = getattr(mitsuba, "__version__", "Mitsuba version not found")
    except ImportError:
        mitsuba_version = "Mitsuba not installed"

    cuda_version = get_cuda_version()
    cudnn_version = get_cudnn_version_tf()
    gpu_active = check_gpu_tf()
    libtinfo_version = get_lib_tinfo_version()
    nvidia_driver = get_nvidia_driver_version()
    optix_version = get_optix_version()
    tensorrt_version = get_tensorrt_version()
    
    # Get detailed TF build info
    tf_build_info = get_tf_build_info()

    print("============ Environment Info ============")
    print(f"TensorFlow version:        {tf_version}")
    print(f"TensorFlow path:           {tf_file}")
    print(f"CUDA version:              {cuda_version}")
    print(f"cuDNN version:             {cudnn_version}")
    print(f"TensorRT version:          {tensorrt_version}")
    print(f"GPU active (via TF):       {gpu_active}")
    print(f"Sionna version:            {sionna_version}")
    print(f"Mitsuba version:           {mitsuba_version}")
    print(f"libtinfo6 version:         {libtinfo_version}")
    print(f"NVIDIA driver version:     {nvidia_driver}")
    print(f"OptiX version:             {optix_version}")
    print(f"Dr.Jit CUDA support:       {dr.has_backend(dr.JitBackend.CUDA)}")
    print(f"Dr.Jit LLVM support:       {dr.has_backend(dr.JitBackend.LLVM)}")
    print("==========================================")
    
    # Print detailed TensorFlow build info
    if tf_build_info:
        print("\nTensorFlow Build Information:")
        for key, value in tf_build_info.items():
            print(f"  {key}: {value}")
        print("==========================================")

if __name__ == "__main__":
    main()