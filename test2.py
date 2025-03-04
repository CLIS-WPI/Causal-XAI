#!/usr/bin/env python3
import re
import subprocess

def get_cuda_version():
    """Return the installed CUDA version via nvcc."""
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode()
        for line in output.split("\n"):
            if "Cuda compilation tools" in line:
                match = re.search(r"release\s+([\d.]+)", line)
                if match:
                    return match.group(1)
        return "Unknown CUDA version (nvcc not found or parsing failed)"
    except Exception as e:
        return f"Error checking CUDA version: {e}"

def get_cudnn_version_tf():
    """Return the cuDNN version as reported by TensorFlow build info."""
    try:
        import tensorflow as tf
        info = tf.sysconfig.get_build_info()
        # برخی نسخه‌ها در فیلد 'cudnn_version' یا 'cudnn_library_path' قرار دارد
        cudnn_ver = info.get('cudnn_version', None)
        if cudnn_ver is not None:
            return str(cudnn_ver)
        else:
            return "cuDNN version not found in TF build info"
    except Exception as e:
        return f"Error reading cuDNN version from TensorFlow: {e}"

def check_gpu_tf():
    """Check if a GPU is visible to TensorFlow."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        return len(gpus) > 0
    except Exception:
        return False

def get_lib_tinfo_version():
    """Check the installed version of libtinfo6 using dpkg -s."""
    try:
        output = subprocess.check_output(["dpkg", "-s", "libtinfo6"]).decode()
        for line in output.splitlines():
            if line.startswith("Version: "):
                return line.split("Version: ")[1].strip()
        return "libtinfo6 installed, but version not found in dpkg -s output"
    except Exception as e:
        return f"Error checking libtinfo6 version: {e}"

def get_nvidia_driver_version():
    """Return the NVIDIA driver version using nvidia-smi."""
    try:
        # در صورت وجود چند GPU، این دستور نسخه درایور را در چند خط برمی‌گرداند.
        output = subprocess.check_output(["nvidia-smi", 
                                          "--query-gpu=driver_version", 
                                          "--format=csv,noheader"]).decode()
        lines = output.strip().splitlines()
        if lines:
            # فرض می‌کنیم نسخه همه GPUها یکسان است، بنابراین اولین خط را برمی‌گردانیم.
            return lines[0]
        return "No driver info found (empty nvidia-smi output)"
    except Exception as e:
        return f"Error checking NVIDIA driver version: {e}"

def main():
    # TensorFlow version
    try:
        import tensorflow as tf
        tf_version = tf.__version__
    except ImportError:
        tf_version = "TensorFlow not installed"

    # Sionna version
    try:
        import sionna
        sionna_version = sionna.__version__
    except ImportError:
        sionna_version = "Sionna not installed"

    # Mitsuba version
    try:
        import mitsuba
        # برخی نسخه‌های Mitsuba ممکن است __version__ نداشته باشند؛ در این صورت مقدار پیش‌فرض را برمی‌گردانیم.
        mitsuba_version = getattr(mitsuba, "__version__", "Mitsuba version not found")
    except ImportError:
        mitsuba_version = "Mitsuba not installed"

    # CUDA و cuDNN
    cuda_version = get_cuda_version()
    cudnn_version = get_cudnn_version_tf()

    # GPU availability
    gpu_active = check_gpu_tf()

    # libtinfo6 version
    libtinfo_version = get_lib_tinfo_version()

    # NVIDIA driver
    nvidia_driver = get_nvidia_driver_version()

    # Print all info
    print("============ Environment Info ============")
    print(f"TensorFlow version:        {tf_version}")
    print(f"CUDA version:              {cuda_version}")
    print(f"cuDNN version:             {cudnn_version}")
    print(f"GPU active (via TF):       {gpu_active}")
    print(f"Sionna version:            {sionna_version}")
    print(f"Mitsuba version:           {mitsuba_version}")
    print(f"libtinfo6 version:         {libtinfo_version}")
    print(f"NVIDIA driver version:     {nvidia_driver}")
    print("==========================================")

if __name__ == "__main__":
    main()
