import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("cuDNN version:", tf.sysconfig.get_build_info().get("cudnn_version", "Not found"))
