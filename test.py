import tensorflow as tf
import sionna as sn

# تست TensorFlow
print("TensorFlow version:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)
print("Physical devices:", tf.config.list_physical_devices('GPU'))

# تست Sionna
print("Sionna version:", sn.__version__)

# تعریف صحنه
scene = sn.rt.Scene()

# تعریف فرستنده و آرایه آنتنش
tx = sn.rt.Transmitter("tx", [0, 0, 0])
tx_array = sn.rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso")
tx.array = tx_array
scene.add(tx)

# تعریف گیرنده و آرایه آنتنش
rx = sn.rt.Receiver("rx", [1, 1, 1])
rx_array = sn.rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso")
rx.array = rx_array
scene.add(rx)

# محاسبه مسیرها
paths = scene.compute_paths()
print("Sionna works!")