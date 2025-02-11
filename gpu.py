import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Gefundene GPUs:", tf.config.list_physical_devices('GPU'))
print(tf.config.list_physical_devices('GPU'))
physical_devices = tf.config.list_physical_devices('GPU')
print("Verfügbare GPUs:", physical_devices)

# Dynamische Speicherzuweisung für GPUs aktivieren
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)