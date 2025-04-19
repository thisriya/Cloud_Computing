# accelerator_check.py
import tensorflow as tf

def check_accelerator():
    """Check which accelerator is available in the current Colab session."""
    try:
        # Check for GPU
        device_type = tf.test.gpu_device_name()
        if device_type != '':
            return "GPU"
    except:
        pass
    
    try:
        # Check for TPU
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        return "TPU"
    except:
        return "CPU"

if __name__ == "__main__":
    accelerator = check_accelerator()
    print(f"Available accelerator: {accelerator}")