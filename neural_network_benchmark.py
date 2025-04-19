# neural_network_benchmark.py
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from accelerator_check import check_accelerator

def create_model():
    """Create a simple neural network model."""
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

def benchmark_nn_training(batch_size=128, epochs=5):
    """Benchmark neural network training performance on different accelerators."""
    # Load MNIST dataset
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    
    results = {}
    
    # CPU training
    with tf.device('/CPU:0'):
        model_cpu = create_model()
        model_cpu.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
        start_time = time.time()
        model_cpu.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
        results['CPU'] = time.time() - start_time
    
    # GPU training (if available)
    if tf.test.is_gpu_available():
        with tf.device('/GPU:0'):
            model_gpu = create_model()
            model_gpu.compile(optimizer='adam',
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                             metrics=['accuracy'])
            start_time = time.time()
            model_gpu.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
            results['GPU'] = time.time() - start_time
    
    # TPU training (if available)
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        
        with strategy.scope():
            model_tpu = create_model()
            model_tpu.compile(optimizer='adam',
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                             metrics=['accuracy'])
            start_time = time.time()
            model_tpu.fit(x_train, y_train, 
                         batch_size=batch_size*strategy.num_replicas_in_sync, 
                         epochs=epochs, verbose=0)
            results['TPU'] = time.time() - start_time
    except:
        pass
    
    return results

def plot_results(results, title):
    """Plot the benchmark results."""
    devices = []
    times = []
    for device, time_taken in results.items():
        if time_taken is not None:
            devices.append(device)
            times.append(time_taken)
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(devices, times, color=['blue', 'green', 'red'])
    plt.title(title)
    plt.ylabel('Time (seconds)')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')
    
    plt.show()

if __name__ == "__main__":
    print("Neural Network Training Benchmark")
    print("----------------------------------")
    accelerator = check_accelerator()
    print(f"Available accelerator: {accelerator}")
    
    results = benchmark_nn_training()
    print("\nBenchmark Results:")
    for device, time_taken in results.items():
        if time_taken is not None:
            print(f"{device}: {time_taken:.2f} seconds for 5 epochs")
    
    plot_results(results, 'Neural Network Training Benchmark (5 epochs on MNIST)')