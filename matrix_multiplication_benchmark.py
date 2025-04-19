# matrix_multiplication_benchmark.py
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from accelerator_check import check_accelerator

def benchmark_matrix_multiplication(size=1000, iterations=10):
    """Benchmark matrix multiplication performance on different accelerators."""
    # Create random matrices
    a = np.random.rand(size, size).astype(np.float32)
    b = np.random.rand(size, size).astype(np.float32)
    
    results = {}
    
    # CPU benchmark
    start_time = time.time()
    for _ in range(iterations):
        np.dot(a, b)
    results['CPU'] = (time.time() - start_time) / iterations
    
    # GPU benchmark (if available)
    if tf.test.is_gpu_available():
        a_gpu = tf.constant(a)
        b_gpu = tf.constant(b)
        start_time = time.time()
        for _ in range(iterations):
            tf.matmul(a_gpu, b_gpu)
        results['GPU'] = (time.time() - start_time) / iterations
    
    # TPU benchmark (if available)
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        
        with strategy.scope():
            a_tpu = tf.constant(a)
            b_tpu = tf.constant(b)
            start_time = time.time()
            for _ in range(iterations):
                strategy.run(lambda: tf.matmul(a_tpu, b_tpu))
            results['TPU'] = (time.time() - start_time) / iterations
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
                 f'{height:.4f}',
                 ha='center', va='bottom')
    
    plt.show()

if __name__ == "__main__":
    print("Matrix Multiplication Benchmark")
    print("--------------------------------")
    accelerator = check_accelerator()
    print(f"Available accelerator: {accelerator}")
    
    results = benchmark_matrix_multiplication()
    print("\nBenchmark Results:")
    for device, time_taken in results.items():
        if time_taken is not None:
            print(f"{device}: {time_taken:.4f} seconds per operation")
    
    plot_results(results, 'Matrix Multiplication Benchmark (1000x1000 matrices)')