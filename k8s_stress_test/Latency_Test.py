import numpy as np
import time
from collections import defaultdict

matmul_latency = defaultdict(lambda: {})

n = 9216
dtype = np.float32

a = np.ones((n, n), dtype=dtype)
b = np.ones((n, n), dtype=dtype)

num_tests = 10

latencies = []

for _ in range(num_tests):
    start_time = time.time()
    np.matmul(a, b)
    end_time = time.time()

    latency = (end_time - start_time) * 1000
    latencies.append(latency)

average_latency = np.mean(latencies)

matmul_latency[f'n={n}'][dtype] = average_latency

for n, dtypes in matmul_latency.items():
    print(f"Matrix size: {n}")
    for dtype, latency in dtypes.items():
        print(f"  Data type: {dtype}, Average Latency: {latency:.3f} ms")
