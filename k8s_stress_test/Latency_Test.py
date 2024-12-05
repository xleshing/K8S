import numpy as np
import time
import logging
from collections import defaultdict
from flask import Flask

app = Flask(__name__)

# 配置 logging
logging.basicConfig(level=logging.INFO)

# 儲存每個情況下的延遲
matmul_latency = defaultdict(lambda: {})


@app.route('/start', methods=['GET'])
def start_matrix_multiply():
    # 固定矩陣大小
    n = 9216
    dtype = np.float32
    a = np.ones((n, n), dtype=np.float32)
    b = np.ones((n, n), dtype=np.float32)

    # 計測矩陣乘法所需時間
    start_time = time.time()
    np.matmul(a, b)
    end_time = time.time()

    latency = (end_time - start_time) * 1000  # 轉換為毫秒
    matmul_latency[f'n={n}'][dtype] = latency

    # 顯示測試結果到標準輸出
    response = ""
    for n, dtypes in matmul_latency.items():
        response += f"Matrix size: {n}\n"
        for dtype, latency in dtypes.items():
            response += f"  Data type: {dtype}, Latency: {latency:.3f} ms\n"

    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # 監聽所有IP的 5000 端口
