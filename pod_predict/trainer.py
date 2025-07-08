import os
import time
from datetime import datetime
from lstm_model import get_pod_data, prepare_data, train_model
import torch

POD_LIST = [("default")]  # 可擴充
MODEL_DIR = "/models"
INTERVAL = 600  # 每 10 分鐘

while True:
    for namespace, pod_name in POD_LIST:
        try:
            df = get_pod_data(namespace, pod_name)
            if len(df) < 31:
                print(f"Not enough data for {namespace}/{pod_name}")
                continue
            X, y, _ = prepare_data(df)
            model = train_model(X, y)
            path = f"{MODEL_DIR}/{namespace}_{pod_name}.pt"
            torch.save(model.state_dict(), path)
            print(f"[{datetime.utcnow()}] Model saved to {path}")
        except Exception as e:
            print(f"Failed training {namespace}/{pod_name}: {e}")
    time.sleep(INTERVAL)