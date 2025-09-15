import os
import time
from datetime import datetime
import pandas as pd

from lstm_model import (
    get_connection,
    get_pod_data,
    init_model_and_optim,
    incremental_finetune,
    save_model,
    load_model,
    save_scaler,
    load_scaler,
)

MODEL_DIR = os.getenv("MODEL_DIR", "/models")
INTERVAL = int(os.getenv("TRAIN_INTERVAL", "60"))
EPOCHS_PER_TICK = int(os.getenv("EPOCHS_PER_TICK", "1"))
SEQ_LEN = int(os.getenv("SEQ_LEN", "30"))
WINDOW_SIZE = int(os.getenv("SCALER_WINDOW", "1000"))
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "10"))

models = {}     # (ns, pod) -> model
optimizers = {} # (ns, pod) -> optimizer
criteria = {}   # (ns, pod) -> criterion
scalers = {}    # (ns, pod) -> MinMaxScaler
last_ts = {}    # (ns, pod) -> 最新 timestamp
tick_count = 0

def fetch_all_pods():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT pod_namespace, pod_name FROM pod_cpu_usage")
            rows = cur.fetchall()
        return [(r["pod_namespace"], r["pod_name"]) for r in rows]
    finally:
        conn.close()

def get_new_data(ns, pod, since_ts):
    df = get_pod_data(ns, pod)
    if df is None or df.empty:
        return pd.DataFrame()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    if since_ts is not None:
        df = df[df["timestamp"] > since_ts]
    return df

def ensure_model(ns, pod):
    key = (ns, pod)
    if key not in models:
        model, optim, crit = init_model_and_optim(input_size=6, hidden_size=64, num_layers=2, lr=1e-3)
        ckpt_path = os.path.join(MODEL_DIR, f"{ns}_{pod}.pt")
        scaler_path = os.path.join(MODEL_DIR, f"{ns}_{pod}_scaler.pkl")
        if os.path.exists(ckpt_path):
            try:
                load_model(model, optim, ckpt_path)
                print(f"[{datetime.now()}] Loaded checkpoint for {ns}/{pod}")
            except Exception as e:
                print(f"[{datetime.now()}] Load checkpoint failed for {ns}/{pod}: {e}")
        if os.path.exists(scaler_path):
            try:
                scalers[key] = load_scaler(scaler_path)
                print(f"[{datetime.now()}] Loaded scaler for {ns}/{pod}")
            except Exception as e:
                print(f"[{datetime.now()}] Load scaler failed for {ns}/{pod}: {e}")
        models[key] = model
        optimizers[key] = optim
        criteria[key] = crit

def loop_once():
    global tick_count
    pods = fetch_all_pods()
    print(f"[{datetime.now()}] Pods discovered: {len(pods)}")
    updated = 0

    for ns, pod in pods:
        key = (ns, pod)
        ensure_model(ns, pod)
        df_new = get_new_data(ns, pod, last_ts.get(key))
        if df_new.empty:
            continue

        steps, scaler = incremental_finetune(
            model=models[key],
            optimizer=optimizers[key],
            criterion=criteria[key],
            df=df_new,
            epochs=EPOCHS_PER_TICK,
            batch_size=64,
            seq_len=SEQ_LEN,
            window_size=WINDOW_SIZE,
        )
        if steps > 0:
            updated += 1
            last_ts[key] = df_new["timestamp"].max()
            scalers[key] = scaler
            print(f"[{datetime.now()}] Fine-tuned {ns}/{pod}: {len(df_new)} new rows, steps={steps}")

    tick_count += 1
    if tick_count % CHECKPOINT_EVERY == 0 and updated > 0:
        os.makedirs(MODEL_DIR, exist_ok=True)
        for (ns, pod), model in models.items():
            try:
                save_model(model, optimizers[(ns, pod)], os.path.join(MODEL_DIR, f"{ns}_{pod}.pt"))
                if (ns, pod) in scalers:
                    save_scaler(scalers[(ns, pod)], os.path.join(MODEL_DIR, f"{ns}_{pod}_scaler.pkl"))
            except Exception as e:
                print(f"[{datetime.now()}] Save checkpoint failed for {ns}/{pod}: {e}")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    while True:
        try:
            loop_once()
        except Exception as e:
            print(f"[{datetime.now()}] ERROR in loop: {e}")
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()