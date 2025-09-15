import os
import time
import logging
from datetime import datetime

import pymysql
from kubernetes import client, config
from kubernetes.client import CustomObjectsApi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def get_connection():
    return pymysql.connect(
        host=os.getenv('DB_HOST', 'mysql'),
        user=os.getenv('DB_USER', 'root'),
        password=os.getenv('DB_PASSWORD', 'rootpass'),
        database=os.getenv('DB_NAME', 'podmetrics'),
        cursorclass=pymysql.cursors.DictCursor
    )

# in-cluster 優先，否則用 kubeconfig（方便本地測試）
try:
    config.load_incluster_config()
except:
    config.load_kube_config()

v1 = client.CoreV1Api()
metrics_api = CustomObjectsApi()

def init_db():
    conn = get_connection()
    with conn.cursor() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS pod_cpu_usage (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                pod_namespace VARCHAR(255) NOT NULL,
                pod_name VARCHAR(255) NOT NULL,
                cpu_usage FLOAT NOT NULL
            );
        """)
    conn.commit()
    conn.close()

def collect_pod_cpu_usage(namespaces):
    usage_per_pod = {}
    for ns in namespaces:
        try:
            metrics = metrics_api.list_namespaced_custom_object(
                group="metrics.k8s.io",
                version="v1beta1",
                namespace=ns,
                plural="pods"
            )
            for item in metrics["items"]:
                pod_name = item["metadata"]["name"]
                pod_namespace = item["metadata"]["namespace"]
                cpu_total = 0.0
                for container in item["containers"]:
                    usage_cpu = container["usage"]["cpu"]
                    if usage_cpu.endswith("n"):
                        cpu_val = int(usage_cpu[:-1]) / 1e9
                    elif usage_cpu.endswith("u"):
                        cpu_val = int(usage_cpu[:-1]) / 1e6
                    elif usage_cpu.endswith("m"):
                        cpu_val = int(usage_cpu[:-1]) / 1000
                    else:
                        cpu_val = float(usage_cpu)
                    cpu_total += cpu_val
                usage_per_pod[f"{pod_namespace}/{pod_name}"] = cpu_total
        except Exception as e:
            logging.error(f"Error getting metrics for {ns}: {e}")
    return usage_per_pod

def store_cpu_usages(pod_usages):
    conn = get_connection()
    with conn.cursor() as c:
        now = datetime.utcnow()
        for full_name, cpu in pod_usages.items():
            namespace, pod = full_name.split("/", 1)
            c.execute("""
                INSERT INTO pod_cpu_usage (timestamp, pod_namespace, pod_name, cpu_usage)
                VALUES (%s, %s, %s, %s)
            """, (now.strftime('%Y-%m-%d %H:%M:%S'), namespace, pod, cpu))
    conn.commit()
    conn.close()

def remove_disappeared_pods(current_pods):
    conn = get_connection()
    with conn.cursor() as c:
        c.execute("SELECT DISTINCT pod_namespace, pod_name FROM pod_cpu_usage")
        known_pods = {(row['pod_namespace'], row['pod_name']) for row in c.fetchall()}
        current_set = set(tuple(name.split("/", 1)) for name in current_pods)
        missing_pods = known_pods - current_set
        for ns, pod in missing_pods:
            c.execute("DELETE FROM pod_cpu_usage WHERE pod_namespace = %s AND pod_name = %s", (ns, pod))
    conn.commit()
    conn.close()

def run(interval=60):
    init_db()
    targets = os.getenv("COLLECT_NAMESPACES", "default").split(",")
    targets = [t.strip() for t in targets if t.strip()]
    while True:
        try:
            pod_usages = collect_pod_cpu_usage(targets)
            store_cpu_usages(pod_usages)
            remove_disappeared_pods(list(pod_usages.keys()))
            logging.info(f"Collected and cleaned {len(pod_usages)} pods")
        except Exception as e:
            logging.error(f"Collector loop error: {e}")
        time.sleep(int(os.getenv("COLLECT_INTERVAL", "60")))

if __name__ == "__main__":
    run()