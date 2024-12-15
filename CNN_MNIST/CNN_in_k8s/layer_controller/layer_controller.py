from flask import Flask, request, jsonify
from kubernetes import client, config
import json
import time
import logging
import os
import requests

app = Flask(__name__)

# 設置日誌級別
logging.basicConfig(level=logging.INFO)

# 初始化 Kubernetes API
config.load_incluster_config()
k8s_api = client.CoreV1Api()

# 層列表
layers = []


@app.route('/create_layers', methods=['POST'])
def create_layers():
    """
    接收主程式的請求，動態創建多個層（Pod 和 Service），並串接起來。
    """
    global layers
    layers = request.json["layers"]
    created_pods = []

    for idx, layer in enumerate(layers):
        pod_name = f"layer-{idx}"
        service_name = f"layer-service-{idx}"
        layer_type = layer["type"]
        config = layer["config"]

        # 重試次數
        max_retries = 3
        success = False

        for attempt in range(max_retries):
            try:
                app.logger.info(f"Attempting to create layer {idx}: Attempt {attempt + 1}/{max_retries}")

                # 創建 Pod 和 Service
                create_pod(pod_name, layer_type, config)
                create_service(service_name, pod_name)

                # 等待 Pod 準備就緒
                wait_for_pod_ready(pod_name)

                # 記錄成功創建的層
                created_pods.append({"name": pod_name, "service": service_name, "type": layer_type})
                app.logger.info(f"Layer {idx} created successfully.")
                success = True
                break  # 跳出重試循環
            except Exception as e:
                app.logger.error(f"Failed to create layer {idx}: {e}")
                if attempt < max_retries - 1:
                    app.logger.info(f"Retrying layer {idx} creation...")
                else:
                    app.logger.error(f"Layer {idx} creation failed after {max_retries} attempts.")
                    raise e

        if not success:
            app.logger.info(f"Failed to create layer {idx} after {max_retries} attempts")

    app.logger.info(f"Layers created, created_pods: {created_pods}")


@app.route('/forward', methods=['POST'])
def forward():
    """
    接收主程式的訓練請求，讀取數據集並進行反向傳播訓練。
    """
    global layers

    app.logger.info(f"Starting forward")
    input_data = request.json()("input")

    for idx, layer in enumerate(layers):
        service_name = f"layer-service-{idx}"
        url = f"http://{service_name}:5000/forward"
        try:
            response = requests.post(url, json={"input": input_data})
            if response.json()("output"):
                input_data = response.json()["output"]
                app.logger.info(f"Successful forward on layer {idx}")
            else:
                app.logger.error(f"Failed forward on layer {idx}: {response.json()("message")}")
        except Exception as e:
            app.logger.error(f"Failed to requests layer {idx}: {e}")
            raise e

    app.logger.info("Forward complete")
    return jsonify({"output": input_data})


@app.route('/backward', methods=['POST'])
def backward():
    global layers

    app.logger.info(f"Starting backward")
    learning_rate = request.json()("learning_rate")
    loss = request.json()("loss")

    for idx, layer in enumerate(layers):
        service_name = f"layer-service-{idx}"
        url = f"http://{service_name}:5000/backward"
        try:
            response = requests.post(url, json={"learning_rate": learning_rate, "loss": loss})
            if response:
                app.logger.error(f"Failed backward on layer {idx}: {response.json()("message")}")
            else:
                app.logger.info(f"Successful backward on layer {idx}")
        except Exception as e:
            app.logger.error(f"Failed to requests layer {idx}: {e}")
            raise e

    app.logger.info("Backward complete")


@app.route('/initialize', methods=['POST'])
def initialize():
    app.logger.info(f"Starting initialize")
    for idx, layer in enumerate(layers):
        service_name = f"layer-service-{idx}"
        url = f"http://{service_name}:5000/initialize"
        try:
            response = requests.post(url)
            if response:
                app.logger.error(f"Failed initialize layer {idx}: {response.json()("message")}")
            else:
                app.logger.info(f"Successful initialize layer {idx}")
        except Exception as e:
            app.logger.error(f"Failed to requests layer {idx}: {e}")
            raise e

    app.logger.info("initialize complete")


@app.route('/save_model', methods=['POST'])
def save_model():
    app.logger.info(f"Starting save_model")
    for idx, layer in enumerate(layers):
        service_name = f"layer-service-{idx}"
        url = f"http://{service_name}:5000/save"
        try:
            response = requests.post(url, json={"path": os.path.join(request.json()("model_path"), f"layer-{idx}.pth")})
            if response:
                app.logger.error(f"Failed save layer {idx}: {response.json()("message")}")
            else:
                app.logger.info(f"Successful save layer {idx}")
        except Exception as e:
            app.logger.error(f"Failed to requests layer {idx}: {e}")
            raise e

    app.logger.info("Save_model complete")


def create_pod(pod_name, layer_type, config):
    """
    創建 Kubernetes Pod。
    """
    app.logger.info(f"Starting create_pod")
    try:
        container = client.V1Container(
            name=pod_name,
            image="ycair/cnn_layer_service",  # 替換為您的 Docker 映像名稱
            env=[
                client.V1EnvVar(name="LAYER_TYPE", value=layer_type),
                client.V1EnvVar(name="LAYER_CONFIG", value=json.dumps(config)),
            ],
            ports=[client.V1ContainerPort(container_port=5000)],
        )
        spec = client.V1PodSpec(containers=[container])
        metadata = client.V1ObjectMeta(name=pod_name, labels={"app": pod_name})
        pod = client.V1Pod(spec=spec, metadata=metadata)

        k8s_api.create_namespaced_pod(namespace="default", body=pod)
        app.logger.info(f"Pod {pod_name} created successfully.")
    except Exception as e:
        app.logger.error(f"Error creating pod {pod_name}: {e}")
        raise e


def create_service(service_name, pod_name):
    """
    創建 Kubernetes Service。
    """
    app.logger.info(f"Starting create_service")
    try:
        service = client.V1Service(
            metadata=client.V1ObjectMeta(name=service_name),
            spec=client.V1ServiceSpec(
                selector={"app": pod_name},
                ports=[client.V1ServicePort(port=5000, target_port=5000)],
            ),
        )
        k8s_api.create_namespaced_service(namespace="default", body=service)
        app.logger.info(f"Service {service_name} created successfully.")
    except Exception as e:
        app.logger.error(f"Error creating service {service_name}: {e}")
        raise e


def wait_for_pod_ready(pod_name, timeout=120):
    """
    等待 Pod 變為 Ready 狀態。
    """
    app.logger.info(f"Wait_for_pod_ready")
    start_time = time.time()
    while True:
        pod = k8s_api.read_namespaced_pod(name=pod_name, namespace="default")
        if pod.status.phase == "Running" and all(
                condition.status == "True" for condition in pod.status.conditions if condition.type == "Ready"
        ):
            app.logger.info(f"Pod {pod_name} is ready.")
            break
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for pod {pod_name} to become ready.")
        time.sleep(5)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
