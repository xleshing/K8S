from flask import Flask, request, jsonify
from kubernetes import client, config
import json
import time
import logging
import os
import requests
import torch

app = Flask(__name__)

# 設置日誌級別
logging.basicConfig(level=logging.INFO)

# 初始化 Kubernetes API
config.load_incluster_config()
k8s_api = client.CoreV1Api()

# 層列表
LAYERS_FILE = "./layers.json"  # 保存層結構的默認檔案路徑
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
    return jsonify({"message": "Layers created successfully", "created_pods": created_pods}), 200


@app.route('/forward', methods=['POST'])
def forward():
    """
    接收主程式的訓練請求，讀取數據集並進行反向傳播訓練。
    """
    global layers

    app.logger.info(f"Starting forward")
    input_data = request.json["input"]

    for idx, layer in enumerate(layers):
        service_name = f"layer-service-{idx}"
        url = f"http://{service_name}:5000/forward"
        try:
            response = requests.post(url, json={"input": input_data})
            if response.status_code == 200:
                input_data = response.json()["output"]
                app.logger.info(f"Successful forward on layer {idx}")
            else:
                app.logger.error(f"Failed forward on layer {idx}: {response.json()['message']}")
        except Exception as e:
            app.logger.error(f"Failed to requests layer {idx}: {e}")
            raise e

    app.logger.info("Forward complete")
    return jsonify({"output": input_data}), 200


@app.route('/backward', methods=['POST'])
def backward():
    global layers

    app.logger.info(f"Starting backward")

    learning_rate = request.json["learning_rate"]
    loss_grad = request.json["output_grad"]  # 初始梯度（最后一层的输出梯度）

    # 从最后一层逐层反向传递
    for idx in range(len(layers) - 1, -1, -1):  # 从最后一层向前
        service_name = f"layer-service-{idx}"
        url = f"http://{service_name}:5000/backward"
        try:
            response = requests.post(url, json={"learning_rate": learning_rate, "output_grad": loss_grad})
            if response.status_code == 200:
                # 成功获取当前层的输入梯度
                loss_grad = response.json()["input_grad"]
                app.logger.info(f"Successful backward on layer {idx}")
            else:
                app.logger.error(f"Failed backward on layer {idx}: {response.json()['message']}")
        except Exception as e:
            app.logger.error(f"Failed to requests layer {idx}: {e}")
            raise e

    app.logger.info("Backward complete")
    return jsonify({"message": "Backward complete"}), 200


@app.route('/initialize', methods=['POST'])
def initialize():
    app.logger.info(f"Starting initialize")
    for idx, layer in enumerate(layers):
        service_name = f"layer-service-{idx}"

        try:
            wait_for_pod_ready(f"layer-{idx}")
        except TimeoutError as e:
            app.logger.error(f"Service {service_name} failed to become ready: {e}")
            return jsonify({"error": f"Service {service_name} not ready"}), 500

        url = f"http://{service_name}:5000/initialize"
        try:
            response = requests.post(url)
            if response.status_code == 500:
                app.logger.error(f"Failed initialize layer {idx}: {response.json()['message']}")
            else:
                app.logger.info(f"Successful initialize layer {idx}")
        except Exception as e:
            app.logger.error(f"Failed to requests layer {idx}: {e}")
            raise e

    app.logger.info("initialize complete")
    return jsonify({"message": "Initialization complete"}), 200


@app.route('/save_model', methods=['POST'])
def save_model():
    app.logger.info(f"Starting save_model")
    for idx, layer in enumerate(layers):
        service_name = f"layer-service-{idx}"
        url = f"http://{service_name}:5000/save"
        try:
            response = requests.post(url, json={"path": os.path.join(request.json["model_path"], f"layer-{idx}.pth")})
            if response.status_code == 500:
                app.logger.error(f"Failed save layer {idx}: {response.json()['message']}")
            else:
                app.logger.info(f"Successful save layer {idx}")
        except Exception as e:
            app.logger.error(f"Failed to requests layer {idx}: {e}")
            raise e

    app.logger.info("Save_model complete")
    return jsonify({"message": "Save_model complete"}), 200


@app.route('/load_model', methods=['POST'])
def load_model():
    app.logger.info(f"Starting load_model")
    for idx, layer in enumerate(layers):
        service_name = f"layer-service-{idx}"
        url = f"http://{service_name}:5000/load"
        try:
            response = requests.post(url, json={"path": os.path.join(request.json["model_path"], f"layer-{idx}.pth")})
            if response.status_code == 500:
                app.logger.error(f"Failed load layer {idx}: {response.json()['message']}")
            else:
                app.logger.info(f"Successful load layer {idx}")
        except Exception as e:
            app.logger.error(f"Failed to requests layer {idx}: {e}")
            raise e

    app.logger.info("Load_model complete")
    return jsonify({"message": "Load_model complete"}), 200


@app.route('/save_layers', methods=['POST'])
def save_layers(file_path=LAYERS_FILE):
    """
    將 layers 保存到指定檔案。
    """
    global layers
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    try:
        with open(file_path, "w") as f:
            json.dump(layers, f)
        app.logger.info(f"Successful save layers")
    except Exception as e:
        app.logger.error(f"Failed to save layers: {str(e)}")
        raise e
    app.logger.info("Save layers complete")
    return jsonify({"message": "Save layers complete"}), 200


def load_layers(file_path=LAYERS_FILE):
    """
    從指定檔案讀取 layers。如果檔案不存在，返回空列表。
    """
    global layers
    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                layers = json.load(f)
            app.logger.info(f"Layers loaded successfully from {file_path}")
        else:
            app.logger.error(f"No layers file found at {file_path}. Returning empty list.")
            layers = []
    except Exception as e:
        app.logger.error(f"Failed to load layers: {str(e)}")
        raise e


def create_pod(pod_name, layer_type, config):
    """
    創建 Kubernetes Pod。如果同名 Pod 已存在，則刪除並重新創建。
    """
    app.logger.info(f"Starting create_pod for {pod_name}")
    try:
        # 檢查是否已存在同名的 Pod
        existing_pod = k8s_api.list_namespaced_pod(namespace="default", field_selector=f"metadata.name={pod_name}")
        if existing_pod.items:
            app.logger.info(f"Pod {pod_name} already exists. Forcing deletion...")
            # 使用 force delete（grace_period_seconds=0, propagation_policy='Foreground'）
            delete_opts = client.V1DeleteOptions(
                grace_period_seconds=0,
                propagation_policy='Foreground'
            )
            k8s_api.delete_namespaced_pod(
                name=pod_name,
                namespace="default",
                body=delete_opts
            )
            # 等待 Pod 被刪除
            wait_for_deletion(pod_name, "pod")

        # 創建新的 Pod
        container = client.V1Container(
            name=pod_name,
            image="icanlab/cnn_layer_service",  # 替換為您的 Docker 映像名稱
            env=[
                client.V1EnvVar(name="LAYER_TYPE", value=layer_type),
                client.V1EnvVar(name="LAYER_CONFIG", value=json.dumps(config)),
            ],
            ports=[client.V1ContainerPort(container_port=5000)],
        )
        spec = client.V1PodSpec(
            containers=[container],
            node_selector={"beta.kubernetes.io/arch": "amd64"}
        )
        metadata = client.V1ObjectMeta(name=pod_name, labels={"app": pod_name})
        pod = client.V1Pod(spec=spec, metadata=metadata)

        k8s_api.create_namespaced_pod(namespace="default", body=pod)
        app.logger.info(f"Pod {pod_name} created successfully.")
    except Exception as e:
        app.logger.error(f"Error creating pod {pod_name}: {e}")
        raise e


def create_service(service_name, pod_name):
    """
    創建 Kubernetes Service。如果同名 Service 已存在，則刪除並重新創建。
    """
    app.logger.info(f"Starting create_service for {service_name}")
    try:
        # 檢查是否已存在同名的 Service
        existing_service = k8s_api.list_namespaced_service(namespace="default",
                                                           field_selector=f"metadata.name={service_name}")
        if existing_service.items:
            app.logger.info(f"Service {service_name} already exists. Deleting it...")
            k8s_api.delete_namespaced_service(name=service_name, namespace="default")
            # 等待 Service 被刪除
            wait_for_deletion(service_name, "service")

        # 創建新的 Service
        service = client.V1Service(
            metadata=client.V1ObjectMeta(name=service_name),
            spec=client.V1ServiceSpec(
                selector={"app": pod_name},  # 確保與 Pod 的 labels 匹配
                ports=[client.V1ServicePort(port=5000, target_port=5000)],
            ),
        )
        k8s_api.create_namespaced_service(namespace="default", body=service)
        app.logger.info(f"Service {service_name} created successfully.")
    except Exception as e:
        app.logger.error(f"Error creating service {service_name}: {e}")
        raise e


def wait_for_deletion(resource_name, resource_type, namespace="default", timeout=60):
    """
    等待資源（Pod 或 Service）被刪除。
    """
    app.logger.info(f"Waiting for {resource_type} {resource_name} to be deleted...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            if resource_type == "pod":
                k8s_api.read_namespaced_pod(name=resource_name, namespace=namespace)
            elif resource_type == "service":
                k8s_api.read_namespaced_service(name=resource_name, namespace=namespace)
            else:
                raise ValueError(f"Unsupported resource type: {resource_type}")
            # 如果沒有拋出異常，說明資源仍存在
            time.sleep(1)
        except client.exceptions.ApiException as e:
            if e.status == 404:  # 資源已刪除
                app.logger.info(f"{resource_type.capitalize()} {resource_name} deleted successfully.")
                return
            else:
                raise e

    raise TimeoutError(f"Timeout waiting for {resource_type} {resource_name} to be deleted.")


def wait_for_pod_ready(pod_name, timeout=180):
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
    load_layers()
