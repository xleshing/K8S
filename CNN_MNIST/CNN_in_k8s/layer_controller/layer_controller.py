from flask import Flask, request, jsonify
from kubernetes import client, config
import time

app = Flask(__name__)

# 初始化 Kubernetes API
config.load_incluster_config()
k8s_api = client.CoreV1Api()

# 層列表
layers = []

@app.route('/create_layers', methods=['POST'])
def create_layers():
    global layers
    layers = request.json["layers"]

    created_pods = []

    for idx, layer in enumerate(layers):
        pod_name = f"layer-{idx}"
        service_name = f"layer-service-{idx}"
        layer_type = layer["type"]
        config = layer["config"]

        # 創建 Pod 和 Service
        create_pod(pod_name, layer_type, config)
        create_service(service_name, pod_name)

        # 註冊層到列表
        created_pods.append({"name": pod_name, "service": service_name, "type": layer_type})

        # 等待 Pod 啟動
        wait_for_pod_ready(pod_name)

    return jsonify({"message": "Layers created", "created_pods": created_pods})

@app.route('/forward', methods=['POST'])
def forward():
    global layers
    input_data = request.json["input"]

    for idx, layer in enumerate(layers):
        service_name = f"layer-service-{idx}"
        url = f"http://{service_name}:5000/forward"
        response = requests.post(url, json={"input": input_data})
        input_data = response.json()["output"]

    return jsonify({"output": input_data})

def create_pod(name, layer_type, config):
    container = client.V1Container(
        name=name,
        image="your-layer-image",
        env=[client.V1EnvVar(name="LAYER_TYPE", value=layer_type),
             client.V1EnvVar(name="LAYER_CONFIG", value=json.dumps(config))],
        ports=[client.V1ContainerPort(container_port=5000)],
    )
    spec = client.V1PodSpec(containers=[container])
    metadata = client.V1ObjectMeta(name=name, labels={"app": "layer"})
    pod = client.V1Pod(spec=spec, metadata=metadata)

    k8s_api.create_namespaced_pod(namespace="default", body=pod)

def create_service(name, pod_name):
    service = client.V1Service(
        metadata=client.V1ObjectMeta(name=name),
        spec=client.V1ServiceSpec(
            selector={"app": "layer", "pod-name": pod_name},
            ports=[client.V1ServicePort(port=5000, target_port=5000)],
        )
    )
    k8s_api.create_namespaced_service(namespace="default", body=service)

def wait_for_pod_ready(name):
    while True:
        pod = k8s_api.read_namespaced_pod(name=name, namespace="default")
        if pod.status.phase == "Running":
            break
        time.sleep(1)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
