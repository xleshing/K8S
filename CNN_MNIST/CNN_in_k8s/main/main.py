import requests

LAYER_CONTROLLER_URL = "http://layer-controller-service:5000"

def main():
    # 指定層數與配置
    layers = [
        {"type": "ConvLayer", "config": {"in_channels": 1, "out_channels": 32}},
        {"type": "ConvLayer", "config": {"in_channels": 32, "out_channels": 64}},
        {"type": "FcLayer", "config": {"in_features": 64 * 7 * 7, "out_features": 128}},
        {"type": "FcLayer", "config": {"in_features": 128, "out_features": 10, "activation": False}},
    ]

    # 發送到 LayerController
    response = requests.post(LAYER_CONTROLLER_URL + "/create_layers", json={"layers": layers})
    if response.status_code == 200:
        print("Layers created successfully!")
        print(response.json())
    else:
        print("Failed to create layers:", response.text)

    # 測試前向傳播
    test_input = [[[[0.5] * 28] * 28]]  # 假設是 MNIST 的一個樣本
    response = requests.post(LAYER_CONTROLLER_URL + "/forward", json={"input": test_input})
    if response.status_code == 200:
        print("Forward pass result:", response.json())
    else:
        print("Forward pass failed:", response.text)

if __name__ == "__main__":
    main()
