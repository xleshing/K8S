from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json

app = Flask(__name__)

LAYER_TYPE = os.getenv("LAYER_TYPE", "ConvLayer")
LAYER_CONFIG = json.loads(os.getenv("LAYER_CONFIG", "{}"))
global model, optimizer, output_data, input_data


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, pool_size=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(pool_size, stride=pool_size)

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))


class FcLayer(nn.Module):
    def __init__(self, in_features, out_features, activation=True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU() if activation else nn.Identity()

    def forward(self, x):
        # 展平操作，從第二維展平到最後一維
        x = torch.flatten(x, start_dim=1)
        return self.relu(self.fc(x))

@app.route('/initialize', methods=['POST'])
def initialize():
    global model
    try:
        if LAYER_TYPE == "ConvLayer":
            model = ConvLayer(**LAYER_CONFIG)
        elif LAYER_TYPE == "FcLayer":
            model = FcLayer(**LAYER_CONFIG)
        return jsonify({"message": "Layer initialized successfully"}), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route('/forward', methods=['POST'])
def forward():
    global model, output_data, input_data
    try:
        input_data = torch.tensor(request.json["input"], dtype=torch.float32, requires_grad=True)

        output_data = model(input_data)

        return jsonify({"output": output_data.tolist(), "message": "Layer initialized successfully"}), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route('/backward', methods=['POST'])
def backward():
    global model, optimizer, output_data, input_data
    try:
        learning_rate = request.json["learning_rate"]
        output_grad = torch.tensor(request.json["output_grad"], dtype=torch.float32)

        # 设置优化器
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.zero_grad()

        # 反向传播
        output_data.backward(gradient=output_grad)

        # 获取输入梯度 (上一层需要的梯度)
        input_grad = input_data.grad.clone()

        # 优化模型参数
        optimizer.step()

        return jsonify({"message": "Backward success", "input_grad": input_grad.tolist()}), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route('/save', methods=['POST'])
def save():
    global model
    try:
        # 獲取請求中的保存路徑，如果沒有提供則使用默認路徑
        path = request.json.get("path", "./default_model.pth")

        # 確保目錄存在
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # 保存模型參數
        torch.save(model.state_dict(), path)
        return jsonify({"message": f"Layer saved successfully at {path}"}), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route('/load', methods=['POST'])
def load():
    global model
    try:
        # 獲取請求中的模型路徑，如果沒有提供則使用默認路徑
        path = request.json.get("path", "./default_model.pth")

        # 確認模型檔案是否存在
        if not os.path.exists(path):
            return jsonify({"message": f"Model file not found at {path}"}), 404

        # 加載模型參數
        model.load_state_dict(torch.load(path))
        model.eval()  # 設置為評估模式

        return jsonify({"message": f"Layer loaded successfully from {path}"}), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
