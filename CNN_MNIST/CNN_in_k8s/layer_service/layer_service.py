from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json

app = Flask(__name__)

LAYER_TYPE = os.getenv("LAYER_TYPE", "ConvLayer")
LAYER_CONFIG = json.loads(os.getenv("LAYER_CONFIG", "{}"))
global criterion, model, optimizer, layer_name

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
        return self.relu(self.fc(x))

@app.route('/initialize', methods=['POST'])
def initialize():
    global criterion, model
    try:
        criterion = nn.MSELoss()
        if LAYER_TYPE == "ConvLayer":
            model = ConvLayer(**LAYER_CONFIG)
        elif LAYER_TYPE == "FcLayer":
            model = FcLayer(**LAYER_CONFIG)
        return jsonify({"message": f"{LAYER_TYPE} initialized"})
    except Exception as e:
        app.logger.error(f"Error initialize layer {layer_name}: {e}")
        raise e

@app.route('/forward', methods=['POST'])
def forward():
    global model
    try:

        input_data = torch.tensor(request.json["input"], dtype=torch.float32)

        output = model(input_data)

        return jsonify({"output": output.detach().tolist()})
    except Exception as e:
        app.logger.error(f"Error forward layer {layer_name}: {e}")
        raise e

@app.route('/backward', methods=['POST'])
def backward():
    global model, optimizer
    try:
        optimizer = optim.Adam(model.parameters(), lr=request.json["learning_rate"])

        optimizer.zero_grad()

        loss = request.json["loss"]

        loss.backward()

        optimizer.step()
    except Exception as e:
        app.logger.error(f"Error backward layer {layer_name}: {e}")
        raise e

@app.route('/save', methods=['POST'])
def save():
    global model
    try:
        path = request.json["path"]
        torch.save(model.state_dict(), path)
        app.logger.error({"message": f"Model saved to {path}"})
    except Exception as e:
        app.logger.error(f"Error save layer {layer_name}: {e}")
        raise e

@app.route('/load', methods=['POST'])
def load():
    global model
    path = request.json["path"]
    model.load_state_dict(torch.load(path))
    model.eval()
    return jsonify({"message": f"Model loaded from {path}"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)