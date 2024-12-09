from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import os
import json

app = Flask(__name__)

LAYER_TYPE = os.getenv("LAYER_TYPE", "ConvLayer")
LAYER_CONFIG = json.loads(os.getenv("LAYER_CONFIG", "{}"))

model = None

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
    global model
    if LAYER_TYPE == "ConvLayer":
        model = ConvLayer(**LAYER_CONFIG)
    elif LAYER_TYPE == "FcLayer":
        model = FcLayer(**LAYER_CONFIG)
    return jsonify({"message": f"{LAYER_TYPE} initialized"})

@app.route('/forward', methods=['POST'])
def forward():
    global model
    input_data = torch.tensor(request.json["input"], dtype=torch.float32)
    with torch.no_grad():
        output = model(input_data).tolist()
    return jsonify({"output": output})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
