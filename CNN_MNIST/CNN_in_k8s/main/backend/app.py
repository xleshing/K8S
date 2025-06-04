from flask import Flask, request, jsonify
import requests
import torch

app = Flask(__name__)
LAYER_CONTROLLER_URL = "http://layer-controller-service:5000"
model_path = "./model"
@app.route('/predict', methods=['POST'])
def predict():
    requests.post(LAYER_CONTROLLER_URL + "/load_model", json={"model_path": model_path})
    data = request.json
    input_pixels = data.get('input')
    if not input_pixels:
        return jsonify({'error': 'No input data provided'}), 400

    # 将数据转换为 PyTorch Tensor
    input_tensor = torch.tensor(input_pixels, dtype=torch.float32).view(1, 1, 28, 28)

    # 前向传播
    with torch.no_grad():
        response = requests.post(LAYER_CONTROLLER_URL + "/forward", json={"input": input_tensor.tolist()})
        outputs = torch.tensor(response.json()["output"], dtype=torch.float32)
        _, predicted = torch.max(outputs, 1)

    return jsonify({'prediction': predicted.item()})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
