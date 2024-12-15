import requests
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

LAYER_CONTROLLER_URL = "http://localhost:32323"
# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()

learning_rate = 0.001  # 假設的學習率

model_path = "./model"


def train_model(epochs=5):

    # 指定層數與配置
    layers = [
        {"type": "ConvLayer", "config": {"in_channels": 1, "out_channels": 32}},
        {"type": "ConvLayer", "config": {"in_channels": 32, "out_channels": 64}},
        {"type": "FcLayer", "config": {"in_features": 64 * 7 * 7, "out_features": 128}},
        {"type": "FcLayer", "config": {"in_features": 128, "out_features": 10, "activation": False}},
    ]

    # 創建層
    requests.post(LAYER_CONTROLLER_URL + "/create_layers", json={"layers": layers})

    # 初始化層
    requests.post(LAYER_CONTROLLER_URL + "/initialize")

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # 前向传播
            response = requests.post(LAYER_CONTROLLER_URL + "/forward", json={"input": inputs})

            # 计算损失
            loss = criterion(response.json()("output"), labels)

            # 反向传播和优化
            requests.post(LAYER_CONTROLLER_URL + "/backward", json={"learning_rate": learning_rate, "loss": loss})

            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, 平均损失: {epoch_loss:.4f}")

    print("训练完成")

    # 保存模型
    requests.post(LAYER_CONTROLLER_URL + "/save_model", json={"model_path": model_path})


if __name__ == "__main__":
    train_model()
