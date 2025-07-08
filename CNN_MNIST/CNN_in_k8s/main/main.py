import requests
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import logging

LAYER_CONTROLLER_URL = "http://172.0.0.100:32323"
# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 1024

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()

learning_rate = 0.001  # 假設的學習率

model_path = "./model.pth"

input_size = (1, 28, 28)  # MNIST 圖像大小

# 配置日志记录
logging.basicConfig(
    filename="log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="w"
)
logger = logging.getLogger()


def initialize():
    conv_layers = [
        {"out_channels": 32, "kernel_size": 3, "padding": 0, "pool_size": 2},  # ConvLayer 1
        {"out_channels": 64, "kernel_size": 3, "padding": 0, "pool_size": 2},  # ConvLayer 2
    ]

    # 計算展平大小
    flatten_size = compute_flatten_size(input_size, conv_layers)

    # 動態生成層結構
    layers = [
        {"type": "ConvLayer", "config": {"in_channels": 1, "out_channels": 32, "padding": 1}},
        {"type": "ConvLayer", "config": {"in_channels": 32, "out_channels": 64, "padding": 1}},
        {"type": "FcLayer", "config": {"in_features": flatten_size, "out_features": 128}},
        {"type": "FcLayer", "config": {"in_features": 128, "out_features": 10, "activation": False}},
    ]

    # 創建層
    requests.post(LAYER_CONTROLLER_URL + "/create_layers", json={"layers": layers})

    # 初始化層
    requests.post(LAYER_CONTROLLER_URL + "/initialize")

    requests.post(LAYER_CONTROLLER_URL + "/save_layers")


def train_model(epochs=5):
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            message = (f"Epoch [{epoch + 1}/{epochs}], Current batch index: {batch_idx + 1}, "
                       f"Total batches: {len(train_loader)}")
            logger.info(message)
            print(message)

            message = f"Batch {batch_idx}: Inputs shape {inputs.shape}, Labels shape {labels.shape}"
            logger.info(message)
            print(message)

            # 前向传播
            response = requests.post(LAYER_CONTROLLER_URL + "/forward", json={"input": inputs.tolist()})

            # 转换输出为计算图中的张量
            output = torch.tensor(response.json()["output"], dtype=torch.float32, requires_grad=True)

            # 计算损失
            loss = criterion(output, labels)

            # 检查梯度是否启用
            if not output.requires_grad:
                raise RuntimeError("Output tensor does not have requires_grad=True")

            # 计算损失的梯度
            loss.backward()  # 对最后一层进行反向传播
            loss_grad = output.grad.clone()  # 获取最后一层的梯度

            requests.post(LAYER_CONTROLLER_URL + "/backward",
                          json={"learning_rate": learning_rate, "output_grad": loss_grad.tolist()})

            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                message = (f"Epoch [{epoch + 1}/{epochs}], Step [{batch_idx + 1}/{len(train_loader)}], "
                           f"Loss: {loss.item():.4f}")
                logger.info(message)
                print(message)

        epoch_loss = running_loss / len(train_loader)
        message = f"Epoch {epoch + 1}/{epochs}, 平均损失: {epoch_loss:.4f}"
        logger.info(message)
        print(message)

    logger.info("训练完成")
    print("训练完成")

    # 保存模型
    requests.post(LAYER_CONTROLLER_URL + "/save_model", json={"model_path": model_path})


def compute_flatten_size(input_size, conv_layers):
    _, H, W = input_size  # 預設輸入的通道數、寬度、高度
    for layer in conv_layers:
        # 計算卷積後的高度和寬度
        H = (H + 2 * layer.get("padding", 0) - layer.get("kernel_size", 3)) // layer.get("stride", 1) + 1
        W = (W + 2 * layer.get("padding", 0) - layer.get("kernel_size", 3)) // layer.get("stride", 1) + 1
        C = layer.get("out_channels")

        # 池化操作後的高度和寬度
        H = H // layer.get("pool_size", 2)
        W = W // layer.get("pool_size", 2)

    # 計算展平大小
    return C * H * W


def test_model():
    # 加载模型
    requests.post(LAYER_CONTROLLER_URL + "/load_model", json={"model_path": model_path})
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            message = (f"Current batch index: {batch_idx + 1}, "
                       f"Total batches: {len(test_loader)}")
            logger.info(message)
            print(message)

            message = f"Batch {batch_idx}: Inputs shape {inputs.shape}, Labels shape {labels.shape}"
            logger.info(message)
            print(message)

            # 前向传播
            response = requests.post(LAYER_CONTROLLER_URL + "/forward", json={"input": inputs.tolist()})

            outputs = torch.tensor(response.json()["output"], dtype=torch.float32)

            # 获取预测结果
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    logger.info(f"测试准确率: {100 * correct / total:.2f}%")
    print(f"测试准确率: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    initialize()
    train_model()
    test_model()
