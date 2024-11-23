import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib
from radam import RAdam
import torch.nn.init as init

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self, input_channels, input_height, input_width, output_size):
        super(CNN, self).__init__()
        # 第一層卷積區塊
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)

        # 第二層卷積區塊
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.2)

        # 第三層卷積區塊
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.2)

        # 計算展平後的輸出大小
        conv_output_size = self._get_conv_output_size(input_channels, input_height, input_width)

        # 全連接層
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, output_size)

        # 初始化權重
        self._init_weights()

    def _init_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight, nonlinearity="relu")
            elif isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def _get_conv_output_size(self, input_channels, input_height, input_width):
        dummy_input = torch.zeros(1, input_channels, input_height, input_width)  # 假設輸入
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = self.pool1(self.batch_norm1(x))
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(self.batch_norm2(x))
        x = self.dropout2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(self.batch_norm3(x))
        x = self.dropout3(x)

        return x.numel()

    def forward(self, x):
        # 第一層卷積區塊
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(self.batch_norm1(x))
        x = self.dropout1(x)

        # 第二層卷積區塊
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(self.batch_norm2(x))
        x = self.dropout2(x)

        # 第三層卷積區塊
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(self.batch_norm3(x))
        x = self.dropout3(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全連接層
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)

        return x

    def save(self, file_name='model_cnn.pth'):
        """
        (CNN_QNet, str) -> None
        file_name: 保存状态文件的路径
        将模型状态保存到file_name
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model_cnn, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model_cnn = model_cnn
        self.optimizer = RAdam(model_cnn.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        self.losses = []
        self.gradient_norms = []
        self.parameter_gradient_norms = []

    def train_step(self, inputs, labels):
        """
        inputs: 圖像數據張量
        labels: 標籤張量
        """
        self.optimizer.zero_grad()

        # 前向傳播
        outputs = self.model_cnn(inputs)

        # 計算損失
        loss = self.criterion(outputs, labels)

        # 反向傳播與優化
        loss.backward()

        # 記錄全局梯度范數
        self.gradient_norms.append(
            torch.norm(
                torch.cat([param.grad.flatten() for param in self.model_cnn.parameters() if param.grad is not None]))
        )

        # 記錄每個參數的梯度范數
        for param in self.model_cnn.parameters():
            if param.grad is not None:
                self.parameter_gradient_norms.append(param.grad.norm().item())

        torch.nn.utils.clip_grad_norm_(self.model_cnn.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        # 保存損失
        self.losses.append(loss.item())

    def save_checkpoint(self, epoch, loss, file_path):
        """
        保存檢查點
        """
        checkpoint = {
            "model_state_dict": self.model_cnn.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss
        }
        torch.save(checkpoint, file_path)

    def load_checkpoint(self, file_path):
        """
        加載檢查點
        """
        if not torch.cuda.is_available():
            map_location = torch.device("cpu")
        else:
            map_location = None

        if not os.path.exists(file_path):
            print(f"No checkpoint found at {file_path}")
            return 0, 0.0

        checkpoint = torch.load(file_path, map_location=map_location)
        self.model_cnn.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print(f"Checkpoint loaded: epoch {epoch}, loss {loss:.4f}")
        return epoch, loss

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses, label='Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.show()

    def plot_gradient_norms(self):
        plt.figure(figsize=(10, 5))
        plt.plot([g.cpu().numpy() for g in self.gradient_norms], label='Gradient Norm')  # 移動到 CPU 並轉為 NumPy
        plt.xlabel('Training Steps')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm over Steps')
        plt.legend()
        plt.show()

    def plot_parameter_gradient_norms(self):
        # 將梯度范數從 Tensor 轉為純浮點數
        gradients = [g.item() if isinstance(g, torch.Tensor) else g for g in self.parameter_gradient_norms]

        # 繪圖
        plt.figure(figsize=(10, 5))
        plt.plot(gradients, label='Parameter Gradient Norm')
        plt.xlabel('Training Steps')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm over Steps')
        plt.legend()
        plt.show()