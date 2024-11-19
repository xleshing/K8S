import torch
import torch.nn as nn
from torch_optimizer import Lookahead
import torch.nn.functional as F
import os
import matplotlib
from radam import RAdam
import torch.nn.init as init
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self, input_channels, output_size, input_height, input_width, conv_channels=[4, 8]):
        super(CNN, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.conv1 = nn.Conv2d(input_channels, conv_channels[0], kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(conv_channels[0])
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(conv_channels[1])
        self.flatten = nn.Flatten()  # 替代手動計算
        self.fc1 = nn.Linear(self._get_conv_output_size(input_channels, self.input_height, self.input_width), output_size)

        # 初始化權重
        self._init_weights()

    def _init_weights(self):
        init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")

    def _get_conv_output_size(self, input_channels, input_height, input_width):
        dummy_input = torch.zeros(1, input_channels, input_height, input_width)  # 假設輸入維度 (C, H, W)
        x = F.relu(self.conv1(dummy_input))
        x = F.max_pool2d(self.batch_norm1(x), 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(self.batch_norm2(x), 2)
        return x.numel()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.batch_norm1(x), 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(self.batch_norm2(x), 2)
        x = self.flatten(x)
        x = self.fc1(x)
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
        self.optimizer = Lookahead(RAdam(model_cnn.parameters(), lr=self.lr), k=5, alpha=0.5)
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        self.losses = []
        self.gradient_norms = []
        self.parameter_gradient_norms = []


    def train_step(self, state_cnn, action, reward, next_state_cnn, done):
        # 將狀態轉換為張量
        state_cnn = torch.tensor(state_cnn, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        next_state_cnn = torch.tensor(next_state_cnn, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.bool)

        # 前向傳播，預測 Q 值
        pred = self.model_cnn(state_cnn)  # (batch_size, num_actions)
        target = pred.clone().detach()

        # 計算新的 Q 值
        Q_new = reward + (0 if done else self.gamma * torch.max(self.model_cnn(next_state_cnn)))

        # 更新目標值（索引基於動作的編號）
        target[0, action] = Q_new  # 假設 action 是 0, 1, 或 2

        # 計算損失並反向傳播
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
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

        # 梯度裁剪與參數更新
        torch.nn.utils.clip_grad_norm_(self.model_cnn.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        # 保存損失
        self.losses.append(loss.item())

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
        plt.plot(self.gradient_norms, label='Gradient Norm')
        plt.xlabel('Training Steps')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm over Steps')
        plt.legend()
        plt.show()

    def plot_parameter_gradient_norms(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.parameter_gradient_norms, label='Parameter Gradient Norm')
        plt.xlabel('Training Steps')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm over Steps')
        plt.legend()
        plt.show()

