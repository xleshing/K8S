import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import CNN, QTrainer
from settings import *
from dataset import TestDataset, TrainingDataset
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt


class Main:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 訓練數據轉換
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 加載數據集
        self.train_dataset = TrainingDataset(root_dir=TRAIN_DATASET_DIR, transform=self.transform)
        self.test_dataset = TestDataset(root_dir=TEST_DATASET_DIR, csv_file=TEST_CSV_FILE, transform=self.transform)

        # 定義 DataLoader
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 檢查數據加載
        for images, labels in self.train_loader:
            print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
            break

        # 檢查數據加載
        for images, filenames in self.test_loader:
            print(f"Images shape: {images.shape}")  # 批次圖像的形狀
            print(f"Filenames: {filenames[:5]}")  # 顯示前 5 個文件名
            break

        # 初始化模型
        self.model = CNN(input_channels=INPUT_CHANNELS,
                         output_size=NUM_CLASSES,
                         input_height=INPUT_HEIGHT,
                         input_width=INPUT_WIDTH).to(self.device)

        # 初始化 QTrainer
        self.trainer = QTrainer(model_cnn=self.model, lr=LR, gamma=GAMMA)

        self.patience = PATIENCE
        self.no_improve_epochs = 0
        self.best_avg_loss = float('inf')  # 初始化最佳驗證損失
        self.avg_losses = []  # 保存每個 epoch 的平均 Loss

    def main(self, is_train=1):

        if is_train:
            # 加載檢查點（如果有）
            start_epoch, _ = self.trainer.load_checkpoint(CHECKPOINT_PATH, mode="full")

            # 訓練模型
            self.train_model_with_step(epochs=EPOCHS, checkpoint_path=CHECKPOINT_PATH,
                                       start_epoch=start_epoch)

            # 測試模型
            self.test_model()

            self.trainer.save_checkpoint(file_path=MODEL_LOAD_PATH, epoch=None, loss=None, mode="model_only")
        else:
            self.trainer.load_checkpoint(MODEL_LOAD_PATH, mode="model_only")  # 僅加載模型參數
            self.test_model()  # 測試模型

    def train_model_with_step(self, epochs, checkpoint_path, start_epoch=0):
        """
        使用 trainer 的 train_step 方法進行模型訓練
        trainer: 包含模型、優化器與損失計算的封裝類（如 QTrainer）
        train_loader: PyTorch DataLoader，提供訓練數據
        epochs: 訓練週期數
        """
        total_batches = len(self.train_loader)
        for epoch in range(start_epoch, epochs):
            self.trainer.model_cnn.train()  # 確保模型處於訓練模式
            running_loss = 0.0

            progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}", unit="batch")

            for images, labels in progress_bar:
                # 將數據轉移到 GPU（如果可用）
                images, labels = images.to(self.device), labels.to(self.device)

                # 使用 train_step 處理單個批次
                self.trainer.train_step(images, labels)

                # 累計損失
                running_loss += self.trainer.losses[-1]  # 獲取當前批次的損失

                # 動態更新進度條後綴
                progress_bar.set_postfix({"Batch Loss": self.trainer.losses[-1],
                                          "Avg Loss": running_loss / total_batches})

            current_avg_loss = running_loss / total_batches

            self.avg_losses.append(current_avg_loss)

            # 如果當前正確率是最高的，保存模型
            if current_avg_loss < self.best_avg_loss:
                self.best_avg_loss = current_avg_loss
                self.no_improve_epochs = 0
                self.trainer.save_checkpoint(file_path=checkpoint_path, epoch=epoch + 1, loss=running_loss / total_batches, mode="full")
                print(f"New best model saved to {checkpoint_path}")
            else:
                self.no_improve_epochs += 1

            if self.no_improve_epochs >= self.patience:
                print(f"Early stopping triggered. Best Avg Loss: {self.best_avg_loss:.4f}")
                break

            # 每個 epoch 輸出平均損失
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / total_batches:.4f}")

        # 可視化損失與梯度變化
        self.trainer.plot_losses()
        self.trainer.plot_gradient_norms()
        self.trainer.plot_parameter_gradient_norms()

        self.plot_avg_losses(self.avg_losses)
        # 保存平均 Loss 到文件
        self.save_avg_losses(self.avg_losses, f"CSV/avg_losses.csv")

        print("Training completed. AVG Losses saved to CSV/avg_losses.csv.")

    def save_avg_losses(self, avg_losses, file_path):
        """
        保存每個 epoch 的平均 Loss 到 CSV 文件
        avg_losses: 包含平均 Loss 的列表
        file_path: 保存 CSV 文件的路徑
        """
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "AVG Loss"])  # 寫入標題
            for epoch, loss in enumerate(avg_losses, start=1):
                writer.writerow([epoch, loss])  # 寫入每一行

    def test_model(self):  # , output_csv=OUTPUT_CSV):
        """
        測試模型並保存結果到 CSV，包含進度條顯示
        model: 測試的模型
        test_loader: 測試數據的 DataLoader
        output_csv: 保存測試結果的 CSV 文件路徑
        """
        self.model.eval()  # 設置模型為測試模式
        self.model.to(self.device)  # 確保模型在正確的設備上

        # 初始化變量
        correct = 0
        total = 0
        # results = []  # 保存每個文件的測試結果

        # 加入 tqdm 顯示進度
        with torch.no_grad():
            with tqdm(total=len(self.test_loader), desc="Testing Progress", unit="batch") as pbar:
                for images, labels in self.test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)  # 獲取預測類別

                    # 記錄準確率計算
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # 保存真實類別和預測類別
                    # results.extend(zip(labels.cpu().numpy(), predicted.cpu().numpy()))

                    # 更新進度條
                    pbar.update(1)

        # 計算總體準確率
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")

        """# 保存結果到 CSV
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["TrueClass", "PredictedClass"])  # CSV 標題
            writer.writerows(results)  # 寫入每行數據

        print(f"Predictions saved to {output_csv}")"""

    def plot_avg_losses(self, avg_losses):
        """
        繪製平均 Loss 隨 epoch 的變化圖
        avg_losses: 包含每個 epoch 平均 Loss 的列表
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(avg_losses) + 1), avg_losses, marker='o', label="AVG Loss")
        plt.xlabel("Epoch")
        plt.ylabel("AVG Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    main = Main()
    main.main(IS_TRAIN)
