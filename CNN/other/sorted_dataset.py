import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class SortedTrainingDataset(Dataset):
    def __init__(self, csv_file, transform=None, range_start=None, range_end=None):
        """
        csv_file: 排序後的 CSV 文件路徑
        transform: 圖像轉換操作
        range_start: 起始百分比 (0-100)
        range_end: 結束百分比 (0-100)
        """
        self.data = pd.read_csv(csv_file)  # 從 CSV 文件加載數據

        if range_start is not None and range_end is not None:  # 如果指定了百分比區間
            if 0 <= range_start < range_end <= 100:  # 確保範圍有效
                total_length = len(self.data)
                start_idx = (range_start * total_length) // 100
                end_idx = (range_end * total_length) // 100
                self.data = self.data.iloc[start_idx:end_idx]  # 提取目標區間的數據
            else:
                raise ValueError("range_start 和 range_end 必須在 0 到 100 之間，且 range_start < range_end")

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = row["FilePath"]  # 從 CSV 中提取圖片路徑
        class_id = row["Class"] if "Class" in row else -1  # 如果有類別信息，提取；否則默認 -1
        image = Image.open(file_path).convert("RGB")  # 加載圖像

        if self.transform:
            image = self.transform(image)

        return image, class_id  # 返回圖像和類別標籤


class SortedTestDataset(Dataset):
    def __init__(self, csv_file, transform=None, range_start=None, range_end=None):
        """
        csv_file: 排序後的 CSV 文件路徑
        transform: 圖像轉換操作
        range_start: 起始百分比 (0-100)
        range_end: 結束百分比 (0-100)
        """
        self.data = pd.read_csv(csv_file)  # 從 CSV 文件加載數據

        if range_start is not None and range_end is not None:  # 如果指定了百分比區間
            if 0 <= range_start < range_end <= 100:  # 確保範圍有效
                total_length = len(self.data)
                start_idx = (range_start * total_length) // 100
                end_idx = (range_end * total_length) // 100
                self.data = self.data.iloc[start_idx:end_idx]  # 提取目標區間的數據
            else:
                raise ValueError("range_start 和 range_end 必須在 0 到 100 之間，且 range_start < range_end")

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = row["FilePath"]  # 從 CSV 中提取圖片路徑
        class_id = row["Class"] if "Class" in row else -1  # 測試集類別可選
        image = Image.open(file_path).convert("RGB")  # 加載圖像

        if self.transform:
            image = self.transform(image)

        return image, class_id  # 返回圖像和類別標籤
