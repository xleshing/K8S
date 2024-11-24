import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class TrainingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: 包含 'Images/' 和各類別目錄的根目錄
        transform: 圖像轉換操作
        """
        self.data = []
        self.transform = transform

        # 遍歷每個類別目錄
        image_dir = os.path.join(root_dir, "Images")
        for folder_name in os.listdir(image_dir):
            folder_path = os.path.join(image_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # 加載對應的 GT-xxxxx.csv 文件
            csv_file = os.path.join(folder_path, f"GT-{folder_name}.csv")
            annotations = pd.read_csv(csv_file, sep=";")

            # 保存每張圖像的信息
            for _, row in annotations.iterrows():
                file_path = os.path.join(folder_path, row["Filename"])
                class_id = row["ClassId"]
                self.data.append((file_path, class_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, class_id = self.data[idx]
        image = Image.open(file_path).convert("RGB")  # 加載圖像

        if self.transform:
            image = self.transform(image)

        return image, class_id


class TestDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        root_dir: 測試圖像所在目錄
        csv_file: 包含標籤和 ROI 的 CSV 文件路徑
        transform: 圖像轉換操作
        """
        self.root_dir = root_dir
        self.transform = transform

        # 讀取 CSV 文件
        self.annotations = pd.read_csv(csv_file, sep=";")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        file_path = os.path.join(self.root_dir, row["Filename"])
        class_id = row["ClassId"]  # 類別標籤

        # 加載圖像
        image = Image.open(file_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, class_id  # 返回圖像和類別標籤

