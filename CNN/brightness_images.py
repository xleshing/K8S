import os
import pandas as pd
from PIL import Image, ImageEnhance
from shutil import copyfile
from tqdm import tqdm
from settings import *


class BrightnessSorter:
    def __init__(self, input_dir, output_dir, csv_file=None):
        """
        input_dir: 包含圖像的根目錄
        output_dir: 調整後的圖像保存目錄
        csv_file: （可選）包含圖片類別信息的 CSV 文件
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.csv_file = csv_file

    def adjust_brightness(self, file_path, target_brightness):
        """
        調整圖像亮度至目標值
        """
        image = Image.open(file_path).convert("L")  # 灰度轉換
        pixel_values = list(image.getdata())
        original_brightness = sum(pixel_values) / len(pixel_values)  # 原亮度

        # 計算調整比例
        scale_factor = target_brightness / original_brightness
        enhancer = ImageEnhance.Brightness(image)
        adjusted_image = enhancer.enhance(scale_factor)
        return adjusted_image

    def process_and_save(self, target_brightness_levels, has_subdirs=True):
        """
        處理並保存所有圖像和相關的 CSV 文件
        target_brightness_levels: 目標亮度列表
        has_subdirs: 是否有子目錄
        """
        all_files = []
        csv_files = []

        if has_subdirs:
            # 如果有子目錄，遍歷每個類別目錄
            for class_dir in os.listdir(self.input_dir):
                class_path = os.path.join(self.input_dir, class_dir)
                if not os.path.isdir(class_path):  # 確保是子目錄
                    continue

                # 搜集該目錄內的所有圖片文件
                for file in os.listdir(class_path):
                    if file.lower().endswith('.ppm'):  # 僅處理 ppm 格式的圖片
                        file_path = os.path.join(class_path, file)
                        all_files.append((file_path, class_dir))
                    elif file.lower().endswith('.csv'):  # 保存 CSV 文件
                        csv_files.append((os.path.join(class_path, file), class_dir))
        else:
            # 如果沒有子目錄，根據 CSV 文件中的信息處理圖片
            if not self.csv_file:
                raise ValueError("CSV file must be provided when there are no subdirectories.")
            annotations = pd.read_csv(self.csv_file, sep=";")
            for _, row in annotations.iterrows():
                file_path = os.path.join(self.input_dir, row["Filename"])
                class_id = row["ClassId"]
                all_files.append((file_path, class_id))

            # 添加 CSV 文件
            csv_files.append((self.csv_file, ""))

        # 遍歷所有文件並進行亮度調整
        for file_path, class_id in tqdm(all_files, desc="Processing Images", unit="file"):
            for target_brightness in target_brightness_levels:
                output_subdir = os.path.join(self.output_dir, f"{target_brightness}/Images")
                os.makedirs(output_subdir, exist_ok=True)

                if has_subdirs:
                    output_subdir = os.path.join(output_subdir, class_id)
                    os.makedirs(output_subdir, exist_ok=True)

                adjusted_image = self.adjust_brightness(file_path, target_brightness)
                file_name = os.path.basename(file_path)
                output_path = os.path.join(output_subdir, file_name)
                adjusted_image.save(output_path)

        # 複製 CSV 文件
        for csv_path, class_id in csv_files:
            for target_brightness in target_brightness_levels:
                output_subdir = os.path.join(self.output_dir, f"{target_brightness}/Images")
                os.makedirs(output_subdir, exist_ok=True)

                if has_subdirs:
                    output_subdir = os.path.join(output_subdir, class_id)
                    os.makedirs(output_subdir, exist_ok=True)

                csv_output_path = os.path.join(output_subdir, os.path.basename(csv_path))
                copyfile(csv_path, csv_output_path)


if __name__ == "__main__":
    # 目標亮度級別
    target_brightness_levels = [10, 50, 100, 150, 200, 250]

    # 處理訓練集（有子目錄）
    train_sorter = BrightnessSorter(
        input_dir="dataset/Final_Training/Images",
        output_dir="dataset/Train",
    )
    train_sorter.process_and_save(target_brightness_levels, has_subdirs=True)

    # 處理測試集（無子目錄，但有 CSV 文件記錄類別）
    test_sorter = BrightnessSorter(
        input_dir="dataset/Final_Test/Images",
        output_dir="dataset/Test",
        csv_file=TEST_CSV_FILE,
    )
    test_sorter.process_and_save(target_brightness_levels, has_subdirs=False)
