import os
import pandas as pd
from PIL import Image

class ImageExtractor:
    def __init__(self, csv_file, output_dir):
        """
        csv_file: 包含圖像排序結果的 CSV 文件
        output_dir: 保存提取 JPG 圖像的目錄
        """
        self.csv_file = csv_file
        self.output_dir = output_dir

        # 創建輸出目錄
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_and_convert(self, step=2000):
        """
        提取每 step 張圖像並轉換為 JPG 格式
        step: 提取間隔
        """
        # 讀取 CSV 文件
        df = pd.read_csv(self.csv_file)

        # 獲取每 step 張圖像的文件路徑
        selected_files = df.iloc[::step]["FilePath"].tolist()

        # 轉換並保存圖像
        for i, file_path in enumerate(selected_files, start=1):
            try:
                with Image.open(file_path) as img:
                    output_path = os.path.join(self.output_dir, f"image_{i}.jpg")
                    img.convert("RGB").save(output_path, "JPEG")
                    print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    # 設定提取參數
    train_extractor = ImageExtractor(csv_file="sorted_train.csv", output_dir="train_jpgs")
    train_extractor.extract_and_convert(step=2000)

    test_extractor = ImageExtractor(csv_file="sorted_test.csv", output_dir="test_jpgs")
    test_extractor.extract_and_convert(step=2000)
