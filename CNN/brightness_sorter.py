import os
import pandas as pd
from PIL import Image
from tqdm import tqdm


class BrightnessSorter:
    def __init__(self, input_dir, output_csv, csv_file=None):
        """
        input_dir: 包含圖像的根目錄
        output_csv: 排序後保存的 CSV 路徑
        csv_file: （可選）包含圖片類別信息的 CSV 文件
        """
        self.input_dir = input_dir
        self.output_csv = output_csv
        self.csv_file = csv_file
        self.sorted_data = []

    def calculate_brightness(self, file_path):
        """
        計算圖像的平均亮度
        """
        image = Image.open(file_path).convert("L")  # 灰度轉換
        pixel_values = list(image.getdata())  # 轉換為列表
        return sum(pixel_values) / len(pixel_values)  # 計算平均值

    def process_and_sort(self, has_subdirs=True):
        """
        遍歷所有圖像文件，計算每張圖片的亮度並排序
        has_subdirs: 是否有子目錄
        """
        data = []
        all_files = []

        if has_subdirs:
            # 如果有子目錄，遍歷每個類別目錄
            for class_dir in os.listdir(self.input_dir):
                class_path = os.path.join(self.input_dir, class_dir)
                if not os.path.isdir(class_path):  # 確保是子目錄
                    continue
                for file in os.listdir(class_path):
                    if file.lower().endswith('.ppm'):  # 僅處理 ppm 格式的圖片
                        file_path = os.path.join(class_path, file)
                        all_files.append((file_path, class_dir))
        else:
            # 如果沒有子目錄，根據 CSV 文件中的信息處理圖片
            if not self.csv_file:
                raise ValueError("CSV file must be provided when there are no subdirectories.")
            annotations = pd.read_csv(self.csv_file, sep=";")
            for _, row in annotations.iterrows():
                file_path = os.path.join(self.input_dir, row["Filename"])
                class_id = row["ClassId"]
                all_files.append((file_path, class_id))

        # 遍歷所有文件並計算亮度，顯示進度條
        for file_path, class_id in tqdm(all_files, desc="Processing Images", unit="file"):
            brightness = self.calculate_brightness(file_path)
            data.append((file_path, brightness, class_id))

        # 根據亮度排序
        self.sorted_data = sorted(data, key=lambda x: x[1])

    def save_to_csv(self):
        """
        保存排序結果到 CSV 文件
        """
        df = pd.DataFrame(self.sorted_data, columns=["FilePath", "Brightness", "Class"])
        df.to_csv(self.output_csv, index=False)
        print(f"排序結果已保存到 {self.output_csv}")


if __name__ == "__main__":
    # 訓練集（有子目錄）
    train_sorter = BrightnessSorter(input_dir="dataset/Final_Training/Images", output_csv="CSV/sorted_train.csv")
    train_sorter.process_and_sort(has_subdirs=True)
    train_sorter.save_to_csv()

    # 測試集（無子目錄，但有 CSV 文件記錄類別）
    test_sorter = BrightnessSorter(
        input_dir="dataset/Final_Test/Images",
        output_csv="CSV/sorted_test.csv",
        csv_file="dataset/Final_Test/Images/GT-final_test.csv",
    )
    test_sorter.process_and_sort(has_subdirs=False)
    test_sorter.save_to_csv()
