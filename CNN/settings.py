# 路標分類設定
NUM_CLASSES = 43  # 路標的類別數
INPUT_CHANNELS = 3  # RGB 圖像
INPUT_HEIGHT = 32  # 圖像高度
INPUT_WIDTH = 32  # 圖像寬度

# 訓練參數
BATCH_SIZE = 64  # 每次訓練的批次大小
LR = 0.001  # 學習率
GAMMA = 0.9  # 折扣因子
EPOCHS = 20  # 訓練週期數
MAX_MEMORY = 10000  # 記憶體大小（僅強化學習用）
EPSILON = 0.1  # ε 探索率（僅強化學習用）
EPS_RANGE = 0.05  # 探索範圍
PATIENCE = 5  # 設置耐心次數
TRAIN_LIMIT_START = 0  # 只保留 sorted_train.csv 前 TRAIN_LIMIT_START% 到 TRAIN_LIMIT_END% 行
TRAIN_LIMIT_END = 30
TEST_LIMIT_START = 70  # 只保留 sorted_test.csv 前 TEST_LIMIT_START% 到 TEST_LIMIT_END% 行
TEST_LIMIT_END = 100
IS_TRAIN = True  # 預設為訓練模型

# 數據集路徑
TRAIN_DATASET_DIR = "dataset/Train/50"  # 訓練資料路徑
TEST_DATASET_DIR = "dataset/Test/50/Images"  # 測試資料路徑
TEST_CSV_FILE = "dataset/Final_Test/Images/GT-final_test.csv"  # 測試資料標籤表路徑
OUTPUT_CSV = "CSV/predictions.csv"  # 預測結果路徑
CHECKPOINT_PATH = "model/checkpoint.pth"  # 檢查點路徑
MODEL_LOAD_PATH = "model/traffic_sign_cnn.pth"  # 加載模型路徑
# SORTED_TRAIN_CSV_PATH = "CSV/sorted_train.csv"  # 依照亮度排序原始訓練資料之結果表路徑
# SORTED_TEST_CSV_PATH = "CSV/sorted_test.csv"  # 依照亮度排序原始測試資料之結果表路徑
