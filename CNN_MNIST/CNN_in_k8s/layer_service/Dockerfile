# 使用更輕量級的基礎映像
FROM python:3.9-slim

# 設置工作目錄
WORKDIR /app

# 複製應用代碼
COPY layer_service.py /app/

# 安裝必要的 Python 依賴
RUN pip install --no-cache-dir numpy flask torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# 指定執行命令
CMD ["python", "layer_service.py"]
