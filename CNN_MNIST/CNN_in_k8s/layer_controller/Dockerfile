# 使用 Python slim 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

COPY * /app/

# 安装必要依赖
RUN pip install --no-cache-dir numpy==1.24.4 flask kubernetes requests torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# 指定容器启动命令
CMD ["python", "layer_controller.py"]