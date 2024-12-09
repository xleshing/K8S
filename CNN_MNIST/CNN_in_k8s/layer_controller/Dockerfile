# 使用 Python slim 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制 LayerController 文件
COPY layer_controller.py /app/

# 安装必要依赖
RUN pip install flask kubernetes requests

# 指定容器启动命令
CMD ["python", "layer_controller.py"]
