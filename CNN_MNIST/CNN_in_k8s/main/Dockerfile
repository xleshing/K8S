# 使用 Python slim 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制 main 程序文件
COPY main.py /app/

# 安装必要依赖
RUN pip install requests

# 指定容器启动命令
CMD ["python", "main.py"]
