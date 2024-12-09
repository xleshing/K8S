FROM python:3.9-slim

WORKDIR /app

COPY layer_service.py /app/

RUN pip install flask torch

CMD ["python", "layer_service.py"]
