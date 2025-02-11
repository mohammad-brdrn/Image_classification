# defining the basic image
FROM python:3.10-slim
# working directory
WORKDIR /app

COPY . .
#COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]