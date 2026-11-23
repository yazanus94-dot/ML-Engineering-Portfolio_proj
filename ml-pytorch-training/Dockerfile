FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt requirements-serving.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-serving.txt

COPY pytorch_lab ./pytorch_lab
RUN mkdir -p reports

EXPOSE 8000
# Mount a trained checkpoint at /app/reports/checkpoint.pt before running.
CMD ["python", "-m", "pytorch_lab.serve", "--checkpoint", "reports/checkpoint.pt"]
