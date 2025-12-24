FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy Model dari folder Sprint 3 (Kita akan mount via Docker Compose, tapi copy juga oke)
# Pastikan path ini sesuai nanti saat build context

# Copy Script App
COPY . .

# Expose port App (5000) dan Prometheus Metrics (8000)
EXPOSE 5000 8000

CMD ["python", "app.py"]