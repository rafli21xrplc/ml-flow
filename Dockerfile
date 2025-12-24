FROM python:3.9-slim

# Install dependencies dasar
RUN pip install mlflow>=2.0 scikit-learn pandas cloudpickle

# Copy model dari folder lokal ke dalam image
COPY model_lokal_output /opt/ml/model

# Set environment variable agar MLflow tahu lokasi model
ENV MLFLOW_MODEL_DIR=/opt/ml/model

# Perintah default untuk serving
CMD ["mlflow", "models", "serve", "-m", "/opt/ml/model", "-h", "0.0.0.0", "-p", "8080", "--env-manager=local"]