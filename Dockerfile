FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY models ./models
ENV PYTHONPATH=/app/src MODEL_PATH=/app/models/model.joblib
EXPOSE 8000
CMD ["uvicorn", "churn_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
