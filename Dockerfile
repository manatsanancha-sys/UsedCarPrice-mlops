FROM python:3.12-slim
WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/__init__.py src/__init__.py
COPY src/data_cleaning.py src/data_cleaning.py
COPY src/api.py src/api.py
COPY static/ static/
COPY model_export/ model_export/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
