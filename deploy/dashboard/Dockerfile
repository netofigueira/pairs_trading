FROM python:3.14-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY artifacts ./artifacts
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "quant_pairs.dashboard_api:app", "--host", "0.0.0.0", "--port", "8000"]
