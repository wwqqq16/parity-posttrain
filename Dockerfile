FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY enterprise_eval ./enterprise_eval
COPY scripts ./scripts

RUN python -m pip install --no-cache-dir ".[distributed]"

EXPOSE 8000 50051

CMD ["python", "-m", "uvicorn", \
     "enterprise_eval.distributed.kafka_api:app", \
     "--host", "0.0.0.0", "--port", "8000"]
