FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

COPY cli.py .
COPY engine/ engine/
COPY config/ config/
COPY dictionary/ dictionary/
COPY tests/ tests/

RUN mkdir -p runtime/multivar_anomaly runtime/logs/cli \
    && chgrp -R 0 /app \
    && chmod -R g=u /app

USER 1001

ENTRYPOINT ["python", "cli.py"]
CMD ["run-multivar-anomaly", "oracle"]
