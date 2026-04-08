FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY engine/ engine/
COPY config/pipeline_config.yaml config/pipeline_config.yaml
COPY generate_data.py .
COPY cli.py .

RUN mkdir -p logs models

ENTRYPOINT ["python", "cli.py"]
CMD ["run"]
