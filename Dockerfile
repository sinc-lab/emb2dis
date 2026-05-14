FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

WORKDIR /app

COPY requirements-container.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 \
 && pip install -r requirements-container.txt

COPY src/ /app/src/
COPY config/ /app/config/
COPY model/ProtT5/config.yaml /app/model/ProtT5/config.yaml
COPY model/ProtT5/weights.pk /app/model/ProtT5/weights.pk
COPY predict_disorder.py /app/

ENTRYPOINT ["python", "/app/predict_disorder.py", \
            "--model", "ProtT5", \
            "--device", "cpu", \
            "--caid", \
            "--threads", "4"]
CMD ["--help"]
