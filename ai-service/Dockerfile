FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    # Transformers must never phone home during a container start.
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU-only torch wheel first. The default PyPI wheel bundles CUDA
# and would add ~2.5 GB to an image that only ever runs on CPU here.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.3.0"

COPY requirements.txt .
# torch is already satisfied above; the flag stops pip re-resolving it from PyPI.
RUN pip install --no-cache-dir -r requirements.txt

COPY entrypoint.sh ./
COPY app ./app
COPY knowledge ./knowledge
COPY evaluation ./evaluation

# Unprivileged user; the HF cache volume is mounted at /models.
RUN adduser --disabled-password --gecos "" --uid 10002 aiuser \
    && mkdir -p /models/huggingface /app/app/ml/artifacts \
    && chown -R aiuser:aiuser /app /models
USER aiuser

EXPOSE 8001

HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=5 \
    CMD curl -fsS http://localhost:8001/health || exit 1

CMD ["./entrypoint.sh"]
