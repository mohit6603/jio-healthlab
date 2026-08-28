#!/usr/bin/env sh
# AI service entrypoint.
#
# Model weights are NOT downloaded at build time -- they land in the mounted
# Hugging Face cache volume on first use, so the image stays small and the
# container becomes healthy immediately.
set -e

echo "[entrypoint] HF cache: ${HF_HOME:-<default>}"
echo "[entrypoint] embedding model: ${EMBEDDING_MODEL:-<default>}"
echo "[entrypoint] llm provider: ${LLM_PROVIDER:-local}"

exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${AI_SERVICE_PORT:-8001}" \
  --proxy-headers \
  --forwarded-allow-ips '*'
