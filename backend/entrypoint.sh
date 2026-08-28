#!/usr/bin/env sh
# Container entrypoint: apply migrations, optionally seed, then serve.
set -e

echo "[entrypoint] applying database migrations"
alembic upgrade head

if [ "${SEED_ON_STARTUP:-true}" = "true" ]; then
  echo "[entrypoint] seeding demo data (idempotent)"
  python -m app.seed
fi

echo "[entrypoint] starting uvicorn on 0.0.0.0:${PORT:-8000}"
exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --proxy-headers \
  --forwarded-allow-ips '*'
