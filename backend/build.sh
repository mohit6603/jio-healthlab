#!/usr/bin/env bash
# Render build script for the backend
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Seed demo data
python -m app.seed
