# JIO Healthlab

An agentic full end-to-end diagnostics report system rebuilt from the original single-file FastAPI demo.

## What changed

- FastAPI backend with SQLAlchemy models instead of in-memory dictionaries.
- MySQL database support via `DATABASE_URL`.
- Alembic migration for the first production schema.
- Expanded report fields: status, priority, branch, city, doctor, contact details, due times, and notes.
- Dashboard API for totals, urgent work, status mix, test demand, city load, due-soon queue, and recent reports.
- React + Vite frontend with a creative lab operations dashboard.
- Docker Compose setup for local MySQL/API/frontend.
- AWS EC2 + RDS deployment guide.

## Local Docker Run

1. Copy the environment sample:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Start the stack:

   ```powershell
   docker compose up --build
   ```

3. Open:

   - Frontend: http://localhost
   - API docs: http://localhost/docs
   - Backend health: http://localhost/health

The backend container runs `alembic upgrade head`, seeds demo reports, then starts Uvicorn.

## Local Development

Backend:

```powershell
cd backend
pip install -r requirements.txt
alembic upgrade head
python -m app.seed
uvicorn app.main:app --reload
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

The Vite dev server proxies `/api`, `/docs`, `/health`, and `/openapi.json` to `localhost:8000`.

## Database

The default local Docker URL is:

```text
mysql+pymysql://healthlab:healthlab@mysql:3306/jio_healthlab
```

For a local backend outside Docker, use a host reachable from your machine, for example:

```text
mysql+pymysql://healthlab:healthlab@localhost:3306/jio_healthlab
```

Run migrations manually with:

```powershell
cd backend
alembic upgrade head
```

Create a new migration after model changes with:

```powershell
cd backend
alembic revision --autogenerate -m "describe change"
```

## Project Layout

```text
backend/
  app/
    main.py          FastAPI routes
    models.py        SQLAlchemy models
    schemas.py       Pydantic request/response models
    database.py      DB engine/session
    seed.py          demo data loader
  alembic/           migration environment
frontend/
  src/
    App.tsx          React dashboard
    api.ts           API client
deploy/
  aws-ec2-rds.md     AWS live hosting guide
```

## License

MIT License. See [LICENSE](LICENSE).
