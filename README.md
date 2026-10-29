# AI-Generated Video Forgery Detection (SKIT CSE 2026-27)

CNN-based system that analyses video frames and classifies a video as REAL or AI_GENERATED_FORGED.

## Run backend
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
python -m uvicorn app.main:app --reload      # Swagger UI: http://localhost:8000/docs
pytest
```
Docker: `docker build -t forgery-api . && docker run -p 8000:8000 forgery-api`

## Layout
- `app/` FastAPI backend (routes, services, model adapters) - see `docs/architecture.md`
- `docs/api-spec.md` API contract for the frontend
- `docs/TASKS.md` task tracker per Form 2

## Plugging in the trained model
Set `FORGERY_PREDICTOR=torch` (TorchScript) or `keras` and `FORGERY_MODEL_PATH=...`. Match the
training preprocessing via `FORGERY_FRAME_SIZE` and `FORGERY_NORMALIZE_MODE`.

## Run with Docker (API + PostgreSQL)

1. Create a `.env` file in the project root:
   ```
   POSTGRES_PASSWORD=choose_a_password
   ```
2. Start the services:
   ```
   docker compose up --build
   ```
3. Open http://localhost:8000/docs to try the API (upload, analyze, poll the job).

Data is stored in the `pgdata` volume and survives restarts.
To stop: `docker compose down`. To also delete all data: `docker compose down -v`.

Inspect the database:
```
docker compose exec db psql -U forgery -d forgery -c "SELECT id, filename FROM videos;"
```
