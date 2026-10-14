# Database design (FR-005)

The backend stores **metadata and prediction results only. Video files are never stored** (NFR-005 deletes them after processing).

Enable it with `FORGERY_DATABASE_URL`:
- unset -> in-memory store (jobs vanish on restart; used by the default tests)
- `sqlite:///data/forgery.db` -> local file, good for development
- `postgresql+psycopg://forgery:<password>@db:5432/forgery` -> Docker/PostgreSQL (set by `docker-compose.yml`)

Tables are created automatically on startup (`app/services/db_job_store.py`).

## videos
| column | type | meaning |
|---|---|---|
| id | text, primary key | `video_id` returned by the API |
| filename | text | original file name |
| path | text | temporary upload path (file is deleted after processing) |
| created_at | timestamptz | upload time |

## predictions
One row per analysis run, so re-analysing a video keeps the earlier results as history.
| column | type | meaning |
|---|---|---|
| id | serial, primary key | |
| video_id | text, foreign key -> videos.id | deleted together with the video |
| status | text | processing / completed / failed |
| label | text | REAL or AI_GENERATED_FORGED |
| forged_probability | float | 0..1 |
| frame_scores | json(b) | per-frame probabilities |
| frames_analyzed | int | |
| video_info | json(b) | width, height, fps, frame_count, duration_sec |
| model_name | text | e.g. torch-cnn |
| processing_sec | float | for NFR-001 (< 60 s) |
| error | text | message if failed |
| created_at / finished_at | timestamptz | |

The API shows the latest prediction of each video. History: `GET /api/v1/jobs?limit=20&offset=0` (newest first).

## Run with Docker
1. Create a `.env` file next to `docker-compose.yml` containing `POSTGRES_PASSWORD=choose_a_password`
2. `docker compose up --build`
3. Open http://localhost:8000/docs
4. Stop: `docker compose down` (data is kept). Delete the data too: `docker compose down -v`
