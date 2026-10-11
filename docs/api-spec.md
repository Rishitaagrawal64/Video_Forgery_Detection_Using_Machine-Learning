# API Specification (v1)

Interactive docs: `GET /docs` (Swagger) · `GET /openapi.json`

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness + active predictor |
| POST | `/api/v1/videos` | Upload video (multipart field `file`). 201 → `{video_id, filename, status}` |
| POST | `/api/v1/videos/{video_id}/analyze` | Start analysis. Body optional: `{num_frames: 1-128, detect_faces: bool}`. 202 → job |
| GET | `/api/v1/jobs/{video_id}` | Job status and result |
| GET | `/api/v1/jobs` | List all jobs |
| DELETE | `/api/v1/videos/{video_id}` | Delete video and extracted frames. 204 |

## Errors

| Code | When |
|---|---|
| 404 | Unknown `video_id` |
| 409 | Analyze requested while already processing |
| 410 | Analyze requested after the file was auto-deleted (re-upload) |
| 413 | Upload larger than `FORGERY_MAX_UPLOAD_MB` |
| 415 | Extension not in `.mp4 .avi .mov .mkv .webm` |
| 422 | Request validation error |

## Job response

```json
{
  "video_id": "3f9c…",
  "filename": "clip.mp4",
  "status": "completed",
  "created_at": "2026-10-09T12:00:00Z",
  "result": {
    "label": "REAL",
    "forged_probability": 0.31,
    "frame_scores": [0.28, 0.35, "..."],
    "frames_analyzed": 16,
    "video": {"width": 1280, "height": 720, "fps": 30.0, "frame_count": 300, "duration_sec": 10.0},
    "model": "heuristic-placeholder",
    "processing_sec": 0.42
  },
  "error": null
}
```

`label` is `REAL` or `AI_GENERATED_FORGED`. On failure `status` is `failed` and `error` has the reason.
