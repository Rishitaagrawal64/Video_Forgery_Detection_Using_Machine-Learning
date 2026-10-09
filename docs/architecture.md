# Backend Architecture (Sprint 2 deliverable)

```
 React UI (Sachin)
      │  HTTP/JSON
      ▼
 ┌────────────────────────── FastAPI (app/main.py) ──────────────────────────┐
 │ routes/videos.py    upload · analyze · jobs · delete                       │
 │ routes/health.py                                                           │
 └───────────┬───────────────────────────────┬────────────────────────────────┘
             │ BackgroundTasks               │ read/write job state
             ▼                               ▼
   services/pipeline.py              services/job_store.py  ──► (later) PostgreSQL/MongoDB (Tapesh)
     │ 1. frame_extractor.py (OpenCV: uniform / fps sampling)
     │ 2. preprocessing.py   (resize → [face crop] → normalize → batch)
     │ 3. models/base.BasePredictor.predict(batch)  ◄── CNN from Rishita via registry.py
     ▼
   AnalysisResult {label, forged_probability, frame_scores, video info}
```

## Design decisions

- **Layered:** routes are thin; all logic lives in `services/` so it can be reused by the CLI and tests.
- **Pluggable model:** the only coupling to the ML work is `BasePredictor.predict_frames(batch) -> (N,)`.
  Batches are channels-last float32 `(N, 224, 224, 3)` (Keras/TF layout; transpose for PyTorch).
- **Async job model:** upload returns immediately; analysis runs in a background task; the client polls
  `GET /api/v1/jobs/{id}`. Statuses: `uploaded → processing → completed | failed`.
- **Storage seam:** `JobStore` is the single persistence interface (in-memory now). Replacing it with a DB
  repository requires no route changes.
- **Uniform sampling** (default 16 frames) covers the whole clip, which suits video-level classification;
  fps-based sampling is available for temporal models.
- **Video-level score** = mean of per-frame forged probabilities; threshold 0.5.
- **Safety:** extension allow-list, streamed size limit (default 200 MB), corrupt files mark the job `failed`.

## Known limits / next steps (from 19/10/26)

- Replace in-memory store with DB (Tapesh) and move long jobs to a worker queue if needed.
- Integrate the trained CNN; tune frame count/size/normalization to match training.
- Add auth/rate limiting before public deployment.
