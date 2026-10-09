# Sumit Todwal — Backend/Pipeline Lead: Task Tracker

Source: Project Abstract Forms 1 & 2 (session 2026-27). Status as of 09/10/26.

| Dates | Task (Form 2) | Status | Where |
|---|---|---|---|
| 10/08/26 – 13/09/26 | Set up preprocessing pipeline and frame extraction module | Done | `app/services/frame_extractor.py`, `preprocessing.py`, `scripts/extract_frames.py`, `tests/test_frame_extractor.py`, `tests/test_preprocessing.py` |
| 14/09/26 – 18/10/26 | Design backend architecture and API structure | Done | `docs/architecture.md`, `docs/api-spec.md`, `app/` layout, `app/schemas/` |
| 19/10/26 – 22/11/26 | Develop backend APIs and integrate model inference module | Started early | `app/api/routes/videos.py`, `app/models/` (interface + placeholder; real CNN pending from Rishita) |
| 23/11/26 – 27/12/26 | Frontend–backend API linking and endpoints | Pending | CORS enabled; contract in `docs/api-spec.md` |
| 28/12/26 – 31/01/27 | Integrate backend, DB and model into one pipeline | Pending | `JobStore` is the DB seam (Tapesh) |
| 01/02/27 – 10/03/27 | Backend testing, bug fixing, deployment | Pending | `Dockerfile`, `tests/` as starting point |

## Dependencies on teammates

- **Rishita:** trained CNN + expected input size/normalization → plug into `app/models/registry.py`.
- **Sachin:** consume `docs/api-spec.md` (upload → analyze → poll job).
- **Tapesh:** replace `JobStore` with PostgreSQL/MongoDB; Docker/K8s deployment.
