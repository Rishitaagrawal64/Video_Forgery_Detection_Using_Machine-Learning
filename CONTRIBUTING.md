# Contributing

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
cp .env.example .env
pip install pre-commit && pre-commit install           # optional: lint on every commit
cd frontend && npm ci
```

## Workflow
1. Branch from `main`: `git checkout -b feat/short-description`.
2. Keep commits small and focused. Messages follow `type(scope): summary`, e.g. `fix(api): ...`,
   `feat(eval): ...`, `docs: ...`, `test: ...`, `ci: ...`, `chore: ...`.
3. Before pushing:
   ```bash
   ruff check .
   pytest -q --cov=app
   cd frontend && npm run lint && npm run build
   ```
4. Open a pull request. CI runs the same checks.

## Guidelines
- New backend behavior needs a test in `tests/`. Tests must not need model weights or a GPU.
- Settings live in `app/core/config.py` and are read from `FORGERY_*` environment variables. Document new ones in `.env.example`.
- The serving pipeline must match training (frame size, normalization, face cropping, threshold). See `docs/model_card.md`.
- Never commit videos, model weights (`*.pt`, `*.keras`), or `.env` files.
