# carzam-api

FastAPI backend for the Carzam iOS app. Auth (Google + Apple), car CRUD, audio recognize endpoint, discoveries.

## Local dev

From the monorepo root:

```bash
cd apps/carzam-api
cp .env.example .env  # fill in the blanks
uv pip install -e .[dev]
uvicorn app.main:app --reload --port 8000
```

The lifespan hook will:
1. Create tables in the configured Postgres (idempotent).
2. Seed `car_classes` from `app/seeds.py` (upsert).
3. Load the PyTorch checkpoint into memory.

OpenAPI docs at http://localhost:8000/docs.

## Deploy to Coolify

1. Push this repo to GitHub.
2. In Coolify: **Servers → carzam-server → + New Project → Carzam**.
3. **+ New Resource → PostgreSQL 16** in the project. Coolify exposes `DATABASE_URL` to other resources in the same project.
4. **+ New Resource → Application → GitHub repo**:
   - Build pack: **Dockerfile**
   - Dockerfile path: `apps/carzam-api/Dockerfile`
   - Base directory: `/` (the monorepo root — the Dockerfile copies both `src/carzam` and `apps/carzam-api/`)
5. Set environment variables (copy from `.env.example`). For `DATABASE_URL`, change `postgresql+asyncpg://...` to use the Coolify-injected hostname (Coolify gives you `DATABASE_URL=postgresql://...` — prefix the driver: `postgresql+asyncpg://...`).
6. Add a domain (e.g. `api.carzam.app`).
7. **Deploy.**

## Routes

| Method | Path | Auth | What it does |
|--------|------|------|--------------|
| GET    | `/health` | — | DB-less liveness check |
| POST   | `/auth/google` | — | Verify Google ID token → JWT |
| POST   | `/auth/apple`  | — | Verify Apple ID token → JWT |
| GET    | `/me` | JWT | Current user |
| GET    | `/me/cars` | JWT | List my owned cars |
| POST   | `/me/cars` | JWT | Add a car |
| DELETE | `/me/cars/{id}` | JWT | Remove a car |
| POST   | `/me/cars/{id}/clips` | JWT | Upload audio of a car you own |
| POST   | `/recognize` | JWT | Multipart WAV → top-1 + top-3 + state + discovery_unlocked |
| POST   | `/clips/{id}/submit-as-new-car` | JWT | Tag a low-confidence clip with user-supplied specs |
| GET    | `/me/discoveries` | JWT | Classes I've matched, with first-found timestamps |
| GET    | `/me/stats` | JWT | n_discovered / n_total / completion %, total recognitions |
| GET    | `/car-classes` | — | All classes the model knows about |

## Notes

- Inference loads the checkpoint **once at startup**. Single-process; ~320 MB resident.
- Audio always lands in R2, even on prediction failure — that's the data flywheel.
- `discoveries` is denormalized for fast Found-grid reads; updated transactionally with `clips` insert.
