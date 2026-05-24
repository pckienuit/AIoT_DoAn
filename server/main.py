from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.database import DB_PATH, init_db, iter_tables
from server.face_routes import router as face_router
from server.routes import router as api_router
from server.vector_service import ensure_face_collection, get_vector_status

app = FastAPI(
    title="AIoT Flight Face Lookup API",
    version="0.1.0",
    description="Prototype backend for flight information lookup via face recognition.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)
app.include_router(face_router)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health_check() -> dict[str, object]:
    tables = list(iter_tables())
    return {
        "status": "ok",
        "service": "flight-face-api",
        "database": {
            "path": str(DB_PATH),
            "tables": tables,
        },
    }


@app.post("/health/vector")
def vector_health_check() -> dict[str, object]:
    created = ensure_face_collection()
    status = get_vector_status()
    return {"status": "ok", "created": created, "vector": status}
