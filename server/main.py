"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from server.database import DB_PATH, init_db, iter_tables, USE_MYSQL
from server.face_routes import router as face_router
from server.routes import register_routes
from server.seed import maintain_prototype_flights
from server.vector_service import ensure_face_collection, get_vector_status
from server.admin_routes import router as admin_router


app = FastAPI(
    title="AIoT Flight Face Lookup API",
    version="0.2.0",
    description="Airline booking system with face-recognition check-in.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routers
register_routes(app)
app.include_router(face_router)
app.include_router(admin_router)

WEB_ROOT = Path(__file__).resolve().parent.parent / "web_stage3"
PAGE_ROUTES = {
    "/login": "login.html",
    "/register": "register.html",
    "/search": "search.html",
    "/booking": "booking.html",
    "/seat-map": "seat-map.html",
    "/payment": "payment.html",
    "/confirmation": "confirmation.html",
    "/my-tickets": "my-tickets.html",
    "/checkin": "checkin.html",
    "/lookup": "lookup.html",
    "/register-face": "register-face.html",
    "/kiosk": "kiosk.html",
    "/admin": "admin.html",
}


@app.on_event("startup")
def startup() -> None:
    init_db()
    if not USE_MYSQL:
        maintain_prototype_flights()


@app.get("/health")
def health_check() -> dict:
    tables = list(iter_tables())
    return {
        "status": "ok",
        "service": "aiot-flight-api",
        "db_type": "mysql" if USE_MYSQL else "sqlite",
        "database": {
            "path": str(DB_PATH) if not USE_MYSQL else "mysql",
            "tables": tables,
        },
    }


@app.post("/health/vector")
def vector_health_check() -> dict:
    created = ensure_face_collection()
    status = get_vector_status()
    return {"status": "ok", "created": created, "vector": status}


@app.get("/")
def web_home() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/{page}", include_in_schema=False)
def web_page(page: str) -> FileResponse:
    route = f"/{page}"
    filename = PAGE_ROUTES.get(route)
    if filename:
        return FileResponse(WEB_ROOT / "pages" / filename)
    return FileResponse(WEB_ROOT / "index.html")


# Mount models folder so browser can load ONNX models
app.mount("/models", StaticFiles(directory=str(Path(__file__).resolve().parent.parent / "models")), name="models")

# Mount static assets after API and page routes so app URLs are not swallowed.
app.mount("/", StaticFiles(directory=str(WEB_ROOT), html=True), name="static")
