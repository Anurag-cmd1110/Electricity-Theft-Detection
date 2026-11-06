import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes.meters import router as meters_router

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="GridGuard Core API",
    description="AI-powered electricity theft detection and feeder energy balancing system",
    version="1.0.0",
)

# Robust CORS Configuration: covers localhost and 127.0.0.1 across any Vite/React port
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handler to guarantee CORS headers on unhandled 500 errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Server Error at {request.url.path}: {exc}", exc_info=True)
    origin = request.headers.get("origin", "*")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
        headers={
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
        },
    )


@app.get("/")
def home():
    return {
        "message": "GridGuard Core API is running",
        "project": "GridGuard Electricity Intelligence System",
        "status": "success",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# Include Meters Router
app.include_router(meters_router)