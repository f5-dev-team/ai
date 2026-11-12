from fastapi import FastAPI

from src.service.api.health import router as health_router


app = FastAPI(
    title="Scholarship AI Service",
    description="장학금 추천 시스템 AI Service API",
    version="0.1.0",
)

app.include_router(health_router)