from fastapi import FastAPI
from road_damage_tracking.models import RoadDamageDetector
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Road Damage Detection API",
              description="API for Road Damage Detection",
              version="0.1.0",
              docs_url="/",
              redoc_url=None,
              openapi_url="/openapi.json",
              contact={"name": "Brice AKPALO, Gédéon GBEDONOU"})


class Request(BaseModel):
    source: str
    height: Optional[int]
    width: Optional[int]


@app.post("/detect")
def detect(request: Request):
    detector = RoadDamageDetector(request.source, request.height, request.width)

    return dict([response for response in detector()])
