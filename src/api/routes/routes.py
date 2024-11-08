from typing import List
from fastapi import APIRouter
from pydantic import BaseModel

from config import BaseConfig
from .image import ImageRouter
from .depth import DepthRouter

router = APIRouter()

router.include_router(
    ImageRouter(
        project_id=BaseConfig.PROJECT_ID,
        api_key=BaseConfig.GOOGLE_MAPS_API_KEY,
        bucket_name=BaseConfig.BUCKET_NAME,
    ).get_router(),
    prefix="/image",
)

router.include_router(DepthRouter().get_router())


class DepthResponse(BaseModel):
    message: str
    image_id: str
    depth_areas: List[List[float]]


class PredictionResponse(BaseModel):
    message: str
    prediction: str


class HealthResponse(BaseModel):
    status: str

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify the service status."""
    return {"status": "ok"}


@router.get("/predict/house/{image_id}", response_model=PredictionResponse)
async def predict_house(image_id: str):
    """Predict details of a house based on the image ID."""
    prediction_result = "House predicted successfully for ID: " + image_id
    return {"message": "Prediction complete", "prediction": prediction_result}


@router.get("/predict/field/{image_id}", response_model=PredictionResponse)
async def predict_field(image_id: str):
    """Predict details of a field based on the image ID."""
    prediction_result = "Field predicted successfully for ID: " + image_id
    return {"message": "Prediction complete", "prediction": prediction_result}
