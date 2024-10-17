import os
from fastapi import APIRouter

from api.earth_engine import GEEImageFetcher
from api.earth_engine.engine import GoogleMapsImageFetcher

router = APIRouter()
# fetcher = GEEImageFetcher()
maps_fetcher = GoogleMapsImageFetcher(api_key=os.getenv("GOOGLE_MAPS_API_KEY"))

# router.include_router(health.router, tags=["health"])
# router.include_router(predict.router, tags=["predict"])


@router.get("/")
async def read_root():
    return {"message": "Welcome to the Earth Engine API!"}


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.get("/predict")
async def predict():
    return {"message": "Predicting..."}


@router.get("/fetch_images")
async def fetch_images():
    # fetcher.fetch_images()
    return {"message": "Fetching images..."}


# @router.get("/satellite-map")
# def get_gee_image():
#     # Get the first image from the collection (choose a less cloudy image)
#     image = fetcher.dataset.first()

#     # Visualization parameters for true color (RGB bands)
#     vis_params = {
#         'bands': ['B4', 'B3', 'B2'],  # Red, Green, Blue
#         'min': 0,
#         'max': 3000,
#         'gamma': 1.4
#     }

#     # Get the URL for the tile layer
#     map_id_dict = image.getMapId(vis_params)
#     tile_url = map_id_dict['tile_fetcher'].url_format
#     print(tile_url)
#     return {"tile_url": tile_url}


@router.get("/image")
async def export_image(location: str):
    print("location", location)
    return {"message": "success", "bytes": maps_fetcher.fetch_image(location)}
