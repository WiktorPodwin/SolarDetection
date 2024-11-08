from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.operations.map_operations import GoogleMapsImageFetcher
from api.operations.data_operations import GSOperations

maps_fetcher = GoogleMapsImageFetcher(api_key=os.getenv("GOOGLE_MAPS_API_KEY"))


class ImageResponse(BaseModel):
    message: str
    bytes: bytes


class ImageRouter:
    def __init__(self, project_id: str, api_key: str, bucket_name: str):
        self.maps_fetcher = GoogleMapsImageFetcher(api_key=api_key)
        self.storage_client = GSOperations(project_id, bucket_name)

    def get_router(self) -> APIRouter:
        """Creates and returns an APIRouter with all the route definitions."""
        router = APIRouter()

        @router.get("/google-maps/{location}", response_model=ImageResponse)
        async def fetch_google_maps_image(location: str):
            return maps_fetcher.fetch_image(location)

        @router.get("/bucket/{image_id}", response_model=ImageResponse)
        async def get_image_from_bucket(image_id: str):
            image = self.storage_client.list_files(image_id)[0]
            return await image

        @router.post("/bucket/{image_id}", response_model=ImageResponse)
        async def upload_image_to_bucket(image_id: str, file: UploadFile):
            image = self.storage_client.upload_file(file, image_id)
            return await image

        @router.put("/bucket/{image_id}", response_model=ImageResponse)
        async def update_image_in_bucket(image_id: str, file: UploadFile):
            image = self.storage_client.update_file(image_id, file)
            return await image

        @router.delete("/bucket/{image_id}", response_model=ImageResponse)
        async def delete_image_from_bucket(image_id: str):
            image = self.storage_client.delete_file(image_id)
            return await image
