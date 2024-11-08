from typing import List
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src import depth_pro


class DepthResponse(BaseModel):
    message: str
    image_id: str
    depth_areas: List[List[float]]


class DepthRouter:
    def __init__(self) -> None:
        pass

    def get_router(self) -> APIRouter:
        """Creates and returns an APIRouter with all the route definitions."""
        router = APIRouter()

        router.get("/depth/{image_id}", response_model=DepthResponse)

        async def image_depth_detection(image_id: str):
            """Uses depth_pro package to detect depth in image."""
            model, transform = depth_pro.create_model_and_transforms(
                # see depth_pro.cli.run
                # device=get_torch_device(),
                # precision=torch.half,
            )
            model.eval()

            image, _, f_px = depth_pro.load_rgb("data/output/example.png")
            image = transform(image)

            print("Running inference.")
            prediction = model.infer(image, f_px=f_px)
            depth = prediction["depth"]
            focallength_px = prediction["focallength_px"]
            inverse_depth = 1 / depth
            # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
            max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
            min_invdepth_vizu = max(1 / 250, inverse_depth.min())
            inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
                max_invdepth_vizu - min_invdepth_vizu
            )
            return JSONResponse(
                {
                    "message": "success",
                    "image_id": image_id,
                    "depth_areas": [[0.0]],
                    "depth": depth,
                    "focallength_px": focallength_px,
                    "inverse_depth_normalized": inverse_depth_normalized,
                }
            )

        return router
