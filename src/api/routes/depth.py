import logging
from typing import List

import numpy as np
import PIL.Image
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import BaseConfig
from src import depth_pro
from src.api.utils import get_torch_device
from src.api.operations import GSOperations


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

    def __run(self, image_id: str, save: bool = False, display: bool = False):
        """Runs the depth detection process."""

        # Load model.
        model, transform = depth_pro.create_model_and_transforms(
            device=get_torch_device(),
            precision=torch.half,
        )
        model.eval()

        image_paths = []

        if display:
            plt.ion()
            fig = plt.figure()
            ax_rgb = fig.add_subplot(121)
            ax_disp = fig.add_subplot(122)

        for image_path in tqdm(image_paths):
            # Load image and focal length from exif info (if found.).
            try:
                logging.info("Loading image %s ...", image_path)
                image, _, f_px = depth_pro.load_rgb(image_path)
            except Exception as e:
                logging.error(str(e))
                continue
            # Run prediction. If `f_px` is provided, it is used to estimate the final metric depth,
            # otherwise the model estimates `f_px` to compute the depth metricness.
            prediction = model.infer(transform(image), f_px=f_px)

            # Extract the depth and focal length.
            depth = prediction["depth"].detach().cpu().numpy().squeeze()
            if f_px is not None:
                logging.debug("Focal length (from exif): %.2f", f_px)
            elif prediction["focallength_px"] is not None:
                focallength_px = prediction["focallength_px"].detach().cpu().item()
                logging.info("Estimated focal length: %s", focallength_px)

            inverse_depth = 1 / depth
            # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
            max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
            min_invdepth_vizu = max(1 / 250, inverse_depth.min())
            inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
                max_invdepth_vizu - min_invdepth_vizu
            )

            # Save Depth as npz file.
            if save:
                output_file = f"depth/{image_id}"
                logging.info("Saving depth map to: %s", output_file)
                gso = GSOperations(BaseConfig.PROJECT_ID, BaseConfig.BUCKET_NAME)
                gso.upload_file(depth, output_file, source_file_type="file_object")

                # Save as color-mapped "turbo" jpg image.
                cmap = plt.get_cmap("turbo")
                color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(
                    np.uint8
                )
                color_map_output_file = output_file + ".jpg"
                logging.info(
                    "Saving color-mapped depth to: : %s", color_map_output_file
                )
                PIL.Image.fromarray(color_depth)
                gso.upload_file(
                    color_depth, color_map_output_file, source_file_type="file_object"
                )

            # Display the image and estimated depth map.
            if display:
                ax_rgb.imshow(image)
                ax_disp.imshow(inverse_depth_normalized, cmap="turbo")
                fig.canvas.draw()
                fig.canvas.flush_events()

        logging.info("Done predicting depth!")
        if display:
            plt.show(block=True)
