import logging
from typing import List

import numpy as np
from PIL import Image
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from config.config import BaseConfig
from src import depth_pro
from src.depth_pro.eval.boundary_metrics import (
    SI_boundary_Recall,
    get_thresholds_and_weights,
    boundary_f1,
    SI_boundary_F1,
)
from src.api.utils import get_torch_device
from src.api.operations import DirectoryOperations, GSOperations


class DepthProcessing:
    def __init__(self) -> None:
        self.model, self.transform = depth_pro.create_model_and_transforms(
            device=get_torch_device(),
            precision=torch.half,
        )
        self.model.eval()

    def run(self, image_paths: List[str], save: bool = False, display: bool = False):
        """Runs the depth detection process."""
        if BaseConfig.USE_GS:
            self.__run_gs(image_paths, save, display)
        else:
            self.__run_local(image_paths, save, display)

    def __run_gs(self, image_ids: List[str], save: bool = False, display: bool = False):
        """Runs the depth detection process."""

        gso = GSOperations(BaseConfig.PROJECT_ID, BaseConfig.BUCKET_NAME)
        for i, image_id in enumerate(gso.get_files(image_ids)):
            # Load image and focal length from exif info (if found.).
            try:
                logging.info("Loading image %s ...", image_ids[i])
                image, _, f_px = depth_pro.load_rgb(image_id)  # this needs change
            except Exception as e:
                logging.error(str(e))
                continue
            # Run prediction. If `f_px` is provided, it is used to estimate the final metric depth,
            # otherwise the model estimates `f_px` to compute the depth metricness.
            prediction = self.model.infer(self.transform(image), f_px=f_px)

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
                Image.fromarray(color_depth)
                gso.upload_file(
                    color_depth, color_map_output_file, source_file_type="file_object"
                )

    def __run_local(
        self, image_paths: List[str], save: bool = False, display: bool = False
    ):
        """Runs the depth detection process."""

        # Load model.

        if display:
            plt.ion()
            fig = plt.figure()
            ax_rgb = fig.add_subplot(121)
            ax_disp = fig.add_subplot(122)

        for image_path in tqdm(image_paths):
            # Load image and focal length from exif info (if found.).
            try:
                logging.info("Loading image %s ...", image_path)
                image, _, f_px = depth_pro.load_rgb(image_path, remove_alpha=False)
            except Exception as e:
                logging.error(str(e))
                continue
            # Run prediction. If `f_px` is provided, it is used to estimate the final metric depth,
            # otherwise the model estimates `f_px` to compute the depth metricness.
            prediction = self.model.infer(self.transform(image), f_px=f_px)

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
            ) # over 0.39 should be a house
            print("inverse_depth_normalized", inverse_depth_normalized)
            print("inverse_depth_normalized[0]", inverse_depth_normalized[0])
            print("inverse_depth_normalized[0][0]", inverse_depth_normalized[0][0])
            print("inverse_depth_normalized[..., :3][0] * 255", inverse_depth_normalized[..., :3][0] * 255)
            # print(inverse_depth_normalized.shape)
            # print("max_invdepth_vizu", max_invdepth_vizu)
            # print("min_invdepth_vizu", min_invdepth_vizu)
            # print("inverse_depth", inverse_depth)
            # print("depth", depth)
            mean_depth = np.mean(depth)
            mean_depth_matrix = np.full_like(depth, mean_depth)
            print("boundary metric", self.calculate_boundary_metrics(depth, mean_depth_matrix))

            # Save Depth as npz file.
            if save:
                image_directory = f"{BaseConfig.DATA_DIR}/depth"
                output_file = f"{image_directory}/{image_path.split("/")[-1].removesuffix(".png")}"

                logging.info("Saving depth map to: %s", output_file)
                DirectoryOperations.create_directory(image_directory)
                np.savez_compressed(output_file, depth=depth)
                print("inverse_depth_normalized", inverse_depth_normalized)
                inverse_depth_normalized[inverse_depth_normalized < 0.42] = 0
                print("inverse_depth_normalized", inverse_depth_normalized)
                # Save as color-mapped "turbo" jpg image.
                cmap = plt.get_cmap("turbo")
                color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(
                    np.uint8
                )
                color_map_output_file = output_file + ".jpg"
                logging.info(
                    "Saving color-mapped depth to: : %s", color_map_output_file
                )
                Image.fromarray(color_depth).save(
                    color_map_output_file, format="JPEG", quality=90
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

        return None

    def calculate_boundary_metrics(self, depth: np.ndarray, target_depth: np.ndarray):
        """Calculates the boundary metrics for the given depth map and ground truth."""
        thresholds, weights = get_thresholds_and_weights(0.1, 250, 100)
        boundary_metrics = SI_boundary_F1(depth, target_depth)
        return boundary_metrics

    def calculate_boundary_gt(self, depth: np.ndarray, gt: np.ndarray, t: float = 1.25):
        """Calculates the boundary metrics for the given depth map and ground truth."""
        thresholds, weights = get_thresholds_and_weights(0.1, 250, 100)
        boundary_metrics = boundary_f1(depth, gt, t)
        boundary_recall = SI_boundary_Recall(depth, gt)
        return boundary_metrics
