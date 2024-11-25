import logging
from typing import List, Tuple
import cv2

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
from src.utils import get_torch_device
from src.api.operations import DirectoryOperations, GSOperations
from src.processing.image_processing.image_process import ImageProcessing


class DepthProcessing:
    def __init__(self) -> None:
        self.model, self.transform = depth_pro.create_model_and_transforms(
            device=get_torch_device(),
            precision=torch.half,
        )
        self.model.eval()

    def run(
        self,
        image_paths: List[str],
        rectangle_shapes: List[Tuple[int, int, int, int]] | None = None,
        masks: List[np.ndarray] | None = None,
        save: bool = False,
        display: bool = False,
    ) -> None:
        """Runs the depth detection process."""
        if BaseConfig.USE_GS:
            self.__run_gs(image_paths, save, display)
        else:
            self.__run_local(image_paths, rectangle_shapes, masks, save, display)

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
        self,
        image_paths: List[str],
        rectangle_shapes: List[Tuple[int, int, int, int]] | None = None,
        masks: List[np.ndarray] | None = None,
        save: bool = False,
        display: bool = False,
    ):
        """Runs the depth detection process."""

        # Load model.
        number_of_images = len(image_paths)
        image_processing = ImageProcessing()

        for i, image_path in enumerate(tqdm(image_paths)):
            if display and i == number_of_images - 1:
                plt.ion()
                fig = plt.figure()
                ax_rgb = fig.add_subplot(121)
                ax_disp = fig.add_subplot(122)

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

            masked_image = self.get_masked_image(
                image_processing, depth, rectangle_shapes[i], masks[i]
            )

            # Save Depth as npz file.
            if save:
                cut_out_plot = image_processing.crop_plot(masked_image)
                self._save_local(image_path, depth, cut_out_plot)

            # Display the image and estimated depth map.
            if display and i == number_of_images - 1:
                cropped_rectangle = image_processing.crop_rectangle_around_plot(
                    masked_image, with_mask=True
                )

                original_image_original_size = (
                    image_processing.restore_original_size_from_rectangle(
                        image, rectangle_shapes[i]
                    )
                )
                original_image_with_mask = image_processing.apply_mask(
                    original_image_original_size, masks[i]
                )
                cut_out_original_plot = image_processing.crop_rectangle_around_plot(
                    original_image_with_mask, with_mask=True
                )
                ax_rgb.imshow(cut_out_original_plot)
                ax_disp.imshow(cropped_rectangle)
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.show(block=True)
                plt.clf()

        logging.info("Done predicting depth!")
        return None

    def _save_local(self, image_path: str, depth: np.ndarray, cut_out_plot: np.ndarray):
        image_directory = f"{BaseConfig.DATA_DIR}/depth"
        output_file = (
            f"{image_directory}/{image_path.split('/')[-1].removesuffix('.png')}"
        )

        logging.info("Saving depth map to: %s", output_file)
        DirectoryOperations.create_directory(image_directory)
        np.savez_compressed(output_file, depth=depth)

        color_map_output_file = output_file + ".jpg"
        logging.info("Saving color-mapped depth to: : %s", color_map_output_file)
        Image.fromarray(cut_out_plot).save(
            color_map_output_file, format="JPEG", quality=90
        )

    def _save_gs(self, image_id: str, depth: np.ndarray, cut_out_plot: np.ndarray): ...

    def get_masked_image(
        self,
        image_processing: ImageProcessing,
        depth: np.ndarray,
        rectangle_shape: Tuple[int, int, int, int] | None = None,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Returns the masked image."""
        inverse_depth = 1 / depth
        # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
        max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
        min_invdepth_vizu = max(1 / 250, inverse_depth.min())
        inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
            max_invdepth_vizu - min_invdepth_vizu
        )
        mean_depth = np.mean(depth)
        mean_depth_matrix = np.full_like(depth, mean_depth)
        # threshold displayed value
        inverse_depth_normalized[inverse_depth_normalized < 0.42] = 0

        cmap = plt.get_cmap("turbo")
        color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
        color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)

        original_size_inverse_depth_normalized = (
            image_processing.restore_original_size_from_rectangle(
                color_depth_bgr, rectangle_shape
            )
        )
        original_size_inverse_depth_normalized = (
            original_size_inverse_depth_normalized.astype("uint8")
        )
        return image_processing.apply_mask(
            original_size_inverse_depth_normalized, mask
        )

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
