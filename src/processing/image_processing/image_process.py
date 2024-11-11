"""
# Example usage:
image_path = "data/images/281411_2.0001.295_5.png"
output_path = "data/result_image.png"

image_processing = ImageProcessing()
image = image_processing.load_image(image_path)
cropped_image = image_processing.remove_outside_frame(image)
image_processing.save_image(output_path, cropped_image)

"""

import logging
from typing import Tuple
import cv2
import numpy as np


class ImageProcessing:
    """
    Class to process an image
    """

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Loads an image from specified path and returns it as matrix

        Args:
            image_path: Path to the saved image
        Returns:
            np.ndarray: Image matrix
        """
        image = cv2.imread(image_path)
        return image

    def save_image(self, save_path: str, image: np.ndarray) -> None:
        """
        Saves an image to specified path

        Args:
            save_path: Path to save the image
            image: Matrix respresenting image
        """
        cv2.imwrite(save_path, image)

    def remove_outside_frame(
        self,
        image: np.ndarray,
        frame_color: Tuple[int, int, int] = (80, 150, 200),
        tolerance: int = 60,
        clear_pixels: bool = False,
    ) -> np.ndarray:
        """
        Removes all elements outside the frame, including noise reduction for isolated pixels.

        Args:
            image (np.ndarray): Image matrix
            frame_color (tuple): 3 element tuple representing the RGB color of the frame.
            output_path (str): Path where the output image will be saved.
            tolerance (int): Tolerance for color matching in frame detection.
        """
        image_shape = image.shape
        x_half = int(image_shape[1] / 2)
        y_half = int(image_shape[0] / 2)

        # Convert the frame color from RGB to BGR (since OpenCV uses BGR)
        frame_color_bgr = tuple(reversed(frame_color))  # convert RGB to BGR

        # Define tolerance for color matching (to handle slight variations in color)
        lower_bound = np.array(
            [max(c - tolerance, 0) for c in frame_color_bgr], dtype=np.uint8
        )
        upper_bound = np.array(
            [min(c + tolerance, 255) for c in frame_color_bgr], dtype=np.uint8
        )

        # Create a mask where the frame color is present
        mask = cv2.inRange(image, lower_bound, upper_bound)
        # Create a mask of zeros (black image)
        frame_mask = np.zeros_like(image)
        # Find contours of the frame color regions
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        measure_differences = {}
        measures = {}

        for i, contour in enumerate(contours):
            if i == 0:
                continue

            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            x, y, w, h = cv2.boundingRect(approx)
            x_mid = int(x + w / 2)
            y_mid = int(y + h / 2)

            if w > 30 and h > 30 and 2 < approx.shape[0] < 10:
                difference = np.abs(x_mid - x_half) + np.abs(y_mid - y_half)
                measure_differences[i] = difference
                measures[i] = approx
        try:
            closest_point = min(measure_differences, key=measure_differences.get)
        except ValueError:
            logging.error("No frame detected in the image.")
            return image
        final_approx = tuple([measures[closest_point]])
        cv2.drawContours(
            frame_mask, final_approx, -1, (255, 255, 255), thickness=cv2.FILLED
        )

        gray_mask = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)

        # Create an alpha channel based on the final cleaned mask
        alpha_channel = np.where(gray_mask > 0, 255, 0).astype(np.uint8)
        image_with_alpha = cv2.merge(
            [image[:, :, 0], image[:, :, 1], image[:, :, 2], alpha_channel]
        )
        if clear_pixels:
            # Crop the image to remove transparent pixels
            image_with_alpha[image_with_alpha[..., 3] == 0] = [0, 0, 0, 0]

        return image_with_alpha

    def crop_image(
        self, image: np.ndarray, x: int, y: int, w: int, h: int
    ) -> np.ndarray:
        """
        Crops an image to the specified dimensions

        Args:
            image (np.ndarray): Image matrix
            x (int): x-coordinate of the top-left corner
            y (int): y-coordinate of the top-left corner
            w (int): Width of the cropped image
            h (int): Height of the cropped image

        Returns:
            np.ndarray: Cropped image matrix
        """
        return image[y : y + h, x : x + w]
