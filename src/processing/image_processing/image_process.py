import logging
from typing import Tuple, Optional, List, Any
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
        image = cv2.imread(image_path, cv2.COLOR_RGB2BGR)
        return image

    def save_image(self, save_path: str, image: np.ndarray) -> None:
        """
        Saves an image to specified path

        Args:
            save_path: Path to save the image
            image: Matrix respresenting image
        """
        cv2.imwrite(save_path, image)

    def generate_mask_around_plot(
        self,
        image: np.ndarray,
        frame_color: Tuple[int, int, int] = (80, 150, 200),
        tolerance: int = 60
        ) -> np.ndarray:
        """
        Generates a mask around the searched plot

        Args:
            image (np.ndarray): Image matrix
            frame_color (tuple): 3 element tuple representing the RGB color of the frame.
            tolerance (int): Tolerance for color matching in frame detection.

        Returns:
            np.ndarray: Mask matrix
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
        return frame_mask
    
    def generate_mask_around_potential_building(self,
                                      image: np.ndarray,
                                      low_boundary: Tuple[int, int, int],
                                      high_boundary: Tuple[int, int, int]
                                      ) -> List[np.ndarray | Any]:
        """
        Generates a mask around the building 

        Args:
            image (np.ndarray): Image matrix
            low_boundary (tuple): 3 element tuple representing the low boundary of RGB color.
            high_boundary (tuple): 3 element tuple representing the top boundary of RGB color.

        Returns:
            List[np.ndarray | Any]: List of images with masks around potential buildings
        """

        low_boundary_bgr = tuple(reversed(low_boundary))
        high_boundary_bgr = tuple(reversed(high_boundary))

        reversed_mask = cv2.inRange(image, low_boundary_bgr, high_boundary_bgr)

        mask = cv2.bitwise_not(reversed_mask)
        frame_mask = np.zeros_like(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        coordinates = {}
        areas = {}
        for i, contour in enumerate(contours):

            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            area = cv2.contourArea(approx)
            _, _, w, h = cv2.boundingRect(approx)

            if w > 30 and h > 30 and 4 <= approx.shape[0] <= 15 and area > 4200:
                coordinates[i] = approx
                areas[i] = area

        shapes = []
        for _, value in coordinates.items():
            frame_mask = np.zeros_like(image)
            cv2.drawContours(frame_mask, [value], -1, (255, 255, 255), cv2.FILLED)
            shapes.append(frame_mask)
        return shapes


    def apply_mask(self, image: np.ndarray, mask: np.ndarray, clear_pixels: bool = False) -> np.ndarray:
        """
        Removes all elements outside the frame, including noise reduction for isolated pixels.

        Args:
            image (np.ndarray): Image matrix
            mask (np.ndarray): Mask matrix
        
        Returns:
            np.ndarray: Cut out plot
        """
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Create an alpha channel based on the final cleaned mask
        alpha_channel = np.where(gray_mask > 0, 255, 0).astype(np.uint8)
        image = image.astype(np.uint8)

        image_with_alpha = cv2.merge(
            [image[:, :, 0], image[:, :, 1], image[:, :, 2], alpha_channel]
        )
        if clear_pixels:
            # Crop the image to remove transparent pixels
            image_with_alpha[image_with_alpha[..., 3] == 0] = [0, 0, 0, 0]

        return image_with_alpha
    
    def crop_plot(self, image: np.ndarray) -> np.ndarray:
        """
        Cuts out plot from image with mask
        
        Args:
            image (np.ndarray): Image matrix

        Returns:
            np.ndarray: Cropped plot matrix
        """
        if image.shape[2] != 4:
            logging.error("ValueError: Input image must have exacly 4 channels.")
            return image
        
        rgb_image = image[:, :, :3]
        mask = image[:, :, 3]
        
        black_background = np.zeros_like(rgb_image)

        cropped_image = np.where(mask[..., None] != 0, rgb_image, black_background)

        return cropped_image
    
    def crop_rectangle_around_plot(self, 
                                   image: np.ndarray, 
                                   return_coordinates: bool = False,
                                   with_mask: bool = False,
                                   ) -> Tuple[np.ndarray, 
                                              Optional[Tuple[int,...]]]:
        """
        Cuts out rescatgle around plot from image with mask
        
        Args:
            image (np.ndarray): Image matrix
            return_coordinates (bool): If should return rectangle shape
            with_mask (bool): If should return image with mask

        Returns:
            np.ndarray: Cropped rectangle around plot matrix
            Optional[Tuple[int,...]]: Size of rectangle (x_min, x_max, y_min, y_max)
        """
        try:
            mask = image[:, :, 3]
            if with_mask:
                final_image = image.copy()
            else:
                final_image = image[:, :, :3]
        except IndexError:
            logging.error("Image doesn't have marked any plot")
            if return_coordinates:
                return image, None
            else: 
                return image
        y_indicies, x_indicies = np.where(mask != 0)

        if y_indicies.size == 0 or x_indicies.size == 0:
            logging.error("Plot is not detected, impossible to create a rectangle around")
            return image
        
        x_min, x_max = x_indicies.min(), x_indicies.max()
        y_min, y_max = y_indicies.min(), y_indicies.max()

        cropped_rectangle = final_image[y_min: y_max + 1, x_min: x_max + 1]

        if return_coordinates:
            coordinates = (x_min, x_max, y_min, y_max)
            return cropped_rectangle, coordinates
        else: 
            return cropped_rectangle

    def restore_original_size_from_rectangle(self, 
                                             cropped_rectangle: np.ndarray,
                                             rectangle_cords: Tuple[int, int, int, int],
                                             original_size: Tuple[int, int, int] = (780, 1450, 3)
                                             ) -> np.ndarray:
      """
      Restores a cropped rectangle back to the original image size.

      Args:
          cropped_rectangle (np.ndarray): The cropped rectangle image.
          original_size (Tuple[int, int, int]): The size of the original image (height, width, channels).
          rectangle_coords (Tuple[int, int, int, int]): Coordinates of the rectangle (x_min, x_max, y_min, y_max).

      Returns:
          np.ndarray: Restored image with the rectangle placed back.
      """
      try:
          x_min, x_max, y_min, y_max = rectangle_cords
      except TypeError:
          x_min, x_max, y_min, y_max = 0, 0, 0, 0
      restored_image = np.ones(original_size)
      restored_image[y_min: y_max + 1, x_min: x_max + 1] = cropped_rectangle
      return restored_image
    
    def resize_image(self, image: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Resizes the image into specified size
        
        Args:
            image (np.ndarray): Image matrix
            size (Tuple[int, int]): Size of resized image
        
        Returns:
            np.ndarray: Resized image
        """
        resized_image = cv2.resize(image, size)
        return resized_image


        