"""
# Example usage:
image_path = "data/images/281411_2.0001.295_6.png"
output_path = "data/result2_image.png"
frame_color = (80, 150, 200)

remove_outside_frame(image_path, frame_color, output_path, tolerance=60)
"""

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
    
    def crop_image(self, image: np.ndarray) -> np.ndarray:
        """
        Crops an image to cut of redundant elements

        Args:
            image: Image matrix
        Returns:
            np.ndarray: Cropped image matrix
        """
        h1, h2, w1, w2 = 80, 800, 300, 1250
        image = image[h1:h2, w1:w2]
        return image

    def remove_outside_frame(self,
        image: np.ndarray,
        frame_color: Tuple[int, int, int] = (80, 150, 200),
        tolerance: int = 60,
        noise_removal_kernel_size: int = 3,
        iterations: int = 9
        ) -> np.ndarray:
        """
        Removes all elements outside the frame, including noise reduction for isolated pixels.
        
        Args:
            image (np.ndarray): Image matrix
            frame_color (tuple): 3 element tuple representing the RGB color of the frame.
            output_path (str): Path where the output image will be saved.
            tolerance (int): Tolerance for color matching in frame detection.
            noise_removal_kernel_size (int): Size of the kernel for noise removal. Larger kernels remove more noise.
            iterations (int): Number of iterations in erosion
        """
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
        # Find contours of the frame color regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Create a mask of zeros (black image)
        frame_mask = np.zeros_like(image)
        # Draw the contours on the mask
        cv2.drawContours(frame_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

        gray_mask = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((noise_removal_kernel_size, noise_removal_kernel_size), np.uint8)
        eroded = cv2.erode(gray_mask, kernel, iterations=iterations)

        # Create an alpha channel based on the final cleaned mask
        alpha_channel = np.where(eroded > 0, 255, 0).astype(np.uint8)
        image_with_alpha = cv2.merge([image[:, :, 0], image[:, :, 1], image[:, :, 2], alpha_channel])

        return image_with_alpha



def image_loader(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return image, cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def remove_red_filter(image_path, output_path):
    # Load the image
    # Convert the image to HSV (Hue, Saturation, Value) color space
    image, hsv_image = image_loader(image_path)
    red_mask = create_red_mask(hsv_image)
    # Invert the mask to get non-red areas
    non_red_mask = cv2.bitwise_not(red_mask)

    # Reduce the red channel in the red-filtered regions
    image[:, :, 2] = cv2.bitwise_and(image[:, :, 2], non_red_mask)
    # Set the non-red areas to black
    image[np.where(non_red_mask == 255)] = [0, 0, 0]
    # Save the output image
    cv2.imwrite(output_path, image)


def cut_outside_red_area(image_path, output_path):
    # Load the image
    # Convert the image to HSV (Hue, Saturation, Value) color space
    image, hsv_image = image_loader(image_path)
    red_mask = create_red_mask(hsv_image)

    # Find the bounding box of the red area
    x, y, w, h = cv2.boundingRect(red_mask)
    print(f"Bounding box of red area: x={x}, y={y}, w={w}, h={h}")

    # Crop the image using the bounding box
    cropped_image = image[y : y + h, x : x + w]

    cv2.imwrite(output_path, cropped_image)


def create_red_mask(hsv_image):
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    return mask1 + mask2
