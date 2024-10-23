"""
# Example usage:
image_path = "example.png"

# Frame samples
frame_color = (76, 146, 207)
frame_color = (76, 146, 207)
frame_color = (87, 149, 179)
frame_color = (79, 153, 214)
# More or less blue frame
frame_color = (80, 150, 200)

# Black frame
frame_color = (0, 0, 0)

output_path = "output/example.png"

cut_outside_red_area(image_path, output_path)
remove_red_filter(image_path, output_path)
remove_outside_frame(image_path, frame_color, output_path)
"""

from typing import Tuple
import cv2
import numpy as np


def image_loader(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return image, cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def remove_outside_frame(
    image_path: str,
    frame_color: Tuple[int, int, int],
    output_path: str,
    tolerance: int = 0,
):
    """removes all elements outside the frame, excluding elements of the same color as a frame is.

    Args:
        image_path (str): Path to the input image.
        frame_color (tuple): 3 element tuple representing the RGB color of the frame.
        output_path (str): Path where the output image will be saved.
    """
    image = cv2.imread(image_path)
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
    # Create an alpha channel based on the mask (255 inside the frame, 0 outside)
    alpha_channel = np.where(
        cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY) > 0, 255, 0
    ).astype(np.uint8)
    # Add the alpha channel to the original image
    image_with_alpha = cv2.merge(
        [image[:, :, 0], image[:, :, 1], image[:, :, 2], alpha_channel]
    )

    print(f"Saving the output to {output_path}")
    cv2.imwrite(output_path, image_with_alpha)


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
