from config.config import BaseConfig as config
from pathlib import Path
from src.processing.image_processing.image_process import ImageProcessing
from src.api.operations.data_operations import DirectoryOperations
import os



if __name__ == "__main__":
    original_dir_path = config.IMAGES_DIR
    depth_dir_path = os.path.join(config.BASE_DIR, "data/depth/")
    buildings_dir_path = os.path.join(config.BASE_DIR, "data/buildings/")

    dir_oper = DirectoryOperations
    dir_oper.create_directory(buildings_dir_path)
    files = dir_oper.list_directory(depth_dir_path)

    for file in files:
        file_path = Path(file)
        if file_path.suffix == ".jpg":
            original_image_path = os.path.join(original_dir_path, str(file_path))
            png_original_image_path = os.path.join(original_dir_path, file_path.with_suffix('.png').name)

            input_file_path = os.path.join(depth_dir_path, file_path)
            building_file_path = os.path.join(buildings_dir_path, file_path.with_suffix('.png').name)

            image_processing = ImageProcessing()
            segmented_image = image_processing.load_image(input_file_path)
            original_image = image_processing.load_image(png_original_image_path)
            low_boundary = (0, 2, 110)
            high_boundary = (100, 255, 255)
            mask = image_processing.generate_mask_around_building(segmented_image, low_boundary, high_boundary)

            extracted_buliding = image_processing.apply_mask(original_image, mask)
            image_processing.save_image(building_file_path, extracted_buliding)