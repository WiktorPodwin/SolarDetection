from config.config import BaseConfig as config
from pathlib import Path
from src.processing.image_processing.image_process import ImageProcessing
from src.api.operations.data_operations import DirectoryOperations
from src.datatypes import Image
import os
from typing import List

def extract_potential_roofs() -> List[Image]:
    original_dir_path = config.IMAGES_DIR
    depth_dir_path = os.path.join(config.BASE_DIR, "data/depth/")
    buildings_dir_path = os.path.join(config.BASE_DIR, "data/potential_buildings/")

    dir_oper = DirectoryOperations
    dir_oper.create_directory(buildings_dir_path)
    files = dir_oper.list_directory(depth_dir_path)
    potential_roofs = []

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
            shapes = image_processing.generate_mask_around_potential_building(segmented_image, low_boundary, high_boundary)
            
            for i, shape in enumerate(shapes):
                plot_id_nosuffix = file_path.with_suffix('')
                plot_id_i = str(plot_id_nosuffix) + f"_{i}.png"
                building_file_path = os.path.join(buildings_dir_path, plot_id_i)

                building_mask = image_processing.apply_mask(original_image, shape)
                extracted_rectangle = image_processing.crop_rectangle_around_plot(building_mask, with_mask=True)
                extracted_buliding = image_processing.crop_plot(extracted_rectangle)
                resized_building = image_processing.resize_image(extracted_buliding)
                image_processing.save_image(building_file_path, resized_building)

                potential_roofs.append(Image(name=str(file_path), new_name=plot_id_i, potential_building=building_mask))
    
    return potential_roofs