from typing import List
from src.datatypes import Image
from src.api.operations import DirectoryOperations
from .image_processing.image_process import ImageProcessing


def process_image(input_directory: str, output_directory: str) -> List[Image]:
    files = DirectoryOperations.list_directory(input_directory)
    ret = []
    for file in files:
        input_path = input_directory + "/" + file
        output_path = output_directory + "/" + file
        print(f"Processing image: {input_path}")
        image_processing = ImageProcessing()
        # image = image_processing.load_image(input_path)
        # cropped_image = image_processing.remove_outside_frame(image, clear_pixels=True)
        # image_processing.save_image(output_path, cropped_image)
        ret.append(Image(name=file, location=input_path))

    return ret
