from src.api.operations import DirectoryOperations
from .image_processing.image_process import ImageProcessing

def process_image(input_directory: str, output_directory: str) -> None:
    files = DirectoryOperations.list_directory(input_directory)

    for file in files:
        input_path = input_directory + "/" + file
        output_path = output_directory + "/" + file

        image_processing = ImageProcessing()
        image = image_processing.load_image(input_path)
        cropped_image = image_processing.remove_outside_frame(image)
        image_processing.save_image(output_path, cropped_image)
