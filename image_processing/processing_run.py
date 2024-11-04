from image_processing.processing.image_processing import ImageProcessing
from api.src.operations.data_operations import DirectoryOperations

def process_image(input_directory: str, output_directory: str) -> None:
    dir_oper = DirectoryOperations()
    files = dir_oper.list_directory(input_directory)

    for file in files:
        input_path = input_directory + "/" + file
        output_path = output_directory + "/" + file
        image_processing = ImageProcessing()

        image = image_processing.load_image(input_path)
        image = image_processing.crop_image(image)
        image = image_processing.remove_outside_frame(image)
        image_processing.save_image(output_path, image)
