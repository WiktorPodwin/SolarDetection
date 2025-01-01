from typing import List
from solar_detection.datatypes import Image
from solar_detection.api.operations import DirectoryOperations
from solar_detection.processing.image_processing.image_process import ImageProcessing
from solar_detection.processing.depth_processing.depth_processing import DepthProcessing
from solar_detection.config import BaseConfig as Config


def prepare_image(input_directory: str, output_directory: str) -> List[Image]:
    files = DirectoryOperations.list_directory(input_directory)
    ret = []
    for file in files:
        input_path = input_directory + "/" + file
        output_path = output_directory + "/" + file
        # print(f"Processing image: {input_path}")
        image_processing = ImageProcessing()
        image = image_processing.load_image(input_path)
        mask = image_processing.generate_mask_around_plot(image)
        masked_image = image_processing.apply_mask(image, mask)
        cropped_plot, rectangle_shape = image_processing.crop_rectangle_around_plot(masked_image, True)
        # image_processing.save_image(output_path, cropped_plot)
        ret.append(Image(name=file, location=output_path, rectangle_shape=rectangle_shape, mask=mask))
    return ret

def depth_run():
    cut_out_plots_dir = f"{Config.DATA_DIR}/cut_out_plots"
    # process images
    images = prepare_image(Config.IMAGES_DIR, cut_out_plots_dir)
    # detect image depth
    depth_processing = DepthProcessing()
    depth_processing.run(
        image_paths=[
            image.location for image in images
        ],
        rectangle_shapes=[
          image.rectangle_shape for image in images
        ],
        masks = [
          image.mask for image in images
        ],
        save=True,
        display=False,
    )