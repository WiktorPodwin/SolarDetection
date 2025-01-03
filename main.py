from solar_detection.api.operations.data_operations import DirectoryOperations
from solar_detection.datatypes.datatypes import Image
from solar_detection.processing.depth_processing.depth_processing import DepthProcessing
from solar_detection.processing.image_processing.image_process import ImageProcessing
from solar_detection.utils.utils import plot
from solar_detection.config.config import BaseConfig as SolarConfig
from solar_detection.pipelines.roof_detection import extract_potential_roofs
from solar_detection.pipelines.roof_detection import prediction as roof_prediction
from solar_detection.utils import get_torch_device
from solar_detection.pipelines.solar_detection import prediction as solar_prediction

field_id = "281411_2.0001.295_3"
filename = f"{field_id}.png"
raw_image_path = f"{SolarConfig.IMAGES_DIR}/{filename}"
cut_out_image_path = f"{SolarConfig.CUT_OUT_IMAGES_DIR}/{filename}"
image_processing = ImageProcessing()
depth_processing = DepthProcessing()
device = get_torch_device()
# DirectoryOperations.create_directory(SolarConfig.CUT_OUT_IMAGES_DIR)
# plot(field_ids=[field_id], website=SolarConfig.GEOPORTAL_URL, images_dir=SolarConfig.IMAGES_DIR)

image = image_processing.load_image(raw_image_path)
mask = image_processing.generate_mask_around_plot(image)
masked_image = image_processing.apply_mask(image, mask)
cropped_plot, rectangle_shape = image_processing.crop_rectangle_around_plot(
    masked_image, True
)
image_processing.save_image(cut_out_image_path, cropped_plot)

cut_out_image = Image(
    name=field_id,
    location=cut_out_image_path,
    rectangle_shape=rectangle_shape,
    mask=mask,
)

print("Depth processing")
depth_processing.run(
    image_paths=[cut_out_image.location],
    rectangle_shapes=[cut_out_image.rectangle_shape],
    masks=[cut_out_image.mask],
    save=True,
    display=False,
)

print("Roof extraction")
potential_roofs = extract_potential_roofs(
    SolarConfig.IMAGES_DIR,
    SolarConfig.DEPTH_DIR,
    SolarConfig.BUILDING_DETECTION_DIR,
)

print("Roof prediction")
roof_prediction(device, potential_roofs, SolarConfig.ROOF_MODEL, SolarConfig.ROOFS_DIR)

print("Solar prediction")
pred, _ = solar_prediction(device, SolarConfig.SOLAR_ROOF_MODEL, SolarConfig.ROOFS_DIR)
