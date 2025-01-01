from solar_detection.pipelines.roof_detection import extract_potential_roofs, prediction
from solar_detection.config import BaseConfig as config
from solar_detection.utils import get_torch_device

if __name__ == "__main__":
    potential_roofs = extract_potential_roofs(
        config.IMAGES_DIR, config.DEPTH_DIR, config.BUILDING_DETECTION_DIR
    )
    device = get_torch_device()
    prediction(device, potential_roofs, config.ROOF_MODEL, config.ROOFS_DIR)
