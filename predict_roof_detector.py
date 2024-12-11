from src.pipelines.roof_detection import extract_potential_roofs, prediction
from config.config import BaseConfig as config

if __name__ == "__main__":
    potential_roofs = extract_potential_roofs(config.IMAGES_DIR, config.DEPTH_DIR, config.BUILDING_DETECTION_DIR) 
    prediction(potential_roofs, config.ROOF_MODEL, config.ROOFS_DIR)
