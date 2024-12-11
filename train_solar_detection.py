from src.pipelines.solar_detection import generate_model
from config.config import BaseConfig as config

if __name__ == "__main__":
    generate_model(config.LOCATION_CSV_FILE,
                   config.ROOFS_DIR,
                   20,
                   config.SOLAR_ROOF_MODEL,
                   config.SOLAR_ROOF_METRICS_DIR,
                   2,
                   (512, 512))
