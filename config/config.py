import os
import logging
from attrs import define

logging.basicConfig(level=logging.INFO)


@define
class BaseConfig:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    GEOPORTAL_URL = "https://polska.geoportal2.pl/map/www/mapa.php?mapa=polska"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    IMAGES_DIR = os.path.join(DATA_DIR, "images")
    DEPTH_DIR =  os.path.join(DATA_DIR, "depth")
    BUILDING_DETECTION_DIR = os.path.join(DATA_DIR, "potential_buildings")
    ROOFS_DIR = os.path.join(DATA_DIR, "roofs")

    # MODEL_DIR = os.path.join(DATA_DIR, "models")

    PROJECT_ID = os.getenv("GCP_PROJECT_ID", "solar-panels")
    BUCKET_NAME = os.getenv("GS_BUCKET_NAME", "solar-panels")
    SA_CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json")
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

    LOCATION_FIELD_CSV_DIR = os.path.join(DATA_DIR, "locations.csv")
    BUILDINGS_CSV_FILE = os.path.join(DATA_DIR, "potential_buildings.csv")
    
    SOLAR_DETECTION_TRAINING_DIR = os.path.join(DATA_DIR, "solar-detection/training")
    SOLAR_DETECTION_TEST_DIR = os.path.join(DATA_DIR, "solar-detection/test")
    SOLAR_DETECTION_VALIDATION_DIR = os.path.join(DATA_DIR, "solar-detection/validation")

    CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

    ROOF_MODEL = os.path.join(CHECKPOINTS_DIR, "roof_detector.pt")
    ROOF_METRICS_DIR = os.path.join(BASE_DIR, "src/roofs_detection/metrics")

    USE_GS = os.getenv("USE_GS", "False") == "True"
