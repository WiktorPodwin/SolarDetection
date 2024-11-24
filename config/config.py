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
    MODEL_DIR = os.path.join(DATA_DIR, "models")
    LOCATION_FIELD_CSV_DIR = os.path.join(DATA_DIR, "locations.csv")

    PROJECT_ID = os.getenv("GCP_PROJECT_ID", "solar-panels")
    BUCKET_NAME = os.getenv("GS_BUCKET_NAME", "solar-panels")
    SA_CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json")
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

    SOLAR_DETECTION_TRAINING_DIR = os.path.join(DATA_DIR, "solar-detection/training")
    SOLAR_DETECTION_TEST_DIR = os.path.join(DATA_DIR, "solar-detection/test")
    SOLAR_DETECTION_VALIDATION_DIR = os.path.join(DATA_DIR, "solar-detection/validation")


    USE_GS = os.getenv("USE_GS", "False") == "True"
