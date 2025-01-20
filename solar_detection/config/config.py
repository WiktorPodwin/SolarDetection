import os
import logging
from attrs import define

logging.basicConfig(level=logging.INFO)


@define
class BaseConfig:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    GEOPORTAL_URL = "https://polska.geoportal2.pl/map/www/mapa.php?mapa=polska"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    IMAGES_DIR = os.path.join(DATA_DIR, "images")
    CUT_OUT_IMAGES_DIR = os.path.join(DATA_DIR, "cut_out_plots")
    DEPTH_DIR =  os.path.join(DATA_DIR, "depth")
    BUILDING_DETECTION_DIR = os.path.join(DATA_DIR, "potential_buildings")
    ROOFS_DIR = os.path.join(DATA_DIR, "roofs")

    PROJECT_ID = os.getenv("GCP_PROJECT_ID", "solar-panels")
    BUCKET_NAME = os.getenv("GS_BUCKET_NAME", "solar-panels")
    SA_CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json")
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

    LOCATION_CSV_FILE = os.path.join(DATA_DIR, "locations_1.csv")
    BUILDINGS_CSV_FILE = os.path.join(DATA_DIR, "potential_buildings.csv")

    CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

    ROOF_MODEL = os.path.join(CHECKPOINTS_DIR, "roof_detector.pt")
    ROOF_METRICS_DIR = os.path.join(BASE_DIR, "solar_detection/roofs_detector/metrics")

    SOLAR_ROOF_MODEL = os.path.join(CHECKPOINTS_DIR, "solar_roof_detector.pt")
    SOLAR_ROOF_METRICS_DIR = os.path.join(BASE_DIR, "solar_detection/solar_detector/metrics")

    USE_GS = os.getenv("USE_GS", "False") == "True"
