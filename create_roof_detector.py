from pipelines.create_roof_detector import roof_detector
from config.config import BaseConfig as config

if __name__ == "__main__":
    roof_detector(
        csv_file_path=config.BUILDINGS_CSV_FILE, 
        potential_roofs_dir=config.BUILDING_DETECTION_DIR, 
        num_epochs=18, 
        model_path=config.ROOF_MODEL, 
        metrics_dir=config.ROOF_METRICS_DIR
        )


