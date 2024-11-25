import os
from src.solar_detection.train_model import setup_training
from src.pipelines.depth_pipeline import process_image
from src.processing.depth_processing.depth_processing import depth_run
from src.utils import plot, upload_to_gs
from config import BaseConfig as Config


if __name__ == "__main__":
    depth_run()
    # plot(
    #     list(range(1, 100)),
    #     csv_file=Config.LOCATION_FIELD_CSV_DIR,
    #     website=Config.GEOPORTAL_URL,
    #     images_dir=Config.IMAGES_DIR,
    # )

    # # upload images to Google Storage
    # for dirpath, _, filenames in os.walk(Config.IMAGES_DIR):
    #     for filename in filenames:
    #         upload_to_gs(
    #             project_id="solar-detection-ai",
    #             bucket_name=Config.BUCKET_NAME,
    #             source_file_path=f"{dirpath}/{filename}",
    #             destination_blob_name=filename,
    #         )
