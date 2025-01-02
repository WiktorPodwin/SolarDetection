import solar_detection.solar_detector.metrics
import solar_detection.solar_detector.solar_detector
from solar_detection.pipelines.depth_pipeline import depth_run
from solar_detection.config import BaseConfig as Config

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
