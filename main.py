import os
from src.processing.processing_run import process_image
from src.processing.depth_processing.depth_processing import DepthProcessing
from src.api.utils import plot, upload_to_gs
from config import BaseConfig as Config


def run():
    cut_out_plots_dir = f"{Config.DATA_DIR}/cut_out_plots"
    # process images
    images = process_image(Config.IMAGES_DIR, cut_out_plots_dir)
    # detect image depth
    depth_processing = DepthProcessing()
    depth_processing.run(
        image_paths=[
            image.location for image in images
        ],
        save=False,
        display=False,
    )


    # predict solar panel locations


if __name__ == "__main__":
    run()
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
