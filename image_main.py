from processing.processing_run import process_image
from config.config import BaseConfig as config

if __name__ == "__main__":
    process_image(config.IMAGES_DIR, "data/cut_out_plots")