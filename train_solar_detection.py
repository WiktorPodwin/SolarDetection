from src.pipelines.solar_detection import generate_model
from config.config import BaseConfig as config
from src.utils import get_torch_device


if __name__ == "__main__":
    device = get_torch_device()

    generate_model(device=device,
                   csv_file_path=config.LOCATION_CSV_FILE,
                   potential_roofs_dir=config.ROOFS_DIR,
                   num_epochs=25,
                   model_path=config.SOLAR_ROOF_MODEL,
                   metrics_dir=config.SOLAR_ROOF_METRICS_DIR,
                   data_multiplier=1,
                   resize_val=(224, 224),
                   learning_rate=0.0001,
                   batch_size=8,
                   accumulation_steps=1
                   )