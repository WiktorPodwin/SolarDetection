from src.pipelines.solar_detection import generate_model
from config.config import BaseConfig as config

if __name__ == "__main__":
    generate_model(config.LOCATION_CSV_FILE,
                   config.ROOFS_DIR,
                   num_epochs=30,
                   model_path=config.SOLAR_ROOF_MODEL,
                   metrics_dir=config.SOLAR_ROOF_METRICS_DIR,
                   data_multiplier=1,
                   resize_val=(224, 224),
                   learning_rate=0.001,
                   batch_size=4,
                   accumulation_steps=1
                   )