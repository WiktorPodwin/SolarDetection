from src.pipelines.roof_detection import generate_model
from config.config import BaseConfig as config
from src.utils import get_torch_device


if __name__ == "__main__":
    device = get_torch_device()

    generate_model(
        device=device,
        csv_file_path=config.BUILDINGS_CSV_FILE, 
        potential_roofs_dir=config.BUILDING_DETECTION_DIR, 
        num_epochs=30,
        model_path=config.ROOF_MODEL, 
        metrics_dir=config.ROOF_METRICS_DIR,
        data_multiplier=2,
        resize_val=(128, 128),
        batch_size=32,
        learning_rate=0.001,
        accumulation_steps=4
        )
