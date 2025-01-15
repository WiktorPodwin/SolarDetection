from solar_detection.config import BaseConfig as config
from solar_detection.utils import get_torch_device, fill_data_for_training
from solar_detection.pipelines.solar_detection import generate_model


if __name__ == "__main__":
    # Generate more images
    # fill_data_for_training(
    #     config.LOCATION_CSV_FILE, config.ROOFS_DIR, [60, 150, 270]
    # )

    device = get_torch_device()

    generate_model(
        device=device,
        csv_file_path=config.BUILDINGS_CSV_FILE,
        roof_label="is_roof",
        solar_label="panel_on_roof",
        potential_roofs_dir=config.BUILDING_DETECTION_DIR,
        roofs_dir=config.ROOFS_DIR,
        num_epochs=70,
        model_path=config.SOLAR_ROOF_MODEL,
        metrics_dir=config.SOLAR_ROOF_METRICS_DIR,
        data_multiplier=2,
        resize_val=256,
        learning_rate=0.0001,
        batch_size=8,
        accumulation_steps=1
    )
