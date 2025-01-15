from solar_detection.pipelines.roof_detection import extract_potential_roofs, prediction
from solar_detection.config import BaseConfig as config
from solar_detection.utils import get_torch_device, apply_pred_to_csv, compare_prediction_and_labels

if __name__ == "__main__":
    potential_roofs = extract_potential_roofs(
        config.IMAGES_DIR, config.DEPTH_DIR, config.BUILDING_DETECTION_DIR
    )
    device = get_torch_device()
    pred, labels = prediction(device, potential_roofs, config.ROOF_MODEL, config.ROOFS_DIR)
    apply_pred_to_csv(config.BUILDINGS_CSV_FILE, pred, labels, "roof_pred", change_dash=False)
    compare_prediction_and_labels(config.BUILDINGS_CSV_FILE, 'is_roof', 'roof_pred')

    