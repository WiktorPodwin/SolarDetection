from solar_detection.pipelines.solar_detection import prediction
from solar_detection.config import BaseConfig as config
from solar_detection.utils import get_torch_device, apply_pred_to_csv, compare_prediction_and_labels, update_field_csv

if __name__ == "__main__":
    device = get_torch_device()
    pred, labels = prediction(device, config.SOLAR_ROOF_MODEL, config.ROOFS_DIR)
    apply_pred_to_csv(config.BUILDINGS_CSV_FILE, pred, labels, "roof_solar_pred", change_dash=False)
    compare_prediction_and_labels(config.BUILDINGS_CSV_FILE, 'panel_on_roof', 'roof_solar_pred', 'is_roof')
    update_field_csv(config.BUILDINGS_CSV_FILE, 'roof_solar_pred', config.LOCATION_CSV_FILE, 'roof_solar_pred')