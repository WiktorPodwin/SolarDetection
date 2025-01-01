from solar_detection.pipelines.solar_detection import prediction
from config.config import BaseConfig as config
from solar_detection.utils import get_torch_device, apply_pred_to_csv

if __name__ == "__main__":
    device = get_torch_device()
    pred, labels = prediction(device, config.SOLAR_ROOF_MODEL, config.ROOFS_DIR)
    for i in range(len(pred)):
        print("---")
        print(labels[i])
        print(pred[i])
        print("---")
    apply_pred_to_csv(config.LOCATION_CSV_FILE, pred, labels, "roof_solar_pred")
