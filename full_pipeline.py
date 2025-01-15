from solar_detection.pipelines.preprocessing import plot
from solar_detection.config.config import BaseConfig as SolarConfig
from solar_detection.pipelines.depth_pipeline import depth_run
from solar_detection.pipelines.roof_detection import extract_potential_roofs
from solar_detection.pipelines.roof_detection import prediction as roof_prediction
from solar_detection.utils import get_torch_device, apply_pred_to_csv
from solar_detection.pipelines.solar_detection import prediction as solar_prediction


if __name__ == "__main__":
    device = get_torch_device()
    # plot(3, website=SolarConfig.GEOPORTAL_URL, images_dir=SolarConfig.IMAGES_DIR)
    depth_run()
    potential_roofs = extract_potential_roofs(
        SolarConfig.IMAGES_DIR,
        SolarConfig.DEPTH_DIR,
        SolarConfig.BUILDING_DETECTION_DIR,
    )
    pred, labels = roof_prediction(
        device, potential_roofs, SolarConfig.ROOF_MODEL, SolarConfig.ROOFS_DIR
        )
    apply_pred_to_csv(SolarConfig.BUILDINGS_CSV_FILE, pred, labels, "roof_pred", change_dash=False)

    pred, labels = solar_prediction(
        device, SolarConfig.SOLAR_ROOF_MODEL, SolarConfig.ROOFS_DIR
    )
    apply_pred_to_csv(SolarConfig.BUILDINGS_CSV_FILE, pred, labels, "roof_solar_pred", change_dash=False)

    print({"field_id": 3,
        "is_solar": 1 in pred,
        "is_solar_on_roof": 1 in pred,
    })