from src.utils import prepare_data, train_model
from config.config import BaseConfig as config
from src.roofs_detection.test_model import test_model


if __name__ == "__main__":
    train_loader, test_loader = prepare_data(config.BUILDINGS_CSV_FILE, config.BUILDING_DETECTION_DIR)
    train_model(train_loader, num_epochs=18, save_path=config.ROOF_MODEL)
    predictions = test_model(test_loader, config.ROOF_MODEL)