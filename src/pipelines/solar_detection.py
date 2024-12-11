from src.utils import prepare_from_csv_and_dir, prepare_for_prediction, train_model
from src.utils.model.evaluate_model import EvaluateMetrics
from src.utils.model.model_prediction import predict
from src.solar_detection.solar_detector import SolarRoofDetector
from typing import Tuple

def generate_model(csv_file_path: str, 
                   potential_roofs_dir: str, 
                   num_epochs: int,
                   model_path: str, 
                   metrics_dir: str,
                   enhance_val: int = 1,
                   resize_val: int | Tuple[int, int] = None,
                   learning_rate: float = 0.0001
                   ) -> None:
    """
    Prepares the data, trains model and tests his performance
    
    Args:
        csv_file_path (str): A path to the csv file storing plots ID and roofs labels
        potential_roofs_dir (str): A path to the directory storing roofs
        num_epochs (int): Number of epochs
        model_path (str): A path to storing the model
        metrics_dir (str): A directory path for storing metrics
        enhance_val (int): Data replication number
        resize_val (int | Tuple[int, int]): Shape of resized image
        learning_rate (float): Learning rate for model training
    """
    train_loader, test_loader, labels = prepare_from_csv_and_dir(csv_file_path, potential_roofs_dir, enhance_val, resize_val)
    model = SolarRoofDetector()
    train_model(model, train_loader, num_epochs=num_epochs, lr=learning_rate, save_path=model_path)
    predictions = predict(model, test_loader, model_path)
    evaluate_metrics = EvaluateMetrics(labels, predictions)
    evaluate_metrics.calculate_accuracy()
    evaluate_metrics.display_conf_matrix(metrics_dir)