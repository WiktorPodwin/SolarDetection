from src.utils import prepare_train_test_data, train_model
from src.roofs_detection.model_testing import model_testing
from src.roofs_detection.evaluate_model import EvaluateMetrics


def roof_detector(csv_file_path: str, potential_roofs_dir: str, num_epochs: int, model_path: str, metrics_dir: str) -> None:
    """
    Prepares the data, trains model and tests his performance
    
    Args:
        csv_file_path (str): A path to the csv file storing potential roofs labels
        potential_roofs_dir (str): A path to the directory storing potential roofs
        num_epochs (int): Number of epochs
        model_path (str): A path to storing the model
        metrics_dir (str): A directory path for storing metrics
    """
    train_loader, test_loader = prepare_train_test_data(csv_file_path, potential_roofs_dir)
    train_model(train_loader, num_epochs=num_epochs, save_path=model_path)
    predictions, labels = model_testing(test_loader, model_path)
    evaluate_metrics = EvaluateMetrics(labels, predictions)
    evaluate_metrics.calculate_accuracy()
    evaluate_metrics.display_conf_matrix(metrics_dir)