from src.utils import prepare_from_csv_and_dir, prepare_for_prediction, train_model, predict, EvaluateMetrics
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
    history = train_model(model, train_loader, num_epochs=num_epochs, lr=learning_rate, save_path=model_path)
    predictions = predict(model, test_loader, model_path)
    evaluate_metrics = EvaluateMetrics(labels, predictions)
    evaluate_metrics.calculate_accuracy()
    evaluate_metrics.display_conf_matrix(metrics_dir)
    evaluate_metrics.display_history(history, metrics_dir)


# def prediction(potential_roofs: List[Image], model_path: str, img_dir: str) -> None:
#     """
#     Prepares the data to roof predictions, selects only images with predicted roof, 
#     combines them and saves to specified directory

#     Args:
#         potential_roofs (List[Image]): List of Image classes, storing potential roof parameters
#         model_path (str): Path to the saved model
#         img_dir (str): Directory path to store extracted roofs
#     """
#     dataloader = prepare_for_prediction(potential_roofs)
#     pred = predict(dataloader, model_path)
#     img_processing = ImageProcessing()
#     roofs = dict()

#     for i, Image in enumerate(potential_roofs):
#         if pred[i] == 1:
#             masked_roof = Image.potential_building
#             cropped_roof = img_processing.crop_plot(masked_roof)

#             if Image.name in roofs:
#                 roofs[Image.name].append(cropped_roof)
#             else:
#                 roofs[Image.name] = [cropped_roof]

#     for key, values in roofs.items():
#         roofs_extracted = img_processing.connect_images(values)
#         save_path = os.path.join(img_dir, key)
#         img_processing.save_image(save_path, roofs_extracted)