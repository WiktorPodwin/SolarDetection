from src.utils import prepare_from_csv_and_dir, prepare_for_prediction, train_model, upload_csv_file
from config.config import BaseConfig as config
from src.roofs_detection.model_prediction import predict
from src.roofs_detection.evaluate_model import EvaluateMetrics
from src.processing.image_processing.image_process import ImageProcessing
from src.api.operations.data_operations import DirectoryOperations
from src.datatypes import Image
from pathlib import Path
from typing import List
import os
from src.datatypes import Image as Img

def extract_potential_roofs(base_dir: str, depth_dir: str, potential_roofs_dir: str) -> List[Image]:
    """
    Uploads images from the depth directory, extracts potential candidates for roofs and save them
    
    Args:
        base_dir (str): A directory storing original images
        depth_dir (str): A directory storing images with a depth
        potentoal_roofs_dir (str): A directory to save potential roofs
    
    Returns:
        List[Image]: List of Image classes, storing potential roof parameters
    """
    dir_oper = DirectoryOperations
    dir_oper.create_directory(potential_roofs_dir)
    files = dir_oper.list_directory(depth_dir)
    potential_roofs = []

    for file in files:
        file_path = Path(file)
        if file_path.suffix == ".jpg":
            png_original_image_path = os.path.join(base_dir, file_path.with_suffix('.png').name)

            input_file_path = os.path.join(depth_dir, file_path)
            building_file_path = os.path.join(potential_roofs_dir, file_path.with_suffix('.png').name)
            
            image_processing = ImageProcessing()
            segmented_image = image_processing.load_image(input_file_path)
            original_image = image_processing.load_image(png_original_image_path)
            low_boundary = (0, 0, 0)
            high_boundary = (65, 30, 54)
            shapes = image_processing.generate_mask_around_potential_building(segmented_image, low_boundary, high_boundary)
            
            for i, shape in enumerate(shapes):
                plot_id_nosuffix = file_path.with_suffix('')
                plot_id_i = str(plot_id_nosuffix) + f"_{i}.png"
                building_file_path = os.path.join(potential_roofs_dir, plot_id_i)

                building_mask = image_processing.apply_mask(original_image, shape)
                extracted_rectangle = image_processing.crop_rectangle_around_plot(building_mask, with_mask=True)
                extracted_buliding = image_processing.crop_plot(extracted_rectangle)
                resized_building = image_processing.resize_image(extracted_buliding)
                image_processing.save_image(building_file_path, resized_building)

                potential_roofs.append(Image(name=str(file_path), new_name=plot_id_i, potential_building=building_mask, potential_building_transformed=resized_building, is_building=False))

    return potential_roofs


def generate_model(csv_file_path: str, 
                   potential_roofs_dir: str, 
                   num_epochs: int, 
                   model_path: str, 
                   metrics_dir: str) -> None:
    """
    Prepares the data, trains model and tests his performance
    
    Args:
        csv_file_path (str): A path to the csv file storing potential roofs labels
        potential_roofs_dir (str): A path to the directory storing potential roofs
        num_epochs (int): Number of epochs
        model_path (str): A path to storing the model
        metrics_dir (str): A directory path for storing metrics
    """
    train_loader, test_loader, labels = prepare_from_csv_and_dir(csv_file_path, potential_roofs_dir)
    train_model(train_loader, num_epochs=num_epochs, save_path=model_path)
    predictions = predict(test_loader, model_path)
    evaluate_metrics = EvaluateMetrics(labels, predictions)
    evaluate_metrics.calculate_accuracy()
    evaluate_metrics.display_conf_matrix(metrics_dir)

def prediction(potential_roofs: List[Img], model_path: str, img_dir: str) -> None:
    """
    Prepares the data to roof predictions, selects only images with predicted roof, 
    combines them and saves to specified directory

    Args:
        potential_roofs (List[Img]): List of Image classes, storing potential roof parameters
        model_path (str): Path to the saved model
        img_dir (str): Directory path to store extracted roofs
    """
    dataloader = prepare_for_prediction(potential_roofs)
    pred = predict(dataloader, model_path)
    img_processing = ImageProcessing()
    roofs = dict()

    for i, Img in enumerate(potential_roofs):
        if pred[i] == 1:
            masked_roof = Img.potential_building
            cropped_roof = img_processing.crop_plot(masked_roof)

            if Img.name in roofs:
                roofs[Img.name].append(cropped_roof)
            else:
                roofs[Img.name] = [cropped_roof]

    for key, values in roofs.items():
        roofs_extracted = img_processing.connect_images(values)
        save_path = os.path.join(img_dir, key)
        img_processing.save_image(save_path, roofs_extracted)