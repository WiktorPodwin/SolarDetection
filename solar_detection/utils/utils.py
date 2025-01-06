import logging
from typing import Any, List
import cv2
import pandas as pd
import selenium.common
import torch
from solar_detection.api.operations import (
    DirectoryOperations,
    MapOperations,
    GSOperations,
)
from solar_detection.datatypes import Image


def rotate_image(image: Image, angle: int) -> Image:
    """
    Rotate an image by a given angle.

    Args:
        image (Image): The image to rotate.
        angle (int): The angle to rotate the image by.

    Returns:
        Image: The rotated image.
    """
    cv2_image = cv2.imread(image.location)
    rotated_image = cv2.rotate(cv2_image, angle)
    cv2.imwrite(image.location, rotated_image)
    return Image(
        name=image.name,
        location=image.location,
        rectangle_shape=image.rectangle_shape,
        mask=image.mask,
    )


def rotate_images_in_dir(images_dir: str, images: List[str], angle: int) -> None:
    """
    Rotate all images in a directory by a given angle.

    Args:
        images_dir (str): The directory containing the images.
        angle (int): The angle to rotate the images by.
    """
    for image in images:
        image_path = f"{images_dir}/{image}"
        rotated_image_path = f"{images_dir}/rotated-{angle}-{image}"
        cv2_image = cv2.imread(image_path)
        rotated_image = cv2.rotate(cv2_image, angle)
        cv2.imwrite(rotated_image_path, rotated_image)
        print(f"Rotated image: {rotated_image_path}")
        print(f"Original image: {image_path}")


def rotate_for_training(images_dir: str, angles: List[int]) -> None:
    """Create rotated images for training

    Args:
        images_dir (str): Directory containing the images.
    """
    dir_oper = DirectoryOperations()
    images = dir_oper.list_directory(images_dir)
    for angle in angles:
        rotate_images_in_dir(images_dir, images, angle)


def fill_csv_for_training(csv_file: str, angles: List[int]) -> None:
    """
    Fill a CSV file with the names and tags of the rotated images.

    Args:
        csv_file (str): Path to the CSV file.
        images_dir (str): Directory containing the images.
    """
    df = pd.read_csv(csv_file)
    df_copy = df.copy()
    for angle in angles:
        for _, row in df.iterrows():
            new_row = row.copy()
            new_row["id"] = f"rotated-{angle}-{row['id']}"
            df_copy.loc[len(df_copy)] = new_row

    print("New rows:")
    for i, row in df_copy.iterrows():
        print(row["id"])

    df_copy.to_csv(csv_file, index=False)


def fill_data_for_training(csv_file: str, images_dir: str, angles: List[int]) -> None:
    fill_csv_for_training(csv_file, angles)
    rotate_for_training(images_dir, angles)


def plot(
    field_ids: List[str | int] | int | str,
    csv_file: str = "",
    website: str = "",
    images_dir: str = "",
    plot_id: str = "281411_2.0001.295",
):

    if not isinstance(field_ids, list):
        field_ids = [field_ids]

    dir_oper = DirectoryOperations()
    dir_oper.create_directory(images_dir)
    dir_oper.clear_directory(images_dir)

    map_oper = MapOperations(website=website, image_path=images_dir)
    map_oper.prepare_map()

    for field_id in field_ids:
        try:
            map_oper.handle_plot(f"{plot_id}/{field_id}")
        except selenium.common.exceptions.UnexpectedAlertPresentException:
            print("Alert present")

    map_oper.quit_map()


def upload_to_gs(
    project_id: str = "",
    bucket_name: str = "",
    source_file_path: str = "",
    destination_blob_name: str = "",
):
    gs_oper = GSOperations(project_id=project_id, bucket_name=bucket_name)
    gs_oper.upload_file(
        source_file=source_file_path, destination_blob_name=destination_blob_name
    )


def get_torch_device() -> torch.device:
    """Get the Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device


def load_csv_df(csv_file: str, header: int | List[Any] | None = None) -> pd.DataFrame:
    return pd.read_csv(csv_file, skipinitialspace=True, header=header)


def upload_csv_file(csv_file: str, images_params: List[Image]) -> None:
    plot_names = [image_param.new_name[:-4] for image_param in images_params]
    df = pd.DataFrame(plot_names, columns=["plot_id"])
    df.to_csv(csv_file, index=False)
    logging.info("Successfully saved the csv file: %s", csv_file)


def apply_pred_to_csv(
    csv_file: str, pred: List[int], labels: List[str], col_name: str
) -> None:
    """
    Updates a CSV file with a new column based on predictions.

    Args:
        csv_file (str): Path to the CSV file.
        pred (List[int]): List of predictions.
        labels (List[str]): List of plot labels.
        col_name (str): Name of the new column to be added/changed.
    """
    plots = [plot.rsplit("_", 1)[0] + "-" + plot.rsplit("_", 1)[1] for plot in labels]
    df = pd.read_csv(csv_file)

    df[col_name] = 0
    plot_to_pred = dict(zip(plots, pred))

    df[col_name] = df["id"].apply(lambda x: plot_to_pred.get(x, 0))
    df.to_csv(csv_file, index=False)
