import logging
from typing import Any, List
import pandas as pd
import selenium.common
import torch
from src.api.operations import DirectoryOperations, MapOperations, GSOperations
from src.datatypes import Image


def plot(
    field_ids: List[str | int],
    csv_file: str = "",
    website: str = "",
    images_dir: str = "",
):
    plot_id = "281411_2.0001.295"
    df = pd.read_csv(csv_file, skipinitialspace=True)
    logging.debug(df.head())

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

def apply_pred_to_csv(csv_file: str, pred: List[int], labels: List[str], col_name: str) -> None:
    """
    Updates a CSV file with a new column based on predictions.

    Args:
        csv_file (str): Path to the CSV file.
        pred (List[int]): List of predictions.
        labels (List[str]): List of plot labels.
        col_name (str): Name of the new column to be added/changed.
    """
    plots = [plot.rsplit('_', 1)[0] + '-' + plot.rsplit('_', 1)[1] for plot in labels]
    df = pd.read_csv(csv_file)

    df[col_name] = 0
    plot_to_pred = dict(zip(plots, pred))

    df[col_name] = df["id"].apply(lambda x: plot_to_pred.get(x, 0))
    df.to_csv(csv_file, index=False)

