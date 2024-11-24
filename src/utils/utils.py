import logging
from typing import Any, List
import pandas as pd
import selenium.common
import torch
from src.api.operations import DirectoryOperations, MapOperations, GSOperations


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
