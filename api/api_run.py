import logging
import pandas as pd
import selenium.common
from .src import DirectoryOperations, MapOperations, GSOperations


def plot(
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

    for i in range(0, 100):
        try:
            map_oper.handle_plot(f"{plot_id}/{i}")
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
        source_file_path=source_file_path, destination_blob_name=destination_blob_name
    )
