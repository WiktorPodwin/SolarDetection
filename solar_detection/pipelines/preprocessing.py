import selenium.common
import pandas as pd

from typing import List
from solar_detection.api.operations import (
    DirectoryOperations,
    MapOperations,
    GSOperations,
)

def plot(
    csv_file: str,
    website: str = "",
    images_dir: str = "",
    id_col: str = "id"
):
    df = pd.read_csv(csv_file)
    df = df[id_col]

    dir_oper = DirectoryOperations()
    dir_oper.create_directory(images_dir)

    map_oper = MapOperations(website=website, image_path=images_dir)
    map_oper.prepare_map()

    for field_id in df:
        try:
            field_id = field_id.replace("-", "/")
            print(field_id)
            # map_oper.handle_plot(f"{plot_id}/{field_id}")
            map_oper.handle_plot(field_id)
        except selenium.common.exceptions.UnexpectedAlertPresentException:
            print("Alert present")

    map_oper.quit_map()