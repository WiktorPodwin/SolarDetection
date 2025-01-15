import selenium.common

from typing import List
from solar_detection.api.operations import (
    DirectoryOperations,
    MapOperations,
    GSOperations,
)

def plot(
    field_ids: List[str | int] | int | str,
    website: str = "",
    images_dir: str = "",
    plot_id: str = "281411_2.0001.295",
):

    if not isinstance(field_ids, list):
        field_ids = [field_ids]

    dir_oper = DirectoryOperations()
    dir_oper.create_directory(images_dir)
    # dir_oper.clear_directory(images_dir)

    map_oper = MapOperations(website=website, image_path=images_dir)
    map_oper.prepare_map()

    for field_id in field_ids:
        try:
            map_oper.handle_plot(f"{plot_id}/{field_id}")
        except selenium.common.exceptions.UnexpectedAlertPresentException:
            print("Alert present")

    map_oper.quit_map()