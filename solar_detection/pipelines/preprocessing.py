import selenium.common
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from solar_detection.api.operations import (
    DirectoryOperations,
    MapOperations,
    GSOperations,
)


def plot(csv_file: str, website: str = "", images_dir: str = "", id_col: str = "id"):
    def handle_field_id(field_id):
        """Process a single field_id in a separate thread."""
        nonlocal map_oper
        try:
            field_id = field_id.replace("-", "/")
            print(f"Processing: {field_id}")
            map_oper.handle_plot(field_id)
        except selenium.common.exceptions.UnexpectedAlertPresentException:
            print(f"Alert present while processing {field_id}")
        except Exception as e:
            print(f"Error while processing {field_id}: {e}")

    map_oper = MapOperations(website=website, image_path=images_dir)
    map_oper.prepare_map()
    # Load CSV and extract IDs
    df = pd.read_csv(csv_file)
    field_ids = df[id_col]

    # Create the images directory
    dir_oper = DirectoryOperations()
    dir_oper.create_directory(images_dir)

    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(handle_field_id, field_id       )
            for field_id in field_ids
        ]
        for future in as_completed(futures):
            try:
                future.result()  # Raise exceptions if any occurred
            except Exception as e:
                print(f"Exception in thread: {e}")
    map_oper.quit_map()
