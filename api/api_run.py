import pandas as pd
from steps.handle_browser import MapOperations
from src.data_operations import DirectoryOperations

def plot(csv_file: str = "../data/plots.csv", website: str = 'https://polska.geoportal2.pl/map/www/mapa.php?mapa=polska'):
    df = pd.read_csv(csv_file, skipinitialspace=True)
    print(df.head())

    dir_oper = DirectoryOperations()
    dir_oper.create_directory("../data/images/")
    dir_oper.clear_directory("../data/images/")

    map_oper = MapOperations(website=website)
    map_oper.prepare_map()

    for plot_id in df["id"]:
        map_oper.handle_plot(plot_id)

    map_oper.quit_map()

if __name__ == "__main__":
    plot()