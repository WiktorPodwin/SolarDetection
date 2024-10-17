from .handle_browser import MapOperations

def run_browser(website: str='https://polska.geoportal2.pl/map/www/mapa.php?mapa=polska'):
    map_oper = MapOperations(website=website)
    map_oper.prepare_map()
    map_oper.find_plot()