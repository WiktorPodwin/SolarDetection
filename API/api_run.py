from steps.handle_browser import MapOperations

map_oper = MapOperations(website='https://polska.geoportal2.pl/map/www/mapa.php?mapa=polska')
map_oper.prepare_map()
map_oper.find_plot()
