import os
import re
from pathlib import Path
from typing import List

import requests



def replace_line(file_name: str, line_num: int, text: str):
    """ Replace a specific line in a file

    Args:
        file_name (str): file path
        line_num (int): line to replace.
        text (str): Text to put in the line.

    Returns:
        bool: If the line was replaced will return True.
    """
    with open(file_name, 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        lines[line_num] = text
        f.writelines(lines)
    with open(file_name, 'w', encoding='utf-8') as fw:    
        fw.writelines(lines)
    return True

def searching_all_files(path:str = ".", pattern : str="\.tiff$|\.csv$") -> List:
    dirpath = Path(path)
    assert(dirpath.is_dir())
    file_list = []
    for x in dirpath.iterdir():
        if x.is_file() and re.search(pattern, x.as_posix()):
            file_list.append(x.absolute().as_posix())
        elif x.is_dir():
            file_list.extend(searching_all_files(x, pattern))
    return file_list

def raster_to_vector(image, geom):
    vector_img = image.unmask(0).reduceToVectors(
        geometry = geom,
        scale = 10,
        bestEffort =True
    )
    return vector_img.getInfo()
    


def get_save_image(location: str, zoom=20, size="640x640"):
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={location}&zoom={zoom}&size={size}&maptype=satellite&key={os.getenv("GOOGLE_MAPS_API_KEY")}"
    response = requests.get(url, timeout=10)
    img = response.content
    print(img)
    print("Image fetched from Google Maps")
    with open("image2.jpg", "wb", encoding='utf-8') as f:
        f.write(img)
    print("Image saved to 'image.jpg'")
