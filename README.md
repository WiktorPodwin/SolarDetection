# SolarDetection Project

## Overview
SolarDetection is a project aimed at identifying and analyzing solar panel installations using satellite imagery. The project leverages advanced image processing techniques and integrates with external geospatial services to provide accurate and up-to-date information.

## Features
- **Satellite Imagery Analysis**: Utilizes high-resolution satellite images to detect solar panels.
- **Data Visualization**: Presents data in an easy-to-understand format.
- **Integration with Geospatial Services**: Enhances data accuracy and provides additional context.

## Integration with Google Earth
SolarDetection integrates with Google Earth to provide a visual representation of detected solar panels. By overlaying detection results on Google Earth's satellite imagery, users can easily verify and explore the locations of solar installations.

### Steps to Integrate:
1. Export detection results in KML format.
2. Open Google Earth and import the KML file.
3. Visualize and analyze the detected solar panels on the map.

## Integration with polska.geoportal2.pl and geoportal.gov.pl
geoportal.gov.pl is a Polish geospatial service that provides detailed maps and geospatial data. SolarDetection integrates with geoportal.gov.pl to enhance the accuracy of solar panel detection and provide additional geospatial context.

### Steps to Integrate:
1. Access geoportal.gov.pl API to retrieve geospatial data.
2. Use the data to refine detection algorithms and improve accuracy.
3. Overlay detection results on geoportal.gov.pl maps for detailed analysis.

## Getting Started
To get started with SolarDetection, follow these steps:
1. Install the required dependencies: `pip install -r requirements.txt`
2. Run the server: `python main.py`

## Contributing
We welcome contributions from the community. Please read our [contributing guidelines](CONTRIBUTING.md) for more information.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or feedback, you can reach us on Github.

## Links and Resources

* [Google Earth Engine Dataset](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED#colab-python)
* [Google Earth Engine Documentation](https://developers.google.com/earth-engine/guides)
* [Geoportal 2](https://polska.geoportal2.pl/map/www/mapa.php?mapa=polska)
* [Remote Sensing Image Analysis (RSiM) Group @ TU Berlin code](https://git.tu-berlin.de/rsim)
* [Remote Sensing Image Analysis Datasets](https://bigearth.net/#downloads)
* [Large Datasets Collection](https://github.com/Agri-Hub/Callisto-Dataset-Collection)
* [Reliable Building Footprints Change Extraction](https://github.com/liaochengcsu/BCE-Net)
* [Simulated Multimodal Aerial Remote Sensing dataset](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/smars)
* [Esri Geoportal Server](https://github.com/Esri/geoportal-server-catalog)
* [Esri Geoportal Server Search Catalog](https://gpt.geocloud.com/geoportal2/#searchPanel)