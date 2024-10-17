import logging
import os
import ee
import requests


class GEEImageFetcher:
    """
    A class for fetching satellite images of house roofs from Google Earth Engine.
    """

    def __init__(self, ee_credentials_path=None):
        """
        Initializes the class with the path to the Earth Engine credentials file.

        Args:
            ee_credentials_path (str): The path to the Earth Engine credentials file.
        """
        # Initialize the Earth Engine API
        if ee_credentials_path:
            # Authenticate using the provided credentials file
            ee.Authenticate(ee_credentials_path)
        else:
            ee.Authenticate()
            ee.Initialize(project=os.getenv("GCP_PROJECT_ID"))
        self.dataset = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate("2020-01-01", "2020-01-30")
            # Pre-filter to get less cloudy granules.
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .map(self.mask_s2_clouds)
        )

    def fetch_images(self, roi, start_date, end_date, collection_name, bands, scale=10):
        """
        Fetches satellite images for a given region of interest (ROI), date range,
        collection name, bands, and scale.

        Args:
            roi (ee.Geometry): The region of interest.
            start_date (str): The start date of the image acquisition (e.g., '2024-01-01').
            end_date (str): The end date of the image acquisition (e.g., '2024-12-31').
            collection_name (str): The name of the satellite image collection (e.g., 'LANDSAT/LC08_L1_TOA').
            bands (list): A list of band names to include in the image (e.g., ['B2', 'B3', 'B4']).
            scale (int, optional): The scale of the image in meters. Defaults to 10.

        Returns:
            ee.ImageCollection: An Earth Engine ImageCollection containing the fetched images.
        """

        # Load the image collection
        image_collection = (
            ee.ImageCollection(collection_name)
            .filterBounds(roi)
            .filterDate(start_date, end_date)
        )

        # Select the desired bands
        image_collection = image_collection.select(bands)

        # Scale the image to the desired resolution
        # image_collection = image_collection.scale(scale)

        return image_collection

    def export_image_to_cloud_storage(self, image, filename, description, region=None):
        """
        Exports an Earth Engine image to Google Drive.

        Args:
            image (ee.Image): The Earth Engine image to export.
            filename (str): The name of the exported file.
            description (str): A description of the exported file.
            region (ee.Geometry, optional): The region to export. Defaults to the image's footprint.
        """

        task = ee.batch.Export.image.toCloudStorage(
            image=image,
            description=description,
            bucket=os.getenv("GS_BUCKET_NAME"),
            fileNamePrefix=filename,
            region=region,
        )
        task.start()

    def get_image_url(self, image: ee.image.Image) -> str:
        """
        Gets a URL for visualizing an Earth Engine image.

        Args:
            image (ee.Image): The Earth Engine image.

        Returns:
            str: The URL for visualizing the image.
        """

        map_id = image.getMapId(
            {"bands": ["SR_B2", "SR_B3", "SR_B4"], "min": 0, "max": 1}
        )
        url = f"https://earthengine.google.com/map/{map_id['token']}"
        return url

    def mask_s2_clouds(self, image: ee.image.Image):
        """Masks clouds in a Sentinel-2 image using the QA band.

        Args:
            image (ee.Image): A Sentinel-2 image.

        Returns:
            ee.Image: A cloud-masked Sentinel-2 image.
        """
        qa = image.select("QA60")

        # Bits 10 and 11 are clouds and cirrus, respectively.
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11

        # Both flags should be set to zero, indicating clear conditions.
        mask = (
            qa.bitwiseAnd(cloud_bit_mask)
            .eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )

        return image.updateMask(mask).divide(10000)


class GoogleMapsImageFetcher:
    """
    A class for fetching satellite images from Google Maps.
    """

    def __init__(self, api_key):
        """
        Initializes the class with the Google Maps API key.

        Args:
            api_key (str): The Google Maps API key.
        """
        self._api_key = api_key

    def fetch_image(self, location: str, zoom=20, size="640x640"):
        """
        Fetches a satellite image from Google Maps for a given location.

        Args:
            location (str): The location to fetch the image for (e.g., "New York City").
            zoom (int, optional): The zoom level of the image. Defaults to 15.
            size (str, optional): The size of the image in pixels (e.g., "640x640"). Defaults to "640x640".

        Returns:
            bytes: The image data.
        """
        url = f"https://maps.googleapis.com/maps/api/staticmap?center={location}&zoom={zoom}&size={size}&maptype=satellite&key={self._api_key}"
        response = requests.get(url, timeout=10)
        logging.debug("Response status code: %s", response.status_code)
        return response.content
