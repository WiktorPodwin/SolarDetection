import logging
from .browser_operations import BrowserInitiation, BrowserOperations
from .data_operations import DirectoryOperations


class MapException(BaseException):
    pass


class MapOperations:
    """
    Handles map operations
    """

    def __init__(self, website: str, image_path: str) -> None:
        """
        Args:
            website: Website's URL
        """
        self.image_path = image_path
        self.driver = BrowserInitiation(website=website).driver
        self.browser_operation = BrowserOperations(self.driver)

    def prepare_map(self) -> None:
        """
        Initiates map and sets up configuration
        """
        try:
            browser_oper = BrowserOperations(self.driver)
            browser_oper.find_by_class_name(class_name="ui-dialog-buttonset")
            browser_oper.find_by_id(element_id="check_widoczna_8")
        except MapException as e:
            logging.error("Error in map preparing: %s", e)

    def handle_plot(self, plot_id: str = "") -> None:
        """
        Manages browser to get specified plot and take screenshot of it

        Args:
            plot_id: Unique identificator of plot
        """
        try:
            browse_oper = BrowserOperations(self.driver)
            # Search button
            browse_oper.find_by_id("szukaj_dzialki_north")
            # Search iframe
            browse_oper.switch_iframe("frame_szukaj_dzgb")
            # Search plot
            browse_oper.input_text("iddz", plot_id)
            # Default iframe
            browse_oper.switch_to_default_iframe()
            # Unmark checkbox
            browse_oper.find_by_id("check_widoczna_25", False)
            # Unmark checkbox
            browse_oper.find_by_id("check_widoczna_27", False)

            dir_oper = DirectoryOperations()
            dir_oper.create_directory(self.image_path)


            browse_oper.take_screenshot("map_canvas", self.image_path, plot_id.replace("/", "_"))
        except MapException as e:
            logging.error("Error in plot searching: %s", e)

    def quit_map(self) -> None:
        """
        Closes the window with opened map
        """
        try:
            self.browser_operation.close_service()
        except MapException as e:
            logging.error("Error in closing the service: %s", e)

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