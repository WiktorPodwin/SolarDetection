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
            browser_oper.find_by_id(element_id="check_widoczna_7")
            browser_oper.find_by_id(element_id="check_widoczna_8")
            browser_oper.find_by_id("button_rozwin_27")
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
            # Unmark checkboxes
            browse_oper.find_by_id("check_widoczna_29", False)
            browse_oper.find_by_id("check_widoczna_30", False)
            # Search button
            browse_oper.find_by_id("szukaj_dzialki_north")
            # Search iframe
            browse_oper.switch_iframe("frame_szukaj_dzgb")
            # Clean mark
            browse_oper.find_by_id("wyczysc_marker")
            # Default iframe
            browse_oper.switch_to_default_iframe()
            # Close search window
            browse_oper.find_element_by_xpath("/html/body/div[35]/div[1]/a/span")            

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
