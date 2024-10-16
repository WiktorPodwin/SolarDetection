import logging
from src.browser_operations import BrowserInitiation, BrowserOperations
from src.data_operations import DirectoryOperations
from src.convert_symbols import ConvertSymbols

class MapException(BaseException):
    pass

class MapOperations:
    """
    Handles map operations
    """

    def __init__(self, website: str) -> None:
        """
        Args:
            website: Website's URL
        """
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

    def handle_plot(self, plot_id: str = None):
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
            dir_oper.create_directory("../data/images")

            convert_symbols = ConvertSymbols()
            new_id = convert_symbols.slash_into_dash(plot_id)

            browse_oper.take_screenshot("map_canvas", "../data/images/", new_id)
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

