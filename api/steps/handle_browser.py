import logging
from selenium import webdriver

from src.browser_operations import BrowserInitiation, BrowserOperations

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

    def find_plot(self):
        """
        Search for specified plot
        """
        try:
            browse_oper = BrowserOperations(self.driver)
            browse_oper.find_by_id("szukaj_dzialki_north")
            browse_oper.switch_iframe("frame_szukaj_dzgb")
            # browse_oper.find_by_class_name('ui-button-icon-primary ui-icon marker1')
            browse_oper.input_text("iddz", "022305_2.0010.9/2")
            browse_oper.switch_to_default_iframe()
            browse_oper.find_by_id("button_rozwin_27")
            browse_oper.find_by_id("check_widoczna_29")
            browse_oper.find_by_id("check_widoczna_30")
        except MapException as e:
            logging.error("Error in plot searching: %s", e)
