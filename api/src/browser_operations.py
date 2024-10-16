import logging
import os
from attrs import define
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time


@define
class Driver:
    driver: webdriver.Chrome


class BrowserException(BaseException):
    pass


class BrowserInitiation:
    """
    Class for web service initiation
    """

    def __init__(self, website: str) -> None:
        """
        Initiates Chrome service and connects with specified website

        Args:
            website: Website URL
        Returns:
            webdriver.Chrome: Connected Chrome service with specified website
        """
        try:
            options = Options()
            options.add_experimental_option("detach", True)
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), options=options
            )
        except BrowserException as e:
            logging.error("Error in service initiation: %s", e)

        driver.get(website)
        driver.maximize_window()
        self.driver: webdriver.Chrome = driver


class BrowserOperations:
    """
    Class for browser operations
    """

    def __init__(self, driver) -> None:
        """
        Args:
            driver: Active Chrome service
        """
        self.driver = driver

    def close_service(self) -> None:
        """
        Close Chrome service

        Args:
            driver: Actice web service
        """
        try:
            self.driver.quit()
        except BrowserException as e:
            logging.error("Error while closing service: %s", e)

    def find_by_id(self, element_id: str, checkbox_selected: bool = None) -> None:
        """
        Clicks on an element by specified ID

        Args:
            element_id: ID of searched element
            checkbox_selected: True if checkbox needs to be selected, False otherweise
        """
        try:
            if checkbox_selected is not None:
                time.sleep(1.5)
            object = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.ID, element_id))
            )
            if checkbox_selected is None or (checkbox_selected == True and object.is_selected() == False) or (checkbox_selected == False and object.is_selected() == True):
                object.click()
        except BrowserException as e:
            logging.error("Error while searching by id %s: %s", element_id, e)

    def find_by_class_name(self, class_name: str) -> None:
        """
        Clicks on element by specified class name

        Args:
            class_name: Class name of searched element
        """
        try:
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CLASS_NAME, class_name))
            ).click()
        except BrowserException as e:
            logging.error(
                "Error while searching element by class name %s: %s", class_name, e
            )

    def switch_iframe(self, element_id) -> None:
        """
        Switches iframe to the specified one

        Args:
            element_id: Input field id
        """
        try:
            iframe = WebDriverWait(self.driver, 10).until(
                EC.visibility_of_element_located((By.ID, element_id))
            )
            self.driver.switch_to.frame(iframe)
        except BrowserException as e:
            logging.error("Error while swittching iframe %s: %s", element_id, e)

    def switch_to_default_iframe(self) -> None:
        """
        Switches iframe to the default one
        """
        try:
            self.driver.switch_to.default_content()
        except BrowserException as e:
            logging.error("Error while switching iframe to the default one: %s", e)

    def input_text(self, element_id: str, text: str) -> None:
        """
        Inputs the text into field

        Args:
            element_id: Input field id
            text: Text to input
        """
        try:
            element = WebDriverWait(self.driver, 10).until(
                EC.visibility_of_element_located((By.ID, element_id))
            )
            element.clear()
            element.send_keys(text)
            element.send_keys(Keys.ENTER)
        except BrowserException as e:
            logging.error(
                "Error while inserting text to the element %s: %s", element_id, e
            )
    
    def take_screenshot(self, element_id: str, directory_path: str, file_name: str) -> None:
        """
        Takes a screenshot of selected element and saves into specified path
        
        Args:
            element_id: ID of element to select
            directory_path: Path to the directory which will store image
            file_name: Name of file to save the image
        """
        try:
            element = WebDriverWait(self.driver, 10).until(
                EC.visibility_of_element_located((By.ID, element_id))
            )
            full_path = directory_path + file_name + ".png"
            element.screenshot(full_path)
        except BrowserException as e:
            logging.error(
                "Error in a screenshot process: %s", e
                )



# class GoogleEarthEngine:
#     """Class For initializing Earth Engine"""

#     def __init__(self) -> None:
#         ee.Authenticate()
#         ee.Initialize(project=os.getenv("GCP_PROJECT_ID"))
