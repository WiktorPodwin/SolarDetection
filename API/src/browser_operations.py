from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import logging

class BrowserInitiation:
    """
    Class for web service initiation
    """
    def initiate(self, website: str) -> webdriver.Chrome:
        """
        Initiates Chrome servie and connects with specified website
        
        Args:
            website: Website URL
        Returns:
            webdriver.Chrome: Connected Chrome service with specified website
        """
        try:
            options = Options()
            options.add_experimental_option("detach", True)
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                          options=options)
            driver.get(website)
            driver.maximize_window()
            return driver
        except Exception as e:
            logging.error(f"Error in service initiation: {e}")
            raise e

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
        except Exception as e:
            logging.error(f"Error while closing service: {e}")
            raise e

    def find_by_id(self, id: str) -> None:
        """
        Clicks on an element by specified ID 
        
        Args:
            id: Id of searched element 
        """
        try:
            WebDriverWait(self.driver, 10) \
                .until(EC.element_to_be_clickable((By.ID, id))) \
                .click()
        except Exception as e:
            logging.error(f"Error while searching by id {id}: {e}")
            raise e
    
    def find_by_class_name(self, class_name: str) -> None:
        """
        Clicks on element by specified class name
            
        Args:
            class_name: Class name of searched element
        """    
        try:
            WebDriverWait(self.driver, 10) \
                .until(EC.element_to_be_clickable((By.CLASS_NAME, class_name))) \
                .click()
        except Exception as e:
            logging.error(F"Error while searching element by class name {class_name}: {e}")
            raise e
    
    def switch_iframe(self, id) -> None:
        """
        Switches iframe to the specified one

        Args:
            id: Input field id
        """
        try:
            iframe = WebDriverWait(self.driver, 10).until(EC.visibility_of_element_located((By.ID, id)))
            self.driver.switch_to.frame(iframe)
        except Exception as e:
            logging.error(f"Error while swittching iframe {id}: {e}")
            raise e 

    def switch_to_default_iframe(self) -> None:
        """
        Switches iframe to the default one
        """
        try:
            self.driver.switch_to.default_content()
        except Exception as e:
            logging.error(f"Error while switching iframe to the default one: {e}")
            raise e

    
    def input_text(self, id: str, text: str) -> None:
        """
        Inputs the text into field

        Args:
            id: Input field id
            text: Text to input
        """
        try:
            element = WebDriverWait(self.driver, 10).until(EC.visibility_of_element_located((By.ID, id)))
            element.clear()
            element.send_keys(text)
            element.send_keys(Keys.ENTER)
        except Exception as e:
            logging.error(f"Error while inserting text to the element {id}: {e}")
            raise e