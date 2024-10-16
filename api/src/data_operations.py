import os
import glob
import logging

class DirectoryOperations:
    """
    Class to handle directory operations
    """
    def create_directory(self, directory_path: str) -> None:
        """
        Creates a directory
        
        Args:
            directory_path: Path to the directory
        """
        try:
            os.makedirs(directory_path, exist_ok=True)
        except OSError as e:
            logging.error("Error in creating directory %s: %s", directory_path, e)
        
    def clear_directory(self, directory_path: str) -> None:
        """
        Deletes all files from directory
        
        Args:
            directory_path: Path to the dorectory
        """
        try:
            files = glob.glob(directory_path + "/*")
            for file in files:
                os.remove(file)
        except OSError as e:
            logging.error("Error while deleting %s: %s", directory_path, e)
        except FileNotFoundError:
            logging.error("Directory %s not found", directory_path)