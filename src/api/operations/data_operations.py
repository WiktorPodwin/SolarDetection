import os
import glob
import logging
from google.cloud import storage


# this class should contain static methods as it does not require any class level attributes
class DirectoryOperations:
    """
    Class to handle directory operations
    """

    @staticmethod
    def list_directory(directory_path: str) -> list:
        """
        Lists all files in the directory

        Args:
            directory_path: Path to the directory
            returns: List of files in the directory
        """
        try:
            files = glob.glob(directory_path + "/*")
            file_names = [os.path.basename(file) for file in files]
            return file_names
        except FileNotFoundError:
            logging.error("Directory %s not found", directory_path)
            return []
        except OSError as e:
            logging.error("Error while listing files in %s: %s", directory_path, e)
            return []

    @staticmethod
    def create_directory(directory_path: str) -> None:
        """
        Creates a directory

        Args:
            directory_path: Path to the directory
        """
        try:
            os.makedirs(directory_path, exist_ok=True)
        except OSError as e:
            logging.error("Error in creating directory %s: %s", directory_path, e)

    @staticmethod
    def clear_directory(directory_path: str) -> None:
        """
        Deletes all files from directory

        Args:
            directory_path: Path to the dorectory
        """
        try:
            files = glob.glob(directory_path + "/*")
            for file in files:
                os.remove(file)
        except FileNotFoundError:
            logging.error("Directory %s not found", directory_path)
        except OSError as e:
            logging.error("Error while deleting %s: %s", directory_path, e)


class GSOperations:
    """
    Class to handle Google Cloud Storage operations
    """

    def __init__(self, project_id, bucket_name):
        """Initialize the GCPStorage class with the specified bucket name."""
        self.bucket_name = bucket_name
        self.client = storage.Client(project_id)
        self.bucket = self.client.get_bucket(bucket_name)

    def get_files(self, file_names):
        """
        get files in binary format from the GCP bucket.

        Args:
            file_names: List of file names to get.

        Returns:
            List of file objects.
        """
        try:
            file_objects = []
            for file_name in file_names:
                blob = self.bucket.blob(file_name)
                file_object = blob.download_as_bytes()
                file_objects.append(file_object)
            return file_objects
        except Exception as e:
            logging.error("Error getting files from GCP: %s", e)
            return []

    def upload_file(
        self, source_file, destination_blob_name, source_file_type: str = "local_file"
    ):
        """
        Upload a file to the GCP bucket.

        Args:
            source_file (str or file-like object): Path to the local file or file-like object.
            destination_blob_name (str): The name of the destination blob (file) in the bucket.
            source_file_type (str, optional): Type of the source file, either "local_file" for a file path or "file_object" for a file-like object. Defaults to "local_file".

        Returns:
            str: The public URL of the uploaded file, or None if an error occurred.
        """
        try:
            # Upload the file
            blob = self.bucket.blob(destination_blob_name)
            if source_file_type == "local_file":
                blob.upload_from_filename(source_file)
            elif source_file_type == "file_object":
                blob.upload_from_file(source_file)

            # Make the blob publicly accessible (optional)
            blob.make_public()

            logging.info("File %s uploaded to %s.", source_file, destination_blob_name)

            # Return the public URL
            return blob.public_url

        except Exception as e:
            logging.error("Error uploading file to GCP: %s", e)
            return None

    def download_file(self, blob_name, destination_file_path):
        """
        Download a file from the GCP bucket.

        Args:
            blob_name: The name of the blob (file) in the bucket.
            destination_file_path: The local path where the file will be saved.
        """
        try:
            # Retrieve the blob (file) from the bucket
            blob = self.bucket.blob(blob_name)

            # Download the file to the specified local destination
            blob.download_to_filename(destination_file_path)

            logging.info("File %s downloaded to %s.", blob_name, destination_file_path)

        except Exception as e:
            logging.error("Error downloading file from GCP: %s", e)

    def list_files(self, prefix=None):
        """
        List all files in the GCP bucket.

        Args:
            prefix: Optional prefix to filter files by.
            returns: A list of blob names in the bucket.
        """
        try:
            # List all blobs (files) in the bucket
            blobs = self.bucket.list_blobs(prefix=prefix)

            file_list = [blob.name for blob in blobs]
            return file_list

        except Exception as e:
            logging.error("Error listing files in GCP bucket: %s", e)
            return []

    def delete_file(self, blob_name):
        """
        Delete a file from the GCP bucket.

        Args:
            blob_name: The name of the blob (file) in the bucket.
        """
        try:
            # Retrieve the blob and delete it
            blob = self.bucket.blob(blob_name)
            blob.delete()

            logging.info("Blob %s deleted.", blob_name)

        except Exception as e:
            logging.error("Error deleting file from GCP: %s", e)
