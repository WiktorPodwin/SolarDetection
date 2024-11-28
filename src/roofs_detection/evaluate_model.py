from sklearn.metrics import confusion_matrix, accuracy_score
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt
import logging

class EvaluateMetrics:
    """
    Class to define evaluate metrics
    """
    def __init__(self, labels: List[int], predictions: List[int]):
        """
        Args:
            labels: (List[int]): List of true labels
            predictions (List[int]): List of predictions
        """
        self.labels = labels
        self.predictions = predictions

    def calculate_accuracy(self) -> None:
        """
        Calculates accuracy score and display the value
        """
        accuracy = accuracy_score(self.labels, self.predictions)
        logging.info(f"Test Accuracy: {accuracy:.4f}")

    def display_conf_matrix(self, dir_path: str) -> None:
        """
        Calculates and saves confusion matrix as a graph
        
        Args:
            dir_path (str): Path to directory for storing graph
        """
        conf_matrix = confusion_matrix(self.labels, self.predictions)
        
        plt.figure(figsize=(12, 7))
        sns.heatmap(conf_matrix, cmap='Blues', annot=True, fmt='d')
        plt.title("Confusion matrix")
        plt.xlabel("True labels")
        plt.ylabel("Predictions")

        save_path = dir_path + "/conf_matrix.png"
        plt.savefig(save_path)
        plt.close()
        logging.info("Successfully saved confusion matrix to the file: %s", save_path)

