import logging
from typing import List
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

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
        # logging.info("Test Accuracy: %.4f", accuracy)
        print("Test Accuracy:", round(accuracy, 3))
    
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

    def display_history(self, history: List[float], dir_path: str) -> None:
        """
        Display history of loss function during training

        Args:
            history (List[float]): List of luss function history during training
            dir_path (str): Path to directory for storing graph
        """
        epochs = [epoch + 1 for epoch in range(len(history))]

        plt.figure(figsize=(12, 7))
        plt.plot(epochs, history)
        plt.title("Loss function history")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid()
        plt.tight_layout()

        plt.xticks(range(1, len(epochs) + 1, 5))
        
        save_path = dir_path + "/history.png"
        plt.savefig(save_path)
        plt.close()
        logging.info("Successfully saved loss history graph to the file: %s", save_path)
