from src.roofs_detection.roof_detector import RoofDetector
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch


def train_model(train_loader: torch.Tensor, num_epochs: int = 25, save_path: str = None) -> None:
    """
    Trains the model and saves into specified path

    Args:
        train_loader: The training data DataLoader instance
        num_epochs: The number of epochs during training
        save_path: A path to save the model
    """       
    model = RoofDetector()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, labels in tqdm(train_loader):
            optimizer.zero_grad()
            
            outputs = model(data)
            loss = criterion(outputs.view(-1), labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predicted = (outputs.squeeze() > 0.5)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    if save_path:
        torch.save(model.state_dict(), save_path)
