import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
from typing import List

def train_model(model: torch.Tensor, 
                train_loader: torch.Tensor, 
                num_epochs: int = 25, 
                lr: float = 0.0001, 
                save_path: str | None = None
                ) -> List[float]:
    """
    Trains the model and saves into specified path

    Args:
        model (torch.Tensor): Model instance
        train_loader: The training data DataLoader instance
        num_epochs: The number of epochs during training
        lr (float): Learning rate
        save_path: A path to save the model
    
    Returns:
        List[float]: List of accuracy history during training
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(num_epochs/3), gamma=0.5)
    history = []

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

            predicted = outputs.squeeze() > 0.5
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        history.append(accuracy)
        scheduler.step()

    if save_path:
        torch.save(model.state_dict(), save_path)

    return history