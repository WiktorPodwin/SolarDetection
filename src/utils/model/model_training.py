import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
from typing import List
from torch.amp import GradScaler, autocast

def train_model(model: torch.Tensor, 
                train_loader: torch.Tensor, 
                num_epochs: int = 25, 
                lr: float = 0.0001, 
                accumulation_steps: int = 5,
                save_path: str | None = None
                ) -> List[float]:
    """
    Trains the model and saves into specified path

    Args:
        model (torch.Tensor): Model instance
        train_loader: The training data DataLoader instance
        num_epochs: The number of epochs during training
        lr (float): Learning rate
        accumulation_steps (int): Number of batches to accumulate gradients before performing an optimizer step
        save_path: A path to save the model
    
    Returns:
        List[float]: List of accuracy history during training
    """
    decision = torch.cuda.is_available()
    if decision:
        scaler = GradScaler()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(num_epochs/3)), gamma=0.8)
    history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        optimizer.zero_grad()
        for i, (data, labels) in enumerate(tqdm(train_loader)):
            if decision:
                with autocast():
                    outputs = model(data)
                    loss = criterion(outputs.view(-1), labels.float())
                scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
            else:
                outputs = model(data)
                loss = criterion(outputs.view(-1), labels.float())

                loss = loss /accumulation_steps
                loss.backward()
            
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps

            predicted = outputs.squeeze() > 0.5
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            del loss, outputs

        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        history.append(accuracy)
        scheduler.step()

    if save_path:
        torch.save(model.state_dict(), save_path)

    return history