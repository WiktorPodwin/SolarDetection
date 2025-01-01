import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
from typing import List
from torch.amp import GradScaler, autocast
import pandas as pd
# from sklearn.model_selection import StratifiedKFold

def train_model(device: torch.device, 
                model: torch.Tensor, 
                train_loader: torch.Tensor, 
                class_distribution: pd.Series,
                num_epochs: int = 25, 
                lr: float = 0.0001,
                step_size: int = None, 
                accumulation_steps: int = 1,
                save_path: str | None = None
                ) -> List[float]:
    """
    Trains the model and saves into specified path

    Args:
        device (torch.device): Calculation units
        model (torch.Tensor): Model instance
        train_loader: The training data DataLoader instance
        class_distribution (pd.Series): Data classes distribution
        num_epochs: The number of epochs during training
        lr (float): Learning rate
        step_size (int): Step size in learning rate scheduler
        accumulation_steps (int): Number of batches to accumulate gradients before performing an optimizer step
        save_path: A path to save the model
    
    Returns:
        List[float]: List of loss function history during training
    """
    decision = torch.cuda.is_available()
    if decision:
        scaler = GradScaler()
    
    pos_weight = torch.tensor([class_distribution[0] / class_distribution[1]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    
    if step_size is None:
        step_size = max(1, int(num_epochs/3))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.8)
    history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        optimizer.zero_grad()
        for i, (data, labels) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            labels = labels.to(device)
            if decision:
                with autocast("cuda"):
                    outputs = model(data)
                    loss = criterion(outputs.view(-1), labels.float())
                    loss = loss / accumulation_steps
                scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(data)
                loss = criterion(outputs.view(-1), labels.float())

                loss = loss /accumulation_steps
                loss.backward()
            
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            running_loss += loss.item()

            predicted = outputs.squeeze() > 0.5
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        history.append(avg_loss)
        scheduler.step()

    if save_path:
        torch.save(model.state_dict(), save_path)
    return history




# def train_model(device: torch.device, 
#                 model: torch.Tensor, 
#                 train_loader: torch.Tensor, 
#                 class_distribution: pd.Series,
#                 num_epochs: int = 25, 
#                 lr: float = 0.0001,
#                 step_size: int = None, 
#                 accumulation_steps: int = 5,
#                 save_path: str | None = None
#                 ) -> List[float]:
#     """
#     Trains the model and saves into specified path

#     Args:
#         device (torch.device): Calculation units
#         model (torch.Tensor): Model instance
#         train_loader: The training data DataLoader instance
#         class_distribution (pd.Series): Data classes distribution
#         num_epochs: The number of epochs during training
#         lr (float): Learning rate
#         step_size (int): Step size in learning rate scheduler
#         accumulation_steps (int): Number of batches to accumulate gradients before performing an optimizer step
#         save_path: A path to save the model
    
#     Returns:
#         List[float]: List of accuracy history during training
#     """
#     n_splits = 3
#     decision = torch.cuda.is_available()
#     if decision:
#         scaler = GradScaler()

#     # Create pos_weight for imbalanced data
#     pos_weight = torch.tensor([class_distribution[0] / class_distribution[1]]).to(device)
#     criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    
#     # If step_size is None, set it to the number of epochs
#     if step_size is None:
#         step_size = num_epochs

#     # Initialize StratifiedKFold
#     kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
#     history = []

#     # Assuming the DataLoader has a list of tuples (data, labels), extract data and labels
#     all_data = []
#     all_labels = []
#     for data, labels in train_loader:
#         all_data.append(data)
#         all_labels.append(labels)

#     all_data = torch.cat(all_data, dim=0)
#     all_labels = torch.cat(all_labels, dim=0)

#     # Perform K-Fold Cross Validation
#     for fold, (train_idx, val_idx) in enumerate(kf.split(all_data, all_labels)):
#         print(f"\nFold {fold + 1}/{n_splits}")

#         # Split data into train and validation sets
#         train_data, val_data = all_data[train_idx], all_data[val_idx]
#         train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]

#         # Create DataLoader for this fold
#         train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
#         val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
#         train_loader_fold = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
#         val_loader_fold = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

#         # Reset the model for each fold
#         # model.reset_parameters()

#         # Training loop for each fold
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.8)

#         for epoch in range(num_epochs):
#             model.train()
#             running_loss = 0.0
#             correct = 0
#             total = 0

#             optimizer.zero_grad()
#             for i, (data, labels) in enumerate(tqdm(train_loader_fold)):
#                 data = data.to(device)
#                 labels = labels.to(device)
#                 if decision:
#                     with autocast("cuda"):
#                         outputs = model(data)
#                         loss = criterion(outputs.view(-1), labels.float())
#                         loss = loss / accumulation_steps
#                     scaler.scale(loss).backward()

#                     if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader_fold):
#                         scaler.step(optimizer)
#                         scaler.update()
#                         optimizer.zero_grad()
#                 else:
#                     outputs = model(data)
#                     loss = criterion(outputs.view(-1), labels.float())
#                     loss = loss / accumulation_steps
#                     loss.backward()

#                     if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader_fold):
#                         optimizer.step()
#                         optimizer.zero_grad()

#                 running_loss += loss.item()
#                 # running_loss += loss.item() * accumulation_steps

#                 predicted = outputs.squeeze() > 0.5
#                 correct += (predicted == labels).sum().item()
#                 total += labels.size(0)

#             avg_loss = running_loss / len(train_loader_fold)
#             accuracy = correct / total

#             print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

#             # Step the scheduler
#             scheduler.step()

#         # Record accuracy for this fold
#         history.append(accuracy)

#     # Optionally, save the final model after all folds
#     if save_path:
#         torch.save(model.state_dict(), save_path)

#     return history