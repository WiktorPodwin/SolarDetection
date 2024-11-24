import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import BaseConfig
from src.utils import get_torch_device, load_csv_df
from .classifier import SolarPanelClassifier, SolarPanelDataset


def setup_training():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    csv_dataset = load_csv_df(BaseConfig.LOCATION_FIELD_CSV_DIR)

    # Paths to training and validation datasets
    train_dir = BaseConfig.SOLAR_DETECTION_TRAINING_DIR
    test_dir = BaseConfig.SOLAR_DETECTION_TEST_DIR

    # Define data transformations
    transform = transforms.Compose(
        [
            transforms.Resize((150, 150)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # Normalize to [-1, 1]
        ]
    )

    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model, loss function, and optimizer
    device = get_torch_device()
    model = SolarPanelClassifier().to(device)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model = train_model(
        model, device, train_loader, val_loader, criterion, optimizer, num_epochs=20
    )

    # Save the trained model
    torch.save(model.state_dict(), "solar_panel_classifier.pth")

    # Load the model for inference
    # model.load_state_dict(torch.load("solar_panel_classifier.pth"))


def train_model(
    model, device, train_loader, val_loader, criterion, optimizer, num_epochs=20
) -> SolarPanelClassifier:
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)  # Match output shape
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Compute accuracy
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Validation Loss: {val_loss/len(val_loader):.4f}, "
              f"Validation Accuracy: {correct/total:.4f}")
    return model
