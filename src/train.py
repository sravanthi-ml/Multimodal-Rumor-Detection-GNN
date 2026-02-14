import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model import MultimodalRumorModel
from src.dataset import RumorDataset


# -----------------------------------
# Configuration
# -----------------------------------
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------
# Training Function
# -----------------------------------
def train():

    # Initialize Dataset (placeholder path)
    train_dataset = RumorDataset(split="train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model
    model = MultimodalRumorModel()
    model.to(DEVICE)

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Optimizer (as described in manuscript)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # -----------------------------------
    # Training Loop
    # -----------------------------------
    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for batch in train_loader:

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            pixel_values = batch["pixel_values"].to(DEVICE)
            metadata = batch["metadata"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask, pixel_values, metadata)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    train()

