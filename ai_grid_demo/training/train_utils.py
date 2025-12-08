import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                config, model_path: str):
    """Generic training loop with early stopping and learning rate scheduling. Returns training history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)  # Reduce LR every 10 epochs
    criterion = nn.MSELoss()  # Or BCE for classification
    device = torch.device('cpu')  # Use CPU for demo
    model.to(device)

    best_loss = float('inf')
    patience_counter = 0

    # Store training history
    history = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rates': [],
        'best_epoch': 0
    }

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x, y, edge_index = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x, edge_index)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = evaluate_model(model, val_loader, criterion, device)
        train_loss_avg = train_loss / len(train_loader)

        # Store in history
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch+1}: Train Loss {train_loss_avg:.4f}, Val Loss {val_loss:.4f}, LR {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            history['best_epoch'] = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("Early stopping")
                break

        # Step the scheduler
        scheduler.step()

    return history


def evaluate_model(model: nn.Module, data_loader: DataLoader, criterion, device):
    """Evaluate model on data loader."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            x, y, edge_index = batch
            x, y = x.to(device), y.to(device)
            output = model(x, edge_index)
            loss = criterion(output, y)
            total_loss += loss.item()
    return total_loss / len(data_loader)

