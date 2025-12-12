import torch
from torch.utils.data import DataLoader, Dataset

class SimpleDataset(Dataset):
    """
    Basic dataset wrapper for numpy → PyTorch tensors.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (N, ...)
    y : np.ndarray
        Integer labels of shape (N,)
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def train_torch(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs=15,
    batch_size=64,
    lr=1e-3,
    wandb_run=None,
    save_best_path=None,
):
    """
    Train a classification model using PyTorch.

    Parameters
    ----------
    model : torch.nn.Module
        Model with forward() returning logits of shape (N, num_classes).

    X_train : np.ndarray
        Training input features.

    y_train : np.ndarray
        Training labels as integers.

    X_val : np.ndarray, optional
        Validation input features.

    y_val : np.ndarray, optional
        Validation labels.

    epochs : int
        Number of training epochs.

    batch_size : int
        Mini-batch size.

    lr : float
        Learning rate for Adam optimizer.

    wandb_run : wandb.Run, optional
        If provided, logs training and validation metrics.

    save_best_path : str, optional
        If provided, saves the best model (highest val accuracy).

    Returns
    -------
    torch.nn.Module
        The trained model.
    """
    
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create training dataset + loader
    ds_train = SimpleDataset(X_train, y_train)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    # Apply Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = -1

    # Training loop
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0

        # Mini-batch training
        for xb, yb in loader_train:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(loader_train)

        # Validation (if provided)
        val_acc = None
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
                yv = torch.tensor(y_val, dtype=torch.long).to(device)
                logits = model(Xv)
                preds = logits.argmax(dim=1)
                val_acc = (preds == yv).float().mean().item()

        # Logging: console + Weights&Biases
        if wandb_run:
            wandb_run.log({
                "train_loss": avg_train_loss,
                "epoch": ep,
                "val_accuracy": val_acc,
            })

        print(f"Epoch {ep}/{epochs}: loss={avg_train_loss:.4f}, val_acc={val_acc}")

        # Save best model (if enabled)
        if save_best_path and val_acc is not None:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"New best model discovered! Saving to {save_best_path}")
                model.save(save_best_path)

    return model

