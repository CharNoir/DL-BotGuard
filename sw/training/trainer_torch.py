import torch
from torch.utils.data import DataLoader, Dataset

class SimpleDataset(Dataset):
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
    epochs=15,
    batch_size=64,
    lr=1e-3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ds = SimpleDataset(X_train, y_train)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {ep+1} loss={total_loss:.4f}")

    return model
