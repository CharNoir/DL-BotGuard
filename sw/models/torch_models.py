import torch
import torch.nn as nn

class GRUMouseNet(nn.Module):
    """
    A GRU-based classifier for mouse-dynamics sequences.

    Each input sample is a sequence of shape (T, F) where:
        T = window length (e.g., 100)
        F = number of features per event (e.g., dx, dy, dt, speed)

    The model processes the full sequence through a GRU stack and uses
    the final hidden state as a summary representation for classification.
    """
    def __init__(
        self,
        input_dim, 
        hidden_dim=64,
        num_layers=1,
        bidirectional=True,
        dropout=0.1,
    ):
        super().__init__()
        
        # Store config so the model can reconstruct itself later
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        # GRU encoder
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output dimension doubles for bidirectional GRUs (forward + backward)
        gru_out_dim = hidden_dim * (2 if bidirectional else 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def get_config(self):
        """Return model hyperparameters so the model can be rebuilt."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout,
        }

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): shape (B, T, F)

        Returns:
            Tensor: shape (B, 2) logits
        """
        out, _ = self.gru(x)

        last = out[:, -1, :]

        logits = self.classifier(last)
        return logits
    
    def save(self, filename):
        """
        Save model weights + configuration.
        """
        torch.save({
            "config": self.get_config(),
            "state_dict": self.state_dict()
        }, filename)
        print(f"Model saved: {filename}")
        
    @classmethod
    def load(cls, filename, map_location="cpu"):
        """
        Load a model saved with `save()`.

        Returns:
            GRUMouseNet: reconstructed model with weights loaded.
        """
        checkpoint = torch.load(filename, map_location=map_location)

        config = checkpoint["config"]
        model = cls(**config)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        print(f"Loaded model from {filename}")
        return model
    
    @classmethod
    def from_config(cls, config: dict, *, input_dim: int):
        """
        Create a GRUMouseNet instance from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Dictionary containing model hyperparameters.
            Required keys:
                - "hidden_dim" (int)
                - "num_layers" (int)
                - "bidirectional" (bool)
                - "dropout" (float)
        input_dim : int
            The size of each input feature vector.

        Returns
        -------
        GRUMouseNet
            A fully constructed model instance.

        Raises
        ------
        KeyError
            If required parameters are missing from the configuration.
        TypeError
            If parameters exist but have incorrect types.
        """
        REQUIRED_PARAMS = [
            "hidden_dim",
            "num_layers",
            "bidirectional",
            "dropout"
        ]
        
        missing = [k for k in REQUIRED_PARAMS if k not in config]
        if missing:
            raise KeyError(
                f"Missing required config parameters for GRUMouseNet: {missing}"
            )

        return cls(
            input_dim=input_dim,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            bidirectional=config["bidirectional"],
            dropout=config["dropout"]
        )
    
class CNNMouseNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_filters: int = 64,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(num_filters, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)

        x = torch.max(x, dim=2).values
        logits = self.fc(x)
        return logits