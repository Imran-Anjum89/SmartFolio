"""
PyTorch LSTM Sequence Model for Return Forecasting in SmartFolio.
Captures sequential temporal dynamics across multi-day feature trajectories.
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from config.settings import RANDOM_SEED, set_seed

class LSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Use final time step hidden state
        last_step = lstm_out[:, -1, :]
        out = self.fc(last_step)
        return out.squeeze(-1)

class SmartFolioLSTM:
    def __init__(
        self,
        lookback_window: int = 20,
        hidden_dim: int = 64,
        num_layers: int = 2,
        lr: float = 0.001,
        epochs: int = 25,
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        set_seed(RANDOM_SEED)
        self.lookback_window = lookback_window
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[LSTMNet] = None
        self.input_dim: int = 0
        self.mean_: np.ndarray = np.array([])
        self.std_: np.ndarray = np.array([])

    def _create_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sliding windows of shape (N - lookback + 1, lookback, features)."""
        xs = []
        ys = []
        n = len(X)
        if n < self.lookback_window:
            # Pad with first row if fewer rows than lookback
            pad = np.repeat(X[:1], self.lookback_window - n, axis=0)
            X = np.vstack([pad, X])
            n = len(X)

        for i in range(n - self.lookback_window + 1):
            xs.append(X[i : i + self.lookback_window])
            if y is not None:
                ys.append(y[i + self.lookback_window - 1])

        X_seq = np.array(xs, dtype=np.float32)
        y_seq = np.array(ys, dtype=np.float32) if y is not None else None
        return X_seq, y_seq

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SmartFolioLSTM":
        """Normalize features, construct sequences, and train PyTorch LSTM."""
        X_mat = X.values.astype(np.float32)
        y_vec = y.values.astype(np.float32)
        
        # Standardize features
        self.mean_ = np.nanmean(X_mat, axis=0)
        self.std_ = np.nanstd(X_mat, axis=0) + 1e-8
        X_norm = np.nan_to_num((X_mat - self.mean_) / self.std_)

        X_seq, y_seq = self._create_sequences(X_norm, y_vec)
        self.input_dim = X_seq.shape[2]

        self.model = LSTMNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        ).to(self.device)

        dataset = TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_seq))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate forecasts from sequences."""
        if self.model is None:
            raise ValueError("LSTM model is not fitted yet.")

        self.model.eval()
        X_mat = X.values.astype(np.float32)
        X_norm = np.nan_to_num((X_mat - self.mean_) / self.std_)
        X_seq, _ = self._create_sequences(X_norm, None)

        with torch.no_grad():
            tensor_x = torch.from_numpy(X_seq).to(self.device)
            preds = self.model(tensor_x).cpu().numpy()

        # If sequences resulted in fewer rows than input, pad the beginning with the first prediction
        diff = len(X) - len(preds)
        if diff > 0:
            pad = np.repeat(preds[:1], diff)
            preds = np.concatenate([pad, preds])
        return preds
