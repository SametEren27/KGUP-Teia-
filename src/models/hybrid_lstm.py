import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

MODEL_DIR = "src/models"
LSTM_WEIGHTS_PATH = os.path.join(MODEL_DIR, "lstm_residual.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "residual_scaler.joblib")


class ResidualLSTM(nn.Module):

  def __init__(
      self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1
  ):
    super().__init__()
    self.lstm = nn.LSTM(
        input_size, hidden_size, num_layers, batch_first=True
    )
    self.fc = nn.Linear(hidden_size, 1)

  def forward(self, x):
    out, _ = self.lstm(x)
    return self.fc(out[:, -1, :])


def create_sequences(data: np.ndarray, seq_len: int = 24):
  xs, ys = [], []
  for i in range(len(data) - seq_len):
    xs.append(data[i : (i + seq_len)])
    ys.append(data[i + seq_len])
  return np.array(xs), np.array(ys)


def train_residual_lstm(
    residuals: np.ndarray, seq_len: int = 24, epochs: int = 15, lr: float = 0.01
):
  """LightGBM artıklarını (residuals) eğitir ve ağırlıkları kaydeder."""
  from sklearn.preprocessing import StandardScaler

  os.makedirs(MODEL_DIR, exist_ok=True)

  # Normalizasyon
  scaler = StandardScaler()
  scaled_residuals = scaler.fit_transform(residuals.reshape(-1, 1))

  X, y = create_sequences(scaled_residuals, seq_len)
  X_tensor = torch.tensor(X, dtype=torch.float32)
  y_tensor = torch.tensor(y, dtype=torch.float32)

  dataset = TensorDataset(X_tensor, y_tensor)
  loader = DataLoader(dataset, batch_size=32, shuffle=True)

  model = ResidualLSTM()
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  model.train()
  for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_x, batch_y in loader:
      optimizer.zero_grad()
      pred = model(batch_x)
      loss = criterion(pred, batch_y)
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()

  torch.save(model.state_dict(), LSTM_WEIGHTS_PATH)
  joblib.dump(scaler, SCALER_PATH)
  print(
      f"✅ LSTM Artık Modeli eğitildi ve kaydedildi (Son Epoch Kaybı:"
      f" {epoch_loss/len(loader):.4f})"
  )
  return model, scaler


def predict_residual(recent_residuals: np.ndarray, seq_len: int = 24) -> float:
  """Son 24 saatin artık değerlerini alıp bir sonraki saatin hatasını tahmin eder."""
  if not (os.path.exists(LSTM_WEIGHTS_PATH) and os.path.exists(SCALER_PATH)):
    return 0.0

  scaler = joblib.load(SCALER_PATH)
  model = ResidualLSTM()
  model.load_state_dict(
      torch.load(LSTM_WEIGHTS_PATH, map_location=torch.device("cpu"))
  )
  model.eval()

  scaled = scaler.transform(recent_residuals[-seq_len:].reshape(-1, 1))
  inp = torch.tensor(scaled.reshape(1, seq_len, 1), dtype=torch.float32)

  with torch.no_grad():
    pred_scaled = model(inp).item()

  return float(scaler.inverse_transform([[pred_scaled]])[0][0])


if __name__ == "__main__":
  # Modül bağımsız çalıştığında test eğitimi
  mock_residuals = np.random.normal(0, 300, 200)
  train_residual_lstm(mock_residuals)