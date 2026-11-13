import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
train = pd.read_csv("data/train.csv")
holidays = pd.read_csv("data/holidays_events.csv")

# Date conversion
train["date"] = pd.to_datetime(train["date"])
holidays["date"] = pd.to_datetime(holidays["date"])

# Holiday feature
holiday_dates = holidays["date"].drop_duplicates()

train["is_holiday"] = (
    train["date"]
    .isin(holiday_dates)
    .astype(int)
)

# Filter dataset
df = train[
    (train["store_nbr"] == 1)
    &
    (train["family"] == "GROCERY I")
].copy()

df = df.sort_values("date")

# Select features
features = [
    "sales",
    "onpromotion",
    "is_holiday"
]

data = df[features]

print("Dataset shape:")
print(data.shape)

print("\nFeature Preview")
print(data.head())

# Scale data
scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)

print("\nScaled shape:")
print(scaled_data.shape)
import numpy as np


def create_sequences(data, sequence_length):

    X = []
    y = []

    for i in range(len(data) - sequence_length):

        X.append(
            data[i:i + sequence_length]
        )

        y.append(
            data[i + sequence_length, 0]
        )

    return (
        np.array(X),
        np.array(y)
    )


SEQ_LENGTH = 30

X, y = create_sequences(
    scaled_data,
    SEQ_LENGTH
)

print("\nSequence Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

# Train-test split

split_size = int(len(X) * 0.8)

X_train = X[:split_size]
X_test = X[split_size:]

y_train = y[:split_size]
y_test = y[split_size:]

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Naive forecasting
naive_predictions = X_test[:, -1, 0]

naive_mae = mean_absolute_error(
    y_test,
    naive_predictions
)

naive_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        naive_predictions
    )
)

print("\nNaive Baseline")
print("MAE:", naive_mae)
print("RMSE:", naive_rmse)

# Moving average forecast

moving_avg_predictions = np.mean(
    X_test[:, -7:, 0],
    axis=1
)

moving_mae = mean_absolute_error(
    y_test,
    moving_avg_predictions
)

moving_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        moving_avg_predictions
    )
)

print("\nMoving Average Baseline")
print("MAE:", moving_mae)
print("RMSE:", moving_rmse)

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SalesDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        return (
            self.X[index],
            self.y[index]
        )


train_dataset = SalesDataset(
    X_train,
    y_train
)

test_dataset = SalesDataset(
    X_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("\nDatasets Created Successfully")

import torch.nn as nn


class LSTMForecast(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        output = output[:, -1, :]

        output = self.fc(output)

        return output.squeeze()


lstm_model = LSTMForecast(
    input_size=3
)

print("\nLSTM Model Built")

def train_model(
    model,
    loader,
    optimizer,
    criterion,
    epochs=10
):

    model.train()

    for epoch in range(epochs):

        running_loss = 0

        for X_batch, y_batch in loader:

            optimizer.zero_grad()

            predictions = model(
                X_batch
            )

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"Loss: {running_loss/len(loader):.4f}"
        )
        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(
            lstm_model.parameters(),
            lr=0.001
        )

        print("\nTraining LSTM...")

        train_model(
            lstm_model,
            train_loader,
            optimizer,
            criterion,
            epochs=10
        )

        # Loss Function
        criterion = nn.MSELoss()

        # Optimiser
        optimizer = torch.optim.Adam(
            lstm_model.parameters(),
            lr=0.001
        )
def train_model(
    model,
    loader,
    optimizer,
    criterion,
    epochs=10
):

    model.train()

    for epoch in range(epochs):

        running_loss = 0

        for X_batch, y_batch in loader:

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_loss = (
            running_loss /
            len(loader)
        )

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"Loss: {avg_loss:.4f}"
        )
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    lstm_model.parameters(),
    lr=0.001
)

print("\nTraining LSTM...")

train_model(
    lstm_model,
    train_loader,
    optimizer,
    criterion,
    epochs=10
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

# Evaluation

lstm_model.eval()

predictions = []
actuals = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        pred = lstm_model(X_batch)

        predictions.extend(
            pred.numpy()
        )

        actuals.extend(
            y_batch.numpy()
        )

predictions = np.array(
    predictions
)

actuals = np.array(
    actuals
)

lstm_mae = mean_absolute_error(
    actuals,
    predictions
)

lstm_rmse = np.sqrt(
    mean_squared_error(
        actuals,
        predictions
    )
)

print("\nLSTM Results")

print(
    "MAE:",
    lstm_mae
)

print(
    "RMSE:",
    lstm_rmse
)
lstm_mape = np.mean(
    np.abs(
        (actuals - predictions)
        / actuals
    )
) * 100

print(
    "MAPE:",
    lstm_mape
)

class GRUForecast(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        output, _ = self.gru(x)

        output = output[:, -1, :]

        output = self.fc(output)

        return output.squeeze()

gru_model = GRUForecast(
    input_size=3
)

print("\nGRU Model Built")

gru_optimizer = torch.optim.Adam(
    gru_model.parameters(),
    lr=0.001
)

print("\nTraining GRU...")

train_model(
    gru_model,
    train_loader,
    gru_optimizer,
    criterion,
    epochs=10
)

gru_model.eval()

gru_predictions = []
gru_actuals = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        pred = gru_model(X_batch)

        gru_predictions.extend(
            pred.numpy()
        )

        gru_actuals.extend(
            y_batch.numpy()
        )

gru_predictions = np.array(
    gru_predictions
)

gru_actuals = np.array(
    gru_actuals
)

gru_mae = mean_absolute_error(
    gru_actuals,
    gru_predictions
)

gru_rmse = np.sqrt(
    mean_squared_error(
        gru_actuals,
        gru_predictions
    )
)

print("\nGRU Results")

print(
    "MAE:",
    gru_mae
)

print(
    "RMSE:",
    gru_rmse
)

plt.figure(figsize=(12,5))

plt.plot(
    gru_actuals,
    label="Actual Sales"
)

plt.plot(
    gru_predictions,
    label="GRU Prediction"
)

plt.title(
    "GRU Forecast vs Actual Sales"
)

plt.xlabel("Observations")
plt.ylabel("Scaled Sales")

plt.legend()

plt.tight_layout()

plt.savefig(
    "plots/gru_predictions.png"
)

plt.show()
torch.save(
    lstm_model.state_dict(),
    "saved_models/lstm_model.pth"
)

torch.save(
    gru_model.state_dict(),
    "saved_models/gru_model.pth"
)

gru_mape = np.mean(
    np.abs(
        (gru_actuals - gru_predictions)
        / gru_actuals
    )
) * 100

print("MAPE:", gru_mape)

print(
    "RMSE:",
    gru_rmse
)

import matplotlib.pyplot as plt

models = [
    "Naive",
    "Moving Avg",
    "LSTM",
    "GRU"
]

mae_values = [
    1.059,
    0.756,
    0.849,
    0.603
]

plt.figure(figsize=(8,5))

plt.bar(
    models,
    mae_values
)

plt.title(
    "MAE Comparison of Forecasting Models"
)

plt.ylabel(
    "MAE"
)

plt.savefig(
    "plots/model_comparison_mae.png"
)

plt.show()