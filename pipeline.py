import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pandas as pd
from sklearn.model_selection import train_test_split


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data.reset_index(drop=True).values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=3, nhead=8, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)

        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
            for _ in range(num_layers)
        ])
        self.variable_selection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, input_size)
        )
        self.output_layer = nn.Linear(hidden_size, 7)  # 7-day predictions

    def forward(self, x):
        x = self.input_layer(x)
        x = x.permute(1, 0, 2)

        for i, gate in enumerate(self.gates):
            transformer_out = self.transformer_encoder(x)
            gate_values = gate(transformer_out)
            x = gate_values * transformer_out + (1 - gate_values) * x

        x = x.permute(1, 0, 2)
        output = self.output_layer(x[:, -1, :])
        return output


class ReformerModel(nn.Module):
    def __init__(self, input_size=25, hidden_size=256, num_layers=4, num_classes=3):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)

        self.reformer_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])

        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)

        self.risk_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size // 4), nn.ReLU(),
            nn.Linear(hidden_size // 4, num_classes)
        )

        self.residual_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])

    def forward(self, x):
        x = self.input_layer(x)

        for reformer_layer, residual_layer in zip(self.reformer_layers, self.residual_layers):
            residual = residual_layer(x)
            x = reformer_layer(x) + residual

        x, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = x.squeeze(1)

        risk_scores = self.risk_layers(x)
        return risk_scores


# Training function for TFT
# Adjusted Training Function for TFT
def train_tft_model(model, data_loader, epochs=5):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for weather_batch in data_loader:
            if weather_batch.dim() == 2:
                weather_batch = weather_batch.unsqueeze(1)

            optimizer.zero_grad()
            output = model(weather_batch)

            # Ensure that the target size matches the model output size
            target = weather_batch[:, -1, :7]  # Adjusted to take the last step with 7 outputs

            if torch.isnan(output).any() or torch.isnan(target).any():
                print("NaN detected in output or target.")
                continue

            # Check if output and target shapes match
            if output.shape != target.shape:
                print(f"Shape mismatch - Output shape: {output.shape}, Target shape: {target.shape}")
                continue

            loss = criterion(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            print(f"Epoch {epoch + 1}, TFT Loss: {loss.item()}")



# Function to generate TFT predictions
def get_tft_predictions(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for weather_batch in data_loader:
            if weather_batch.dim() == 2:
                weather_batch = weather_batch.unsqueeze(1)

            output = model(weather_batch)
            predictions.append(output)
    return torch.cat(predictions, dim=0)


# Training function for Reformer
def train_reformer_model(model, weather_predictions, vitals_data_loader, epochs=5):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for weather_pred_batch, vitals_batch in zip(DataLoader(weather_predictions, batch_size=16), vitals_data_loader):
            combined_input = torch.cat((weather_pred_batch, vitals_batch), dim=1)
            optimizer.zero_grad()

            output = model(combined_input)

            target = torch.randint(0, 3, (combined_input.size(0),))  # Adjust this based on your actual targets
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}, Reformer Loss: {loss.item()}")


# Initialize models
tft_model = TemporalFusionTransformer()
reformer_model = ReformerModel()

# Sample data
weather_data = pd.read_csv(
    r"D:\Siddharth\Textbooks\Final Yr\Project\open-meteo-38.00N97.00E4318m.csv",
    on_bad_lines='skip',
    usecols=['temperature_2m (°C)', 'relative_humidity_2m (%)', 'dew_point_2m (°C)',
              'apparent_temperature (°C)', 'wind_speed_180m (km/h)', 'temperature_180m (°C)']
)
vitals_data = pd.read_csv(
    r"D:\Siddharth\Textbooks\Final Yr\Project\HHI Data 2024 United States.csv",
    on_bad_lines='skip',
    usecols=['P_CHD', 'PR_CHD', 'F_CHD', 'P_DIABETES', 'PR_DIABETES', 'F_DIABETES',
              'P_COPD', 'PR_COPD', 'F_COPD', 'P_ASTHMA', 'PR_ASTHMA', 'F_ASTHMA',
              'P_OBS', 'PR_OBS', 'F_OBS', 'P_MNTHL', 'PR_MNTHL', 'F_MNTHL']
)

weather_data = weather_data.dropna()
vitals_data = vitals_data.dropna()

# Split datasets into training and validation sets
min_size = min(len(weather_data), len(vitals_data))
weather_data = weather_data[:min_size].reset_index(drop=True)
vitals_data = vitals_data[:min_size].reset_index(drop=True)

# Use train_test_split to create training and validation sets
weather_train, weather_val = train_test_split(weather_data.values, test_size=0.2, random_state=42)
vitals_train, vitals_val = train_test_split(vitals_data.values, test_size=0.2, random_state=42)

# Initialize CustomDataset with DataFrames
weather_train_dataset = CustomDataset(pd.DataFrame(weather_train))
weather_val_dataset = CustomDataset(pd.DataFrame(weather_val))
vitals_train_dataset = CustomDataset(pd.DataFrame(vitals_train))
vitals_val_dataset = CustomDataset(pd.DataFrame(vitals_val))

# Define data loaders
weather_loader = DataLoader(weather_train_dataset, batch_size=16, shuffle=True)
vitals_loader = DataLoader(vitals_train_dataset, batch_size=16, shuffle=True)

# Use the validation dataset to create a DataLoader for TFT predictions
tft_predictions = get_tft_predictions(tft_model, DataLoader(weather_val_dataset, batch_size=16))

# Continue with training the Reformer model
train_tft_model(tft_model, weather_loader)
train_reformer_model(reformer_model, tft_predictions, DataLoader(vitals_val_dataset, batch_size=16))
