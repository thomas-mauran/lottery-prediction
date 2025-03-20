import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

class LotteryDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class LotteryPredictor(nn.Module):
    def __init__(self, input_size):
        super(LotteryPredictor, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        
        # Separate branches for main numbers and lucky number
        self.main_numbers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 5)  # 5 main numbers
        )
        
        self.lucky_number = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)  # 1 lucky number
        )
        
    def forward(self, x):
        shared_features = self.shared(x)
        main_pred = self.main_numbers(shared_features)
        lucky_pred = self.lucky_number(shared_features)
        return main_pred, lucky_pred

def load_data(file_path):
    df = pd.read_csv(file_path, sep=';')
    df['date_de_tirage'] = pd.to_datetime(df['date_de_tirage'], format='%d/%m/%Y')
    return df

def prepare_data(df):
    # Extract features from the date
    df['year'] = df['date_de_tirage'].dt.year
    df['month'] = df['date_de_tirage'].dt.month
    df['day'] = df['date_de_tirage'].dt.day
    df['day_of_week'] = df['date_de_tirage'].dt.dayofweek
    
    # Normalize temporal features
    df['year'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
    df['month'] = df['month'] / 12
    df['day'] = df['day'] / 31
    df['day_of_week'] = df['day_of_week'] / 6
    
    # Add cyclical features for month, day, and day_of_week
    df['month_sin'] = np.sin(2 * np.pi * df['month'])
    df['month_cos'] = np.cos(2 * np.pi * df['month'])
    df['day_sin'] = np.sin(2 * np.pi * df['day'])
    df['day_cos'] = np.cos(2 * np.pi * df['day'])
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'])
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'])
    
    features = ['year', 
                'month_sin', 'month_cos',
                'day_sin', 'day_cos',
                'day_of_week_sin', 'day_of_week_cos']
    
    # Normalize targets
    main_numbers = df[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].values / 49.0
    lucky_number = df['numero_chance'].values.reshape(-1, 1) / 10.0
    
    return (df[features].values, 
            np.hstack([main_numbers, lucky_number]))

def train_model(train_loader, val_loader, model, num_epochs=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            main_targets = batch_targets[:, :5].to(device)
            lucky_targets = batch_targets[:, 5:].to(device)
            
            optimizer.zero_grad()
            main_pred, lucky_pred = model(batch_features)
            
            loss = criterion(main_pred, main_targets) + criterion(lucky_pred, lucky_targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                main_targets = batch_targets[:, :5].to(device)
                lucky_targets = batch_targets[:, 5:].to(device)
                
                main_pred, lucky_pred = model(batch_features)
                loss = criterion(main_pred, main_targets) + criterion(lucky_pred, lucky_targets)
                val_loss += loss.item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    model.load_state_dict(best_model)
    return model

def predict_unique_numbers(raw_predictions, num_range, count):
    """Ensure predictions are unique and within range"""
    predictions = np.clip(raw_predictions, 0, 1) * num_range
    predictions = predictions.round().astype(int)
    
    # If we have duplicates, resolve them
    used_numbers = set()
    final_predictions = []
    
    for pred in predictions:
        while pred in used_numbers or pred < 1:
            if pred < 1:
                pred = 1
            elif pred < num_range:
                pred += 1
            else:
                pred = 1
                while pred in used_numbers:
                    pred += 1
        used_numbers.add(pred)
        final_predictions.append(pred)
    
    return sorted(final_predictions)

def main():
    # Load and prepare data
    print("Loading and preparing data...")
    df = load_data('nouveau_loto.csv')
    features, targets = prepare_data(df)
    
    # Create datasets
    dataset = LotteryDataset(features, targets)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Create and train model
    print("Training model...")
    model = LotteryPredictor(input_size=features.shape[1])
    model = train_model(train_loader, val_loader, model)
    
    # Prepare next draw features
    next_draw = pd.DataFrame({
        'date_de_tirage': [pd.Timestamp.now()],
    })
    next_draw['year'] = next_draw['date_de_tirage'].dt.year
    next_draw['month'] = next_draw['date_de_tirage'].dt.month
    next_draw['day'] = next_draw['date_de_tirage'].dt.day
    next_draw['day_of_week'] = next_draw['date_de_tirage'].dt.dayofweek
    
    # Normalize features
    next_draw['year'] = (next_draw['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
    next_draw['month'] = next_draw['month'] / 12
    next_draw['day'] = next_draw['day'] / 31
    next_draw['day_of_week'] = next_draw['day_of_week'] / 6
    
    # Add cyclical features
    next_draw['month_sin'] = np.sin(2 * np.pi * next_draw['month'])
    next_draw['month_cos'] = np.cos(2 * np.pi * next_draw['month'])
    next_draw['day_sin'] = np.sin(2 * np.pi * next_draw['day'])
    next_draw['day_cos'] = np.cos(2 * np.pi * next_draw['day'])
    next_draw['day_of_week_sin'] = np.sin(2 * np.pi * next_draw['day_of_week'])
    next_draw['day_of_week_cos'] = np.cos(2 * np.pi * next_draw['day_of_week'])
    
    features = ['year', 
                'month_sin', 'month_cos',
                'day_sin', 'day_cos',
                'day_of_week_sin', 'day_of_week_cos']
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        next_features = torch.FloatTensor(next_draw[features].values)
        main_pred, lucky_pred = model(next_features)
        
        # Convert predictions to actual numbers
        main_numbers = predict_unique_numbers(main_pred.numpy()[0], 49, 5)
        lucky_number = int(round(lucky_pred.numpy()[0][0] * 10))
        lucky_number = max(1, min(10, lucky_number))
        
        print("\nPredicted numbers for next draw:")
        print(f"Main balls: {main_numbers}")
        print(f"Lucky number: {lucky_number}")
    
    # Plot historical distribution
    plt.figure(figsize=(15, 5))
    for i, column in enumerate(['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']):
        plt.hist(df[column], bins=49, alpha=0.3, label=f'Ball {i+1}')
    plt.title('Historical Distribution of Lottery Numbers')
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('number_distribution.png')
    plt.close()

if __name__ == "__main__":
    main() 