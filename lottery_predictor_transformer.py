import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import argparse
from collections import Counter
import random
import os

class LotteryDataset(Dataset):
    def __init__(self, features, targets, tokenizer, max_length=128):
        self.features = features
        self.targets = torch.FloatTensor(targets)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Convert numerical features to text format
        feature_text = self.features[idx]
        
        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            feature_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': self.targets[idx]
        }

class LotteryTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Separate prediction heads for main numbers and lucky number
        self.main_numbers_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # Predict 5 main numbers
        )
        
        self.lucky_number_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Predict 1 lucky number
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(0)  # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(0)  # Remove sequence dimension
        
        main_numbers = self.main_numbers_head(x)
        lucky_number = self.lucky_number_head(x)
        
        return main_numbers, lucky_number

def load_and_prepare_data(csv_dir='csv'):
    """Load and combine all CSV files"""
    all_data = []
    for file in os.listdir(csv_dir):
        if file.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(csv_dir, file), sep=';')
                if 'date_de_tirage' in df.columns:
                    try:
                        df['date_de_tirage'] = pd.to_datetime(df['date_de_tirage'], format='%d/%m/%Y')
                    except:
                        try:
                            df['date_de_tirage'] = pd.to_datetime(df['date_de_tirage'], format='%Y%m%d')
                        except:
                            print(f"Warning: Could not parse dates in {file}")
                            continue
                
                if 'numero_chance' in df.columns:
                    df = df.dropna(subset=['numero_chance'])
                    df['numero_chance'] = df['numero_chance'].astype(int)
                
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue
    
    if not all_data:
        raise ValueError("No valid CSV files found")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    if 'date_de_tirage' in combined_df.columns:
        combined_df = combined_df.sort_values('date_de_tirage')
    
    return combined_df

def prepare_features(df):
    """Prepare features for the transformer model"""
    # Extract date features
    features = pd.DataFrame()
    features['year'] = df['date_de_tirage'].dt.year
    features['month'] = df['date_de_tirage'].dt.month
    features['day'] = df['date_de_tirage'].dt.day
    features['day_of_week'] = df['date_de_tirage'].dt.dayofweek
    
    # Add historical draw features (last 5 draws)
    for i in range(1, 6):
        for j in range(1, 6):
            col_name = f'prev_{i}_ball_{j}'
            features[col_name] = df[f'boule_{j}'].shift(i)
        features[f'prev_{i}_lucky'] = df['numero_chance'].shift(i)
    
    # Drop rows with NaN values (first 5 rows due to shift)
    features = features.dropna()
    
    # Scale features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, scaler

def evaluate_prediction(prediction, actual_numbers):
    """Evaluate a single prediction against actual numbers"""
    prediction_set = set(prediction)
    actual_set = set(actual_numbers)
    
    matches = len(prediction_set.intersection(actual_set))
    return {
        'matches': matches,
        'sum_diff': abs(sum(prediction) - sum(actual_numbers)),
        'numbers_within_5': sum(1 for p in prediction if any(abs(p - a) <= 5 for a in actual_numbers))
    }

def evaluate_model(model, test_features, test_data):
    """Evaluate transformer model predictions"""
    model.eval()
    results = []
    lucky_correct = 0
    total_predictions = len(test_features)
    
    with torch.no_grad():
        for i in range(len(test_features)):
            # Get model predictions
            features = torch.FloatTensor(test_features[i:i+1])
            main_numbers, lucky_number = model(features)
            
            # Process main numbers predictions
            main_pred = main_numbers.squeeze().numpy()
            # Convert to valid lottery numbers (1-49)
            main_pred = np.clip(np.round(main_pred * 48 + 1), 1, 49).astype(int)
            # Ensure unique numbers
            main_pred = sorted(list(set(main_pred)))[:5]
            while len(main_pred) < 5:
                new_num = random.randint(1, 49)
                if new_num not in main_pred:
                    main_pred.append(new_num)
            main_pred = sorted(main_pred)
            
            # Process lucky number prediction
            lucky_pred = int(round(lucky_number.item() * 9 + 1))
            lucky_pred = max(1, min(10, lucky_pred))
            
            # Get actual numbers for this draw
            actual = sorted([
                test_data.iloc[i]['boule_1'],
                test_data.iloc[i]['boule_2'],
                test_data.iloc[i]['boule_3'],
                test_data.iloc[i]['boule_4'],
                test_data.iloc[i]['boule_5']
            ])
            
            # Evaluate prediction
            eval_result = evaluate_prediction(main_pred, actual)
            results.append(eval_result)
            
            # Check lucky number
            if lucky_pred == test_data.iloc[i]['numero_chance']:
                lucky_correct += 1
    
    # Calculate metrics
    avg_matches = np.mean([r['matches'] for r in results])
    avg_sum_diff = np.mean([r['sum_diff'] for r in results])
    avg_within_5 = np.mean([r['numbers_within_5'] for r in results])
    match_distribution = Counter([r['matches'] for r in results])
    lucky_accuracy = lucky_correct / total_predictions
    
    return {
        'avg_matches': avg_matches,
        'avg_sum_diff': avg_sum_diff,
        'avg_within_5': avg_within_5,
        'match_distribution': match_distribution,
        'lucky_accuracy': lucky_accuracy
    }

def train_model(train_features, train_data, epochs=10, batch_size=32):
    """Train the transformer model"""
    input_dim = train_features.shape[1]
    model = LotteryTransformer(input_dim)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    # Convert targets to normalized form (0-1)
    train_main_targets = np.column_stack([
        train_data['boule_1'],
        train_data['boule_2'],
        train_data['boule_3'],
        train_data['boule_4'],
        train_data['boule_5']
    ])
    train_main_targets = (train_main_targets - 1) / 48  # Normalize to 0-1
    
    train_lucky_targets = (train_data['numero_chance'].values - 1) / 9  # Normalize to 0-1
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = len(train_features) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_features = torch.FloatTensor(train_features[start_idx:end_idx])
            batch_main_targets = torch.FloatTensor(train_main_targets[start_idx:end_idx])
            batch_lucky_targets = torch.FloatTensor(train_lucky_targets[start_idx:end_idx]).unsqueeze(1)
            
            optimizer.zero_grad()
            main_pred, lucky_pred = model(batch_features)
            
            main_loss = criterion(main_pred, batch_main_targets)
            lucky_loss = criterion(lucky_pred, batch_lucky_targets)
            loss = main_loss + lucky_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Transformer-based lottery prediction')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation on test data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    args = parser.parse_args()
    
    # Load and prepare data
    print("Loading data...")
    df = load_and_prepare_data()
    
    # Prepare features
    features, scaler = prepare_features(df)
    
    if args.evaluate:
        print("\nRunning evaluation on test data...")
        
        # Split data into training and testing sets (80-20 split)
        train_size = int(0.8 * len(features))
        train_features = features[:train_size]
        test_features = features[train_size:]
        train_data = df.iloc[5:train_size+5]  # Offset by 5 due to historical features
        test_data = df.iloc[train_size+5:]
        
        print(f"\nTraining data: {len(train_features)} draws")
        print(f"Testing data: {len(test_features)} draws")
        
        # Train model
        print("\nTraining transformer model...")
        model = train_model(train_features, train_data, epochs=args.epochs)
        
        # Evaluate model
        print("\nEvaluating model performance...")
        results = evaluate_model(model, test_features, test_data)
        
        print("\nTransformer Model Results:")
        print(f"Average matches per draw: {results['avg_matches']:.2f}")
        print(f"Average sum difference: {results['avg_sum_diff']:.2f}")
        print(f"Average numbers within ±5: {results['avg_within_5']:.2f}")
        print(f"\nLucky Number Accuracy: {results['lucky_accuracy']:.2%}")
        
        print("\nMatch distribution:")
        for matches, count in sorted(results['match_distribution'].items()):
            percentage = (count / sum(results['match_distribution'].values())) * 100
            print(f"{matches} matches: {percentage:.1f}%")
    
    else:
        # Train on all data for prediction
        train_data = df.iloc[5:]  # Offset by 5 due to historical features
        model = train_model(features, train_data, epochs=args.epochs)
        
        # Make prediction for next draw
        with torch.no_grad():
            latest_features = torch.FloatTensor(features[-1:])
            main_numbers, lucky_number = model(latest_features)
            
            # Process main numbers predictions
            main_pred = main_numbers.squeeze().numpy()
            main_pred = np.clip(np.round(main_pred * 48 + 1), 1, 49).astype(int)
            main_pred = sorted(list(set(main_pred)))[:5]
            while len(main_pred) < 5:
                new_num = random.randint(1, 49)
                if new_num not in main_pred:
                    main_pred.append(new_num)
            main_pred = sorted(main_pred)
            
            # Process lucky number prediction
            lucky_pred = int(round(lucky_number.item() * 9 + 1))
            lucky_pred = max(1, min(10, lucky_pred))
            
            print("\nTransformer Model Predictions:")
            print(f"Main numbers: {main_pred}")
            print(f"Sum: {sum(main_pred)}")
            print(f"Lucky number: {lucky_pred}")

if __name__ == "__main__":
    main() 