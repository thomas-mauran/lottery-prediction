import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import argparse
from collections import Counter
import random
import os
import math
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import sys

class EuroMillionsTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=2, num_layers=1, dropout=0.1):
        super().__init__()
        
        # Very simple architecture
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Single transformer encoder layer with reduced complexity
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model,  # Reduced feedforward dimension
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Direct prediction heads
        self.main_numbers_head = nn.Linear(d_model, 50)
        self.star_numbers_head = nn.Linear(d_model, 12)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        return torch.sigmoid(self.main_numbers_head(x)), torch.sigmoid(self.star_numbers_head(x))

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        return x + self.layers(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def load_and_prepare_data(csv_dir='csv'):
    """Load and combine all CSV files"""
    all_data = []
    for file in os.listdir(csv_dir):
        if file.endswith('.csv') and 'euro' in file.lower():  # Only load EuroMillions files
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
                
                # Ensure all required columns exist
                required_columns = ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 
                                 'etoile_1', 'etoile_2']
                if not all(col in df.columns for col in required_columns):
                    print(f"Warning: Missing required columns in {file}")
                    continue
                
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue
    
    if not all_data:
        raise ValueError("No valid EuroMillions CSV files found")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    if 'date_de_tirage' in combined_df.columns:
        combined_df = combined_df.sort_values('date_de_tirage')
    
    return combined_df

def prepare_features(df):
    """Prepare minimal features for the transformer model"""
    feature_dict = {}
    
    # Historical draw features with minimal context (last 3 draws only)
    for i in range(1, 4):  # Reduced from 10 to 3 previous draws
        # Main numbers
        for j in range(1, 6):
            feature_dict[f'prev_{i}_ball_{j}'] = df[f'boule_{j}'].shift(i)
        
        # Star numbers (both stars)
        feature_dict[f'prev_{i}_star_1'] = df['etoile_1'].shift(i)
        feature_dict[f'prev_{i}_star_2'] = df['etoile_2'].shift(i)
    
    # Add only the most relevant rolling statistics with a small window
    window_size = 5  # Only use one small window size
    
    # Rolling statistics for main numbers
    for j in range(1, 6):
        roll_mean = df[f'boule_{j}'].rolling(window_size).mean()
        feature_dict[f'roll_{window_size}_mean_ball_{j}'] = roll_mean
    
    # Rolling statistics for star numbers
    for j in range(1, 3):
        roll_mean = df[f'etoile_{j}'].rolling(window_size).mean()
        feature_dict[f'roll_{window_size}_mean_star_{j}'] = roll_mean
    
    # Create DataFrame all at once to avoid fragmentation
    features = pd.DataFrame(feature_dict)
    
    # Drop rows with NaN values
    features = features.dropna()
    
    # Scale features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, scaler

def evaluate_prediction(prediction, actual_numbers, star_prediction=None, actual_stars=None):
    """Evaluate a single prediction against actual numbers"""
    prediction_set = set(prediction)
    actual_set = set(actual_numbers)
    
    matches = len(prediction_set.intersection(actual_set))
    
    if star_prediction is not None and actual_stars is not None:
        # Compare both star numbers
        star_matches = len(set(star_prediction).intersection(set(actual_stars)))
    else:
        star_matches = 0
    
    return {
        'matches': matches,
        'star_matches': star_matches,
        'sum_diff': abs(sum(prediction) - sum(actual_numbers)),
        'numbers_within_5': sum(1 for p in prediction if any(abs(p - a) <= 5 for a in actual_numbers))
    }

def process_predictions(raw_predictions, num_count, min_val, max_val, min_gap=2):
    """Process raw predictions with spacing constraints"""
    # Handle single number prediction (for star numbers)
    if num_count == 1:
        pred = raw_predictions.squeeze().numpy()
        if isinstance(pred, np.ndarray):
            pred = pred.item()
        return [int(np.clip(np.round(pred * (max_val - min_val) + min_val), min_val, max_val))]
    
    # Convert predictions to valid range
    pred = raw_predictions.squeeze().numpy()
    pred = np.clip(np.round(pred * (max_val - min_val) + min_val), min_val, max_val).astype(int)
    
    # Sort by confidence (use absolute values as predictions are normalized)
    confidence = np.abs(raw_predictions.squeeze().numpy())
    indices = np.argsort(confidence)[::-1]
    sorted_pred = pred[indices]
    
    # Generate final numbers with spacing constraints
    final_numbers = []
    all_candidates = list(range(min_val, max_val + 1))
    
    # Start with the highest confidence prediction
    if len(sorted_pred) > 0:
        final_numbers.append(sorted_pred[0])
    
    while len(final_numbers) < num_count:
        valid_candidates = [n for n in all_candidates 
                          if n not in final_numbers and
                          all(abs(n - x) >= min_gap for x in final_numbers)]
        
        if not valid_candidates:
            # If no candidates meet spacing criteria, relax the constraint
            valid_candidates = [n for n in all_candidates if n not in final_numbers]
        
        # Prefer numbers from model predictions if available
        remaining_preds = [p for p in sorted_pred if p in valid_candidates]
        if remaining_preds:
            next_num = remaining_preds[0]
        else:
            # If no valid predictions, choose randomly with preference for numbers
            # that maintain good distribution
            weights = [1.0 / (1 + min(abs(n - x) for x in final_numbers)) 
                      for n in valid_candidates]
            next_num = random.choices(valid_candidates, weights=weights)[0]
        
        final_numbers.append(next_num)
    
    return sorted(final_numbers)

# Add Dataset class for better data loading
class LotteryDataset(Dataset):
    def __init__(self, features, main_numbers, star_numbers):
        """Initialize dataset with tensor conversions"""
        self.features = torch.FloatTensor(features)
        self.main_numbers = torch.FloatTensor(main_numbers)
        self.star_numbers = torch.FloatTensor(star_numbers)
        assert len(self.features) == len(self.main_numbers) == len(self.star_numbers), \
            "All inputs must have the same length"
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")
        return (self.features[idx], self.main_numbers[idx], self.star_numbers[idx])

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def train_model(features, data, epochs=50, model_path='models/euromillions_model.pt'):
    """Train the model or load from file if it exists"""
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Define model
    input_dim = features.shape[1]
    model = EuroMillionsTransformer(input_dim)
    
    # If model exists and we're not in evaluation mode, just load it
    if os.path.exists(model_path) and not sys.argv[1:] == ['--evaluate']:
        print(f"\nLoading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path))
        return model
    
    print("\nTraining new model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Prepare data
    train_main_numbers = np.column_stack([
        data[f'boule_{i}'].values - 1 for i in range(1, 6)
    ])
    train_star_numbers = np.column_stack([
        data[f'etoile_{i}'].values - 1 for i in range(1, 3)
    ])
    
    def to_one_hot(numbers, max_val):
        one_hot = torch.zeros(len(numbers), max_val)
        for i, nums in enumerate(numbers):
            one_hot[i, nums.astype(int)] = 1
        return one_hot
    
    train_main_one_hot = to_one_hot(train_main_numbers, 50)
    train_star_one_hot = to_one_hot(train_star_numbers, 12)
    
    # Create dataset and dataloader
    dataset = LotteryDataset(features, train_main_one_hot, train_star_one_hot)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,  # Reduced workers
        pin_memory=True
    )
    
    # Simple optimizer with high learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    best_loss = float('inf')
    best_state = None
    patience = 5  # Very short patience
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_features, batch_main_targets, batch_star_targets in dataloader:
            batch_features = batch_features.to(device)
            batch_main_targets = batch_main_targets.to(device)
            batch_star_targets = batch_star_targets.to(device)
            
            optimizer.zero_grad()
            main_pred, star_pred = model(batch_features)
            
            # Simple BCE loss
            loss = F.binary_cross_entropy(main_pred, batch_main_targets) + \
                   F.binary_cross_entropy(star_pred, batch_star_targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if epoch % 5 == 0:  # Print less frequently
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
    
    # Save best model
    model = model.cpu()
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), model_path)
    
    return model

def evaluate_model(model, test_features, test_data):
    """Evaluate transformer model predictions"""
    model.eval()
    results = []
    
    # Ensure all inputs have the same length
    n_samples = len(test_features)
    assert n_samples == len(test_data), "Test features and data must have the same length"
    
    # Create dataset and dataloader for test data
    test_main_numbers = np.column_stack([
        test_data[f'boule_{i}'].values - 1 for i in range(1, 6)
    ])
    test_star_numbers = np.column_stack([
        test_data[f'etoile_{i}'].values - 1 for i in range(1, 3)
    ])
    
    def to_one_hot(numbers, max_val):
        one_hot = torch.zeros(len(numbers), max_val)
        for i, nums in enumerate(numbers):
            one_hot[i, nums.astype(int)] = 1
        return one_hot
    
    test_main_one_hot = to_one_hot(test_main_numbers, 50)
    test_star_one_hot = to_one_hot(test_star_numbers, 12)
    
    # Create dataset and dataloader
    dataset = LotteryDataset(test_features, test_main_one_hot, test_star_one_hot)
    dataloader = DataLoader(
        dataset,
        batch_size=min(8, n_samples),  # Ensure batch size doesn't exceed dataset size
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch_features, batch_main_targets, batch_star_targets in dataloader:
            # Move batch to device
            batch_features = batch_features.to(device)
            
            # Get model predictions
            main_pred, star_pred = model(batch_features)
            
            # Move predictions back to CPU for processing
            main_pred = main_pred.cpu()
            star_pred = star_pred.cpu()
            
            # Process each prediction in the batch
            for i in range(len(batch_features)):
                # Process predictions
                main_numbers = main_pred[i].numpy()
                star_numbers = star_pred[i].numpy()
                
                # Get top 5 main numbers and top 2 star numbers
                main_indices = np.argsort(main_numbers)[-5:]
                star_indices = np.argsort(star_numbers)[-2:]
                
                prediction = np.concatenate([
                    main_indices + 1,  # Add 1 to convert from 0-based to 1-based
                    star_indices + 1
                ])
                
                # Get actual numbers for this draw
                actual_main = sorted([
                    test_data.iloc[i]['boule_1'],
                    test_data.iloc[i]['boule_2'],
                    test_data.iloc[i]['boule_3'],
                    test_data.iloc[i]['boule_4'],
                    test_data.iloc[i]['boule_5']
                ])
                
                actual_stars = sorted([
                    test_data.iloc[i]['etoile_1'],
                    test_data.iloc[i]['etoile_2']
                ])
                
                # Calculate matches
                main_matches = len(set(prediction[:5]) & set(actual_main))
                star_matches = len(set(prediction[5:]) & set(actual_stars))
                
                results.append({
                    'main_matches': main_matches,
                    'star_matches': star_matches
                })
    
    # Convert results to arrays
    main_matches = np.array([r['main_matches'] for r in results])
    star_matches = np.array([r['star_matches'] for r in results])
    
    return {
        'main_matches': main_matches,
        'star_matches': star_matches
    }

def main():
    parser = argparse.ArgumentParser(description='EuroMillions prediction with transformer model')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation on test data')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--model-path', type=str, default='models/euromillions_model.pt', help='Path to save/load model')
    args = parser.parse_args()
    
    try:
        # Load and prepare data
        print("Loading EuroMillions data...")
        df = load_and_prepare_data()
        
        # Calculate statistics from historical data
        historical_sums = []
        historical_gaps = []
        for _, row in df.iterrows():
            numbers = sorted([row['boule_1'], row['boule_2'], row['boule_3'], row['boule_4'], row['boule_5']])
            historical_sums.append(sum(numbers))
            gaps = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
            historical_gaps.extend(gaps)
        
        avg_sum = np.mean(historical_sums)
        std_sum = np.std(historical_sums)
        avg_gap = np.mean(historical_gaps)
        print(f"\nHistorical Statistics:")
        print(f"Average sum: {avg_sum:.1f} ± {std_sum:.1f}")
        print(f"Average gap between numbers: {avg_gap:.1f}")
        
        if args.evaluate:
            print("\nRunning evaluation on test data...")
            
            # Prepare features first
            features, scaler = prepare_features(df)
            
            # Split data into training and testing sets (90-10 split)
            train_size = int(0.9 * len(features))
            train_features = features[:train_size]
            test_features = features[train_size:]
            
            # Split the original dataframe at the same point
            train_data = df.iloc[len(df)-len(features):].iloc[:train_size]
            test_data = df.iloc[len(df)-len(features):].iloc[train_size:]
            
            print(f"\nTraining data: {len(train_features)} draws")
            print(f"Testing data: {len(test_features)} draws")
            
            # Train model
            model = train_model(train_features, train_data, epochs=args.epochs, model_path=args.model_path)
            
            # Evaluate model
            print("\nEvaluating model performance...")
            results = evaluate_model(model, test_features, test_data)
            
            # Print results
            total_predictions = len(results['main_matches'])
            print(f"\nEuroMillions Transformer Model Results:")
            print(f"Total draws evaluated: {total_predictions}")
            print(f"Average main number matches per draw: {np.mean(results['main_matches']):.2f}")
            print(f"Average star number matches per draw: {np.mean(results['star_matches']):.2f}")
            
            print(f"\nMain number match distribution (out of {total_predictions} draws):")
            for matches in range(6):  # 0 to 5 matches
                count = np.sum(results['main_matches'] == matches)
                percentage = (count / total_predictions) * 100
                print(f"{matches} matches: {count}/{total_predictions} ({percentage:.1f}%)")
            
            print(f"\nStar number match distribution (out of {total_predictions} draws):")
            for matches in range(3):  # 0 to 2 matches for star numbers
                count = np.sum(results['star_matches'] == matches)
                percentage = (count / total_predictions) * 100
                print(f"{matches} matches: {count}/{total_predictions} ({percentage:.1f}%)")
            
            # Add summary statistics
            print("\nSummary:")
            print(f"Draws with at least 1 main number match: {np.sum(results['main_matches'] > 0)}/{total_predictions}")
            print(f"Draws with at least 1 star match: {np.sum(results['star_matches'] > 0)}/{total_predictions}")
            max_main_matches = np.max(results['main_matches'])
            max_star_matches = np.max(results['star_matches'])
            print(f"Best performance: {max_main_matches} main numbers and {max_star_matches} star numbers matched")
        
        else:
            # Train or load model for prediction
            features, scaler = prepare_features(df)
            train_data = df.iloc[len(df)-len(features):]
            model = train_model(features, train_data, epochs=args.epochs, model_path=args.model_path)
            
            # Make prediction for next draw
            print("\nGenerating predictions...")
            with torch.no_grad():
                latest_features = torch.FloatTensor(features[-1:])
                main_pred, star_pred = model(latest_features)
                
                # Process predictions
                main_numbers = main_pred[0].numpy()
                star_numbers = star_pred[0].numpy()
                
                # Get top 5 main numbers and top 2 star numbers
                main_indices = np.argsort(main_numbers)[-5:] + 1  # Add 1 for 1-based indexing
                star_indices = np.argsort(star_numbers)[-2:] + 1
                
                # Sort the predictions
                main_indices = sorted(main_indices)
                star_indices = sorted(star_indices)
                
                # Calculate gaps for display
                gaps = [main_indices[i+1] - main_indices[i] for i in range(len(main_indices)-1)]
                
                print("\nEuroMillions Transformer Model Predictions:")
                print(f"Main numbers: {main_indices}")
                print(f"Sum: {sum(main_indices)} (historical avg: {avg_sum:.1f})")
                print(f"Gaps between numbers: {gaps}")
                print(f"Average gap: {np.mean(gaps):.1f} (historical avg: {avg_gap:.1f})")
                print(f"Star numbers: {star_indices}")
    
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please ensure you have the correct data files and all required packages installed.")
        raise

if __name__ == "__main__":
    main() 