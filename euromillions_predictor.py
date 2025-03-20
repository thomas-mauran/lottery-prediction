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

class EuroMillionsTransformer(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=6, dropout=0.2):
        super().__init__()
        
        # Split input dimensions
        self.temporal_dim = 10  # First 10 features are temporal
        self.other_dim = input_dim - self.temporal_dim
        
        # Enhanced embedding with multiple layers
        self.embedding = nn.Sequential(
            nn.Linear(self.other_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Separate embedding for temporal features
        self.temporal_embedding = nn.Sequential(
            nn.Linear(self.temporal_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Main numbers prediction head
        self.main_numbers_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 50),  # Output probabilities for numbers 1-50
            nn.Sigmoid()
        )
        
        # Star numbers prediction head
        self.star_numbers_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 12),  # Output probabilities for numbers 1-12
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Split temporal and other features
        temporal_features = x[:, :self.temporal_dim]
        other_features = x[:, self.temporal_dim:]
        
        # Process features
        temporal_embedded = self.temporal_embedding(temporal_features)
        other_embedded = self.embedding(other_features)
        
        # Combine embeddings
        combined = temporal_embedded + other_embedded
        
        # Apply transformer
        x = combined.unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(1)  # Remove sequence dimension
        
        # Generate predictions
        main_numbers = self.main_numbers_head(x)
        star_numbers = self.star_numbers_head(x)
        
        return main_numbers, star_numbers

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
    """Prepare enhanced features for the transformer model"""
    feature_dict = {}
    
    # Enhanced date features
    feature_dict.update({
        'year': df['date_de_tirage'].dt.year,
        'month': df['date_de_tirage'].dt.month,
        'day': df['date_de_tirage'].dt.day,
        'day_of_week': df['date_de_tirage'].dt.dayofweek,
        'week_of_year': df['date_de_tirage'].dt.isocalendar().week,
        'day_of_year': df['date_de_tirage'].dt.dayofyear,
    })
    
    # Cyclical encoding for temporal features
    feature_dict.update({
        'month_sin': np.sin(2 * np.pi * feature_dict['month']/12),
        'month_cos': np.cos(2 * np.pi * feature_dict['month']/12),
        'day_sin': np.sin(2 * np.pi * feature_dict['day']/31),
        'day_cos': np.cos(2 * np.pi * feature_dict['day']/31)
    })
    
    # Historical draw features with more context (last 10 draws)
    for i in range(1, 11):
        # Main numbers
        for j in range(1, 6):
            feature_dict[f'prev_{i}_ball_{j}'] = df[f'boule_{j}'].shift(i)
        
        # Add sum and statistical features for previous draws
        prev_numbers = df[[f'boule_{j}' for j in range(1, 6)]].shift(i)
        feature_dict[f'prev_{i}_sum'] = prev_numbers.sum(axis=1)
        feature_dict[f'prev_{i}_mean'] = prev_numbers.mean(axis=1)
        feature_dict[f'prev_{i}_std'] = prev_numbers.std(axis=1)
        
        # Star numbers (both stars)
        feature_dict[f'prev_{i}_star_1'] = df['etoile_1'].shift(i)
        feature_dict[f'prev_{i}_star_2'] = df['etoile_2'].shift(i)
    
    # Add rolling statistics
    window_sizes = [5, 10, 20]
    for window in window_sizes:
        # Rolling statistics for main numbers
        for j in range(1, 6):
            roll_mean = df[f'boule_{j}'].rolling(window).mean()
            roll_std = df[f'boule_{j}'].rolling(window).std()
            feature_dict[f'roll_{window}_mean_ball_{j}'] = roll_mean
            feature_dict[f'roll_{window}_std_ball_{j}'] = roll_std
        
        # Rolling statistics for star numbers (both stars)
        for j in range(1, 3):
            roll_mean = df[f'etoile_{j}'].rolling(window).mean()
            roll_std = df[f'etoile_{j}'].rolling(window).std()
            feature_dict[f'roll_{window}_mean_star_{j}'] = roll_mean
            feature_dict[f'roll_{window}_std_star_{j}'] = roll_std
    
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

def evaluate_model(model, test_features, test_data):
    """Evaluate transformer model predictions"""
    model.eval()
    results = []
    total_predictions = len(test_features)
    
    with torch.no_grad():
        for i in range(len(test_features)):
            # Get model predictions
            features = torch.FloatTensor(test_features[i:i+1])
            main_numbers, star_numbers = model(features)
            
            # Process main numbers predictions (1-50)
            main_pred = process_predictions(main_numbers, 5, 1, 50, min_gap=2)
            
            # Process star numbers predictions (1-12)
            star_pred = process_predictions(star_numbers, 2, 1, 12, min_gap=1)
            
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
            
            # Evaluate prediction
            eval_result = evaluate_prediction(main_pred, actual_main, star_pred, actual_stars)
            results.append(eval_result)
    
    # Calculate metrics
    avg_matches = np.mean([r['matches'] for r in results])
    avg_star_matches = np.mean([r['star_matches'] for r in results])
    avg_sum_diff = np.mean([r['sum_diff'] for r in results])
    avg_within_5 = np.mean([r['numbers_within_5'] for r in results])
    match_distribution = Counter([r['matches'] for r in results])
    star_match_distribution = Counter([r['star_matches'] for r in results])
    
    return {
        'avg_matches': avg_matches,
        'avg_star_matches': avg_star_matches,
        'avg_sum_diff': avg_sum_diff,
        'avg_within_5': avg_within_5,
        'match_distribution': match_distribution,
        'star_match_distribution': star_match_distribution
    }

def train_model(train_features, train_data, epochs=2000, batch_size=128):
    """Train the transformer model with improved training process"""
    input_dim = train_features.shape[1]
    model = EuroMillionsTransformer(input_dim)
    
    # Reduce initial learning rate and adjust weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
    
    # Custom loss function for number selection with adjusted weights
    def number_selection_loss(pred, target, num_to_select):
        # Sort predictions and get top k indices
        topk_values, topk_indices = torch.topk(pred, k=num_to_select)
        
        # Create a binary mask for selected numbers
        selected = torch.zeros_like(pred)
        selected.scatter_(1, topk_indices, 1)
        
        # Calculate BCE loss for selected vs. target numbers
        bce_loss = nn.BCELoss()(selected.float(), target.float())
        
        # Add regularization to encourage diverse predictions with reduced weight
        diversity_loss = -torch.std(pred, dim=1).mean()
        
        return bce_loss + 0.05 * diversity_loss  # Reduced diversity loss weight
    
    # Convert targets to one-hot encoding
    def to_one_hot(numbers, max_val):
        one_hot = torch.zeros(len(numbers), max_val)
        for i, nums in enumerate(numbers):
            one_hot[i, (nums - 1).astype(int)] = 1
        return one_hot
    
    # Prepare targets
    train_main_numbers = np.column_stack([
        train_data['boule_1'].values,
        train_data['boule_2'].values,
        train_data['boule_3'].values,
        train_data['boule_4'].values,
        train_data['boule_5'].values
    ])
    
    train_star_numbers = np.column_stack([
        train_data['etoile_1'].values,
        train_data['etoile_2'].values
    ])
    
    train_main_one_hot = to_one_hot(train_main_numbers, 50)
    train_star_one_hot = to_one_hot(train_star_numbers, 12)
    
    # Learning rate scheduler with longer warmup and gentler cosine annealing
    total_steps = epochs * (len(train_features) // batch_size)
    warmup_steps = total_steps // 5  # 20% warmup
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress)) + 0.1  # Add minimum learning rate
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_loss = float('inf')
    patience = 200  # Increased patience significantly
    patience_counter = 0
    min_delta = 0.0001  # Slightly increased improvement threshold
    
    # Keep track of losses for smoothing
    loss_history = []
    smoothing_window = 50  # Increased smoothing window
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = len(train_features) // batch_size
        
        # Shuffle data for each epoch
        indices = np.random.permutation(len(train_features))
        train_features = train_features[indices]
        train_main_one_hot = train_main_one_hot[indices]
        train_star_one_hot = train_star_one_hot[indices]
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_features = torch.FloatTensor(train_features[start_idx:end_idx])
            batch_main_targets = train_main_one_hot[start_idx:end_idx]
            batch_star_targets = train_star_one_hot[start_idx:end_idx]
            
            optimizer.zero_grad()
            main_pred, star_pred = model(batch_features)
            
            # Calculate losses with number selection constraint
            main_loss = number_selection_loss(main_pred, batch_main_targets, 5)
            star_loss = number_selection_loss(star_pred, batch_star_targets, 2)
            
            # Adjusted loss weights
            loss = 0.8 * main_loss + 0.2 * star_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Increased gradient clipping
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        loss_history.append(avg_loss)
        
        # Calculate smoothed loss with longer window
        if len(loss_history) >= smoothing_window:
            smoothed_loss = np.mean(loss_history[-smoothing_window:])
        else:
            smoothed_loss = avg_loss
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Smoothed Loss: {smoothed_loss:.4f}")
        
        # Early stopping with increased patience and threshold
        if smoothed_loss < best_loss - min_delta:
            best_loss = smoothed_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best loss: {best_loss:.4f}")
                break
    
    # Restore best model
    model.load_state_dict(best_state)
    return model

def main():
    parser = argparse.ArgumentParser(description='EuroMillions prediction with transformer model')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation on test data')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs (recommended: 500+)')
    args = parser.parse_args()
    
    # Add warning for low epoch count
    if args.epochs < 200:
        print("\nWarning: Using less than 200 epochs may result in suboptimal model performance.")
        print("Recommended: Use 500+ epochs for better results.")
        user_input = input("Do you want to continue with the current number of epochs? (y/n): ")
        if user_input.lower() != 'y':
            print("Exiting. Please run again with more epochs using --epochs argument.")
            return
    
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
        
        # Prepare features
        features, scaler = prepare_features(df)
        
        if args.evaluate:
            print("\nRunning evaluation on test data...")
            
            # Split data into training and testing sets (90-10 split)
            train_size = int(0.9 * len(features))
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
            
            print("\nEuroMillions Transformer Model Results:")
            total_predictions = sum(results['match_distribution'].values())
            print(f"Total draws evaluated: {total_predictions}")
            print(f"Average main number matches per draw: {results['avg_matches']:.2f}")
            print(f"Average star number matches per draw: {results['avg_star_matches']:.2f}")
            print(f"Average sum difference: {results['avg_sum_diff']:.2f}")
            print(f"Average numbers within ±5: {results['avg_within_5']:.2f}")
            
            print(f"\nMain number match distribution (out of {total_predictions} draws):")
            for matches in range(6):  # 0 to 5 matches
                count = results['match_distribution'].get(matches, 0)
                percentage = (count / total_predictions) * 100
                print(f"{matches} matches: {count}/{total_predictions} ({percentage:.1f}%)")
            
            print(f"\nStar number match distribution (out of {total_predictions} draws):")
            for matches in range(3):  # 0 to 2 matches for star numbers
                count = results['star_match_distribution'].get(matches, 0)
                percentage = (count / total_predictions) * 100
                print(f"{matches} matches: {count}/{total_predictions} ({percentage:.1f}%)")
            
            # Add summary statistics
            print("\nSummary:")
            print(f"Draws with at least 1 main number match: {sum(count for matches, count in results['match_distribution'].items() if matches > 0)}/{total_predictions}")
            print(f"Draws with at least 1 star match: {sum(count for matches, count in results['star_match_distribution'].items() if matches > 0)}/{total_predictions}")
            max_main_matches = max(results['match_distribution'].keys())
            max_star_matches = max(results['star_match_distribution'].keys())
            print(f"Best performance: {max_main_matches} main numbers and {max_star_matches} star numbers matched")
        
        else:
            # Train on all data for prediction
            train_data = df.iloc[5:]  # Offset by 5 due to historical features
            model = train_model(features, train_data, epochs=args.epochs)
            
            # Make prediction for next draw
            print("\nGenerating predictions...")
            with torch.no_grad():
                # Convert features to tensor properly
                latest_features = torch.FloatTensor(features[-1:])  # Get last row and convert to tensor
                main_numbers, star_numbers = model(latest_features)  # Use the tensor
                
                # Process predictions with spacing constraints
                main_pred = process_predictions(main_numbers, 5, 1, 50, min_gap=2)
                star_pred = process_predictions(star_numbers, 2, 1, 12, min_gap=1)
                
                # Calculate gaps for display
                gaps = [main_pred[i+1] - main_pred[i] for i in range(len(main_pred)-1)]
                
                print("\nEuroMillions Transformer Model Predictions:")
                print(f"Main numbers: {main_pred}")
                print(f"Sum: {sum(main_pred)} (historical avg: {avg_sum:.1f})")
                print(f"Gaps between numbers: {gaps}")
                print(f"Average gap: {np.mean(gaps):.1f} (historical avg: {avg_gap:.1f})")
                print(f"Star numbers: {star_pred}")  # Show both star numbers
                
                # Validate prediction
                if min(gaps) < 2:
                    print("\nWarning: Some numbers are too close together!")
                if abs(sum(main_pred) - avg_sum) > std_sum:
                    print("\nWarning: Sum is outside the typical range!")
    
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please ensure you have the correct data files and all required packages installed.")
        raise

if __name__ == "__main__":
    main() 