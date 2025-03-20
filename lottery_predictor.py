import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from collections import Counter
import random
import argparse

# Load the data
def load_data(file_path):
    df = pd.read_csv(file_path, sep=';')
    df['date_de_tirage'] = pd.to_datetime(df['date_de_tirage'], format='%d/%m/%Y')
    return df

# Prepare features and target
def prepare_data(df):
    # Extract features from the date
    df['year'] = df['date_de_tirage'].dt.year
    df['month'] = df['date_de_tirage'].dt.month
    df['day'] = df['date_de_tirage'].dt.day
    df['day_of_week'] = df['date_de_tirage'].dt.dayofweek
    
    # Create features from historical draws
    features = ['year', 'month', 'day', 'day_of_week']
    
    # Create separate targets for each ball and lucky number
    targets = ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'numero_chance']
    
    return df[features], df[targets]

# Train models for each number
def train_models(X, y):
    models = {}
    for column in y.columns:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y[column])
        models[column] = model
    return models

# Make predictions and ensure unique numbers for main balls
def predict_numbers(models, X):
    predictions = {}
    used_numbers = set()
    
    # First, get raw predictions for all balls
    raw_predictions = {}
    for column, model in models.items():
        if column != 'numero_chance':
            # Get multiple predictions to choose from
            pred = model.predict(X)[0]
            raw_predictions[column] = pred
    
    # Sort columns by confidence (using distance from nearest whole number as a proxy for confidence)
    columns = list(raw_predictions.keys())
    columns.sort(key=lambda col: abs(raw_predictions[col] - round(raw_predictions[col])))
    
    # Assign unique numbers for main balls
    for column in columns:
        base_pred = int(round(raw_predictions[column]))
        pred = base_pred
        
        # If number is already used, find the nearest unused number
        offset = 0
        while pred in used_numbers:
            offset += 1
            # Try alternating between adding and subtracting offset
            if offset % 2 == 0:
                pred = base_pred + offset
            else:
                pred = base_pred - offset
            
            # Ensure number is within valid range
            pred = max(1, min(49, pred))
            
            # If we've tried too many times, just find first available number
            if offset > 10:
                for i in range(1, 50):
                    if i not in used_numbers:
                        pred = i
                        break
        
        predictions[column] = np.array([pred])
        used_numbers.add(pred)
    
    # Add lucky number prediction
    lucky_pred = models['numero_chance'].predict(X)
    predictions['numero_chance'] = np.clip(np.round(lucky_pred), 1, 10)
    
    return predictions

def load_and_prepare_data(csv_dir='csv'):
    """Load and combine all CSV files"""
    all_data = []
    for file in os.listdir(csv_dir):
        if file.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(csv_dir, file), sep=';')
                # Convert date column if it exists
                if 'date_de_tirage' in df.columns:
                    try:
                        df['date_de_tirage'] = pd.to_datetime(df['date_de_tirage'], format='%d/%m/%Y')
                    except:
                        try:
                            df['date_de_tirage'] = pd.to_datetime(df['date_de_tirage'], format='%Y%m%d')
                        except:
                            print(f"Warning: Could not parse dates in {file}")
                            continue
                
                # Filter out rows with missing lucky numbers
                if 'numero_chance' in df.columns:
                    df = df.dropna(subset=['numero_chance'])
                    # Convert lucky numbers to integers
                    df['numero_chance'] = df['numero_chance'].astype(int)
                
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue
    
    if not all_data:
        raise ValueError("No valid CSV files found")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    # Sort by date if available
    if 'date_de_tirage' in combined_df.columns:
        combined_df = combined_df.sort_values('date_de_tirage')
    
    return combined_df

def get_temporal_weight(draw_date, target_date):
    """Calculate temporal weight based on how similar the time of year is"""
    # Convert dates to day of year (1-366)
    draw_day = draw_date.timetuple().tm_yday
    target_day = target_date.timetuple().tm_yday
    
    # Calculate difference in days, considering year wrap-around
    diff = min(abs(draw_day - target_day), 366 - abs(draw_day - target_day))
    
    # Convert to weight (closer dates get higher weights)
    # Using a Gaussian-like decay
    sigma = 30  # About a month of variance
    weight = np.exp(-0.5 * (diff / sigma) ** 2)
    return weight

def get_number_frequencies(df, target_date=None, window=100):
    """Calculate number frequencies with weighted recency and temporal patterns"""
    all_numbers = []
    weights = []
    
    # Calculate weights that decay exponentially
    total_draws = len(df)
    decay_factor = 0.995  # Slower decay for more historical influence
    
    for idx, row in df.iterrows():
        for col in ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']:
            all_numbers.append(row[col])
            
            # Base weight from recency
            recency_weight = decay_factor ** (total_draws - idx - 1)
            
            # Add temporal weight if target date is provided
            if target_date is not None:
                temporal_weight = get_temporal_weight(row['date_de_tirage'], target_date)
                final_weight = recency_weight * (0.7 + 0.3 * temporal_weight)  # Blend of recency and temporal weights
            else:
                final_weight = recency_weight
                
            weights.append(final_weight)
    
    # Calculate weighted frequencies
    frequencies = {}
    for num in range(1, 50):
        indices = [i for i, n in enumerate(all_numbers) if n == num]
        frequency = sum(weights[i] for i in indices)
        frequencies[num] = frequency
    
    return frequencies

def calculate_sum_probability(sum_value):
    """Calculate probability weight based on sum distribution (approximating observed Gaussian)"""
    # Parameters derived from the observed distribution
    mean = 124.9
    std = 20
    
    # Add penalty for extreme values
    if sum_value < 75 or sum_value > 175:  # These are rough bounds based on your distribution
        penalty = 0.5
    else:
        penalty = 1.0
    
    # Calculate probability weight using Gaussian distribution
    weight = np.exp(-0.5 * ((sum_value - mean) / std) ** 2) * penalty
    return weight

def generate_prediction_hot(frequencies, n_numbers=5):
    """Generate prediction favoring hot numbers"""
    # Sort numbers by frequency
    sorted_numbers = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    
    # Select from top 20 most frequent numbers (increased from 15)
    top_numbers = [num for num, _ in sorted_numbers[:20]]
    prediction = []
    
    while len(prediction) < n_numbers:
        # Weight probabilities towards hotter numbers
        weights = [frequencies[num] for num in top_numbers]
        # Add small random factor to avoid getting stuck in patterns
        weights = [w + random.random() * 0.1 for w in weights]
        num = random.choices(top_numbers, weights=weights)[0]
        if num not in prediction:
            prediction.append(num)
    
    return sorted(prediction)

def generate_prediction_cold(frequencies, n_numbers=5):
    """Generate prediction favoring cold numbers"""
    # Sort numbers by frequency
    sorted_numbers = sorted(frequencies.items(), key=lambda x: x[1])
    
    # Select from 20 least frequent numbers (increased from 15)
    cold_numbers = [num for num, _ in sorted_numbers[:20]]
    prediction = []
    
    while len(prediction) < n_numbers:
        # Inverse weights to favor colder numbers
        weights = [1/(freq + 0.1) for _, freq in sorted_numbers[:20]]  # Added small constant to avoid division by zero
        # Add small random factor
        weights = [w + random.random() * 0.1 for w in weights]
        num = random.choices(cold_numbers, weights=weights)[0]
        if num not in prediction:
            prediction.append(num)
    
    return sorted(prediction)

def generate_prediction_balanced(frequencies, n_numbers=5):
    """Generate balanced prediction considering both hot/cold and sum distribution"""
    all_numbers = list(range(1, 50))
    prediction = []
    
    while len(prediction) < n_numbers:
        # Generate a candidate number
        remaining_numbers = [n for n in all_numbers if n not in prediction]
        
        # Calculate current sum
        current_sum = sum(prediction)
        
        valid_candidates = []
        valid_weights = []
        
        for num in remaining_numbers:
            candidate_sum = current_sum + num
            if len(prediction) == 4:  # Last number
                sum_weight = calculate_sum_probability(candidate_sum)
            else:
                sum_weight = 1
                
            # Combine frequency and sum weights
            freq_weight = (frequencies[num] + 1) / (max(frequencies.values()) + 1)
            total_weight = sum_weight * freq_weight
            
            valid_candidates.append(num)
            valid_weights.append(total_weight)
        
        # Normalize weights
        weights = np.array(valid_weights) / sum(valid_weights)
        
        # Select number
        selected = random.choices(valid_candidates, weights=weights)[0]
        prediction.append(selected)
    
    return sorted(prediction)

def predict_lucky_number(df):
    """Predict lucky number based on historical frequency"""
    # Use all historical data with exponential decay
    lucky_numbers = df['numero_chance'].tolist()
    total_draws = len(lucky_numbers)
    
    frequencies = Counter()
    decay_factor = 0.995
    
    for idx, num in enumerate(lucky_numbers):
        try:
            num = int(num)  # Ensure number is integer
            if 1 <= num <= 10:  # Only count valid lucky numbers
                weight = decay_factor ** (total_draws - idx - 1)
                frequencies[num] += weight
        except (ValueError, TypeError):
            continue
    
    # Weight towards more frequent numbers but allow some randomness
    numbers = list(range(1, 11))
    weights = [frequencies.get(n, 0.1) for n in numbers]  # Use 0.1 as base weight for numbers never drawn
    
    return random.choices(numbers, weights=weights)[0]

def parse_date(date_str):
    """Parse date string in various formats"""
    formats = ['%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d']
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError("Invalid date format. Please use YYYY-MM-DD or DD/MM/YYYY")

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

def evaluate_strategy(strategy_func, frequencies, test_data, n_predictions=100):
    """Evaluate a prediction strategy over multiple test cases"""
    results = []
    
    for _ in range(n_predictions):
        prediction = strategy_func(frequencies)
        
        # Evaluate against each test draw
        for _, row in test_data.iterrows():
            actual = sorted([row['boule_1'], row['boule_2'], row['boule_3'], row['boule_4'], row['boule_5']])
            eval_result = evaluate_prediction(prediction, actual)
            results.append(eval_result)
    
    # Calculate average metrics
    avg_matches = np.mean([r['matches'] for r in results])
    avg_sum_diff = np.mean([r['sum_diff'] for r in results])
    avg_within_5 = np.mean([r['numbers_within_5'] for r in results])
    match_distribution = Counter([r['matches'] for r in results])
    
    return {
        'avg_matches': avg_matches,
        'avg_sum_diff': avg_sum_diff,
        'avg_within_5': avg_within_5,
        'match_distribution': match_distribution
    }

def evaluate_lucky_number(test_data, n_predictions=100):
    """Evaluate lucky number predictions"""
    matches = 0
    total_predictions = n_predictions * len(test_data)
    
    for _ in range(n_predictions):
        prediction = predict_lucky_number(test_data)
        matches += sum(1 for _, row in test_data.iterrows() if row['numero_chance'] == prediction)
    
    return matches / total_predictions

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate lottery predictions')
    parser.add_argument('--date', type=str, help='Target date for prediction (YYYY-MM-DD or DD/MM/YYYY)')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation on test data')
    args = parser.parse_args()
    
    # Load data
    df = load_and_prepare_data()
    
    if args.evaluate:
        print("\nRunning evaluation on test data...")
        
        # Split data into training and testing sets (80-20 split)
        train_size = int(0.8 * len(df))
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]
        
        print(f"\nTraining data: {len(train_data)} draws")
        print(f"Testing data: {len(test_data)} draws")
        
        # Get frequencies from training data
        frequencies = get_number_frequencies(train_data)
        
        # Evaluate each strategy
        print("\nEvaluating prediction strategies...")
        
        strategies = {
            'Hot Numbers': lambda f: generate_prediction_hot(f),
            'Cold Numbers': lambda f: generate_prediction_cold(f),
            'Balanced': lambda f: generate_prediction_balanced(f)
        }
        
        for name, strategy in strategies.items():
            results = evaluate_strategy(strategy, frequencies, test_data)
            
            print(f"\n{name} Strategy Results:")
            print(f"Average matches per draw: {results['avg_matches']:.2f}")
            print(f"Average sum difference: {results['avg_sum_diff']:.2f}")
            print(f"Average numbers within ±5: {results['avg_within_5']:.2f}")
            print("\nMatch distribution:")
            for matches, count in sorted(results['match_distribution'].items()):
                percentage = (count / sum(results['match_distribution'].values())) * 100
                print(f"{matches} matches: {percentage:.1f}%")
        
        # Evaluate lucky number predictions
        lucky_accuracy = evaluate_lucky_number(test_data)
        print(f"\nLucky Number Accuracy: {lucky_accuracy:.2%}")
        
    else:
        # Regular prediction mode
        # Parse target date if provided
        target_date = None
        if args.date:
            try:
                target_date = parse_date(args.date)
                print(f"\nGenerating predictions for: {target_date.strftime('%Y-%m-%d')}")
                
                # Validate target date
                min_date = df['date_de_tirage'].min()
                max_date = df['date_de_tirage'].max() + timedelta(days=365)
                if target_date < min_date or target_date > max_date:
                    print(f"Warning: Target date is outside the range of historical data ({min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')})")
            except ValueError as e:
                print(f"Error: {str(e)}")
                return
        
        # Get frequencies from historical draws
        frequencies = get_number_frequencies(df, target_date)
        
        # Generate predictions
        hot_prediction = generate_prediction_hot(frequencies)
        cold_prediction = generate_prediction_cold(frequencies)
        balanced_prediction = generate_prediction_balanced(frequencies)
        lucky_number = predict_lucky_number(df)
        
        # Print predictions with additional context
        print("\nPredictions for next draw:")
        
        print("\n1. Hot Numbers Strategy (favoring recently frequent numbers)")
        print(f"Main numbers: {hot_prediction}")
        print(f"Sum: {sum(hot_prediction)}")
        
        print("\n2. Cold Numbers Strategy (favoring less frequent numbers)")
        print(f"Main numbers: {cold_prediction}")
        print(f"Sum: {sum(cold_prediction)}")
        
        print("\n3. Balanced Strategy (considering both frequency and sum distribution)")
        print(f"Main numbers: {balanced_prediction}")
        print(f"Sum: {sum(balanced_prediction)}")
        
        print(f"\nRecommended Lucky Number: {lucky_number}")
        
        # Print statistics
        print("\nSum Distribution in Predictions:")
        print(f"Hot Strategy Sum: {sum(hot_prediction)} (target mean: 124.9)")
        print(f"Cold Strategy Sum: {sum(cold_prediction)} (target mean: 124.9)")
        print(f"Balanced Strategy Sum: {sum(balanced_prediction)} (target mean: 124.9)")
        
        if target_date:
            # Find similar historical draws
            similar_dates = []
            for _, row in df.iterrows():
                # Only consider draws from 2008 onwards (when lucky numbers were consistently recorded)
                if row['date_de_tirage'].year >= 2008:
                    temporal_weight = get_temporal_weight(row['date_de_tirage'], target_date)
                    if temporal_weight > 0.7:  # Only show highly similar dates
                        # Only include draws with valid lucky numbers
                        lucky_num = row.get('numero_chance')
                        if not pd.isna(lucky_num):  # Only include if lucky number exists
                            try:
                                lucky_num = int(lucky_num)
                                if 1 <= lucky_num <= 10:  # Validate lucky number range
                                    similar_dates.append({
                                        'date': row['date_de_tirage'],
                                        'numbers': sorted([row['boule_1'], row['boule_2'], row['boule_3'], row['boule_4'], row['boule_5']]),
                                        'lucky': lucky_num,
                                        'similarity': temporal_weight
                                    })
                            except (ValueError, TypeError):
                                continue
            
            if similar_dates:
                print("\nHistorical draws from similar dates (since 2008):")
                similar_dates.sort(key=lambda x: x['similarity'], reverse=True)
                for i, draw in enumerate(similar_dates[:3]):  # Show top 3 similar dates
                    print(f"{draw['date'].strftime('%Y-%m-%d')}: {draw['numbers']} + {draw['lucky']}")
            else:
                print("\nNo similar historical draws found with recorded lucky numbers since 2008.")

if __name__ == "__main__":
    main() 