import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def prepare_features(df, lookback=5):
    """
    Prepare features from historical data:
    - Previous n draws
    - Rolling statistics (mean, std, min, max)
    - Day of week, month features
    """
    features = []
    targets = []
    
    # Convert date to datetime if it's not already
    df['date_de_tirage'] = pd.to_datetime(df['date_de_tirage'])
    
    # Sort by date
    df = df.sort_values('date_de_tirage')
    
    # Create features for main numbers and stars
    main_numbers = df[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].values
    star_numbers = df[['etoile_1', 'etoile_2']].values
    
    # Add time-based features
    df['day_of_week'] = df['date_de_tirage'].dt.dayofweek
    df['month'] = df['date_de_tirage'].dt.month
    
    for i in range(lookback, len(df)):
        # Previous draws features
        prev_main = main_numbers[i-lookback:i].flatten()
        prev_stars = star_numbers[i-lookback:i].flatten()
        
        # Rolling statistics for main numbers
        rolling_mean_main = np.mean(main_numbers[i-lookback:i], axis=0)
        rolling_std_main = np.std(main_numbers[i-lookback:i], axis=0)
        rolling_min_main = np.min(main_numbers[i-lookback:i], axis=0)
        rolling_max_main = np.max(main_numbers[i-lookback:i], axis=0)
        
        # Rolling statistics for star numbers
        rolling_mean_stars = np.mean(star_numbers[i-lookback:i], axis=0)
        rolling_std_stars = np.std(star_numbers[i-lookback:i], axis=0)
        rolling_min_stars = np.min(star_numbers[i-lookback:i], axis=0)
        rolling_max_stars = np.max(star_numbers[i-lookback:i], axis=0)
        
        # Combine all features
        feature = np.concatenate([
            prev_main,
            prev_stars,
            rolling_mean_main,
            rolling_std_main,
            rolling_min_main,
            rolling_max_main,
            rolling_mean_stars,
            rolling_std_stars,
            rolling_min_stars,
            rolling_max_stars,
            [df.iloc[i]['day_of_week']],
            [df.iloc[i]['month']]
        ])
        
        # Target is the next draw
        target = np.concatenate([main_numbers[i], star_numbers[i]])
        
        features.append(feature)
        targets.append(target)
    
    return np.array(features), np.array(targets)

def train_model(csv_path, test_size=52):
    """Train the sklearn model and save it"""
    # Load data
    df = pd.read_csv(csv_path, sep=';')
    
    # Prepare features and targets
    X, y = prepare_features(df)
    
    # Split into train and test
    X_train = X[:-test_size]
    y_train = y[:-test_size]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train separate models for main numbers and stars
    main_model = RandomForestRegressor(n_estimators=100, random_state=42)
    star_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    main_model.fit(X_train_scaled, y_train[:, :5])
    star_model.fit(X_train_scaled, y_train[:, 5:])
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save models and scaler
    joblib.dump(main_model, 'models/sklearn_main_model.joblib')
    joblib.dump(star_model, 'models/sklearn_star_model.joblib')
    joblib.dump(scaler, 'models/sklearn_scaler.joblib')
    
    return main_model, star_model, scaler

def generate_predictions(csv_path, num_predictions=100, test_size=52):
    """Generate predictions using the trained sklearn model"""
    # Load data
    df = pd.read_csv(csv_path, sep=';')
    
    # Load models and scaler
    main_model = joblib.load('models/sklearn_main_model.joblib')
    star_model = joblib.load('models/sklearn_star_model.joblib')
    scaler = joblib.load('models/sklearn_scaler.joblib')
    
    # Prepare features for the last available data point
    X, _ = prepare_features(df)
    X_last = X[-test_size:]  # Use last test_size points for predictions
    X_last_scaled = scaler.transform(X_last)
    
    predictions = []
    for i in range(num_predictions):
        # Get base predictions
        main_pred = main_model.predict(X_last_scaled)
        star_pred = star_model.predict(X_last_scaled)
        
        # Round to nearest integers and ensure unique values
        for pred_idx in range(len(main_pred)):
            main_numbers = np.clip(np.round(main_pred[pred_idx]), 1, 50)
            star_numbers = np.clip(np.round(star_pred[pred_idx]), 1, 12)
            
            # Ensure unique values
            while len(set(main_numbers)) < 5:
                missing = 5 - len(set(main_numbers))
                additional = np.random.choice(range(1, 51), size=missing, replace=False)
                main_numbers = np.unique(np.concatenate([main_numbers, additional]))[:5]
            
            while len(set(star_numbers)) < 2:
                missing = 2 - len(set(star_numbers))
                additional = np.random.choice(range(1, 13), size=missing, replace=False)
                star_numbers = np.unique(np.concatenate([star_numbers, additional]))[:2]
            
            predictions.append(np.concatenate([main_numbers, star_numbers]))
    
    return np.array(predictions)

if __name__ == "__main__":
    csv_path = 'csv/euromillions_202002.csv'
    
    print("Training sklearn models...")
    main_model, star_model, scaler = train_model(csv_path)
    
    print("\nGenerating predictions...")
    predictions = generate_predictions(csv_path, num_predictions=10)
    
    print("\nSample predictions:")
    for i, pred in enumerate(predictions[:5], 1):
        print(f"Prediction {i}:")
        print(f"Main numbers: {pred[:5]}")
        print(f"Star numbers: {pred[5:]}")
        print() 