import pandas as pd
import numpy as np
from collections import Counter
import random

def load_test_data(csv_path):
    """Load and prepare test data"""
    # Load CSV with semicolon separator and French date format
    df = pd.read_csv(csv_path, sep=';')
    df['date_de_tirage'] = pd.to_datetime(df['date_de_tirage'], format='%d/%m/%Y', dayfirst=True)
    
    # Sort by date and get the most recent 52 draws for comparison
    df = df.sort_values('date_de_tirage', ascending=False)
    return df.head(52)  # Get exactly 52 draws

def generate_random_prediction():
    """Generate random EuroMillions numbers"""
    # Generate 5 unique main numbers between 1 and 50
    main_numbers = sorted(random.sample(range(1, 51), 5))
    # Generate 2 unique star numbers between 1 and 12
    star_numbers = sorted(random.sample(range(1, 13), 2))
    return main_numbers, star_numbers

def evaluate_prediction(prediction, actual_numbers, star_prediction, actual_stars):
    """Evaluate a prediction against actual numbers"""
    # Count matching main numbers
    matches = len(set(prediction).intersection(set(actual_numbers)))
    # Count matching star numbers
    star_matches = len(set(star_prediction).intersection(set(actual_stars)))
    # Calculate sum difference
    sum_diff = abs(sum(prediction) - sum(actual_numbers))
    # Count numbers within ±5 of actual numbers
    numbers_within_5 = sum(1 for p in prediction if any(abs(p - a) <= 5 for a in actual_numbers))
    
    return {
        'matches': matches,
        'star_matches': star_matches,
        'sum_diff': sum_diff,
        'numbers_within_5': numbers_within_5
    }

def run_evaluation(test_data, num_predictions=52):
    """Run random number evaluation against test data"""
    results = []
    
    # Evaluate exactly num_predictions draws
    for i in range(num_predictions):
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
        
        # Generate random prediction
        pred_main, pred_stars = generate_random_prediction()
        
        # Evaluate prediction
        result = evaluate_prediction(pred_main, actual_main, pred_stars, actual_stars)
        results.append(result)
    
    # Calculate metrics
    avg_matches = np.mean([r['matches'] for r in results])
    avg_star_matches = np.mean([r['star_matches'] for r in results])
    avg_sum_diff = np.mean([r['sum_diff'] for r in results])
    avg_within_5 = np.mean([r['numbers_within_5'] for r in results])
    match_distribution = Counter([r['matches'] for r in results])
    star_match_distribution = Counter([r['star_matches'] for r in results])
    
    print("\nRandom Baseline Results:")
    print(f"Total draws evaluated: {num_predictions}")
    print(f"Average main number matches per draw: {avg_matches:.2f}")
    print(f"Average star number matches per draw: {avg_star_matches:.2f}")
    print(f"Average sum difference: {avg_sum_diff:.2f}")
    print(f"Average numbers within ±5: {avg_within_5:.2f}")
    
    print(f"\nMain number match distribution (out of {num_predictions} draws):")
    for matches in range(6):  # 0 to 5 matches
        count = match_distribution.get(matches, 0)
        percentage = (count / num_predictions) * 100
        print(f"{matches} matches: {count}/{num_predictions} ({percentage:.1f}%)")
    
    print(f"\nStar number match distribution (out of {num_predictions} draws):")
    for matches in range(3):  # 0 to 2 matches
        count = star_match_distribution.get(matches, 0)
        percentage = (count / num_predictions) * 100
        print(f"{matches} matches: {count}/{num_predictions} ({percentage:.1f}%)")
    
    print("\nSummary:")
    print(f"Draws with at least 1 main number match: {sum(count for matches, count in match_distribution.items() if matches > 0)}/{num_predictions}")
    print(f"Draws with at least 1 star match: {sum(count for matches, count in star_match_distribution.items() if matches > 0)}/{num_predictions}")
    max_main_matches = max(match_distribution.keys())
    max_star_matches = max(star_match_distribution.keys())
    print(f"Best performance: {max_main_matches} main numbers and {max_star_matches} star numbers matched")

def main():
    print("Loading test data...")
    test_data = load_test_data('csv/euromillions_202002.csv')
    print(f"Loaded {len(test_data)} test draws")
    
    # Run evaluation with exactly 52 draws
    run_evaluation(test_data, num_predictions=52)

if __name__ == "__main__":
    main() 