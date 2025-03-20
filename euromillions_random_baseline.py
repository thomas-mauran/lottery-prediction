import numpy as np
import pandas as pd
from datetime import datetime

def generate_random_prediction():
    """Generate random EuroMillions numbers"""
    main_numbers = np.random.choice(50, size=5, replace=False) + 1
    star_numbers = np.random.choice(12, size=2, replace=False) + 1
    return sorted(main_numbers), sorted(star_numbers)

def load_test_data(csv_path):
    """Load and prepare test data"""
    df = pd.read_csv(csv_path, parse_dates=['date_de_tirage'])
    
    # Get the most recent 52 draws (same as transformer test set)
    test_data = df.tail(52).copy()
    
    return test_data

def evaluate_prediction(pred_main, pred_stars, true_main, true_stars):
    """Evaluate a single prediction against true numbers"""
    main_matches = len(set(pred_main) & set(true_main))
    star_matches = len(set(pred_stars) & set(true_stars))
    return main_matches, star_matches

def calculate_statistics(numbers):
    """Calculate sum and gaps for a set of numbers"""
    numbers = sorted(numbers)
    gaps = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
    return {
        'sum': sum(numbers),
        'gaps': gaps,
        'avg_gap': np.mean(gaps) if gaps else 0
    }

def run_evaluation(test_data, num_predictions=1000):
    """Run multiple random predictions and evaluate them"""
    main_matches_dist = {i: 0 for i in range(6)}  # 0 to 5 matches
    star_matches_dist = {i: 0 for i in range(3)}  # 0 to 2 matches
    
    total_main_matches = 0
    total_star_matches = 0
    
    sum_stats = []
    gap_stats = []
    
    for _ in range(num_predictions):
        pred_main, pred_stars = generate_random_prediction()
        
        # Calculate statistics for this prediction
        stats = calculate_statistics(pred_main)
        sum_stats.append(stats['sum'])
        gap_stats.append(stats['avg_gap'])
        
        # Compare with each test draw
        for _, row in test_data.iterrows():
            true_main = [row[f'boule_{i}'] for i in range(1, 6)]
            true_stars = [row[f'etoile_{i}'] for i in range(1, 3)]
            
            main_matches, star_matches = evaluate_prediction(pred_main, pred_stars, true_main, true_stars)
            
            total_main_matches += main_matches
            total_star_matches += star_matches
            
            main_matches_dist[main_matches] += 1
            star_matches_dist[star_matches] += 1
    
    total_comparisons = num_predictions * len(test_data)
    
    # Calculate averages
    avg_main_matches = total_main_matches / total_comparisons
    avg_star_matches = total_star_matches / total_comparisons
    
    # Convert distributions to percentages
    for k in main_matches_dist:
        main_matches_dist[k] = (main_matches_dist[k] / total_comparisons) * 100
    for k in star_matches_dist:
        star_matches_dist[k] = (star_matches_dist[k] / total_comparisons) * 100
    
    return {
        'avg_main_matches': avg_main_matches,
        'avg_star_matches': avg_star_matches,
        'main_matches_dist': main_matches_dist,
        'star_matches_dist': star_matches_dist,
        'avg_sum': np.mean(sum_stats),
        'std_sum': np.std(sum_stats),
        'avg_gap': np.mean(gap_stats),
        'std_gap': np.std(gap_stats)
    }

def main():
    print("EuroMillions Random Number Generator Baseline")
    print("-------------------------------------------")
    
    # Load test data
    test_data = load_test_data('csv/euromillions_202002.csv')
    print(f"\nEvaluating on {len(test_data)} test draws...")
    
    # Run evaluation with different numbers of predictions
    for num_preds in [100, 1000, 10000]:
        print(f"\nRunning evaluation with {num_preds} predictions...")
        results = run_evaluation(test_data, num_predictions=num_preds)
        
        print(f"\nResults for {num_preds} predictions:")
        print(f"Average main number matches: {results['avg_main_matches']:.4f}")
        print(f"Average star number matches: {results['avg_star_matches']:.4f}")
        
        print("\nMain number matches distribution:")
        for matches, percentage in results['main_matches_dist'].items():
            print(f"{matches} matches: {percentage:.2f}%")
        
        print("\nStar number matches distribution:")
        for matches, percentage in results['star_matches_dist'].items():
            print(f"{matches} matches: {percentage:.2f}%")
        
        print("\nNumber Statistics:")
        print(f"Average sum: {results['avg_sum']:.1f} ± {results['std_sum']:.1f}")
        print(f"Average gap: {results['avg_gap']:.1f} ± {results['std_gap']:.1f}")

if __name__ == "__main__":
    main() 