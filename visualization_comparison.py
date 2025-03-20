import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import subprocess
import json
import os
import torch
from sklearn_euromillions import train_model, generate_predictions

def run_models_and_collect_data(num_runs=3):
    """Run both models multiple times and collect their results"""
    random_results = []
    model_results = []
    
    print("Running evaluations...")
    
    for i in range(num_runs):
        print(f"\nRun {i+1}/{num_runs}")
        
        # Run random baseline
        print("Running random baseline...")
        random_output = subprocess.check_output(['python3', 'euromillions_random_baseline.py']).decode()
        random_results.append(parse_output(random_output, 'random'))
        
        # Run trained model with fewer epochs for testing
        print("Running trained model...")
        model_output = subprocess.check_output([
            'python3', 
            'euromillions_predictor.py', 
            '--evaluate', 
            '--epochs', 
            '200'  # Reduced epochs for testing
        ]).decode()
        model_results.append(parse_output(model_output, 'model'))
        
        print(f"Completed run {i+1}")
    
    return random_results, model_results

def parse_output(output, model_type):
    """Parse the output text to extract metrics"""
    lines = output.split('\n')
    results = {'type': model_type}
    
    for line in lines:
        if 'Average main number matches per draw:' in line:
            results['avg_main_matches'] = float(line.split(':')[1].strip())
        elif 'Average star number matches per draw:' in line:
            results['avg_star_matches'] = float(line.split(':')[1].strip())
        elif 'Average sum difference:' in line:
            results['avg_sum_diff'] = float(line.split(':')[1].strip())
        elif 'Average numbers within ±5:' in line:
            results['avg_within_5'] = float(line.split(':')[1].strip())
    
    return results

def create_visualizations(random_results, model_results):
    """Create various comparison visualizations"""
    # Convert results to DataFrames
    random_df = pd.DataFrame(random_results)
    model_df = pd.DataFrame(model_results)
    
    # Set style
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Create directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # 1. Box plots comparison
    plt.figure(figsize=(15, 8))
    metrics = ['avg_main_matches', 'avg_star_matches', 'avg_within_5']
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        data = pd.concat([
            random_df[[metric]].assign(Model='Random'),
            model_df[[metric]].assign(Model='Trained')
        ])
        sns.boxplot(x='Model', y=metric, data=data)
        plt.title(f'{metric.replace("_", " ").title()}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('plots/metrics_comparison_boxplot.png')
    plt.close()
    
    # 2. Distribution plots
    plt.figure(figsize=(15, 5))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        sns.kdeplot(data=random_df[metric], label='Random', alpha=0.6)
        sns.kdeplot(data=model_df[metric], label='Trained', alpha=0.6)
        plt.title(f'{metric.replace("_", " ").title()} Distribution')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/metrics_distribution.png')
    plt.close()
    
    # 3. Bar plot of averages with error bars
    plt.figure(figsize=(12, 6))
    
    for i, metric in enumerate(metrics):
        random_mean = random_df[metric].mean()
        random_std = random_df[metric].std()
        model_mean = model_df[metric].mean()
        model_std = model_df[metric].std()
        
        plt.bar([i-0.2], [random_mean], width=0.4, label='Random' if i==0 else '', 
                yerr=random_std, capsize=5, color='skyblue')
        plt.bar([i+0.2], [model_mean], width=0.4, label='Trained' if i==0 else '', 
                yerr=model_std, capsize=5, color='orange')
    
    plt.xticks(range(len(metrics)), [m.replace('avg_', '').replace('_', ' ').title() for m in metrics])
    plt.ylabel('Average Value')
    plt.title('Performance Comparison: Random vs Trained Model')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/average_comparison.png')
    plt.close()
    
    # Save numerical results
    results_summary = {
        'random': {
            metric: {
                'mean': random_df[metric].mean(),
                'std': random_df[metric].std()
            } for metric in metrics
        },
        'trained': {
            metric: {
                'mean': model_df[metric].mean(),
                'std': model_df[metric].std()
            } for metric in metrics
        }
    }
    
    with open('plots/results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print("\nVisualizations have been saved to the 'plots' directory:")
    print("1. metrics_comparison_boxplot.png - Box plots comparing key metrics")
    print("2. metrics_distribution.png - Distribution plots of metrics")
    print("3. average_comparison.png - Bar plot of averages with error bars")
    print("4. results_summary.json - Numerical summary of results")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for metric in metrics:
        print(f"\n{metric.replace('avg_', '').replace('_', ' ').title()}:")
        print(f"Random: {random_df[metric].mean():.3f} ± {random_df[metric].std():.3f}")
        print(f"Trained: {model_df[metric].mean():.3f} ± {model_df[metric].std():.3f}")

def get_hot_cold_predictions(df, test_size=52):
    """Generate predictions based on hot and cold numbers from training data only"""
    # Split the data into training and test sets
    train_data = df.iloc[:-test_size]  # All but the last 52 draws
    
    # Main numbers analysis using only training data
    main_numbers = pd.Series()
    for col in ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']:
        main_numbers = pd.concat([main_numbers, train_data[col]])
    
    main_freq = main_numbers.value_counts()
    hot_numbers = main_freq.head(10).index.tolist()  # Top 10 frequent numbers
    cold_numbers = main_freq.tail(10).index.tolist()  # 10 least frequent numbers
    
    # Star numbers analysis using only training data
    star_numbers = pd.Series()
    for col in ['etoile_1', 'etoile_2']:
        star_numbers = pd.concat([star_numbers, train_data[col]])
    
    star_freq = star_numbers.value_counts()
    hot_stars = star_freq.head(5).index.tolist()  # Top 5 frequent stars
    cold_stars = star_freq.tail(5).index.tolist()  # 5 least frequent stars
    
    # Generate predictions using hot and cold numbers
    num_predictions = 1000  # Increased for smoother distributions
    hot_predictions = []
    cold_predictions = []
    
    for _ in range(num_predictions):
        # Hot predictions
        hot_pred_main = np.random.choice(hot_numbers, size=5, replace=False)
        hot_pred_stars = np.random.choice(hot_stars, size=2, replace=False)
        hot_predictions.append(np.concatenate([hot_pred_main, hot_pred_stars]))
        
        # Cold predictions
        cold_pred_main = np.random.choice(cold_numbers, size=5, replace=False)
        cold_pred_stars = np.random.choice(cold_stars, size=2, replace=False)
        cold_predictions.append(np.concatenate([cold_pred_main, cold_pred_stars]))
    
    return np.array(hot_predictions), np.array(cold_predictions), df.iloc[-test_size:]

def plot_comparison_distributions(random_metrics, trained_metrics, hot_metrics, cold_metrics):
    """Plot actual match distribution comparisons"""
    plt.style.use('seaborn-whitegrid')
    sns.set_context("notebook", font_scale=1.2)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Color palette
    colors = {
        'Random': '#2E86C1',      # Blue
        'Trained': '#28B463',     # Green
        'Hot': '#E74C3C',         # Red
        'Cold': '#8E44AD'         # Purple
    }
    
    # Main number matches distribution (0-5 matches)
    main_data = []
    labels = []
    for name, metrics in [('Random', random_metrics), ('Trained', trained_metrics), 
                         ('Hot', hot_metrics), ('Cold', cold_metrics)]:
        counts = np.bincount(metrics['main_matches'], minlength=6)
        percentages = (counts / len(metrics['main_matches'])) * 100
        main_data.append(percentages)
        labels.append(name)
    
    main_data = np.array(main_data).T  # Transpose for proper plotting
    x = np.arange(6)  # 0-5 matches
    width = 0.2
    
    # Plot bars for main numbers
    for i, (name, color) in enumerate(colors.items()):
        ax1.bar(x + i*width, main_data[:, i], width, label=name, color=color, alpha=0.7)
    
    ax1.set_title('Main Number Matches Distribution\n(5 numbers)', fontsize=14, pad=20)
    ax1.set_xlabel('Number of Matches', fontsize=12)
    ax1.set_ylabel('Percentage of Draws (%)', fontsize=12)
    ax1.set_xticks(x + width*1.5)
    ax1.set_xticklabels(['0', '1', '2', '3', '4', '5'])
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Star matches distribution (0-2 matches)
    star_data = []
    for name, metrics in [('Random', random_metrics), ('Trained', trained_metrics), 
                         ('Hot', hot_metrics), ('Cold', cold_metrics)]:
        counts = np.bincount(metrics['star_matches'], minlength=3)
        percentages = (counts / len(metrics['star_matches'])) * 100
        star_data.append(percentages)
    
    star_data = np.array(star_data).T  # Transpose for proper plotting
    x = np.arange(3)  # 0-2 matches
    
    # Plot bars for star numbers
    for i, (name, color) in enumerate(colors.items()):
        ax2.bar(x + i*width, star_data[:, i], width, label=name, color=color, alpha=0.7)
    
    ax2.set_title('Star Number Matches Distribution\n(2 numbers)', fontsize=14, pad=20)
    ax2.set_xlabel('Number of Matches', fontsize=12)
    ax2.set_ylabel('Percentage of Draws (%)', fontsize=12)
    ax2.set_xticks(x + width*1.5)
    ax2.set_xticklabels(['0', '1', '2'])
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    # Add a main title
    fig.suptitle('EuroMillions Prediction Performance Comparison\nActual Match Distribution', 
                 fontsize=16, y=1.05)
    
    # Adjust layout
    plt.tight_layout()
    return fig

def generate_random_predictions(num_predictions):
    """Generate random predictions for EuroMillions"""
    predictions = []
    for _ in range(num_predictions):
        main_numbers = np.random.choice(range(1, 51), size=5, replace=False)
        star_numbers = np.random.choice(range(1, 13), size=2, replace=False)
        predictions.append(np.concatenate([main_numbers, star_numbers]))
    return np.array(predictions)

def calculate_metrics(predictions, test_data):
    """Calculate metrics for predictions against test data"""
    metrics = {
        'main_matches': [],  # Will store actual match counts (0-5)
        'star_matches': []   # Will store actual match counts (0-2)
    }
    
    # Convert test data columns to numpy array
    test_numbers = test_data[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 
                             'etoile_1', 'etoile_2']].values
    
    # For each prediction, calculate matches against each test draw
    for pred in predictions:
        for test_draw in test_numbers:
            # Calculate main number matches (exact count 0-5)
            main_match = sum(1 for x in pred[:5] if x in test_draw[:5])
            metrics['main_matches'].append(main_match)
            
            # Calculate star matches (exact count 0-2)
            star_match = sum(1 for x in pred[5:] if x in test_draw[5:])
            metrics['star_matches'].append(star_match)
    
    return {k: np.array(v) for k, v in metrics.items()}

def load_test_data(csv_path):
    """Load and prepare test data"""
    df = pd.read_csv(csv_path, sep=';')
    return df

def main():
    print("Starting performance comparison visualization...")
    
    csv_path = 'csv/euromillions_202002.csv'
    
    # Load test data
    test_data = pd.read_csv(csv_path, sep=';')
    if test_data is None or len(test_data) == 0:
        print("Error: Could not load test data")
        return
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate predictions using different methods
    print("\nGenerating predictions...")
    n_predictions = 100  # Reduced for faster processing
    
    # Random predictions
    random_predictions = generate_random_predictions(n_predictions)
    
    # Hot and cold predictions using proper train/test split
    print("Generating hot/cold predictions...")
    hot_predictions, cold_predictions, test_set = get_hot_cold_predictions(test_data, test_size=52)
    
    # Train and generate sklearn model predictions
    print("Training and generating sklearn model predictions...")
    train_model(csv_path)  # Train and save the model
    trained_predictions = generate_predictions(csv_path, num_predictions=n_predictions)
    
    # Calculate metrics using only test set
    random_metrics = calculate_metrics(random_predictions, test_set)
    hot_metrics = calculate_metrics(hot_predictions, test_set)
    cold_metrics = calculate_metrics(cold_predictions, test_set)
    trained_metrics = calculate_metrics(trained_predictions, test_set)
    
    # Create and save visualization
    print("\nCreating visualizations...")
    fig = plot_comparison_distributions(random_metrics, trained_metrics, hot_metrics, cold_metrics)
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Save the plot with high DPI
    plt.savefig('plots/match_distribution_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics with improved formatting
    print("\nSummary Statistics:")
    print("-" * 50)
    print(f"Using {len(test_data) - 52} draws for training and last 52 draws for testing")
    print("-" * 50)
    methods = ['Random', 'Trained (RF)', 'Hot Numbers', 'Cold Numbers']
    metrics = [random_metrics, trained_metrics, hot_metrics, cold_metrics]
    
    for method, metric in zip(methods, metrics):
        print(f"\n{method} Predictions:")
        main_counts = np.bincount(metric['main_matches'], minlength=6)
        star_counts = np.bincount(metric['star_matches'], minlength=3)
        
        print(f"Main number matches distribution:")
        for i, count in enumerate(main_counts):
            percentage = (count / len(metric['main_matches'])) * 100
            print(f"{i} matches: {percentage:.2f}%")
        
        print(f"\nStar matches distribution:")
        for i, count in enumerate(star_counts):
            percentage = (count / len(metric['star_matches'])) * 100
            print(f"{i} matches: {percentage:.2f}%")
        print("-" * 50)
    
    print("\nVisualization saved as 'plots/match_distribution_comparison.png'")

if __name__ == "__main__":
    main() 