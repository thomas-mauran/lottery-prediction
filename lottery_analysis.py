import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def load_data(file_path):
    # Check both in csv folder and current directory
    if os.path.exists(os.path.join('csv', file_path)):
        file_path = os.path.join('csv', file_path)
    elif not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path} in current directory or csv folder")
    
    df = pd.read_csv(file_path, sep=';')
    df['date_de_tirage'] = pd.to_datetime(df['date_de_tirage'], format='%d/%m/%Y')
    return df

def analyze_number_frequency(df):
    # Create figure for all number frequencies
    plt.figure(figsize=(15, 8))
    
    # Analyze main numbers
    all_numbers = pd.Series()
    for col in ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']:
        all_numbers = pd.concat([all_numbers, df[col]])
    
    # Count frequencies
    number_freq = all_numbers.value_counts().sort_index()
    
    # Plot main numbers frequency
    plt.bar(number_freq.index, number_freq.values)
    plt.title('Frequency of Main Numbers (2007-2017)')
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add average line
    avg_freq = number_freq.mean()
    plt.axhline(y=avg_freq, color='r', linestyle='--', label=f'Average Frequency ({avg_freq:.1f})')
    plt.legend()
    
    plt.savefig('main_numbers_frequency.png')
    plt.close()
    
    # Analyze lucky numbers separately
    plt.figure(figsize=(10, 6))
    lucky_freq = df['numero_chance'].value_counts().sort_index()
    plt.bar(lucky_freq.index, lucky_freq.values, color='orange')
    plt.title('Frequency of Lucky Numbers (2007-2017)')
    plt.xlabel('Lucky Number')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add average line for lucky numbers
    avg_lucky_freq = lucky_freq.mean()
    plt.axhline(y=avg_lucky_freq, color='r', linestyle='--', label=f'Average Frequency ({avg_lucky_freq:.1f})')
    plt.legend()
    
    plt.savefig('lucky_numbers_frequency.png')
    plt.close()
    
    return number_freq, lucky_freq

def analyze_temporal_patterns(df):
    # Analyze patterns by day of week
    plt.figure(figsize=(12, 6))
    df['day_of_week'] = df['date_de_tirage'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['day_of_week'].value_counts().reindex(day_order)
    
    plt.bar(day_counts.index, day_counts.values)
    plt.title('Number of Draws by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Draws')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('draws_by_day.png')
    plt.close()
    
    # Analyze patterns by month
    plt.figure(figsize=(12, 6))
    df['month'] = df['date_de_tirage'].dt.month_name()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    month_counts = df['month'].value_counts().reindex(month_order)
    
    plt.bar(month_counts.index, month_counts.values)
    plt.title('Number of Draws by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Draws')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('draws_by_month.png')
    plt.close()

def analyze_winning_patterns(df):
    # Analyze prize amounts over time
    plt.figure(figsize=(15, 8))
    plt.plot(df['date_de_tirage'], df['rapport_du_rang1'], label='1st Prize')
    plt.title('1st Prize Amount Over Time')
    plt.xlabel('Date')
    plt.ylabel('Prize Amount (EUR)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('prize_amounts.png')
    plt.close()
    
    # Analyze number of winners
    plt.figure(figsize=(12, 6))
    winner_cols = ['nombre_de_gagnant_au_rang1', 'nombre_de_gagnant_au_rang2', 
                  'nombre_de_gagnant_au_rang3']
    winner_data = df[winner_cols].mean()
    
    plt.bar(range(len(winner_data)), winner_data.values)
    plt.title('Average Number of Winners by Rank')
    plt.xlabel('Rank')
    plt.ylabel('Average Number of Winners')
    plt.xticks(range(len(winner_data)), ['1st', '2nd', '3rd'])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('average_winners.png')
    plt.close()

def analyze_number_combinations(df):
    # Analyze common pairs of numbers
    pairs = []
    for i in range(1, 6):
        for j in range(i+1, 6):
            col1 = f'boule_{i}'
            col2 = f'boule_{j}'
            pairs.extend(list(zip(df[col1], df[col2])))
    
    pair_counts = pd.Series(pairs).value_counts().head(20)
    
    plt.figure(figsize=(15, 8))
    plt.bar(range(len(pair_counts)), pair_counts.values)
    plt.title('Top 20 Most Common Number Pairs')
    plt.xlabel('Pair Combination')
    plt.ylabel('Frequency')
    plt.xticks(range(len(pair_counts)), [str(pair) for pair in pair_counts.index], rotation=45)
    
    plt.tight_layout()
    plt.savefig('common_pairs.png')
    plt.close()

def generate_summary_stats(df, number_freq, lucky_freq):
    # Create a text file with summary statistics
    with open('lottery_analysis_summary.txt', 'w') as f:
        f.write("Lottery Analysis Summary (2007-2017)\n")
        f.write("=" * 50 + "\n\n")
        
        # Most common numbers
        f.write("Most Common Main Numbers:\n")
        for num, freq in number_freq.nlargest(5).items():
            f.write(f"Number {num}: {freq} times\n")
        f.write("\n")
        
        # Most common lucky numbers
        f.write("Most Common Lucky Numbers:\n")
        for num, freq in lucky_freq.nlargest(3).items():
            f.write(f"Number {num}: {freq} times\n")
        f.write("\n")
        
        # Prize statistics
        f.write("Prize Statistics:\n")
        f.write(f"Average 1st Prize: €{df['rapport_du_rang1'].mean():,.2f}\n")
        f.write(f"Maximum 1st Prize: €{df['rapport_du_rang1'].max():,.2f}\n")
        f.write(f"Minimum 1st Prize: €{df['rapport_du_rang1'].min():,.2f}\n")
        f.write("\n")
        
        # Winner statistics
        f.write("Winner Statistics:\n")
        f.write(f"Average number of 1st prize winners: {df['nombre_de_gagnant_au_rang1'].mean():.2f}\n")
        f.write(f"Average number of 2nd prize winners: {df['nombre_de_gagnant_au_rang2'].mean():.2f}\n")
        f.write(f"Average number of 3rd prize winners: {df['nombre_de_gagnant_au_rang3'].mean():.2f}\n")

def main():
    print("Loading data...")
    df = load_data('nouveau_loto.csv')
    
    print("Analyzing number frequencies...")
    number_freq, lucky_freq = analyze_number_frequency(df)
    
    print("Analyzing temporal patterns...")
    analyze_temporal_patterns(df)
    
    print("Analyzing winning patterns...")
    analyze_winning_patterns(df)
    
    print("Analyzing number combinations...")
    analyze_number_combinations(df)
    
    print("Generating summary statistics...")
    generate_summary_stats(df, number_freq, lucky_freq)
    
    print("\nAnalysis complete! The following files have been generated:")
    print("- main_numbers_frequency.png")
    print("- lucky_numbers_frequency.png")
    print("- draws_by_day.png")
    print("- draws_by_month.png")
    print("- prize_amounts.png")
    print("- average_winners.png")
    print("- common_pairs.png")
    print("- lottery_analysis_summary.txt")

if __name__ == "__main__":
    main() 