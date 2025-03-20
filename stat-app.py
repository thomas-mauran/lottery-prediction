import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os

def convert_date(date_str):
    """Convert date string to datetime using multiple possible formats"""
    formats = [
        '%d/%m/%Y',  # 31/12/2023
        '%Y%m%d',    # 20231231
        '%d-%m-%Y',  # 31-12-2023
        '%Y-%m-%d'   # 2023-12-31
    ]
    
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    return None

def clean_numeric_column(series):
    """Clean numeric columns by removing currency symbols and converting to float"""
    if series.dtype == 'object':
        # Remove currency symbols, spaces, and commas
        cleaned = series.astype(str).str.replace('€', '')\
                       .str.replace(' ', '')\
                       .str.replace(',', '.')\
                       .str.replace('\\xa0', '')  # Remove non-breaking spaces
        # Convert to float, replacing errors with NaN
        return pd.to_numeric(cleaned, errors='coerce')
    return series

def load_all_data():
    """Load and combine all EuroMillions CSV files"""
    all_data = []
    csv_dir = 'csv'
    
    for file in os.listdir(csv_dir):
        if file.endswith('.csv') and 'euro' in file.lower():
            try:
                df = pd.read_csv(os.path.join(csv_dir, file), sep=';')
                
                # Convert dates
                df['date_de_tirage'] = df['date_de_tirage'].apply(convert_date)
                
                # Clean numeric columns for prize amounts
                prize_columns = [col for col in df.columns if 'rapport_du_rang' in col]
                for col in prize_columns:
                    df[col] = clean_numeric_column(df[col])
                
                # Skip files where date conversion failed
                if df['date_de_tirage'].isna().any():
                    st.warning(f"Skipping {file} due to date format issues")
                    continue
                
                df['source_file'] = file
                all_data.append(df)
            except Exception as e:
                st.warning(f"Could not load {file}: {str(e)}")
    
    if not all_data:
        st.error("No EuroMillions CSV files could be loaded!")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values('date_de_tirage')
    combined_df = combined_df.drop_duplicates(subset=['date_de_tirage'], keep='first')
    
    return combined_df

def plot_main_numbers_frequency(df):
    """Interactive plot of main numbers frequencies"""
    all_numbers = pd.Series()
    for col in ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']:
        all_numbers = pd.concat([all_numbers, df[col]])
    
    number_freq = all_numbers.value_counts().sort_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=number_freq.index,
        y=number_freq.values,
        name='Frequency',
        marker_color='blue'
    ))
    
    avg_freq = number_freq.mean()
    fig.add_hline(y=avg_freq, line_dash="dash", line_color="red",
                  annotation_text=f"Average ({avg_freq:.1f})")
    
    fig.update_layout(
        title='Frequency of Main Numbers (1-50)',
        xaxis_title='Number',
        yaxis_title='Frequency',
        showlegend=False
    )
    
    return fig

def plot_star_numbers_frequency(df):
    """Interactive plot of star numbers frequencies"""
    all_stars = pd.Series()
    for col in ['etoile_1', 'etoile_2']:
        all_stars = pd.concat([all_stars, df[col]])
    
    star_freq = all_stars.value_counts().sort_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=star_freq.index,
        y=star_freq.values,
        marker_color='gold',
        name='Frequency'
    ))
    
    avg_freq = star_freq.mean()
    fig.add_hline(y=avg_freq, line_dash="dash", line_color="red",
                  annotation_text=f"Average ({avg_freq:.1f})")
    
    fig.update_layout(
        title='Frequency of Star Numbers (1-12)',
        xaxis_title='Star Number',
        yaxis_title='Frequency',
        showlegend=False
    )
    
    return fig

def plot_temporal_patterns(df):
    """Interactive plot of temporal patterns"""
    df['day_of_week'] = df['date_de_tirage'].dt.day_name()
    df['month'] = df['date_de_tirage'].dt.month_name()
    
    # Day of week plot
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['day_of_week'].value_counts().reindex(day_order)
    
    fig_day = go.Figure()
    fig_day.add_trace(go.Bar(
        x=day_counts.index,
        y=day_counts.values,
        marker_color='lightblue'
    ))
    
    fig_day.update_layout(
        title='EuroMillions Draws by Day of Week',
        xaxis_title='Day of Week',
        yaxis_title='Number of Draws',
        showlegend=False
    )
    
    # Month plot
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    month_counts = df['month'].value_counts().reindex(month_order)
    
    fig_month = go.Figure()
    fig_month.add_trace(go.Bar(
        x=month_counts.index,
        y=month_counts.values,
        marker_color='lightgreen'
    ))
    
    fig_month.update_layout(
        title='EuroMillions Draws by Month',
        xaxis_title='Month',
        yaxis_title='Number of Draws',
        showlegend=False
    )
    
    return fig_day, fig_month

def plot_prize_trends(df):
    """Interactive plot of prize trends"""
    fig = go.Figure()
    
    # Add traces for different prize ranks
    for rank in range(1, 7):
        col = f'rapport_du_rang{rank}'
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date_de_tirage'],
                y=df[col],
                mode='lines',
                name=f'Rank {rank} Prize'
            ))
    
    fig.update_layout(
        title='EuroMillions Prize Amounts Over Time',
        xaxis_title='Date',
        yaxis_title='Prize Amount (EUR)',
        showlegend=True
    )
    
    return fig

def analyze_hot_cold_numbers(df, window=30):
    """Analyze hot and cold numbers for both main and star numbers"""
    recent_df = df.tail(window)
    
    # Main numbers analysis
    main_numbers = pd.Series()
    for col in ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']:
        main_numbers = pd.concat([main_numbers, recent_df[col]])
    
    main_freq = main_numbers.value_counts()
    hot_main = main_freq.head(5)
    cold_main = main_freq.tail(5)
    
    # Star numbers analysis
    star_numbers = pd.Series()
    for col in ['etoile_1', 'etoile_2']:
        star_numbers = pd.concat([star_numbers, recent_df[col]])
    
    star_freq = star_numbers.value_counts()
    hot_stars = star_freq.head(3)
    cold_stars = star_freq.tail(3)
    
    return hot_main, cold_main, hot_stars, cold_stars

def plot_sum_statistics(df):
    """Interactive plot of sum statistics for main numbers"""
    # Calculate sum for each draw
    df['sum_main_numbers'] = df[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].sum(axis=1)
    df['sum_star_numbers'] = df[['etoile_1', 'etoile_2']].sum(axis=1)
    
    # Create figures for both main and star numbers
    fig_main = go.Figure()
    sum_freq_main = df['sum_main_numbers'].value_counts().sort_index()
    fig_main.add_trace(go.Bar(
        x=sum_freq_main.index,
        y=sum_freq_main.values,
        name='Main Numbers',
        marker_color='blue'
    ))
    
    avg_sum_main = df['sum_main_numbers'].mean()
    fig_main.add_hline(y=sum_freq_main.mean(), line_dash="dash", line_color="red",
                       annotation_text=f"Average Frequency")
    
    fig_main.update_layout(
        title=f'Distribution of Main Numbers Sum (Average: {avg_sum_main:.1f})',
        xaxis_title='Sum of Main Numbers',
        yaxis_title='Frequency',
        showlegend=False
    )
    
    fig_star = go.Figure()
    sum_freq_star = df['sum_star_numbers'].value_counts().sort_index()
    fig_star.add_trace(go.Bar(
        x=sum_freq_star.index,
        y=sum_freq_star.values,
        name='Star Numbers',
        marker_color='gold'
    ))
    
    avg_sum_star = df['sum_star_numbers'].mean()
    fig_star.add_hline(y=sum_freq_star.mean(), line_dash="dash", line_color="red",
                       annotation_text=f"Average Frequency")
    
    fig_star.update_layout(
        title=f'Distribution of Star Numbers Sum (Average: {avg_sum_star:.1f})',
        xaxis_title='Sum of Star Numbers',
        yaxis_title='Frequency',
        showlegend=False
    )
    
    return fig_main, fig_star

def plot_winner_stats(df):
    """Interactive plot of winner statistics"""
    winner_cols = [col for col in df.columns if 'nombre_de_gagnant' in col]
    if not winner_cols:
        return None
        
    winner_means = df[winner_cols].mean()
    winner_maxs = df[winner_cols].max()
    
    fig = go.Figure()
    
    # Add bar for average winners
    fig.add_trace(go.Bar(
        name='Average Winners',
        x=[f'Rank {i+1}' for i in range(len(winner_cols))],
        y=winner_means.values,
        marker_color='lightblue'
    ))
    
    # Add bar for maximum winners
    fig.add_trace(go.Bar(
        name='Maximum Winners',
        x=[f'Rank {i+1}' for i in range(len(winner_cols))],
        y=winner_maxs.values,
        marker_color='darkblue'
    ))
    
    fig.update_layout(
        title='Winners by Prize Rank',
        xaxis_title='Prize Rank',
        yaxis_title='Number of Winners',
        barmode='group',
        showlegend=True
    )
    
    return fig

def plot_prize_distribution(df):
    """Create a box plot of prize distributions"""
    prize_cols = [col for col in df.columns if 'rapport_du_rang' in col]
    if not prize_cols:
        return None
        
    fig = go.Figure()
    
    for i, col in enumerate(prize_cols, 1):
        fig.add_trace(go.Box(
            y=df[col].dropna(),
            name=f'Rank {i}',
            boxpoints='outliers',
            marker_color=f'hsl({360 * i / len(prize_cols)}, 70%, 50%)'
        ))
    
    fig.update_layout(
        title='Prize Distribution by Rank',
        yaxis_title='Prize Amount (EUR)',
        showlegend=True,
        yaxis_type='log'  # Use log scale for better visualization
    )
    
    return fig

def main():
    st.set_page_config(page_title="EuroMillions Analysis Dashboard", layout="wide")
    
    st.title("EuroMillions Analysis Dashboard")
    st.write("Interactive analysis of historical EuroMillions lottery data")
    
    # Load data
    df = load_all_data()
    if df is None:
        return
    
    # Display date range
    st.sidebar.write("Data Range:")
    st.sidebar.write(f"From: {df['date_de_tirage'].min().strftime('%Y-%m-%d')}")
    st.sidebar.write(f"To: {df['date_de_tirage'].max().strftime('%Y-%m-%d')}")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['date_de_tirage'].min().date(), df['date_de_tirage'].max().date()),
        min_value=df['date_de_tirage'].min().date(),
        max_value=df['date_de_tirage'].max().date()
    )
    
    if len(date_range) == 2:
        mask = (df['date_de_tirage'].dt.date >= date_range[0]) & (df['date_de_tirage'].dt.date <= date_range[1])
        df_filtered = df[mask]
    else:
        df_filtered = df
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Number Analysis", "Temporal Patterns", "Prize Analysis", "Hot & Cold Numbers"])
    
    with tab1:
        st.subheader("Number Frequencies")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_main_numbers_frequency(df_filtered), use_container_width=True)
        with col2:
            st.plotly_chart(plot_star_numbers_frequency(df_filtered), use_container_width=True)
        
        st.subheader("Sum Statistics")
        col1, col2 = st.columns(2)
        fig_main_sum, fig_star_sum = plot_sum_statistics(df_filtered)
        with col1:
            st.plotly_chart(fig_main_sum, use_container_width=True)
        with col2:
            st.plotly_chart(fig_star_sum, use_container_width=True)
    
    with tab2:
        fig_day, fig_month = plot_temporal_patterns(df_filtered)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_day, use_container_width=True)
        with col2:
            st.plotly_chart(fig_month, use_container_width=True)
    
    with tab3:
        
        with col2:
            st.subheader("Winner Statistics")
            winner_fig = plot_winner_stats(df_filtered)
            if winner_fig:
                st.plotly_chart(winner_fig, use_container_width=True)
            else:
                st.write("No winner statistics available in the dataset.")
        
        st.subheader("Prize Distribution Analysis")
        prize_dist_fig = plot_prize_distribution(df_filtered)
        if prize_dist_fig:
            st.plotly_chart(prize_dist_fig, use_container_width=True)
        else:
            st.write("No prize distribution data available in the dataset.")
        
        # Add summary statistics
        st.subheader("Summary Statistics")
        total_prize_money = sum(
            df_filtered[col].sum() 
            for col in df_filtered.columns 
            if 'rapport_du_rang' in col
        )
        total_winners = sum(
            df_filtered[col].sum() 
            for col in df_filtered.columns 
            if 'nombre_de_gagnant' in col
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Total Prize Money Awarded",
                f"€{total_prize_money:,.2f}"
            )
        with col2:
            st.metric(
                "Total Number of Winners",
                f"{int(total_winners):,}"
            )
        with col3:
            avg_prize_per_winner = total_prize_money / total_winners if total_winners > 0 else 0
            st.metric(
                "Average Prize per Winner",
                f"€{avg_prize_per_winner:,.2f}"
            )
    
    with tab4:
        window = st.slider("Analysis Window (number of recent draws)", min_value=10, max_value=100, value=30)
        hot_main, cold_main, hot_stars, cold_stars = analyze_hot_cold_numbers(df_filtered, window)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Hot Numbers")
            st.write("Main Numbers (Most Frequent Recently):")
            st.write(hot_main)
            st.write("\nStar Numbers (Most Frequent Recently):")
            st.write(hot_stars)
        with col2:
            st.subheader("Cold Numbers")
            st.write("Main Numbers (Least Frequent Recently):")
            st.write(cold_main)
            st.write("\nStar Numbers (Least Frequent Recently):")
            st.write(cold_stars)
    
    # Display summary statistics in sidebar
    st.sidebar.subheader("Summary Statistics")
    st.sidebar.write(f"Total number of draws: {len(df_filtered):,}")
    
    # Calculate and display main number statistics
    main_sums = df_filtered[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].sum(axis=1)
    st.sidebar.write("\nMain Numbers Statistics:")
    st.sidebar.write(f"Average sum: {main_sums.mean():.1f}")
    st.sidebar.write(f"Most common sum: {main_sums.mode().iloc[0]}")
    
    # Calculate and display star number statistics
    star_sums = df_filtered[['etoile_1', 'etoile_2']].sum(axis=1)
    st.sidebar.write("\nStar Numbers Statistics:")
    st.sidebar.write(f"Average sum: {star_sums.mean():.1f}")
    st.sidebar.write(f"Most common sum: {star_sums.mode().iloc[0]}")
    
    # Download filtered data
    if st.sidebar.button("Download Filtered Data"):
        csv = df_filtered.to_csv(index=False)
        st.sidebar.download_button(
            label="Click to Download",
            data=csv,
            file_name="filtered_euromillions_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main() 