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
    """Load and combine all CSV files from the csv directory"""
    all_data = []
    csv_dir = 'csv'
    
    for file in os.listdir(csv_dir):
        if file.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(csv_dir, file), sep=';')
                
                # Convert dates using multiple formats
                df['date_de_tirage'] = df['date_de_tirage'].apply(convert_date)
                
                # Clean numeric columns
                numeric_columns = ['rapport_du_rang1', 'rapport_du_rang2', 'rapport_du_rang3',
                                 'rapport_du_rang4', 'rapport_du_rang5', 'rapport_du_rang6']
                
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = clean_numeric_column(df[col])
                
                # Skip files where date conversion failed
                if df['date_de_tirage'].isna().any():
                    st.warning(f"Skipping {file} due to date format issues")
                    continue
                
                df['source_file'] = file  # Keep track of which file the data came from
                all_data.append(df)
            except Exception as e:
                st.warning(f"Could not load {file}: {str(e)}")
    
    if not all_data:
        st.error("No CSV files could be loaded!")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by date
    combined_df = combined_df.sort_values('date_de_tirage')
    
    # Remove any duplicate draws
    combined_df = combined_df.drop_duplicates(subset=['date_de_tirage'], keep='first')
    
    return combined_df

def plot_number_frequency(df):
    """Interactive plot of number frequencies"""
    all_numbers = pd.Series()
    for col in ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']:
        all_numbers = pd.concat([all_numbers, df[col]])
    
    number_freq = all_numbers.value_counts().sort_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=number_freq.index,
        y=number_freq.values,
        name='Frequency'
    ))
    
    # Add average line
    avg_freq = number_freq.mean()
    fig.add_hline(y=avg_freq, line_dash="dash", line_color="red",
                  annotation_text=f"Average ({avg_freq:.1f})")
    
    fig.update_layout(
        title='Frequency of Main Numbers',
        xaxis_title='Number',
        yaxis_title='Frequency',
        showlegend=False
    )
    
    return fig

def plot_lucky_number_frequency(df):
    """Interactive plot of lucky number frequencies"""
    lucky_freq = df['numero_chance'].value_counts().sort_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=lucky_freq.index,
        y=lucky_freq.values,
        marker_color='orange',
        name='Frequency'
    ))
    
    avg_freq = lucky_freq.mean()
    fig.add_hline(y=avg_freq, line_dash="dash", line_color="red",
                  annotation_text=f"Average ({avg_freq:.1f})")
    
    fig.update_layout(
        title='Frequency of Lucky Numbers',
        xaxis_title='Lucky Number',
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
        y=day_counts.values
    ))
    
    fig_day.update_layout(
        title='Number of Draws by Day of Week',
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
        y=month_counts.values
    ))
    
    fig_month.update_layout(
        title='Number of Draws by Month',
        xaxis_title='Month',
        yaxis_title='Number of Draws',
        showlegend=False
    )
    
    return fig_day, fig_month

def plot_prize_trends(df):
    """Interactive plot of prize trends"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date_de_tirage'],
        y=df['rapport_du_rang1'],
        mode='lines',
        name='1st Prize'
    ))
    
    fig.update_layout(
        title='1st Prize Amount Over Time',
        xaxis_title='Date',
        yaxis_title='Prize Amount (EUR)',
        showlegend=True
    )
    
    return fig

def plot_winner_stats(df):
    """Interactive plot of winner statistics"""
    winner_cols = ['nombre_de_gagnant_au_rang1', 'nombre_de_gagnant_au_rang2',
                   'nombre_de_gagnant_au_rang3']
    winner_data = df[winner_cols].mean()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['1st Prize', '2nd Prize', '3rd Prize'],
        y=winner_data.values
    ))
    
    fig.update_layout(
        title='Average Number of Winners by Rank',
        xaxis_title='Prize Rank',
        yaxis_title='Average Number of Winners',
        showlegend=False
    )
    
    return fig

def analyze_hot_cold_numbers(df, window=30):
    """Analyze hot and cold numbers based on recent draws"""
    recent_df = df.tail(window)
    all_numbers = pd.Series()
    for col in ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']:
        all_numbers = pd.concat([all_numbers, recent_df[col]])
    
    recent_freq = all_numbers.value_counts()
    
    hot_numbers = recent_freq.head(5)
    cold_numbers = recent_freq.tail(5)
    
    return hot_numbers, cold_numbers

def plot_sum_statistics(df):
    """Interactive plot of sum statistics for drawn numbers"""
    # Calculate sum for each draw
    df['sum_numbers'] = df[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].sum(axis=1)
    
    # Create histogram of sums
    sum_freq = df['sum_numbers'].value_counts().sort_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sum_freq.index,
        y=sum_freq.values,
        name='Frequency'
    ))
    
    # Add average line
    avg_sum = df['sum_numbers'].mean()
    fig.add_hline(y=sum_freq.mean(), line_dash="dash", line_color="red",
                  annotation_text=f"Average Frequency")
    
    fig.update_layout(
        title=f'Distribution of Number Sums (Average Sum: {avg_sum:.1f})',
        xaxis_title='Sum of Numbers',
        yaxis_title='Frequency',
        showlegend=False
    )
    
    return fig

def main():
    st.set_page_config(page_title="Lottery Analysis Dashboard", layout="wide")
    
    st.title("French Lottery Analysis Dashboard")
    st.write("Interactive analysis of historical lottery data")
    
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
        st.subheader("Individual Number Frequencies")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_number_frequency(df_filtered), use_container_width=True)
        with col2:
            st.plotly_chart(plot_lucky_number_frequency(df_filtered), use_container_width=True)
            
        st.subheader("Sum Statistics")
        st.plotly_chart(plot_sum_statistics(df_filtered), use_container_width=True)
        
        # Add sum statistics to sidebar
        st.sidebar.subheader("Sum Statistics")
        sum_series = df_filtered[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].sum(axis=1)
        st.sidebar.write(f"Minimum sum: {sum_series.min()}")
        st.sidebar.write(f"Maximum sum: {sum_series.max()}")
        st.sidebar.write(f"Average sum: {sum_series.mean():.1f}")
        st.sidebar.write(f"Most common sum: {sum_series.mode().iloc[0]}")
    
    with tab2:
        fig_day, fig_month = plot_temporal_patterns(df_filtered)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_day, use_container_width=True)
        with col2:
            st.plotly_chart(fig_month, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_prize_trends(df_filtered), use_container_width=True)
        with col2:
            st.plotly_chart(plot_winner_stats(df_filtered), use_container_width=True)
    
    with tab4:
        window = st.slider("Analysis Window (number of recent draws)", min_value=10, max_value=100, value=30)
        hot_numbers, cold_numbers = analyze_hot_cold_numbers(df_filtered, window)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Hot Numbers (Most Frequent Recently)")
            st.write(hot_numbers)
        with col2:
            st.subheader("Cold Numbers (Least Frequent Recently)")
            st.write(cold_numbers)
    
    # Display summary statistics
    st.sidebar.subheader("Summary Statistics")
    st.sidebar.write(f"Total number of draws: {len(df_filtered):,}")
    
    # Safely calculate prize statistics
    mean_prize = df_filtered['rapport_du_rang1'].mean()
    max_prize = df_filtered['rapport_du_rang1'].max()
    
    if pd.notna(mean_prize):
        st.sidebar.write(f"Average 1st prize: €{mean_prize:,.2f}")
    else:
        st.sidebar.write("Average 1st prize: Not available")
        
    if pd.notna(max_prize):
        st.sidebar.write(f"Highest 1st prize: €{max_prize:,.2f}")
    else:
        st.sidebar.write("Highest 1st prize: Not available")
    
    # Download filtered data
    if st.sidebar.button("Download Filtered Data"):
        csv = df_filtered.to_csv(index=False)
        st.sidebar.download_button(
            label="Click to Download",
            data=csv,
            file_name="filtered_lottery_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main() 