"""
Sample Stock Data Generator
Creates realistic multi-stock historical data for testing the analysis system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_stock_data(output_file='stock_data.csv', days=1260):
    """
    Generate realistic multi-stock historical data (5 years)
    
    Parameters:
    - output_file: CSV filename to save
    - days: number of trading days (approx 5 years = 1260 days)
    """
    
    print("Generating sample stock data...")
    
    np.random.seed(42)
    
    # Company list
    companies = ['TECH_A', 'TECH_B', 'FINANCE_A', 'PHARMA_A', 'AUTO_A']
    
    # Starting prices
    base_prices = {
        'TECH_A': 150,
        'TECH_B': 200,
        'FINANCE_A': 300,
        'PHARMA_A': 250,
        'AUTO_A': 400,
    }
    
    # Volatility per stock
    volatility = {
        'TECH_A': 0.025,
        'TECH_B': 0.030,
        'FINANCE_A': 0.015,
        'PHARMA_A': 0.020,
        'AUTO_A': 0.022,
    }
    
    # Trend (drift)
    trend = {
        'TECH_A': 0.0008,      # Slight uptrend
        'TECH_B': 0.0005,      # Slight uptrend
        'FINANCE_A': 0.0003,   # Minor uptrend
        'PHARMA_A': 0.0006,    # Slight uptrend
        'AUTO_A': -0.0002,     # Slight downtrend
    }
    
    data = []
    start_date = datetime.now() - timedelta(days=days)
    
    for company in companies:
        current_price = base_prices[company]
        current_date = start_date
        
        for day in range(days):
            # Skip weekends (simplified)
            if current_date.weekday() > 4:
                current_date += timedelta(days=1)
                continue
            
            # Generate price movement (log returns)
            daily_return = np.random.normal(
                trend[company],
                volatility[company]
            )
            
            # Update price
            new_price = current_price * (1 + daily_return)
            
            # Generate OHLCV data
            open_price = current_price
            close_price = new_price
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
            
            volume = np.random.uniform(1_000_000, 10_000_000)
            dividends = np.random.choice([0, 0, 0, 0.5])  # 25% chance of dividends
            stock_splits = 1
            
            data.append({
                'Date': current_date,
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': int(volume),
                'Dividends': dividends,
                'Stock Splits': stock_splits,
                'Company': company,
            })
            
            current_price = new_price
            current_date += timedelta(days=1)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df = df.sort_values(['Company', 'Date']).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✓ Sample data saved to '{output_file}'")
    print(f"  - Records: {len(df)}")
    print(f"  - Companies: {df['Company'].nunique()}")
    print(f"  - Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df


if __name__ == '__main__':
    # Generate sample data
    df = generate_sample_stock_data()
    print("\nSample data preview:")
    print(df.head(10))
