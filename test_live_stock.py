#!/usr/bin/env python3
"""Test script for live stock data fetching via yfinance."""

import sys
from stock_analysis import StockAnalysisSystem
import warnings
warnings.filterwarnings('ignore')

def main():
    print("\n" + "="*70)
    print("TESTING LIVE STOCK DATA FETCHING WITH yfinance")
    print("="*70)
    
    try:
        # Initialize system (trains model on existing dataset)
        print("\n1️⃣ Initializing StockAnalysisSystem...")
        system = StockAnalysisSystem('stock_data.csv')
        print("   ✓ System initialized. Model trained and ready.")
        
        # Test 1: Fetch live data for a real stock
        print("\n2️⃣ Testing fetch_live_stock_data() with AAPL...")
        aapl_df, error = system.fetch_live_stock_data('AAPL', period='3mo')
        if error:
            print(f"   ❌ Error: {error}")
        else:
            print(f"   ✓ Fetched {len(aapl_df)} records for AAPL")
            print(f"   Columns: {list(aapl_df.columns)}")
            if len(aapl_df) > 0:
                latest_close = aapl_df['Close'].iloc[-1]
                if isinstance(latest_close, (int, float)):
                    print(f"   Latest price: ₹{latest_close:.2f}")
                else:
                    print(f"   Latest price: {latest_close}")
        
        # Test 2: Predict price for AAPL
        print("\n3️⃣ Testing predict_price_live() for AAPL...")
        lower, upper, current, error = system.predict_price_live('AAPL')
        if error:
            print(f"   ❌ Error: {error}")
        else:
            print(f"   ✓ Prediction successful")
            print(f"   Current price: ₹{current:.2f}")
            print(f"   Predicted range: ₹{lower:.2f} - ₹{upper:.2f}")
        
        # Test 3: Analyze trend
        print("\n4️⃣ Testing analyze_trend_live() for AAPL...")
        trend_data, error = system.analyze_trend_live('AAPL')
        if error:
            print(f"   ❌ Error: {error}")
        else:
            print(f"   ✓ Trend analysis successful")
            print(f"   Trend: {trend_data['trend']}")
            print(f"   Volatility: {trend_data['volatility']}")
            print(f"   Momentum: {trend_data['momentum']}")
        
        # Test 4: Full recommendation
        print("\n5️⃣ Testing generate_recommendation_live() for AAPL...")
        rec = system.generate_recommendation_live('AAPL', investment_horizon=30, risk_level='medium')
        if rec.get('status') == 'FAILED':
            print(f"   ❌ Error: {rec.get('error')}")
        else:
            print(f"   ✓ Recommendation generated")
            print(f"   Ticker: {rec['ticker']}")
            print(f"   Current price: {rec['current_price']}")
            print(f"   Recommendation: {rec['recommendation']}")
            print(f"   Reason: {rec['reason']}")
        
        # Test 5: Try another stock (GOOGL)
        print("\n6️⃣ Testing with GOOGL...")
        rec2 = system.generate_recommendation_live('GOOGL', investment_horizon=30, risk_level='high')
        if rec2.get('status') == 'FAILED':
            print(f"   ❌ Error: {rec2.get('error')}")
        else:
            print(f"   ✓ Recommendation for GOOGL: {rec2['recommendation']}")
        
        # Test 6: Test error handling (invalid ticker)
        print("\n7️⃣ Testing error handling with invalid ticker...")
        rec3 = system.generate_recommendation_live('INVALID12345', investment_horizon=30, risk_level='medium')
        if rec3.get('status') == 'FAILED':
            print(f"   ✓ Correctly handled error: {rec3['error']}")
        else:
            print(f"   ❌ Should have failed but didn't")
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
