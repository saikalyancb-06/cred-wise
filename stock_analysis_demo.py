"""
Stock Analysis System - Demo & Test Script
Shows how to use the StockAnalysisSystem for investment decisions
"""

from stock_analysis import StockAnalysisSystem


def demo_programmatic():
    """Run the system programmatically without user input."""
    
    print("\n" + "=" * 80)
    print("STOCK ANALYSIS SYSTEM - PROGRAMMATIC DEMO")
    print("=" * 80)
    
    # Initialize system
    system = StockAnalysisSystem('stock_data.csv')
    
    # Test cases
    test_cases = [
        {
            'stock_name': 'TECH_A',
            'buy_date': None,  # Use today/latest
            'investment_horizon': 30,
            'risk_level': 'medium',
        },
        {
            'stock_name': 'FINANCE_A',
            'buy_date': None,
            'investment_horizon': 60,
            'risk_level': 'low',
        },
        {
            'stock_name': 'PHARMA_A',
            'buy_date': None,
            'investment_horizon': 90,
            'risk_level': 'high',
        },
    ]
    
    # Run analysis for each test case
    for i, test in enumerate(test_cases, 1):
        print(f"\n\n{'─' * 80}")
        print(f"Test Case {i}")
        print(f"{'─' * 80}")
        
        result = system.generate_recommendation(
            stock_name=test['stock_name'],
            buy_date=test['buy_date'],
            investment_horizon=test['investment_horizon'],
            risk_level=test['risk_level']
        )
        
        system.print_recommendation(result)


def demo_individual_functions():
    """Demonstrate individual functions of the system."""
    
    print("\n" + "=" * 80)
    print("STOCK ANALYSIS SYSTEM - INDIVIDUAL FUNCTIONS DEMO")
    print("=" * 80)
    
    system = StockAnalysisSystem('stock_data.csv')
    
    stock_name = 'TECH_A'
    print(f"\nAnalyzing: {stock_name}")
    
    # 1. Price prediction
    print("\n[1] Price Prediction")
    lower, upper, error = system.predict_buy_price(stock_name)
    if error:
        print(f"   Error: {error}")
    else:
        print(f"   Buy Price Range: ₹{lower:.2f} - ₹{upper:.2f}")
    
    # 2. Trend analysis
    print("\n[2] Trend Analysis")
    trend_data, error = system.analyze_trend(stock_name)
    if error:
        print(f"   Error: {error}")
    else:
        print(f"   Trend: {trend_data['trend']}")
        print(f"   Momentum: {trend_data['momentum']}")
        print(f"   Volatility: {trend_data['volatility']}")
        print(f"   MA_10: ₹{trend_data['ma_10']:.2f}")
        print(f"   MA_50: ₹{trend_data['ma_50']:.2f}")
    
    # 3. Future prediction
    print("\n[3] Future Trend Prediction (30 days)")
    forecast, strength = system.predict_future_trend(stock_name, horizon=30)
    print(f"   Forecast: {forecast}")
    print(f"   Strength: {strength * 100:.0f}%")
    
    # 4. Complete recommendation
    print("\n[4] Complete Recommendation")
    result = system.generate_recommendation(
        stock_name=stock_name,
        buy_date=None,
        investment_horizon=30,
        risk_level='medium'
    )
    system.print_recommendation(result)


def demo_compare_stocks():
    """Compare multiple stocks for investment."""
    
    print("\n" + "=" * 80)
    print("STOCK COMPARISON ANALYSIS")
    print("=" * 80)
    
    system = StockAnalysisSystem('stock_data.csv')
    
    companies = ['TECH_A', 'TECH_B', 'FINANCE_A', 'PHARMA_A', 'AUTO_A']
    
    print(f"\n{'Stock':<15} {'Latest Price':>15} {'Trend':>15} {'Volatility':>15} {'Recommendation':>30}")
    print("─" * 90)
    
    results = []
    
    for stock in companies:
        # Get latest price
        result = system.generate_recommendation(
            stock_name=stock,
            buy_date=None,
            investment_horizon=30,
            risk_level='medium'
        )
        
        if result.get('status') == 'SUCCESS':
            results.append(result)
            
            # Extract recommendation type
            rec_text = result['recommendation'].split(' - ')[0].strip()
            
            print(f"{stock:<15} {result['latest_price']:>15} {result['trend']:>15} {result['volatility']:>15} {rec_text:>30}")
    
    # Sort and show best opportunities
    print("\n\nBest Buy Opportunities:")
    print("─" * 50)
    
    buy_results = [r for r in results if '✅' in r['recommendation']]
    if buy_results:
        for i, r in enumerate(buy_results[:3], 1):
            print(f"{i}. {r['stock_name']}")
            print(f"   Price Range: {r['buy_price_range']}")
            print(f"   Reason: {r['reason']}")


if __name__ == '__main__':
    import sys
    
    print("\n" + "=" * 80)
    print("STOCK ANALYSIS SYSTEM - DEMO OPTIONS")
    print("=" * 80)
    print("\nChoose demo mode:")
    print("  1. Programmatic Analysis (Multiple test cases)")
    print("  2. Individual Functions (Deep dive into each feature)")
    print("  3. Stock Comparison (Find best opportunities)")
    print("  4. Interactive Mode (Enter your own parameters)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    try:
        if choice == '1':
            demo_programmatic()
        elif choice == '2':
            demo_individual_functions()
        elif choice == '3':
            demo_compare_stocks()
        elif choice == '4':
            system = StockAnalysisSystem('stock_data.csv')
            user_input = system.get_user_input()
            result = system.generate_recommendation(
                stock_name=user_input['stock_name'],
                buy_date=user_input['buy_date'],
                investment_horizon=user_input['investment_horizon'],
                risk_level=user_input['risk_level']
            )
            system.print_recommendation(result)
        else:
            print("Invalid choice. Running default demo (Programmatic)...")
            demo_programmatic()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
