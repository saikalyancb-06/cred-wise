import google.generativeai as genai
import yfinance as yf

def get_elite_ai_analysis(ticker, ml_data, risk_profile, investment_horizon, api_key):
    """Call Gemini API using the Elite AI Analyst template, enriched by ML data and YFinance fundamental data."""
    
    if not api_key:
        return None, "Gemini API Key is missing. Please provide your API Key."
        
    try:
        genai.configure(api_key=api_key)
        # Use gemini-2.5-flash
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        return None, f"Failed to initialize Gemini client: {str(e)}"
    
    try:
        horizon_days = int(investment_horizon)
        horizon_str = "short-term" if horizon_days < 180 else "long-term"
    except (ValueError, TypeError):
        horizon_str = "short-term"
        
    # Fetch live fundamental data to give Gemini "extra details"
    fund_data = {}
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        fund_data = {
            "Forward P/E": info.get("forwardPE", "N/A"),
            "Trailing P/E": info.get("trailingPE", "N/A"),
            "Beta": info.get("beta", "N/A"),
            "Debt to Equity": info.get("debtToEquity", "N/A"),
            "Profit Margin": f"{info.get('profitMargins', 0)*100:.2f}%" if info.get('profitMargins') else "N/A",
            "Return on Equity": f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else "N/A",
            "Revenue Growth": f"{info.get('revenueGrowth', 0)*100:.2f}%" if info.get('revenueGrowth') else "N/A",
            "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52 Week Low": info.get("fiftyTwoWeekLow", "N/A"),
        }
    except Exception:
        # Silently fail if YFinance blocks or errors, just provide what we can
        fund_data = {"Error": "Could not fetch fundamental data"}

    fund_context = "\\n".join([f"- {k}: {v}" for k, v in fund_data.items()])
    
    system_prompt = f"""You are an elite AI investment analyst.

Analyze the stock {ticker} based on:
- Current price
- Technical indicators
- Fundamental metrics
- Market sentiment

**PROPRIETARY ML ALGORITHM CONTEXT TO USE (Very Important):**
- Current Price: {ml_data.get('current_price', 'N/A')}
- ML Predicted Range: {ml_data.get('predicted_range', 'N/A')}
- Momentum Signal: {ml_data.get('momentum', 'N/A')}
- Volatility: {ml_data.get('volatility', 'N/A')}
- Live Trend: {ml_data.get('trend', 'N/A')} (Confidence: {ml_data.get('trend_confidence', 'N/A')})
- ML Recommendation: {ml_data.get('recommendation', 'N/A')}
- ML Core Reason: {ml_data.get('reason', 'N/A')}
- User Risk Profile: {risk_profile}
- Investment Horizon: {horizon_str}

**LIVE FUNDAMENTAL METRICS (From Yahoo Finance):**
{fund_context}

Return a structured report including:

1. Trend & Forecast (short-term & long-term)
2. Fundamental Analysis (growth, profitability, debt)
3. Valuation (intrinsic value, P/E comparison)
4. Technical Indicators (RSI, MACD, support/resistance)
5. Risk Analysis (beta, volatility, drawdown)
6. Scenario Forecast (bull, base, bear case)
7. AI Reasoning (top 3 factors)
8. Entry/Exit Strategy (buy zone, stop loss, targets)
9. Portfolio Fit (allocation suggestion)
10. Final Recommendation (Buy/Hold/Sell with confidence %)

Keep explanations concise but incredibly insightful, giving specific numbers and avoiding vague filler.
"""

    try:
        response = model.generate_content(
            system_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,
                max_output_tokens=2500,
            )
        )
        return response.text, None
    except Exception as e:
        return None, f"Gemini API Error: {str(e)}"
