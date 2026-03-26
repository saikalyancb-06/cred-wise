"""
AI-Based Stock Price Prediction and Investment Analysis System
Implements full pipeline: data preprocessing, feature engineering, model training, and prediction
Now supports ANY stock ticker using yfinance for live data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime, timedelta
import yfinance as yf

warnings.filterwarnings('ignore')


class StockAnalysisSystem:
    """Complete stock analysis and prediction system."""
    
    def __init__(self, dataset_path):
        """Initialize system with dataset."""
        self.dataset_path = dataset_path
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = ['MA_10', 'MA_50', 'returns', 'volatility', 'momentum', 'Volume']
        print("🔄 Initializing Stock Analysis System...")
        self._load_and_preprocess()
        self._engineer_features()
        self._train_model()
        print("✓ System initialized and model trained")

    # ========================================================================
    # STEP 1: DATA PREPROCESSING
    # ========================================================================
    def _load_and_preprocess(self):
        """Load dataset and perform initial preprocessing."""
        print("\n[1] Loading and Preprocessing Data...")
        
        # Load dataset
        self.df = pd.read_csv(self.dataset_path)
        print(f"   ✓ Loaded {len(self.df)} records")
        
        # Convert Date to datetime
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Sort by Company and Date
        if 'Company' in self.df.columns:
            self.df = self.df.sort_values(['Company', 'Date']).reset_index(drop=True)
        else:
            self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Handle missing values
        self.df = self.df.dropna(subset=['Close', 'Volume'])
        print(f"   ✓ After handling missing values: {len(self.df)} records")
        
        # Display info
        if 'Company' in self.df.columns:
            companies = self.df['Company'].unique()
            print(f"   ✓ Companies in dataset: {len(companies)} - {list(companies)[:5]}")

    # ========================================================================
    # STEP 2: FEATURE ENGINEERING
    # ========================================================================
    def _engineer_features(self):
        """Calculate technical features for all companies."""
        print("\n[2] Feature Engineering...")
        
        if 'Company' in self.df.columns:
            # Process each company separately
            for company in self.df['Company'].unique():
                company_mask = self.df['Company'] == company
                
                # Calculate returns
                self.df.loc[company_mask, 'returns'] = self.df.loc[company_mask, 'Close'].pct_change()
                
                # Moving averages
                self.df.loc[company_mask, 'MA_10'] = self.df.loc[company_mask, 'Close'].rolling(window=10, min_periods=1).mean()
                self.df.loc[company_mask, 'MA_50'] = self.df.loc[company_mask, 'Close'].rolling(window=50, min_periods=1).mean()
                
                # Volatility (rolling std dev of returns)
                self.df.loc[company_mask, 'volatility'] = self.df.loc[company_mask, 'returns'].rolling(window=10, min_periods=1).std()
                
                # Momentum (10-day price change)
                self.df.loc[company_mask, 'momentum'] = self.df.loc[company_mask, 'Close'] - self.df.loc[company_mask, 'Close'].shift(10)
        else:
            # Single stock
            self.df['returns'] = self.df['Close'].pct_change()
            self.df['MA_10'] = self.df['Close'].rolling(window=10, min_periods=1).mean()
            self.df['MA_50'] = self.df['Close'].rolling(window=50, min_periods=1).mean()
            self.df['volatility'] = self.df['returns'].rolling(window=10, min_periods=1).std()
            self.df['momentum'] = self.df['Close'] - self.df['Close'].shift(10)
        
        # Fill NaN created by rolling/shift operations
        self.df = self.df.fillna(method='bfill').fillna(method='ffill')
        
        print(f"   ✓ Engineered features: {self.feature_cols}")
        print(f"   ✓ Features shape: {len(self.df)} x {len(self.df.columns)}")

    # ========================================================================
    # STEP 3 & 4: TRAINING DATASET & MODEL
    # ========================================================================
    def _train_model(self):
        """Prepare training data and train RandomForest model."""
        print("\n[3] Preparing Training Data...")
        
        # Drop rows with NaN in feature columns
        train_df = self.df.dropna(subset=self.feature_cols + ['Close']).copy()
        print(f"   ✓ Training samples: {len(train_df)}")
        
        # Prepare features and target
        X = train_df[self.feature_cols]
        y = train_df['Close']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"   ✓ Train set: {len(X_train)}, Test set: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("\n[4] Training RandomForest Model...")
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"   ✓ Model trained successfully")
        print(f"   ✓ MAE: ₹{mae:.2f}")
        print(f"   ✓ R² Score: {r2:.4f}")
        
        # Feature importance
        importances = self.model.feature_importances_
        print(f"\n   Feature Importance:")
        for feat, imp in zip(self.feature_cols, importances):
            print(f"   - {feat}: {imp:.4f}")

    # ========================================================================
    # STEP 5: USER INPUT FUNCTION
    # ========================================================================
    def get_user_input(self):
        """Get user input for stock analysis."""
        print("\n" + "=" * 70)
        print("STOCK INVESTMENT ANALYSIS")
        print("=" * 70)
        
        stock_name = input("\nEnter stock name/company: ").strip()
        buy_date_str = input("Enter buy date (YYYY-MM-DD) or press Enter for today: ").strip()
        
        if not buy_date_str:
            buy_date = datetime.now().date()
        else:
            try:
                buy_date = datetime.strptime(buy_date_str, '%Y-%m-%d').date()
            except ValueError:
                print("Invalid date format. Using today.")
                buy_date = datetime.now().date()
        
        investment_horizon = input("Enter investment horizon (days, e.g., 30): ").strip()
        try:
            investment_horizon = int(investment_horizon)
        except ValueError:
            investment_horizon = 30
        
        risk_level = input("Enter risk level (low/medium/high): ").strip().lower()
        if risk_level not in ['low', 'medium', 'high']:
            risk_level = 'medium'
        
        return {
            'stock_name': stock_name,
            'buy_date': buy_date,
            'investment_horizon': investment_horizon,
            'risk_level': risk_level,
        }

    # ========================================================================
    # LIVE DATA FETCHING (supports ANY stock ticker)
    # ========================================================================
    def fetch_live_stock_data(self, ticker, period='5y'):
        """
        Fetch live historical data for any stock ticker using yfinance.
        
        Parameters:
        - ticker: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TCS.NS')
        - period: Data period ('5y', '1y', '6mo', etc.)
        
        Returns:
        - DataFrame with OHLCV data and engineered features
        - Error message if failed
        """
        try:
            print(f"\n🔄 Fetching live data for {ticker}...")
            
            # Fetch data from yfinance
            stock_data = yf.download(ticker, period=period, progress=False)
            
            if stock_data is None or len(stock_data) == 0:
                return None, f"No data found for ticker '{ticker}'. Please check the symbol."
            
            # Flatten MultiIndex columns if present (yfinance returns MultiIndex for single ticker)
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = [col[0] if col[1] == '' else col[0] for col in stock_data.columns.values]
            
            # Reset index to make Date a column
            stock_data = stock_data.reset_index()
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            stock_data = stock_data.sort_values('Date').reset_index(drop=True)
            
            # Standardize column names (remove spaces)
            stock_data.columns = [col.strip() for col in stock_data.columns]
            
            print(f"   ✓ Downloaded {len(stock_data)} records from {stock_data['Date'].min().date()} to {stock_data['Date'].max().date()}")
            
            # Check minimum data requirement
            if len(stock_data) < 50:
                return None, f"Insufficient data: only {len(stock_data)} records found. Need at least 50."
            
            # Engineer features
            stock_data = self._engineer_features_live(stock_data)
            
            return stock_data, None
            
        except Exception as e:
            return None, f"Error fetching data for '{ticker}': {str(e)}"

    def _engineer_features_live(self, df):
        """
        Apply same feature engineering as training data.
        Must match exact features used in model training:
        [MA_10, MA_50, returns, volatility, momentum, Volume]
        """
        df = df.copy()
        
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        
        # Moving averages
        df['MA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
        df['MA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        
        # Volatility (rolling std dev of returns)
        df['volatility'] = df['returns'].rolling(window=10, min_periods=1).std()
        
        # Momentum (10-day price change)
        df['momentum'] = df['Close'] - df['Close'].shift(10)
        
        # Fill NaN created by rolling/shift operations
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df

    def predict_price_live(self, ticker, prediction_date=None):
        """
        Predict price for ANY stock ticker using live data from yfinance.
        
        Parameters:
        - ticker: Stock symbol (e.g., 'AAPL')
        - prediction_date: Optional date to predict for
        
        Returns:
        - (lower_bound, upper_bound, current_price, error_message)
        """
        
        # Fetch live data
        stock_df, fetch_error = self.fetch_live_stock_data(ticker)
        
        if fetch_error:
            return None, None, None, fetch_error
        
        if prediction_date is None:
            prediction_date = stock_df.iloc[-1]['Date'].date() + timedelta(days=30)
        
        # Use latest available data
        if len(stock_df) < 50:
            return None, None, None, f"Insufficient data for prediction: {len(stock_df)} records"
        
        latest = stock_df.iloc[-1]
        current_price = float(latest['Close'])
        
        # Prepare features for prediction (MUST match training features exactly)
        features = np.array([[
            latest['MA_10'],
            latest['MA_50'],
            latest['returns'],
            latest['volatility'],
            latest['momentum'],
            latest['Volume']
        ]])
        
        # Scale and predict
        try:
            features_scaled = self.scaler.transform(features)
            model_price = float(self.model.predict(features_scaled)[0])
        except Exception as e:
            return None, None, current_price, f"Prediction error: {str(e)}"

        # Horizon-aware projection from recent drift/volatility to avoid flat forecasts.
        latest_date = latest['Date'].date()
        horizon_days = max(1, (prediction_date - latest_date).days)

        recent_returns = stock_df['returns'].tail(60).dropna()
        if len(recent_returns) >= 5:
            daily_drift = float(recent_returns.mean())
            daily_vol = float(recent_returns.std())
        else:
            daily_drift = 0.0
            daily_vol = float(latest['volatility']) if not pd.isna(latest['volatility']) else 0.02

        # Annualized cap converted to horizon to keep projections realistic.
        horizon_cap = 0.35 * np.sqrt(horizon_days / 252.0)
        projected_return = np.clip(daily_drift * horizon_days, -horizon_cap, horizon_cap)
        drift_price = current_price * (1.0 + projected_return)

        # Blend ML point estimate with drift projection.
        predicted_price = 0.6 * model_price + 0.4 * drift_price

        # Ensure there is meaningful movement when model/drift collapse near current price.
        min_move_pct = min(0.003 * np.sqrt(horizon_days), 0.04)
        if abs(predicted_price - current_price) / max(current_price, 1e-9) < min_move_pct:
            direction = 1.0 if daily_drift >= 0 else -1.0
            predicted_price = current_price * (1.0 + direction * min_move_pct)
        
        # Volatility-scaled uncertainty band based on horizon.
        vol_for_band = max(daily_vol, 0.005)
        price_range_pct = float(np.clip(1.64 * vol_for_band * np.sqrt(horizon_days), 0.02, 0.18))
        lower_bound = predicted_price * (1 - price_range_pct)
        upper_bound = predicted_price * (1 + price_range_pct)
        
        return lower_bound, upper_bound, current_price, None

    def analyze_trend_live(self, ticker):
        """Analyze current trend for ANY stock using live data."""
        
        # Fetch live data
        stock_df, fetch_error = self.fetch_live_stock_data(ticker)
        
        if fetch_error:
            return None, fetch_error
        
        if len(stock_df) < 50:
            return None, f"Insufficient data: {len(stock_df)} records (need 50+)"
        
        # Get latest data
        latest = stock_df.iloc[-1]
        
        # Determine trend
        ma_10 = latest['MA_10']
        ma_50 = latest['MA_50']
        
        if ma_10 > ma_50:
            trend = "Uptrend"
        elif ma_10 < ma_50:
            trend = "Downtrend"
        else:
            trend = "Sideways"
        
        # Classify volatility
        volatility = latest['volatility']
        if pd.isna(volatility):
            volatility_level = "Unknown"
        elif volatility < 0.02:
            volatility_level = "Low"
        elif volatility < 0.05:
            volatility_level = "Moderate"
        else:
            volatility_level = "High"
        
        # Momentum signal
        momentum = latest['momentum']
        momentum_signal = "Positive" if momentum > 0 else "Negative"
        
        return {
            'trend': trend,
            'ma_10': ma_10,
            'ma_50': ma_50,
            'volatility': volatility_level,
            'momentum': momentum_signal,
            'confidence': 0.7 if trend != "Sideways" else 0.5,
            'latest_price': latest['Close'],
        }, None

    def predict_future_trend_live(self, ticker, horizon=30):
        """Predict trend for ANY stock using live data."""
        
        # Fetch live data
        stock_df, fetch_error = self.fetch_live_stock_data(ticker)
        
        if fetch_error:
            return "Insufficient data", 0
        
        if len(stock_df) < 50:
            return f"Insufficient data: {len(stock_df)} records", 0
        
        # Get recent prices (last 30 days if available)
        recent_prices = stock_df['Close'].tail(30).values
        
        # Calculate trend direction
        price_change = recent_prices[-1] - recent_prices[0]
        price_change_pct = (price_change / recent_prices[0]) * 100
        
        # Get latest momentum
        latest = stock_df.iloc[-1]
        momentum = latest['momentum']
        
        # Predict direction
        if momentum > 0 and price_change_pct > 0:
            forecast = "📈 Uptrend likely"
            strength = min(0.9, abs(price_change_pct) / 20)
        elif momentum < 0 and price_change_pct < 0:
            forecast = "📉 Downtrend likely"
            strength = min(0.9, abs(price_change_pct) / 20)
        else:
            forecast = "↔️ Sideways movement expected"
            strength = 0.5
        
        # Adjust based on volatility
        volatility = latest['volatility']
        if volatility and volatility > 0.05:
            forecast += " (High volatility - high risk)"
        
        return forecast, strength

    # ========================================================================
    # PREDICTION FOR ANY STOCK (NEW - GENERALIZED)
    # ========================================================================
    def generate_recommendation_live(self, ticker, investment_horizon=30, risk_level='medium'):
        """
        Generate investment recommendation for ANY stock ticker using live data.
        
        This generalizes beyond the training dataset to support any stock.
        """
        
        print(f"\n{'='*70}")
        print(f"GENERATING LIVE INVESTMENT RECOMMENDATION FOR: {ticker.upper()}")
        print(f"{'='*70}")
        
        target_date = datetime.now().date() + timedelta(days=max(1, int(investment_horizon)))

        # Get horizon-aware price prediction
        lower_bound, upper_bound, current_price, price_error = self.predict_price_live(
            ticker,
            prediction_date=target_date,
        )
        if price_error:
            return {
                'error': price_error,
                'status': 'FAILED'
            }
        
        # Get trend analysis
        trend_data, trend_error = self.analyze_trend_live(ticker)
        if trend_error:
            return {
                'error': trend_error,
                'status': 'FAILED'
            }
        
        # Get future prediction
        forecast, forecast_strength = self.predict_future_trend_live(ticker, investment_horizon)
        
        volatility_pct = trend_data['volatility']
        
        # Generate recommendation logic
        recommendation = self._generate_recommendation_logic(
            trend_data['trend'],
            trend_data['momentum'],
            volatility_pct,
            forecast_strength,
            risk_level
        )
        
        # OUTPUT
        result = {
            'ticker': ticker.upper(),
            'current_price': f"₹{current_price:.2f}",
            'predicted_range': f"₹{lower_bound:.2f} - ₹{upper_bound:.2f}",
            'price_margin': f"±{((upper_bound - lower_bound) / ((upper_bound + lower_bound) / 2) * 100):.1f}%",
            'trend': trend_data['trend'],
            'trend_confidence': f"{trend_data['confidence'] * 100:.0f}%",
            'volatility': volatility_pct,
            'momentum': trend_data['momentum'],
            'future_forecast': forecast,
            'forecast_strength': f"{forecast_strength * 100:.0f}%",
            'investment_horizon_days': investment_horizon,
            'risk_level_input': risk_level,
            'recommendation': recommendation['label'],
            'reason': recommendation['reason'],
            'score': recommendation.get('score', 0),
            'status': 'SUCCESS'
        }
        
        return result


        """Predict buy price range for stock on given date."""
        
        # Filter data for stock
        if 'Company' in self.df.columns:
            stock_df = self.df[self.df['Company'].str.lower() == stock_name.lower()].copy()
        else:
            stock_df = self.df.copy()
        
        if len(stock_df) == 0:
            return None, None, f"Stock '{stock_name}' not found in dataset"
        
        if buy_date is None:
            buy_date = stock_df['Date'].max().date()
        
        # Check if buy_date exists in data
        matching_rows = stock_df[stock_df['Date'].dt.date == buy_date]
        
        if len(matching_rows) > 0:
            # Exact date found - return actual price
            actual_price = matching_rows['Close'].values[0]
            return actual_price, actual_price, None
        
        # Date not found - predict using latest available data before buy_date
        available_data = stock_df[stock_df['Date'].dt.date < buy_date]
        
        if len(available_data) < 10:
            # Not enough historical data
            if len(stock_df) > 0:
                last_price = stock_df['Close'].iloc[-1]
                return last_price, last_price, f"Insufficient data for {buy_date}. Using last known price."
            return None, None, f"Insufficient data for '{stock_name}'"
        
        # Use latest available data point
        latest = available_data.iloc[-1]
        
        # Prepare features for prediction
        features = np.array([[
            latest['MA_10'],
            latest['MA_50'],
            latest['returns'],
            latest['volatility'],
            latest['momentum'],
            latest['Volume']
        ]])
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        predicted_price = self.model.predict(features_scaled)[0]
        
        # Calculate price range (±3%)
        price_range_pct = 0.03
        lower_bound = predicted_price * (1 - price_range_pct)
        upper_bound = predicted_price * (1 + price_range_pct)
        
        return lower_bound, upper_bound, None

    # ========================================================================
    # STEP 7: TREND ANALYSIS
    # ========================================================================
    def analyze_trend(self, stock_name):
        """Analyze current trend for stock."""
        
        # Filter data for stock
        if 'Company' in self.df.columns:
            stock_df = self.df[self.df['Company'].str.lower() == stock_name.lower()].copy()
        else:
            stock_df = self.df.copy()
        
        if len(stock_df) < 50:
            return None, "Insufficient data for trend analysis"
        
        # Get latest data
        latest = stock_df.iloc[-1]
        
        # Determine trend
        ma_10 = latest['MA_10']
        ma_50 = latest['MA_50']
        
        if ma_10 > ma_50:
            trend = "Uptrend"
            trend_signal = 1
        elif ma_10 < ma_50:
            trend = "Downtrend"
            trend_signal = -1
        else:
            trend = "Sideways"
            trend_signal = 0
        
        # Classify volatility
        volatility = latest['volatility']
        if pd.isna(volatility):
            volatility_level = "Unknown"
        elif volatility < 0.02:
            volatility_level = "Low"
        elif volatility < 0.05:
            volatility_level = "Moderate"
        else:
            volatility_level = "High"
        
        # Momentum signal
        momentum = latest['momentum']
        momentum_signal = "Positive" if momentum > 0 else "Negative"
        
        return {
            'trend': trend,
            'ma_10': ma_10,
            'ma_50': ma_50,
            'volatility': volatility_level,
            'momentum': momentum_signal,
            'confidence': 0.7 if trend != "Sideways" else 0.5,
        }, None

    # ========================================================================
    # STEP 8: FUTURE TREND PREDICTION
    # ========================================================================
    def predict_future_trend(self, stock_name, horizon=30):
        """Predict trend for next N days."""
        
        # Filter data for stock
        if 'Company' in self.df.columns:
            stock_df = self.df[self.df['Company'].str.lower() == stock_name.lower()].copy()
        else:
            stock_df = self.df.copy()
        
        if len(stock_df) < 50:
            return "Insufficient data", 0
        
        # Get recent prices (last 30 days)
        recent_prices = stock_df['Close'].tail(30).values
        
        # Calculate trend direction
        price_change = recent_prices[-1] - recent_prices[0]
        price_change_pct = (price_change / recent_prices[0]) * 100
        
        # Get latest momentum
        latest = stock_df.iloc[-1]
        momentum = latest['momentum']
        
        # Predict direction
        if momentum > 0 and price_change_pct > 0:
            forecast = "📈 Uptrend likely"
            strength = min(0.9, abs(price_change_pct) / 20)
        elif momentum < 0 and price_change_pct < 0:
            forecast = "📉 Downtrend likely"
            strength = min(0.9, abs(price_change_pct) / 20)
        else:
            forecast = "↔️ Sideways movement expected"
            strength = 0.5
        
        # Adjust based on volatility
        volatility = latest['volatility']
        if volatility > 0.05:
            forecast += " (High volatility - high risk)"
        
        return forecast, strength

    # ========================================================================
    # STEP 9: FINAL DECISION ENGINE
    # ========================================================================
    def generate_recommendation(self, stock_name, buy_date, investment_horizon, risk_level):
        """Generate final investment recommendation."""
        
        print("\n" + "=" * 70)
        print("GENERATING INVESTMENT RECOMMENDATION")
        print("=" * 70)
        
        # Get price prediction
        lower_bound, upper_bound, price_error = self.predict_buy_price(stock_name, buy_date)
        if price_error:
            return {
                'error': price_error,
                'status': 'FAILED'
            }
        
        # Get trend analysis
        trend_data, trend_error = self.analyze_trend(stock_name)
        if trend_error:
            return {
                'error': trend_error,
                'status': 'FAILED'
            }
        
        # Get future prediction
        forecast, forecast_strength = self.predict_future_trend(stock_name, investment_horizon)
        
        # Get stock data for additional metrics
        if 'Company' in self.df.columns:
            stock_df = self.df[self.df['Company'].str.lower() == stock_name.lower()].copy()
        else:
            stock_df = self.df.copy()
        
        latest_price = stock_df['Close'].iloc[-1]
        volatility_pct = trend_data['volatility']
        
        # Generate recommendation logic
        recommendation = self._generate_recommendation_logic(
            trend_data['trend'],
            trend_data['momentum'],
            volatility_pct,
            forecast_strength,
            risk_level
        )
        
        # ====== STEP 10: OUTPUT FORMAT ======
        result = {
            'stock_name': stock_name,
            'buy_date': str(buy_date),
            'latest_price': f"₹{latest_price:.2f}",
            'buy_price_range': f"₹{lower_bound:.2f} - ₹{upper_bound:.2f}",
            'price_margin': f"±{((upper_bound - lower_bound) / ((upper_bound + lower_bound) / 2) * 100):.1f}%",
            'trend': trend_data['trend'],
            'trend_confidence': f"{trend_data['confidence'] * 100:.0f}%",
            'volatility': volatility_pct,
            'momentum': trend_data['momentum'],
            'future_forecast': forecast,
            'forecast_strength': f"{forecast_strength * 100:.0f}%",
            'investment_horizon_days': investment_horizon,
            'risk_level_input': risk_level,
            'recommendation': recommendation['label'],
            'reason': recommendation['reason'],
            'status': 'SUCCESS'
        }
        
        return result

    def _generate_recommendation_logic(self, trend, momentum, volatility, forecast_strength, risk_level):
        """Generate recommendation based on analysis."""
        
        # Risk compatibility check
        risk_volatility_map = {
            'low': 'Low',
            'medium': ['Low', 'Moderate'],
            'high': ['Low', 'Moderate', 'High']
        }
        
        compatible = False
        if isinstance(risk_volatility_map[risk_level], list):
            compatible = volatility in risk_volatility_map[risk_level]
        else:
            compatible = volatility == risk_volatility_map[risk_level]
        
        # Scoring
        score = 0
        
        # Trend score
        if trend == "Uptrend":
            score += 3
        elif trend == "Downtrend":
            score -= 2
        else:
            score += 1
        
        # Momentum score
        if momentum == "Positive":
            score += 2
        else:
            score -= 1
        
        # Volatility score
        if volatility == "Low":
            score += 1
        elif volatility == "High":
            score -= 1
        
        # Forecast strength
        score += forecast_strength * 2
        
        # Risk adjustment
        if not compatible:
            score -= 2
        
        # Generate label and reason
        if score >= 4:
            label = "✅ BUY - Likely to perform well"
            reason = f"Strong {trend.lower()}, positive momentum, and good forecast confidence"
        elif score >= 2:
            label = "⚠️ MODERATE - Consider carefully"
            reason = f"Mixed signals: {trend.lower()} with {volatility.lower()} volatility"
        else:
            label = "❌ HOLD/AVOID - High risk"
            reason = f"Negative trend and/or high volatility. Wait for better entry point."
        
        if not compatible:
            label += f" (⚠️ Volatility mismatch with {risk_level} risk tolerance)"
        
        return {
            'label': label,
            'reason': reason,
            'score': score
        }

    # ========================================================================
    # REPORTING
    # ========================================================================
    def print_recommendation(self, result):
        """Pretty-print recommendation."""
        
        if result.get('status') == 'FAILED':
            print(f"\n❌ Error: {result['error']}")
            return
        
        print("\n" + "=" * 70)
        print("INVESTMENT ANALYSIS REPORT")
        print("=" * 70)
        print(f"\nStock:                  {result['stock_name']}")
        print(f"Latest Price:           {result['latest_price']}")
        print(f"Buy Price Range:        {result['buy_price_range']}")
        print(f"Price Margin:           {result['price_margin']}")
        print(f"\nTrend:                  {result['trend']} ({result['trend_confidence']} confidence)")
        print(f"Momentum:               {result['momentum']}")
        print(f"Volatility:             {result['volatility']}")
        print(f"\nFuture Forecast:        {result['future_forecast']}")
        print(f"Forecast Strength:      {result['forecast_strength']}")
        print(f"Investment Horizon:     {result['investment_horizon_days']} days")
        print(f"Your Risk Level:        {result['risk_level_input']}")
        print(f"\n{result['recommendation']}")
        print(f"Reason:                 {result['reason']}")
        print("=" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    import os
    
    # Check if dataset exists
    dataset_path = 'stock_data.csv'  # Change this to your actual dataset path
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset '{dataset_path}' not found.")
        print("Please ensure you have a CSV file with stock data.")
        print("\nExpected columns: Date, Open, High, Low, Close, Volume, Dividends, Stock Splits, Company")
        exit(1)
    
    try:
        # Initialize system
        system = StockAnalysisSystem(dataset_path)
        
        # Get user input
        user_input = system.get_user_input()
        
        # Generate recommendation
        result = system.generate_recommendation(
            stock_name=user_input['stock_name'],
            buy_date=user_input['buy_date'],
            investment_horizon=user_input['investment_horizon'],
            risk_level=user_input['risk_level']
        )
        
        # Print report
        system.print_recommendation(result)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please check your dataset and try again.")
