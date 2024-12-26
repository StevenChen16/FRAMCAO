import yfinance as yf
import talib as ta
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from tqdm import tqdm

def rolling_normalize(series, window=21):
    """
    Apply rolling window normalization to a time series
    
    Parameters:
    series (pd.Series): Input time series
    window (int): Size of rolling window
    
    Returns:
    pd.Series: Normalized series
    """
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    
    # Add small constant to avoid division by zero
    eps = 1e-8
    normalized = (series - rolling_mean) / (rolling_std + eps)
    
    # Forward fill NaN values at the beginning
    normalized = normalized.ffill()
    # Backward fill any remaining NaN values
    normalized = normalized.bfill()
    
    return normalized

def calculate_connors_rsi(data, rsi_period=3, streak_period=2, rank_period=100):
    """Calculate Connors RSI"""
    # Component 1: Regular RSI on price changes
    price_rsi = pd.Series(ta.RSI(data['Close'].values, timeperiod=rsi_period), index=data.index)
    
    # Component 2: Streak RSI
    daily_returns = data['Close'].diff()
    streak = pd.Series(0.0, index=data.index)
    streak_count = 0.0
    
    for i in range(1, len(data)):
        if daily_returns.iloc[i] > 0:
            if streak_count < 0:
                streak_count = 1.0
            else:
                streak_count += 1.0
        elif daily_returns.iloc[i] < 0:
            if streak_count > 0:
                streak_count = -1.0
            else:
                streak_count -= 1.0
        else:
            streak_count = 0.0
        streak.iloc[i] = streak_count
    
    streak_values = streak.values.astype(np.float64)
    streak_rsi = pd.Series(ta.RSI(streak_values, timeperiod=streak_period), index=data.index)
    
    # Component 3: Percentage Rank (ROC)
    def percent_rank(series, period):
        return series.rolling(period).apply(
            lambda x: pd.Series(x).rank().iloc[-1] / float(period) * 100,
            raw=True
        )
    
    pct_rank = percent_rank(data['Close'], rank_period)
    
    # Combine components with equal weighting
    crsi = (price_rsi + streak_rsi + pct_rank) / 3.0
    return crsi

def apply_kalman_filter(data, measurement_noise=0.1, process_noise=0.01):
    """Apply Kalman filter to price series"""
    prices = data['Close'].values
    state = np.array([prices[0], 0])
    P = np.array([[1, 0], [0, 1]])
    
    F = np.array([[1, 1], [0, 1]])
    H = np.array([[1, 0]])
    Q = np.array([[process_noise/10, 0], [0, process_noise]])
    R = np.array([[measurement_noise]])
    
    filtered_prices = []
    trends = []
    
    for price in prices:
        # Predict
        state = np.dot(F, state)
        P = np.dot(np.dot(F, P), F.T) + Q
        
        # Update
        y = price - np.dot(H, state)
        S = np.dot(np.dot(H, P), H.T) + R
        K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
        
        state = state + np.dot(K, y)
        P = P - np.dot(np.dot(K, H), P)
        
        filtered_prices.append(state[0])
        trends.append(state[1])
    
    return pd.Series(filtered_prices, index=data.index), pd.Series(trends, index=data.index)

def apply_fft_filter(data, cutoff_period):
    """Apply FFT filtering with specified cutoff period"""
    prices = data['Close'].values.astype(np.float64)
    n = len(prices)
    
    # Detrend the prices to reduce edge effects
    trend = np.linspace(prices[0], prices[-1], n)
    detrended = prices - trend
    
    # Perform FFT
    fft_result = fft(detrended)
    freqs = fftfreq(n, d=1)
    
    # Create low-pass filter
    filter_threshold = 1/cutoff_period
    filter_mask = np.abs(freqs) < filter_threshold
    fft_result_filtered = fft_result * filter_mask
    
    # Inverse FFT and add trend back
    filtered_detrended = np.real(ifft(fft_result_filtered))
    filtered_prices = filtered_detrended + trend
    
    return pd.Series(filtered_prices, index=data.index)

def download_and_prepare_data(symbol, start_date, end_date, window=21):
    """
    Download and prepare stock data with rolling window normalization
    
    Parameters:
    symbol (str): Stock symbol
    start_date (str): Start date
    end_date (str): End date
    window (int): Size of rolling window for normalization
    """
    # Download stock data
    stock = yf.download(symbol, start=start_date, end=end_date, progress=False)

    if isinstance(stock.columns, pd.MultiIndex):
        stock = stock.xs(symbol, level=1, axis=1) if symbol in stock.columns.levels[1] else stock
    
    # Ensure we have data
    if stock.empty:
        raise ValueError(f"No data downloaded for {symbol}")
    
    # Convert price columns to float64
    price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    stock = stock.astype({col: np.float64 for col in price_columns})
    
    # Create a copy to avoid SettingWithCopyWarning
    stock = stock.copy()
    
    # Calculate basic technical indicators
    stock['MA5'] = ta.SMA(stock['Close'].values, timeperiod=5)
    stock['MA20'] = ta.SMA(stock['Close'].values, timeperiod=20)
    stock['MACD'], stock['MACD_Signal'], stock['MACD_Hist'] = ta.MACD(stock['Close'].values)
    stock['RSI'] = ta.RSI(stock['Close'].values)
    stock['Upper'], stock['Middle'], stock['Lower'] = ta.BBANDS(stock['Close'].values)
    stock['Volume_MA5'] = ta.SMA(stock['Volume'].values, timeperiod=5)
    
    # Add Connors RSI
    stock['CRSI'] = calculate_connors_rsi(stock)
    
    # Add Kalman Filter estimates
    stock['Kalman_Price'], stock['Kalman_Trend'] = apply_kalman_filter(stock)
    
    # Add FFT filtered prices
    stock['FFT_21'] = apply_fft_filter(stock, 21)
    stock['FFT_63'] = apply_fft_filter(stock, 63)
    
    # Forward fill any NaN values from indicators
    stock = stock.ffill()
    
    # Backward fill any remaining NaN values at the beginning
    stock = stock.bfill()
    
    # Apply rolling window normalization to all columns
    columns_to_normalize = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MA5', 'MA20', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'RSI', 'Upper', 'Middle', 'Lower', 'Volume_MA5',
        'CRSI', 'Kalman_Price', 'Kalman_Trend',
        'FFT_21', 'FFT_63'
    ]
    
    for col in columns_to_normalize:
        if col in stock.columns:
            stock[col] = rolling_normalize(stock[col], window=window)
    
    return stock

def load_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

def combine_stock_data(symbols, start_date, end_date):
    """
    下载多只股票的数据并拼接
    
    Args:
        symbols (list): 股票代码列表
        start_date (str): 开始日期
        end_date (str): 结束日期
    
    Returns:
        pd.DataFrame: 拼接后的数据
    """
    all_data = []
    
    for symbol in tqdm(symbols):
        # 获取单个股票数据
        data = download_and_prepare_data(symbol, start_date, end_date)

        if not data.empty:
            # 将数据添加到列表中
            all_data.append(data)
    
    # 直接拼接所有数据
    combined_data = pd.concat(all_data, axis=0, ignore_index=True)
    
    return combined_data