# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter
from freqtrade.strategy import stoploss_from_open

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from scipy.linalg import pinv
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

'''
      ______   __          __              __    __   ______   __    __        __     __    __             ______            
 /      \ /  |       _/  |            /  |  /  | /      \ /  \  /  |      /  |   /  |  /  |           /      \           
/$$$$$$  |$$ |____  / $$ |    _______ $$ | /$$/ /$$$$$$  |$$  \ $$ |     _$$ |_  $$ |  $$ |  _______ /$$$$$$  |  _______ 
$$ |  $$/ $$      \ $$$$ |   /       |$$ |/$$/  $$ ___$$ |$$$  \$$ |    / $$   | $$ |__$$ | /       |$$$  \$$ | /       |
$$ |      $$$$$$$  |  $$ |  /$$$$$$$/ $$  $$<     /   $$< $$$$  $$ |    $$$$$$/  $$    $$ |/$$$$$$$/ $$$$  $$ |/$$$$$$$/ 
$$ |   __ $$ |  $$ |  $$ |  $$ |      $$$$$  \   _$$$$$  |$$ $$ $$ |      $$ | __$$$$$$$$ |$$ |      $$ $$ $$ |$$      \ 
$$ \__/  |$$ |  $$ | _$$ |_ $$ \_____ $$ |$$  \ /  \__$$ |$$ |$$$$ |      $$ |/  |     $$ |$$ \_____ $$ \$$$$ | $$$$$$  |
$$    $$/ $$ |  $$ |/ $$   |$$       |$$ | $$  |$$    $$/ $$ | $$$ |______$$  $$/      $$ |$$       |$$   $$$/ /     $$/ 
 $$$$$$/  $$/   $$/ $$$$$$/  $$$$$$$/ $$/   $$/  $$$$$$/  $$/   $$//      |$$$$/       $$/  $$$$$$$/  $$$$$$/  $$$$$$$/  
                                                                   $$$$$$/                                               
                                                                                                                         
'''     


class GPR(IStrategy):
    """
    Gaussian Process Regression Strategy
    
    This strategy uses Gaussian Process Regression to predict future price movements
    and generates buy/sell signals based on the forecasted trend direction.
    
    Original Pine Script by LuxAlgo - adapted for Freqtrade
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Timeframe for the strategy - works best on higher timeframes
    timeframe = '1h'

    # Can this strategy go short?
    can_short = False

    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.10,
        "40": 0.05,
        "100": 0.02,
        "200": 0.01
    }

    # Optimal stoploss
    stoploss = -0.05

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Strategy parameters
    window = IntParameter(50, 200, default=200, space="buy", optimize=True)
    forecast_length = IntParameter(10, 50, default=20, space="buy", optimize=True)
    smooth_length = DecimalParameter(10.0, 50.0, default=20.0, space="buy", optimize=True)
    sigma = DecimalParameter(0.01, 0.1, default=0.51, decimals=2, space="buy", optimize=True)
    
    # Signal thresholds
    buy_threshold = DecimalParameter(0.001, 0.02, default=0.005, decimals=3, space="buy", optimize=True)
    sell_threshold = DecimalParameter(0.001, 0.02, default=0.005, decimals=3, space="sell", optimize=True)

    # Required startup candles
    startup_candle_count: int = 250

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # Cache for storing model predictions
        self._model_cache = {}
        
    def rbf_kernel(self, x1: np.ndarray, x2: np.ndarray, length_scale: float) -> np.ndarray:
        """
        Radial Basis Function (RBF) kernel implementation
        """
        # Calculate squared Euclidean distances
        distances = cdist(x1.reshape(-1, 1), x2.reshape(-1, 1), 'sqeuclidean')
        return np.exp(-distances / (2.0 * length_scale ** 2))
    
    def gaussian_process_regression(self, y_train: np.ndarray, window: int, 
                                   forecast_length: int, smooth_length: float, 
                                   sigma: float) -> tuple:
        """
        Perform Gaussian Process Regression
        
        Returns:
            tuple: (fitted_values, forecast_values)
        """
        try:
            # Create training and test indices
            x_train = np.arange(window)
            x_test = np.arange(window + forecast_length)
            
            # Compute kernel matrices
            K_train = self.rbf_kernel(x_train, x_train, smooth_length)
            K_star = self.rbf_kernel(x_train, x_test, smooth_length)
            
            # Add noise to diagonal (regularization)
            K_train_reg = K_train + sigma ** 2 * np.eye(window)
            
            # Compute inverse of regularized kernel matrix
            try:
                K_inv = pinv(K_train_reg)
            except:
                return None, None
            
            # Compute predictions
            mu = K_star.T @ K_inv @ y_train
            
            # Split into fitted and forecast parts
            fitted_values = mu[:window]
            forecast_values = mu[window:] if forecast_length > 0 else np.array([])
            
            return fitted_values, forecast_values
            
        except Exception as e:
            self.dp.send_msg(f"GPR Error: {str(e)}")
            return None, None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add GPR indicators to the dataframe
        """
        # Ensure we have enough data
        if len(dataframe) < self.window.value + self.forecast_length.value:
            return dataframe
            
        # Initialize columns
        dataframe['gpr_fitted'] = np.nan
        dataframe['gpr_forecast'] = np.nan
        dataframe['gpr_trend'] = 0
        dataframe['gpr_signal'] = 0
        dataframe['price_mean'] = ta.SMA(dataframe, timeperiod=self.window.value)
        
        # Calculate GPR for each row (only for recent data to save computation)
        start_idx = max(0, len(dataframe)-200)  # Only calculate for last 100 bars
        
        for i in range(start_idx, len(dataframe)):
            if i < self.window.value:
                continue
                
            # Get training data
            end_idx = i + 1
            start_idx_train = end_idx - self.window.value
            
            y_data = dataframe['close'].iloc[start_idx_train:end_idx].values
            price_mean = dataframe['price_mean'].iloc[i]
            
            if np.isnan(price_mean):
                continue
                
            # Normalize data (subtract mean)
            y_train = y_data - price_mean
            
            # Perform GPR
            fitted, forecast = self.gaussian_process_regression(
                y_train, 
                self.window.value,
                self.forecast_length.value,
                self.smooth_length.value,
                self.sigma.value
            )
            
            if fitted is not None and forecast is not None:
                # Store fitted value (add back the mean)
                dataframe.loc[dataframe.index[i], 'gpr_fitted'] = fitted[-1] + price_mean
                
                # Store forecast trend
                if len(forecast) > 0:
                    forecast_trend = forecast[-1] - forecast[0] if len(forecast) > 1 else forecast[0]
                    dataframe.loc[dataframe.index[i], 'gpr_forecast'] = forecast[-1] + price_mean
                    dataframe.loc[dataframe.index[i], 'gpr_trend'] = forecast_trend
                    
                    # Generate signals based on forecast trend
                    current_price = dataframe['close'].iloc[i]
                    forecast_price = forecast[-1] + price_mean
                    price_change_pct = (forecast_price - current_price) / current_price
                    
                    if price_change_pct > self.buy_threshold.value:
                        dataframe.loc[dataframe.index[i], 'gpr_signal'] = 1  # Buy signal
                    elif price_change_pct < -self.sell_threshold.value:
                        dataframe.loc[dataframe.index[i], 'gpr_signal'] = -1  # Sell signal
        
        # Add additional indicators for confirmation
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on GPR analysis, populate the entry trend for buys and sells
        """
        conditions_long = [
            (dataframe['gpr_signal'] == 1),  # GPR predicts upward movement
            (dataframe['gpr_trend'] > 0),    # Positive trend
            (dataframe['rsi'] < 70),         # Not overbought
            (dataframe['ema_fast'] > dataframe['ema_slow']),  # EMA confirmation
            (dataframe['volume'] > dataframe['volume_sma']),  # Volume confirmation
        ]
        
        conditions_short = [
            (dataframe['gpr_signal'] == -1),  # GPR predicts downward movement
            (dataframe['gpr_trend'] < 0),     # Negative trend
            (dataframe['rsi'] > 30),          # Not oversold
            (dataframe['ema_fast'] < dataframe['ema_slow']),  # EMA confirmation
            (dataframe['volume'] > dataframe['volume_sma']),  # Volume confirmation
        ]
        
        # Combine all conditions
        if conditions_long:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions_long),
                'enter_long'
            ] = 1

        if conditions_short and self.can_short:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions_short),
                'enter_short'
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on GPR analysis, populate the exit trend for buys and sells
        """
        conditions_exit_long = [
            (dataframe['gpr_signal'] == -1) |  # GPR predicts downward movement
            (dataframe['gpr_trend'] < -0.001) |  # Negative trend developing
            (dataframe['rsi'] > 75)  # Overbought
        ]
        
        conditions_exit_short = [
            (dataframe['gpr_signal'] == 1) |  # GPR predicts upward movement
            (dataframe['gpr_trend'] > 0.001) |  # Positive trend developing
            (dataframe['rsi'] < 25)  # Oversold
        ]
        
        if conditions_exit_long:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_exit_long),
                'exit_long'
            ] = 1

        if conditions_exit_short and self.can_short:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_exit_short),
                'exit_short'
            ] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss implementation using GPR signals
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Use GPR signal for dynamic stoploss
        if 'gpr_signal' in last_candle:
            if trade.is_short and last_candle['gpr_signal'] == 1:
                return 0.01  # Exit short position if strong buy signal
            elif not trade.is_short and last_candle['gpr_signal'] == -1:
                return 0.01  # Exit long position if strong sell signal
        
        return self.stoploss

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                           rate: float, time_in_force: str, current_time: datetime,
                           entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """
        Confirm trade entry based on GPR confidence
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return False
            
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Check if we have valid GPR data
        if pd.isna(last_candle.get('gpr_signal', np.nan)):
            return False
            
        # Additional confirmation for entry
        if side == "long":
            return (last_candle['gpr_signal'] == 1 and 
                   last_candle['gpr_trend'] > 0)
        elif side == "short":
            return (last_candle['gpr_signal'] == -1 and 
                   last_candle['gpr_trend'] < 0)
                   
        return True


# Helper function for reducing conditions
def reduce(function, iterable, initializer=None):
    """
    Python's built-in reduce function
    """
    it = iter(iterable)
    if initializer is None:
        value = next(it)
    else:
        value = initializer
    for element in it:
        value = function(value, element)
    return value