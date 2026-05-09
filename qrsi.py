from freqtrade.strategy import IStrategy
from pandas import DataFrame
import pandas_ta as ta
import pandas as pd
import numpy as np

class qrsi(IStrategy):
    # Strategy parameters
    timeframe = "15m"  # Base timeframe for trading
    minimal_roi = {
        "0": 0.1,  # 10% ROI target
    }
    stoploss = -0.05  # 5% stop-loss
    trailing_stop = True
    trailing_stop_positive = 0.03  # Trailing stop activates after 3% profit
    trailing_stop_positive_offset = 0.04  # Trailing stop starts at 4% profit
    trailing_only_offset_is_reached = True

    # Hyperparameters for RSI, RMI, and filters
    rsi_period = 14
    rsi_overbought = 70  # Relaxed for exits
    rsi_oversold = 30   # Relaxed for exits
    ma_period = 20      # Shortened from 50 for faster trend detection
    atr_period = 14
    atr_threshold = 0.02
    adx_period = 14
    adx_threshold = 20
    ma_trend_period = 50
    rmi_period = 14
    rmi_momentum = 5
    rmi_overbought = 60
    rmi_oversold = 40

    def calculate_rmi(self, series: pd.Series, length: int, momentum: int) -> pd.Series:
        momentum_change = series.diff(momentum)
        up = momentum_change.where(momentum_change > 0, 0)
        down = -momentum_change.where(momentum_change < 0, 0)
        avg_up = up.ewm(span=length, adjust=False).mean()
        avg_down = down.ewm(span=length, adjust=False).mean()
        avg_down = avg_down.replace(0, np.nan)
        rs = avg_up / avg_down
        rmi = 100 - (100 / (1 + rs))
        return rmi.fillna(50)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Debug: Print detailed DataFrame info
        print(f"Pair: {metadata['pair']}")
        print(f"DataFrame columns: {list(dataframe.columns)}")
        print(f"Index type: {type(dataframe.index)}")
        print(f"First few index values: {dataframe.index[:5]}")
        print(f"First few rows:\n{dataframe.head()}")

        # Check for 'date' or 'timestamp' column and set it as index if needed
        date_column = None
        if 'date' in dataframe.columns:
            date_column = 'date'
        elif 'timestamp' in dataframe.columns:
            date_column = 'timestamp'
            dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], unit='ms')
        else:
            raise ValueError(f"DataFrame for {metadata['pair']} is missing 'date' or 'timestamp' column and does not have a DatetimeIndex")

        # Set the index to the date column for resampling
        if not isinstance(dataframe.index, pd.DatetimeIndex) and date_column:
            dataframe = dataframe.set_index(date_column)

        # Calculate RSI on the base timeframe (5m)
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=self.rsi_period)

        # Calculate Moving Average for trend confirmation (5m)
        dataframe['ma'] = ta.sma(dataframe['close'], length=self.ma_period)

        # Calculate ATR for volatility filter
        dataframe['atr'] = ta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=self.atr_period)

        # Calculate ADX for trend strength
        dataframe['adx'] = ta.adx(dataframe['high'], dataframe['low'], dataframe['close'], length=self.adx_period)['ADX_14']

        # Calculate RMI manually
        dataframe['rmi'] = self.calculate_rmi(dataframe['close'], length=self.rmi_period, momentum=self.rmi_momentum)

        # Resample to higher timeframes for RSI and trend
        for timeframe in ['1h', '4h', '1d']:
            resampled = dataframe.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            # RSI for exits
            resampled[f'rsi_{timeframe}'] = ta.rsi(resampled['close'], length=self.rsi_period)

            # Moving Average for trend (only on 1h for market trend)
            if timeframe == '1h':
                resampled['ma_trend'] = ta.sma(resampled['close'], length=self.ma_trend_period)
            resampled = resampled[[f'rsi_{timeframe}', 'ma_trend'] if timeframe == '1h' else [f'rsi_{timeframe}']].resample(self.timeframe).ffill()
            dataframe = dataframe.join(resampled, how='left')

        # Add 'date' column back from the index for Freqtrade compatibility
        dataframe['date'] = dataframe.index

        # Debug: Print DataFrame after modifications
        print(f"After processing - Pair: {metadata['pair']}")
        print(f"DataFrame columns: {list(dataframe.columns)}")
        print(f"First few rows:\n{dataframe.head()}")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Relaxed conditions: RSI and 5m MA
        rsi_oversold = (dataframe['rsi'] < 60)  # Further relaxed from 50
        rsi_overbought = (dataframe['rsi'] > 40)  # Further relaxed from 50
        close_above_ma = (dataframe['close'] > dataframe['ma'])
        close_below_ma = (dataframe['close'] < dataframe['ma'])

        # Debug: Check conditions for long entry with actual values
        long_conditions = pd.DataFrame({
            'rsi': dataframe['rsi'],
            'rsi_oversold': rsi_oversold,
            'close': dataframe['close'],
            'ma': dataframe['ma'],
            'close_above_ma': close_above_ma
        })
        print(f"Long entry conditions for {metadata['pair']} (last 5 rows):")
        print(long_conditions.tail(5))

        # Debug: Check conditions for short entry with actual values
        short_conditions = pd.DataFrame({
            'rsi': dataframe['rsi'],
            'rsi_overbought': rsi_overbought,
            'close': dataframe['close'],
            'ma': dataframe['ma'],
            'close_below_ma': close_below_ma
        })
        print(f"Short entry conditions for {metadata['pair']} (last 5 rows):")
        print(short_conditions.tail(5))

        # Relaxed long entry: RSI < 60 and price above 5m MA
        dataframe.loc[
            rsi_oversold &
            close_above_ma,
            ['enter_long', 'enter_tag']] = [1, 'long-enter']

        # Relaxed short entry: RSI > 40 and price below 5m MA
        dataframe.loc[
            rsi_overbought &
            close_below_ma,
            ['enter_short', 'enter_tag']] = [1, 'short-enter']

        # Debug: Check if any signals were generated
        print(f"Long signals for {metadata['pair']}: {dataframe['enter_long'].sum()}")
        print(f"Short signals for {metadata['pair']}: {dataframe['enter_short'].sum()}")

        # Export DataFrame with signals to CSV for inspection
        dataframe.to_csv(f"signals_{metadata['pair'].replace('/', '_')}.csv")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Initialize exit columns
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_tag'] = ''

        # Relaxed exit conditions: Use RSI on 5m timeframe
        dataframe.loc[
            (dataframe['enter_long'] == 1) & (dataframe['rsi'] > self.rsi_overbought),
            ['exit_long', 'exit_tag']] = [1, 'long_exit_rsi_overbought']

        dataframe.loc[
            (dataframe['enter_short'] == 1) & (dataframe['rsi'] < self.rsi_oversold),
            ['exit_short', 'exit_tag']] = [1, 'short_exit_rsi_oversold']

        return dataframe