from freqtrade.strategy import IStrategy
from pandas import DataFrame
import pandas_ta as ta

class 2Candle(IStrategy):
    # Strategy parameters
    timeframe = "5m"  # 5-minute timeframe as per the video example
    minimal_roi = {"0": 0.015}  # 1.5% ROI (1:1.5 risk-reward ratio)
    stoploss = -0.03  # 1% stop loss (adjustable)
    trailing_stop = False

    # Define custom variables
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate candle range (high - low)
        dataframe['range'] = dataframe['high'] - dataframe['low']
        dataframe['range_third'] = dataframe['range'] / 3

        # Close position: Determine if close is in upper, mid, or lower third
        dataframe['close_position'] = 0  # Default: mid
        dataframe.loc[dataframe['close'] > (dataframe['high'] - dataframe['range_third']), 'close_position'] = 1  # High close
        dataframe.loc[dataframe['close'] < (dataframe['low'] + dataframe['range_third']), 'close_position'] = -1  # Low close

        # Close comparison: Compare current close to previous candle's range
        dataframe['prev_high'] = dataframe['high'].shift(1)
        dataframe['prev_low'] = dataframe['low'].shift(1)
        dataframe['close_comparison'] = 0  # Default: range
        dataframe.loc[dataframe['close'] > dataframe['prev_high'], 'close_comparison'] = 1  # Bull candle
        dataframe.loc[dataframe['close'] < dataframe['prev_low'], 'close_comparison'] = -1  # Bear candle

        # Combine close position and close comparison into 9 patterns
        dataframe['pattern'] = dataframe['close_position'] * 3 + dataframe['close_comparison'] + 4  # Maps to 0-8 (9 patterns)

        # Simple support/resistance levels using rolling min/max
        dataframe['support'] = dataframe['low'].rolling(window=20).min()
        dataframe['resistance'] = dataframe['high'].rolling(window=20).max()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Buy conditions based on Two Candle Theory
        dataframe.loc[
            (
                # High close bull candle (pattern 4: most bullish)
                (dataframe['pattern'] == 4) &
                # Breakout above resistance
                (dataframe['close'] > dataframe['resistance'].shift(1)) &
                # Previous candle was not bearish (avoid false breakouts)
                (dataframe['pattern'].shift(1) != 0)  # Not low close bear
            ),
            'enter_long'] = 1

        # Alternative buy: High close bull after support bounce
        dataframe.loc[
            (
                (dataframe['pattern'] == 4) &
                (dataframe['close'] > dataframe['support']) &
                (dataframe['pattern'].shift(1) == 0)  # Previous was low close bear
            ),
            'enter_long'] = 1

        # Sell conditions (short entry)
        dataframe.loc[
            (
                # Low close bear candle (pattern 0: most bearish)
                (dataframe['pattern'] == 0) &
                # Breakdown below support
                (dataframe['close'] < dataframe['support'].shift(1)) &
                # Previous candle was not bullish (avoid false breakdowns)
                (dataframe['pattern'].shift(1) != 4)  # Not high close bull
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long on strong bearish signal
        dataframe.loc[
            (dataframe['enter_long'].shift(1) == 1) &
            (dataframe['pattern'] == 0),  # Low close bear candle
            'exit_long'] = 1

        # Exit short on strong bullish signal
        dataframe.loc[
            (dataframe['enter_short'].shift(1) == 1) &
            (dataframe['pattern'] == 4),  # High close bull candle
            'exit_short'] = 1

        return dataframe