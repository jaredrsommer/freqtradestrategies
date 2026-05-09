from freqtrade.strategy import IStrategy
from pandas import DataFrame
import pandas_ta as ta
from technical import qtpylib

class TwoCandle(IStrategy):
    # Strategy parameters
    timeframe = "1h"  # 5-minute timeframe as per the video example
    minimal_roi = {"0": 0.015, "15": 0.012, "30": 0.05, "60": 0.025}  # 1.5% ROI (1:1.5 risk-reward ratio)
    stoploss = -0.03  # 1% stop loss (adjustable)
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.013
    trailing_stop_positive_offset = 0.035
    trailing_only_offset_is_reached = True

    plot_config = {
        "main_plot": {
        "support": {
          "color": "green",
          "type": "line"
        },
        "resistance": {
          "color": "red",
          "type": "line"
        }
        },
        "subplots": {
        "close": {
          "close_position": {
            "color": "#76f639",
            "type": "bar"
          },
          "close_comparison": {
            "color": "#dedcf9",
            "type": "line"
          }
        },
        "pattern": {
          "pattern": {
            "color": "#165e70",
            "type": "bar"
          },
          "pattern diff": {
            "color": "#744bce",
            "type": "bar"
          }
        }
        }
        }

    # Define custom variables
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        heikinashi = qtpylib.heikinashi(dataframe)

        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']
        # Calculate candle range (high - low)
        dataframe['range'] = dataframe['high'] - dataframe['low']
        dataframe['range_third'] = dataframe['range'] / 3

        # Calculate candle body (close - open)
        dataframe['body'] = abs(dataframe['close'] - dataframe['open'])
        dataframe['body_third'] = dataframe['body'] / 3

        # Simple support/resistance levels using rolling min/max
        dataframe['support'] = dataframe['low'].rolling(window=80).min().shift(80)
        dataframe['resistance'] = dataframe['high'].rolling(window=80).max().shift(80)
        dataframe['midpoint'] = ((dataframe['resistance']-dataframe['support'])/2) + dataframe['support']
        dataframe['trading_range'] = abs(dataframe['support'] - dataframe['resistance']) / 8

        # Close position: Determine if close is in upper, mid, or lower third
        dataframe['close_position'] = 0  # Default: mid
        dataframe.loc[dataframe['close'] > (dataframe['high'] - dataframe['range_third']), 'close_position'] = 1  # High close
        dataframe.loc[dataframe['close'] < (dataframe['low'] + dataframe['range_third']), 'close_position'] = -1  # Low close

        # Close comparison: Compare current close to previous candle's range
        dataframe['prev_high'] = dataframe['high'].shift(1)
        dataframe['prev_low'] = dataframe['low'].shift(1)
        dataframe['hl_close_comparison'] = 0  # Default: range
        dataframe.loc[dataframe['close'] > dataframe['prev_high'], 'hl_close_comparison'] = 1  # Bull candle
        dataframe.loc[dataframe['close'] < dataframe['prev_low'], 'hl_close_comparison'] = -1  # Bear candle

        # Close position: Determine if close is in upper, mid, or lower third
        dataframe['close_body_position'] = 0  # Default: mid
        dataframe.loc[dataframe['close'] > (dataframe['high'] - dataframe['range_third']), 'close_body_position'] = 1  # High close
        dataframe.loc[dataframe['close'] < (dataframe['low'] + dataframe['range_third']), 'close_body_position'] = -1  # Low close

        # Close comparison: Compare current close to previous candle's range
        dataframe['prev_close'] = dataframe['close'].shift(1)
        dataframe['close_comparison'] = 0  # Default: range
        dataframe.loc[dataframe['close'] > dataframe['prev_high'], 'close_comparison'] = 1  # Bull candle
        dataframe.loc[dataframe['close'] < dataframe['prev_low'], 'close_comparison'] = -1  # Bear candle



        # Combine close position and close comparison into 9 patterns
        dataframe['pattern'] = dataframe['close_position'] * 3 + dataframe['hl_close_comparison'] + 4  # Maps to 0-8 (9 patterns)
        dataframe['body_pattern'] = dataframe['close_body_position'] * 3 + dataframe['close_comparison'] + 4  # Maps to 0-8 (9 patterns)
        dataframe['pattern_mean'] = dataframe['pattern'].rolling(2).mean()
        dataframe['pattern_trend'] = dataframe['pattern'].rolling(9).mean()
        dataframe['pattern_trend_ma'] = dataframe['support'] + (dataframe['pattern_trend'] * dataframe['trading_range'])
        dataframe['pattern diff'] = dataframe['pattern'].diff()



        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Buy conditions based on Two Candle Theory
        # dataframe.loc[
        #     (
        #         # High close bull candle (pattern 4: most bullish)
        #         (dataframe['pattern'] == 4) &
        #         # Breakout above resistance
        #         (dataframe['close'] > dataframe['resistance'].shift(1)) &
        #         (dataframe['pattern_mean'] >= 4) &
        #         # Previous candle was not bearish (avoid false breakouts)
        #         (dataframe['pattern'].shift(1) != 0)  # Not low close bear
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'long 1')

        # dataframe.loc[
        #     (
        #         # High close bull candle (pattern 4: most bullish)
        #         (dataframe['pattern'] == 8) &
        #         # Breakout above resistance
        #         (dataframe['pattern_mean'] >= 4) &
        #         # Previous candle was not bearish (avoid false breakouts)
        #         (dataframe['pattern'].shift(1) == 0)  # Not low close bear
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'long 3')

        # dataframe.loc[
        #     (
        #         # High close bull candle (pattern 4: most bullish)
        #         (dataframe['pattern'] <= 7) &
        #         # Breakout above resistance
        #         (dataframe['pattern_mean'] >= 4) &
        #         # Previous candle was not bearish (avoid false breakouts)
        #         (dataframe['pattern'].shift(1) <= 0)  # Not low close bear
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'long 4')

        # dataframe.loc[
        #     (
        #         # High close bull candle (pattern 4: most bullish)
        #         (dataframe['pattern'] <= 6) &
        #         # Breakout above resistance
        #         (dataframe['pattern_mean'] >= 4) &
        #         # Previous candle was not bearish (avoid false breakouts)
        #         (dataframe['pattern'].shift(1) <= 0)  # Not low close bear
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'long 5')


        # Alternative buy: High close bull after support bounce
        dataframe.loc[
            (
                (dataframe['pattern_mean'].shift() < 4) &
                (dataframe['pattern_mean'].shift() < 4) &
                (dataframe['pattern_mean'] > 4) 
                # (dataframe['close'] > dataframe['support']) &
                # (dataframe['pattern'].shift(1) == 0)  # Previous was low close bear
            ),
            ['enter_long', 'enter_tag']] = (1, 'long 2')


        # Track bias (simplified as a rolling score)
        dataframe['bias'] = 0  # Neutral
        dataframe.loc[dataframe['pattern'] == 4, 'bias'] = 1  # Bullish on high close bull
        dataframe.loc[dataframe['pattern'] == 0, 'bias'] = -1  # Bearish on low close bear
        dataframe['bias'] = dataframe['bias'].rolling(window=3).mean().fillna(0)  # 3-candle bias

        # # Bullish entries
        # dataframe.loc[
        #     # High close bull at breakout
        #     ((dataframe['pattern'] == 4) & (dataframe['close'] > dataframe['resistance'].shift(1))) |
        #     # High close bull after support bounce
        #     ((dataframe['pattern'] == 4) & (dataframe['pattern'].shift(1) == 0) & (dataframe['close'] > dataframe['support'])) |
        #     # Two consecutive high close bull candles
        #     ((dataframe['pattern'] == 4) & (dataframe['pattern'].shift(1) == 4) & (dataframe['bias'] > 0)) |
        #     # High close range after pullback in uptrend
        #     ((dataframe['pattern'] == 1) & (dataframe['bias'] > 0) & (dataframe['close'] > dataframe['support'])),
        #     'enter_long'] = 1

        # # Bearish entries
        # dataframe.loc[
        #     # Low close bear at breakdown
        #     ((dataframe['pattern'] == 0) & (dataframe['close'] < dataframe['support'].shift(1))) |
        #     # Two consecutive low close bear candles
        #     ((dataframe['pattern'] == 0) & (dataframe['pattern'].shift(1) == 0) & (dataframe['bias'] < 0)) |
        #     # Low close bear after resistance rejection
        #     ((dataframe['pattern'] == 0) & (dataframe['close'] < dataframe['resistance']) & (dataframe['bias'] < 0)),
        #     'enter_short'] = 1


        # # Sell conditions (short entry)
        # dataframe.loc[
        #     (
        #         # Low close bear candle (pattern 0: most bearish)
        #         (dataframe['pattern'] == 0) &
        #         # Breakdown below support
        #         (dataframe['close'] < dataframe['support'].shift(1)) &
        #         # Previous candle was not bullish (avoid false breakdowns)
        #         (dataframe['pattern'].shift(1) != 4)  # Not high close bull
        #     ),
        #     'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long on strong bearish signal
        dataframe.loc[
            (
                # High close bull candle (pattern 4: most bullish)
                (dataframe['pattern'] >= 0) &
                # Breakout above resistance
                (dataframe['pattern_mean'] < 3.5) &
                # Previous candle was not bearish (avoid false breakouts)
                (dataframe['pattern'].shift(1) == 8)  # Not low close bear
            ),
            ['exit_long', 'exit_tag']] = (1, 'short1')

        dataframe.loc[
            (
                # High close bull candle (pattern 4: most bullish)
                (dataframe['pattern'] >= 0) &
                # Breakout above resistance3.5
                (dataframe['pattern_mean'] < 3.5) &
                # Previous candle was not bearish (avoid false breakouts)
                (dataframe['pattern'].shift(1) == 7)  # Not low close bear
            ),
            ['exit_long', 'exit_tag']] = (1, 'short2')

        dataframe.loc[
            (
                # High close bull candle (pattern 4: most bullish)
                (dataframe['pattern'] >= 0) &
                # Breakout above resistance
                (dataframe['pattern_mean'] < 3.5) &
                # Previous candle was not bearish (avoid false breakouts)
                (dataframe['pattern'].shift(1) == 6)  # Not low close bear
            ),
            ['exit_long', 'exit_tag']] = (1, 'short1')

        dataframe.loc[
            (
                (dataframe['pattern_mean'].shift() > 4) &
                (dataframe['pattern_mean'] < 4) 
            ),            
            ['exit_long', 'exit_tag']] = (1, 'exit 2')

        # # Bearish entries
        # dataframe.loc[
        #     # Low close bear at breakdown
        #     ((dataframe['pattern'] == 0) & (dataframe['close'] < dataframe['support'].shift(1))) |
        #     # Two consecutive low close bear candles
        #     ((dataframe['pattern'] == 0) & (dataframe['pattern'].shift(1) == 0) & (dataframe['bias'] < 0)) |
        #     # Low close bear after resistance rejection
        #     ((dataframe['pattern'] == 0) & (dataframe['close'] < dataframe['resistance']) & (dataframe['bias'] < 0)),
        #     'exit_long'] = 1

        # Exit short on strong bullish signal
        # dataframe.loc[
        #     (dataframe['enter_short'].shift(1) == 1) &
        #     (dataframe['pattern'] == 4),  # High close bull candle
        #     'exit_short'] = 1

        return dataframe