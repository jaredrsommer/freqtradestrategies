from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, informative
from pandas import DataFrame
import numpy
import talib.abstract as ta
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)

class Candle2(IStrategy):
    # Strategy parameters
    timeframe = "1h"  # 5-minute timeframe as per the video example
    minimal_roi = {}  # 1.5% ROI (1:1.5 risk-reward ratio)
    stoploss = -0.04  # 1% stop loss (adjustable)
    # Trailing stop:
    trailing_stop = True  # value loaded from strategy
    trailing_stop_positive = 0.025 # value loaded from strategy
    trailing_stop_positive_offset = 0.10 # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy
    buy_threshold = DecimalParameter(3.0, 8.0, default=4.0, decimals=1, space="buy")
    rsi_threshold = DecimalParameter(40.0, 70.0, default=65.0, decimals=1, space="buy")
    sell_threshold = DecimalParameter(3.0, 4.0, default=4.0, decimals=1, space="sell")
    sr_length = IntParameter(12, 50, default=24, space="buy")
    sr_shift = IntParameter(12, 50, default=24, space="buy")
    fast_rsi = IntParameter(4, 20, default=12, space="buy")

    @informative('4h')
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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
        dataframe['CO2'] = ((dataframe['close'] - dataframe['open']) / 2) + dataframe['open']


        # Combine close position and close comparison into 9 patterns
        dataframe['pattern'] = dataframe['close_position'] * 3 + dataframe['close_comparison'] + 4  # Maps to 0-8 (9 patterns)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=6)
        return dataframe

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
        dataframe['pattern_avg'] = ((dataframe['pattern'] + dataframe['pattern_4h']) / 2).rolling(2).mean()
        # Simple support/resistance levels using rolling min/max
        dataframe['support'] = dataframe['low_4h'].rolling(window=self.sr_length.value).min().shift(self.sr_shift.value)
        dataframe['resistance'] = dataframe['high_4h'].rolling(window=self.sr_length.value).max().shift(self.sr_shift.value)

        # 4h S/R 
        dataframe['range_4h'] = dataframe['resistance'] - dataframe['support']
        dataframe['inflection'] = (dataframe['range_4h']/2) + dataframe['support']
        dataframe['range_third_4h'] = dataframe['range_4h'] / 3

        # Close position: Determine if close is in upper, mid, or lower third
        dataframe['CO2_position'] = 0  # Default: mid
        dataframe.loc[dataframe['CO2_4h'] > (dataframe['resistance'] - dataframe['range_third_4h']), 'CO2_position'] = 1  # High close
        dataframe.loc[dataframe['CO2_4h'] < (dataframe['support'] + dataframe['range_third_4h']), 'CO2_position'] = -1  # Low close

        # Close comparison: Compare current close to previous candle's range
        dataframe['prev_high_4h'] = dataframe['resistance'].shift(1)
        dataframe['prev_low_4h'] = dataframe['support'].shift(1)
        dataframe['close_c'] = 0
        dataframe.loc[dataframe['close'] > dataframe['close_4h'].shift(4), 'close_c'] = 1  # Bull candle
        dataframe.loc[dataframe['close'] < dataframe['close_4h'].shift(4), 'close_c'] = -1  # Bear candle
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.fast_rsi.value)

        # # Combine close position and close comparison into 9 patterns
        dataframe['pattern_CO2'] = dataframe['CO2_position'] * 3 + dataframe['close_c'] + 4  # Maps to 0-8 (9 patterns)
        dataframe['bull_bear'] = 0 
        dataframe.loc[dataframe['CO2_4h'] < dataframe['close'], 'bull_bear'] = 1  # High close
        dataframe.loc[dataframe['CO2_4h'] > dataframe['close'], 'bull_bear'] = -1  # Low close

        # timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
        # pair = metadata['pair'].replace('/', '_')  # Replace '/' with '_' for valid filename
        # filename = f"{pair}_{timestamp}.csv"
        # dataframe.to_csv(filename, index=True)
        # logger.info(f"Exported DataFrame for {pair} to {filename}")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Buy conditions based on Two Candle Theory
        dataframe.loc[
            (
                # High close bull candle (pattern 4: most bullish)
                (dataframe['pattern_avg'] >= self.buy_threshold.value) &
                (dataframe['pattern_avg'].shift() < 8) &
                (dataframe['rsi_4h'] < self.rsi_threshold.value) &
                (dataframe['rsi'] > dataframe['rsi_4h'])
                # Breakout above resistance
                # (dataframe['close'] > dataframe['resistance'].shift(1)) &
                # Previous candle was not bearish (avoid false breakouts)
                # (dataframe['pattern'].shift(1) != 0)  # Not low close bear
            ),
            ['enter_long', 'enter_tag']] = (1, 'Pattern and RSI')

        dataframe.loc[
            (
                # High close bull candle (pattern 4: most bullish)
                (dataframe['pattern_avg'] >= self.buy_threshold.value) &
                (dataframe['pattern_avg'].shift() < 8) &
                (dataframe['CO2_4h'] > dataframe['resistance']) &
                (dataframe['CO2_4h'].shift() < dataframe['resistance'].shift()) & 
                (dataframe['rsi_4h'] < self.rsi_threshold.value) &
                (dataframe['rsi'] > dataframe['rsi_4h'])
                # Breakout above resistance
                # (dataframe['close'] > dataframe['resistance'].shift(1)) &
                # Previous candle was not bearish (avoid false breakouts)
                # (dataframe['pattern'].shift(1) != 0)  # Not low close bear
            ),
            ['enter_long', 'enter_tag']] = (1, 'Resistance Cross')

        dataframe.loc[
            (
                # High close bull candle (pattern 4: most bullish)
                (dataframe['pattern_avg'] >= self.buy_threshold.value) &
                # (dataframe['pattern_avg'].shift() < 8) &
                (dataframe['CO2_4h'] > dataframe['support']) &
                (dataframe['CO2_4h'].shift() < dataframe['support'].shift()) 
                # (dataframe['rsi_4h'] < self.rsi_threshold.value) 
                # (dataframe['rsi'] > dataframe['rsi_4h'])
                # Breakout above resistance
                # (dataframe['close'] > dataframe['resistance'].shift(1)) &
                # Previous candle was not bearish (avoid false breakouts)
                # (dataframe['pattern'].shift(1) != 0)  # Not low close bear
            ),
            ['enter_long', 'enter_tag']] = (1, 'Support Cross')

        dataframe.loc[
            (
                # High close bull candle (pattern 4: most bullish)
                (dataframe['pattern_avg'] >= self.buy_threshold.value) &
                # (dataframe['pattern_avg'].shift() < 8) &
                (dataframe['CO2_4h'] < dataframe['support']) &
                (dataframe['CO2_4h'].shift(4) < dataframe['CO2_4h']) &
                # (dataframe['rsi_4h'] < self.rsi_threshold.value) 
                (dataframe['rsi'] > dataframe['rsi_4h'])
                # Breakout above resistance
                # (dataframe['close'] > dataframe['resistance'].shift(1)) &
                # Previous candle was not bearish (avoid false breakouts)
                # (dataframe['pattern'].shift(1) != 0)  # Not low close bear
            ),
            ['enter_long', 'enter_tag']] = (1, 'Below Support 4h pull up')


        dataframe.loc[
            (
                # High close bull candle (pattern 4: most bullish)
                # (dataframe['pattern_avg'] >= self.buy_threshold.value) &
                # (dataframe['pattern_avg'].shift() < 8) &
                (dataframe['CO2_4h'] < dataframe['support']) &
                (dataframe['rsi_4h'] < 10) &
                (dataframe['rsi'] > dataframe['rsi_4h'])
                # Breakout above resistance
                # (dataframe['close'] > dataframe['resistance'].shift(1)) &
                # Previous candle was not bearish (avoid false breakouts)
                # (dataframe['pattern'].shift(1) != 0)  # Not low close bear
            ),
            ['enter_long', 'enter_tag']] = (1, '4h Extreme RSI')

        # # Alternative buy: High close bull after support bounce
        # dataframe.loc[
        #     (
        #         (dataframe['pattern'] == 4) &
        #         (dataframe['close'] > dataframe['support']) &
        #         (dataframe['pattern'].shift(1) == 0)  # Previous was low close bear
        #     ),
        #     'enter_long'] = 1

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
        # dataframe.loc[
        #     # (dataframe['enter_long'].shift(1) == 1) &
        #     (dataframe['pattern_avg'] <= self.sell_threshold.value),  # Low close bear candle
        #     'exit_long'] = 1

        # # Exit short on strong bullish signal
        # dataframe.loc[
        #     (dataframe['enter_short'].shift(1) == 1) &
        #     (dataframe['pattern'] == 4),  # High close bull candle
        #     'exit_short'] = 1

        dataframe.loc[
            (
                # High close bull candle (pattern 4: most bullish)
                # (dataframe['pattern_avg'] >= self.buy_threshold.value) &
                # (dataframe['pattern_avg'].shift() < 8) &
                (dataframe['CO2_4h'] > dataframe['resistance']) &
                (dataframe['close'] > dataframe['resistance']) &
                (dataframe['close'] < dataframe['CO2_4h']) &
                (dataframe['rsi_4h'] > 85) 
                # (dataframe['rsi'] > dataframe['rsi_4h'])
                # Breakout above resistance
                # (dataframe['close'] > dataframe['resistance'].shift(1)) &
                # Previous candle was not bearish (avoid false breakouts)
                # (dataframe['pattern'].shift(1) != 0)  # Not low close bear
            ),
            ['exit_long', 'exit_tag']] = (1, 'Above Resistance 4h')

        # dataframe.loc[
        #     (
        #         # High close bull candle (pattern 4: most bullish)
        #         # (dataframe['pattern_avg'] >= self.buy_threshold.value) &
        #         # (dataframe['pattern_avg'].shift() < 8) &
        #         (dataframe['CO2_4h'] < dataframe['resistance']) &
        #         (dataframe['CO2_4h'].shift() > dataframe['resistance'].shift()) 
        #         # (dataframe['rsi'] > dataframe['rsi_4h'])
        #         # Breakout above resistance
        #         # (dataframe['close'] > dataframe['resistance'].shift(1)) &
        #         # Previous candle was not bearish (avoid false breakouts)
        #         # (dataframe['pattern'].shift(1) != 0)  # Not low close bear
        #     ),
        #     ['exit_long', 'exit_tag']] = (1, 'Resistance Cross')


        # dataframe.loc[
        #     (
        #         # High close bull candle (pattern 4: most bullish)
        #         # (dataframe['pattern_avg'] >= self.buy_threshold.value) &
        #         # (dataframe['pattern_avg'].shift() < 8) &
        #         (dataframe['CO2_4h'] < dataframe['support']) &
        #         (dataframe['CO2_4h'].shift() > dataframe['support'].shift()) 
        #         # (dataframe['rsi'] > dataframe['rsi_4h'])
        #         # Breakout above resistance
        #         # (dataframe['close'] > dataframe['resistance'].shift(1)) &
        #         # Previous candle was not bearish (avoid false breakouts)
        #         # (dataframe['pattern'].shift(1) != 0)  # Not low close bear
        #     ),
        #     ['exit_long', 'exit_tag']] = (1, 'Below Support 4h cross')

        return dataframe