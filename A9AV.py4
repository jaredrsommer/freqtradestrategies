from freqtrade.strategy import IStrategy
from freqtrade.strategy.interface import ITrend
from pandas import DataFrame
from datetime import datetime, timedelta

class A9AV(IStrategy):
    # Strategy interface version
    INTERFACE_VERSION = 3

    # Minimal candle length for strategy
    MIN_CANDLE_LENGTH = 1

    # Set the plot configuration
    plot_config = {
        'main_plot': {
            ' SMA_9': {'color': 'blue'},
            'current_volume': {'color': 'green'}
        },
        'subplots': [
            {"SMA_9": {'color': 'blue'}},
            {"current_volume": {'color': 'green'}}
        ]
    }

    # Define the set of parameters that will be used in the strategy
    timeframe = IntParameter(1, 15, default=1, space='space', optimize=False, load=True)
    source = StringParameter('close', 'open', 'high', 'low', 'hl2', 'hlc3', 'hlcc4', default='close', space='space', optimize=False, load=True)

    # Define the set of parameters that will be used in the strategy for the 9-period average
    length = IntParameter(5, 15, default=9, space='space', optimize=False, load=True)

    # Define the set of parameters for the opposing signal filter
    opposing_signal_filter = IntParameter(1, 5, default=2, space='space', optimize=False, load=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate the 9-period average of the volume
        dataframe['SMA_9'] = dataframe['volume'].rolling(window=self.length.value).mean()

        # Create columns to track buy and sell signals
        dataframe['buy_signal'] = 0
        dataframe['sell_signal'] = 0

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Buy signal when the current volume is above the 9-period average
        # and the previous candle's close is higher than the current candle's close
        # and there's no opposing sell signal in the last `opposing_signal_filter` candles
        dataframe.loc[
            (dataframe['volume'] > dataframe['SMA_9']) &
            (dataframe[self.source.value].shift(1) < dataframe[self.source.value]) &
            (~dataframe['sell_signal'].rolling(window=self.opposing_signal_filter.value).any()),
            'buy_signal'
        ] = 1

        # Set buy signal in the `buy` column
        dataframe.loc[dataframe['buy_signal'] == 1, 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Sell signal when the current volume is above the 9-period average
        # and the previous candle's close is lower than the current candle's close
        # and there's no opposing buy signal in the last `opposing_signal_filter` candles
        dataframe.loc[
            (dataframe['volume'] > dataframe['SMA_9']) &
            (dataframe[self.source.value].shift(1) > dataframe[self.source.value]) &
            (~dataframe['buy_signal'].rolling(window=self.opposing_signal_filter.value).any()),
            'sell_signal'
        ] = 1

        # Set sell signal in the `sell` column
        dataframe.loc[dataframe['sell_signal'] == 1, 'sell'] = 1

        return dataframe