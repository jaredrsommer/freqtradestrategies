import logging
import numpy as np
import pandas_ta as pta
import math
import pywt
import warnings
import numpy as np
import scipy as sp
import os
import time
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce
import datetime
from datetime import timedelta, datetime
import talib.abstract as ta
from pandas import DataFrame, Series
from technical import qtpylib
from typing import Optional
import pandas as pd
from freqtrade.strategy.interface import IStrategy
from technical.pivots_points import pivots_points
from freqtrade.strategy import (CategoricalParameter, informative, IStrategy, merge_informative_pair,
                                DecimalParameter, IntParameter, BooleanParameter, timeframe_to_minutes)

from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_minutes
from freqtrade.persistence import Trade
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, butter, filtfilt
from technical import qtpylib
from datetime import timedelta, datetime, timezone
from pandas import DataFrame, Series
from technical import qtpylib
from typing import List, Tuple, Optional
from technical.pivots_points import pivots_points
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_minutes
from freqtrade.persistence import Trade

pd.options.mode.chained_assignment = None
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import IStrategy, merge_informative_pair

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
    timeframe = '5m'
    stoploss = -0.20
    # source = CategoricalParameter('close', 'open', 'high', 'low', 'hl2', 'hlc3', 'hlcc4', default='close', space='space', optimize=False, load=True)

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
            (dataframe['close'].shift(1) < dataframe['close']) &
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
            (dataframe['close'].shift(1) > dataframe['close']) &
            (~dataframe['buy_signal'].rolling(window=self.opposing_signal_filter.value).any()),
            'sell_signal'
        ] = 1

        # Set sell signal in the `sell` column
        dataframe.loc[dataframe['sell_signal'] == 1, 'sell'] = 1

        return dataframe