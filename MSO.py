from freqtrade.strategy import (BooleanParameter, CategoricalParameter, 
                               DecimalParameter, IStrategy, IntParameter, 
                               RealParameter, merge_informative_pair, informative)
import pandas as pd
import numpy as np
import talib.abstract as ta
from pandas import DataFrame
import logging

class MSO(IStrategy):
    # Strategy interface version
    INTERFACE_VERSION = 3

    # Minimal ROI
    minimal_roi = {
        "0": 0.05,
        "30": 0.03,
        "60": 0.02,
        "120": 0
    }

    # Stop loss
    stoploss = -0.1

    # Trailing stop
    trailing_stop = False

    # Timeframe
    timeframe = '1h'

    # Hyperopt parameters
    ms_weight_k1 = DecimalParameter(0.0, 5.0, default=1.0, space='buy')
    ms_weight_k2 = DecimalParameter(0.0, 5.0, default=3.0, space='buy')
    ms_weight_k3 = DecimalParameter(0.0, 5.0, default=5.0, space='buy')
    norm_smooth = IntParameter(1, 10, default=3, space='buy')
    cyc_smooth = IntParameter(1, 14, default=9, space='buy')
    buy_threshold = DecimalParameter(50.0, 85.0, default=60.0, space='buy')
    sell_threshold = DecimalParameter(15.0, 50.0, default=40.0, space='sell')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Initialize swing structures
        st_high = {'last_price': 0, 'mid_price': 0, 'prev_price': 0, 'is_crossed': False}
        st_low = {'last_price': 0, 'mid_price': 0, 'prev_price': 0, 'is_crossed': False}
        it_high = {'last_price': 0, 'mid_price': 0, 'prev_price': 0, 'is_crossed': False}
        it_low = {'last_price': 0, 'mid_price': 0, 'prev_price': 0, 'is_crossed': False}
        lt_high = {'last_price': 0, 'mid_price': 0, 'prev_price': 0, 'is_crossed': False}
        lt_low = {'last_price': 0, 'mid_price': 0, 'prev_price': 0, 'is_crossed': False}

        # Short-term market structure
        def short_market_structure(df, high, low, close):
            bull = np.zeros(len(df))
            bear = np.zeros(len(df))
            osc = np.zeros(len(df))
            max_vals = np.zeros(len(df))
            min_vals = np.zeros(len(df))

            for i in range(2, len(df)):
                # High swing detection
                if high[i-2] < high[i-1] and high[i-1] >= high[i]:
                    st_high['prev_price'] = st_high['mid_price']
                    st_high['mid_price'] = st_high['last_price']
                    st_high['last_price'] = high[i-1]
                    st_high['is_crossed'] = False

                if close[i] > st_high['last_price'] and not st_high['is_crossed']:
                    st_high['is_crossed'] = True
                    bull[i] = 1

                # Low swing detection
                if low[i-2] > low[i-1] and low[i-1] <= low[i]:
                    st_low['prev_price'] = st_low['mid_price']
                    st_low['mid_price'] = st_low['last_price']
                    st_low['last_price'] = low[i-1]
                    st_low['is_crossed'] = False

                if close[i] < st_low['last_price'] and not st_low['is_crossed']:
                    st_low['is_crossed'] = True
                    bear[i] = 1

                # Normalize oscillator
                os = 1 if bull[i] else -1 if bear[i] else osc[i-1] if i > 0 else 0
                max_vals[i] = close[i] if os > (osc[i-1] if i > 0 else 0) else max(close[i], max_vals[i-1] if i > 0 else close[i])
                min_vals[i] = close[i] if os < (osc[i-1] if i > 0 else 0) else min(close[i], min_vals[i-1] if i > 0 else close[i])
                
                if max_vals[i] != min_vals[i] and i >= self.norm_smooth.value:
                    series = np.array([(close[j] - min_vals[j]) / (max_vals[j] - min_vals[j]) * 100 for j in range(i - self.norm_smooth.value + 1, i + 1)])
                    sma_result = ta.SMA(series, timeperiod=self.norm_smooth.value)
                    osc[i] = sma_result[-1] if sma_result is not None and not np.isnan(sma_result[-1]) else osc[i-1] if i > 0 else 50.0
                else:
                    osc[i] = osc[i-1] if i > 0 else 50.0

            return osc

        # General market structure
        def market_structure(df, h_swing_high, h_swing_low, l_swing_high, l_swing_low, close):
            bull = np.zeros(len(df))
            bear = np.zeros(len(df))
            osc = np.zeros(len(df))
            max_vals = np.zeros(len(df))
            min_vals = np.zeros(len(df))

            for i in range(2, len(df)):
                # High swing detection
                c_swing_high = l_swing_high['prev_price'] < l_swing_high['mid_price'] and l_swing_high['mid_price'] >= l_swing_high['last_price']
                if c_swing_high:
                    h_swing_high['prev_price'] = h_swing_high['mid_price']
                    h_swing_high['mid_price'] = h_swing_high['last_price']
                    h_swing_high['last_price'] = l_swing_high['mid_price']
                    h_swing_high['is_crossed'] = False

                if close[i] > h_swing_high['last_price'] and not h_swing_high['is_crossed']:
                    h_swing_high['is_crossed'] = True
                    bull[i] = 1

                # Low swing detection
                c_swing_low = l_swing_low['prev_price'] > l_swing_low['mid_price'] and l_swing_low['mid_price'] <= l_swing_low['last_price']
                if c_swing_low:
                    h_swing_low['prev_price'] = h_swing_low['mid_price']
                    h_swing_low['mid_price'] = h_swing_low['last_price']
                    h_swing_low['last_price'] = l_swing_low['mid_price']
                    h_swing_low['is_crossed'] = False

                if close[i] < h_swing_low['last_price'] and not h_swing_low['is_crossed']:
                    h_swing_low['is_crossed'] = True
                    bear[i] = 1

                # Normalize oscillator
                os = 1 if bull[i] else -1 if bear[i] else osc[i-1] if i > 0 else 0
                max_vals[i] = close[i] if os > (osc[i-1] if i > 0 else 0) else max(close[i], max_vals[i-1] if i > 0 else close[i])
                min_vals[i] = close[i] if os < (osc[i-1] if i > 0 else 0) else min(close[i], min_vals[i-1] if i > 0 else close[i])
                
                if max_vals[i] != min_vals[i] and i >= self.norm_smooth.value:
                    series = np.array([(close[j] - min_vals[j]) / (max_vals[j] - min_vals[j]) * 100 for j in range(i - self.norm_smooth.value + 1, i + 1)])
                    sma_result = ta.SMA(series, timeperiod=self.norm_smooth.value)
                    osc[i] = sma_result[-1] if sma_result is not None and not np.isnan(sma_result[-1]) else osc[i-1] if i > 0 else 50.0
                else:
                    osc[i] = osc[i-1] if i > 0 else 50.0

            return osc

        # Calculate oscillators
        dataframe['st_osc'] = short_market_structure(dataframe, dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['it_osc'] = market_structure(dataframe, it_high, it_low, st_high, st_low, dataframe['close'])
        dataframe['lt_osc'] = market_structure(dataframe, lt_high, lt_low, it_high, it_low, dataframe['close'])

        # Market Structure Oscillator
        weights_sum = self.ms_weight_k1.value + self.ms_weight_k2.value + self.ms_weight_k3.value
        weights_count = (1 if not np.isnan(dataframe['st_osc']).any() else 0) + \
                        (1 if not np.isnan(dataframe['it_osc']).any() else 0) + \
                        (1 if not np.isnan(dataframe['lt_osc']).any() else 0)
        dataframe['ms_osc'] = (self.ms_weight_k1.value * dataframe['st_osc'].fillna(0) +
                              self.ms_weight_k2.value * dataframe['it_osc'].fillna(0) +
                              self.ms_weight_k3.value * dataframe['lt_osc'].fillna(0)) / (weights_sum if weights_sum > 0 else 1)

        dataframe['ms_osc_smo'] = dataframe['ms_osc'].rolling(self.cyc_smooth.value).mean()
        # Cycle Oscillator
        dataframe['cycle_osc'] = (dataframe['ms_osc'] - dataframe['ms_osc_smo']) + 50

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['ms_osc'] > self.buy_threshold.value) &
                (dataframe['cycle_osc'] > 50)
            ),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['ms_osc'] < self.sell_threshold.value) &
                (dataframe['cycle_osc'] < 50)
            ),
            'exit_long'] = 1
        return dataframe