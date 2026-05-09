from freqtrade.strategy import IStrategy
import pandas as pd
import numpy as np
import talib
from scipy.fft import fft
from typing import Dict, List
import logging
import numpy as np
import pandas as pd
from technical import qtpylib
from pandas import DataFrame, Series
from datetime import datetime, timezone
from typing import Optional
from functools import reduce
import talib.abstract as ta
import talib
import pandas_ta as pta
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
import warnings
from pandas.errors import PerformanceWarning
from scipy.signal import savgol_filter
import math

class HurstCycle3(IStrategy):
    timeframe = '4h'
    minimal_roi = {"0": 0.05, "60": 0.03, "120": 0.01}
    stoploss = -0.04
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.02

    # Hyperparameters to optimize
    base_cycle_period = 20
    envelope_factor_short = 1.2
    envelope_factor_long = 2.0
    filter_weights = [1, 2, 3, 2, 1]  # Symmetric weights for smoothing (p. 177)
    convergence_threshold = DecimalParameter(0.002, 0.01, default=0.005, space='buy', optimize=True)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Spectral Analysis
        heikinashi = qtpylib.heikinashi(dataframe)

        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        close = dataframe['ha_close'].values
        n = min(len(close), 500)
        yf = fft(close[-n:])
        freq = np.fft.fftfreq(n)
        power = np.abs(yf) ** 2
        dominant_idx = np.argmax(power[1:n//2]) + 1
        dominant_period = int(1 / freq[dominant_idx]) if freq[dominant_idx] != 0 else self.base_cycle_period
        self.short_cycle = max(10, min(30, dominant_period))
        self.long_cycle = self.short_cycle * 2

        # Numerical Filter
        weights = np.array(self.filter_weights) / sum(self.filter_weights)
        filtered = np.convolve(close, weights, mode='valid')
        dataframe['filtered_close'] = pd.Series(filtered, index=dataframe.index[-len(filtered):])

        # FLDs (for convergence check)
        half_short = math.ceil(self.short_cycle / 4)
        half_mid = math.ceil(self.short_cycle / 2)
        half_long = math.ceil(self.long_cycle / 2)
        dataframe['filtered_close_smooth'] = dataframe['filtered_close'].rolling(window=2).mean()
        fld_short_base = dataframe['filtered_close'].rolling(window=int(self.short_cycle/2), center=True).mean()
        fld_mid_base = dataframe['filtered_close'].rolling(window=self.short_cycle, center=True).mean()
        fld_long_base = dataframe['filtered_close'].rolling(window=self.long_cycle, center=True).mean()
        dataframe['fld_short'] = fld_short_base.shift(half_short)
        dataframe['fld_mid'] = fld_mid_base.shift(half_mid)
        dataframe['fld_long'] = fld_long_base.shift(half_long)
        for fld in ['fld_short', 'fld_mid', 'fld_long']:
            last_valid_idx = dataframe[fld].last_valid_index()
            if last_valid_idx is not None:
                last_valid_row = dataframe.index.get_loc(last_valid_idx)
                if last_valid_row < len(dataframe) - 1 and last_valid_row > 0:
                    for i in range(last_valid_row + 1, len(dataframe)):
                        prev_slope = (dataframe[fld].iloc[last_valid_row] - 
                                      dataframe[fld].iloc[last_valid_row - 1])
                        dataframe[fld].iloc[i] = dataframe[fld].iloc[last_valid_row] + \
                                                 prev_slope * (i - last_valid_row)

        # Convergence-Based Agreeance Band
        fld_spread = dataframe[['fld_short', 'fld_mid', 'fld_long']].max(axis=1) - \
                     dataframe[['fld_short', 'fld_mid', 'fld_long']].min(axis=1)
        relative_spread = fld_spread / dataframe['filtered_close']
        dataframe['converging'] = relative_spread < self.convergence_threshold.value
        band_center = dataframe[['fld_short', 'fld_mid', 'fld_long']].mean(axis=1)
        band_width = dataframe['filtered_close'] * 0.01
        dataframe['agreeance_upper'] = np.where(dataframe['converging'],
                                               band_center + band_width,
                                               np.nan)
        dataframe['agreeance_lower'] = np.where(dataframe['converging'],
                                               band_center - band_width,
                                               np.nan)


        # [Rest of indicators: envelopes, VTLs, trend...]
        dataframe['sma_short'] = talib.SMA(dataframe['filtered_close'], timeperiod=self.short_cycle)
        dataframe['sma_long'] = talib.SMA(dataframe['filtered_close'], timeperiod=self.long_cycle)
        dataframe['atr_short'] = talib.ATR(dataframe['high'], dataframe['low'], dataframe['filtered_close'], timeperiod=self.short_cycle)
        dataframe['atr_long'] = talib.ATR(dataframe['high'], dataframe['low'], dataframe['filtered_close'], timeperiod=self.long_cycle)
        dataframe['upper_env_short'] = dataframe['sma_short'] + self.envelope_factor_short * dataframe['atr_short']
        dataframe['lower_env_short'] = dataframe['sma_short'] - self.envelope_factor_short * dataframe['atr_short']
        dataframe['upper_env_long'] = dataframe['sma_long'] + self.envelope_factor_long * dataframe['atr_long']
        dataframe['lower_env_long'] = dataframe['sma_long'] - self.envelope_factor_long * dataframe['atr_long']
        dataframe['trough'] = dataframe['filtered_close'].rolling(self.long_cycle).min()
        dataframe['crest'] = dataframe['filtered_close'].rolling(self.long_cycle).max()
        dataframe['is_trough'] = np.where(dataframe['filtered_close'] == dataframe['trough'], 1, 0)
        dataframe['is_crest'] = np.where(dataframe['filtered_close'] == dataframe['crest'], 1, 0)
        troughs = dataframe[dataframe['is_trough'] == 1].index
        crests = dataframe[dataframe['is_crest'] == 1].index
        if len(troughs) >= 2:
            t1, t2 = troughs[-2], troughs[-1]
            dataframe['vtl_up_slope'] = (dataframe['filtered_close'][t2] - dataframe['filtered_close'][t1]) / (t2 - t1)
        if len(crests) >= 2:
            c1, c2 = crests[-2], crests[-1]
            dataframe['vtl_down_slope'] = (dataframe['filtered_close'][c2] - dataframe['filtered_close'][c1]) / (c2 - c1)
        dataframe['trend'] = np.where(dataframe['filtered_close'] > dataframe['sma_long'], 1, -1)

        # Calculate distances from filtered_close to each line
        dataframe['dist_upper_short'] = dataframe['upper_env_short'] - dataframe['filtered_close']
        dataframe['dist_lower_short'] = dataframe['filtered_close'] - dataframe['lower_env_short']
        dataframe['dist_upper_long'] = dataframe['upper_env_long'] - dataframe['filtered_close']
        dataframe['dist_lower_long'] = dataframe['filtered_close'] - dataframe['lower_env_long']
        dataframe['dist_fld_short'] = abs(dataframe['filtered_close'] - dataframe['fld_short'])
        dataframe['dist_fld_mid'] = abs(dataframe['filtered_close'] - dataframe['fld_mid'])
        dataframe['dist_fld_long'] = abs(dataframe['filtered_close'] - dataframe['fld_long'])

        # Normalize distances as percentage of filtered_close
        for col in ['dist_upper_short', 'dist_lower_short', 'dist_upper_long', 
                    'dist_lower_long', 'dist_fld_short', 'dist_fld_mid', 'dist_fld_long']:
            dataframe[f'{col}_norm'] = dataframe[col] / dataframe['filtered_close']

        # Calculate spread metrics for nesting detection
        envelope_spread_short = dataframe['upper_env_short'] - dataframe['lower_env_short']
        envelope_spread_long = dataframe['upper_env_long'] - dataframe['lower_env_long']
        fld_spread = dataframe[['fld_short', 'fld_mid', 'fld_long']].max(axis=1) - \
                     dataframe[['fld_short', 'fld_mid', 'fld_long']].min(axis=1)
        
        # Normalized spreads
        dataframe['env_spread_short_norm'] = envelope_spread_short / dataframe['filtered_close']
        dataframe['env_spread_long_norm'] = envelope_spread_long / dataframe['filtered_close']
        dataframe['fld_spread_norm'] = fld_spread / dataframe['filtered_close']

        # Nesting detection
        # We'll consider lines "nesting" when:
        # 1. All distances are within a small threshold
        # 2. Spreads between lines are compressing
        nesting_threshold = 0.01  # 1% of price, adjustable
        dataframe['is_nesting'] = (
            (dataframe['dist_upper_short_norm'] < nesting_threshold) &
            (dataframe['dist_lower_short_norm'] < nesting_threshold) &
            (dataframe['dist_upper_long_norm'] < nesting_threshold) &
            (dataframe['dist_lower_long_norm'] < nesting_threshold) &
            (dataframe['dist_fld_short_norm'] < nesting_threshold) &
            (dataframe['dist_fld_mid_norm'] < nesting_threshold) &
            (dataframe['dist_fld_long_norm'] < nesting_threshold)
        )

        # Nesting trend (are lines coming together?)
        lookback = 5  # Lookback period to detect convergence
        dataframe['nesting_score'] = 0.0
        for col in ['env_spread_short_norm', 'env_spread_long_norm', 'fld_spread_norm']:
            spread_change = dataframe[col] - dataframe[col].shift(lookback)
            # Negative change means spreads are compressing
            dataframe['nesting_score'] += np.where(spread_change < 0, 1, 0)
        # Normalize score (0-3, where 3 means all spreads are compressing)
        dataframe['nesting_score'] = dataframe['nesting_score'] / 3.0

        # Additional metric: average normalized distance to all lines
        distance_cols = [col for col in dataframe.columns if col.startswith('dist_') and col.endswith('_norm')]
        dataframe['avg_line_distance'] = dataframe[distance_cols].mean(axis=1)
        dataframe['avg_dist_mean'] = ta.SMA(dataframe['avg_line_distance'], 200)

        return dataframe
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Long Entry: Trough, below FLD, short envelope breach, VTL breakout
        # dataframe.loc[
        #     # (dataframe['is_trough'] == 1) &
        #     (dataframe['filtered_close'] > dataframe['filtered_close_smooth']) &
        #     (dataframe['filtered_close'].shift() < dataframe['filtered_close_smooth'].shift()),
        #     # (dataframe['trend'] == 1),
        #     'enter_long'] = 1

        dataframe.loc[
            # (dataframe['is_trough'] == 1) &
            (dataframe['ha_close'] > dataframe['filtered_close']) &
            (dataframe['ha_close'].shift() < dataframe['filtered_close'].shift()) &
            (dataframe['ha_close'] > dataframe['ha_open']),
    
            # (dataframe['trend'] == 1),
            'enter_long'] = 1

        # # Short Entry: Crest, above FLD, short envelope breach, VTL breakout
        # dataframe.loc[
        #     (dataframe['is_crest'] == 1) &
        #     (dataframe['filtered_close'] > dataframe['fld_short']) &
        #     (dataframe['filtered_close'] > dataframe['upper_env_short']) &
        #     (dataframe['trend'] == -1) &
        #     (dataframe['vtl_down_slope'].notna() & (dataframe['vtl_down_slope'] < 0)),
        #     'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Exit Long: Hit FLD or upper long envelope
        # dataframe.loc[
        #     (dataframe['filtered_close'] < dataframe['filtered_close_smooth']) &
        #     (dataframe['filtered_close'].shift() > dataframe['filtered_close_smooth'].shift()),
        #     'exit_long'] = 1

        dataframe.loc[
            (dataframe['ha_open'] > dataframe['filtered_close']) &
            (dataframe['ha_open'].shift() < dataframe['filtered_close'].shift()) &
            (dataframe['ha_close'] < dataframe['ha_open']),
            'exit_long'] = 1

        # # Exit Short: Hit FLD or lower long envelope
        # dataframe.loc[
        #     (dataframe['enter_short'].shift(1) == 1) &
        #     ((dataframe['filtered_close'] <= dataframe['fld_long']) | 
        #      (dataframe['filtered_close'] <= dataframe['lower_env_long'])),
        #     'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time, current_rate: float, proposed_leverage: float, 
                 max_leverage: float, side: str, **kwargs) -> float:
        return 3.0
