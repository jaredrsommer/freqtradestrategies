# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import numpy as np
import pandas as pd
import talib.abstract as ta
from functools import reduce
from freqtrade.strategy import DecimalParameter, IntParameter, BooleanParameter, CategoricalParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib


class FVGChannel(IStrategy):
    """
    FVG Channel Strategy based on LuxAlgo's indicator
    Adjusted to maintain Fibonacci spacing within extremes
    """
    INTERFACE_VERSION = 3

    minimal_roi = {"0": 0.05, "30": 0.025, "60": 0.015, "120": 0.01}
    stoploss = -0.1
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
    timeframe = '1h'

    fvg_len = IntParameter(1, 50, default=12, space="buy", optimize=False)  # Match TradingView lookback=12
    smooth_len = IntParameter(1, 50, default=9, space="buy", optimize=False)  # Match TradingView smoothing=9
    correction_factor = DecimalParameter(low=0.99, high=1.01, default=1.00, decimals=2, space="buy", optimize=False)

    plot_config = {
        'main_plot': {
            'upper_extreme': {'color': '#089981'},
            'upper_inner': {'color': 'rgba(8, 153, 129, 0.5)'},
            'mid_point': {'color': 'gray'},
            'lower_inner': {'color': 'rgba(242, 54, 69, 0.5)'},
            'lower_extreme': {'color': '#f23645'}
        },
        'subplots': {
            "signals": {
                'up_signal': {'color': '#089981'},
                'down_signal': {'color': '#f23645'}
            }
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        filter_weights = [1, 2, 4, 8, 4]
        pair = metadata['pair']
        heikinashi = qtpylib.heikinashi(dataframe)

        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']
        # Numerical Filter
        close = dataframe['ha_close'].values
        weights = np.array(filter_weights) / sum(filter_weights)
        filtered = np.convolve(close, weights, mode='valid')
        dataframe['filtered_close'] = pd.Series(filtered, index=dataframe.index[-len(filtered):])
        dataframe['filter_sma'] = ta.SMA(dataframe['filtered_close'], 9)
        dataframe['filter_sma'] = ta.SMA(dataframe['filtered_close'], 21)

        for i in range(3):
            dataframe[f'high_{i}'] = dataframe['high'].shift(i)
            dataframe[f'low_{i}'] = dataframe['low'].shift(i)
            dataframe[f'close_{i}'] = dataframe['close'].shift(i)
            dataframe[f'open_{i}'] = dataframe['open'].shift(i)

        dataframe = self.calculate_fvg_levels(dataframe)
        dataframe = self.calculate_fvg_channel(dataframe)
        dataframe = self.generate_signals(dataframe)
        return dataframe

    def calculate_fvg_levels(self, dataframe: DataFrame) -> DataFrame:
        """
        Calculate FVGs and maintain a rolling 'array' of unmitigated levels
        """
        dataframe['bull_fvg'] = np.nan
        dataframe['bear_fvg'] = np.nan
        
        bull_fvg_mask = (dataframe['low'] > dataframe['high_2']) & (dataframe['close_1'] > dataframe['high_2'])
        dataframe.loc[bull_fvg_mask, 'bull_fvg'] = dataframe['high_2']
        
        bear_fvg_mask = (dataframe['high'] < dataframe['low_2']) & (dataframe['close_1'] < dataframe['low_2'])
        dataframe.loc[bear_fvg_mask, 'bear_fvg'] = dataframe['low_2']
        
        bull_fvgs = []
        bear_fvgs = []
        dataframe['bull_lvls_avg'] = np.nan
        dataframe['bear_lvls_avg'] = np.nan
        
        for i in range(len(dataframe)):
            if not np.isnan(dataframe['bull_fvg'].iloc[i]):
                bull_fvgs.append(dataframe['bull_fvg'].iloc[i])
            if not np.isnan(dataframe['bear_fvg'].iloc[i]):
                bear_fvgs.append(dataframe['bear_fvg'].iloc[i])
            
            if bull_fvgs and dataframe['close'].iloc[i] < max(bull_fvgs):
                bull_fvgs = [fvg for fvg in bull_fvgs if dataframe['close'].iloc[i] >= fvg]
            if bear_fvgs and dataframe['close'].iloc[i] > min(bear_fvgs):
                bear_fvgs = [fvg for fvg in bear_fvgs if dataframe['close'].iloc[i] <= fvg]
            
            if len(bull_fvgs) > self.fvg_len.value:
                bull_fvgs = bull_fvgs[-self.fvg_len.value:]
            if len(bear_fvgs) > self.fvg_len.value:
                bear_fvgs = bear_fvgs[-self.fvg_len.value:]
            
            dataframe.at[i, 'bull_lvls_avg'] = np.mean(bull_fvgs) if bull_fvgs else np.nan
            dataframe.at[i, 'bear_lvls_avg'] = np.mean(bear_fvgs) if bear_fvgs else np.nan
        
        return dataframe

    def calculate_fvg_channel(self, dataframe: DataFrame) -> DataFrame:
        """
        Calculate channel with correct spacing between extremes
        """
        # Bars since last FVG
        dataframe['bull_bs'] = dataframe['bull_lvls_avg'].notna().cumsum().where(dataframe['bull_lvls_avg'].isna(), 0).shift(1).fillna(0) + 1
        dataframe['bear_bs'] = dataframe['bear_lvls_avg'].notna().cumsum().where(dataframe['bear_lvls_avg'].isna(), 0).shift(1).fillna(0) + 1
        
        # Progressive SMAs
        dataframe['bull_sma'] = np.nan
        dataframe['bear_sma'] = np.nan
        
        for i in range(len(dataframe)):
            bull_window = min(int(dataframe['bull_bs'].iloc[i]) if not np.isnan(dataframe['bull_bs'].iloc[i]) else 1, 
                             self.smooth_len.value)
            bear_window = min(int(dataframe['bear_bs'].iloc[i]) if not np.isnan(dataframe['bear_bs'].iloc[i]) else 1, 
                             self.smooth_len.value)
            
            if i >= bull_window - 1:
                dataframe.at[i, 'bull_sma'] = dataframe['close'].iloc[i-bull_window+1:i+1].mean()
            else:
                dataframe.at[i, 'bull_sma'] = dataframe['close'].iloc[:i+1].mean()
                
            if i >= bear_window - 1:
                dataframe.at[i, 'bear_sma'] = dataframe['close'].iloc[i-bear_window+1:i+1].mean()
            else:
                dataframe.at[i, 'bear_sma'] = dataframe['close'].iloc[:i+1].mean()
        
        # Smooth FVGs or fallback to SMA
        dataframe['bull_disp'] = ta.SMA(dataframe['bull_lvls_avg'].fillna(dataframe['bull_sma']), timeperiod=self.smooth_len.value)
        dataframe['bear_disp'] = ta.SMA(dataframe['bear_lvls_avg'].fillna(dataframe['bear_sma']), timeperiod=self.smooth_len.value)
        
        # Ensure bear_disp is above bull_disp with minimum range
        range_fallback = (dataframe['high'].rolling(self.fvg_len.value * 1).max() - 
                         dataframe['low'].rolling(self.fvg_len.value * 1).min()) * 0.8
        min_range = dataframe['close'] * 0.2367  # Target fvg_rng ≈ 23.67% for 6.77% mid-to-upper gap
        dataframe['bear_disp'] = np.where(dataframe['bear_disp'] < dataframe['bull_disp'] + min_range, 
                                         dataframe['bull_disp'] + min_range, dataframe['bear_disp'])
        
        # Define fvg_rng as the actual range between extremes
        dataframe['fvg_rng'] = (dataframe['bear_disp'] - dataframe['bull_disp']) #/ 2
        
        # Define channel levels within extremes
        dataframe['lower_extreme'] = dataframe['bull_disp'] * self.correction_factor.value
        dataframe['lower_inner'] = (dataframe['bull_disp'] + dataframe['fvg_rng'] * 0.236) * self.correction_factor.value
        dataframe['mid_point'] = (dataframe['bull_disp'] + dataframe['fvg_rng'] * 0.5) * self.correction_factor.value
        dataframe['upper_inner'] = (dataframe['bull_disp'] + dataframe['fvg_rng'] * 0.786) * self.correction_factor.value
        dataframe['upper_extreme'] = dataframe['bear_disp'] * self.correction_factor.value
        
        # Debug gap
        gap = (dataframe['upper_inner'] - dataframe['mid_point']) / dataframe['mid_point'] * 100
        print(f"Average Mid-to-Upper Gap (%): {gap.mean():.2f}")
        
        return dataframe

    def generate_signals(self, dataframe: DataFrame) -> DataFrame:
        dataframe['down_check'] = False
        dataframe['up_check'] = False
        dataframe['down_signal'] = False
        dataframe['up_signal'] = False
        
        for i in range(1, len(dataframe)):
            dataframe.at[i, 'down_check'] = dataframe['down_check'].iloc[i-1]
            dataframe.at[i, 'up_check'] = dataframe['up_check'].iloc[i-1]
            
            if dataframe['close'].iloc[i] < dataframe['upper_inner'].iloc[i]:
                dataframe.at[i, 'down_check'] = True
            if dataframe['close'].iloc[i] > dataframe['lower_inner'].iloc[i]:
                dataframe.at[i, 'up_check'] = True
                
            if (dataframe['down_check'].iloc[i] and 
                np.isnan(dataframe['bear_lvls_avg'].iloc[i]) and 
                dataframe['close_1'].iloc[i] > dataframe['open_1'].iloc[i] and 
                dataframe['close'].iloc[i] < dataframe['open'].iloc[i] and 
                dataframe['close'].iloc[i] < dataframe['open_1'].iloc[i]):
                dataframe.at[i, 'down_signal'] = True
                dataframe.at[i, 'down_check'] = False
                
            if (dataframe['up_check'].iloc[i] and 
                np.isnan(dataframe['bull_lvls_avg'].iloc[i]) and 
                dataframe['close_1'].iloc[i] < dataframe['open_1'].iloc[i] and 
                dataframe['close'].iloc[i] > dataframe['open'].iloc[i] and 
                dataframe['close'].iloc[i] > dataframe['open_1'].iloc[i]):
                dataframe.at[i, 'up_signal'] = True
                dataframe.at[i, 'up_check'] = False
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe['up_signal'], 'entry'] = 1

        dataframe.loc[
            (
                (dataframe["close"].shift(3) < dataframe['lower_extreme'].shift(3)) & 
                (dataframe['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['enter_long', 'enter_tag']] = (1, 'lower_extreme dip')

        dataframe.loc[
            (
                (dataframe["close"] > dataframe['lower_extreme']) & 
                (dataframe["close"].shift(1) < dataframe['lower_extreme'].shift(1)) & 
                (dataframe['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['enter_long', 'enter_tag']] = (1, 'lower_extreme cross')

        dataframe.loc[
            (
                (dataframe["close"] > dataframe['lower_inner']) & 
                (dataframe["close"].shift(1) < dataframe['lower_inner'].shift(1)) & 
                (dataframe['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['enter_long', 'enter_tag']] = (1, 'lower_inner cross')

        dataframe.loc[
            (
                (dataframe["close"] > dataframe['mid_point']) & 
                (dataframe["close"].shift(1) < dataframe['mid_point'].shift(1)) & 
                (dataframe['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['enter_long', 'enter_tag']] = (1, 'mid_point cross')

        dataframe.loc[
            (
                (dataframe["close"] > dataframe['upper_inner']) & 
                (dataframe["close"].shift(1) < dataframe['upper_inner'].shift(1)) & 
                (dataframe['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['enter_long', 'enter_tag']] = (1, 'mid_point cross')


        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe['down_signal'], 'exit'] = 1

        dataframe.loc[
            (
                (dataframe["close"] < dataframe['lower_inner']) & 
                (dataframe["close"].shift(1) > dataframe['lower_inner'].shift(1)) & 
                (dataframe['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['exit_long', 'exit_tag']] = (1, 'lower_inner cross')

        dataframe.loc[
            (
                (dataframe["close"] < dataframe['mid_point']) & 
                (dataframe["close"].shift(1) > dataframe['mid_point'].shift(1)) & 
                (dataframe['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['exit_long', 'exit_tag']] = (1, 'mid_point cross')

        dataframe.loc[
            (
                (dataframe["close"] < dataframe['upper_inner']) & 
                (dataframe["close"].shift(1) > dataframe['upper_inner'].shift(1)) & 
                (dataframe['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['exit_long', 'exit_tag']] = (1, 'upper_inner cross')

        dataframe.loc[
            (
                (dataframe["close"] < dataframe['upper_extreme']) & 
                (dataframe["close"].shift(1) > dataframe['upper_extreme'].shift(1)) & 
                (dataframe['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['exit_long', 'exit_tag']] = (1, 'upper_extreme cross')

        return dataframe