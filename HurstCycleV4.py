import math
import numpy as np
from freqtrade.strategy import IStrategy
import pandas as pd
import talib
from scipy.fft import fft
from technical import qtpylib
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)

class HurstCycleV4(IStrategy):
    timeframe = '4h'
    minimal_roi = {"0": 0.5}
    stoploss = -0.04
    trailing_stop = True
    trailing_stop_positive = 0.015
    can_short = False

    base_cycle_period = 20
    filter_weights = [1, 2, 4, 8, 4]

    ### Hyperoptable parameters ###
    u_window_size = IntParameter(70, 150, default=150, space='buy', optimize=True, load=True)
    l_window_size = IntParameter(20, 50, default=42, space='buy', optimize=True, load=True)
    convergence_threshold = DecimalParameter(0.002, 0.01, default=0.005, space='buy', optimize=True)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata['pair']
        # Heikin-Ashi Candles
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # FFT for Dominant Cycle and Harmonics
        if len(dataframe) < self.u_window_size.value:
            raise ValueError(f"Insufficient data points for FFT: {len(dataframe)}.")
        freq, power = perform_fft(dataframe['ha_close'], window_size=self.u_window_size.value)
        positive_mask = (1 / freq > self.l_window_size.value) & (1 / freq < self.u_window_size.value)
        positive_freqs = freq[positive_mask]
        positive_power = power[positive_mask]
        if len(positive_power) == 0:
            raise ValueError("No valid frequencies.")
        cycle_periods = 1 / positive_freqs
        power_threshold = 0.01 * np.max(positive_power)
        significant_indices = positive_power > power_threshold
        significant_periods = cycle_periods[significant_indices]
        significant_power = positive_power[significant_indices]
        dominant_freq_index = np.argmax(significant_power)
        dominant_freq = positive_freqs[dominant_freq_index]
        cycle_period = int(np.abs(1 / dominant_freq)) if dominant_freq != 0 else 100
        harmonics = [cycle_period / (i + 1) for i in range(1, 4)]
        self.cp = int(cycle_period)
        self.h0 = int(harmonics[0])
        self.h1 = int(harmonics[1])
        self.h2 = int(harmonics[2])

        # EWMA for cycles
        dataframe['cp'] = dataframe['ha_close'].ewm(span=self.cp).mean()
        dataframe['h0'] = dataframe['ha_close'].ewm(span=self.h0).mean()
        dataframe['h1'] = dataframe['ha_close'].ewm(span=self.h1).mean()
        dataframe['h2'] = dataframe['ha_close'].ewm(span=self.h2).mean()

        # Peak-to-Peak Movement (Amplitude Proxy)
        rolling_windowc = dataframe['ha_close'].rolling(self.cp)
        rolling_windowh0 = dataframe['ha_close'].rolling(self.h0)
        rolling_windowh1 = dataframe['ha_close'].rolling(self.h1)
        rolling_windowh2 = dataframe['ha_close'].rolling(self.h2)
        ptp_valuec = rolling_windowc.apply(lambda x: np.ptp(x))
        ptp_valueh0 = rolling_windowh0.apply(lambda x: np.ptp(x))
        ptp_valueh1 = rolling_windowh1.apply(lambda x: np.ptp(x))
        ptp_valueh2 = rolling_windowh2.apply(lambda x: np.ptp(x))
        dataframe['cycle_move'] = ptp_valuec / dataframe['ha_close']
        dataframe['h0_move'] = ptp_valueh0 / dataframe['ha_close']
        dataframe['h1_move'] = ptp_valueh1 / dataframe['ha_close']
        dataframe['h2_move'] = ptp_valueh2 / dataframe['ha_close']
        dataframe['cycle_move_mean'] = dataframe['cycle_move'].rolling(self.cp).mean()
        dataframe['h0_move_mean'] = dataframe['h0_move'].rolling(self.cp).mean()
        dataframe['h1_move_mean'] = dataframe['h1_move'].rolling(self.cp).mean()
        dataframe['h2_move_mean'] = dataframe['h2_move'].rolling(self.cp).mean()

        # Numerical Filter
        close = dataframe['ha_close'].values
        weights = np.array(self.filter_weights) / sum(self.filter_weights)
        filtered = np.convolve(close, weights, mode='valid')
        dataframe['filtered_close'] = pd.Series(filtered, index=dataframe.index[-len(filtered):])

        # Cyclic Model (Figures II-9 to II-11)
        # Sine waves for each cycle, amplitude from move_mean
        t = np.arange(len(dataframe))
        cycles = {
            'cp': self.cp,
            'h0': self.h0,
            'h1': self.h1,
            'h2': self.h2
        }
        amplitudes = {
            'cp': dataframe['cycle_move_mean'].fillna(method='ffill').fillna(0),
            'h0': dataframe['h0_move_mean'].fillna(method='ffill').fillna(0),
            'h1': dataframe['h1_move_mean'].fillna(method='ffill').fillna(0),
            'h2': dataframe['h2_move_mean'].fillna(method='ffill').fillna(0)
        }
        composite = np.zeros(len(dataframe))
        for cycle_name, period in cycles.items():
            # Sine wave: A * sin(2π * t / T)
            wave = amplitudes[cycle_name] * np.sin(2 * np.pi * t / period)
            dataframe[f'wave_{cycle_name}'] = wave
            composite += wave
        dataframe['composite_wave'] = composite    
        # Center the composite around the long-term trend (cp)
        dataframe['cyclic_model'] = dataframe['cp'] + composite

        # Troughs and Crests of the Cyclic Model
        dataframe['model_trough'] = dataframe['cyclic_model'].rolling(self.h0).min()
        dataframe['model_crest'] = dataframe['cyclic_model'].rolling(self.h0).max()
        dataframe['is_model_trough'] = np.where(dataframe['cyclic_model'] == dataframe['model_trough'], 1, 0)
        dataframe['is_model_crest'] = np.where(dataframe['cyclic_model'] == dataframe['model_crest'], 1, 0)

        # FLDs (for convergence check)
        half_short = math.ceil(self.h2 / 2)
        half_mid = math.ceil(self.h0 / 2)
        half_long = math.ceil(self.cp / 2)
        fld_short_base = dataframe['filtered_close'].rolling(window=self.h2, center=True).mean()
        fld_mid_base = dataframe['filtered_close'].rolling(window=self.h0, center=True).mean()
        fld_long_base = dataframe['filtered_close'].rolling(window=self.cp, center=True).mean()
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

        # Trend
        dataframe['trend'] = np.where(dataframe['filtered_close'] > dataframe['cp'], 1, -1)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (dataframe['is_model_trough'] == 1) &
            (dataframe['filtered_close'] < dataframe['cyclic_model']) &
            (dataframe['filtered_close'] < dataframe['agreeance_lower']) &
            (dataframe['converging'] == True) &
            (dataframe['trend'] == 1),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (dataframe['enter_long'].shift(1) == 1) &
            (dataframe['filtered_close'] >= dataframe['cyclic_model']),
            'exit_long'] = 1
        return dataframe

    def leverage(self, pair: str, current_time, current_rate: float, proposed_leverage: float, 
                 max_leverage: float, side: str, **kwargs) -> float:
        return 3.0

def perform_fft(price_data, window_size=None):
    if window_size is not None:
        price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
    normalized_data = (price_data - np.mean(price_data)) / np.std(price_data)
    n = len(normalized_data)
    fft_data = np.fft.fft(normalized_data)
    freq = np.fft.fftfreq(n)
    power = np.abs(fft_data) ** 2
    power[np.isinf(power)] = 0
    return freq, power