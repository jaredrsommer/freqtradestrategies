import math
import numpy as np
from freqtrade.strategy import IStrategy
import pandas as pd
import talib
from scipy.fft import fft
from technical import qtpylib
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)

class HurstCycleV6(IStrategy):
    timeframe = '15m'
    minimal_roi = {"0": 1}
    stoploss = -0.15
    trailing_stop = True
    trailing_stop_positive = 0.015
    can_short = False

    base_cycle_period = 20
    filter_weights = [1, 2, 4, 8, 4]
    # New parameter for convergence threshold (tunable)
    convergence_threshold = 0.005  # 0.5% of price as max spread

    ### Hyperoptable parameters ###
    u_window_size = IntParameter(250, 350, default=250, space='buy', optimize=True, load=True)
    l_window_size = IntParameter(20, 50, default=42, space='buy', optimize=True, load=True)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata['pair']
        # Heikin-Ashi Candles
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # Initialize Hurst Cycles
        cycle_period = 80
        harmonics = [0, 0, 0]
        harmonics[0] = 40
        harmonics[1] = 27
        harmonics[2] = 20

        if len(dataframe) < self.u_window_size.value:
            raise ValueError(f"Insufficient data points for FFT: {len(dataframe)}. Need at least {self.u_window_size.value} data points.")

        # Perform FFT
        freq, power = perform_fft(dataframe['ha_close'], window_size=self.u_window_size.value)
        if len(freq) == 0 or len(power) == 0:
            raise ValueError("FFT resulted in zero or invalid frequencies.")

        positive_mask = (1 / freq > self.l_window_size.value) & (1 / freq < self.u_window_size.value)
        positive_freqs = freq[positive_mask]
        positive_power = power[positive_mask]
        if len(positive_power) == 0:
            raise ValueError("No positive frequencies meet the filtering criteria.")

        cycle_periods = 1 / positive_freqs
        power_threshold = 0 if len(positive_power) == 0 else 0.01 * np.max(positive_power)
        significant_indices = positive_power > power_threshold
        significant_periods = cycle_periods[significant_indices]
        significant_power = positive_power[significant_indices]

        dominant_freq_index = np.argmax(significant_power)
        dominant_freq = positive_freqs[dominant_freq_index]
        cycle_period = int(np.abs(1 / dominant_freq)) if dominant_freq != 0 else 100
        if cycle_period == np.inf:
            raise ValueError("No dominant frequency found.")

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

        # Peak-to-Peak Movement
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

        # FLDs
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
        # Calculate spread between FLDs
        fld_spread = dataframe[['fld_short', 'fld_mid', 'fld_long']].max(axis=1) - \
                     dataframe[['fld_short', 'fld_mid', 'fld_long']].min(axis=1)
        # Relative spread as a percentage of current price
        relative_spread = fld_spread / dataframe['filtered_close']
        
        # Define convergence condition
        dataframe['converging'] = relative_spread < self.convergence_threshold
        
        # Agreeance Band: Mean of FLDs ± a small buffer when converging
        band_center = dataframe[['fld_short', 'fld_mid', 'fld_long']].mean(axis=1)
        band_width = dataframe['filtered_close'] * 0.01  # 1% of price as buffer
        dataframe['agreeance_upper'] = np.where(dataframe['converging'],
                                               band_center + band_width,
                                               np.nan)
        dataframe['agreeance_lower'] = np.where(dataframe['converging'],
                                               band_center - band_width,
                                               np.nan)

        # Generate VTLs for each harmonic
        harmonics_dict = {'cp': self.cp, 'h0': self.h0, 'h1': self.h1, 'h2': self.h2}
        
        for harmonic_name, harmonic_period in harmonics_dict.items():
            self._create_vtl_for_harmonic(dataframe, harmonic_name, harmonic_period)

        # Calculate slopes for each VTL
        for harmonic_name in harmonics_dict.keys():
            # Calculate slopes for up VTLs
            dataframe[f'vtl_up_{harmonic_name}_slope'] = dataframe[f'vtl_up_{harmonic_name}'].diff()
            # Calculate slopes for down VTLs
            dataframe[f'vtl_down_{harmonic_name}_slope'] = dataframe[f'vtl_down_{harmonic_name}'].diff()

        # Sum and average the VTL slopes
        # Sum of all up VTL slopes divided by 4
        up_slope_cols = [f'vtl_up_{harmonic}_slope' for harmonic in harmonics_dict.keys()]
        dataframe['vtl_up_slopes_sum'] = dataframe[up_slope_cols].sum(axis=1)
        dataframe['vtl_up_slopes_avg'] = dataframe['vtl_up_slopes_sum'] / 4

        # Sum of all down VTL slopes divided by 4
        down_slope_cols = [f'vtl_down_{harmonic}_slope' for harmonic in harmonics_dict.keys()]
        dataframe['vtl_down_slopes_sum'] = dataframe[down_slope_cols].sum(axis=1)
        dataframe['vtl_down_slopes_avg'] = dataframe['vtl_down_slopes_sum'] / 4

        # Combined average of all VTL slopes (up and down combined)
        dataframe['vtl_all_slopes_avg'] = ((dataframe['vtl_up_slopes_sum'] + dataframe['vtl_down_slopes_sum']) / dataframe['filtered_close']) * 100

        # Slopes for cycle sync
        dataframe['fld_short_slope'] = dataframe['fld_short'].diff()
        dataframe['fld_mid_slope'] = dataframe['fld_mid'].diff()
        dataframe['fld_long_slope'] = dataframe['fld_long'].diff()

        # Trend
        dataframe['trend'] = np.where(dataframe['filtered_close'] > dataframe['cp'], 1, -1)

        return dataframe

    def _create_vtl_for_harmonic(self, dataframe: pd.DataFrame, harmonic_name: str, harmonic_period: int):
        """Create VTL lines for a specific harmonic period"""
        
        # Troughs and Crests for this harmonic
        dataframe[f'trough_{harmonic_name}'] = dataframe['filtered_close'].rolling(harmonic_period).min()
        dataframe[f'crest_{harmonic_name}'] = dataframe['filtered_close'].rolling(harmonic_period).max()
        dataframe[f'is_trough_{harmonic_name}'] = np.where(
            dataframe['filtered_close'] == dataframe[f'trough_{harmonic_name}'], 1, 0)
        dataframe[f'is_crest_{harmonic_name}'] = np.where(
            dataframe['filtered_close'] == dataframe[f'crest_{harmonic_name}'], 1, 0)

        # Initialize VTL columns for this harmonic
        dataframe[f'vtl_up_{harmonic_name}'] = np.nan
        dataframe[f'vtl_down_{harmonic_name}'] = np.nan

        # Group troughs and crests within a window
        group_window = harmonic_period
        trough_groups = []
        crest_groups = []
        current_trough_group = []
        current_crest_group = []
        last_trough_idx = None
        last_crest_idx = None

        # Identify trough and crest groups
        for idx in dataframe.index:
            if dataframe.at[idx, f'is_trough_{harmonic_name}'] == 1:
                if last_trough_idx is None or (dataframe.index.get_loc(idx) - dataframe.index.get_loc(last_trough_idx)) <= group_window:
                    current_trough_group.append(idx)
                else:
                    if len(current_trough_group) > 0:
                        prices = [dataframe.at[i, 'filtered_close'] for i in current_trough_group]
                        min_idx = current_trough_group[np.argmin(prices)]
                        trough_groups.append(min_idx)
                    current_trough_group = [idx]
                last_trough_idx = idx
            
            if dataframe.at[idx, f'is_crest_{harmonic_name}'] == 1:
                if last_crest_idx is None or (dataframe.index.get_loc(idx) - dataframe.index.get_loc(last_crest_idx)) <= group_window:
                    current_crest_group.append(idx)
                else:
                    if len(current_crest_group) > 0:
                        prices = [dataframe.at[i, 'filtered_close'] for i in current_crest_group]
                        max_idx = current_crest_group[np.argmax(prices)]
                        crest_groups.append(max_idx)
                    current_crest_group = [idx]
                last_crest_idx = idx

        # Add final groups
        if len(current_trough_group) > 0:
            prices = [dataframe.at[i, 'filtered_close'] for i in current_trough_group]
            min_idx = current_trough_group[np.argmin(prices)]
            trough_groups.append(min_idx)
        if len(current_crest_group) > 0:
            prices = [dataframe.at[i, 'filtered_close'] for i in current_crest_group]
            max_idx = current_crest_group[np.argmax(prices)]
            crest_groups.append(max_idx)

        # Price bounds
        price_min = dataframe['filtered_close'].min()
        price_max = dataframe['filtered_close'].max()

        # Store VTL segments: [start_idx, end_idx, slope, intercept]
        up_vtl_segments = []
        down_vtl_segments = []

        # Create VTL segments for trough groups
        for i in range(1, len(trough_groups)):
            x1 = dataframe.index.get_loc(trough_groups[i-1])
            y1 = dataframe.at[trough_groups[i-1], 'filtered_close']
            x2 = dataframe.index.get_loc(trough_groups[i])
            y2 = dataframe.at[trough_groups[i], 'filtered_close']
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            intercept = y1 - slope * x1
            up_vtl_segments.append([trough_groups[i-1], None, slope, intercept])

        # Create VTL segments for crest groups
        for i in range(1, len(crest_groups)):
            x1 = dataframe.index.get_loc(crest_groups[i-1])
            y1 = dataframe.at[crest_groups[i-1], 'filtered_close']
            x2 = dataframe.index.get_loc(crest_groups[i])
            y2 = dataframe.at[crest_groups[i], 'filtered_close']
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            intercept = y1 - slope * x1
            down_vtl_segments.append([crest_groups[i-1], None, slope, intercept])

        # Apply VTL segments across the DataFrame
        for idx in dataframe.index:
            current_x = dataframe.index.get_loc(idx)
            current_price = dataframe.at[idx, 'filtered_close']
            
            # Apply vtl_up
            for i, segment in enumerate(up_vtl_segments):
                start_idx, end_idx, slope, intercept = segment
                if idx >= start_idx and (end_idx is None or idx <= end_idx):
                    vtl_value = slope * current_x + intercept
                    if price_min <= vtl_value <= price_max:
                        dataframe.at[idx, f'vtl_up_{harmonic_name}'] = vtl_value
                        # Check for break (price below VTL)
                        if current_price < vtl_value and i + 1 < len(up_vtl_segments):
                            segment[1] = idx  # End this segment
                    else:
                        segment[1] = idx if end_idx is None else end_idx
            
            # Apply vtl_down
            for i, segment in enumerate(down_vtl_segments):
                start_idx, end_idx, slope, intercept = segment
                if idx >= start_idx and (end_idx is None or idx <= end_idx):
                    vtl_value = slope * current_x + intercept
                    if price_min <= vtl_value <= price_max:
                        dataframe.at[idx, f'vtl_down_{harmonic_name}'] = vtl_value
                        # Check for break (price above VTL)
                        if current_price > vtl_value and i + 1 < len(down_vtl_segments):
                            segment[1] = idx
                    else:
                        segment[1] = idx if end_idx is None else end_idx

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Entry conditions for each harmonic VTL
        entry_conditions = []

        # CP (dominant cycle) entry condition
        cp_entry = (
            (dataframe['close'] < dataframe['vtl_up_h0'])
        )
        entry_conditions.append(cp_entry)
        
        # H0 (longest harmonic) entry condition
        h0_entry = (
            (dataframe['close'] < dataframe['vtl_up_h0']) 
        )
        entry_conditions.append(h0_entry)
        
        # H1 (medium harmonic) entry condition
        h1_entry = (
            (dataframe['close'] < dataframe['vtl_up_h1'])
        )
        entry_conditions.append(h1_entry)
        
        # H2 (shortest harmonic) entry condition
        h2_entry = (
            (dataframe['close'] < dataframe['vtl_up_h2']) 
        )
        entry_conditions.append(h2_entry)
        
        # Combined entry: any harmonic VTL breach with trend confirmation
        dataframe.loc[
            cp_entry | h0_entry | h1_entry | h2_entry,
            'enter_long'
        ] = 1

        # Optional: Individual entry signals for each harmonic
        dataframe.loc[cp_entry, 'enter_tag'] = 'cp'
        dataframe.loc[h0_entry, 'enter_tag'] = 'h0'
        dataframe.loc[h1_entry, 'enter_tag'] = 'h1'
        dataframe.loc[h2_entry, 'enter_tag'] = 'h2'

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Exit conditions for each harmonic VTL
        exit_conditions = []
        
        # CP (dominant cycle) exit condition
        cp_exit = (dataframe['close'] > dataframe['vtl_down_cp'])
        exit_conditions.append(cp_exit)
        
        # H0 (longest harmonic) exit condition
        h0_exit = (dataframe['close'] > dataframe['vtl_down_h0'])
        exit_conditions.append(h0_exit)
        
        # H1 (medium harmonic) exit condition
        h1_exit = (dataframe['close'] > dataframe['vtl_down_h1'])
        exit_conditions.append(h1_exit)
        
        # H2 (shortest harmonic) exit condition
        h2_exit = (dataframe['close'] > dataframe['vtl_down_h2'])
        exit_conditions.append(h2_exit)
        
        # Combined exit: any harmonic VTL breach upward
        dataframe.loc[
            cp_exit | h0_exit | h1_exit | h2_exit, 
            'exit_long'
        ] = 1

        # Optional: Individual exit signals for each harmonic
        dataframe.loc[cp_exit, 'exit_tag'] = 'cp'
        dataframe.loc[h0_exit, 'exit_tag'] = 'h0'
        dataframe.loc[h1_exit, 'exit_tag'] = 'h1'
        dataframe.loc[h2_exit, 'exit_tag'] = 'h2'

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