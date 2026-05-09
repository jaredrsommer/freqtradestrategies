import math
import numpy as np
from freqtrade.strategy import IStrategy
import pandas as pd
import talib
from scipy.fft import fft
from technical import qtpylib
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
import pandas_ta as pta
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
import pandas_ta as pta
import logging

logger = logging.getLogger(__name__)

class HurstCycleV5(IStrategy):
    timeframe = '1h'
    minimal_roi = {"0": 1}
    stoploss = -0.99
    trailing_stop = False
    trailing_stop_positive = 0.04
    can_short = True
    position_adjustment_enable = True

    base_cycle_period = 20
    filter_weights = [1, 2, 4, 8, 4]
    # New parameter for convergence threshold (tunable)
    convergence_threshold = 0.005  # 0.5% of price as max spread

    ### Hyperoptable parameters ###
    u_window_size = IntParameter(70, 150, default=250, space='buy', optimize=True, load=True)
    l_window_size = IntParameter(20, 50, default=42, space='buy', optimize=True, load=True)
    rsi_period = IntParameter(7, 21, default=21, space="buy", optimize=True, load=True)
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: float, max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        
        # Dividiamo lo stake proposto in tre parti uguali
        return proposed_stake / 3
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, min_stake: float, max_stake: float,
                        **kwargs) -> Optional[float]:
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        trade_duration = (current_time - trade.open_date_utc).seconds / 60

        # Contiamo il numero di ordini già eseguiti per questo trade
        filled_entries = trade.nr_of_successful_entries

        # Se abbiamo già 3 entrate, non ne aggiungiamo altre
        if filled_entries >= 3:
            return None
        
        if current_profit > 0:
            return None

        # Verifichiamo il tempo trascorso dall'ultima entrata
        if filled_entries > 0 and trade_duration < 240:
            return None
        if filled_entries > 1 and trade_duration < 240 * 1.5:
            return None
        if filled_entries > 2 and trade_duration < 240 * 2.25:
            return None

        # Verifichiamo se ci sono segnali di entrata nelle ultime candele
       
        long_entry_signals = dataframe['enter_long'].rolling(window=3).sum().iloc[-1]
        short_entry_signals = dataframe['enter_short'].rolling(window=3).sum().iloc[-1]
        
        # Se abbiamo segnali di entrata e il prezzo è sotto il VTL up
        if not trade.is_short and long_entry_signals > 0:
            logger.info(f"{trade.pair} DCA Long Signal Detected")
            # Calcoliamo il nuovo stake (uguale allo stake iniziale)
            return trade.stake_amount
        if trade.is_short and short_entry_signals > 0:
            logger.info(f"{trade.pair} DCA Short Signal Detected")
            # Calcoliamo il nuovo stake (uguale allo stake iniziale)
            return trade.stake_amount
            
        return None
    
    def linear_regression_channel(self, data: pd.Series, window: int, num_dev: float):
        """    Calcola la linea di regressione e il canale di deviazione standard (bande superiore e inferiore).    
        :param data: Serie di dati (prezzi di chiusura)    
        :param window: Lunghezza della finestra di regressione    
        :param num_dev: Numero di deviazioni standard per le bande    
        :return: Linea centrale (regressione), banda superiore e banda inferiore    
        """    
        # Lista per contenere i valori di output    
        lr_channel = {'mid': [], 'upper': [], 'lower': []}

        for i in range(window, len(data)):
            # Seleziona la finestra corrente        
            y = data[i-window:i]

            # Calcola l'indice del tempo per la finestra        
            x = np.arange(window)

            # Regressione lineare sui dati della finestra        
            A = np.vstack([x, np.ones(len(x))]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

            # Calcola la linea centrale (y = mx + b)
            y_line = intercept + slope * x[-1]

            # Calcola la deviazione standard        
            residuals = y - (intercept + slope * x)
            std_dev = np.std(residuals)

            # Linea centrale, banda superiore e inferiore        
            lr_channel['mid'].append(y_line)
            lr_channel['upper'].append(y_line + num_dev * std_dev)
            lr_channel['lower'].append(y_line - num_dev * std_dev)

        # Riempire i valori iniziali con NaN per mantenere la lunghezza della serie uguale    
        lr_channel['mid'] = [np.nan] * window + lr_channel['mid']
        lr_channel['upper'] = [np.nan] * window + lr_channel['upper']
        lr_channel['lower'] = [np.nan] * window + lr_channel['lower']

        return pd.DataFrame(lr_channel)

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

        # Troughs and Crests
        dataframe['trough'] = dataframe['filtered_close'].rolling(self.h2).min()
        dataframe['crest'] = dataframe['filtered_close'].rolling(self.h2).max()
        dataframe['is_trough'] = np.where(dataframe['filtered_close'] == dataframe['trough'], 1, 0)
        dataframe['is_crest'] = np.where(dataframe['filtered_close'] == dataframe['crest'], 1, 0)

        # VTL based on last two troughs/crests
        # Initialize VTL columns
        dataframe['vtl_up'] = np.nan
        dataframe['vtl_down'] = np.nan

        # Group troughs and crests within a window (self.h2 candles)
        group_window = self.h2
        trough_groups = []
        crest_groups = []
        current_trough_group = []
        current_crest_group = []
        last_trough_idx = None
        last_crest_idx = None

        # Identify trough and crest groups
        for idx in dataframe.index:
            if dataframe.at[idx, 'is_trough'] == 1:
                if last_trough_idx is None or (dataframe.index.get_loc(idx) - dataframe.index.get_loc(last_trough_idx)) <= group_window:
                    current_trough_group.append(idx)
                else:
                    if len(current_trough_group) > 0:
                        prices = [dataframe.at[i, 'filtered_close'] for i in current_trough_group]
                        min_idx = current_trough_group[np.argmin(prices)]
                        trough_groups.append(min_idx)
                    current_trough_group = [idx]
                last_trough_idx = idx
            
            if dataframe.at[idx, 'is_crest'] == 1:
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
                        dataframe.at[idx, 'vtl_up'] = vtl_value
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
                        dataframe.at[idx, 'vtl_down'] = vtl_value
                        # Check for break (price above VTL)
                        if current_price > vtl_value and i + 1 < len(down_vtl_segments):
                            segment[1] = idx
                    else:
                        segment[1] = idx if end_idx is None else end_idx

        # Slopes for cycle sync
        dataframe['fld_short_slope'] = dataframe['fld_short'].diff()
        dataframe['fld_mid_slope'] = dataframe['fld_mid'].diff()
        dataframe['fld_long_slope'] = dataframe['fld_long'].diff()

        # Trend
        dataframe['trend'] = np.where(dataframe['filtered_close'] > dataframe['cp'], 1, -1)
        dataframe['rsi'] = pta.rsi(dataframe['close'], length=self.rsi_period.value)
        dataframe['rsi_fast'] = pta.rsi(dataframe['close'], length=4)
        dataframe['rsi1'] = ((pta.rsi(dataframe['close'], length=self.rsi_period.value*4) -50) *2.15) +50
        dataframe['rsi4'] = ((pta.rsi(dataframe['close'], length=self.rsi_period.value*4*4) -50) *3.75) +50
        dataframe['rsi8'] = ((pta.rsi(dataframe['close'], length=self.rsi_period.value*4*8) -50) *5) +50
        
        regression_channel = self.linear_regression_channel(dataframe['rsi_fast'], window=150, num_dev=1.0)

        dataframe['lr_mid_rsi'] = regression_channel['mid']
        dataframe['lr_upper_rsi'] = regression_channel['upper']
        dataframe['lr_lower_rsi'] = regression_channel['lower']

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        
        dataframe.loc[
            # (dataframe['is_trough'] == 1) &
            # (dataframe['filtered_close'] < dataframe['fld_short']) &
            # (dataframe['filtered_close'] < dataframe['agreeance_lower']) &
            # (dataframe['converging'] == True) &
            # (dataframe['trend'] == 1) &
            ~(dataframe['vtl_up'] > dataframe['vtl_down']) &
            (dataframe['lr_mid_rsi'] < 35) &
            (dataframe['close'].rolling(5).min() < dataframe['vtl_up']) &
            (dataframe['close'] > dataframe['vtl_up']),
            'enter_long'] = 1
        
        dataframe.loc[
            # (dataframe['is_trough'] == 1) &
            # (dataframe['filtered_close'] < dataframe['fld_short']) &
            # (dataframe['filtered_close'] < dataframe['agreeance_lower']) &
            # (dataframe['converging'] == True) &
            # (dataframe['trend'] == 1) &
            ~(dataframe['vtl_up'] > dataframe['vtl_down']) &
            (dataframe['lr_mid_rsi'] > 65) &
            (dataframe['close'].rolling(5).max() > dataframe['vtl_down']) &
            (dataframe['close'] < dataframe['vtl_down']),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            # (dataframe['enter_long'].shift(1) == 1) &
            (dataframe['close'] > dataframe['vtl_down']), 
            'exit_long'] = 1
        dataframe.loc[
            # (dataframe['enter_long'].shift(1) == 1) &
            (dataframe['close'] < dataframe['vtl_up']), 
            'exit_short'] = 1
        return dataframe

    def leverage(self, pair: str, current_time, current_rate: float, proposed_leverage: float, 
                 max_leverage: float, side: str, **kwargs) -> float:
        return 5.0

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