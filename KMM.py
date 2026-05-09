import numpy as np
import pandas as pd
import openai
import re
import time
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
from freqtrade.persistence import Trade
from pandas_ta.utils import get_offset
from technical import qtpylib
import logging
from typing import Dict, List, Optional
from functools import reduce
from uuid import uuid4
from datetime import datetime, timezone
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_minutes
from scipy.signal import argrelextrema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KMM(IStrategy):

    '''
          ______   __          __              __    __   ______   __    __        __     __    __             ______            
     /      \ /  |       _/  |            /  |  /  | /      \ /  \  /  |      /  |   /  |  /  |           /      \           
    /$$$$$$  |$$ |____  / $$ |    _______ $$ | /$$/ /$$$$$$  |$$  \ $$ |     _$$ |_  $$ |  $$ |  _______ /$$$$$$  |  _______ 
    $$ |  $$/ $$      \ $$$$ |   /       |$$ |/$$/  $$ ___$$ |$$$  \$$ |    / $$   | $$ |__$$ | /       |$$$  \$$ | /       |
    $$ |      $$$$$$$  |  $$ |  /$$$$$$$/ $$  $$<     /   $$< $$$$  $$ |    $$$$$$/  $$    $$ |/$$$$$$$/ $$$$  $$ |/$$$$$$$/ 
    $$ |   __ $$ |  $$ |  $$ |  $$ |      $$$$$  \   _$$$$$  |$$ $$ $$ |      $$ | __$$$$$$$$ |$$ |      $$ $$ $$ |$$      \ 
    $$ \__/  |$$ |  $$ | _$$ |_ $$ \_____ $$ |$$  \ /  \__$$ |$$ |$$$$ |      $$ |/  |     $$ |$$ \_____ $$ \$$$$ | $$$$$$  |
    $$    $$/ $$ |  $$ |/ $$   |$$       |$$ | $$  |$$    $$/ $$ | $$$ |______$$  $$/      $$ |$$       |$$   $$$/ /     $$/ 
     $$$$$$/  $$/   $$/ $$$$$$/  $$$$$$$/ $$/   $$/  $$$$$$/  $$/   $$//      |$$$$/       $$/  $$$$$$$/  $$$$$$/  $$$$$$$/  
                                                                       $$$$$$/                                               
                                                                                                                             
    '''          

    timeframe = "15m"
    minimal_roi = {}
    locked_stoploss = {}
    can_short = False
    stoploss = -0.08
    trailing_stop = False
    use_custom_stoploss = True
    exit_profit_only = False


    q_noise = DecimalParameter(0.0001, 0.01, decimals=6, default=0.001, space="buy")
    r_noise = DecimalParameter(0.1, 10.0, decimals=2, default=1.0, space="buy")

    ### Custom Functions
    # Threshold and Limits
    u_window_size = IntParameter(100, 160, default=100, space='buy', optimize=True)
    l_window_size = IntParameter(5, 40, default=40, space='buy', optimize=True)
    exclusion_zone = DecimalParameter(0.0005, 0.005, default=0.001, decimals=4, optimize=True)

    # Custom Entry
    increment = DecimalParameter(low=1.0005, high=1.002, default=1.001, decimals=4 ,space='buy', optimize=True, load=True)
    last_entry_price = None

    # protections
    cooldown_lookback = IntParameter(24, 48, default=12, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # negative stoploss
    use_stop1 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    use_stop2 = BooleanParameter(default=True, space="protection", optimize=True, load=True)
    use_stop3 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    use_stop4 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    # roi
    time0 = IntParameter(low=1440, high=2600, default=1440, space='sell', optimize=True, load=True)
    time1 = IntParameter(low=1440, high=2600, default=2000, space='sell', optimize=True, load=True)
    time2 = IntParameter(low=2600, high=4000, default=3200, space='sell', optimize=True, load=True)
    time3 = IntParameter(low=2500, high=5000, default=4500, space='sell', optimize=True, load=True)  

    def __init__(self, config: Dict):
        super().__init__(config)
        self.api_key = config.get("openai_api_key", "sk-DaLfGHdZ4tyHpvqC2rHZBg")
        if not self.api_key:
            logger.error("Akash API key not provided! 🚫")
            raise ValueError("Akash API key required")
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://chatapi.akash.network/api/v1"
        )
        self.custom_data = {
            "recent_trades": {},
            "signal_queue": {},
            "signal_history": {},  # Cache signals by datetime
            "api_failures": 0,
            "last_prompts": [],
            "last_api_call": {},
            "last_signal": {},
            "current_candle_signal": {}
        }

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        trade_duration = (current_time - trade.open_date_utc).seconds / 60
        last_fill = (current_time - trade.date_last_filled_utc).seconds / 60 

        current_candle = dataframe.iloc[-1].squeeze()
        TP0 = current_candle[f'h2_move_mean'] 
        TP1 = current_candle[f'h1_move_mean'] 
        TP2 = current_candle[f'h0_move_mean'] 
        TP3 = current_candle[f'cycle_move_mean']
        display_profit = current_profit * 100

        tp0 = TP0 * 100
        tp1 = TP1 * 100
        tp2 = TP2 * 100
        tp3 = TP3 * 100
        dca_str = ''
        dcaDisp = display_profit+tp1
        if display_profit < 0:
            dca_str = f'| DCA Dist: {dcaDisp:.3}% | Last Fill: {last_fill:.5}m"' 

        if current_candle['enter_long'] is not None:
            signal = current_candle['enter_long']

        if current_profit is not None:
            logger.info(f"{trade.pair} - Current Profit: {display_profit:.3}% # of Entries: {trade.nr_of_successful_entries} {dca_str}")
            logger.info(f"{trade.pair} - TP0: {tp0:.3}% | TP1: {tp1:.3}% | TP2: {tp2:.3}% | TP3: {tp3:.3}%")
        # Take Profit if m00n
        if current_profit > TP2 and trade.nr_of_successful_exits == 0:
            # Take quarter of the profit at next fib%%
            return -(trade.stake_amount / 2)
        if current_profit > TP3 and trade.nr_of_successful_exits == 1:
            # Take half of the profit at last fib%%
            return -(trade.stake_amount / 2)
        if current_profit > (TP3 * 1.5) and trade.nr_of_successful_exits == 2:
            # Take half of the profit at last fib%%
            return -(trade.stake_amount / 2)
        if current_profit > (TP3 * 2.0) and trade.nr_of_successful_exits == 3:
            # Take profit at last fib%%
            return -(trade.stake_amount)
        # Take Profit Early if DCA was used 
        if current_profit > TP0 and trade.nr_of_successful_exits == 0 and count_of_entries == 2:
            # Take half of the profit at next fib%%
            return -(trade.stake_amount / 2)
        if current_profit > TP1 and trade.nr_of_successful_exits == 1 and count_of_entries == 2:
            # Take profit at last fib%%
            return -(trade.stake_amount)
            
        return None


    ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        SLT0 = current_candle[f'h2_move_mean']
        SLT1 = current_candle[f'h1_move_mean']
        SLT2 = current_candle[f'h0_move_mean']
        SLT3 = current_candle[f'cycle_move_mean']
        
        display_profit = current_profit * 100

        if current_profit < -0.01:
            if pair in self.locked_stoploss:
                del self.locked_stoploss[pair]
                self.dp.send_msg(f'*** {pair} *** Stoploss reset.')
                logger.info(f'*** {pair} *** Stoploss reset.')
            return self.stoploss

        new_stoploss = None
        if SLT3 is not None and current_profit > SLT3:
            new_stoploss = (SLT2 - SLT1)
            level = 4
        elif SLT2 is not None and current_profit > SLT2:
            new_stoploss = (SLT2 - SLT1)
            level = 3

        # in the future toggle these on certain conditions with indicators.
        elif SLT1 is not None and current_profit > SLT1:
            new_stoploss = (SLT1 - SLT0)
            level = 2
        elif SLT0 is not None and current_profit > SLT0:
            new_stoploss = (SLT1 - SLT0)
            level = 1

        if new_stoploss is not None:
            if pair not in self.locked_stoploss or new_stoploss > self.locked_stoploss[pair]:
                self.locked_stoploss[pair] = new_stoploss
                self.dp.send_msg(f'*** {pair} *** Profit {level} {display_profit:.3f}%% - New stoploss: {new_stoploss:.4f} activated')
                logger.info(f'*** {pair} *** Profit {level} {display_profit:.3f}%% - New stoploss: {new_stoploss:.4f} activated')
            return self.locked_stoploss[pair]

        return self.stoploss


    def custom_entry_price(self, pair: str, trade: Optional['Trade'], current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)

        entry_price = (dataframe['close'].iat[-1] + dataframe['open'].iat[-1] + proposed_rate) / 3
        logger.info(f"{pair} Using Entry Price: {entry_price} | close: {dataframe['close'].iat[-1]} open: {dataframe['open'].iat[-1]} proposed_rate: {proposed_rate}")
        self.dp.send_msg(f"{pair} Using Entry Price: {entry_price} | close: {dataframe['close'].iat[-1]} open: {dataframe['open'].iat[-1]} proposed_rate: {proposed_rate}")

        # Check if there is a stored last entry price and if it matches the proposed entry price
        if self.last_entry_price is not None and abs(entry_price - self.last_entry_price) < 0.0001:  # Tolerance for floating-point comparison
            entry_price *= self.increment.value # Increment by 0.2%%
            logger.info(f"{pair} Incremented entry price: {entry_price} based on previous entry price : {self.last_entry_price}.")

        # Update the last entry price
        self.last_entry_price = entry_price

        return entry_price

    # Custom_Exits
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
    
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        trade_duration = (current_time - trade.open_date_utc).seconds / 60
        last_fill = (current_time - trade.date_last_filled_utc).seconds / 60 

        current_candle = dataframe.iloc[-1].squeeze()
        TP0 = current_candle[f'h2_move_mean'] 
        TP1 = current_candle[f'h1_move_mean']
        TP2 = current_candle[f'h0_move_mean']
        TP3 = current_candle[f'cycle_move_mean']

        ### roi ###
        if current_profit > TP3 and trade_duration > self.time0.value:
            return 'Roi 0 - Easy $$$'
        if current_profit > TP2 and trade_duration > self.time1.value:
            return 'Roi 1 - ol reliable' 
        if current_profit > TP1 and trade_duration > self.time2.value:
            return 'Roi 2 - Avg Joe'
        if current_profit > TP0 and trade_duration > self.time3.value:
            return 'Roi 3 - Better than Nothing'

        return False

    def custom_kalman_filter(self, data: pd.Series, q: float, r: float) -> tuple:
        n = len(data)
        state = np.zeros((2, n))
        state[:, 0] = [data.iloc[0], 0]
        P = np.eye(2) * 0.1
        F = np.array([[1, 1], [0, 1]])
        H = np.array([[1, 0]])
        Q = np.eye(2) * q
        R = np.array([[r]])
        I = np.eye(2)

        for t in range(1, n):
            state[:, t] = F @ state[:, t-1]
            P = F @ P @ F.T + Q
            Z = data.iloc[t]
            y = Z - H @ state[:, t]
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            state[:, t] = state[:, t] + K @ y
            P = (I - K @ H) @ P

        return state[0, :], state[1, :]

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        pair = metadata['pair']
        heikinashi = qtpylib.heikinashi(dataframe)

        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # Initialize Hurst Cycles for startup errors
        cycle_period = 80
        harmonics = [0, 0, 0]
        harmonics[0] = 40
        harmonics[1] = 27
        harmonics[2] = 20

        if len(dataframe) < self.u_window_size.value:
            raise ValueError(f"Insufficient data points for FFT: {len(dataframe)}. Need at least {self.u_window_size.value} data points.")

        # Perform FFT to identify cycles with a rolling window
        freq, power = perform_fft(dataframe['ha_close'], window_size=self.u_window_size.value)

        if len(freq) == 0 or len(power) == 0:
            raise ValueError("FFT resulted in zero or invalid frequencies. Check the data or the FFT implementation.")

        # Filter out the zero-frequency component and limit the frequency to below 500
        positive_mask = (1 / freq > self.l_window_size.value) & (1 / freq < self.u_window_size.value)
        positive_freqs = freq[positive_mask]
        positive_power = power[positive_mask]

        # Check if there are valid frequencies after filtering
        if len(positive_power) == 0:
            raise ValueError("No positive frequencies meet the filtering criteria. Adjust window sizes or check the data.")

        # Convert frequencies to periods
        cycle_periods = 1 / positive_freqs

        # Set a threshold to filter out insignificant cycles based on power
        power_threshold = 0 if len(positive_power) == 0 else 0.01 * np.max(positive_power)
        significant_indices = positive_power > power_threshold
        significant_periods = cycle_periods[significant_indices]
        significant_power = positive_power[significant_indices]

        # Identify the dominant cycle
        dominant_freq_index = np.argmax(significant_power)
        dominant_freq = positive_freqs[dominant_freq_index]
        # logger.info(f'{pair} Hurst Exponent: {dominant_freq}')
        cycle_period = int(np.abs(1 / dominant_freq)) if dominant_freq != 0 else 100

        if cycle_period == np.inf:
            raise ValueError("No dominant frequency found. Check the data or the method used.")

        # Calculate harmonics for the dominant cycle
        harmonics = [cycle_period / (i + 1) for i in range(1, 4)]
        # print(cycle_period, harmonics)
        self.cp = int(cycle_period)
        self.h0 = int(harmonics[0])
        self.h1 = int(harmonics[1])
        self.h2 = int(harmonics[2])
        dataframe['zero'] = 0

        dataframe['cp'] = dataframe['ha_close'].ewm(span=int(cycle_period)).mean()
        dataframe['h0'] = dataframe['ha_close'].ewm(span=int(harmonics[0])).mean()
        dataframe['h1'] = dataframe['ha_close'].ewm(span=int(harmonics[1])).mean()
        dataframe['h2'] = dataframe['ha_close'].ewm(span=int(harmonics[2])).mean()

        # Apply rolling window operation to the 'OHLC4' column
        rolling_windowc = dataframe['ha_close'].rolling(cycle_period) 
        rolling_windowh0 = dataframe['ha_close'].rolling(int(harmonics[0]))
        rolling_windowh1 = dataframe['ha_close'].rolling(int(harmonics[1])) 
        rolling_windowh2 = dataframe['ha_close'].rolling(int(harmonics[2])) 

        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_valuec = rolling_windowc.apply(lambda x: np.ptp(x))
        ptp_valueh0 = rolling_windowh0.apply(lambda x: np.ptp(x))
        ptp_valueh1 = rolling_windowh1.apply(lambda x: np.ptp(x))
        ptp_valueh2 = rolling_windowh2.apply(lambda x: np.ptp(x))

        # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['cycle_move'] = ptp_valuec / dataframe['ha_close']
        dataframe['h0_move'] = ptp_valueh0 / dataframe['ha_close']
        dataframe['h1_move'] = ptp_valueh1 / dataframe['ha_close']
        dataframe['h2_move'] = ptp_valueh2 / dataframe['ha_close']

        dataframe['cycle_move_mean'] = dataframe['cycle_move'].rolling(self.cp).mean()        
        dataframe['h0_move_mean'] = dataframe['h0_move'].rolling(self.cp).mean()
        dataframe['h1_move_mean'] = dataframe['h1_move'].rolling(self.cp).mean() 
        dataframe['h2_move_mean'] = dataframe['h2_move'].rolling(self.cp).mean()

        dataframe['max_high_h2'] = dataframe['high'].rolling(self.h2).max()
        dataframe['min_low_h2'] = dataframe['low'].rolling(self.h2).min()
        dataframe['closePos_h2'] = (dataframe['ha_close'] - dataframe['min_low_h2']) / (dataframe['max_high_h2'] - dataframe['min_low_h2'])

        dataframe['max_high_h1'] = dataframe['high'].rolling(self.h1).max()
        dataframe['min_low_h1'] = dataframe['low'].rolling(self.h1).min()
        dataframe['closePos_h1'] = (dataframe['ha_close'] - dataframe['min_low_h1']) / (dataframe['max_high_h1'] - dataframe['min_low_h1'])

        dataframe['max_high_h0'] = dataframe['high'].rolling(self.h0).max()
        dataframe['min_low_h0'] = dataframe['low'].rolling(self.h0).min()
        dataframe['closePos_h0'] = (dataframe['ha_close'] - dataframe['min_low_h0']) / (dataframe['max_high_h0'] - dataframe['min_low_h0'])

        dataframe['max_high_cp'] = dataframe['high'].rolling(self.cp).max()
        dataframe['min_low_cp'] = dataframe['low'].rolling(self.cp).min()
        dataframe['closePos_cp'] = (dataframe['ha_close'] - dataframe['min_low_cp']) / (dataframe['max_high_cp'] - dataframe['min_low_cp'])

        dataframe['candle_size'] = abs((dataframe['high'] - dataframe['low']) / dataframe['low'])
        dataframe['candle_size_lower'] = dataframe['candle_size'].rolling(self.h2).mean()
        dataframe['candle_size_upper'] = dataframe['candle_size_lower'] * 2.312
        dataframe['candle_size_target'] = dataframe['candle_size_lower'] * 1.618

        kalman_estimate, kalman_velocity = self.custom_kalman_filter(
            dataframe["close"], self.q_noise.value, self.r_noise.value
        )
        dataframe["kalman_estimate"] = kalman_estimate 
        dataframe["kalman_velocity"] = kalman_velocity / kalman_estimate
        minima, maxima = calculate_minima_maxima(dataframe, self.cp)
        dataframe['ex_up'] = self.exclusion_zone.value
        dataframe['ex_dn'] = -self.exclusion_zone.value

        dataframe["minima"] = minima
        dataframe["maxima"] = maxima

        dataframe["volume_sma"] = dataframe["volume"].rolling(window=self.h2).mean()
        # Calculate ATR and RSI
        dataframe["rsi"] = dataframe.ta.rsi(length=self.h2)

        if not self.dp.runmode.value in ("backtest", "plot", "hyperopt"):
            logger.info(f'{pair} - DC: {cycle_period:.2f} | 1/2: {harmonics[0]:.2f} | 1/3: {harmonics[1]:.2f} | 1/4: {harmonics[2]:.2f}')

        return dataframe



    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:

        dataframe.loc[
            (
                (dataframe["kalman_estimate"] < dataframe['close']) & 
                (dataframe["kalman_velocity"] < -self.exclusion_zone.value) & 
                (dataframe["minima"] == 1) & 
                (dataframe['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['enter_long', 'enter_tag']] = (1, 'Kal Min')
            

        if self.can_short == True:
            dataframe.loc[
                (
                    (dataframe["kalman_estimate"] > dataframe['close']) & 
                    (dataframe["kalman_velocity"] > self.exclusion_zone.value) & 
                    (dataframe["maxima"] == 1) & 
                    (dataframe['volume'] > 0)   # Make sure Volume is not 0

                ),
                ['enter_short', 'enter_tag']] = (1, 'Kal Max')
            

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe.loc[
                (
                    (dataframe["kalman_estimate"] > dataframe['close']) &  
                    (dataframe["kalman_velocity"] > self.exclusion_zone.value) & 
                    (dataframe["maxima"] == 1) & 
                    (dataframe['volume'] > 0)   # Make sure Volume is not 0

                ),
                ['exit_long', 'exit_tag']] = (1, 'Kal Max')

        if self.can_short == True:
            dataframe.loc[
                (
                    (dataframe["kalman_estimate"] < dataframe['close']) & 
                    (dataframe["kalman_velocity"] < -self.exclusion_zone.value) & 
                    (dataframe["minima"] == 1) & 
                    (dataframe['volume'] > 0)   # Make sure Volume is not 0

                ),
                ['exit_short', 'exit_tag']] = (1, 'Kal Min')

        return dataframe

    def leverage(self, pair: str, current_time, current_rate: float,
                proposed_leverage: float, max_leverage: float, side: str,
                **kwargs) -> float:
        return 3.0


def perform_fft(price_data, window_size=None):
    if window_size is not None:
        # Apply rolling window to smooth the data
        price_data = price_data.rolling(window=window_size, center=True).mean().dropna()

    normalized_data = (price_data - np.mean(price_data)) / np.std(price_data)
    n = len(normalized_data)
    fft_data = np.fft.fft(normalized_data)
    freq = np.fft.fftfreq(n)
    power = np.abs(fft_data) ** 2
    power[np.isinf(power)] = 0
    return freq, power

def calculate_minima_maxima(df, window):
    if df is None or df.empty:
        return np.zeros(0), np.zeros(0)  # Return empty arrays instead of None

    minima = np.zeros(len(df))
    maxima = np.zeros(len(df))

    for i in range(window, len(df)):  # Ensure index does not go out of bounds
        window_data = df['kalman_estimate'].iloc[i - window:i + 1]

        if df['kalman_estimate'].iloc[i] == window_data.min() and (window_data == df['kalman_estimate'].iloc[i]).sum() == 1:
            minima[i] = 1
        if df['kalman_estimate'].iloc[i] == window_data.max() and (window_data == df['kalman_estimate'].iloc[i]).sum() == 1:
            maxima[i] = 1

    return minima, maxima