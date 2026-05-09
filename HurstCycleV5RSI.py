import math
import numpy as np
from freqtrade.strategy import IStrategy
import pandas as pd
import talib as ta
from scipy.fft import fft
from technical import qtpylib
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
import pandas_ta as pta
import logging
import warnings
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)


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


class HurstCycleV5RSI(IStrategy):
    timeframe = '1h'
    minimal_roi = {"0": 1}
    locked_stoploss = {}
    stoploss = -0.99
    trailing_stop = False
    trailing_stop_positive = 0.015
    can_short = False
    use_custom_stoploss = True
    ignore_roi_if_entry_signal = True

    base_cycle_period = 20
    filter_weights = [1, 2, 4, 8, 4]
    # New parameter for convergence threshold (tunable)
    convergence_threshold = 0.005  # 0.5% of price as max spread

    ### Hyperoptable parameters ###
    u_window_size = IntParameter(70, 120, default=104, space='buy', optimize=True, load=True)
    l_window_size = IntParameter(35, 50, default=42, space='buy', optimize=True, load=True)
    rsi_period = IntParameter(7, 21, default=21, space="buy", optimize=True, load=True)
    buy_thres = DecimalParameter(low=0.05, high=0.2, default=0.12, decimals=2 ,space='buy', optimize=True, load=True)
    sell_thres = DecimalParameter(low=0.8, high=0.95, default=0.95, decimals=2 ,space='sell', optimize=True, load=True)
    rsibuy_thres = DecimalParameter(low=35, high=55, default=52.8, decimals=1 ,space='buy', optimize=True, load=True)
    rsisell_thres = DecimalParameter(low=55, high=85, default=71.4, decimals=1 ,space='sell', optimize=True, load=True)

    # DCA
    position_adjustment_enable = True
    max_epa = IntParameter(1, 3, default = 1 ,space='buy', optimize=True, load=True) # of additional buys.
    filldelay = IntParameter(100, 300, default = 100 ,space='buy', optimize=True, load=True)

    # trailing stoploss enables
    level1 = DecimalParameter(low=0.5, high=0.8, default=0.69, decimals=2 ,space='sell', optimize=True, load=True)
    level0 = DecimalParameter(low=0.4, high=0.7, default=0.59, decimals=2 ,space='sell', optimize=True, load=True)

    # negative stoploss
    use_stop1 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    use_stop2 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    use_stop3 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    use_stop4 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    # roi
    time0 = IntParameter(low=1440, high=2600, default=1440, space='sell', optimize=True, load=True)
    time1 = IntParameter(low=1440, high=2600, default=2000, space='sell', optimize=True, load=True)
    time2 = IntParameter(low=2600, high=4000, default=3200, space='sell', optimize=True, load=True)
    time3 = IntParameter(low=2500, high=5000, default=4500, space='sell', optimize=True, load=True)  

    # Logic Selection
    use0 = BooleanParameter(default=False, space="sell", optimize=True, load=True)
    use1 = BooleanParameter(default=False, space="sell", optimize=True, load=True)
    use2 = BooleanParameter(default=True, space="sell", optimize=True, load=True)

    plot_config = {
            "main_plot": {
            "vtl_up": {
              "color": "#05db83"
            },
            "vtl_down": {
              "color": "#9e209a",
              "type": "line"
            },
            "rollingMin": {
              "color": "green",
              "type": "line"
            },
            "rollingMax": {
              "color": "red",
              "type": "line"
            },
            "lr_mid": {
              "color": "#3075b5",
              "type": "line"
            },
            "lr_upper": {
              "color": "#c1354c",
              "type": "line"
            },
            "lr_lower": {
              "color": "#21fd3c",
              "type": "line"
            }
            },
            "subplots": {
            "trend": {
              "trend_location": {
                "color": "#1761bb"
              }
            },
            "slope": {
              "vtl_up_slope": {
                "color": "#398820"
              },
              "vtl_dn_slope": {
                "color": "#3638dd"
              }
            }
          }
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

        TP0 = current_candle['h2_move_mean'] 
        TP1 = current_candle['h1_move_mean'] 
        TP2 = current_candle['h0_move_mean'] 
        TP3 = current_candle['cycle_move_mean']
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
        if current_profit > TP2 and trade.nr_of_successful_exits == 0 and self.use0.value == True:
            # Take quarter of the profit at next fib%%
            return -(trade.stake_amount / 2)
        if current_profit > TP3 and trade.nr_of_successful_exits == 1 and self.use1.value == True:
            # Take half of the profit at last fib%%
            return -(trade.stake_amount / 2)
        if current_profit > (TP3 * 1.5) and trade.nr_of_successful_exits == 2:
            # Take half of the profit at last fib%%
            return -(trade.stake_amount / 2)
        if current_profit > (TP3 * 2.0) and trade.nr_of_successful_exits == 3:
            # Take profit at last fib%%
            return -(trade.stake_amount)
        # Take Profit Early if DCA was used 
        if current_profit > TP1 and trade.nr_of_successful_exits == 0 and count_of_entries == 2 and self.use2.value == True:
            # Take half of the profit at next fib%%
            return -(trade.stake_amount / 2)
        if current_profit > TP2 and trade.nr_of_successful_exits == 1 and count_of_entries == 2 and self.use2.value == True:
            # Take profit at last fib%%
            return -(trade.stake_amount)
            
        if trade.nr_of_successful_entries == self.max_epa.value + 1:
            return None 
        # Block concurrent buys and Hold otherwise if the dip is not large enough
        if current_profit > -TP1:
            return None

        try:
            # This returns first order stake size 
            # Modify the following parameters to enable more levels or different buy size:
            # max_entry_position_adjustment = 3 
            # max_dca_multiplier = 3.5 

            stake_amount = filled_entries[0].cost
            # This then calculates current safety order size when a secondary buy signal is generated.
            if (last_fill > self.filldelay.value):
                if (signal == 1 and current_profit < -TP0):
                    if count_of_entries >= 1: 
                        stake_amount = stake_amount * 2
                    else:
                        stake_amount = stake_amount

                    return stake_amount

            # This then calculates current safety order size when below -Take Profit 1.
            if (last_fill > self.filldelay.value):
                if current_profit < -TP1:
                    if count_of_entries >= 1: 
                        stake_amount = stake_amount * 1.5
                    else:
                        stake_amount = stake_amount

                    return stake_amount

            # This accommadates a one shot at buying the dip on a big wick with one 
            # large buy if the funds are available...
            if (last_fill > self.filldelay.value):
                if (current_profit < -TP3):
                    if count_of_entries == 1: 
                        stake_amount = stake_amount * 4
                    else:
                        stake_amount = stake_amount

                    return stake_amount

        except Exception as exception:
            return None

        return None


    ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        SLT0 = current_candle['h2_move_mean']
        SLT1 = current_candle['h1_move_mean']
        SLT2 = current_candle['h0_move_mean']
        SLT3 = current_candle['cycle_move_mean']
        enable = current_candle['trend_location']
        
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
        elif SLT1 is not None and current_profit > SLT1 and enable < self.level1.value:
            new_stoploss = (SLT1 - SLT0)
            level = 2
        elif SLT0 is not None and current_profit > SLT0 and enable < self.level0.value:
            new_stoploss = (SLT1 - SLT0)
            level = 1

        if new_stoploss is not None:
            if pair not in self.locked_stoploss or new_stoploss > self.locked_stoploss[pair]:
                self.locked_stoploss[pair] = new_stoploss
                self.dp.send_msg(f'*** {pair} *** Profit {level} {display_profit:.3f}%% - New stoploss: {new_stoploss:.4f} activated')
                logger.info(f'*** {pair} *** Profit {level} {display_profit:.3f}%% - New stoploss: {new_stoploss:.4f} activated')
            return self.locked_stoploss[pair]

        return self.stoploss

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
    
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        trade_duration = (current_time - trade.open_date_utc).seconds / 60
        last_fill = (current_time - trade.date_last_filled_utc).seconds / 60 

        current_candle = dataframe.iloc[-1].squeeze()

        TP0 = current_candle['h2_move_mean'] 
        TP1 = current_candle['h1_move_mean']
        TP2 = current_candle['h0_move_mean']
        TP3 = current_candle['cycle_move_mean']

        ### roi ###
        if current_profit > TP3 and trade_duration > self.time0.value:
            return 'Roi 0 - Easy $$$'
        if current_profit > TP2 and trade_duration > self.time1.value:
            return 'Roi 1 - ol reliable' 
        if current_profit > TP1 and trade_duration > self.time2.value:
            return 'Roi 2 - Avg Joe'
        if current_profit > TP0 and trade_duration > self.time3.value:
            return 'Roi 3 - Better than Nothing'

        ### negative stoploss ###
        if current_profit < -TP3 and self.use_stop1.value == True:
            return 'Failsafe 3 - REKTd'
        if current_profit < -TP2 and self.use_stop2.value == True and self.max_epa.value < 2:
            return 'Failsafe 2 - Ooo that hurts'
        if current_profit < -TP1 and self.use_stop3.value == True and self.max_epa.value < 1:
            return 'Failsafe 1 - Leverage is risky'
        if current_profit < -TP0 and self.use_stop4.value == True and self.max_epa.value < 1:
            return 'Failsafe 0 - Wasnt a good idea...'

        return False

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

        dataframe['rollingMin'] = dataframe['close'].rolling(5).min()
        dataframe['rollingMax'] = dataframe['close'].rolling(5).max()
        # Numerical Filter
        close = dataframe['ha_close'].values
        weights = np.array(self.filter_weights) / sum(self.filter_weights)
        filtered = np.convolve(close, weights, mode='valid')
        dataframe['filtered_close'] = pd.Series(filtered, index=dataframe.index[-len(filtered):])
        # channel = self.linear_regression_channel(dataframe['ha_close'], window=self.h2, num_dev=1.618)
        # dataframe['lr_mid'] = channel['mid']
        # dataframe['lr_upper'] = channel['upper']
        # dataframe['lr_lower'] = channel['lower']

        # # FLDs
        # half_short = math.ceil(self.h2 / 2)
        # half_mid = math.ceil(self.h0 / 2)
        # half_long = math.ceil(self.cp / 2)
        # fld_short_base = dataframe['filtered_close'].rolling(window=self.h2, center=True).mean()
        # fld_mid_base = dataframe['filtered_close'].rolling(window=self.h0, center=True).mean()
        # fld_long_base = dataframe['filtered_close'].rolling(window=self.cp, center=True).mean()
        # dataframe['fld_short'] = fld_short_base.shift(half_short)
        # dataframe['fld_mid'] = fld_mid_base.shift(half_mid)
        # dataframe['fld_long'] = fld_long_base.shift(half_long)
        # for fld in ['fld_short', 'fld_mid', 'fld_long']:
        #     last_valid_idx = dataframe[fld].last_valid_index()
        #     if last_valid_idx is not None:
        #         last_valid_row = dataframe.index.get_loc(last_valid_idx)
        #         if last_valid_row < len(dataframe) - 1 and last_valid_row > 0:
        #             for i in range(last_valid_row + 1, len(dataframe)):
        #                 prev_slope = (dataframe[fld].iloc[last_valid_row] - 
        #                               dataframe[fld].iloc[last_valid_row - 1])
        #                 dataframe[fld].iloc[i] = dataframe[fld].iloc[last_valid_row] + \
        #                                          prev_slope * (i - last_valid_row)

        # # Convergence-Based Agreeance Band
        # # Calculate spread between FLDs
        # fld_spread = dataframe[['fld_short', 'fld_mid', 'fld_long']].max(axis=1) - \
        #              dataframe[['fld_short', 'fld_mid', 'fld_long']].min(axis=1)
        # # Relative spread as a percentage of current price
        # relative_spread = fld_spread / dataframe['filtered_close']
        
        # # Define convergence condition
        # dataframe['converging'] = relative_spread < self.convergence_threshold
        
        # # Agreeance Band: Mean of FLDs ± a small buffer when converging
        # band_center = dataframe[['fld_short', 'fld_mid', 'fld_long']].mean(axis=1)
        # band_width = dataframe['filtered_close'] * 0.01  # 1% of price as buffer
        # dataframe['agreeance_upper'] = np.where(dataframe['converging'],
        #                                        band_center + band_width,
        #                                        np.nan)
        # dataframe['agreeance_lower'] = np.where(dataframe['converging'],
        #                                        band_center - band_width,
        #                                        np.nan)

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


        # Forward fill NaN values with the last valid value
        dataframe['vtl_up'] = dataframe['vtl_up'].fillna(method='ffill')
        dataframe['vtl_down'] = dataframe['vtl_down'].fillna(method='ffill')
        dataframe['vtl-spread'] = dataframe['vtl_down'] - dataframe['vtl_up']

        # Then calculate slopes
        dataframe['vtl_up_slope'] = dataframe['vtl_up'].diff()
        dataframe['vtl_dn_slope'] = dataframe['vtl_down'].diff()
        dataframe['vtl_trend'] = dataframe['vtl_dn_slope'] - dataframe['vtl_up_slope']

        dataframe['trend_location'] = (dataframe['filtered_close'] - dataframe['vtl_up']) / (dataframe['vtl_down'] - dataframe['vtl_up'])

        # # Slopes for cycle sync
        # dataframe['fld_short_slope'] = dataframe['fld_short'].diff()
        # dataframe['fld_mid_slope'] = dataframe['fld_mid'].diff()
        # dataframe['fld_long_slope'] = dataframe['fld_long'].diff()

        # Trend
        dataframe['trend'] = np.where(dataframe['filtered_close'] > dataframe['cp'], 1, -1)
        dataframe['rsi'] = pta.rsi(dataframe['close'], length=self.rsi_period.value)
        dataframe['rsi_fast'] = pta.rsi(dataframe['close'], length=4)

        regression_channel = self.linear_regression_channel(dataframe['rsi_fast'], window=self.h2, num_dev=1.0)

        dataframe['lr_mid_rsi'] = regression_channel['mid']
        dataframe['lr_upper_rsi'] = regression_channel['upper']
        dataframe['lr_lower_rsi'] = regression_channel['lower']
        
        dataframe['cci'] = ta.CCI(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=160)
        
        dataframe['norm_cci'] = (dataframe['cci'] - dataframe['cci'].rolling(60).mean()) / dataframe['cci'].rolling(60).std()
        dataframe['zero'] = 0 
        dataframe['one'] = 1
        

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        
        dataframe.loc[
            (
                ~(dataframe['vtl_up'] > dataframe['vtl_down']) &
                (dataframe['trend_location'] < self.buy_thres.value) &
                (dataframe['lr_mid_rsi'] < self.rsibuy_thres.value) &
                (dataframe['close'].rolling(5).min() < dataframe['vtl_up']) &
                (dataframe['close'] > dataframe['vtl_up'])
            ),
            ['enter_long', 'enter_tag']] = (1, 'Minima Full Send')


        if self.can_short == True:
            dataframe.loc[
                ~(dataframe['vtl_up'] > dataframe['vtl_down']) &
                (dataframe['trend_location'] > self.sell_thres.value) &
                (dataframe['close'].rolling(5).max() > dataframe['vtl_down']) &
                (dataframe['close'] < dataframe['vtl_down']),
                'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # dataframe.loc[
        #     (dataframe['trend_location'] > self.sell_thres.value) &
        #     (dataframe['close'] > dataframe['vtl_down']), 
        #     'exit_long'] = 1

        # if self.can_short == True:
        #     dataframe.loc[
        #         (dataframe['trend_location'] < self.buy_thres.value) &
        #         (dataframe['close'] < dataframe['vtl_up']), 
        #         'exit_short'] = 1

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
