import numpy as np
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.fft import fft
import pandas_ta as pta
import talib as ta
from technical import qtpylib
from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, RealParameter)
import logging
import warnings

logger = logging.getLogger(__name__)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)

class HEW(IStrategy):
    """
    Hurst Elliott Wave (HEW) Strategy for Freqtrade
    Combines Hurst cycle analysis with Elliott Wave pattern detection and VTLs
    """
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
    convergence_threshold = 0.005

    # Hyperoptable parameters
    u_window_size = IntParameter(70, 120, default=104, space='buy', optimize=True, load=True)
    l_window_size = IntParameter(35, 50, default=42, space='buy', optimize=True, load=True)
    rsi_period = IntParameter(7, 21, default=21, space="buy", optimize=True, load=True)
    buy_thres = DecimalParameter(low=0.05, high=0.2, default=0.12, decimals=2, space='buy', optimize=True, load=True)
    sell_thres = DecimalParameter(low=0.8, high=0.95, default=0.95, decimals=2, space='sell', optimize=True, load=True)
    rsibuy_thres = DecimalParameter(low=35, high=55, default=52.8, decimals=1, space='buy', optimize=True, load=True)
    rsisell_thres = DecimalParameter(low=55, high=85, default=71.4, decimals=1, space='sell', optimize=True, load=True)

    # DCA
    position_adjustment_enable = True
    max_epa = IntParameter(1, 3, default=1, space='buy', optimize=True, load=True)
    filldelay = IntParameter(100, 300, default=100, space='buy', optimize=True, load=True)

    # Trailing stoploss enables
    level1 = DecimalParameter(low=0.5, high=0.8, default=0.69, decimals=2, space='sell', optimize=True, load=True)
    level0 = DecimalParameter(low=0.4, high=0.7, default=0.59, decimals=2, space='sell', optimize=True, load=True)

    # Negative stoploss
    use_stop1 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    use_stop2 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    use_stop3 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    use_stop4 = BooleanParameter(default=False, space="protection", optimize=True, load=True)

    # ROI
    time0 = IntParameter(low=1440, high=2600, default=1440, space='sell', optimize=True, load=True)
    time1 = IntParameter(low=1440, high=2600, default=2000, space='sell', optimize=True, load=True)
    time2 = IntParameter(low=2600, high=4000, default=3200, space='sell', optimize=True, load=True)
    time3 = IntParameter(low=2500, high=5000, default=4500, space='sell', optimize=True, load=True)

    # Logic Selection
    use0 = BooleanParameter(default=False, space="sell", optimize=True, load=True)
    use1 = BooleanParameter(default=False, space="sell", optimize=True, load=True)
    use2 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    buy_thres = DecimalParameter(0.1, 0.5, default=0.3, space="buy", optimize=True)
    rsibuy_thres = DecimalParameter(30.0, 60.0, default=55.0, space="buy", optimize=True)
    sell_thres = DecimalParameter(0.5, 0.9, default=0.7, space="sell", optimize=True)

    plot_config = {
        'main_plot': {
            'ha_close': {'color': 'blue'},
            'vtl_up': {'color': 'green'},
            'vtl_down': {'color': 'red'},
        },
        'subplots': {
            'RSI': {
                'rsi': {'color': 'purple'},
                'lr_mid_rsi': {'color': 'orange'},
            },
            'CCI': {
                'norm_cci': {'color': 'green'},
            },
            'Signals': {
                'enter_long': {'color': 'green', 'type': 'scatter', 'mode': 'markers', 'symbol': 'triangle-up', 'size': 12},
                'enter_short': {'color': 'red', 'type': 'scatter', 'mode': 'markers', 'symbol': 'triangle-down', 'size': 12},
                'exit_long': {'color': 'blue', 'type': 'scatter', 'mode': 'markers', 'symbol': 'circle', 'size': 10},
                'exit_short': {'color': 'purple', 'type': 'scatter', 'mode': 'markers', 'symbol': 'circle', 'size': 10},
                'pattern_marker': {'color': 'orange', 'type': 'scatter', 'mode': 'markers+text', 'textposition': 'top', 'size': 8},
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
        dcaDisp = display_profit + tp1
        if display_profit < 0:
            dca_str = f'| DCA Dist: {dcaDisp:.3}% | Last Fill: {last_fill:.5}m"' 

        if current_candle['enter_long'] is not None:
            signal = current_candle['enter_long']

        if current_profit is not None:
            logger.info(f"{trade.pair} - Current Profit: {display_profit:.3}% # of Entries: {trade.nr_of_successful_entries} {dca_str}")
            logger.info(f"{trade.pair} - TP0: {tp0:.3}% | TP1: {tp1:.3}% | TP2: {tp2:.3}% | TP3: {tp3:.3}%")

        if current_profit > TP2 and trade.nr_of_successful_exits == 0 and self.use0.value == True:
            return -(trade.stake_amount / 2)
        if current_profit > TP3 and trade.nr_of_successful_exits == 1 and self.use1.value == True:
            return -(trade.stake_amount / 2)
        if current_profit > (TP3 * 1.5) and trade.nr_of_successful_exits == 2:
            return -(trade.stake_amount / 2)
        if current_profit > (TP3 * 2.0) and trade.nr_of_successful_exits == 3:
            return -(trade.stake_amount)
        if current_profit > TP1 and trade.nr_of_successful_exits == 0 and count_of_entries == 2 and self.use2.value == True:
            return -(trade.stake_amount / 2)
        if current_profit > TP2 and trade.nr_of_successful_exits == 1 and count_of_entries == 2 and self.use2.value == True:
            return -(trade.stake_amount)
            
        if trade.nr_of_successful_entries == self.max_epa.value + 1:
            return None 
        if current_profit > -TP1:
            return None

        try:
            stake_amount = filled_entries[0].cost
            if (last_fill > self.filldelay.value):
                if (signal == 1 and current_profit < -TP0):
                    if count_of_entries >= 1: 
                        stake_amount = stake_amount * 2
                    else:
                        stake_amount = stake_amount
                    return stake_amount
            if (last_fill > self.filldelay.value):
                if current_profit < -TP1:
                    if count_of_entries >= 1: 
                        stake_amount = stake_amount * 1.5
                    else:
                        stake_amount = stake_amount
                    return stake_amount
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

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
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

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, 
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        trade_duration = (current_time - trade.open_date_utc).seconds / 60

        TP0 = current_candle['h2_move_mean'] 
        TP1 = current_candle['h1_move_mean']
        TP2 = current_candle['h0_move_mean']
        TP3 = current_candle['cycle_move_mean']

        # ROI exits
        if current_profit > TP3 and trade_duration > self.time0.value:
            return 'Roi 0 - Easy $$$'
        if current_profit > TP2 and trade_duration > self.time1.value:
            return 'Roi 1 - ol reliable' 
        if current_profit > TP1 and trade_duration > self.time2.value:
            return 'Roi 2 - Avg Joe'
        if current_profit > TP0 and trade_duration > self.time3.value:
            return 'Roi 3 - Better than Nothing'

        # Negative stoploss
        if current_profit < -TP3 and self.use_stop1.value == True:
            return 'Failsafe 3 - REKTd'
        if current_profit < -TP2 and self.use_stop2.value == True and self.max_epa.value < 2:
            return 'Failsafe 2 - Ooo that hurts'
        if current_profit < -TP1 and self.use_stop3.value == True and self.max_epa.value < 1:
            return 'Failsafe 1 - Leverage is risky'
        if current_profit < -TP0 and self.use_stop4.value == True and self.max_epa.value < 1:
            return 'Failsafe 0 - Wasnt a good idea...'

        # Wave-based exit
        if current_candle['wave_exit_long'] == 1:
            return 'Wave Exit - Pattern Complete'

        return None

    def perform_fft(self, price_data, window_size=None):
        if window_size is not None:
            price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
        normalized_data = (price_data - np.mean(price_data)) / np.std(price_data)
        n = len(normalized_data)
        fft_data = fft(normalized_data)
        freq = np.fft.fftfreq(n)
        power = np.abs(fft_data) ** 2
        power[np.isinf(power)] = 0
        return freq, power

    def linear_regression_channel(self, data: pd.Series, window: int, num_dev: float):
        lr_channel = {'mid': [], 'upper': [], 'lower': []}
        for i in range(window, len(data)):
            y = data[i-window:i]
            x = np.arange(window)
            A = np.vstack([x, np.ones(len(x))]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            y_line = intercept + slope * x[-1]
            residuals = y - (intercept + slope * x)
            std_dev = np.std(residuals)
            lr_channel['mid'].append(y_line)
            lr_channel['upper'].append(y_line + num_dev * std_dev)
            lr_channel['lower'].append(y_line - num_dev * std_dev)
        lr_channel['mid'] = [np.nan] * window + lr_channel['mid']
        lr_channel['upper'] = [np.nan] * window + lr_channel['upper']
        lr_channel['lower'] = [np.nan] * window + lr_channel['lower']
        return pd.DataFrame(lr_channel)

    def check_min_wave_length(self, wave_points, dataframe):
        """
        Log segment lengths without enforcing minimum length.
        """
        if self.h2 is None:
            logger.info("No h2 cycle period defined, skipping wave length check")
            return True
        bar_lengths = []
        for i in range(1, len(wave_points)):
            start_loc = dataframe.index.get_loc(wave_points[i-1])
            end_loc = dataframe.index.get_loc(wave_points[i])
            length = end_loc - start_loc
            bar_lengths.append(length)
        logger.info(f"Wave segment lengths: {bar_lengths} (h2={self.h2})")
        return True

    def validate_wave_with_fib(self, wave_lengths, pattern_type='impulse'):
        if pattern_type == 'impulse' and len(wave_lengths) == 5:
            w1, w2, w3, w4, w5 = wave_lengths
            if not (0.2 <= abs(w2)/w1 <= 1.0):
                logger.info(f"Impulse rejected: Wave 2 retrace {abs(w2)/w1:.3f} not in [0.2, 1.0]")
                return False
            if w3 < 0.3 * w1:
                logger.info(f"Impulse rejected: Wave 3 {w3:.3f} shorter than 0.3*Wave 1 {0.3*w1:.3f}")
                return False
            if not (0.2 <= abs(w4)/w3 <= 0.9):
                logger.info(f"Impulse rejected: Wave 4 retrace {abs(w4)/w3:.3f} not in [0.2, 0.9]")
                return False
            if w5 > w3 or w5 < 0.3 * w1:
                logger.info(f"Impulse rejected: Wave 5 {w5:.3f} invalid (too long vs Wave 3 {w3:.3f} or too short vs 0.3*Wave 1 {0.3*w1:.3f})")
                return False
            logger.info(f"Impulse pattern validated with Fibonacci ratios: w1={w1:.3f}, w2={w2:.3f}, w3={w3:.3f}, w4={w4:.3f}, w5={w5:.3f}")
            return True
        elif pattern_type == 'zigzag' and len(wave_lengths) == 3:
            wa, wb, wc = wave_lengths
            if not (0.2 <= abs(wb)/wa <= 1.0):
                logger.info(f"Zigzag rejected: Wave B retrace {abs(wb)/wa:.3f} not in [0.2, 1.0]")
                return False
            if not (0.2 <= wc/wa <= 3.0):  # Relaxed Wave C range
                logger.info(f"Zigzag rejected: Wave C {wc/wa:.3f} not in [0.2, 3.0] of Wave A")
                return False
            logger.info(f"Zigzag pattern validated with Fibonacci ratios: wa={wa:.3f}, wb={wb:.3f}, wc={wc:.3f}")
            return True
        elif pattern_type == 'flat' and len(wave_lengths) == 3:
            wa, wb, wc = wave_lengths
            if not (0.3 <= wb/wa <= 2.0):
                logger.info(f"Flat rejected: Wave B {wb/wa:.3f} not in [0.3, 2.0] of Wave A")
                return False
            if not (0.3 <= wc/wa <= 2.0):
                logger.info(f"Flat rejected: Wave C {wc/wa:.3f} not in [0.3, 2.0] of Wave A")
                return False
            logger.info(f"Flat pattern validated with Fibonacci ratios: wa={wa:.3f}, wb={wb:.3f}, wc={wc:.3f}")
            return True
        logger.info(f"Invalid pattern type {pattern_type} or wave count {len(wave_lengths)}")
        return False

    def check_vtl_break(self, date, dataframe, direction='up'):
        try:
            atr = pta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=14).iloc[-1]
            if direction == 'up':
                vtl_value = dataframe.loc[date, 'vtl_down']
                close = dataframe.loc[date, 'close']
                threshold = max(vtl_value * 0.15, atr * 1.5, 0.03)  # 15% or 1.5*ATR or 0.03
                result = close > (vtl_value - threshold)
                logger.info(f"VTL check (up) at {date}: close={close:.4f}, vtl_down={vtl_value:.4f}, threshold={threshold:.4f}, break={result}, diff={close-vtl_value:.4f}, atr={atr:.4f}")
                return result
            else:
                vtl_value = dataframe.loc[date, 'vtl_up']
                close = dataframe.loc[date, 'close']
                threshold = max(vtl_value * 0.15, atr * 1.5, 0.03)  # 15% or 1.5*ATR or 0.03
                result = close < (vtl_value + threshold)
                logger.info(f"VTL check (down) at {date}: close={close:.4f}, vtl_up={vtl_value:.4f}, threshold={threshold:.4f}, break={result}, diff={close-vtl_value:.4f}, atr={atr:.4f}")
                return result
        except KeyError as e:
            logger.error(f"VTL check failed at {date}: {e}")
            return False
        except ValueError as e:
            logger.error(f"VTL check failed at {date}: {e}")
            return False

    def detect_wave_patterns(self, dataframe: pd.DataFrame):
        """
        Detect Elliott Wave patterns and store in DataFrame.
        """
        if len(dataframe) < 200:
            logger.warning(f"Dataframe length {len(dataframe)} too short for reliable wave detection (minimum 200 candles)")
            return []
        extrema = sorted(list(argrelextrema(dataframe['high'].values, np.greater_equal, order=self.h2//2)[0]) +
                         list(argrelextrema(dataframe['low'].values, np.less_equal, order=self.h2//2)[0]))
        extrema = [dataframe.index[i] for i in extrema]
        logger.info(f"Found {len(extrema)} extrema points for wave detection")
        
        patterns = []
        candidate_pattern = None
        dataframe['wave_pattern'] = 'none'
        dataframe['pattern_marker'] = np.nan

        for start in range(len(extrema) - 8):
            impulse_points = extrema[start:start + 6]
            logger.info(f"Checking impulse pattern at start index {start}: {impulse_points}")
            if not self.check_min_wave_length(impulse_points, dataframe):
                continue
            try:
                impulse_prices = dataframe.loc[impulse_points, 'close'].values
                logger.info(f"Impulse prices: {impulse_prices}")
                impulse_diffs = np.diff(impulse_prices)
                impulse_signs = np.sign(impulse_diffs)
            except KeyError as e:
                logger.error(f"Failed to extract impulse prices at {impulse_points}: {e}")
                continue
            
            # Calculate and log fib measurements always
            if len(impulse_diffs) == 5:
                w1, w2, w3, w4, w5 = abs(impulse_diffs)
                logger.info(f"Fib measurements for impulse: w1={w1:.3f}, w2={w2:.3f}, w3={w3:.3f}, w4={w4:.3f}, w5={w5:.3f}, w2/w1={w2/w1:.3f}, w3/w1={w3/w1:.3f}, w4/w3={w4/w3:.3f}, w5/w3={w5/w3:.3f}")
            
            # Calculate VTL threshold (for logging and mapping, not rejection)
            atr = pta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=14).iloc[-1]
            direction = 'down' if list(impulse_signs) == [1, -1, 1, -1, 1] else 'up'
            if direction == 'up':
                vtl_value = dataframe.loc[impulse_points[-1], 'vtl_down']
                threshold = max(vtl_value * 0.15, atr * 1.5, 0.03)
            else:
                vtl_value = dataframe.loc[impulse_points[-1], 'vtl_up']
                threshold = max(vtl_value * 0.15, atr * 1.5, 0.03)
            logger.info(f"VTL threshold for impulse ({direction}): threshold={threshold:.4f}, vtl_value={vtl_value:.4f}, atr={atr:.4f}")

            # Add pattern if signs match (no rejection)
            if list(impulse_signs) == [1, -1, 1, -1, 1]:
                candidate_pattern = {
                    'start': impulse_points[0],
                    'end': impulse_points[-1],
                    'type': 'impulse_up',
                    'waves': impulse_points,
                    'labels': ['1', '2', '3', '4', '5']
                }
                patterns.append(candidate_pattern)
                logger.info(f"Impulse up pattern detected: {impulse_points}")
                # Map signed integers
                signed_labels = [1, -2, 3, -4, 5]
                # Fib ratios for each point
                ratios = [np.nan, w2/w1, w3/w1, w4/w3, w5/w3]
                for i, (idx, label, signed, ratio) in enumerate(zip(impulse_points, ['1', '2', '3', '4', '5'], signed_labels, ratios)):
                    dataframe.loc[idx, 'pattern_marker'] = label
                    dataframe.loc[idx, 'wave_integer'] = signed
                    dataframe.loc[idx, 'fib_ratio'] = ratio
                dataframe.loc[impulse_points[-1], 'wave_pattern'] = 'impulse_up'
                # Store threshold at end point
                dataframe.loc[impulse_points[-1], 'break_threshold'] = threshold
            elif list(impulse_signs) == [-1, 1, -1, 1, -1]:
                candidate_pattern = {
                    'start': impulse_points[0],
                    'end': impulse_points[-1],
                    'type': 'impulse_down',
                    'waves': impulse_points,
                    'labels': ['1', '2', '3', '4', '5']
                }
                patterns.append(candidate_pattern)
                logger.info(f"Impulse down pattern detected: {impulse_points}")
                # Map signed integers
                signed_labels = [-1, 2, -3, 4, -5]
                # Fib ratios for each point
                ratios = [np.nan, w2/w1, w3/w1, w4/w3, w5/w3]
                for i, (idx, label, signed, ratio) in enumerate(zip(impulse_points, ['1', '2', '3', '4', '5'], signed_labels, ratios)):
                    dataframe.loc[idx, 'pattern_marker'] = label
                    dataframe.loc[idx, 'wave_integer'] = signed
                    dataframe.loc[idx, 'fib_ratio'] = ratio
                dataframe.loc[impulse_points[-1], 'wave_pattern'] = 'impulse_down'
                # Store threshold at end point
                dataframe.loc[impulse_points[-1], 'break_threshold'] = threshold

            corr_points = extrema[start:start + 4]
            logger.info(f"Checking correction pattern at start index {start}: {corr_points}")
            if not self.check_min_wave_length(corr_points, dataframe):
                continue
            try:
                corr_prices = dataframe.loc[corr_points, 'close'].values
                logger.info(f"Correction prices: {corr_prices}")
                corr_diffs = np.diff(corr_prices)
                corr_signs = np.sign(corr_diffs)
            except KeyError as e:
                logger.error(f"Failed to extract correction prices at {corr_points}: {e}")
                continue
            
            # Calculate and log fib measurements always
            if len(corr_diffs) == 3:
                wa, wb, wc = abs(corr_diffs)
                logger.info(f"Fib measurements for correction: wa={wa:.3f}, wb={wb:.3f}, wc={wc:.3f}, wb/wa={wb/wa:.3f}, wc/wa={wc/wa:.3f}")
            
            # Calculate VTL threshold (for logging and mapping, not rejection)
            direction = 'up' if list(corr_signs) == [-1, 1, -1] else 'down'
            if direction == 'down':
                vtl_value = dataframe.loc[corr_points[-1], 'vtl_down']
                threshold = max(vtl_value * 0.15, atr * 1.5, 0.03)
            else:
                vtl_value = dataframe.loc[corr_points[-1], 'vtl_up']
                threshold = max(vtl_value * 0.15, atr * 1.5, 0.03)
            logger.info(f"VTL threshold for correction ({direction}): threshold={threshold:.4f}, vtl_value={vtl_value:.4f}, atr={atr:.4f}")

            # Add pattern if signs match (no rejection)
            if list(corr_signs) == [-1, 1, -1]:
                candidate_pattern = {
                    'start': corr_points[0],
                    'end': corr_points[-1],
                    'type': 'zigzag_down',
                    'waves': corr_points,
                    'labels': ['A', 'B', 'C']
                }
                patterns.append(candidate_pattern)
                logger.info(f"Zigzag down pattern detected: {corr_points}")
                # Map signed integers
                signed_labels = [-1, 2, -3]
                # Fib ratios for each point
                ratios = [np.nan, wb/wa, wc/wa]
                for i, (idx, label, signed, ratio) in enumerate(zip(corr_points, ['A', 'B', 'C'], signed_labels, ratios)):
                    dataframe.loc[idx, 'pattern_marker'] = label
                    dataframe.loc[idx, 'wave_integer'] = signed
                    dataframe.loc[idx, 'fib_ratio'] = ratio
                dataframe.loc[corr_points[-1], 'wave_pattern'] = 'zigzag_down'
                # Store threshold at end point
                dataframe.loc[corr_points[-1], 'break_threshold'] = threshold
                # Repeat for flat_down
                candidate_pattern = {
                    'start': corr_points[0],
                    'end': corr_points[-1],
                    'type': 'flat_down',
                    'waves': corr_points,
                    'labels': ['A', 'B', 'C']
                }
                patterns.append(candidate_pattern)
                logger.info(f"Flat down pattern detected: {corr_points}")
                # Same mapping as zigzag_down
                for i, (idx, label, signed, ratio) in enumerate(zip(corr_points, ['A', 'B', 'C'], signed_labels, ratios)):
                    dataframe.loc[idx, 'pattern_marker'] = label
                    dataframe.loc[idx, 'wave_integer'] = signed
                    dataframe.loc[idx, 'fib_ratio'] = ratio
                dataframe.loc[corr_points[-1], 'wave_pattern'] = 'flat_down'
                dataframe.loc[corr_points[-1], 'break_threshold'] = threshold
            elif list(corr_signs) == [1, -1, 1]:
                candidate_pattern = {
                    'start': corr_points[0],
                    'end': corr_points[-1],
                    'type': 'zigzag_up',
                    'waves': corr_points,
                    'labels': ['A', 'B', 'C']
                }
                patterns.append(candidate_pattern)
                logger.info(f"Zigzag up pattern detected: {corr_points}")
                # Map signed integers
                signed_labels = [1, -2, 3]
                # Fib ratios for each point
                ratios = [np.nan, wb/wa, wc/wa]
                for i, (idx, label, signed, ratio) in enumerate(zip(corr_points, ['A', 'B', 'C'], signed_labels, ratios)):
                    dataframe.loc[idx, 'pattern_marker'] = label
                    dataframe.loc[idx, 'wave_integer'] = signed
                    dataframe.loc[idx, 'fib_ratio'] = ratio
                dataframe.loc[corr_points[-1], 'wave_pattern'] = 'zigzag_up'
                # Store threshold at end point
                dataframe.loc[corr_points[-1], 'break_threshold'] = threshold
                # Repeat for flat_up
                candidate_pattern = {
                    'start': corr_points[0],
                    'end': corr_points[-1],
                    'type': 'flat_up',
                    'waves': corr_points,
                    'labels': ['A', 'B', 'C']
                }
                patterns.append(candidate_pattern)
                logger.info(f"Flat up pattern detected: {corr_points}")
                # Same mapping as zigzag_up
                for i, (idx, label, signed, ratio) in enumerate(zip(corr_points, ['A', 'B', 'C'], signed_labels, ratios)):
                    dataframe.loc[idx, 'pattern_marker'] = label
                    dataframe.loc[idx, 'wave_integer'] = signed
                    dataframe.loc[idx, 'fib_ratio'] = ratio
                dataframe.loc[corr_points[-1], 'wave_pattern'] = 'flat_up'
                dataframe.loc[corr_points[-1], 'break_threshold'] = threshold

        patterns = sorted(patterns, key=lambda p: p['start'])
        if patterns:
            latest_pattern = patterns[-1]
            try:
                end_timestamp = dataframe.index[latest_pattern['end']]
                if isinstance(end_timestamp, pd.Timestamp):
                    logger.info(f"Latest pattern detected: {latest_pattern['type']} ending at {latest_pattern['end']} ({end_timestamp.strftime('%Y-%m-%d %H:%M:%S')})")
                else:
                    logger.info(f"Latest pattern detected: {latest_pattern['type']} ending at {latest_pattern['end']} (non-datetime index)")
            except Exception as e:
                logger.error(f"Failed to log timestamp for pattern at {latest_pattern['end']}: {e}")
                logger.info(f"Latest pattern detected: {latest_pattern['type']} ending at {latest_pattern['end']} (timestamp unavailable)")
        else:
            if candidate_pattern:
                logger.info(f"No patterns detected, but candidate pattern found: {candidate_pattern['type']} at {candidate_pattern['waves']}")
            else:
                logger.info("No wave patterns or candidates detected")
        
        return patterns

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata['pair']
        logger.info(f"Dataframe length for {pair}: {len(dataframe)} candles")
        if len(dataframe) < 200:
            logger.warning(f"Dataframe length {len(dataframe)} too short for reliable wave detection (minimum 200 candles)")
            return dataframe

        # Heikin-Ashi Candles
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # FFT for cycles
        freq, power = self.perform_fft(dataframe['ha_close'], window_size=self.u_window_size.value)
        positive_mask = (1 / freq > self.l_window_size.value) & (1 / freq < self.u_window_size.value)
        positive_freqs = freq[positive_mask]
        positive_power = power[positive_mask]
        if len(positive_power) == 0:
            raise ValueError("No positive frequencies meet the filtering criteria.")
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
        logger.info(f"Cycle periods: cp={self.cp}, h0={self.h0}, h1={self.h1}, h2={self.h2}")

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
        dataframe['filtered_close'] = pd.Series(filtered, index=dataframe.index[-len(filtered):]).reindex(dataframe.index).ffill()

        # VTL based on last two troughs/crests
        dataframe['trough'] = dataframe['filtered_close'].rolling(self.h2).min()
        dataframe['crest'] = dataframe['filtered_close'].rolling(self.h2).max()
        dataframe['is_trough'] = np.where(dataframe['filtered_close'] == dataframe['trough'], 1, 0)
        dataframe['is_crest'] = np.where(dataframe['filtered_close'] == dataframe['crest'], 1, 0)

        group_window = self.h2
        trough_groups = []
        crest_groups = []
        current_trough_group = []
        current_crest_group = []
        last_trough_idx = None
        last_crest_idx = None

        for idx in dataframe.index:
            if dataframe.at[idx, 'is_trough'] == 1:
                loc = dataframe.index.get_loc(idx)
                if last_trough_idx is None or (loc - last_trough_idx) <= group_window:
                    current_trough_group.append(idx)
                else:
                    if current_trough_group:
                        prices = [dataframe.at[i, 'filtered_close'] for i in current_trough_group]
                        min_idx = current_trough_group[np.argmin(prices)]
                        trough_groups.append(min_idx)
                    current_trough_group = [idx]
                last_trough_idx = loc
            
            if dataframe.at[idx, 'is_crest'] == 1:
                loc = dataframe.index.get_loc(idx)
                if last_crest_idx is None or (loc - last_crest_idx) <= group_window:
                    current_crest_group.append(idx)
                else:
                    if current_crest_group:
                        prices = [dataframe.at[i, 'filtered_close'] for i in current_crest_group]
                        max_idx = current_crest_group[np.argmax(prices)]
                        crest_groups.append(max_idx)
                    current_crest_group = [idx]
                last_crest_idx = loc

        if current_trough_group:
            prices = [dataframe.at[i, 'filtered_close'] for i in current_trough_group]
            min_idx = current_trough_group[np.argmin(prices)]
            trough_groups.append(min_idx)
        if current_crest_group:
            prices = [dataframe.at[i, 'filtered_close'] for i in current_crest_group]
            max_idx = current_crest_group[np.argmax(prices)]
            crest_groups.append(max_idx)

        logger.info(f"Recent troughs: {trough_groups[-3:]} with prices {[dataframe.at[i, 'filtered_close'] for i in trough_groups[-3:]]}")
        logger.info(f"Recent crests: {crest_groups[-3:]} with prices {[dataframe.at[i, 'filtered_close'] for i in crest_groups[-3:]]}")

        price_min = dataframe['filtered_close'].min()
        price_max = dataframe['filtered_close'].max()

        up_vtl_segments = []
        down_vtl_segments = []

        for i in range(1, len(trough_groups)):
            x1 = dataframe.index.get_loc(trough_groups[i-1])
            y1 = dataframe.at[trough_groups[i-1], 'filtered_close']
            x2 = dataframe.index.get_loc(trough_groups[i])
            y2 = dataframe.at[trough_groups[i], 'filtered_close']
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            intercept = y1 - slope * x1
            up_vtl_segments.append([trough_groups[i-1], None, slope, intercept])
            logger.info(f"Up VTL segment {i}: slope={slope:.6f}, intercept={intercept:.4f}")

        for i in range(1, len(crest_groups)):
            x1 = dataframe.index.get_loc(crest_groups[i-1])
            y1 = dataframe.at[crest_groups[i-1], 'filtered_close']
            x2 = dataframe.index.get_loc(crest_groups[i])
            y2 = dataframe.at[crest_groups[i], 'filtered_close']
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            intercept = y1 - slope * x1
            down_vtl_segments.append([crest_groups[i-1], None, slope, intercept])
            logger.info(f"Down VTL segment {i}: slope={slope:.6f}, intercept={intercept:.4f}")

        dataframe['vtl_up'] = np.nan
        dataframe['vtl_down'] = np.nan

        for idx in dataframe.index:
            current_x = dataframe.index.get_loc(idx)
            current_price = dataframe.at[idx, 'filtered_close']
            
            for i, segment in enumerate(up_vtl_segments):
                start_idx, end_idx, slope, intercept = segment
                start_date = start_idx if isinstance(start_idx, pd.Timestamp) else dataframe.index[start_idx]
                if idx >= start_date and (end_idx is None or idx <= end_idx):
                    vtl_value = slope * current_x + intercept
                    if price_min <= vtl_value <= price_max:
                        dataframe.at[idx, 'vtl_up'] = vtl_value
                        if current_price < vtl_value and i + 1 < len(up_vtl_segments):
                            segment[1] = idx
                    else:
                        if end_idx is None:
                            segment[1] = idx
            
            for i, segment in enumerate(down_vtl_segments):
                start_idx, end_idx, slope, intercept = segment
                start_date = start_idx if isinstance(start_idx, pd.Timestamp) else dataframe.index[start_idx]
                if idx >= start_date and (end_idx is None or idx <= end_idx):
                    vtl_value = slope * current_x + intercept
                    if price_min <= vtl_value <= price_max:
                        dataframe.at[idx, 'vtl_down'] = vtl_value
                        if current_price > vtl_value and i + 1 < len(down_vtl_segments):
                            segment[1] = idx
                    else:
                        if end_idx is None:
                            segment[1] = idx

        dataframe['vtl_up'] = dataframe['vtl_up'].fillna(method='ffill')
        dataframe['vtl_down'] = dataframe['vtl_down'].fillna(method='ffill')
        dataframe['vtl-spread'] = dataframe['vtl_down'] - dataframe['vtl_up']
        dataframe['vtl_up_slope'] = dataframe['vtl_up'].diff()
        dataframe['vtl_dn_slope'] = dataframe['vtl_down'].diff()
        dataframe['vtl_trend'] = dataframe['vtl_dn_slope'] - dataframe['vtl_up_slope']
        dataframe['trend_location'] = (dataframe['filtered_close'] - dataframe['vtl_up']) / (dataframe['vtl_down'] - dataframe['vtl_up'])
        logger.info(f"Last VTL values: up={dataframe['vtl_up'].iloc[-1]:.4f}, down={dataframe['vtl_down'].iloc[-1]:.4f}")

        # Trend and indicators
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

        # Detect wave patterns
        self.detect_wave_patterns(dataframe)
        dataframe['is_down_pattern'] = dataframe['wave_pattern'].isin(['zigzag_down', 'flat_down']).astype(int)
        dataframe['is_up_pattern'] = dataframe['wave_pattern'].isin(['zigzag_up', 'flat_up']).astype(int)
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = ''
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Check for recent down correction patterns
        recent_down_patterns = dataframe['is_down_pattern'].rolling(window=20, min_periods=1).max().fillna(0)
        
        base_long_conditions = (
            # ~(dataframe['vtl_up'] > dataframe['vtl_down']) &
            # (dataframe['trend_location'] < self.buy_thres.value) &
            # (dataframe['lr_mid_rsi'] < self.rsibuy_thres.value) &
            # (dataframe['close'].rolling(5).min() < dataframe['vtl_up']) &
            # (dataframe['close'] > dataframe['vtl_up']) &
            (recent_down_patterns == 1)
        )
        long_signal = base_long_conditions
        if long_signal.any():
            latest_signal_idx = dataframe.index[long_signal][-1]
            logger.info(f"Long entry signal triggered for {metadata['pair']} at {latest_signal_idx} ({latest_signal_idx.strftime('%Y-%m-%d %H:%M:%S') if isinstance(latest_signal_idx, pd.Timestamp) else 'non-datetime index'}) with wave_pattern={dataframe['wave_pattern'][latest_signal_idx]}, trend_location={dataframe['trend_location'][latest_signal_idx]:.4f}, lr_mid_rsi={dataframe['lr_mid_rsi'][latest_signal_idx]:.4f}, buy_thres={self.buy_thres.value}, rsibuy_thres={self.rsibuy_thres.value}")
            signal_indices = dataframe.index.get_loc(latest_signal_idx) + np.arange(5)
            signal_indices = signal_indices[signal_indices < len(dataframe)]
            dataframe.iloc[signal_indices, dataframe.columns.get_loc('enter_long')] = 1
            dataframe.iloc[signal_indices, dataframe.columns.get_loc('enter_tag')] = 'Wave Long - Correction Down Complete'
        else:
            logger.info(f"No long entry signal for {metadata['pair']}: trend_location={dataframe['trend_location'].iloc[-1]:.4f}, lr_mid_rsi={dataframe['lr_mid_rsi'].iloc[-1]:.4f}, buy_thres={self.buy_thres.value}, rsibuy_thres={self.rsibuy_thres.value}, recent_down_patterns={recent_down_patterns.iloc[-1]}")

        if self.can_short:
            recent_up_patterns = dataframe['is_up_pattern'].rolling(window=20, min_periods=1).max().fillna(0)
            base_short_conditions = (
                ~(dataframe['vtl_up'] > dataframe['vtl_down']) &
                (dataframe['trend_location'] > self.sell_thres.value) &
                (dataframe['close'].rolling(5).max() > dataframe['vtl_down']) &
                (dataframe['close'] < dataframe['vtl_down']) &
                (recent_up_patterns == 1)
            )
            short_signal = base_short_conditions
            if short_signal.any():
                latest_signal_idx = dataframe.index[short_signal][-1]
                logger.info(f"Short entry signal triggered for {metadata['pair']} at {latest_signal_idx} ({latest_signal_idx.strftime('%Y-%m-%d %H:%M:%S') if isinstance(latest_signal_idx, pd.Timestamp) else 'non-datetime index'}) with wave_pattern={dataframe['wave_pattern'][short_signal].iloc[-1]}, trend_location={dataframe['trend_location'][latest_signal_idx]:.4f}, sell_thres={self.sell_thres.value}")
                signal_indices = dataframe.index.get_loc(latest_signal_idx) + np.arange(5)
                signal_indices = signal_indices[signal_indices < len(dataframe)]
                dataframe.iloc[signal_indices, dataframe.columns.get_loc('enter_short')] = 1
                dataframe.iloc[signal_indices, dataframe.columns.get_loc('enter_tag')] = 'Wave Short - Correction Up Complete'
            else:
                logger.info(f"No short entry signal for {metadata['pair']}: trend_location={dataframe['trend_location'].iloc[-1]:.4f}, sell_thres={self.sell_thres.value}, recent_up_patterns={recent_up_patterns.iloc[-1]}")

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        wave_exit_conditions = dataframe['wave_pattern'].isin(['impulse_up', 'zigzag_up', 'flat_up'])
        if wave_exit_conditions.any():
            latest_exit_idx = dataframe.index[wave_exit_conditions][-1]
            logger.info(f"Long exit signal triggered for {metadata['pair']} at {latest_exit_idx} ({latest_exit_idx.strftime('%Y-%m-%d %H:%M:%S') if isinstance(latest_exit_idx, pd.Timestamp) else 'non-datetime index'}) with wave_pattern={dataframe['wave_pattern'][wave_exit_conditions].iloc[-1]}")
            signal_indices = dataframe.index.get_loc(latest_exit_idx) + np.arange(5)
            signal_indices = signal_indices[signal_indices < len(dataframe)]
            dataframe.iloc[signal_indices, dataframe.columns.get_loc('exit_long')] = 1

        if self.can_short:
            wave_exit_short_conditions = dataframe['wave_pattern'].isin(['impulse_down', 'zigzag_down', 'flat_down'])
            if wave_exit_short_conditions.any():
                latest_exit_idx = dataframe.index[wave_exit_short_conditions][-1]
                logger.info(f"Short exit signal triggered for {metadata['pair']} at {latest_exit_idx} ({latest_exit_idx.strftime('%Y-%m-%d %H:%M:%S') if isinstance(latest_exit_idx, pd.Timestamp) else 'non-datetime index'}) with wave_pattern={dataframe['wave_pattern'][wave_exit_short_conditions].iloc[-1]}")
                signal_indices = dataframe.index.get_loc(latest_exit_idx) + np.arange(5)
                signal_indices = signal_indices[signal_indices < len(dataframe)]
                dataframe.iloc[signal_indices, dataframe.columns.get_loc('exit_short')] = 1

        return dataframe


    def leverage(self, pair: str, current_time: datetime, current_rate: float, 
                 proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        return 5.0







    # ... [Previous unchanged sections: adjust_trade_position, custom_stoploss, custom_exit, leverage]