import math
import numpy as np
from freqtrade.strategy import IStrategy
import pandas as pd
import talib.abstract as ta # Standard TA-Lib
import pandas_ta as pta # Pandas TA
from scipy.fft import fft
from technical import qtpylib # Freqtrade's technical library
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
import logging
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)

class KitchenSink(IStrategy):
    # Strategy Interface version
    INTERFACE_VERSION = 3

    # Timeframe
    timeframe = '1h'

    # ROI table
    minimal_roi = {"0": 0.1}

    # Stoploss
    stoploss = -0.10  # General default stoploss

    # Trailing Stoploss (using TGMA's as a base, can be adjusted)
    trailing_stop = False
    trailing_stop_positive = 0.035
    trailing_stop_positive_offset = 0.013
    trailing_only_offset_is_reached = True

    # Custom stoploss
    use_custom_stoploss = True # Enabled due to HurstCycleV5RSI

    # Run settings
    can_short = False # Enabled as OmaGann and Hurst have short conditions
    ignore_roi_if_entry_signal = True # Common setting from Hurst, TGMA
    process_only_new_candles = True # Common setting
    startup_candle_count = 350 # To accommodate largest lookback (e.g. vvrp_lookback_period)

    # Position Adjustment (DCA - specific to HurstCycleV5RSI's current implementation)
    position_adjustment_enable = True # From Hurst
    # Related hyperopt params for DCA are with Hurst's parameters below

    # --- Hyperparameters from Merged Strategies ---

    # HurstCycleV5RSI Parameters (Core parameters for its logic)
    base_cycle_period = 20
    filter_weights = [1, 2, 4, 8, 4]
    convergence_threshold = 0.005  # 0.5% of price as max spread

    ### Hyperoptable parameters ###
    u_window_size = IntParameter(70, 120, default=104, space='buy', optimize=True, load=True)
    l_window_size = IntParameter(35, 50, default=42, space='buy', optimize=True, load=True)
    rsi_period = IntParameter(7, 21, default=21, space="buy", optimize=True, load=True)
    action = IntParameter(1, 5, default=3, space="buy", optimize=True, load=True)
    # Note: buy_thres and sell_thres are part of entry/exit conditions, not directly indicators.
    # They will be handled when merging entry/exit logic.
    # rsibuy_thres and rsisell_thres are also for entry/exit.
    # Parameters for HurstCycleV5RSI entry/exit logic (kept unprefixed as they are primary for these conditions)
    buy_thres = DecimalParameter(low=0.05, high=0.2, default=0.12, decimals=2 ,space='buy', optimize=True, load=True)
    sell_thres = DecimalParameter(low=0.8, high=0.95, default=0.95, decimals=2 ,space='sell', optimize=True, load=True)
    rsibuy_thres = DecimalParameter(low=35, high=55, default=52.8, decimals=1 ,space='buy', optimize=True, load=True)
    rsisell_thres = DecimalParameter(low=55, high=85, default=71.4, decimals=1 ,space='sell', optimize=True, load=True)

    # HurstCycleV5RSI DCA specific parameters (already part of its class definition)
    hurst_max_epa = IntParameter(0, 3, default=1, space='buy', optimize=True, load=True)
    hurst_filldelay = IntParameter(100, 300, default=100, space='buy', optimize=True, load=True)
    hurst_level1 = DecimalParameter(low=0.5, high=0.8, default=0.69, decimals=2, space='sell', optimize=True, load=True)
    hurst_level0 = DecimalParameter(low=0.4, high=0.7, default=0.59, decimals=2, space='sell', optimize=True, load=True)
    hurst_use_stop1 = BooleanParameter(default=False, space="protection", optimize=True, load=True) # Note: these are for custom_exit, not stoploss itself
    hurst_use_stop2 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    hurst_use_stop3 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    hurst_use_stop4 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    hurst_time0 = IntParameter(low=1440, high=2600, default=1440, space='sell', optimize=True, load=True)
    hurst_time1 = IntParameter(low=1440, high=2600, default=2000, space='sell', optimize=True, load=True)
    hurst_time2 = IntParameter(low=2600, high=4000, default=3200, space='sell', optimize=True, load=True)
    hurst_time3 = IntParameter(low=2500, high=5000, default=4500, space='sell', optimize=True, load=True)
    hurst_use0 = BooleanParameter(default=False, space="sell", optimize=True, load=True) # For adjust_trade_position
    hurst_use1 = BooleanParameter(default=False, space="sell", optimize=True, load=True)
    hurst_use2 = BooleanParameter(default=True, space="sell", optimize=True, load=True)

    # VVRPV2 Parameters
    vvrp_lookback_period = IntParameter(100, 350, default=100, space='buy', optimize=True)
    vvrp_num_bins = IntParameter(5, 15, default=9, space='buy', optimize=True)
    vvrp_max_bars = IntParameter(50, 300, default=100, space='buy', optimize=True)
    vvrp_high_volume_threshold = DecimalParameter(0.05, 0.3, default=0.1, space='buy', optimize=True)
    vvrp_trend_smoothing = IntParameter(5, 15, default=5, space='buy', optimize=True)
    vvrp_rsi_overbought = IntParameter(45, 75, default=45, space='buy', optimize=True)
    vvrp_rsi_oversold = IntParameter(25, 55, default=55, space='buy', optimize=True)

    # TGMA Parameters
    tgma_u_window_size = IntParameter(60, 150, default=120, space='buy', optimize=True) # Already prefixed
    tgma_l_window_size = IntParameter(20, 40, default=30, space='buy', optimize=True) # Already prefixed
    tgma_buylimit_param = IntParameter(0, 15, default=10, space='buy', optimize=True) # Renamed from tgma_buylimit to avoid confusion with column name
    tgma_selllimit_param = IntParameter(15, 23, default=20, space='sell', optimize=True) # Renamed from tgma_selllimit

    # TGMA Protection Hyperparameters
    tgma_cooldown_lookback = IntParameter(0, 12, default=5, space="protection", optimize=True, load=True)
    tgma_use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True, load=True)
    tgma_stop_duration = IntParameter(6, 40, default=39, space="protection", optimize=True, load=True)
    tgma_stop_protection_only_per_pair = BooleanParameter(default=True, space="protection", optimize=True, load=True)
    tgma_stop_protection_only_per_side = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    tgma_stop_protection_trade_limit = IntParameter(1, 10, default=4, space="protection", optimize=True, load=True)
    tgma_stop_protection_required_profit = DecimalParameter(-0.10, 0.01, default=-0.04, decimals=2, space="protection", optimize=True, load=True)
    tgma_use_lowprofit_protection = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    tgma_lowprofit_protection_lookback = IntParameter(1, 24, default=10, space="protection", optimize=True, load=True)
    tgma_lowprofit_trade_limit = IntParameter(1, 10, default=6, space="protection", optimize=True, load=True)
    tgma_lowprofit_stop_duration = IntParameter(1, 70, default=65, space="protection", optimize=True, load=True)
    tgma_lowprofit_required_profit = DecimalParameter(-0.10, 0.00, default=-0.04, decimals=2, space="protection", optimize=True, load=True)
    tgma_lowprofit_only_per_pair = BooleanParameter(default=True, space="protection", optimize=True, load=True)
    tgma_use_maxdrawdown_protection = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    tgma_maxdrawdown_protection_lookback = IntParameter(1, 10, default=6, space="protection", optimize=True, load=True)
    tgma_maxdrawdown_trade_limit = IntParameter(1, 20, default=10, space="protection", optimize=True, load=True)
    tgma_maxdrawdown_stop_duration = IntParameter(1, 40, default=6, space="protection", optimize=True, load=True)
    tgma_maxdrawdown_allowed_drawdown = DecimalParameter(-0.10, 0.00, default=-0.04, decimals=2, space="protection", optimize=True, load=True)
    # TGMA Custom Entry params (if to be made hyperoptable)
    tgma_increment = DecimalParameter(low=1.0005, high=1.002, default=1.001, decimals=4 ,space='buy', optimize=True, load=True)
    tgma_entryX = DecimalParameter(low=0.995, high=1.01, default=1.00, decimals=3 ,space='buy', optimize=True, load=True)


    # OmaGann Parameters
    omagann_len_param = IntParameter(5, 20, default=10, space="buy", optimize=True)
    omagann_const_param = DecimalParameter(1.5, 5, default=2.5, space="buy", optimize=True) # Note: const_param is not used in its _oma_series_static
    omagann_clsper_param = IntParameter(1, 5, default=1, space="buy", optimize=True)

    # 2Candle does not have specific hyperoptable parameters for its indicators.

    locked_stoploss = {} # For custom_stoploss from Hurst

    plot_config = {
        "main_plot": {
            # From HurstCycleV5RSI
            "vtl_up": {"color": "#05db83"}, # Hurst
            "vtl_down": {"color": "#9e209a", "type": "line"}, # Hurst
            "rollingMin": {"color": "green", "type": "line"}, # Hurst had rollingMin/Max, kept unprefixed for now
            "rollingMax": {"color": "red", "type": "line"},   # Hurst
            "filtered_close": {"color": "gray"}, # Hurst main filtered_close
            "vvrp_filtered_close": {"color": "#6cf8ae"}, # VVRPV2 specific
            "TGMA_Cycle": {"color": "blue"},  # TGMA
            "omagann_close_ma": {"color": "purple"}, # OmaGann
            "omagann_jfghla": {"color": "magenta", "type": "line"}, # OmaGann
        },
        "subplots": {
            "Cycle Analysis":{
                "cp": {"color": "blue"}, # Hurst
                "h0": {"color": "red"},  # Hurst
                "h1": {"color": "green"},# Hurst
                "h2": {"color": "orange"}# Hurst
            },
            "Hurst Trend": {
              "trend_location": {"color": "#1761bb"}, # Hurst
              "vtl_up_slope": {"color": "#398820"},   # Hurst
              "vtl_dn_slope": {"color": "#3638dd"}    # Hurst
            },
            "VVRP Oscillators": {
                "vvrp_volume_osc": {"color": "#32CD32", "type": "line"},
                "vvrp_bars_osc": {"color": "#FF4040", "type": "line"}
            },
            "TGMA Signals": {
                "TGMA_Comp_signal": {"color": "cyan"},
                "tgma_gradientTGMA_H2": {"color": "orange", "type": "line"},
                "tgma_buylimitTGMA_H2": {"color": "yellow", "type": "line"}
            },
            "OmaGann Trend":{
                "omagann_trend": {"color": "teal", "type": "line"}
            },
            "2Candle Pattern":{
                "2candle_pattern": {"color": "salmon", "type": "line"}
            },
            "RSI Plots": {
                "rsi": {"color": "yellow"}, # Hurst's RSI
                "rsi_fast": {"color": "orange"}, # Hurst's Fast RSI
                "lr_mid_rsi": {"color": "lightblue"}, # Hurst's LRC on RSI
                "vvrp_rsi": {"color": "brown"},
                "vvrp_rsiMa": {"color": "darkgoldenrod"}
            }
        }
    }

    # --- Custom methods to be merged ---

    # Methods from HurstCycleV5RSI
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe.empty:
            return self.stoploss
        current_candle = dataframe.iloc[-1].squeeze()

        required_cols = ['h2_move_mean', 'h1_move_mean', 'h0_move_mean', 'cycle_move_mean', 'trend_location']
        if not all(col in current_candle and pd.notnull(current_candle[col]) for col in required_cols):
            logger.warning(f"custom_stoploss for {pair}: Missing one or more required indicator values. Using default stoploss: {self.stoploss}")
            return self.stoploss

        SLT0 = current_candle['h2_move_mean']
        SLT1 = current_candle['h1_move_mean']
        SLT2 = current_candle['h0_move_mean']
        SLT3 = current_candle['cycle_move_mean']
        enable = current_candle['trend_location']

        display_profit = current_profit * 100

        if current_profit < -0.01:
            if pair in self.locked_stoploss:
                del self.locked_stoploss[pair]
                logger.info(f'*** {pair} *** Stoploss reset (profit < -0.01).')
            return self.stoploss

        current_pair_stoploss = self.locked_stoploss.get(pair, self.stoploss)
        new_stoploss_proposal = current_pair_stoploss

        level_triggered_message = ""

        # Using -abs(val1 - val2) based on original structure, assuming magnitude is desired stop distance.
        # This ensures the value is negative, as Freqtrade expects for stoploss.
        # The parameters self.hurst_level0.value and self.hurst_level1.value are used here.
        if pd.notnull(SLT3) and pd.notnull(SLT2) and pd.notnull(SLT1) and current_profit > SLT3:
            proposed_sl = -abs(SLT2 - SLT1)
            if proposed_sl > new_stoploss_proposal :
                 new_stoploss_proposal = proposed_sl
                 level_triggered_message = f'Level 4 (Profit: {display_profit:.2f}%)'
        elif pd.notnull(SLT2) and pd.notnull(SLT1) and current_profit > SLT2:
            proposed_sl = -abs(SLT2 - SLT1)
            if proposed_sl > new_stoploss_proposal:
                 new_stoploss_proposal = proposed_sl
                 level_triggered_message = f'Level 3 (Profit: {display_profit:.2f}%)'
        elif pd.notnull(SLT1) and pd.notnull(SLT0) and current_profit > SLT1 and pd.notnull(enable) and enable < self.hurst_level1.value:
            proposed_sl = -abs(SLT1 - SLT0)
            if proposed_sl > new_stoploss_proposal:
                new_stoploss_proposal = proposed_sl
                level_triggered_message = f'Level 2 (Profit: {display_profit:.2f}%)'
        elif pd.notnull(SLT0) and pd.notnull(SLT1) and current_profit > SLT0 and pd.notnull(enable) and enable < self.hurst_level0.value: # Original used SLT1-SLT0 here too
            proposed_sl = -abs(SLT1 - SLT0)
            if proposed_sl > new_stoploss_proposal:
                new_stoploss_proposal = proposed_sl
                level_triggered_message = f'Level 1 (Profit: {display_profit:.2f}%)'

        # If a new, tighter (less negative) stoploss was proposed:
        if new_stoploss_proposal > current_pair_stoploss:
            self.locked_stoploss[pair] = new_stoploss_proposal
            if level_triggered_message:
                 logger.info(f'*** {pair} *** {level_triggered_message} - New stoploss: {new_stoploss_proposal:.4f} activated')
            return new_stoploss_proposal

        # Otherwise, maintain the current locked stoploss or the default strategy stoploss
        return current_pair_stoploss

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        current_candle = dataframe.iloc[-1].squeeze()
        required_cols = ['h2_move_mean', 'h1_move_mean', 'h0_move_mean', 'cycle_move_mean']
        if not all(col in current_candle and pd.notnull(current_candle[col]) for col in required_cols):
            return None

        trade_duration = (current_time - trade.open_date_utc).seconds / 60

        TP0 = current_candle['h2_move_mean']
        TP1 = current_candle['h1_move_mean']
        TP2 = current_candle['h0_move_mean']
        TP3 = current_candle['cycle_move_mean']

        if pd.notnull(TP3) and current_profit > TP3 and trade_duration > self.hurst_time0.value:
            return 'hurst_roi_0_profit_exit'
        if pd.notnull(TP2) and current_profit > TP2 and trade_duration > self.hurst_time1.value:
            return 'hurst_roi_1_profit_exit'
        if pd.notnull(TP1) and current_profit > TP1 and trade_duration > self.hurst_time2.value:
            return 'hurst_roi_2_profit_exit'
        if pd.notnull(TP0) and current_profit > TP0 and trade_duration > self.hurst_time3.value:
            return 'hurst_roi_3_profit_exit'

        if pd.notnull(TP3) and current_profit < -TP3 and self.hurst_use_stop1.value:
            return 'hurst_failsafe_3_loss_exit'
        if pd.notnull(TP2) and current_profit < -TP2 and self.hurst_use_stop2.value and self.hurst_max_epa.value < 2:
            return 'hurst_failsafe_2_loss_exit'
        if pd.notnull(TP1) and current_profit < -TP1 and self.hurst_use_stop3.value and self.hurst_max_epa.value < 1:
            return 'hurst_failsafe_1_loss_exit'
        if pd.notnull(TP0) and current_profit < -TP0 and self.hurst_use_stop4.value and self.hurst_max_epa.value < 1:
            return 'hurst_failsafe_0_loss_exit'

        return None

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty:
            return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        if not filled_entries:
            return None

        count_of_entries = trade.nr_of_successful_entries

        current_candle = dataframe.iloc[-1].squeeze()
        required_cols = ['h2_move_mean', 'h1_move_mean', 'h0_move_mean', 'cycle_move_mean', 'hurst_entry_long_condition']
        if not all(col in current_candle and pd.notnull(current_candle[col]) for col in required_cols):
            return None

        last_fill_duration = (current_time - trade.date_last_filled_utc).seconds / 60

        TP0 = current_candle['h2_move_mean']
        TP1 = current_candle['h1_move_mean']
        TP2 = current_candle['h0_move_mean']
        TP3 = current_candle['cycle_move_mean']

        signal_active = current_candle['hurst_entry_long_condition'] is True

        if pd.notnull(TP2) and current_profit > TP2 and trade.nr_of_successful_exits == 0 and self.hurst_use0.value:
            return -(trade.stake_amount / 2)
        if pd.notnull(TP3) and current_profit > TP3 and trade.nr_of_successful_exits == 1 and self.hurst_use1.value:
            return -(trade.stake_amount / 2)
        if pd.notnull(TP3) and current_profit > (TP3 * 1.5) and trade.nr_of_successful_exits == 2:
             return -(trade.stake_amount / 2)
        if pd.notnull(TP3) and current_profit > (TP3 * 2.0) and trade.nr_of_successful_exits == 3:
             return -(trade.stake_amount)
        if pd.notnull(TP1) and current_profit > TP1 and trade.nr_of_successful_exits == 0 and count_of_entries == 2 and self.hurst_use2.value:
            return -(trade.stake_amount / 2)
        if pd.notnull(TP2) and current_profit > TP2 and trade.nr_of_successful_exits == 1 and count_of_entries == 2 and self.hurst_use2.value:
            return -(trade.stake_amount)

        if trade.nr_of_successful_entries >= self.hurst_max_epa.value + 1:
            return None

        if pd.notnull(TP1) and current_profit > -TP1 :
             return None

        stake_amount = filled_entries[0].cost

        if last_fill_duration > self.hurst_filldelay.value:
            if signal_active and pd.notnull(TP0) and current_profit < -TP0:
                if count_of_entries >= 1:
                    stake_amount = stake_amount * 2
                return stake_amount

            if pd.notnull(TP1) and current_profit < -TP1:
                if count_of_entries >= 1:
                    stake_amount = stake_amount * 1.5
                return stake_amount

            if pd.notnull(TP3) and current_profit < -TP3:
                if count_of_entries == 1:
                    stake_amount = stake_amount * 4
                return stake_amount
        return None

    @staticmethod
    def perform_fft(price_data, window_size=None):
        if window_size is not None:
            price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
        # Check if price_data is empty after rolling mean and dropna
        if price_data.empty:
            return np.array([]), np.array([])
        normalized_data = (price_data - np.mean(price_data)) / np.std(price_data)
        n = len(normalized_data)
        if n == 0: # Should not happen if price_data is not empty, but as a safeguard
            return np.array([]), np.array([])
        fft_data = np.fft.fft(normalized_data)
        freq = np.fft.fftfreq(n)
        power = np.abs(fft_data) ** 2
        power[np.isinf(power)] = 0
        return freq, power

    def linear_regression_channel(self, data: pd.Series, window: int, num_dev: float):
        """    Calcola la linea di regressione e il canale di deviazione standard (bande superiore e inferiore).
        :param data: Serie di dati (prezzi di chiusura)
        :param window: Lunghezza della finestra di regressione
        :param num_dev: Numero di deviazioni standard per le bande
        :return: Linea centrale (regressione), banda superiore e banda inferiore
        """
        # Lista per contenere i valori di output
        lr_channel = {'mid': [], 'upper': [], 'lower': []}

        # Ensure we have enough data points
        if len(data) < window:
            # Fill with NaNs if not enough data
            nan_series = pd.Series([np.nan] * len(data))
            return pd.DataFrame({'mid': nan_series, 'upper': nan_series, 'lower': nan_series}, index=data.index)

        for i in range(window, len(data) + 1): # Adjusted loop to include the last window
            # Seleziona la finestra corrente
            y = data[i-window:i]

            # Calcola l'indice del tempo per la finestra
            x = np.arange(window)

            # Regressione lineare sui dati della finestra
            A = np.vstack([x, np.ones(len(x))]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

            # Calcola la linea centrale (y = mx + b)
            # We need the value for the *end* of the window, which corresponds to x[-1]
            y_line = intercept + slope * (window -1) # x value for the last point in window is (window - 1)


            # Calcola la deviazione standard
            residuals = y - (intercept + slope * x)
            std_dev = np.std(residuals)

            # Linea centrale, banda superiore e inferiore
            lr_channel['mid'].append(y_line)
            lr_channel['upper'].append(y_line + num_dev * std_dev)
            lr_channel['lower'].append(y_line - num_dev * std_dev)

        # Create a DataFrame with the correct index
        # The results correspond to the end of each window
        result_index = data.index[window-1:]
        df_lr_channel = pd.DataFrame(lr_channel, index=result_index)

        # Reindex to match the original dataframe, filling initial part with NaNs
        return df_lr_channel.reindex(data.index)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # --- Indicators to be merged ---
        # --- HurstCycleV5RSI Indicators ---
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
            # Not enough data, return dataframe with NaNs for indicators or raise error
            # For now, let's return it as is, or fill expected columns with NaN
            # This behavior might need adjustment based on how Freqtrade handles this.
            # logger.warning(f"Insufficient data for {pair} for FFT: {len(dataframe)} < {self.u_window_size.value}")
            # Returning dataframe as is, subsequent operations might fail if they expect these columns.
            # Consider adding NaN columns for all expected outputs here.
            return dataframe


        # Perform FFT
        # Ensure 'ha_close' exists, could be missing if dataframe is too short from above check
        if 'ha_close' not in dataframe.columns:
            # This can happen if the initial data length check caused an early return.
            # Or if heikinashi failed for some reason.
            logger.error(f"ha_close not found in dataframe for {pair}, likely due to insufficient data.")
            return dataframe # Or handle more gracefully

        freq, power = self.perform_fft(dataframe['ha_close'], window_size=self.u_window_size.value)
        if len(freq) == 0 or len(power) == 0:
            # This means perform_fft returned empty arrays, likely due to insufficient data after rolling.
            # logger.warning(f"FFT resulted in zero or invalid frequencies for {pair}.")
            # Fill expected columns with NaN or return dataframe
            # For robustness, let's add NaN columns that would have been created.
            cols_to_nan = ['cp', 'h0', 'h1', 'h2', 'cycle_move', 'h0_move', 'h1_move', 'h2_move',
                           'cycle_move_mean', 'h0_move_mean', 'h1_move_mean', 'h2_move_mean',
                           'filtered_close', 'trough', 'crest', 'is_trough', 'is_crest', 'vtl_up', 'vtl_down',
                           'vtl-spread', 'vtl_up_slope', 'vtl_dn_slope', 'vtl_trend', 'trend_location',
                           'rsi', 'rsi_fast', 'lr_mid_rsi', 'lr_upper_rsi', 'lr_lower_rsi', 'cci', 'norm_cci',
                           'zero', 'one', 'trend']
            for col in cols_to_nan:
                dataframe[col] = np.nan
            return dataframe

        positive_mask = (1 / freq > self.l_window_size.value) & (1 / freq < self.u_window_size.value)
        positive_freqs = freq[positive_mask]
        positive_power = power[positive_mask]

        if len(positive_power) == 0: # Changed from `if not positive_power.any():` for robustness
            # logger.warning(f"No positive frequencies meet the filtering criteria for {pair}.")
            # Fill expected columns with NaN or return dataframe
            cols_to_nan = ['cp', 'h0', 'h1', 'h2', 'cycle_move', 'h0_move', 'h1_move', 'h2_move',
                           'cycle_move_mean', 'h0_move_mean', 'h1_move_mean', 'h2_move_mean',
                           'filtered_close', 'trough', 'crest', 'is_trough', 'is_crest', 'vtl_up', 'vtl_down',
                           'vtl-spread', 'vtl_up_slope', 'vtl_dn_slope', 'vtl_trend', 'trend_location',
                           'rsi', 'rsi_fast', 'lr_mid_rsi', 'lr_upper_rsi', 'lr_lower_rsi', 'cci', 'norm_cci',
                           'zero', 'one', 'trend']
            for col in cols_to_nan:
                dataframe[col] = np.nan
            return dataframe


        cycle_periods = 1 / positive_freqs
        power_threshold = 0 if len(positive_power) == 0 else 0.01 * np.max(positive_power)
        significant_indices = positive_power > power_threshold

        # Check if significant_indices has any True values
        if not np.any(significant_indices):
            # logger.warning(f"No significant frequencies found after power threshold for {pair}")
            # Fallback or fill NaNs
            dominant_freq_index = np.argmax(positive_power) # Fallback to max power if no "significant" ones
            if not positive_power.any(): # If all power is zero (should be caught earlier)
                 cols_to_nan = ['cp', 'h0', 'h1', 'h2', 'cycle_move', 'h0_move', 'h1_move', 'h2_move',
                           'cycle_move_mean', 'h0_move_mean', 'h1_move_mean', 'h2_move_mean',
                           'filtered_close', 'trough', 'crest', 'is_trough', 'is_crest', 'vtl_up', 'vtl_down',
                           'vtl-spread', 'vtl_up_slope', 'vtl_dn_slope', 'vtl_trend', 'trend_location',
                           'rsi', 'rsi_fast', 'lr_mid_rsi', 'lr_upper_rsi', 'lr_lower_rsi', 'cci', 'norm_cci',
                           'zero', 'one', 'trend']
                 for col in cols_to_nan:
                     dataframe[col] = np.nan
                 return dataframe
        else:
            significant_periods = cycle_periods[significant_indices]
            significant_power = positive_power[significant_indices]
            dominant_freq_index = np.argmax(significant_power)
            dominant_freq = positive_freqs[significant_indices][dominant_freq_index] # Index into filtered positive_freqs

        # Re-calculate dominant_freq if significant_indices was empty and we used fallback
        if not np.any(significant_indices) and len(positive_freqs) > 0 : # ensure positive_freqs is not empty
            dominant_freq_index_fallback = np.argmax(positive_power)
            dominant_freq = positive_freqs[dominant_freq_index_fallback]
        elif len(positive_freqs) == 0: # If positive_freqs itself is empty
            # logger.error(f"No positive frequencies available to determine dominant frequency for {pair}.")
            cols_to_nan = ['cp', 'h0', 'h1', 'h2', 'cycle_move', 'h0_move', 'h1_move', 'h2_move',
                           'cycle_move_mean', 'h0_move_mean', 'h1_move_mean', 'h2_move_mean',
                           'filtered_close', 'trough', 'crest', 'is_trough', 'is_crest', 'vtl_up', 'vtl_down',
                           'vtl-spread', 'vtl_up_slope', 'vtl_dn_slope', 'vtl_trend', 'trend_location',
                           'rsi', 'rsi_fast', 'lr_mid_rsi', 'lr_upper_rsi', 'lr_lower_rsi', 'cci', 'norm_cci',
                           'zero', 'one', 'trend']
            for col in cols_to_nan:
                dataframe[col] = np.nan
            return dataframe


        cycle_period = int(np.abs(1 / dominant_freq)) if dominant_freq != 0 else 100
        if cycle_period == np.inf or not np.isfinite(cycle_period): # Added isfinite check
            # logger.warning(f"Dominant frequency is zero or results in infinite cycle period for {pair}. Defaulting to 100.")
            cycle_period = 100 # Default or fallback


        harmonics = [cycle_period / (i + 1) for i in range(1, 4)]
        # Ensure harmonics are at least 1, EWM span must be > 0
        self.cp = max(1, int(cycle_period))
        self.h0 = max(1, int(harmonics[0]))
        self.h1 = max(1, int(harmonics[1]))
        self.h2 = max(1, int(harmonics[2]))


        # EWMA for cycles
        dataframe['cp'] = dataframe['ha_close'].ewm(span=self.cp, adjust=False).mean()
        dataframe['h0'] = dataframe['ha_close'].ewm(span=self.h0, adjust=False).mean()
        dataframe['h1'] = dataframe['ha_close'].ewm(span=self.h1, adjust=False).mean()
        dataframe['h2'] = dataframe['ha_close'].ewm(span=self.h2, adjust=False).mean()

        # Peak-to-Peak Movement
        # Ensure rolling windows are not larger than the dataframe length or the specific series length
        # And that the window size is at least 1
        cp_rolling_window = max(1, min(self.cp, len(dataframe['ha_close'])))
        h0_rolling_window = max(1, min(self.h0, len(dataframe['ha_close'])))
        h1_rolling_window = max(1, min(self.h1, len(dataframe['ha_close'])))
        h2_rolling_window = max(1, min(self.h2, len(dataframe['ha_close'])))

        rolling_windowc = dataframe['ha_close'].rolling(cp_rolling_window)
        rolling_windowh0 = dataframe['ha_close'].rolling(h0_rolling_window)
        rolling_windowh1 = dataframe['ha_close'].rolling(h1_rolling_window)
        rolling_windowh2 = dataframe['ha_close'].rolling(h2_rolling_window)

        ptp_valuec = rolling_windowc.apply(lambda x: np.ptp(x), raw=True)
        ptp_valueh0 = rolling_windowh0.apply(lambda x: np.ptp(x), raw=True)
        ptp_valueh1 = rolling_windowh1.apply(lambda x: np.ptp(x), raw=True)
        ptp_valueh2 = rolling_windowh2.apply(lambda x: np.ptp(x), raw=True)

        dataframe['cycle_move'] = ptp_valuec / dataframe['ha_close']
        dataframe['h0_move'] = ptp_valueh0 / dataframe['ha_close']
        dataframe['h1_move'] = ptp_valueh1 / dataframe['ha_close']
        dataframe['h2_move'] = ptp_valueh2 / dataframe['ha_close']

        dataframe['cycle_move_mean'] = dataframe['cycle_move'].rolling(cp_rolling_window).mean()
        dataframe['h0_move_mean'] = dataframe['h0_move'].rolling(cp_rolling_window).mean() # Original used self.cp
        dataframe['h1_move_mean'] = dataframe['h1_move'].rolling(cp_rolling_window).mean() # Original used self.cp
        dataframe['h2_move_mean'] = dataframe['h2_move'].rolling(cp_rolling_window).mean() # Original used self.cp


        dataframe['rollingMin'] = dataframe['close'].rolling(5).min()
        dataframe['rollingMax'] = dataframe['close'].rolling(5).max()
        # Numerical Filter
        close = dataframe['ha_close'].values
        weights = np.array(self.filter_weights) / sum(self.filter_weights)
        # Ensure there's enough data for convolution
        if len(close) >= len(weights):
            filtered = np.convolve(close, weights, mode='valid')
            dataframe['filtered_close'] = pd.Series(filtered, index=dataframe.index[len(weights)-1:]) # Adjusted index
        else:
            # logger.warning(f"Not enough data points for convolution on {pair}. Skipping numerical filter.")
            dataframe['filtered_close'] = dataframe['ha_close'] # Fallback or NaN

        # Troughs and Crests
        # Ensure rolling window for trough/crest is valid
        h2_crest_trough_window = max(1, min(self.h2, len(dataframe['filtered_close'])))
        dataframe['trough'] = dataframe['filtered_close'].rolling(h2_crest_trough_window).min()
        dataframe['crest'] = dataframe['filtered_close'].rolling(h2_crest_trough_window).max()
        dataframe['is_trough'] = np.where(dataframe['filtered_close'] == dataframe['trough'], 1, 0)
        dataframe['is_crest'] = np.where(dataframe['filtered_close'] == dataframe['crest'], 1, 0)

        # VTL based on last two troughs/crests
        dataframe['vtl_up'] = np.nan
        dataframe['vtl_down'] = np.nan

        group_window = self.h2
        trough_groups = []
        crest_groups = []
        current_trough_group = []
        current_crest_group = []
        last_trough_idx = None
        last_crest_idx = None

        for idx_loc, idx_val in enumerate(dataframe.index): # Use idx_loc for iloc, idx_val for at/loc
            if dataframe.at[idx_val, 'is_trough'] == 1:
                if last_trough_idx is None or (idx_loc - dataframe.index.get_loc(last_trough_idx)) <= group_window:
                    current_trough_group.append(idx_val)
                else:
                    if len(current_trough_group) > 0:
                        prices = [dataframe.at[i, 'filtered_close'] for i in current_trough_group]
                        min_idx = current_trough_group[np.argmin(prices)]
                        trough_groups.append(min_idx)
                    current_trough_group = [idx_val]
                last_trough_idx = idx_val

            if dataframe.at[idx_val, 'is_crest'] == 1:
                if last_crest_idx is None or (idx_loc - dataframe.index.get_loc(last_crest_idx)) <= group_window:
                    current_crest_group.append(idx_val)
                else:
                    if len(current_crest_group) > 0:
                        prices = [dataframe.at[i, 'filtered_close'] for i in current_crest_group]
                        max_idx = current_crest_group[np.argmax(prices)]
                        crest_groups.append(max_idx)
                    current_crest_group = [idx_val]
                last_crest_idx = idx_val

        if len(current_trough_group) > 0:
            prices = [dataframe.at[i, 'filtered_close'] for i in current_trough_group]
            min_idx = current_trough_group[np.argmin(prices)]
            trough_groups.append(min_idx)
        if len(current_crest_group) > 0:
            prices = [dataframe.at[i, 'filtered_close'] for i in current_crest_group]
            max_idx = current_crest_group[np.argmax(prices)]
            crest_groups.append(max_idx)

        price_min = dataframe['filtered_close'].min()
        price_max = dataframe['filtered_close'].max()

        up_vtl_segments = []
        down_vtl_segments = []

        for i in range(1, len(trough_groups)):
            x1_idx = trough_groups[i-1]
            y1 = dataframe.at[x1_idx, 'filtered_close']
            x2_idx = trough_groups[i]
            y2 = dataframe.at[x2_idx, 'filtered_close']

            x1_loc = dataframe.index.get_loc(x1_idx)
            x2_loc = dataframe.index.get_loc(x2_idx)

            slope = (y2 - y1) / (x2_loc - x1_loc) if x2_loc != x1_loc else 0
            intercept = y1 - slope * x1_loc
            up_vtl_segments.append([x1_idx, None, slope, intercept])

        for i in range(1, len(crest_groups)):
            x1_idx = crest_groups[i-1]
            y1 = dataframe.at[x1_idx, 'filtered_close']
            x2_idx = crest_groups[i]
            y2 = dataframe.at[x2_idx, 'filtered_close']

            x1_loc = dataframe.index.get_loc(x1_idx)
            x2_loc = dataframe.index.get_loc(x2_idx)

            slope = (y2 - y1) / (x2_loc - x1_loc) if x2_loc != x1_loc else 0
            intercept = y1 - slope * x1_loc
            down_vtl_segments.append([x1_idx, None, slope, intercept])

        for idx_val in dataframe.index: # Use idx_val for .at access
            current_x_loc = dataframe.index.get_loc(idx_val)
            current_price = dataframe.at[idx_val, 'filtered_close']

            active_up_segment = None
            for seg_idx, segment in enumerate(up_vtl_segments):
                start_idx, end_idx, slope, intercept = segment
                if idx_val >= start_idx and (end_idx is None or idx_val <= end_idx):
                    active_up_segment = segment
                    vtl_value = slope * current_x_loc + intercept
                    if price_min <= vtl_value <= price_max:
                        dataframe.at[idx_val, 'vtl_up'] = vtl_value
                        if current_price < vtl_value: # Price broke below
                            segment[1] = idx_val # End this segment here
                            # Future segments should not be affected by this break for this point
                    else: # VTL value out of bounds
                        segment[1] = idx_val if end_idx is None else end_idx
                # If segment has ended, it's no longer active for subsequent points
                if end_idx is not None and idx_val > end_idx:
                    active_up_segment = None


            active_down_segment = None
            for seg_idx, segment in enumerate(down_vtl_segments):
                start_idx, end_idx, slope, intercept = segment
                if idx_val >= start_idx and (end_idx is None or idx_val <= end_idx):
                    active_down_segment = segment
                    vtl_value = slope * current_x_loc + intercept
                    if price_min <= vtl_value <= price_max:
                        dataframe.at[idx_val, 'vtl_down'] = vtl_value
                        if current_price > vtl_value: # Price broke above
                            segment[1] = idx_val # End this segment here
                    else: # VTL value out of bounds
                        segment[1] = idx_val if end_idx is None else end_idx
                if end_idx is not None and idx_val > end_idx:
                    active_down_segment = None


        dataframe['vtl_up'] = dataframe['vtl_up'].fillna(method='ffill')
        dataframe['vtl_down'] = dataframe['vtl_down'].fillna(method='ffill')
        dataframe['vtl-spread'] = dataframe['vtl_down'] - dataframe['vtl_up']

        dataframe['vtl_up_slope'] = dataframe['vtl_up'].diff()
        dataframe['vtl_dn_slope'] = dataframe['vtl_down'].diff()
        dataframe['vtl_trend'] = dataframe['vtl_dn_slope'] - dataframe['vtl_up_slope']

        dataframe['trend_location'] = (dataframe['filtered_close'] - dataframe['vtl_up']) / (dataframe['vtl_down'] - dataframe['vtl_up'])
        dataframe['trend_location'].replace([np.inf, -np.inf], np.nan, inplace=True) # Handle division by zero if vtl_up == vtl_down

        dataframe['trend'] = np.where(dataframe['filtered_close'] > dataframe['cp'], 1, -1)
        dataframe['rsi'] = pta.rsi(dataframe['close'], length=max(2,self.rsi_period.value)) # RSI length must be > 1
        dataframe['rsi_fast'] = pta.rsi(dataframe['close'], length=max(2,4)) # Ensure length > 1

        # Ensure h2 is valid for rolling window, must be at least 1
        lrc_window = max(1, min(self.h2, len(dataframe['rsi_fast'])))
        if lrc_window <= 1 or len(dataframe['rsi_fast']) < lrc_window : # Min window for LRC is typically 2
            # logger.warning(f"Not enough data or invalid window for RSI LRC on {pair}. Window: {lrc_window}, Data length: {len(dataframe['rsi_fast'])}")
            dataframe['lr_mid_rsi'] = np.nan
            dataframe['lr_upper_rsi'] = np.nan
            dataframe['lr_lower_rsi'] = np.nan
        else:
            regression_channel = self.linear_regression_channel(dataframe['rsi_fast'].dropna(), window=lrc_window, num_dev=1.0)
            if regression_channel is not None and not regression_channel.empty:
                 dataframe['lr_mid_rsi'] = regression_channel['mid']
                 dataframe['lr_upper_rsi'] = regression_channel['upper']
                 dataframe['lr_lower_rsi'] = regression_channel['lower']
            else:
                 dataframe['lr_mid_rsi'] = np.nan
                 dataframe['lr_upper_rsi'] = np.nan
                 dataframe['lr_lower_rsi'] = np.nan


        cci_timeperiod = max(2, 160) # CCI timeperiod must be > 1
        if len(dataframe['high']) >= cci_timeperiod :
            # talib.abstract.CCI uses 'timeperiod', not 'length'
            dataframe['cci'] = ta.CCI(dataframe, timeperiod=cci_timeperiod)
        else:
            # logger.warning(f"Not enough data for CCI calculation on {pair}. Need {cci_timeperiod}, got {len(dataframe['high'])}")
            dataframe['cci'] = np.nan


        norm_cci_window = max(2,60) # Rolling window must be at least 1, usually > 1 for std
        if len(dataframe['cci'].dropna()) >= norm_cci_window :
            dataframe['norm_cci'] = (dataframe['cci'] - dataframe['cci'].rolling(norm_cci_window).mean()) / dataframe['cci'].rolling(norm_cci_window).std()
        else:
            # logger.warning(f"Not enough data for norm_cci calculation on {pair}.")
            dataframe['norm_cci'] = np.nan

        dataframe['zero'] = 0
        dataframe['one'] = 1

        # --- VVRPV2 Indicators ---
        # Note: VVRPV2 uses its own filter_weights for its filtered_close.
        # Hurst's filter_weights is a class attribute [1,2,4,8,4]. VVRPV2 hardcodes it.
        # To avoid conflict and keep VVRPV2's logic self-contained here, we use its hardcoded weights.
        vvrp_filter_weights_list = [1, 2, 4, 8, 4] # VVRPV2's specific weights

        # Calculate filtered close for VVRPV2
        close = dataframe['close'].values # VVRPV2 uses 'close'
        weights_vvrp = np.array(vvrp_filter_weights_list) / sum(vvrp_filter_weights_list)

        if len(close) >= len(weights_vvrp):
            filtered_vvrp = np.convolve(close, weights_vvrp, mode='valid')
            dataframe['vvrp_filtered_close'] = pd.Series(filtered_vvrp, index=dataframe.index[len(weights_vvrp)-1:])
        else:
            dataframe['vvrp_filtered_close'] = np.nan # Not enough data

        high = dataframe['high'].values
        if len(high) >= len(weights_vvrp):
            filteredh_vvrp = np.convolve(high, weights_vvrp, mode='valid')
            dataframe['vvrp_filtered_high'] = pd.Series(filteredh_vvrp, index=dataframe.index[len(weights_vvrp)-1:])
        else:
            dataframe['vvrp_filtered_high'] = np.nan

        low = dataframe['low'].values
        if len(low) >= len(weights_vvrp):
            filteredl_vvrp = np.convolve(low, weights_vvrp, mode='valid')
            dataframe['vvrp_filtered_low'] = pd.Series(filteredl_vvrp, index=dataframe.index[len(weights_vvrp)-1:])
        else:
            dataframe['vvrp_filtered_low'] = np.nan

        # Calculate ATR for dynamic range
        dataframe['vvrp_atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['vvrp_dynamic_range'] = dataframe['vvrp_atr'] / dataframe['close']

        # Calculate RSI for momentum (VVRP specific)
        dataframe['vvrp_rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        dataframe['vvrp_rsiMa'] = dataframe['vvrp_rsi'].rolling(window=10).mean()

        # Calculate rolling average volume for dynamic threshold
        dataframe['vvrp_avg_volume'] = dataframe['volume'].rolling(window=14).mean()

        # Calculate rolling VVRP with buy/sell volumes and oscillators
        dataframe = self.calculate_rolling_vvrp(dataframe, self.vvrp_lookback_period.value,
                                                 self.vvrp_num_bins.value, self.vvrp_max_bars.value)

        # Ensure all mid price columns for VVRP exist before trying to access them
        # This can happen if num_bins is changed and old columns persist or new ones are expected.
        # For simplicity, we'll assume calculate_rolling_vvrp correctly creates up to num_bins.value
        if f'VVRP_Mid_Price_{self.vvrp_num_bins.value}' in dataframe.columns and f'VVRP_Mid_Price_1' in dataframe.columns:
             if self.vvrp_num_bins.value >=9: # volume_sqz requires at least 9 bins as per original VVRP
                dataframe['vvrp_volume_sqz'] = (dataframe[f'VVRP_Mid_Price_{self.vvrp_num_bins.value}'] - dataframe['VVRP_Mid_Price_1']) / dataframe['VVRP_Mid_Price_1']
                if self.vvrp_num_bins.value >=5: # Mid_Price_5 is needed
                    dataframe['vvrp_volume_sqz_trend'] = dataframe['vvrp_volume_sqz'] * np.where(dataframe['close'] > dataframe[f'VVRP_Mid_Price_{self.vvrp_num_bins.value // 2 + 1}'], 1, -1) # Generalize mid price
                else:
                    dataframe['vvrp_volume_sqz_trend'] = np.nan
             else:
                dataframe['vvrp_volume_sqz'] = np.nan
                dataframe['vvrp_volume_sqz_trend'] = np.nan
        else:
            dataframe['vvrp_volume_sqz'] = np.nan
            dataframe['vvrp_volume_sqz_trend'] = np.nan


        # Identify high-volume nodes with dynamic threshold
        dataframe['vvrp_high_volume_node'] = 0 # VVRP_High_Volume_Node in original
        for i in range(len(dataframe)):
            # Check if 'VVRP_Bars_1' exists, implies other VVRP columns from calculate_rolling_vvrp should exist for this row
            if not pd.notnull(dataframe.at[dataframe.index[i], f'VVRP_Bars_1']):
                continue

            total_volume_period = sum(dataframe.at[dataframe.index[i], f'VVRP_Volume_{j}']
                                      for j in range(1, self.vvrp_num_bins.value + 1)
                                      if pd.notnull(dataframe.at[dataframe.index[i], f'VVRP_Volume_{j}']))

            current_avg_volume = dataframe.at[dataframe.index[i], 'vvrp_avg_volume']

            if total_volume_period > 0 and pd.notnull(current_avg_volume) and current_avg_volume > 0:
                for j_bin in range(1, self.vvrp_num_bins.value + 1):
                    bars = dataframe.at[dataframe.index[i], f'VVRP_Bars_{j_bin}']
                    volume_in_bin = dataframe.at[dataframe.index[i], f'VVRP_Volume_{j_bin}']

                    # dynamic_threshold based on total_volume_period and current_avg_volume
                    dynamic_threshold_val = self.vvrp_high_volume_threshold.value * (total_volume_period / current_avg_volume)

                    if (pd.notnull(bars) and pd.notnull(volume_in_bin) and
                        bars >= dynamic_threshold_val * self.vvrp_max_bars.value and
                        volume_in_bin / total_volume_period >= 0.2): # Original had 0.2 hardcoded
                        dataframe.at[dataframe.index[i], 'vvrp_high_volume_node'] = 1
                        break

        if 'VVRP_Bars_Osc' in dataframe.columns:
            dataframe['vvrp_volume_trend'] = ta.SMA(dataframe['VVRP_Bars_Osc'], timeperiod=self.vvrp_trend_smoothing.value) # ta.SMA needs timeperiod
        else:
            dataframe['vvrp_volume_trend'] = np.nan


        # Visualize VVRP for the latest candle - this is for debugging, consider removing for production
        # if len(dataframe) > 0 and all(f'VVRP_Mid_Price_{i}' in dataframe.columns for i in range(1, self.vvrp_num_bins.value + 1)):
        #     self.visualize_rolling_vvrp(dataframe, len(dataframe) - 1, metadata['pair'], self.vvrp_num_bins.value)

        # --- TGMA Indicators ---
        # Heikin-Ashi candles are already calculated by HurstCycleV5RSI, so we skip that part from TGMA.
        # Ensure 'ha_close' is available from previous calculations.
        if 'ha_close' not in dataframe.columns:
            logger.error(f"TGMA: 'ha_close' not found in dataframe. Make sure Heikin-Ashi is calculated before TGMA indicators.")
            return dataframe # Cannot proceed

        # Initialize Hurst Cycles for startup errors (TGMA specific initialization)
        tgma_cycle_period = 80 # TGMA's default
        tgma_harmonics = [40, 27, 20] # TGMA's defaults

        if len(dataframe) < self.tgma_u_window_size.value:
            # logger.warning(f"TGMA: Insufficient data points for FFT: {len(dataframe)}. Need {self.tgma_u_window_size.value}.")
            # Fill expected TGMA columns with NaN or handle as appropriate
            # For now, just return; subsequent merges should also check for their required columns.
            return dataframe

        # Perform FFT using the class's static method
        # Using tgma_u_window_size for TGMA's specific FFT calculation
        freq_tgma, power_tgma = KitchenSink.perform_fft(dataframe['ha_close'], window_size=self.tgma_u_window_size.value)

        if len(freq_tgma) == 0 or len(power_tgma) == 0:
            # logger.warning("TGMA: FFT resulted in zero or invalid frequencies.")
            return dataframe

        positive_mask_tgma = (1 / freq_tgma > self.tgma_l_window_size.value) & (1 / freq_tgma < self.tgma_u_window_size.value)
        positive_freqs_tgma = freq_tgma[positive_mask_tgma]
        positive_power_tgma = power_tgma[positive_mask_tgma]

        if len(positive_power_tgma) == 0:
            # logger.warning("TGMA: No positive frequencies meet the filtering criteria.")
            return dataframe

        cycle_periods_tgma = 1 / positive_freqs_tgma
        power_threshold_tgma = 0 if len(positive_power_tgma) == 0 else 0.01 * np.max(positive_power_tgma)
        significant_indices_tgma = positive_power_tgma > power_threshold_tgma

        if not np.any(significant_indices_tgma):
            # logger.warning(f"TGMA: No significant frequencies found for {pair} after power threshold. Using max power fallback.")
            dominant_freq_index_tgma = np.argmax(positive_power_tgma)
            if not positive_power_tgma.any():
                # logger.error(f"TGMA: All positive power is zero for {pair}. Cannot determine dominant frequency.")
                return dataframe
        else:
            significant_power_tgma_filtered = positive_power_tgma[significant_indices_tgma]
            dominant_freq_index_tgma = np.argmax(significant_power_tgma_filtered)
            # Need to index into the filtered positive_freqs_tgma using significant_indices_tgma first
            positive_freqs_tgma_filtered = positive_freqs_tgma[significant_indices_tgma]
            dominant_freq_tgma = positive_freqs_tgma_filtered[dominant_freq_index_tgma]

        # Fallback if significant_indices_tgma was empty
        if not np.any(significant_indices_tgma) and len(positive_freqs_tgma) > 0:
            dominant_freq_index_tgma_fb = np.argmax(positive_power_tgma) # Fallback to max power
            dominant_freq_tgma = positive_freqs_tgma[dominant_freq_index_tgma_fb]
        elif len(positive_freqs_tgma) == 0: # Should have been caught by len(positive_power_tgma) == 0
            # logger.error(f"TGMA: No positive frequencies available for {pair}.")
            return dataframe

        tgma_cycle_period_calc = int(np.abs(1 / dominant_freq_tgma)) if dominant_freq_tgma != 0 else 100
        if tgma_cycle_period_calc == np.inf or not np.isfinite(tgma_cycle_period_calc):
            # logger.warning(f"TGMA: Dominant frequency is zero or results in infinite cycle period for {pair}. Defaulting to 100.")
            tgma_cycle_period_calc = 100

        tgma_harmonics_calc = [max(1, int(tgma_cycle_period_calc / (i + 1))) for i in range(1, 4)]

        # TGMA sets these instance variables for its own calculations.
        # These might differ from Hurst's self.cp, self.h0, etc. if tgma_u/l_window_size are different.
        self.tgma_cp_val = max(1, int(tgma_cycle_period_calc)) # Ensure > 0 for EWM
        self.tgma_h0_val = max(1, int(tgma_harmonics_calc[0]))
        self.tgma_h1_val = max(1, int(tgma_harmonics_calc[1]))
        self.tgma_h2_val = max(1, int(tgma_harmonics_calc[2]))

        dataframe['TGMA_Cycle'] = dataframe['ha_close'].ewm(span=self.tgma_cp_val, adjust=False).mean()
        dataframe['TGMA_H0'] = dataframe['ha_close'].ewm(span=self.tgma_h0_val, adjust=False).mean()
        dataframe['TGMA_H1'] = dataframe['ha_close'].ewm(span=self.tgma_h1_val, adjust=False).mean()
        dataframe['TGMA_H2'] = dataframe['ha_close'].ewm(span=self.tgma_h2_val, adjust=False).mean()

        # Rolling PTP
        for period_val, name in [(self.tgma_cp_val, 'Cycle'), (self.tgma_h0_val, 'H0'),
                                 (self.tgma_h1_val, 'H1'), (self.tgma_h2_val, 'H2')]:
            rolling_window = dataframe['ha_close'].rolling(max(1, period_val)) # Ensure window > 0
            ptp_values = rolling_window.apply(lambda x: np.ptp(x), raw=True)
            dataframe[f'tgma_{name.lower()}_move'] = ptp_values / dataframe['ha_close']
            dataframe[f'tgma_{name.lower()}_move_mean'] = dataframe[f'tgma_{name.lower()}_move'].rolling(max(1,self.tgma_cp_val)).mean()


        dataframe['TGMA_ema3Cycle'] = ta.EMA(dataframe['TGMA_Cycle'], timeperiod=3)
        dataframe['TGMA_ema3H0'] = ta.EMA(dataframe['TGMA_H0'], timeperiod=3)
        dataframe['TGMA_ema3H1'] = ta.EMA(dataframe['TGMA_H1'], timeperiod=3)
        dataframe['TGMA_ema3H2'] = ta.EMA(dataframe['TGMA_H2'], timeperiod=3)

        dataframe['tgma_selllimit_val'] = self.tgma_selllimit_param.value # Store param value in df, ensure this uses the renamed param

        dataframe = self.calculate_tgma(dataframe, "TGMA_H2", self.tgma_h2_val, "tgma_") # Pass prefix
        dataframe = self.calculate_tgma(dataframe, "TGMA_H1", self.tgma_h1_val, "tgma_")
        dataframe = self.calculate_tgma(dataframe, "TGMA_H0", self.tgma_h0_val, "tgma_")
        dataframe = self.calculate_tgma(dataframe, "TGMA_Cycle", self.tgma_cp_val, "tgma_")

        dataframe['TGMA_Composite'] = (dataframe['tgma_gradientTGMA_H2'] + dataframe['tgma_gradientTGMA_H1'] +
                                     dataframe['tgma_gradientTGMA_H0'] + dataframe['tgma_gradientTGMA_Cycle']) / 4
        dataframe['TGMA_Comp_signal'] = ta.SMA(dataframe['TGMA_Composite'], timeperiod=4) # ta.SMA needs timeperiod

        dataframe['tgma_minh2'], dataframe['tgma_maxh2'] = KitchenSink.calculate_minima_maxima(dataframe, self.tgma_h2_val)
        dataframe['tgma_minh1'], dataframe['tgma_maxh1'] = KitchenSink.calculate_minima_maxima(dataframe, self.tgma_h1_val)
        dataframe['tgma_minh0'], dataframe['tgma_maxh0'] = KitchenSink.calculate_minima_maxima(dataframe, self.tgma_h0_val)
        dataframe['tgma_mincp'], dataframe['tgma_maxcp'] = KitchenSink.calculate_minima_maxima(dataframe, self.tgma_cp_val)

        # --- OmaGann Indicators ---
        # Get hyperopt values
        len_val = int(self.omagann_len_param.value)
        const_val = float(self.omagann_const_param.value) # Not used by _oma_series_static, but kept for consistency with original
        clsper_val = int(self.omagann_clsper_param.value)

        # Calculate OMA values using the static helper method
        # The const_val is not used in the _oma_series_static as per its definition from OmaGann.py (it was unused there too)
        dataframe['omagann_high_ma'] = KitchenSink._oma_series_static(dataframe['high'], len_val, adaptive=True) # const_val removed
        dataframe['omagann_low_ma'] = KitchenSink._oma_series_static(dataframe['low'], len_val, adaptive=True)   # const_val removed
        dataframe['omagann_close_ma'] = KitchenSink._oma_series_static(dataframe['close'], clsper_val, adaptive=True) # const_val removed

        # Calculate Gann HiLo Activator
        jfghla_list = [] # Renamed to jfghla_list to avoid conflict if jfghla was a column from another strategy
        for i in range(len(dataframe)):
            if i == 0:
                jfghla_list.append(np.nan)
                continue

            # Use .iloc for positional access, ensure column names are correct
            close_ma_val = dataframe['omagann_close_ma'].iloc[i]
            prev_high_ma_val = dataframe['omagann_high_ma'].iloc[i-1]
            prev_low_ma_val = dataframe['omagann_low_ma'].iloc[i-1]

            if close_ma_val > prev_high_ma_val:
                jfghla_list.append(dataframe['omagann_low_ma'].iloc[i])
            elif close_ma_val < prev_low_ma_val:
                jfghla_list.append(dataframe['omagann_high_ma'].iloc[i])
            else:
                jfghla_list.append(jfghla_list[-1] if len(jfghla_list) > 0 else np.nan)

        dataframe['omagann_jfghla'] = jfghla_list
        dataframe['omagann_trend'] = np.where(dataframe['omagann_close_ma'] > dataframe['omagann_jfghla'], 1,
                                            np.where(dataframe['omagann_close_ma'] < dataframe['omagann_jfghla'], -1, 0))

        # --- 2Candle Indicators ---
        # Calculate candle range (high - low)
        dataframe['2candle_range'] = dataframe['high'] - dataframe['low']
        dataframe['2candle_range_third'] = dataframe['2candle_range'] / 3

        # Close position: Determine if close is in upper, mid, or lower third
        dataframe['2candle_close_position'] = 0  # Default: mid
        dataframe.loc[dataframe['close'] > (dataframe['high'] - dataframe['2candle_range_third']), '2candle_close_position'] = 1  # High close
        dataframe.loc[dataframe['close'] < (dataframe['low'] + dataframe['2candle_range_third']), '2candle_close_position'] = -1  # Low close

        # Close comparison: Compare current close to previous candle's range
        dataframe['2candle_prev_high'] = dataframe['high'].shift(1)
        dataframe['2candle_prev_low'] = dataframe['low'].shift(1)
        dataframe['2candle_close_comparison'] = 0  # Default: range
        dataframe.loc[dataframe['close'] > dataframe['2candle_prev_high'], '2candle_close_comparison'] = 1  # Bull candle
        dataframe.loc[dataframe['close'] < dataframe['2candle_prev_low'], '2candle_close_comparison'] = -1  # Bear candle

        # Combine close position and close comparison into 9 patterns
        dataframe['2candle_pattern'] = dataframe['2candle_close_position'] * 3 + dataframe['2candle_close_comparison'] + 4  # Maps to 0-8 (9 patterns)

        # Simple support/resistance levels using rolling min/max
        dataframe['2candle_support'] = dataframe['low'].rolling(window=20).min()
        dataframe['2candle_resistance'] = dataframe['high'].rolling(window=20).max()

        # Initialize Freqtrade standard signal columns
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = '' # Default tag

        # --- Entry conditions to be merged ---
        dataframe['sum_long_entries'] = 0 # Placeholder
        dataframe['sum_short_entries'] = 0 # Placeholder for short entries sum

        # Initialize Hurst an VVR RSI boolean condition columns
        dataframe['hurst_entry_long_condition'] = False
        dataframe['hurst_entry_short_condition'] = False
        dataframe['vvrp_entry_long_condition'] = False
        # No short conditions for VVRPV2 in the source, so no vvrp_entry_short_condition needed
        dataframe['tgma_entry_long_condition'] = False
        dataframe['omagann_entry_long_condition'] = False
        dataframe['omagann_entry_short_condition'] = False
        dataframe['2candle_entry_long_condition_1'] = False
        dataframe['2candle_entry_long_condition_2'] = False
        dataframe['2candle_entry_short_condition'] = False

        # HurstCycleV5RSI long entry condition
        # Ensure all required columns exist, otherwise skip (could happen with very short dataframes)
        hurst_long_req_cols = ['vtl_up', 'vtl_down', 'trend_location', 'lr_mid_rsi', 'close']
        if all(col in dataframe.columns for col in hurst_long_req_cols):
            dataframe.loc[
                (
                    ~(dataframe['vtl_up'] > dataframe['vtl_down']) &
                    (dataframe['trend_location'] < self.buy_thres.value) &
                    (dataframe['lr_mid_rsi'] < self.rsibuy_thres.value) &
                    (dataframe['close'].rolling(5).min() < dataframe['vtl_up']) & # Ensure rolling window is valid if data is short
                    (dataframe['close'] > dataframe['vtl_up'])
                ),
                'hurst_entry_long_condition'] = True
        else:
            logger.warning(f"Missing one or more required columns for Hurst long entry for pair {metadata['pair']}. Columns: {dataframe.columns.tolist()}")


        # HurstCycleV5RSI short entry condition (if can_short is True, this will be evaluated later)
        hurst_short_req_cols = ['vtl_up', 'vtl_down', 'trend_location', 'close']
        if all(col in dataframe.columns for col in hurst_short_req_cols):
            dataframe.loc[
                (
                    ~(dataframe['vtl_up'] > dataframe['vtl_down']) &
                    (dataframe['trend_location'] > self.sell_thres.value) &
                    (dataframe['close'].rolling(5).max() > dataframe['vtl_down']) & # Ensure rolling window is valid
                    (dataframe['close'] < dataframe['vtl_down'])
                ),
                'hurst_entry_short_condition'] = True
        else:
            logger.warning(f"Missing one or more required columns for Hurst short entry for pair {metadata['pair']}. Columns: {dataframe.columns.tolist()}")

        # VVRPV2 long entry condition
        vvrp_long_req_cols = ['vvrp_rsiMa', 'vvrp_volume_trend']
        if all(col in dataframe.columns for col in vvrp_long_req_cols):
            # Ensure enough data for shift(2)
            if len(dataframe) > 2:
                dataframe.loc[
                    (
                        (dataframe["vvrp_rsiMa"] < self.vvrp_rsi_oversold.value) &
                        (dataframe["vvrp_volume_trend"].shift(1) < dataframe['vvrp_volume_trend']) &
                        (dataframe["vvrp_volume_trend"].shift(1) > dataframe['vvrp_volume_trend'].shift(2))
                    ),
                    'vvrp_entry_long_condition'] = True
            else:
                logger.warning(f"Not enough data for VVRPV2 long entry condition (shifts) for pair {metadata['pair']}.")
        else:
            logger.warning(f"Missing one or more required columns for VVRPV2 long entry for pair {metadata['pair']}. Columns: {dataframe.columns.tolist()}")

        # TGMA long entry condition
        # Column names like 'tgma_gradientTGMA_H2' and 'tgma_buylimitTGMA_H2' are based on the calculate_tgma method's adaptation.
        tgma_long_req_cols = ['tgma_gradientTGMA_H2', 'tgma_buylimitTGMA_H2']
        if all(col in dataframe.columns for col in tgma_long_req_cols):
             dataframe.loc[
                (
                    (dataframe["tgma_gradientTGMA_H2"] < self.tgma_selllimit_param.value) & # Use renamed parameter
                    (qtpylib.crossed_above(dataframe["tgma_gradientTGMA_H2"], dataframe['tgma_buylimitTGMA_H2']))
                ),
                'tgma_entry_long_condition'] = True
        else:
            logger.warning(f"Missing one or more required columns for TGMA long entry for pair {metadata['pair']}. Columns: {dataframe.columns.tolist()}")

        # OmaGann long entry condition
        omagann_entry_req_cols = ['omagann_trend']
        if all(col in dataframe.columns for col in omagann_entry_req_cols):
            if len(dataframe) > 1: # For shift(1)
                dataframe.loc[
                    (
                        (dataframe['omagann_trend'] == 1) &
                        (dataframe['omagann_trend'].shift(1) <= 0)
                    ),
                    'omagann_entry_long_condition'] = True

                dataframe.loc[
                    (
                        (dataframe['omagann_trend'] == -1) &
                        (dataframe['omagann_trend'].shift(1) >= 0)
                    ),
                    'omagann_entry_short_condition'] = True
            else:
                logger.warning(f"Not enough data for OmaGann entry conditions (shift) for pair {metadata['pair']}.")
        else:
            logger.warning(f"Missing one or more required columns for OmaGann entry conditions for pair {metadata['pair']}. Columns: {dataframe.columns.tolist()}")

        # 2Candle entry conditions
        twocandle_entry_req_cols = ['2candle_pattern', 'close', '2candle_resistance', '2candle_support']
        if all(col in dataframe.columns for col in twocandle_entry_req_cols):
            if len(dataframe) > 1: # For shift(1)
                # Condition 1 for enter_long
                dataframe.loc[
                    (
                        (dataframe['2candle_pattern'] > 4) &
                        # (dataframe['close'] > dataframe['2candle_resistance'].shift(1)) &
                        (dataframe['2candle_pattern'].shift(1) != 0)
                    ),
                    '2candle_entry_long_condition_1'] = True

                # Condition 2 for enter_long
                dataframe.loc[
                    (
                        (dataframe['2candle_pattern'] >= 4) &
                        # (dataframe['close'] > dataframe['2candle_support']) &
                        (dataframe['2candle_pattern'].shift(1) == 0)
                    ),
                    '2candle_entry_long_condition_2'] = True

                # Condition for enter_short
                dataframe.loc[
                    (
                        (dataframe['2candle_pattern'] == 0) &
                        # (dataframe['close'] < dataframe['2candle_support'].shift(1)) &
                        (dataframe['2candle_pattern'].shift(1) != 4)
                    ),
                    '2candle_entry_short_condition'] = True
            else:
                logger.warning(f"Not enough data for 2Candle entry conditions (shift) for pair {metadata['pair']}.")
        else:
            logger.warning(f"Missing one or more required columns for 2Candle entry conditions for pair {metadata['pair']}. Columns: {dataframe.columns.tolist()}")

        # Sum all long entry conditions
        list_of_long_entry_columns = [
            'hurst_entry_long_condition',
            'vvrp_entry_long_condition',
            'tgma_entry_long_condition',
            'omagann_entry_long_condition',
            '2candle_entry_long_condition_1',
            '2candle_entry_long_condition_2'
        ]
        # Ensure all columns in the list actually exist in the dataframe before summing to avoid KeyError
        existing_long_entry_cols = [col for col in list_of_long_entry_columns if col in dataframe.columns]
        if existing_long_entry_cols:
            dataframe['sum_long_entries'] = dataframe[existing_long_entry_cols].sum(axis=1)
        else:
            dataframe['sum_long_entries'] = 0 # Or handle as an error/warning

        # Sum all short entry conditions
        list_of_short_entry_columns = [
            'hurst_entry_short_condition',
            'omagann_entry_short_condition',
            '2candle_entry_short_condition'
        ]
        existing_short_entry_cols = [col for col in list_of_short_entry_columns if col in dataframe.columns]
        if existing_short_entry_cols:
            dataframe['sum_short_entries'] = dataframe[existing_short_entry_cols].sum(axis=1)
        else:
            dataframe['sum_short_entries'] = 0

        #exits
        # Initialize Freqtrade standard signal columns
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_tag'] = '' # Default tag

        # --- Exit conditions to be merged ---
        dataframe['sum_long_exits'] = 0 # Placeholder
        dataframe['sum_short_exits'] = 0 # Placeholder for short exits sum

        # Initialize Hurst boolean condition columns
        dataframe['hurst_exit_long_condition'] = False
        dataframe['hurst_exit_short_condition'] = False
        dataframe['vvrp_exit_long_condition'] = False
        # No short conditions for VVRPV2 in the source
        dataframe['tgma_exit_long_condition'] = False
        dataframe['omagann_exit_long_condition'] = False
        dataframe['omagann_exit_short_condition'] = False
        dataframe['2candle_exit_long_condition'] = False
        dataframe['2candle_exit_short_condition'] = False

        # HurstCycleV5RSI long exit condition (from commented out code)
        hurst_exit_long_req_cols = ['trend_location', 'vtl_down', 'close']
        if all(col in dataframe.columns for col in hurst_exit_long_req_cols):
            dataframe.loc[
                (
                    (dataframe['trend_location'] > self.sell_thres.value) &
                    (dataframe['close'] > dataframe['vtl_down'])
                ),
                'hurst_exit_long_condition'] = True
        else:
            logger.warning(f"Missing one or more required columns for Hurst long exit for pair {metadata['pair']}. Columns: {dataframe.columns.tolist()}")

        # HurstCycleV5RSI short exit condition (if can_short is True, from commented out code)
        hurst_exit_short_req_cols = ['trend_location', 'vtl_up', 'close']
        if all(col in dataframe.columns for col in hurst_exit_short_req_cols):
            dataframe.loc[
                (
                    (dataframe['trend_location'] < self.buy_thres.value) &
                    (dataframe['close'] < dataframe['vtl_up'])
                ),
                'hurst_exit_short_condition'] = True
        else:
            logger.warning(f"Missing one or more required columns for Hurst short exit for pair {metadata['pair']}. Columns: {dataframe.columns.tolist()}")

        # VVRPV2 long exit condition
        vvrp_exit_long_req_cols = ['vvrp_rsiMa', 'vvrp_volume_trend']
        if all(col in dataframe.columns for col in vvrp_exit_long_req_cols):
            # Ensure enough data for shift(2)
            if len(dataframe) > 2:
                dataframe.loc[
                    (
                        (dataframe["vvrp_rsiMa"] > self.vvrp_rsi_overbought.value) &
                        (dataframe["vvrp_volume_trend"].shift(1) > dataframe['vvrp_volume_trend']) &
                        (dataframe["vvrp_volume_trend"].shift(1) < dataframe['vvrp_volume_trend'].shift(2))
                    ),
                    'vvrp_exit_long_condition'] = True
            else:
                logger.warning(f"Not enough data for VVRPV2 long exit condition (shifts) for pair {metadata['pair']}.")
        else:
            logger.warning(f"Missing one or more required columns for VVRPV2 long exit for pair {metadata['pair']}. Columns: {dataframe.columns.tolist()}")

        # TGMA long exit condition
        tgma_exit_long_req_cols = ['tgma_gradientTGMA_H2', 'tgma_buylimitTGMA_H2']
        if all(col in dataframe.columns for col in tgma_exit_long_req_cols):
            dataframe.loc[
                (
                    (qtpylib.crossed_below(dataframe["tgma_gradientTGMA_H2"], dataframe['tgma_buylimitTGMA_H2']))
                ),
                'tgma_exit_long_condition'] = True
        else:
            logger.warning(f"Missing one or more required columns for TGMA long exit for pair {metadata['pair']}. Columns: {dataframe.columns.tolist()}")

        # OmaGann exit conditions
        omagann_exit_req_cols = ['omagann_trend']
        if all(col in dataframe.columns for col in omagann_exit_req_cols):
            if len(dataframe) > 1: # For shift(1)
                dataframe.loc[
                    (
                        (dataframe['omagann_trend'] == -1) &
                        (dataframe['omagann_trend'].shift(1) >= 0)
                    ),
                    'omagann_exit_long_condition'] = True

                dataframe.loc[
                    (
                        (dataframe['omagann_trend'] == 1) &
                        (dataframe['omagann_trend'].shift(1) <= 0)
                    ),
                    'omagann_exit_short_condition'] = True
            else:
                logger.warning(f"Not enough data for OmaGann exit conditions (shift) for pair {metadata['pair']}.")
        else:
            logger.warning(f"Missing one or more required columns for OmaGann exit conditions for pair {metadata['pair']}. Columns: {dataframe.columns.tolist()}")

        # 2Candle exit conditions (simplified interpretation)
        twocandle_exit_req_cols = ['2candle_pattern']
        if all(col in dataframe.columns for col in twocandle_exit_req_cols):
            dataframe.loc[(dataframe['2candle_pattern'] == 0), '2candle_exit_long_condition'] = True
            dataframe.loc[(dataframe['2candle_pattern'] == 4), '2candle_exit_short_condition'] = True
        else:
            logger.warning(f"Missing one or more required columns for 2Candle exit conditions for pair {metadata['pair']}. Columns: {dataframe.columns.tolist()}")

        # Sum all long exit conditions
        list_of_long_exit_columns = [
            'hurst_exit_long_condition',
            'vvrp_exit_long_condition',
            'tgma_exit_long_condition',
            'omagann_exit_long_condition',
            '2candle_exit_long_condition'
        ]
        existing_long_exit_cols = [col for col in list_of_long_exit_columns if col in dataframe.columns]
        if existing_long_exit_cols:
            dataframe['sum_long_exits'] = dataframe[existing_long_exit_cols].sum(axis=1)
        else:
            dataframe['sum_long_exits'] = 0

        # Sum all short exit conditions
        list_of_short_exit_columns = [
            'hurst_exit_short_condition',
            'omagann_exit_short_condition',
            '2candle_exit_short_condition'
        ]
        existing_short_exit_cols = [col for col in list_of_short_exit_columns if col in dataframe.columns]
        if existing_short_exit_cols:
            dataframe['sum_short_exits'] = dataframe[existing_short_exit_cols].sum(axis=1)
        else:
            dataframe['sum_short_exits'] = 0

        dataframe['long_decision'] =  dataframe['sum_long_entries'] - dataframe['sum_long_exits']
        dataframe['short_decision'] =  dataframe['sum_short_entries'] - dataframe['sum_short_exits']
        dataframe['overall_decision'] = dataframe['long_decision'] - dataframe['short_decision']

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe.loc[
            (
                (dataframe["sum_long_entries"] >= self.action.value) &
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'Long Enter')

        # Short entry logic will be guarded by self.can_short
        if self.can_short == True:
            dataframe.loc[
                (
                    (dataframe["sum_short_entries"] >= self.action.value) &
                    (dataframe['volume'] > 0)   # Make sure Volume is not 0
                ),
                ['enter_short', 'enter_tag']] = (1, 'Short Enter')
        else: # Explicitly ensure no short signals if can_short is False
            # This is redundant if 'enter_short' is already initialized to 0,
            # but kept for explicit clarity.
            dataframe['enter_short'] = 0
            # Potentially clear enter_tag if it was set by long condition and short is also true but can_short is false
            # However, current logic assigns 'Short Enter' only if self.can_short is True.
            # If a long signal and a short signal could co-exist based on sums,
            # and can_short is False, enter_tag might remain 'Long Enter'. This is usually fine.

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    
        dataframe.loc[
            (
                (dataframe["sum_long_exits"] >= self.action.value) & # Assuming 2 is the threshold for exits as well
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'Long Exit')

        if self.can_short == True:
            dataframe.loc[
                (
                    (dataframe["sum_short_exits"] >= self.action.value) & # Assuming 2 is the threshold
                    (dataframe['volume'] > 0)   # Make sure Volume is not 0
                ),
                # Corrected tag for short exit as per instruction
                ['exit_short', 'exit_tag']] = (1, 'Short Exit')
        else: # Explicitly ensure no short exit signals
            # Redundant if 'exit_short' is initialized to 0, but for clarity.
            dataframe['exit_short'] = 0
            # Similar consideration for exit_tag as in populate_entry_trend

        return dataframe

    # --- Custom methods to be merged ---

    # Methods from HurstCycleV5RSI
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe.empty:
            return self.stoploss # Default stoploss if no dataframe
        current_candle = dataframe.iloc[-1].squeeze()

        # Ensure all required columns are present
        required_cols = ['h2_move_mean', 'h1_move_mean', 'h0_move_mean', 'cycle_move_mean', 'trend_location']
        if not all(col in current_candle for col in required_cols):
            # logger.warning(f"custom_stoploss: Missing required columns for {pair}. Using default stoploss.")
            return self.stoploss

        SLT0 = current_candle['h2_move_mean']
        SLT1 = current_candle['h1_move_mean']
        SLT2 = current_candle['h0_move_mean']
        SLT3 = current_candle['cycle_move_mean']
        enable = current_candle['trend_location']

        display_profit = current_profit * 100

        if current_profit < -0.01: # Exit if profit is already negative, reset locked stoploss
            if pair in self.locked_stoploss:
                del self.locked_stoploss[pair]
                # self.dp.send_msg(f'*** {pair} *** Stoploss reset.') # Consider managing messages centrally if needed
                logger.info(f'*** {pair} *** Stoploss reset.')
            return self.stoploss # Return the initial stoploss

        new_stoploss = self.stoploss # Default to initial stoploss
        level_triggered = 0 # To track which profit level triggered a new stoploss

        # Check profit levels and update stoploss if a new higher level is reached
        if SLT3 is not None and current_profit > SLT3:
            proposed_sl = SLT2 - SLT1 # Example calculation, ensure this logic is sound
            if proposed_sl > new_stoploss :
                 new_stoploss = proposed_sl
                 level_triggered = 4
        elif SLT2 is not None and current_profit > SLT2:
            proposed_sl = SLT2 - SLT1 # Example
            if proposed_sl > new_stoploss :
                 new_stoploss = proposed_sl
                 level_triggered = 3
        elif SLT1 is not None and current_profit > SLT1 and pd.notnull(enable) and enable < self.hurst_level1.value: # Using prefixed hurst_level1
            proposed_sl = SLT1 - SLT0 # Example
            if proposed_sl > new_stoploss:
                new_stoploss = proposed_sl
                level_triggered = 2
        elif SLT0 is not None and current_profit > SLT0 and pd.notnull(enable) and enable < self.hurst_level0.value: # Using prefixed hurst_level0
            proposed_sl = SLT1 - SLT0 # Example
            if proposed_sl > new_stoploss:
                new_stoploss = proposed_sl
                level_triggered = 1

        # Update locked_stoploss if the new_stoploss is tighter (higher)
        if new_stoploss > self.stoploss : # Ensure we are setting a tighter stoploss than initial
            if pair not in self.locked_stoploss or new_stoploss > self.locked_stoploss[pair]:
                self.locked_stoploss[pair] = new_stoploss
                if level_triggered > 0: # Only log if a new level was actually hit
                    # self.dp.send_msg(f'*** {pair} *** Profit Level {level_triggered} ({display_profit:.2f}%) - New stoploss: {new_stoploss:.4f} activated')
                    logger.info(f'*** {pair} *** Profit Level {level_triggered} ({display_profit:.2f}%) - New stoploss: {new_stoploss:.4f} activated')
            # Return the currently locked (potentially updated) stoploss for the pair
            return self.locked_stoploss.get(pair, self.stoploss)

        return self.stoploss # Return default if no conditions met or new_stoploss isn't tighter

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None # No decision if no data

        current_candle = dataframe.iloc[-1].squeeze()

        required_cols = ['h2_move_mean', 'h1_move_mean', 'h0_move_mean', 'cycle_move_mean']
        if not all(col in current_candle for col in required_cols):
            # logger.warning(f"custom_exit: Missing required columns for {pair}.")
            return None

        trade_duration = (current_time - trade.open_date_utc).seconds / 60

        TP0 = current_candle['h2_move_mean']
        TP1 = current_candle['h1_move_mean']
        TP2 = current_candle['h0_move_mean']
        TP3 = current_candle['cycle_move_mean']

        # ROI based exits from Hurst
        if pd.notnull(TP3) and current_profit > TP3 and trade_duration > self.hurst_time0.value: # Using prefixed hurst_time0
            return 'hurst_roi_0_profit_exit'
        if pd.notnull(TP2) and current_profit > TP2 and trade_duration > self.hurst_time1.value: # Using prefixed hurst_time1
            return 'hurst_roi_1_profit_exit'
        if pd.notnull(TP1) and current_profit > TP1 and trade_duration > self.hurst_time2.value: # Using prefixed hurst_time2
            return 'hurst_roi_2_profit_exit'
        if pd.notnull(TP0) and current_profit > TP0 and trade_duration > self.hurst_time3.value: # Using prefixed hurst_time3
            return 'hurst_roi_3_profit_exit'

        # Negative stoploss based exits from Hurst
        if pd.notnull(TP3) and current_profit < -TP3 and self.hurst_use_stop1.value == True:
            return 'hurst_failsafe_3_loss_exit'
        if pd.notnull(TP2) and current_profit < -TP2 and self.hurst_use_stop2.value == True and self.hurst_max_epa.value < 2:
            return 'hurst_failsafe_2_loss_exit'
        if pd.notnull(TP1) and current_profit < -TP1 and self.hurst_use_stop3.value == True and self.hurst_max_epa.value < 1:
            return 'hurst_failsafe_1_loss_exit'
        if pd.notnull(TP0) and current_profit < -TP0 and self.hurst_use_stop4.value == True and self.hurst_max_epa.value < 1:
            return 'hurst_failsafe_0_loss_exit'

        return None # Explicitly return None if no custom exit condition is met

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float, # These are Freqtrade built-in, not from strategy params
                              current_entry_profit: float, current_exit_profit: float, # Same as above
                              **kwargs) -> Optional[float]:

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty:
            return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        if not filled_entries: # Should not happen if a trade exists, but good check
            return None

        count_of_entries = trade.nr_of_successful_entries

        # Ensure current_candle has all necessary data
        current_candle = dataframe.iloc[-1].squeeze()
        required_cols = ['h2_move_mean', 'h1_move_mean', 'h0_move_mean', 'cycle_move_mean', 'hurst_entry_long_condition']
        if not all(col in current_candle for col in required_cols):
            # logger.warning(f"adjust_trade_position: Missing required columns for {trade.pair}.")
            return None

        trade_duration = (current_time - trade.open_date_utc).seconds / 60
        last_fill_duration = (current_time - trade.date_last_filled_utc).seconds / 60

        TP0 = current_candle['h2_move_mean']
        TP1 = current_candle['h1_move_mean']
        TP2 = current_candle['h0_move_mean']
        TP3 = current_candle['cycle_move_mean']

        # Use the boolean condition column for the signal
        signal_active = current_candle['hurst_entry_long_condition'] # This is now True/False

        # Take Profit logic (using prefixed Hurst parameters)
        if pd.notnull(TP2) and current_profit > TP2 and trade.nr_of_successful_exits == 0 and self.hurst_use0.value:
            return -(trade.stake_amount / 2)
        if pd.notnull(TP3) and current_profit > TP3 and trade.nr_of_successful_exits == 1 and self.hurst_use1.value:
            return -(trade.stake_amount / 2)
        # ... (other TP conditions from Hurst) ...
        if pd.notnull(TP1) and current_profit > TP1 and trade.nr_of_successful_exits == 0 and count_of_entries == 2 and self.hurst_use2.value:
            return -(trade.stake_amount / 2)
        if pd.notnull(TP2) and current_profit > TP2 and trade.nr_of_successful_exits == 1 and count_of_entries == 2 and self.hurst_use2.value:
            return -(trade.stake_amount)

        if trade.nr_of_successful_entries >= self.hurst_max_epa.value + 1: # Use prefixed hurst_max_epa
            return None

        if pd.notnull(TP1) and current_profit > -TP1 : # Block DCA if not dipped enough
             return None

        stake_amount = filled_entries[0].cost # Base stake amount

        # DCA logic (using prefixed Hurst parameters)
        if last_fill_duration > self.hurst_filldelay.value:
            if signal_active and pd.notnull(TP0) and current_profit < -TP0: # signal_active is True/False
                if count_of_entries >= 1:
                    stake_amount = stake_amount * 2
                return stake_amount

            if pd.notnull(TP1) and current_profit < -TP1:
                if count_of_entries >= 1:
                    stake_amount = stake_amount * 1.5
                return stake_amount

            if pd.notnull(TP3) and current_profit < -TP3:
                if count_of_entries == 1:
                    stake_amount = stake_amount * 4
                return stake_amount
        return None

    @staticmethod
    def _oma_series_static(src_series: pd.Series, length: int, adaptive: bool = True) -> pd.Series:
        # Ensure length is at least 1 for EWM span
        length = max(1, int(length))
        series = src_series.copy()
        # The 'const' parameter was in the original OmaGann's oma_series signature but not used.
        # So, it's omitted here.
        e1 = series.ewm(alpha=1/length, adjust=adaptive).mean()
        e2 = e1.ewm(alpha=1/length, adjust=adaptive).mean()
        v1 = 1.5*e1 - 0.5*e2

        e3 = v1.ewm(alpha=1/length, adjust=adaptive).mean()
        e4 = e3.ewm(alpha=1/length, adjust=adaptive).mean()
        v2 = 1.5*e3 - 0.5*e4

        e5 = v2.ewm(alpha=1/length, adjust=adaptive).mean()
        e6 = e5.ewm(alpha=1/length, adjust=adaptive).mean()
        return 1.5*e5 - 0.5*e6

    @staticmethod
    def calculate_minima_maxima(df, window):
        if df is None or df.empty or 'ha_close' not in df.columns: # Added ha_close check
            return pd.Series(np.zeros(len(df)), index=df.index), pd.Series(np.zeros(len(df)), index=df.index)

        minima = np.zeros(len(df))
        maxima = np.zeros(len(df))

        # Ensure window is valid
        window = max(1, int(window))


        for i in range(window, len(df)):  # Ensure index does not go out of bounds
            # Use .iloc for positional indexing
            window_data = df['ha_close'].iloc[i - window : i + 1]
            current_val = df['ha_close'].iloc[i]

            if not window_data.empty:
                 is_min = current_val == window_data.min()
                 is_max = current_val == window_data.max()
                 # Ensure uniqueness within the window to mark a true peak/trough
                 is_unique_min = (window_data == current_val).sum() == 1
                 is_unique_max = (window_data == current_val).sum() == 1

                 if is_min and is_unique_min:
                     minima[i] = -window
                 if is_max and is_unique_max:
                     maxima[i] = window

        return pd.Series(minima, index=df.index), pd.Series(maxima, index=df.index)

    def calculate_tgma(self, dataframe: pd.DataFrame, avg_str: str, length: int, prefix: str = "") -> pd.DataFrame:
        # Parameters
        # ma_length = int(length/3) # Not used in the provided TGMA snippet
        steps = length  # Max gradient steps
        length = max(1, int(length)) # Ensure length is at least 1 for EWM/rolling

        # Initialize Gradient Strength
        qty_adv_dec = np.zeros(len(dataframe))

        # Calculate Gradient
        # Ensure avg_str and ema3_avg_str columns exist
        ema_col = f"{prefix}ema3{avg_str.replace(prefix, '')}" # Construct EMA col name based on avg_str (which might already have prefix)

        # Correctly identify the base signal column and its EMA column
        base_signal_col_name = avg_str # e.g., "TGMA_H2"
        # Construct the expected EMA column name, e.g., "TGMA_ema3H2" from "TGMA_H2"
        if "TGMA_" in base_signal_col_name: # avg_str is "TGMA_H2", "TGMA_H1", etc.
            ema_signal_col_name = base_signal_col_name.replace("TGMA_", "TGMA_ema3") # Results in "TGMA_ema3H2"
        else:
            # This case should ideally not be hit if populate_indicators calls this function
            # with avg_str always having "TGMA_" prefix for TGMA calculations.
            logger.error(f"TGMA calc: Unexpected avg_str format: {avg_str}. Cannot reliably determine EMA column.")
            # Fill output columns with 0/NaN and return to prevent further errors
            # Construct fallback names based on original logic to ensure columns are created if expected by other parts
            gradient_col_name_fb = f"{prefix}gradient{avg_str.replace(prefix, '')}"
            buylimit_col_name_fb = f"{prefix}buylimit{avg_str.replace(prefix, '')}"
            gradient_impulse_col_name_fb = f"{prefix}gradientImpulse{avg_str.replace(prefix, '')}" # Consider making this unique per avg_str
            color_col_name_fb = f"{prefix}color{avg_str.replace(prefix, '')}"
            dataframe[gradient_col_name_fb] = 0.0
            dataframe[buylimit_col_name_fb] = 0.0
            dataframe[gradient_impulse_col_name_fb] = 0.0
            dataframe[color_col_name_fb] = 0.0
            return dataframe

        if not (base_signal_col_name in dataframe.columns and ema_signal_col_name in dataframe.columns):
            logger.warning(f"TGMA calc: Missing required columns for {avg_str}. Need {base_signal_col_name} (exists: {base_signal_col_name in dataframe.columns}) and {ema_signal_col_name} (exists: {ema_signal_col_name in dataframe.columns})")
            # Fill output columns with 0/NaN and return
            gradient_col_name_fb = f"{prefix}gradient{avg_str.replace(prefix, '')}"
            buylimit_col_name_fb = f"{prefix}buylimit{avg_str.replace(prefix, '')}"
            gradient_impulse_col_name_fb = f"{prefix}gradientImpulse{avg_str.replace(prefix, '')}"
            color_col_name_fb = f"{prefix}color{avg_str.replace(prefix, '')}"
            dataframe[gradient_col_name_fb] = 0.0
            dataframe[buylimit_col_name_fb] = 0.0
            dataframe[gradient_impulse_col_name_fb] = 0.0
            dataframe[color_col_name_fb] = 0.0
            return dataframe

        for i in range(1, len(dataframe)):
            idx_current = dataframe.index[i]
            idx_prev = dataframe.index[i-1]

            # Use the corrected column names for accessing data
            if pd.isna(dataframe.at[idx_current, base_signal_col_name]) or pd.isna(dataframe.at[idx_current, ema_signal_col_name]):
                qty_adv_dec[i] = qty_adv_dec[i - 1]
                continue

            chg = dataframe.at[idx_current, base_signal_col_name] - dataframe.at[idx_prev, base_signal_col_name]
            is_bull = dataframe.at[idx_current, base_signal_col_name] > dataframe.at[idx_current, ema_signal_col_name]
            is_bear = dataframe.at[idx_current, base_signal_col_name] < dataframe.at[idx_current, ema_signal_col_name]

            if is_bull:
                qty_adv_dec[i] = qty_adv_dec[i - 1] + 1 if chg > 0 else qty_adv_dec[i - 1] - 1
            elif is_bear:
                qty_adv_dec[i] = qty_adv_dec[i - 1] - 1 if chg < 0 else qty_adv_dec[i - 1] + 1
            else: # Neutral case, no change from previous gradient
                qty_adv_dec[i] = qty_adv_dec[i-1]

            # Corrected Clamping
            qty_adv_dec[i] = max(-steps, min(steps, qty_adv_dec[i]))

        gradient_col_name = f"{prefix}gradient{avg_str.replace(prefix, '')}"
        buylimit_col_name = f"{prefix}buylimit{avg_str.replace(prefix, '')}"
        # For gradientImpulse, it might be better to make it specific to each avg_str if it's meaningful,
        # e.g., f"{prefix}gradientImpulse{avg_str.replace(prefix, '')}"
        # Original TGMA seems to have one 'gradientImpulse', potentially recalculated/overwritten in each call.
        # For safety, let's make it specific.
        gradient_impulse_col_name = f"{prefix}gradientImpulse{avg_str.replace(prefix, '')}"
        color_col_name = f"{prefix}color{avg_str.replace(prefix, '')}"

        dataframe[gradient_col_name] = qty_adv_dec # This is already normalized by clamping
        dataframe[buylimit_col_name] = dataframe[gradient_col_name].shift(2) # Original TGMA shifts by 2

        # Calculate gradient impulse using the newly created gradient_col_name and buylimit_col_name
        dataframe[gradient_impulse_col_name] = (abs(dataframe[buylimit_col_name].shift(1) - # shift(1) on buylimit is grad.shift(3)
                                                     dataframe[gradient_col_name].shift(1)) - # grad.shift(1)
                                                 abs(dataframe[buylimit_col_name].shift(2) - # shift(2) on buylimit is grad.shift(4)
                                                     dataframe[gradient_col_name].shift(2))) # grad.shift(2)

        dataframe[color_col_name] = dataframe[gradient_col_name] # Corrected Color Column Assignment
        return dataframe

    def calculate_rolling_vvrp(self, dataframe: pd.DataFrame, lookback_period: int,
                             num_bins: int, max_bars: int) -> pd.DataFrame:
        """
        Calculate a rolling Visual Volume Range Profile (VVRP) with buy/sell volume blocks and oscillators.
        Args:
            dataframe: DataFrame with OHLCV data
            lookback_period: Number of candles to look back for the profile
            num_bins: Number of price bins (vertical resolution)
            max_bars: Maximum number of bars to extend right (visual length)
        Returns:
            DataFrame with VVRP data (total, buy, sell volumes and bars, and oscillators) added as columns
        """
        # Calculate average price
        dataframe['Average_Price'] = (dataframe['high'] + dataframe['low']) / 2 # Already present in VVRPV2, ensure it's fine if Hurst also adds it

        # Estimate buy/sell volume
        temp_price_range = dataframe['high'] - dataframe['low'] # use temp name

        # Ensure temp_price_range does not have zeros before division
        # Using a small epsilon to avoid division by zero if needed, or handle via np.where
        epsilon = 1e-9 # A very small number
        safe_price_range = np.where(temp_price_range == 0, epsilon, temp_price_range)

        dataframe['temp_buy_volume'] = np.where( # use temp name
            dataframe['close'] > dataframe['open'],
            dataframe['volume'] * (dataframe['close'] - dataframe['open']) / safe_price_range,
            0
        )
        dataframe['temp_sell_volume'] = np.where( # use temp name
            dataframe['close'] < dataframe['open'],
            dataframe['volume'] * (dataframe['open'] - dataframe['close']) / safe_price_range,
            0
        )
        # Fallback for neutral candles or zero range (where safe_price_range was epsilon)
        dataframe['temp_buy_volume'] = np.where(
            temp_price_range == 0, # Check original zero range
            dataframe['volume'] * 0.5,
            dataframe['temp_buy_volume']
        )
        dataframe['temp_sell_volume'] = np.where(
            temp_price_range == 0, # Check original zero range
            dataframe['volume'] * 0.5,
            dataframe['temp_sell_volume']
        )
        dataframe['temp_buy_volume'] = dataframe['temp_buy_volume'].fillna(0)
        dataframe['temp_sell_volume'] = dataframe['temp_sell_volume'].fillna(0)

        # Initialize output columns
        bin_labels = range(1, num_bins + 1)
        mid_price_cols = [f'VVRP_Mid_Price_{i}' for i in bin_labels]
        bar_cols = [f'VVRP_Bars_{i}' for i in bin_labels]
        volume_cols = [f'VVRP_Volume_{i}' for i in bin_labels]
        buy_volume_cols = [f'VVRP_Buy_Volume_{i}' for i in bin_labels]
        sell_volume_cols = [f'VVRP_Sell_Volume_{i}' for i in bin_labels]
        buy_bar_cols = [f'VVRP_Buy_Bars_{i}' for i in bin_labels]
        sell_bar_cols = [f'VVRP_Sell_Bars_{i}' for i in bin_labels]
        volume_diff_cols = [f'VVRP_Volume_Diff_{i}' for i in bin_labels]
        bars_diff_cols = [f'VVRP_Bars_Diff_{i}' for i in bin_labels]

        all_vvrp_cols = (mid_price_cols + bar_cols + volume_cols + buy_volume_cols +
                         sell_volume_cols + buy_bar_cols + sell_bar_cols +
                         volume_diff_cols + bars_diff_cols +
                         ['VVRP_Volume_Osc', 'VVRP_Bars_Osc'])

        for col in all_vvrp_cols:
            if col not in dataframe.columns: # Add column only if it doesn't exist
                 dataframe[col] = np.nan
            else: # If it exists, ensure it's float for future assignments (especially if it was int)
                 dataframe[col] = dataframe[col].astype(float)


        # Rolling calculation
        for i in range(lookback_period - 1, len(dataframe)):
            current_index = dataframe.index[i]
            window_start_index = dataframe.index[max(0, i - lookback_period + 1)]
            window = dataframe.loc[window_start_index:current_index].copy()

            hi = window['high'].max()
            lo = window['low'].min()
            width = hi - lo

            if width == 0:
                # If width is 0, set NaNs for this row for VVRP columns and continue
                for col_name in all_vvrp_cols:
                    dataframe.at[current_index, col_name] = np.nan
                continue

            bin_width = width / num_bins
            # Ensure Price_Bins are created correctly, handling potential issues with cut
            try:
                window['Price_Bins'] = pd.cut(window['Average_Price'], bins=num_bins,
                                             labels=bin_labels, include_lowest=True, right=True) # Added right=True for consistency
            except ValueError: # Happens if all values are the same or other edge cases
                 for col_name in all_vvrp_cols:
                    dataframe.at[current_index, col_name] = np.nan
                 continue


            grouped = window.groupby('Price_Bins', observed=True).agg({
                'volume': 'sum',
                'temp_buy_volume': 'sum',
                'temp_sell_volume': 'sum'
            }).reindex(range(1, num_bins + 1), fill_value=0) # Use temp names here

            volume_bins = grouped['volume'].values
            buy_volume_bins = grouped['temp_buy_volume'].values
            sell_volume_bins = grouped['temp_sell_volume'].values

            max_volume_val = volume_bins.max() if volume_bins.max() > 0 else 1 # Renamed from max_volume
            bars = [int(np.round((vol / max_volume_val) * max_bars)) for vol in volume_bins]
            buy_bars = [int(np.round((vol / max_volume_val) * max_bars)) for vol in buy_volume_bins]
            sell_bars = [int(np.round((vol / max_volume_val) * max_bars)) for vol in sell_volume_bins]

            volume_diff = buy_volume_bins - sell_volume_bins
            bars_diff = np.array(buy_bars) - np.array(sell_bars)

            volume_osc = volume_diff.sum()
            bars_osc = bars_diff.sum()

            starting_point = lo # Adjusted: mid_price should be lo + (bin_idx - 0.5) * bin_width
            mid_prices = [starting_point + (bin_idx - 0.5) * bin_width for bin_idx in bin_labels]

            for j, bin_label_val in enumerate(bin_labels): # Renamed bin_label to bin_label_val
                dataframe.at[current_index, f'VVRP_Mid_Price_{bin_label_val}'] = mid_prices[j]
                dataframe.at[current_index, f'VVRP_Volume_{bin_label_val}'] = volume_bins[j]
                dataframe.at[current_index, f'VVRP_Bars_{bin_label_val}'] = bars[j]
                dataframe.at[current_index, f'VVRP_Buy_Volume_{bin_label_val}'] = buy_volume_bins[j]
                dataframe.at[current_index, f'VVRP_Sell_Volume_{bin_label_val}'] = sell_volume_bins[j]
                dataframe.at[current_index, f'VVRP_Buy_Bars_{bin_label_val}'] = buy_bars[j]
                dataframe.at[current_index, f'VVRP_Sell_Bars_{bin_label_val}'] = sell_bars[j]
                dataframe.at[current_index, f'VVRP_Volume_Diff_{bin_label_val}'] = volume_diff[j]
                dataframe.at[current_index, f'VVRP_Bars_Diff_{bin_label_val}'] = bars_diff[j]
            dataframe.at[current_index, 'VVRP_Volume_Osc'] = volume_osc
            dataframe.at[current_index, 'VVRP_Bars_Osc'] = bars_osc

        dataframe.drop(columns=['Average_Price', 'temp_buy_volume', 'temp_sell_volume'], errors='ignore', inplace=True)

        return dataframe

    def visualize_rolling_vvrp(self, dataframe: pd.DataFrame, index: int, pair: str, num_bins: int):
        """
        Visualize the VVRP with buy/sell volume blocks and oscillators for a specific candle index.
        Args:
            dataframe: DataFrame with VVRP data
            index: Index of the candle to visualize (integer location)
            pair: Trading pair name
            num_bins: Number of price bins
        """
        # Check if the index is valid
        if index < 0 or index >= len(dataframe):
            # logger.warning(f"Index {index} is out of bounds for visualize_rolling_vvrp.")
            return

        actual_index = dataframe.index[index] # Get the actual DataFrame index label

        # Check if VVRP data exists for this row, e.g. by checking a sentinel column
        if not (f'VVRP_Mid_Price_1' in dataframe.columns and pd.notnull(dataframe.at[actual_index, f'VVRP_Mid_Price_1'])):
            # logger.info(f"No VVRP data to visualize for {pair} at index {index} (actual index: {actual_index})")
            return

        bin_labels = range(1, num_bins + 1)
        # logger.info(f"\nRolling VVRP for {pair} at index {index} (Time: {dataframe.at[actual_index, 'date']})")
        # logger.info("-" * 90)
        # logger.info("Price   | Buy Bars   | Sell Bars  | Vol Diff | Bars Diff | Total Vol | Buy Vol  | Sell Vol")
        # logger.info("-" * 90)

        for i in bin_labels:
            price = dataframe.at[actual_index, f'VVRP_Mid_Price_{i}']
            buy_bars = int(dataframe.at[actual_index, f'VVRP_Buy_Bars_{i}']) if pd.notnull(dataframe.at[actual_index, f'VVRP_Buy_Bars_{i}']) else 0
            sell_bars = int(dataframe.at[actual_index, f'VVRP_Sell_Bars_{i}']) if pd.notnull(dataframe.at[actual_index, f'VVRP_Sell_Bars_{i}']) else 0
            volume_diff = dataframe.at[actual_index, f'VVRP_Volume_Diff_{i}'] if pd.notnull(dataframe.at[actual_index, f'VVRP_Volume_Diff_{i}']) else 0
            bars_diff = int(dataframe.at[actual_index, f'VVRP_Bars_Diff_{i}']) if pd.notnull(dataframe.at[actual_index, f'VVRP_Bars_Diff_{i}']) else 0
            volume = dataframe.at[actual_index, f'VVRP_Volume_{i}'] if pd.notnull(dataframe.at[actual_index, f'VVRP_Volume_{i}']) else 0
            buy_volume = dataframe.at[actual_index, f'VVRP_Buy_Volume_{i}'] if pd.notnull(dataframe.at[actual_index, f'VVRP_Buy_Volume_{i}']) else 0
            sell_volume = dataframe.at[actual_index, f'VVRP_Sell_Volume_{i}'] if pd.notnull(dataframe.at[actual_index, f'VVRP_Sell_Volume_{i}']) else 0

            if pd.notnull(price):
                buy_bar_str = "#" * min(buy_bars // 5, 10) + " " * (10 - min(buy_bars // 5, 10))
                sell_bar_str = "#" * min(sell_bars // 5, 10) + " " * (10 - min(sell_bars // 5, 10))
                # logger.info(f"{price:<7.4f} | {buy_bar_str} | {sell_bar_str} | {volume_diff:<8.2f} | {bars_diff:<9d} | "
                #       f"{volume:<9.2f} | {buy_volume:<8.2f} | {sell_volume:.2f}")

        volume_osc = dataframe.at[actual_index, 'VVRP_Volume_Osc'] if pd.notnull(dataframe.at[actual_index, 'VVRP_Volume_Osc']) else 0
        bars_osc = dataframe.at[actual_index, 'VVRP_Bars_Osc'] if pd.notnull(dataframe.at[actual_index, 'VVRP_Bars_Osc']) else 0
        # logger.info(f"Oscillators: Volume_Osc={volume_osc:.2f}, Bars_Osc={bars_osc:.2f}")
        # logger.info("-" * 90)