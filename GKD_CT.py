from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                               IStrategy, IntParameter, RealParameter, merge_informative_pair, informative)
from pandas_ta import ema, sma, wma, hma, tema, dema, linreg, vwma
import talib.abstract as ta
import pandas as pd
import numpy as np
import talib
import logging
from typing import Dict, List
from pandas import DataFrame
from technical import qtpylib
from scipy.fft import fft
from scipy.signal import hilbert
from datetime import datetime, timedelta, timezone
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_seconds
from datetime import datetime, timedelta, timezone
from freqtrade.persistence import Trade
from typing import Any, Dict, List, Optional, Tuple, Union

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class GKD_CT(IStrategy):


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
    ### Inspired by [loxx] GKD trading system on TradingView

    # Strategy parameters
    can_short = False
    timeframe = "1h"
    minimal_roi = {}
    locked_stoploss = {}
    stoploss = -0.10
    trailing_stop = False
    use_custom_stoploss = True
    position_adjustment_enable = True

    # Plot configuration
    plot_config = {
        "main_plot": {
        "baseline": {
          "color": "#0ad5a7"
        },
        "goldie_min": {
          "color": "#eaf4f0",
          "type": "line"
        },
        "goldie_max": {
          "color": "#001d86",
          "type": "line"
        }
        },
        "subplots": {
        "hurst": {
          "hurst_smooth": {
            "color": "#3efc34"
          },
          "hurst_trending": {
            "color": "#bcfaa6",
            "type": "line"
          },
          "hurst_mean_rev": {
            "color": "#82372f",
            "type": "line"
          }
        },
        "pfe": {
          "pfe": {
            "color": "#2776ab"
          },
          "pfe_smooth": {
            "color": "#bdce11"
          },
          "pfe_buy_threshold": {
            "color": "#367429"
          },
          "pfe_sell_threshold": {
            "color": "#fd0481"
          }
        },
        "volameter": {
          "dv_val": {
            "color": "#1e4dd6"
          }
        },
        "fisher": {
          "fisher": {
            "color": "#a29a76"
          },
          "fisher_smooth": {
            "color": "#b415f1"
          },
          "fisher_buy_threshold": {
            "color": "#7ea70b",
            "type": "line"
          },
          "fisher_sell_threshold": {
            "color": "#de7b02",
            "type": "line"
          }
        },
        "rex": {
          "rex": {
            "color": "#50b523"
          },
          "rex_signal": {
            "color": "#b8fcb5"
          }
        }
        }
        }

    # Hyperopt parameters
    u_window_size = IntParameter(60, 150, default=100, space='buy', optimize=True, load=True)
    l_window_size = IntParameter(20, 40, default=30, space='buy', optimize=True, load=True)
    ma_type = CategoricalParameter(
        ["AMA", "ADXvma", "Ahrens", "ALXMA", "Donchian", "DEMA", "DSEMA", "DSFEMA", "DSRWEMA", "DSWEMA",
         "DWMA", "EOTF", "EMA", "FEMA", "FRAMA", "GDEMA", "GDDEMA", "HMA1", "HMA2", "HMA3", "HMA4",
         "IE2", "ILRS", "Instantaneous", "Kalman", "KAMA", "Laguerre", "Leader", "LSMA", "LWMA",
         "McGinley", "McNicholl", "NonLag", "ONMAMA", "OMA", "Parabolic", "PDFMA", "QRMA", "REMA",
         "RWEMA", "Recursive", "SDEC", "SJMA", "SMA", "Sine", "SLWMA", "SMMA", "Smoother",
         "SuperSmoother", "T3", "TMA", "TEMA",
         "VIDYA", "VMA", "VEMA", "VWMA", "ZeroLagDEMA",
         "ZeroLagMA", "ZeroLagTEMA"],
        default="EMA", space="buy"
    )

    threshold_level = DecimalParameter(1.0, 2.0, default=1.4, decimals=2, space="buy")
    pfe_smooth = IntParameter(3, 10, default=5, space="buy")
    pfe_buy_threshold = IntParameter(-95, -20, default=-30, space="buy")
    pfe_sell_threshold = IntParameter(20, 50, default=30, space="sell")
    fisher_smooth = IntParameter(3, 10, default=5, space="buy")
    fisher_sell_threshold = DecimalParameter(0.5, 3.9, default=3.0, decimals=2, space="sell")
    fisher_buy_threshold = DecimalParameter(-3.9, 1.0, default=-0.5, decimals=2, space="buy")
    hurst_smooth_period = IntParameter(3, 10, default=5, space="buy")
    hurst_trending = DecimalParameter(0.5, 0.55, default=0.525, decimals=2, space="buy")
    hurst_mean_rev = DecimalParameter(0.45, 0.50, default=0.475, decimals=2, space="buy")
    goldie_locks = DecimalParameter(0.8, 3.0, default=1.0, decimals=1, space="buy")
    h2_ob = IntParameter(50, 70, default=60, space="sell")
    h1_ob = IntParameter(50, 70, default=60, space="sell")
    h0_ob = IntParameter(50, 70, default=60, space="sell")
    cp_ob = IntParameter(50, 70, default=60, space="sell")
    h2_os = IntParameter(30, 50, default=40, space="buy")
    h1_os = IntParameter(30, 50, default=40, space="buy")
    h0_os = IntParameter(30, 50, default=40, space="buy")
    cp_os = IntParameter(30, 50, default=40, space="buy")
    market_length = IntParameter(5, 30, default=21, space="buy")
    use_tsl1 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use_tsl2 = BooleanParameter(default=True, space="sell", optimize=True, load=True)

    # Fixed parameters
    lag_suppressor = True
    lag_s_k = 0.5

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if self.can_short == True:
            lev = self.lev_X.value
            lev = 3
        else:
            lev = 1

        SLT0 = current_candle['h2_move_mean'] * lev
        SLT1 = current_candle['h1_move_mean'] * lev
        SLT2 = current_candle['h0_move_mean'] * lev
        SLT3 = current_candle['cycle_move_mean'] * lev
        
        display_profit = current_profit * 100

        if current_profit < -0.01:
            if pair in self.locked_stoploss:
                del self.locked_stoploss[pair]
                if (self.dp.runmode.value in ('live', 'dry_run')):
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
        elif SLT1 is not None and current_profit > SLT1 and self.use_tsl2.value == True:
            new_stoploss = (SLT1 - SLT0)
            level = 2
        elif SLT0 is not None and current_profit > SLT0 and self.use_tsl1.value == True:
            new_stoploss = (SLT1 - SLT0)
            level = 1

        if new_stoploss is not None:
            if pair not in self.locked_stoploss or new_stoploss > self.locked_stoploss[pair]:
                self.locked_stoploss[pair] = new_stoploss
                if (self.dp.runmode.value in ('live', 'dry_run')):
                    self.dp.send_msg(f'*** {pair} *** Profit {level} {display_profit:.3f}%% - New stoploss: {new_stoploss:.4f} activated')
                    logger.info(f'*** {pair} *** Profit {level} {display_profit:.3f}%% - New stoploss: {new_stoploss:.4f} activated')
            return self.locked_stoploss[pair]

        return self.stoploss

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        # filled_entries = trade.select_filled_orders(trade.entry_side)
        # count_of_entries = trade.nr_of_successful_entries
        # trade_duration = (current_time - trade.open_date_utc).seconds / 60
        # last_fill = (current_time - trade.date_last_filled_utc).seconds / 60 

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

        if current_profit is not None:
            logger.info(f"{trade.pair} - 💰 Current Profit: {display_profit:.3}% TP0: {tp0:.3}% | TP1: {tp1:.3}% | TP2: {tp2:.3}% | TP3: {tp3:.3}%")

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
            
        # DCA
        # if trade.nr_of_successful_entries == self.max_epa.value + 1:
        #     return None 
        # # Block concurrent buys and Hold otherwise if the dip is not large enough
        # if current_profit > -TP1:
        #     return None

        # try:
        #     # This returns first order stake size 
        #     # Modify the following parameters to enable more levels or different buy size:
        #     # max_entry_position_adjustment = 3 
        #     # max_dca_multiplier = 3.5 

        #     stake_amount = filled_entries[0].cost
        #     # This then calculates current safety order size when a secondary buy signal is generated.
        #     if (last_fill > self.filldelay.value):
        #         if (signal == 1 and current_profit < -TP0):
        #             if count_of_entries >= 1: 
        #                 stake_amount = stake_amount * 2
        #             else:
        #                 stake_amount = stake_amount

        #             return stake_amount

        #     # This then calculates current safety order size when below -Take Profit 1.
        #     if (last_fill > self.filldelay.value):
        #         if current_profit < -TP0:
        #             if count_of_entries >= 1: 
        #                 stake_amount = stake_amount * 1.5
        #             else:
        #                 stake_amount = stake_amount

        #             return stake_amount

        #     # This accommadates a one shot at buying the dip on a big wick with one 
        #     # large buy if the funds are available...
        #     if (last_fill > self.filldelay.value):
        #         if (current_profit < -TP3):
        #             if count_of_entries == 1: 
        #                 stake_amount = stake_amount * 4
        #             else:
        #                 stake_amount = stake_amount

        #             return stake_amount

        # except Exception as exception:
        #     return None

        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

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
        # --- Baseline Indicator ---
        # Purpose: Determines trend direction using a single, hyperoptable moving average.
        # Calculation: Selected MA (e.g., EMA, KAMA) with configurable period, trend direction via difference.
        # Role: Filters trades to align with the trend, per NNFX and GKD requirements.
        ma_functions = {
            "AMA": self.ama, "ADXvma": self.adxvma, "Ahrens": self.ahrens, "ALXMA": lambda df, p: wma(df["close"], length=7),
            "Donchian": self.donchian, "DEMA": lambda df, p: dema(df["close"], length=p),
            "DSEMA": self.dsema, "DSFEMA": self.dsfema, "DSRWEMA": self.dsrwema, "DSWEMA": self.dswema,
            "DWMA": self.dwma, "EOTF": lambda df, p: ema(df["close"], length=p), "EMA": lambda df, p: ema(df["close"], length=p),
            "FEMA": lambda df, p: ema(df["close"], length=int(p/2)), "FRAMA": self.frama, "GDEMA": self.gdema,
            "GDDEMA": self.gddema, "HMA1": lambda df, p: hma(df["close"], length=p), "HMA2": self.hma_ema,
            "HMA3": self.hma_wma, "HMA4": self.hma_smma, "IE2": self.t3, "ILRS": self.ilrs,
            "Instantaneous": lambda df, p: ema(df["close"], length=p), "Kalman": self.kalman, "KAMA": self.kama,
            "Laguerre": self.laguerre, "Leader": self.leader_ema, "LSMA": lambda df, p: linreg(df["close"], length=p),
            "LWMA": lambda df, p: wma(df["close"], length=p), "McGinley": self.mcginley, "McNicholl": lambda df, p: ema(df["close"], length=p),
            "NonLag": self.nonlag, "ONMAMA": lambda df, p: ema(df["close"], length=p), "OMA": self.oma,
            "Parabolic": self.parabolic_wma, "PDFMA": self.pdfma, "QRMA": self.qrma, "REMA": self.rema,
            "RWEMA": self.rwema, "Recursive": self.recursive, "SDEC": self.sdec, "SJMA": lambda df, p: ema(df["close"], length=p),
            "SMA": lambda df, p: sma(df["close"], length=p), "Sine": self.sine_wma, "SLWMA": self.slwma,
            "SMMA": self.smma, "Smoother": self.smoother, "SuperSmoother": self.super_smoother, "T3": self.t3,
            "TMA": self.tma, "TEMA": lambda df, p: tema(df["close"], length=p), "VIDYA": self.vidya, "VMA": self.vma, "VEMA": self.vema,
            "VWMA": lambda df, p: vwma(df["close"], df["volume"], length=p), "ZeroLagDEMA": self.zero_lag_dema,
            "ZeroLagMA": self.zero_lag_ma, "ZeroLagTEMA": self.zero_lag_tema
        }
        if self.ma_type.value in ma_functions:
            dataframe["baseline"] = ma_functions[self.ma_type.value](dataframe, self.h2)
        else:
            raise ValueError(f"MA type {self.ma_type.value} not implemented")
        dataframe["baseline_diff"] = dataframe["baseline"].diff()
        dataframe["baseline_up"] = dataframe["baseline_diff"] > 0
        dataframe["baseline_down"] = dataframe["baseline_diff"] < 0

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
        dataframe['move'] = (dataframe['cycle_move'] + dataframe['h0_move'] + dataframe['h1_move'] + dataframe['h2_move']) / 4 
        dataframe['cycle_move_mean'] = dataframe['cycle_move'].rolling(self.cp).mean()
        dataframe['h0_move_mean'] = dataframe['h0_move'].rolling(self.cp).mean()
        dataframe['h1_move_mean'] = dataframe['h1_move'].rolling(self.cp).mean()
        dataframe['h2_move_mean'] = dataframe['h2_move'].rolling(self.cp).mean()
        dataframe['move_mean'] = (dataframe['cycle_move_mean'] + dataframe['h0_move_mean'] + dataframe['h1_move_mean'] + dataframe['h2_move_mean']) / 4
        dataframe['move_mean_min'] = dataframe['h2_move_mean'].min() / 2
        dataframe['entry_limit'] = dataframe['ha_close'].rolling(self.cp).max() * (1 - dataframe['move_mean_min'])
        dataframe['entry_limit_lower'] = dataframe['ha_close'].rolling(self.cp).min() * (1 + dataframe['move_mean_min'])

        # --- Polarized Fractal Efficiency (PFE) Indicator ---
        # Purpose: Confirms trend efficiency as Confirmation 1.
        # Calculation: PFE = 100 * (straight-line distance / path length), polarized, smoothed with EMA for multiple harmonics.
        # Role: Signals strong trend efficiency for entries.
        dataframe["pfe_h2"] = self.calculate_pfe(dataframe, self.h2)
        dataframe["pfe_h1"] = self.calculate_pfe(dataframe, self.h1)
        dataframe["pfe_h0"] = self.calculate_pfe(dataframe, self.h0)
        dataframe["pfe_cp"] = self.calculate_pfe(dataframe, self.cp)
        dataframe["pfe_smooth_h2"] = ema(dataframe["pfe_h2"], length=self.pfe_smooth.value)
        dataframe["pfe_smooth_h1"] = ema(dataframe["pfe_h1"], length=self.pfe_smooth.value)
        dataframe["pfe_smooth_h0"] = ema(dataframe["pfe_h0"], length=self.pfe_smooth.value)
        dataframe["pfe_smooth_cp"] = ema(dataframe["pfe_cp"], length=self.pfe_smooth.value)
        dataframe["pfe_smooth_avg"] = dataframe[["pfe_smooth_h2", "pfe_smooth_h1", "pfe_smooth_h0", "pfe_smooth_cp"]].mean(axis=1)
        dataframe["pfe_buy_threshold"] = self.pfe_buy_threshold.value
        dataframe["pfe_sell_threshold"] = self.pfe_sell_threshold.value

        # --- Fisher Transform Indicator ---
        # Purpose: Confirms trend reversals as Confirmation 2.
        # Calculation: Fisher Transform = 0.5 * ln((1 + norm) / (1 - norm)), norm is scaled median price, smoothed with EMA for multiple harmonics.
        # Role: Enhances entry signals with reversal detection.
        dataframe["fisher_h2"] = self.calculate_fisher(dataframe, self.h2)
        dataframe["fisher_h1"] = self.calculate_fisher(dataframe, self.h1)
        dataframe["fisher_h0"] = self.calculate_fisher(dataframe, self.h0)
        dataframe["fisher_cp"] = self.calculate_fisher(dataframe, self.cp)
        dataframe["fisher_avg"] = dataframe[["fisher_h2", "fisher_h1", "fisher_h0", "fisher_cp"]].mean(axis=1)
        dataframe["fisher_smooth_h2"] = ema(dataframe["fisher_h2"], length=self.fisher_smooth.value)
        dataframe["fisher_smooth_h1"] = ema(dataframe["fisher_h1"], length=self.fisher_smooth.value)
        dataframe["fisher_smooth_h0"] = ema(dataframe["fisher_h0"], length=self.fisher_smooth.value)
        dataframe["fisher_smooth_cp"] = ema(dataframe["fisher_cp"], length=self.fisher_smooth.value)
        dataframe["fisher_smooth_avg"] = dataframe[["fisher_smooth_h2", "fisher_smooth_h1", "fisher_smooth_h0", "fisher_smooth_cp"]].mean(axis=1)
        dataframe["fisher_buy_threshold"] = self.fisher_buy_threshold.value
        dataframe["fisher_sell_threshold"] = self.fisher_sell_threshold.value

        # --- Hurst Exponent Indicator ---
        # Purpose: Confirms persistent trending behavior.
        # Calculation: H = log(R/S) / log(n) on log-returns, smoothed with EMA for multiple harmonics, with variance-based fallback.
        # Role: Ensures trades occur in persistent market conditions.
        dataframe["log_return"] = np.log(dataframe["close"] / dataframe["close"].shift(1))
        dataframe["hurst_h2"] = self.calculate_hurst(dataframe["log_return"], self.h2)
        dataframe["hurst_h1"] = self.calculate_hurst(dataframe["log_return"], self.h1)
        dataframe["hurst_h0"] = self.calculate_hurst(dataframe["log_return"], self.h0)
        dataframe["hurst_cp"] = self.calculate_hurst(dataframe["log_return"], self.cp)
        dataframe["hurst_smooth_h2"] = ema(dataframe["hurst_h2"], length=self.hurst_smooth_period.value)
        dataframe["hurst_smooth_h1"] = ema(dataframe["hurst_h1"], length=self.hurst_smooth_period.value)
        dataframe["hurst_smooth_h0"] = ema(dataframe["hurst_h0"], length=self.hurst_smooth_period.value)
        dataframe["hurst_smooth_cp"] = ema(dataframe["hurst_cp"], length=self.hurst_smooth_period.value)
        dataframe["hurst_smooth_avg"] = dataframe[["hurst_smooth_h2", "hurst_smooth_h1", "hurst_smooth_h0", "hurst_smooth_cp"]].mean(axis=1)
        dataframe['hurst_trending'] = self.hurst_trending.value
        dataframe['hurst_mean_rev'] = self.hurst_mean_rev.value


        # Numerical Filter for filtered_close
        filter_weights = [1, 2, 4, 8, 4]
        close = dataframe['ha_close'].values
        weights = np.array(filter_weights) / sum(filter_weights)
        filtered = np.convolve(close, weights, mode='valid')
        dataframe['filtered_close'] = pd.Series(filtered, index=dataframe.index[-len(filtered):])

        # Troughs and Crests
        dataframe['trough'] = dataframe['filtered_close'].rolling(self.h2).min()
        dataframe['crest'] = dataframe['filtered_close'].rolling(self.h2).max()
        dataframe['is_trough'] = np.where(dataframe['filtered_close'] == dataframe['trough'], 1, 0)
        dataframe['is_crest'] = np.where(dataframe['filtered_close'] == dataframe['crest'], 1, 0)

        # VTL based on last two troughs/crests
        dataframe['vtl_up'] = np.nan
        dataframe['vtl_down'] = np.nan


        # Idea Smart VTL's change periods, but how????
        # Group troughs and crests within a window
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
            if trough_groups[i-1] in dataframe.index and trough_groups[i] in dataframe.index:
                x1 = dataframe.index.get_loc(trough_groups[i-1])
                y1 = dataframe.at[trough_groups[i-1], 'filtered_close']
                x2 = dataframe.index.get_loc(trough_groups[i])
                y2 = dataframe.at[trough_groups[i], 'filtered_close']
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                intercept = y1 - slope * x1
                up_vtl_segments.append([trough_groups[i-1], None, slope, intercept])
            else:
                logger.warning(f"Invalid trough group indices at i={i}: {trough_groups[i-1]}, {trough_groups[i]} not in dataframe index")

        # Create VTL segments for crest groups
        for i in range(1, len(crest_groups)):
            if crest_groups[i-1] in dataframe.index and crest_groups[i] in dataframe.index:
                x1 = dataframe.index.get_loc(crest_groups[i-1])
                y1 = dataframe.at[crest_groups[i-1], 'filtered_close']
                x2 = dataframe.index.get_loc(crest_groups[i])
                y2 = dataframe.at[crest_groups[i], 'filtered_close']
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                intercept = y1 - slope * x1
                down_vtl_segments.append([crest_groups[i-1], None, slope, intercept])
            else:
                logger.warning(f"Invalid crest group indices at i={i}: {crest_groups[i-1]}, {crest_groups[i]} not in dataframe index")

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
                        if current_price < vtl_value and i + 1 < len(up_vtl_segments):
                            segment[1] = idx
                    else:
                        segment[1] = idx if end_idx is None else end_idx
            
            # Apply vtl_down
            for i, segment in enumerate(down_vtl_segments):
                start_idx, end_idx, slope, intercept = segment
                if idx >= start_idx and (end_idx is None or idx <= end_idx):
                    vtl_value = slope * current_x + intercept
                    if price_min <= vtl_value <= price_max:
                        dataframe.at[idx, 'vtl_down'] = vtl_value
                        if current_price > vtl_value and i + 1 < len(down_vtl_segments):
                            segment[1] = idx
                    else:
                        segment[1] = idx if end_idx is None else end_idx


        # Linear Regression Channel for RSI
        dataframe['rsi_h2'] = ta.RSI(dataframe['close'], timeperiod=self.h2)
        regression_channel2 = self.linear_regression_channel(dataframe['rsi_h2'], window=self.h2, num_dev=1.0)
        dataframe['lr_mid_rsi_h2'] = regression_channel2['mid']
        dataframe['rsi_h1'] = ta.RSI(dataframe['close'], timeperiod=self.h1)
        regression_channel1 = self.linear_regression_channel(dataframe['rsi_h1'], window=self.h1, num_dev=1.0)
        dataframe['lr_mid_rsi_h1'] = regression_channel1['mid']
        dataframe['rsi_h0'] = ta.RSI(dataframe['close'], timeperiod=self.h0)
        regression_channel0 = self.linear_regression_channel(dataframe['rsi_h0'], window=self.h0, num_dev=1.0)
        dataframe['lr_mid_rsi_h0'] = regression_channel0['mid']
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=self.cp)
        regression_channel = self.linear_regression_channel(dataframe['rsi'], window=self.cp, num_dev=1.0)
        dataframe['lr_mid_rsi'] = regression_channel['mid']

        h = self.h2
        r = 8.0
        x_0 = self.cp
        smoothColors = False
        lag = 0

        # # Linear Regression Channel for RSI using Nadaraya-Watson
        # dataframe['rsi_h2'] = ta.RSI(dataframe['close'], timeperiod=self.h2)
        # nadaraya_watson(dataframe, 'rsi_h2', h, r, x_0, smoothColors, lag, mult=2.5)
        # dataframe['lr_mid_rsi_h2'] = dataframe[f'yhat1_{h}']

        # h = self.h1
        # x_0 = self.cp
        # dataframe['rsi_h1'] = ta.RSI(dataframe['close'], timeperiod=self.h1)
        # nadaraya_watson(dataframe, 'rsi_h1', h, r, x_0, smoothColors, lag, mult=2.5)
        # dataframe['lr_mid_rsi_h1'] = dataframe[f'yhat1_{h}']

        # h = self.h0
        # x_0 = self.cp
        # dataframe['rsi_h0'] = ta.RSI(dataframe['close'], timeperiod=self.h0)
        # nadaraya_watson(dataframe, 'rsi_h0', h, r, x_0, smoothColors, lag, mult=2.5)
        # dataframe['lr_mid_rsi_h0'] = dataframe[f'yhat1_{h}']

        # h = self.cp
        # x_0 = self.cp
        # dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=self.cp)
        # nadaraya_watson(dataframe, 'rsi', h, r, x_0, smoothColors, lag, mult=2.5)
        # dataframe['lr_mid_rsi'] = dataframe[f'yhat1_{h}']


        # --- Volatility Filter (Goldie Locks Zone) ---
        # Purpose: Ensures trades occur within acceptable volatility ranges.
        # Calculation: ATR-based bands around the baseline.
        # Role: Filters out extreme volatility or stagnation.
        dataframe["atr"] = talib.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=self.h2
        )
        dataframe["goldie_min"] = dataframe["baseline"] - (dataframe["atr"] * self.goldie_locks.value)
        dataframe["goldie_max"] = dataframe["baseline"] + (dataframe["atr"] * self.goldie_locks.value)
        dataframe['above_goldie_max'] = dataframe["close"] > dataframe['goldie_max']
        dataframe["below_goldie_min"] = dataframe["goldie_min"] > dataframe['close'] 

        # Long Conditions
        dataframe['pfe_bull_cross'] = 0
        dataframe['pfe_rising'] = 0
        dataframe['fisher_bull'] = 0
        dataframe['fisher_cross_bull'] = 0
        dataframe['fisher_rising'] = 0
        dataframe['fisher_extreme_avg_buy'] = 0
        dataframe['mean_reversion_avg_buy'] = 0
        dataframe['hurst_trend_bull'] = 0
        dataframe['rsi_bull_cross'] = 0
        dataframe['lr_rising'] = 0
        dataframe['below_goldie_min_bull'] = 0 
        dataframe["above_goldie_max_bull"] = 0 
        dataframe['ha_bull'] = 0
        # New Market Score Conditions
        dataframe['ha_trend_continuation_bull'] = 0

        # Average Conditions
        dataframe.loc[
            ((dataframe["pfe_smooth_avg"] > 0) & (dataframe["pfe_smooth_avg"].shift() < 0)),
            "pfe_bull_cross"
        ] = 1

        dataframe.loc[
            (
                (dataframe["pfe_smooth_avg"] > self.pfe_buy_threshold.value) & 
                (dataframe["pfe_smooth_avg"].shift() < dataframe["pfe_smooth_avg"])
            ),
            "pfe_rising"
        ] = 1

        dataframe.loc[
            (dataframe["fisher_smooth_avg"] > 0),
            "fisher_bull"
        ] = 1

        dataframe.loc[
            ((dataframe["fisher_smooth_avg"] < dataframe['fisher_avg']) & 
             (dataframe["fisher_smooth_avg"].shift() > dataframe['fisher_avg'].shift())),
            "fisher_cross_bull"
        ] = 1

        dataframe.loc[
            (
                (dataframe["fisher_smooth_avg"].shift() < dataframe["fisher_smooth_avg"])
            ),
            "fisher_rising"
        ] = 1

        dataframe.loc[
            (dataframe["fisher_smooth_avg"] < self.fisher_buy_threshold.value),
            "fisher_extreme_avg_buy"
        ] = 1

        dataframe.loc[
            (
                (dataframe["hurst_smooth_avg"] < self.hurst_mean_rev.value) &
                (dataframe['close'] < dataframe['baseline'])  # Corrected to > for buy
            ),
            "mean_reversion_avg_buy"
        ] = 1

        # dataframe.loc[
        #     (
        #         (dataframe["hurst_smooth_avg"] > self.hurst_trending.value) &
        #         (dataframe['close'] < dataframe['baseline'])
        #     ),
        #     "hurst_trend_bull"
        # ] = 1

        dataframe.loc[
            ((dataframe["lr_mid_rsi"] < dataframe['rsi']) & 
             (dataframe["lr_mid_rsi"].shift() > dataframe['rsi'].shift())),
            "rsi_bull_cross"
        ] = 1

        dataframe.loc[
            (dataframe['lr_mid_rsi'] > dataframe['lr_mid_rsi'].shift()),
            "lr_rising"
        ] = 1

        dataframe.loc[
            ((dataframe["close"] > dataframe['goldie_max']) & 
             (dataframe["ha_close"] > dataframe['ha_open'])),
            "above_goldie_max_bull"
        ] = 1

        dataframe.loc[
            ((dataframe["close"] < dataframe['goldie_min']) & 
             (dataframe["ha_close"] > dataframe['ha_open'])),
            "below_goldie_min_bull"
        ] = 1

        # dataframe.loc[
        #     ((dataframe["ha_close"] > dataframe['ha_open'])),
        #     "ha_bull"
        # ] = 1

        # # New Average Conditions
        # dataframe.loc[
        #     (dataframe['ha_bull'].rolling(self.h2).sum() >= 3),
        #     "ha_trend_continuation_bull"
        # ] = 1

        # Harmonic and CP Conditions
        dataframe['pfe_bull_cross_h2'] = 0
        dataframe['pfe_bull_cross_h1'] = 0
        dataframe['pfe_bull_cross_h0'] = 0
        dataframe['pfe_bull_cross_cp'] = 0
        dataframe['pfe_rising_h2'] = 0
        dataframe['pfe_rising_h1'] = 0
        dataframe['pfe_rising_h0'] = 0
        dataframe['pfe_rising_cp'] = 0
        dataframe['fisher_bull_h2'] = 0
        dataframe['fisher_bull_h1'] = 0
        dataframe['fisher_bull_h0'] = 0
        dataframe['fisher_bull_cp'] = 0
        dataframe['fisher_cross_h2'] = 0
        dataframe['fisher_cross_h1'] = 0
        dataframe['fisher_cross_h0'] = 0
        dataframe['fisher_cross_cp'] = 0
        dataframe['fisher_rising_h2'] = 0
        dataframe['fisher_rising_h1'] = 0
        dataframe['fisher_rising_h0'] = 0
        dataframe['fisher_rising_cp'] = 0
        dataframe['fisher_extreme_h2_buy'] = 0
        dataframe['fisher_extreme_h1_buy'] = 0
        dataframe['fisher_extreme_h0_buy'] = 0
        dataframe['fisher_extreme_cp_buy'] = 0
        dataframe['hurst_trend_h2_bull'] = 0
        dataframe['hurst_trend_h1_bull'] = 0
        dataframe['hurst_trend_h0_bull'] = 0
        dataframe['hurst_trend_cp_bull'] = 0
        dataframe['hurst_sustained_trend_h2'] = 0
        dataframe['hurst_sustained_trend_h1'] = 0
        dataframe['hurst_sustained_trend_h0'] = 0
        dataframe['hurst_sustained_trend_cp'] = 0
        dataframe['mean_reversion_h2'] = 0
        dataframe['mean_reversion_h1'] = 0
        dataframe['mean_reversion_h0'] = 0
        dataframe['mean_reversion_cp'] = 0
        dataframe['rsi_oversold_h2'] = 0
        dataframe['rsi_oversold_h1'] = 0
        dataframe['rsi_oversold_h0'] = 0
        dataframe['rsi_oversold_cp'] = 0
        dataframe['rsi_rising_h2'] = 0
        dataframe['rsi_rising_h1'] = 0
        dataframe['rsi_rising_h0'] = 0
        dataframe['rsi_rising_cp'] = 0

        # Harmonic and CP Conditions
        # dataframe.loc[
        #     ((dataframe["pfe_smooth_h2"] > 0) & (dataframe["pfe_smooth_h2"].shift() < 0)),
        #     "pfe_bull_cross_h2"
        # ] = 1
        # dataframe.loc[
        #     ((dataframe["pfe_smooth_h1"] > 0) & (dataframe["pfe_smooth_h1"].shift() < 0)),
        #     "pfe_bull_cross_h1"
        # ] = 1
        # dataframe.loc[
        #     ((dataframe["pfe_smooth_h0"] > 0) & (dataframe["pfe_smooth_h0"].shift() < 0)),
        #     "pfe_bull_cross_h0"
        # ] = 1
        dataframe.loc[
            ((dataframe["pfe_smooth_cp"] > 0) & (dataframe["pfe_smooth_cp"].shift() < 0)),
            "pfe_bull_cross_cp"
        ] = 1

        # dataframe.loc[
        #     (
        #         (dataframe["pfe_smooth_h2"] > self.pfe_buy_threshold.value) & 
        #         (dataframe["pfe_smooth_h2"].shift() < dataframe["pfe_smooth_h2"])
        #     ),
        #     "pfe_rising_h2"
        # ] = 1
        # dataframe.loc[
        #     (
        #         (dataframe["pfe_smooth_h1"] > self.pfe_buy_threshold.value) & 
        #         (dataframe["pfe_smooth_h1"].shift() < dataframe["pfe_smooth_h1"])
        #     ),
        #     "pfe_rising_h1"
        # ] = 1
        # dataframe.loc[
        #     (
        #         (dataframe["pfe_smooth_h0"] > self.pfe_buy_threshold.value) & 
        #         (dataframe["pfe_smooth_h0"].shift() < dataframe["pfe_smooth_h0"])
        #     ),
        #     "pfe_rising_h0"
        # ] = 1
        dataframe.loc[
            (
                (dataframe["pfe_smooth_cp"] > self.pfe_buy_threshold.value) & 
                (dataframe["pfe_smooth_cp"].shift() < dataframe["pfe_smooth_cp"])
            ),
            "pfe_rising_cp"
        ] = 1

        # dataframe.loc[
        #     (
        #         (dataframe["fisher_smooth_h2"] > 0) &
        #         (dataframe["fisher_smooth_h2"] < self.fisher_sell_threshold.value)  # Corrected threshold
        #     ),
        #     "fisher_bull_h2"
        # ] = 1
        # dataframe.loc[
        #     (
        #         (dataframe["fisher_smooth_h1"] > 0) &
        #         (dataframe["fisher_smooth_h1"] < self.fisher_sell_threshold.value)
        #     ),
        #     "fisher_bull_h1"
        # ] = 1
        # dataframe.loc[
        #     (
        #         (dataframe["fisher_smooth_h0"] > 0) &
        #         (dataframe["fisher_smooth_h0"] < self.fisher_sell_threshold.value)
        #     ),
        #     "fisher_bull_h0"
        # ] = 1
        dataframe.loc[
            (
                (dataframe["fisher_smooth_cp"] > 0) &
                (dataframe["fisher_smooth_cp"] < self.fisher_sell_threshold.value)
            ),
            "fisher_bull_cp"
        ] = 1

        # dataframe.loc[
        #     ((dataframe["fisher_smooth_h2"] < dataframe['fisher_h2']) & 
        #      (dataframe["fisher_smooth_h2"].shift() > dataframe['fisher_h2'].shift())),
        #     "fisher_cross_h2"
        # ] = 1
        # dataframe.loc[
        #     ((dataframe["fisher_smooth_h1"] < dataframe['fisher_h1']) & 
        #      (dataframe["fisher_smooth_h1"].shift() > dataframe['fisher_h1'].shift())),
        #     "fisher_cross_h1"
        # ] = 1
        # dataframe.loc[
        #     ((dataframe["fisher_smooth_h0"] < dataframe['fisher_h0']) & 
        #      (dataframe["fisher_smooth_h0"].shift() > dataframe['fisher_h0'].shift())),
        #     "fisher_cross_h0"
        # ] = 1
        dataframe.loc[
            ((dataframe["fisher_smooth_cp"] < dataframe['fisher_cp']) & 
             (dataframe["fisher_smooth_cp"].shift() > dataframe['fisher_cp'].shift())),
            "fisher_cross_cp"
        ] = 1

        # dataframe.loc[
        #     (
        #         (dataframe["fisher_smooth_h2"].shift() < dataframe["fisher_smooth_h2"])
        #     ),
        #     "fisher_rising_h2"
        # ] = 1
        # dataframe.loc[
        #     (
        #         (dataframe["fisher_smooth_h1"].shift() < dataframe["fisher_smooth_h1"])
        #     ),
        #     "fisher_rising_h1"
        # ] = 1
        # dataframe.loc[
        #     (
        #         (dataframe["fisher_smooth_h0"].shift() < dataframe["fisher_smooth_h0"])
        #     ),
        #     "fisher_rising_h0"
        # ] = 1
        dataframe.loc[
            (
                (dataframe["fisher_smooth_cp"].shift() < dataframe["fisher_smooth_cp"])
            ),
            "fisher_rising_cp"
        ] = 1

        # dataframe.loc[
        #     (dataframe["fisher_smooth_h2"] < self.fisher_buy_threshold.value),
        #     "fisher_extreme_h2_buy"
        # ] = 1
        # dataframe.loc[
        #     (dataframe["fisher_smooth_h1"] < self.fisher_buy_threshold.value),
        #     "fisher_extreme_h1_buy"
        # ] = 1
        # dataframe.loc[
        #     (dataframe["fisher_smooth_h0"] < self.fisher_buy_threshold.value),
        #     "fisher_extreme_h0_buy"
        # ] = 1
        dataframe.loc[
            (dataframe["fisher_smooth_cp"] < self.fisher_buy_threshold.value),
            "fisher_extreme_cp_buy"
        ] = 1

        dataframe.loc[
            (
                (dataframe["hurst_smooth_h2"] > self.hurst_trending.value) &
                (dataframe['close'] > dataframe['baseline'])
            ),
            "hurst_trend_h2_bull"
        ] = 1
        dataframe.loc[
            (
                (dataframe["hurst_smooth_h1"] > self.hurst_trending.value) &
                (dataframe['close'] > dataframe['baseline'])
            ),
            "hurst_trend_h1_bull"
        ] = 1
        dataframe.loc[
            (
                (dataframe["hurst_smooth_h0"] > self.hurst_trending.value) &
                (dataframe['close'] > dataframe['baseline'])
            ),
            "hurst_trend_h0_bull"
        ] = 1
        dataframe.loc[
            (
                (dataframe["hurst_smooth_cp"] > self.hurst_trending.value) &
                (dataframe['close'] < dataframe['baseline'])
            ),
            "hurst_trend_cp_bull"
        ] = 1

        # dataframe.loc[
        #     (dataframe["hurst_smooth_h2"].rolling(3).apply(lambda x: all(x > self.hurst_trending.value)) == 1),
        #     "hurst_sustained_trend_h2"
        # ] = 1
        # dataframe.loc[
        #     (dataframe["hurst_smooth_h1"].rolling(3).apply(lambda x: all(x > self.hurst_trending.value)) == 1),
        #     "hurst_sustained_trend_h1"
        # ] = 1
        # dataframe.loc[
        #     (dataframe["hurst_smooth_h0"].rolling(3).apply(lambda x: all(x > self.hurst_trending.value)) == 1),
        #     "hurst_sustained_trend_h0"
        # ] = 1
        # dataframe.loc[
        #     (dataframe["hurst_smooth_cp"].rolling(3).apply(lambda x: all(x > self.hurst_trending.value)) == 1),
        #     "hurst_sustained_trend_cp"
        # ] = 1

        dataframe.loc[
            (
                (dataframe["hurst_smooth_h2"] < self.hurst_mean_rev.value) &
                (dataframe['close'] < dataframe['baseline'])  # Corrected to > for buy
            ),
            "mean_reversion_h2"
        ] = 1
        dataframe.loc[
            (
                (dataframe["hurst_smooth_h1"] < self.hurst_mean_rev.value) &
                (dataframe['close'] < dataframe['baseline'])
            ),
            "mean_reversion_h1"
        ] = 1
        dataframe.loc[
            (
                (dataframe["hurst_smooth_h0"] < self.hurst_mean_rev.value) &
                (dataframe['close'] < dataframe['baseline'])
            ),
            "mean_reversion_h0"
        ] = 1
        dataframe.loc[
            (
                (dataframe["hurst_smooth_cp"] < self.hurst_mean_rev.value) &
                (dataframe['close'] < dataframe['baseline'])
            ),
            "mean_reversion_cp"
        ] = 1

        dataframe.loc[
            (dataframe['lr_mid_rsi_h2'] < self.h2_os.value),
            "rsi_oversold_h2"
        ] = 1
        # dataframe.loc[
        #     (dataframe['lr_mid_rsi_h1'] < self.h1_os.value),
        #     "rsi_oversold_h1"
        # ] = 1
        # dataframe.loc[
        #     (dataframe['lr_mid_rsi_h0'] < self.h0_os.value),
        #     "rsi_oversold_h0"
        # ] = 1
        # dataframe.loc[
        #     (dataframe['lr_mid_rsi'] < self.cp_os.value),
        #     "rsi_oversold_cp"
        # ] = 1

        dataframe.loc[
            (dataframe['lr_mid_rsi_h2'] > dataframe['lr_mid_rsi_h2'].shift()),
            "rsi_rising_h2"
        ] = 1
        # dataframe.loc[
        #     (dataframe['lr_mid_rsi_h1'] > dataframe['lr_mid_rsi_h1'].shift()),
        #     "rsi_rising_h1"
        # ] = 1
        # dataframe.loc[
        #     (dataframe['lr_mid_rsi_h0'] > dataframe['lr_mid_rsi_h0'].shift()),
        #     "rsi_rising_h0"
        # ] = 1
        # dataframe.loc[
        #     (dataframe['lr_mid_rsi'] > dataframe['lr_mid_rsi'].shift()),
        #     "rsi_rising_cp"
        # ] = 1

        # Short Conditions
        dataframe['pfe_bear_cross'] = 0
        dataframe['pfe_falling'] = 0
        dataframe['fisher_bear'] = 0
        dataframe['fisher_bear_cross'] = 0
        dataframe['fisher_falling'] = 0
        dataframe['fisher_extreme_avg_sell'] = 0
        dataframe['mean_reversion_avg_sell'] = 0
        dataframe['hurst_trend_bear'] = 0
        dataframe['rsi_bear_cross'] = 0
        dataframe['lr_falling'] = 0
        dataframe['below_goldie_min_bear'] = 0 
        dataframe["above_goldie_max_bear"] = 0 
        dataframe['ha_bear'] = 0
        # New Market Score Conditions
        dataframe['ha_trend_continuation_sell'] = 0

        # Average Conditions
        dataframe.loc[
            ((dataframe["pfe_smooth_avg"] < 0) & (dataframe["pfe_smooth_avg"].shift() > 0)),
            "pfe_bear_cross"
        ] = 1

        dataframe.loc[
            (
                (dataframe["pfe_smooth_avg"] < self.pfe_sell_threshold.value) & 
                (dataframe["pfe_smooth_avg"].shift() > dataframe["pfe_smooth_avg"])
            ),
            "pfe_falling"
        ] = 1

        dataframe.loc[
            (dataframe["fisher_smooth_avg"] < 0),
            "fisher_bear"
        ] = 1

        dataframe.loc[
            ((dataframe["fisher_smooth_avg"] > dataframe['fisher_avg']) & 
             (dataframe["fisher_smooth_avg"].shift() < dataframe['fisher_avg'].shift())),
            "fisher_bear_cross"
        ] = 1

        dataframe.loc[
            ( 
                (dataframe["fisher_smooth_avg"].shift() > dataframe["fisher_smooth_avg"])
            ),
            "fisher_falling"
        ] = 1

        dataframe.loc[
            (dataframe["fisher_smooth_avg"] > self.fisher_sell_threshold.value),
            "fisher_extreme_avg_sell"
        ] = 1

        dataframe.loc[
            (
                (dataframe["hurst_smooth_avg"] < self.hurst_mean_rev.value) &
                (dataframe['close'] > dataframe['baseline'])
            ),
            "mean_reversion_avg_sell"
        ] = 1

        # dataframe.loc[
        #     (
        #         (dataframe["hurst_smooth_avg"] > self.hurst_trending.value) &
        #         (dataframe['close'] < dataframe['baseline'])
        #     ),
        #     "hurst_trend_bear"
        # ] = 1

        dataframe.loc[
            ((dataframe["lr_mid_rsi"] > dataframe['rsi']) & 
             (dataframe["lr_mid_rsi"].shift() < dataframe['rsi'].shift())),
            "rsi_bear_cross"
        ] = 1

        dataframe.loc[
            (dataframe['lr_mid_rsi'] < dataframe['lr_mid_rsi'].shift()),
            "lr_falling"
        ] = 1

        dataframe.loc[
            ((dataframe["close"] > dataframe['goldie_max']) & 
             (dataframe["ha_close"] < dataframe['ha_open'])),
            "above_goldie_max_bear"
        ] = 1

        dataframe.loc[
            ((dataframe["close"] < dataframe['goldie_min']) & 
             (dataframe["ha_close"] < dataframe['ha_open'])),
            "below_goldie_min_bear"
        ] = 1

        # dataframe.loc[
        #     ((dataframe["ha_close"] < dataframe['ha_open'])),
        #     "ha_bear"
        # ] = 1

        # # New Average Conditions
        # dataframe.loc[
        #     (dataframe['ha_bear'].rolling(self.h2).sum() >= 3),  # 3 consecutive bearish HA candles
        #     "ha_trend_continuation_sell"
        # ] = 1

        # Harmonic and CP Conditions
        dataframe['pfe_bear_cross_h2'] = 0
        dataframe['pfe_bear_cross_h1'] = 0
        dataframe['pfe_bear_cross_h0'] = 0
        dataframe['pfe_bear_cross_cp'] = 0
        dataframe['pfe_falling_h2'] = 0
        dataframe['pfe_falling_h1'] = 0
        dataframe['pfe_falling_h0'] = 0
        dataframe['pfe_falling_cp'] = 0
        dataframe['fisher_bear_h2'] = 0
        dataframe['fisher_bear_h1'] = 0
        dataframe['fisher_bear_h0'] = 0
        dataframe['fisher_bear_cp'] = 0
        dataframe['fisher_bear_cross_h2'] = 0
        dataframe['fisher_bear_cross_h1'] = 0
        dataframe['fisher_bear_cross_h0'] = 0
        dataframe['fisher_bear_cross_cp'] = 0
        dataframe['fisher_falling_h2'] = 0
        dataframe['fisher_falling_h1'] = 0
        dataframe['fisher_falling_h0'] = 0
        dataframe['fisher_falling_cp'] = 0
        dataframe['fisher_extreme_h2_sell'] = 0
        dataframe['fisher_extreme_h1_sell'] = 0
        dataframe['fisher_extreme_h0_sell'] = 0
        dataframe['fisher_extreme_cp_sell'] = 0
        dataframe['hurst_trend_h2_bear'] = 0
        dataframe['hurst_trend_h1_bear'] = 0
        dataframe['hurst_trend_h0_bear'] = 0
        dataframe['hurst_trend_cp_bear'] = 0
        dataframe['hurst_sustained_trend_h2'] = 0
        dataframe['hurst_sustained_trend_h1'] = 0
        dataframe['hurst_sustained_trend_h0'] = 0
        dataframe['hurst_sustained_trend_cp'] = 0
        dataframe['mean_reversion_h2_bear'] = 0
        dataframe['mean_reversion_h1_bear'] = 0
        dataframe['mean_reversion_h0_bear'] = 0
        dataframe['mean_reversion_cp_bear'] = 0
        dataframe['rsi_overbought_h2'] = 0
        dataframe['rsi_overbought_h1'] = 0
        dataframe['rsi_overbought_h0'] = 0
        dataframe['rsi_overbought_cp'] = 0
        dataframe['rsi_falling_h2'] = 0
        dataframe['rsi_falling_h1'] = 0
        dataframe['rsi_falling_h0'] = 0
        dataframe['rsi_falling_cp'] = 0

        # Harmonic and CP Conditions
        # dataframe.loc[
        #     ((dataframe["pfe_smooth_h2"] < 0) & (dataframe["pfe_smooth_h2"].shift() > 0)),
        #     "pfe_bear_cross_h2"
        # ] = 1
        # dataframe.loc[
        #     ((dataframe["pfe_smooth_h1"] < 0) & (dataframe["pfe_smooth_h1"].shift() > 0)),
        #     "pfe_bear_cross_h1"
        # ] = 1
        # dataframe.loc[
        #     ((dataframe["pfe_smooth_h0"] < 0) & (dataframe["pfe_smooth_h0"].shift() > 0)),
        #     "pfe_bear_cross_h0"
        # ] = 1
        dataframe.loc[
            ((dataframe["pfe_smooth_cp"] < 0) & (dataframe["pfe_smooth_cp"].shift() > 0)),
            "pfe_bear_cross_cp"
        ] = 1

        # dataframe.loc[
        #     (
        #         (dataframe["pfe_smooth_h2"] < self.pfe_sell_threshold.value) & 
        #         (dataframe["pfe_smooth_h2"].shift() > dataframe["pfe_smooth_h2"])
        #     ),
        #     "pfe_falling_h2"
        # ] = 1
        # dataframe.loc[
        #     (
        #         (dataframe["pfe_smooth_h1"] < self.pfe_sell_threshold.value) & 
        #         (dataframe["pfe_smooth_h1"].shift() > dataframe["pfe_smooth_h1"])
        #     ),
        #     "pfe_falling_h1"
        # ] = 1
        # dataframe.loc[
        #     (
        #         (dataframe["pfe_smooth_h0"] < self.pfe_sell_threshold.value) & 
        #         (dataframe["pfe_smooth_h0"].shift() > dataframe["pfe_smooth_h0"])
        #     ),
        #     "pfe_falling_h0"
        # ] = 1
        dataframe.loc[
            (
                (dataframe["pfe_smooth_cp"] < self.pfe_sell_threshold.value) & 
                (dataframe["pfe_smooth_cp"].shift() > dataframe["pfe_smooth_cp"])
            ),
            "pfe_falling_cp"
        ] = 1

        # dataframe.loc[
        #     (
        #         (dataframe["fisher_smooth_h2"] < 0) &
        #         (dataframe["fisher_smooth_h2"] > self.fisher_buy_threshold.value)
        #     ),
        #     "fisher_bear_h2"
        # ] = 1
        # dataframe.loc[
        #     (
        #         (dataframe["fisher_smooth_h1"] < 0) &
        #         (dataframe["fisher_smooth_h1"] > self.fisher_buy_threshold.value)
        #     ),
        #     "fisher_bear_h1"
        # ] = 1
        # dataframe.loc[
        #     (
        #         (dataframe["fisher_smooth_h0"] < 0) &
        #         (dataframe["fisher_smooth_h0"] > self.fisher_buy_threshold.value)
        #     ),
        #     "fisher_bear_h0"
        # ] = 1
        dataframe.loc[
            (
                (dataframe["fisher_smooth_cp"] < 0) &
                (dataframe["fisher_smooth_cp"] > self.fisher_buy_threshold.value)
            ),
            "fisher_bear_cp"
        ] = 1

        # dataframe.loc[
        #     ((dataframe["fisher_smooth_h2"] > dataframe['fisher_h2']) & 
        #      (dataframe["fisher_smooth_h2"].shift() < dataframe['fisher_h2'].shift())),
        #     "fisher_bear_cross_h2"
        # ] = 1
        # dataframe.loc[
        #     ((dataframe["fisher_smooth_h1"] > dataframe['fisher_h1']) & 
        #      (dataframe["fisher_smooth_h1"].shift() < dataframe['fisher_h1'].shift())),
        #     "fisher_bear_cross_h1"
        # ] = 1
        # dataframe.loc[
        #     ((dataframe["fisher_smooth_h0"] > dataframe['fisher_h0']) & 
        #      (dataframe["fisher_smooth_h0"].shift() < dataframe['fisher_h0'].shift())),
        #     "fisher_bear_cross_h0"
        # ] = 1
        # dataframe.loc[
        #     ((dataframe["fisher_smooth_cp"] > dataframe['fisher_cp']) & 
        #      (dataframe["fisher_smooth_cp"].shift() < dataframe['fisher_cp'].shift())),
        #     "fisher_bear_cross_cp"
        # ] = 1

        # dataframe.loc[
        #     (
        #         (dataframe["fisher_smooth_h2"].shift() > dataframe["fisher_smooth_h2"])
        #     ),
        #     "fisher_falling_h2"
        # ] = 1
        # dataframe.loc[
        #     (
        #         (dataframe["fisher_smooth_h1"].shift() > dataframe["fisher_smooth_h1"])
        #     ),
        #     "fisher_falling_h1"
        # ] = 1
        # dataframe.loc[
        #     (
        #         (dataframe["fisher_smooth_h0"].shift() > dataframe["fisher_smooth_h0"])
        #     ),
        #     "fisher_falling_h0"
        # ] = 1
        dataframe.loc[
            (
                (dataframe["fisher_smooth_cp"].shift() > dataframe["fisher_smooth_cp"])
            ),
            "fisher_falling_cp"
        ] = 1

        # dataframe.loc[
        #     (dataframe["fisher_smooth_h2"] > self.fisher_sell_threshold.value),
        #     "fisher_extreme_h2_sell"
        # ] = 1
        # dataframe.loc[
        #     (dataframe["fisher_smooth_h1"] > self.fisher_sell_threshold.value),
        #     "fisher_extreme_h1_sell"
        # ] = 1
        # dataframe.loc[
        #     (dataframe["fisher_smooth_h0"] > self.fisher_sell_threshold.value),
        #     "fisher_extreme_h0_sell"
        # ] = 1
        dataframe.loc[
            (dataframe["fisher_smooth_cp"] > self.fisher_sell_threshold.value),
            "fisher_extreme_cp_sell"
        ] = 1

        dataframe.loc[
            (dataframe['lr_mid_rsi_h2'] > self.h2_ob.value),
            "rsi_overbought_h2"
        ] = 1
        # dataframe.loc[
        #     (dataframe['lr_mid_rsi_h1'] > self.h1_ob.value),
        #     "rsi_overbought_h1"
        # ] = 1
        # dataframe.loc[
        #     (dataframe['lr_mid_rsi_h0'] > self.h0_ob.value),
        #     "rsi_overbought_h0"
        # ] = 1
        # dataframe.loc[
        #     (dataframe['lr_mid_rsi'] > self.cp_ob.value),
        #     "rsi_overbought_cp"
        # ] = 1

        dataframe.loc[
            (dataframe['lr_mid_rsi_h2'] < dataframe['lr_mid_rsi_h2'].shift()),
            "rsi_falling_h2"
        ] = 1
        # dataframe.loc[
        #     (dataframe['lr_mid_rsi_h1'] < dataframe['lr_mid_rsi_h1'].shift()),
        #     "rsi_falling_h1"
        # ] = 1
        # dataframe.loc[
        #     (dataframe['lr_mid_rsi_h0'] < dataframe['lr_mid_rsi_h0'].shift()),
        #     "rsi_falling_h0"
        # ] = 1
        # dataframe.loc[
        #     (dataframe['lr_mid_rsi'] < dataframe['lr_mid_rsi'].shift()),
        #     "rsi_falling_cp"
        # ] = 1

        # Hurst Conditions with Inverted Baseline
        dataframe.loc[
            (
                (dataframe["hurst_smooth_h2"] < self.hurst_trending.value) &
                (dataframe['close'] < dataframe['baseline'])
            ),
            "hurst_trend_h2_bear"
        ] = 1
        dataframe.loc[
            (
                (dataframe["hurst_smooth_h1"] < self.hurst_trending.value) &
                (dataframe['close'] < dataframe['baseline'])
            ),
            "hurst_trend_h1_bear"
        ] = 1
        dataframe.loc[
            (
                (dataframe["hurst_smooth_h0"] < self.hurst_trending.value) &
                (dataframe['close'] < dataframe['baseline'])
            ),
            "hurst_trend_h0_bear"
        ] = 1
        dataframe.loc[
            (
                (dataframe["hurst_smooth_cp"] < self.hurst_trending.value) &
                (dataframe['close'] < dataframe['baseline'])
            ),
            "hurst_trend_cp_bear"
        ] = 1

        # dataframe.loc[
        #     (
        #         (dataframe["hurst_smooth_h2"] < self.hurst_trending.value) & 
        #         (dataframe["hurst_smooth_h2"].rolling(3).apply(lambda x: all(x < self.hurst_trending.value)) == 1) &
        #         (dataframe['close'] < dataframe['baseline'])
        #     ),
        #     "hurst_sustained_trend_h2"
        # ] = 1
        # dataframe.loc[
        #     (
        #         (dataframe["hurst_smooth_h1"] < self.hurst_trending.value) & 
        #         (dataframe["hurst_smooth_h1"].rolling(3).apply(lambda x: all(x < self.hurst_trending.value)) == 1) &
        #         (dataframe['close'] < dataframe['baseline'])
        #     ),
        #     "hurst_sustained_trend_h1"
        # ] = 1
        # dataframe.loc[
        #     (
        #         (dataframe["hurst_smooth_h0"] < self.hurst_trending.value) & 
        #         (dataframe["hurst_smooth_h0"].rolling(3).apply(lambda x: all(x < self.hurst_trending.value)) == 1) &
        #         (dataframe['close'] < dataframe['baseline'])
        #     ),
        #     "hurst_sustained_trend_h0"
        # ] = 1
        # dataframe.loc[
        #     (
        #         (dataframe["hurst_smooth_cp"] < self.hurst_trending.value) & 
        #         (dataframe["hurst_smooth_cp"].rolling(3).apply(lambda x: all(x < self.hurst_trending.value)) == 1) &
        #         (dataframe['close'] < dataframe['baseline'])
        #     ),
        #     "hurst_sustained_trend_cp"
        # ] = 1

        dataframe.loc[
            (
                (dataframe["hurst_smooth_h2"] < self.hurst_mean_rev.value) &
                (dataframe['close'] > dataframe['baseline'])
            ),
            "mean_reversion_h2_bear"
        ] = 1
        dataframe.loc[
            (
                (dataframe["hurst_smooth_h1"] < self.hurst_mean_rev.value) &
                (dataframe['close'] > dataframe['baseline'])
            ),
            "mean_reversion_h1_bear"
        ] = 1
        dataframe.loc[
            (
                (dataframe["hurst_smooth_h0"] < self.hurst_mean_rev.value) &
                (dataframe['close'] > dataframe['baseline'])
            ),
            "mean_reversion_h0_bear"
        ] = 1
        dataframe.loc[
            (
                (dataframe["hurst_smooth_cp"] < self.hurst_mean_rev.value) &
                (dataframe['close'] > dataframe['baseline'])
            ),
            "mean_reversion_cp_bear"
        ] = 1

        # Sum Long and Short Conditions
        long_conditions = [
            'pfe_bull_cross', 'pfe_rising', 'fisher_bull', 'fisher_cross_bull', 'fisher_rising',
            'fisher_extreme_avg_buy', 'mean_reversion_avg_buy', 'hurst_trend_bull', 'rsi_bull_cross',
            'lr_rising', 'below_goldie_min_bull', 'above_goldie_max_bull', 'ha_bull', 'ha_trend_continuation_bull',
            'pfe_bull_cross_h2', 'pfe_bull_cross_h1', 'pfe_bull_cross_h0', 'pfe_bull_cross_cp',
            'pfe_rising_h2', 'pfe_rising_h1', 'pfe_rising_h0', 'pfe_rising_cp',
            'fisher_bull_h2', 'fisher_bull_h1', 'fisher_bull_h0', 'fisher_bull_cp',
            'fisher_cross_h2', 'fisher_cross_h1', 'fisher_cross_h0', 'fisher_cross_cp',
            'fisher_rising_h2', 'fisher_rising_h1', 'fisher_rising_h0', 'fisher_rising_cp',
            'fisher_extreme_h2_buy', 'fisher_extreme_h1_buy', 'fisher_extreme_h0_buy', 'fisher_extreme_cp_buy',
            'hurst_trend_h2_bull', 'hurst_trend_h1_bull', 'hurst_trend_h0_bull', 'hurst_trend_cp_bull',
            'hurst_sustained_trend_h2', 'hurst_sustained_trend_h1', 'hurst_sustained_trend_h0', 'hurst_sustained_trend_cp',
            'mean_reversion_h2', 'mean_reversion_h1', 'mean_reversion_h0', 'mean_reversion_cp',
            'rsi_oversold_h2', 'rsi_oversold_h1', 'rsi_oversold_h0', 'rsi_oversold_cp',
            'rsi_rising_h2', 'rsi_rising_h1', 'rsi_rising_h0', 'rsi_rising_cp'
        ]

        short_conditions = [
            'pfe_bear_cross', 'pfe_falling', 'fisher_bear', 'fisher_bear_cross', 'fisher_falling',
            'fisher_extreme_avg_sell', 'mean_reversion_avg_sell', 'hurst_trend_bear', 'rsi_bear_cross',
            'lr_falling', 'below_goldie_min_bear', 'above_goldie_max_bear', 'ha_bear', 'ha_trend_continuation_sell',
            'pfe_bear_cross_h2', 'pfe_bear_cross_h1', 'pfe_bear_cross_h0', 'pfe_bear_cross_cp',
            'pfe_falling_h2', 'pfe_falling_h1', 'pfe_falling_h0', 'pfe_falling_cp',
            'fisher_bear_h2', 'fisher_bear_h1', 'fisher_bear_h0', 'fisher_bear_cp',
            'fisher_bear_cross_h2', 'fisher_bear_cross_h1', 'fisher_bear_cross_h0', 'fisher_bear_cross_cp',
            'fisher_falling_h2', 'fisher_falling_h1', 'fisher_falling_h0', 'fisher_falling_cp',
            'fisher_extreme_h2_sell', 'fisher_extreme_h1_sell', 'fisher_extreme_h0_sell', 'fisher_extreme_cp_sell',
            'hurst_trend_h2_bear', 'hurst_trend_h1_bear', 'hurst_trend_h0_bear', 'hurst_trend_cp_bear',
            'hurst_sustained_trend_h2', 'hurst_sustained_trend_h1', 'hurst_sustained_trend_h0', 'hurst_sustained_trend_cp',
            'mean_reversion_h2_bear', 'mean_reversion_h1_bear', 'mean_reversion_h0_bear', 'mean_reversion_cp_bear',
            'rsi_overbought_h2', 'rsi_overbought_h1', 'rsi_overbought_h0', 'rsi_overbought_cp',
            'rsi_falling_h2', 'rsi_falling_h1', 'rsi_falling_h0', 'rsi_falling_cp'
        ]

        # Sum Indicator-Specific Scores
        dataframe['pfe_score'] = dataframe[['pfe_bull_cross', 'pfe_rising', 'pfe_bull_cross_h2', 'pfe_bull_cross_h1', 'pfe_bull_cross_h0', 'pfe_bull_cross_cp', 'pfe_rising_h2', 'pfe_rising_h1', 'pfe_rising_h0', 'pfe_rising_cp']].sum(axis=1)
        dataframe['fisher_score'] = dataframe[['fisher_bull', 'fisher_cross_bull', 'fisher_rising', 'fisher_extreme_avg_buy', 'fisher_bull_h2', 'fisher_bull_h1', 'fisher_bull_h0', 'fisher_bull_cp', 'fisher_cross_h2', 'fisher_cross_h1', 'fisher_cross_h0', 'fisher_cross_cp', 'fisher_rising_h2', 'fisher_rising_h1', 'fisher_rising_h0', 'fisher_rising_cp', 'fisher_extreme_h2_buy', 'fisher_extreme_h1_buy', 'fisher_extreme_h0_buy', 'fisher_extreme_cp_buy']].sum(axis=1)
        dataframe['hurst_score'] = dataframe[['mean_reversion_avg_buy', 'hurst_trend_bull', 'hurst_trend_h2_bull', 'hurst_trend_h1_bull', 'hurst_trend_h0_bull', 'hurst_trend_cp_bull', 'hurst_sustained_trend_h2', 'hurst_sustained_trend_h1', 'hurst_sustained_trend_h0', 'hurst_sustained_trend_cp', 'mean_reversion_h2', 'mean_reversion_h1', 'mean_reversion_h0', 'mean_reversion_cp']].sum(axis=1)
        dataframe['rsi_score'] = dataframe[['rsi_bull_cross', 'lr_rising', 'rsi_oversold_h2', 'rsi_oversold_h1', 'rsi_oversold_h0', 'rsi_oversold_cp', 'rsi_rising_h2', 'rsi_rising_h1', 'rsi_rising_h0', 'rsi_rising_cp']].sum(axis=1)
        dataframe['goldie_score'] = dataframe[['below_goldie_min_bull', 'above_goldie_max_bull']].sum(axis=1)
        dataframe['ha_score'] = dataframe[['ha_bull', 'ha_trend_continuation_bull']].sum(axis=1)

        # Total Long Score
        dataframe['long_score'] = dataframe['pfe_score'] + dataframe['fisher_score'] + dataframe['hurst_score'] + dataframe['rsi_score'] + dataframe['goldie_score'] + dataframe['ha_score']

        # Sum Short Conditions
        dataframe['pfe_score_short'] = dataframe[['pfe_bear_cross', 'pfe_falling', 'pfe_bear_cross_h2', 'pfe_bear_cross_h1', 'pfe_bear_cross_h0', 'pfe_bear_cross_cp', 'pfe_falling_h2', 'pfe_falling_h1', 'pfe_falling_h0', 'pfe_falling_cp']].sum(axis=1)
        dataframe['fisher_score_short'] = dataframe[['fisher_bear', 'fisher_bear_cross', 'fisher_falling', 'fisher_extreme_avg_sell', 'fisher_bear_h2', 'fisher_bear_h1', 'fisher_bear_h0', 'fisher_bear_cp', 'fisher_bear_cross_h2', 'fisher_bear_cross_h1', 'fisher_bear_cross_h0', 'fisher_bear_cross_cp', 'fisher_falling_h2', 'fisher_falling_h1', 'fisher_falling_h0', 'fisher_falling_cp', 'fisher_extreme_h2_sell', 'fisher_extreme_h1_sell', 'fisher_extreme_h0_sell', 'fisher_extreme_cp_sell']].sum(axis=1)
        dataframe['hurst_score_short'] = dataframe[['mean_reversion_avg_sell', 'hurst_trend_bear', 'hurst_trend_h2_bear', 'hurst_trend_h1_bear', 'hurst_trend_h0_bear', 'hurst_trend_cp_bear', 'hurst_sustained_trend_h2', 'hurst_sustained_trend_h1', 'hurst_sustained_trend_h0', 'hurst_sustained_trend_cp', 'mean_reversion_h2_bear', 'mean_reversion_h1_bear', 'mean_reversion_h0_bear', 'mean_reversion_cp_bear']].sum(axis=1)
        dataframe['rsi_score_short'] = dataframe[['rsi_bear_cross', 'lr_falling', 'rsi_overbought_h2', 'rsi_overbought_h1', 'rsi_overbought_h0', 'rsi_overbought_cp', 'rsi_falling_h2', 'rsi_falling_h1', 'rsi_falling_h0', 'rsi_falling_cp']].sum(axis=1)
        dataframe['goldie_score_short'] = dataframe[['below_goldie_min_bear', 'above_goldie_max_bear']].sum(axis=1)
        dataframe['ha_score_short'] = dataframe[['ha_bear', 'ha_trend_continuation_sell']].sum(axis=1)

        # Total Short Score
        dataframe['short_score'] = dataframe['pfe_score_short'] + dataframe['fisher_score_short'] + dataframe['hurst_score_short'] + dataframe['rsi_score_short'] + dataframe['goldie_score_short'] + dataframe['ha_score_short']
        # Sum Combined Indicator Market Scores
        dataframe['pfe_market_score'] = dataframe[['pfe_bull_cross', 'pfe_rising', 'pfe_bull_cross_h2', 'pfe_bull_cross_h1', 'pfe_bull_cross_h0', 'pfe_bull_cross_cp', 'pfe_rising_h2', 'pfe_rising_h1', 'pfe_rising_h0', 'pfe_rising_cp']].sum(axis=1) - dataframe[['pfe_bear_cross', 'pfe_falling', 'pfe_bear_cross_h2', 'pfe_bear_cross_h1', 'pfe_bear_cross_h0', 'pfe_bear_cross_cp', 'pfe_falling_h2', 'pfe_falling_h1', 'pfe_falling_h0', 'pfe_falling_cp']].sum(axis=1)
        dataframe['fisher_market_score'] = dataframe[['fisher_bull', 'fisher_cross_bull', 'fisher_rising', 'fisher_extreme_avg_buy', 'fisher_bull_h2', 'fisher_bull_h1', 'fisher_bull_h0', 'fisher_bull_cp', 'fisher_cross_h2', 'fisher_cross_h1', 'fisher_cross_h0', 'fisher_cross_cp', 'fisher_rising_h2', 'fisher_rising_h1', 'fisher_rising_h0', 'fisher_rising_cp', 'fisher_extreme_h2_buy', 'fisher_extreme_h1_buy', 'fisher_extreme_h0_buy', 'fisher_extreme_cp_buy']].sum(axis=1) - dataframe[['fisher_bear', 'fisher_bear_cross', 'fisher_falling', 'fisher_extreme_avg_sell', 'fisher_bear_h2', 'fisher_bear_h1', 'fisher_bear_h0', 'fisher_bear_cp', 'fisher_bear_cross_h2', 'fisher_bear_cross_h1', 'fisher_bear_cross_h0', 'fisher_bear_cross_cp', 'fisher_falling_h2', 'fisher_falling_h1', 'fisher_falling_h0', 'fisher_falling_cp', 'fisher_extreme_h2_sell', 'fisher_extreme_h1_sell', 'fisher_extreme_h0_sell', 'fisher_extreme_cp_sell']].sum(axis=1)
        dataframe['hurst_market_score'] = dataframe[['mean_reversion_avg_buy', 'hurst_trend_bull', 'hurst_trend_h2_bull', 'hurst_trend_h1_bull', 'hurst_trend_h0_bull', 'hurst_trend_cp_bull', 'hurst_sustained_trend_h2', 'hurst_sustained_trend_h1', 'hurst_sustained_trend_h0', 'hurst_sustained_trend_cp', 'mean_reversion_h2', 'mean_reversion_h1', 'mean_reversion_h0', 'mean_reversion_cp']].sum(axis=1) - dataframe[['mean_reversion_avg_sell', 'hurst_trend_bear', 'hurst_trend_h2_bear', 'hurst_trend_h1_bear', 'hurst_trend_h0_bear', 'hurst_trend_cp_bear', 'hurst_sustained_trend_h2', 'hurst_sustained_trend_h1', 'hurst_sustained_trend_h0', 'hurst_sustained_trend_cp', 'mean_reversion_h2_bear', 'mean_reversion_h1_bear', 'mean_reversion_h0_bear', 'mean_reversion_cp_bear']].sum(axis=1)
        dataframe['rsi_market_score'] = dataframe[['rsi_bull_cross', 'lr_rising', 'rsi_oversold_h2', 'rsi_oversold_h1', 'rsi_oversold_h0', 'rsi_oversold_cp', 'rsi_rising_h2', 'rsi_rising_h1', 'rsi_rising_h0', 'rsi_rising_cp']].sum(axis=1) - dataframe[['rsi_bear_cross', 'lr_falling', 'rsi_overbought_h2', 'rsi_overbought_h1', 'rsi_overbought_h0', 'rsi_overbought_cp', 'rsi_falling_h2', 'rsi_falling_h1', 'rsi_falling_h0', 'rsi_falling_cp']].sum(axis=1)
        dataframe['goldie_market_score'] = dataframe[['below_goldie_min_bull', 'above_goldie_max_bull']].sum(axis=1) - dataframe[['below_goldie_min_bear', 'above_goldie_max_bear']].sum(axis=1)
        dataframe['ha_market_score'] = dataframe[['ha_bull', 'ha_trend_continuation_bull']].sum(axis=1) - dataframe[['ha_bear', 'ha_trend_continuation_sell']].sum(axis=1)

        dataframe['long_score'] = dataframe[long_conditions].sum(axis=1)
        dataframe['short_score'] = dataframe[short_conditions].sum(axis=1)

        dataframe['market_score'] = dataframe['long_score'] - dataframe['short_score']
        dataframe['market_signal'] = dataframe['market_score'].rolling(3).mean()
        dataframe['market_cross'] = dataframe['market_signal'].rolling(3).mean()
        dataframe['market_diff'] = dataframe['market_score'].diff()
        analytic_signal = hilbert(dataframe['market_score'])
        dataframe['market_score_shifted'] = np.imag(analytic_signal)  # 90-degree shift
        dataframe['market_scoot'] = (dataframe['market_score_shifted'] + dataframe['market_score']) / 2
        h = self.market_length.value
        x_0 = self.cp
        nadaraya_watson_mc(dataframe, h, r, x_0, smoothColors, lag, mult = 2.0)
        dataframe['nw_market'] = dataframe[f'yhat1_{h}']
        dataframe['nw_diff'] = dataframe['market_score'] - dataframe['nw_market']

        market_channel = self.linear_regression_channel(dataframe['market_score'], window=self.cp, num_dev=1.0)
        market_channel0 = self.linear_regression_channel(dataframe['market_score'], window=self.h0, num_dev=1.0)
        market_channel1 = self.linear_regression_channel(dataframe['market_score'], window=self.h1, num_dev=1.0)
        market_channel2 = self.linear_regression_channel(dataframe['market_score'], window=self.h2, num_dev=1.0)
        dataframe['lr_mid_market'] = market_channel['mid'].rolling(3).mean() 
        dataframe['lr_mid_market0'] = market_channel0['mid'].rolling(3).mean()
        dataframe['lr_mid_market1'] = market_channel1['mid'].rolling(3).mean()
        dataframe['lr_mid_market2'] = market_channel2['mid'].rolling(3).mean()
        dataframe['lr_market_avg'] = (dataframe['lr_mid_market'] + dataframe['lr_mid_market0'] + dataframe['lr_mid_market1'] + dataframe['lr_mid_market2']) / 4 
        dataframe['lr_market_signal'] = dataframe['lr_mid_market'].rolling(3).mean()
        dataframe['lr_up_market'] = market_channel['upper']
        dataframe['lr_dn_market'] = market_channel['lower']

        dataframe['trend'] = np.where(dataframe['market_signal'] > dataframe['market_cross'], 1, -1)
        dataframe['zero'] = 0

        # timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
        # pair = metadata['pair'].replace('/', '_')  # Replace '/' with '_' for valid filename
        # filename = f"{pair}_{timestamp}.csv"
        # dataframe.to_csv(filename, index=True)
        # logger.info(f"Exported DataFrame for {pair} to {filename}")

        return dataframe

    # Baseline MA Implementations
    def ama(self, df, period):
        return ema(df["close"], length=period)

    def adxvma(self, df, period):
        adx = talib.ADX(df["high"], df["low"], df["close"], timeperiod=period)
        return ema(df["close"], length=period) * (adx / 100)

    def ahrens(self, df, period):
        return (df["open"] + df["close"]).rolling(window=period).mean() / 2

    def donchian(self, df, period):
        high = df["high"].rolling(window=period).max()
        low = df["low"].rolling(window=period).min()
        return (high + low) / 2

    def dsema(self, df, period):
        ema1 = ema(df["close"], length=period)
        return ema(ema1, length=period)

    def dsfema(self, df, period):
        ema1 = ema(df["close"], length=int(period/2))
        return ema(ema1, length=int(period/2))

    def dsrwema(self, df, period):
        range_weight = (df["high"] - df["low"]).rolling(window=period).mean()
        weighted_price = df["close"] * range_weight
        ema1 = ema(weighted_price, length=period)
        return ema(ema1, length=period)

    def dswema(self, df, period):
        wilder_period = period * 2 - 1
        ema1 = ema(df["close"], length=wilder_period)
        return ema(ema1, length=wilder_period)

    def dwma(self, df, period):
        wma1 = wma(df["close"], length=period)
        return wma(wma1, length=period)

    def frama(self, df, period):
        return ema(df["close"], length=period)

    def gdema(self, df, period, vol_factor=0.5):
        if isinstance(df, pd.Series):
            close = df
        else:
            close = df["close"]
        ema1 = ema(close, length=period)
        ema2 = ema(ema1, length=period)
        return (1 + vol_factor) * ema1 - vol_factor * ema2

    def gddema(self, df, period):
        gdema1 = self.gdema(df, period)
        return self.gdema(pd.DataFrame({"close": gdema1}), period)

    def hma_ema(self, df, period):
        wma1 = wma(df["close"], length=int(period/2)) * 2
        wma2 = wma(df["close"], length=period)
        raw_hma = wma1 - wma2
        return ema(raw_hma, length=int(np.sqrt(period)))

    def hma_wma(self, df, period):
        wma1 = wma(df["close"], length=int(period/2)) * 2
        wma2 = wma(df["close"], length=period)
        raw_hma = wma1 - wma2
        return wma(raw_hma, length=int(np.sqrt(period)))

    def hma_smma(self, df, period):
        wma1 = wma(df["close"], length=int(period/2)) * 2
        wma2 = wma(df["close"], length=period)
        raw_hma = wma1 - wma2
        return self.smma(pd.DataFrame({"close": raw_hma}), int(np.sqrt(period)))

    def t3(self, df, period, v_factor=0.7):
        ema1 = ema(df["close"], length=period)
        ema2 = ema(ema1, length=period)
        ema3 = ema(ema2, length=period)
        c1 = -v_factor ** 3
        c2 = 3 * v_factor ** 2 * (1 + v_factor)
        c3 = -6 * v_factor ** 2 - 3 * v_factor * (1 + v_factor)
        c4 = 1 + 3 * v_factor + v_factor ** 3 + 3 * v_factor ** 2
        return c1 * ema3 + c2 * ema2 + c3 * ema1 + c4 * df["close"]

    def ilrs(self, df, period):
        lsma = linreg(df["close"], length=period)
        return lsma.cumsum() / period

    def kalman(self, df, period):
        return ema(df["close"], length=period)

    def kama(self, df, period, fast=2, slow=30):
        close_diff = df["close"].diff().abs()
        signal = close_diff.rolling(window=period).sum()
        noise = (df["high"] - df["low"]).rolling(window=period).sum()
        er = signal / noise
        sc = ((er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2)
        kama = pd.Series(index=df.index)
        kama.iloc[period] = df["close"].iloc[period]
        for i in range(period + 1, len(df)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (df["close"].iloc[i] - kama.iloc[i-1])
        return kama

    def laguerre(self, df, period, alpha=0.2):
        l0 = pd.Series(0, index=df.index)
        l1 = pd.Series(0, index=df.index)
        l2 = pd.Series(0, index=df.index)
        l3 = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            l0.iloc[i] = (1 - alpha) * df["close"].iloc[i] + alpha * l0.iloc[i-1]
            l1.iloc[i] = -alpha * l0.iloc[i] + l0.iloc[i-1] + alpha * l1.iloc[i-1]
            l2.iloc[i] = -alpha * l1.iloc[i] + l1.iloc[i-1] + alpha * l2.iloc[i-1]
            l3.iloc[i] = -alpha * l2.iloc[i] + l2.iloc[i-1] + alpha * l3.iloc[i-1]
        return (l0 + 2 * l1 + 2 * l2 + l3) / 6

    def leader_ema(self, df, period):
        return ema(df["close"], length=period)

    def mcginley(self, df, period):
        mg = pd.Series(df["close"].iloc[0], index=df.index)
        for i in range(1, len(df)):
            mg.iloc[i] = mg.iloc[i-1] + (df["close"].iloc[i] - mg.iloc[i-1]) / (period * (df["close"].iloc[i] / mg.iloc[i-1]) ** 4)
        return mg

    def nonlag(self, df, period):
        return ema(df["close"], length=int(period/2))

    def oma(self, df, period):
        ema1 = ema(df["close"], length=period)
        ema2 = ema(ema1, length=period)
        return ema(ema2, length=period)

    def parabolic_wma(self, df, period, power=2):
        weights = np.array([((i+1)/period)**power for i in range(period)])
        weights = weights / weights.sum()
        return df["close"].rolling(window=period).apply(lambda x: np.sum(x * weights), raw=True)

    def pdfma(self, df, period):
        weights = np.exp(-np.linspace(-2, 2, period)**2)
        weights = weights / weights.sum()
        return df["close"].rolling(window=period).apply(lambda x: np.sum(x * weights), raw=True)

    def qrma(self, df, period):
        x = np.arange(period)
        def quad_reg(y):
            coeffs = np.polyfit(x, y, 2)
            return np.polyval(coeffs, period-1)
        return df["close"].rolling(window=period).apply(quad_reg, raw=True)

    def rema(self, df, period):
        alpha = 2 / (period + 1)
        return ema(df["close"], length=period) * (1 + alpha)

    def rwema(self, df, period):
        range_weight = (df["high"] - df["low"]).rolling(window=period).mean()
        return ema(df["close"] * range_weight, length=period) / ema(range_weight, length=period)

    def recursive(self, df, period):
        return linreg(df["close"], length=period)

    def sdec(self, df, period):
        return df["close"] - ema(df["close"], length=period)

    def slwma(self, df, period):
        lwma1 = wma(df["close"], length=period)
        return ema(lwma1, length=period)

    def smma(self, df, period):
        if isinstance(df, pd.Series):
            close = df
        else:
            close = df["close"]
        smma = pd.Series(index=close.index)
        smma.iloc[period-1] = close.iloc[:period].mean()
        for i in range(period, len(close)):
            smma.iloc[i] = (smma.iloc[i-1] * (period - 1) + close.iloc[i]) / period
        return smma

    def smoother(self, df, period):
        return ema(df["close"], length=int(period/2))

    def super_smoother(self, df, period):
        a = np.exp(-1.414 * np.pi / period)
        b = 2 * a * np.cos(1.414 * np.pi / period)
        c = a * a
        d = 1 - b - c
        ss = pd.Series(0, index=df.index)
        for i in range(2, len(df)):
            ss.iloc[i] = d * df["close"].iloc[i] + b * ss.iloc[i-1] + c * ss.iloc[i-2]
        return ss

    def tma(self, df, period):
        sma1 = sma(df["close"], length=period)
        return sma(sma1, length=period)

    def vidya(self, df, period):
        cmo = talib.CMO(df["close"], timeperiod=period)
        alpha = 2 / (period + 1) * (cmo / 100)
        vidya = pd.Series(df["close"].iloc[0], index=df.index)
        for i in range(1, len(df)):
            vidya.iloc[i] = alpha.iloc[i] * df["close"].iloc[i] + (1 - alpha.iloc[i]) * vidya.iloc[i-1]
        return vidya

    def vma(self, df, period):
        return ema(df["close"], length=period)

    def vema(self, df, period):
        vol_price = df["close"] * df["volume"]
        return ema(vol_price, length=period) / ema(df["volume"], length=period)

    def zero_lag_dema(self, df, period):
        ema1 = ema(df["close"], length=period)
        lag = ema1.shift(period)
        return 2 * ema1 - lag

    def zero_lag_ma(self, df, period):
        ema1 = ema(df["close"], length=period)
        lag = ema1.shift(period)
        return 2 * ema1 - lag

    def zero_lag_tema(self, df, period):
        tema1 = tema(df["close"], length=period)
        lag = tema1.shift(period)
        return 2 * tema1 - lag

    def sine_wma(self, df, period):
        weights = np.array([np.sin(np.pi * (i+1) / (period+1)) for i in range(period)])
        weights = weights / weights.sum()
        return df["close"].rolling(window=period).apply(lambda x: np.sum(x * weights), raw=True)

    # PFE Implementation
    def calculate_pfe(self, dataframe: DataFrame, period: int) -> pd.Series:
        # Polarized Fractal Efficiency calculation
        close = dataframe["close"]
        pfe = pd.Series(0.0, index=dataframe.index)

        for i in range(period, len(dataframe)):
            # Straight-line distance
            price_diff = close.iloc[i] - close.iloc[i - period]
            straight_dist = np.sqrt(price_diff**2 + period**2)

            # Path length (sum of segment lengths)
            path_length = 0
            for j in range(i - period + 1, i + 1):
                segment_diff = close.iloc[j] - close.iloc[j - 1]
                segment_length = np.sqrt(segment_diff**2 + 1)
                path_length += segment_length

            # PFE calculation
            if path_length != 0:
                pfe.iloc[i] = 100 * straight_dist / path_length
                # Polarize: positive for upward movement, negative for downward
                if price_diff < 0:
                    pfe.iloc[i] = -pfe.iloc[i]

        return pfe
    # def calculate_pfe(self, dataframe: DataFrame, period: int) -> pd.Series:
    #     close = dataframe["close"]
        
    #     # Auto-scaling: determina il fattore di moltiplicazione necessario
    #     # per portare il prezzo medio sopra 1000
    #     avg_price = close.mean()
        
    #     if avg_price >= 1000:
    #         scale_factor = 100
    #     elif avg_price >= 100:
    #         scale_factor = 1000
    #     elif avg_price >= 10:
    #         scale_factor = 10000
    #     elif avg_price >= 1:
    #         scale_factor = 100000
    #     elif avg_price >= 0.1:
    #         scale_factor = 1000000
    #     elif avg_price >= 0.01:
    #         scale_factor = 10000000
    #     elif avg_price >= 0.001:
    #         scale_factor = 100000000
    #     else:
    #         scale_factor = 1000000000
        
    #     # Scala i prezzi per il calcolo
    #     scaled_close = close * scale_factor
        
    #     pfe = pd.Series(0.0, index=dataframe.index)
        
    #     for i in range(period, len(dataframe)):
    #         price_diff = scaled_close.iloc[i] - scaled_close.iloc[i - period]
    #         straight_dist = np.sqrt(price_diff**2 + period**2)
            
    #         path_length = 0
    #         for j in range(i - period + 1, i + 1):
    #             segment_diff = scaled_close.iloc[j] - scaled_close.iloc[j - 1]
    #             segment_length = np.sqrt(segment_diff**2 + 1)
    #             path_length += segment_length
            
    #         if path_length != 0:
    #             pfe.iloc[i] = 100 * straight_dist / path_length
    #             if price_diff < 0:
    #                 pfe.iloc[i] = -pfe.iloc[i]
        
    #     return pfe

    def linear_regression_channel(self, data: pd.Series, window: int, num_dev: float):
        """Calculate the linear regression line and standard deviation channel (upper and lower bands)."""
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

    # Fisher Transform Implementation
    def calculate_fisher(self, dataframe: DataFrame, period: int) -> pd.Series:
        median_price = (dataframe["high"] + dataframe["low"]) / 2
        fisher = pd.Series(0.0, index=dataframe.index)
        for i in range(period, len(dataframe)):
            price_window = median_price.iloc[i-period:i]
            price_min = price_window.min()
            price_max = price_window.max()
            if price_max != price_min:
                norm = (median_price.iloc[i] - price_min) / (price_max - price_min)
                norm = 2 * norm - 1
                norm = max(min(norm, 0.999), -0.999)
                fisher.iloc[i] = 0.5 * np.log((1 + norm) / (1 - norm))
            else:
                fisher.iloc[i] = 0.0
        return fisher

    # Hurst Exponent Implementation
    def calculate_hurst(self, series: pd.Series, period: int) -> pd.Series:
        hurst = pd.Series(np.nan, index=series.index)
        for i in range(period, len(series)):
            window = series.iloc[i-period:i].dropna()
            if len(window) < period:
                logger.debug(f"Insufficient data at i={i}, len={len(window)}")
                hurst.iloc[i] = 0.5
                continue
            mean = window.mean()
            mean_adj = window - mean
            cum_dev = mean_adj.cumsum()
            r = cum_dev.max() - cum_dev.min()
            s = window.std()
            if s == 0 or r == 0 or np.isnan(s) or np.isnan(r):
                logger.debug(f"Invalid R/S at i={i}: r={r}, s={s}")
                variance = window.var()
                if variance > 0:
                    hurst.iloc[i] = 0.5 + np.log(variance) / (2 * np.log(period))
                    hurst.iloc[i] = np.clip(hurst.iloc[i], 0, 1)
                else:
                    hurst.iloc[i] = 0.5
                continue
            rs = r / s
            if rs <= 0:
                logger.debug(f"Invalid rs at i={i}: rs={rs}")
                hurst.iloc[i] = 0.5
                continue
            h = np.log(rs) / np.log(period)
            hurst.iloc[i] = np.clip(h, 0, 1)
            logger.debug(f"H at i={i}: rs={rs}, h={h}, clipped={hurst.iloc[i]}")
        return hurst

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Standard Entry Logic (GKD-C Confirmation)
        # Adjusted to use dv_val instead of t, aligning with Pine Script's negation
        # dataframe.loc[
        #     ((dataframe["market_score"] == -5) & (dataframe["market_score"].shift() > -2)),
        #     "enter_long"
        # ] = 1

        dataframe.loc[
            (
                # (dataframe["market_score"] >= 0) & (dataframe["market_score"].shift() <= -1) # &
                (dataframe["lr_mid_market"] > dataframe["lr_mid_market"].shift()) &
                (dataframe["lr_mid_market"].shift() < dataframe["lr_mid_market"].shift(2)) &
                (dataframe['close'] < dataframe['entry_limit'])
            ),
            "enter_long"
        ] = 1

        # dataframe.loc[
        #     (
        #         (dataframe["pfe_smooth"] > 0) & 
        #         (dataframe["pfe_smooth"].shift() < 0) &
        #         (dataframe['pfe'] - dataframe['pfe_smooth'] > 5) &
        #         (dataframe['pfe_smooth'] < 95) &
        #     (dataframe["fisher_smooth"] > 0) |
        #     (dataframe["hurst_smooth"] < self.hurst_mean_rev.value) |
        #     (dataframe["volatmeter"] < -dataframe["dv_val"])),
        #     "enter_long"
        # ] = 1


        if self.can_short == True:
            dataframe.loc[
                (dataframe["pfe_smooth"] < 0) |
                (dataframe["fisher_smooth"] < 0) |
                (dataframe["hurst_smooth"] < self.hurst_mean_rev.value) |
                (dataframe["volatmeter"] < -dataframe["dv_val"]),
                "enter_short"
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit Logic
        # Adjusted to use dv_val for exits, aligning with Pine Script
        dataframe.loc[
            (
                (dataframe["market_score"] <= 10) & (dataframe["market_score"].shift() >= 11) &
                (dataframe["lr_mid_market"] < dataframe["lr_mid_market"].shift()) &
                (dataframe["lr_mid_market"].shift() > dataframe["lr_mid_market"].shift(2))
                # (dataframe["fisher_smooth_avg"] > self.fisher_sell_threshold.value)
            ), 
            "exit_long"
        ] = 1

        # dataframe.loc[
        #     ((dataframe["pfe_smooth"] < 0) & (dataframe["pfe_smooth"].shift() > 0)) |
        #     (dataframe["fisher_smooth"] < 0) |
        #     (dataframe["hurst_smooth"] < self.hurst_mean_rev.value) |
        #     (dataframe["volatmeter"] < -dataframe["dv_val"]),  # Equivalent to vol < -(threshold_level - anti_thres - vol)
        #     "exit_long"
        # ] = 1

        if self.can_short == True:
            dataframe.loc[
                (dataframe["pfe_smooth"] > 0) |
                (dataframe["fisher_smooth"] > 0) |
                (dataframe["hurst_smooth"] < self.hurst_mean_rev.value) |
                (dataframe["volatmeter"] < -dataframe["dv_val"]),  # Equivalent to vol < -(threshold_level - anti_thres - vol)
                "exit_short"
            ] = 1

        return dataframe

    # def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    #     dataframe.loc[
    #         (
    #             (dataframe["lr_mid_market"] > dataframe["lr_market_signal"].shift()) &
    #             (dataframe["close"] > dataframe["close"].shift(2)) &
    #             (
    #                 (dataframe["pfe_market_score"] > 0) |
    #                 (dataframe["fisher_market_score"] > 0) |
    #                 (dataframe["hurst_market_score"] > 0)
    #             ) &
    #             (dataframe["ha_close"] > dataframe["ha_open"])
    #         ),
    #         "enter_long"
    #     ] = 1
    #     return dataframe

    # def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    #     dataframe.loc[
    #         (
    #             (dataframe["lr_mid_market"] < dataframe["lr_market_signal"].shift()) &
    #             (
    #                 (dataframe["pfe_market_score"] < 0) |
    #                 (dataframe["fisher_market_score"] < 0) |
    #                 (dataframe["hurst_market_score"] < 0)
    #             ) &
    #             (dataframe["ha_close"] < dataframe["ha_open"])
    #         ) |
    #         (
    #             (dataframe["close"] > dataframe["close"].shift() + 2 * dataframe["atr"]) |
    #             (dataframe["pfe_market_score"] < dataframe["pfe_market_score"].shift() - 1)
    #         ),
    #         "exit_long"
    #     ] = 1
    #     return dataframe

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

def kernel_regression(src, size, h, r, x_0):
    _currentWeight = 0.0
    _cumulativeWeight = 0.0000000001
    for i in range(len(src)):
        y = src.iloc[i]
        w = np.power(1 + (np.power(i, 2) / ((np.power(h, 2) * 2 * r))), -r)
        _currentWeight += y * w
        _cumulativeWeight += w
    if _cumulativeWeight == 0:
        return 0
    return _currentWeight / _cumulativeWeight

def kernel_regression_2(src, size, h, r, x_0):
    _currentWeight = 0.0
    _cumulativeWeight = 0.0000000001
    for i in range(len(src)):
        y = src[i]  # Use array indexing
        w = np.power(1 + (np.power(i, 2) / ((np.power(h, 2) * 2 * r))), -r)
        _currentWeight += y * w
        _cumulativeWeight += w
    if _cumulativeWeight == 0:
        return 0
    return _currentWeight / _cumulativeWeight

def nadaraya_watson(df, src_col, h, r, x_0, smoothColors, lag, mult=2):
    src = df[src_col].values  # NumPy array
    size = len(src)
    yhat1 = []
    yhat2 = []
    nwe_up = []
    nwe_down = []
    nwe_entry = []
    nwe_exit = []
    wasBearish = []
    wasBullish = []
    isBearish = []
    isBullish = []
    isBearishChange = []
    isBullishChange = []
    isBullishCross = []
    isBearishCross = []
    isBullishSmooth = []
    isBearishSmooth = []
    colorByCross = []
    colorByRate = []
    plotColor = []
    alertBullish = []
    alertBearish = []

    for i in range(size):
        if i >= h:
            window = src[i - h:i]
            yhat1_value = kernel_regression_2(window, h, h, r, x_0)
            yhat1.append(yhat1_value)

            # Compute envelopes
            mae = np.mean(np.abs(src[i - h:i] - yhat1_value)) * mult
            nwe_up.append(yhat1_value + mae)
            nwe_down.append(yhat1_value - mae)

            # Set entry and exit signals
            nwe_entry.append(1 if src[i] < nwe_down[-1] else 0)
            nwe_exit.append(1 if src[i] > nwe_up[-1] else 0)

            # Trend and crossover conditions
            if i > 1:
                wasBearish.append(yhat1[i - 2] > yhat1[i - 1])
                wasBullish.append(yhat1[i - 2] < yhat1[i - 1])
            else:
                wasBearish.append(False)
                wasBullish.append(False)

            if i > 0:
                isBearish.append(yhat1[i - 1] > yhat1[i])
                isBullish.append(yhat1[i - 1] < yhat1[i])
            else:
                isBearish.append(False)
                isBullish.append(False)

            isBearishChange.append(isBearish[-1] and wasBullish[-1] if wasBullish else False)
            isBullishChange.append(isBullish[-1] and wasBearish[-1] if wasBearish else False)

            if i >= h + lag:
                window = src[i - h - lag:i - lag]
                yhat2.append(kernel_regression_2(window, h, h, r, x_0))
                
                if i > 0:
                    isBullishCross.append(yhat2[-1] > yhat1[i])
                    isBearishCross.append(yhat2[-1] < yhat1[i])
                    isBullishSmooth.append(yhat2[-1] > yhat1[i])
                    isBearishSmooth.append(yhat2[-1] < yhat1[i])
                else:
                    isBullishCross.append(False)
                    isBearishCross.append(False)
                    isBullishSmooth.append(False)
                    isBearishSmooth.append(False)

                colorByCross.append(1 if isBullishSmooth[-1] else -1 if isBearishSmooth[-1] else 0)
                colorByRate.append(1 if isBullish[-1] else -1 if isBearish[-1] else 0)
                plotColor.append(colorByCross[-1] if smoothColors else colorByRate[-1])
                alertBullish.append(1 if isBullishCross[-1] else 0)
                alertBearish.append(-1 if isBearishCross[-1] else 0)
            else:
                yhat2.append(0)
                isBullishCross.append(False)
                isBearishCross.append(False)
                isBullishSmooth.append(False)
                isBearishSmooth.append(False)
                colorByCross.append(0)
                colorByRate.append(0)
                plotColor.append(0)
                alertBullish.append(0)
                alertBearish.append(0)
        else:
            yhat1.append(0)
            yhat2.append(0)
            nwe_up.append(np.nan)
            nwe_down.append(np.nan)
            nwe_entry.append(0)
            nwe_exit.append(0)
            wasBearish.append(False)
            wasBullish.append(False)
            isBearish.append(False)
            isBullish.append(False)
            isBearishChange.append(False)
            isBullishChange.append(False)
            isBullishCross.append(False)
            isBearishCross.append(False)
            isBullishSmooth.append(False)
            isBearishSmooth.append(False)
            colorByCross.append(0)
            colorByRate.append(0)
            plotColor.append(0)
            alertBullish.append(0)
            alertBearish.append(0)

    # Assign computed values to DataFrame columns
    df[f'yhat1_{h}'] = yhat1
    df[f'yhat2_{h}'] = yhat2
    df[f'nw_up_{h}'] = nwe_up
    df[f'nw_down_{h}'] = nwe_down
    df[f'nw_entry_{h}'] = nwe_entry
    df[f'nw_exit_{h}'] = nwe_exit
    df[f'wasBearish_{h}'] = wasBearish
    df[f'wasBullish_{h}'] = wasBullish
    df[f'isBearish_{h}'] = isBearish
    df[f'isBullish_{h}'] = isBullish
    df[f'isBearishChange_{h}'] = isBearishChange
    df[f'isBullishChange_{h}'] = isBullishChange
    df[f'isBullishCross_{h}'] = isBullishCross
    df[f'isBearishCross_{h}'] = isBearishCross
    df[f'isBullishSmooth_{h}'] = isBullishSmooth
    df[f'isBearishSmooth_{h}'] = isBearishSmooth
    df[f'colorByCross_{h}'] = colorByCross
    df[f'colorByRate_{h}'] = colorByRate
    df[f'plotColor_{h}'] = plotColor
    df[f'alertBullish_{h}'] = alertBullish
    df[f'alertBearish_{h}'] = alertBearish

    return df

def nadaraya_watson_mc(df, h, r, x_0, smoothColors, lag, mult=2):
    src = df['market_score']
    size = len(src)
    yhat1 = []
    yhat2 = []
    nwe_up = []
    nwe_down = []
    nwe_entry = []
    nwe_exit = []
    wasBearish = []
    wasBullish = []
    isBearish = []
    isBullish = []
    isBearishChange = []
    isBullishChange = []
    isBullishCross = []
    isBearishCross = []
    isBullishSmooth = []
    isBearishSmooth = []
    colorByCross = []
    colorByRate = []
    plotColor = []
    alertBullish = []
    alertBearish = []

    for i in range(size):
        if i >= h:
            window = src[i - h:i]
            yhat1_value = kernel_regression(window, h, h, r, x_0)
            yhat1.append(yhat1_value)

            # Compute envelopes
            mae = np.mean(np.abs(src[i - h:i] - yhat1_value)) * mult
            nwe_up.append(yhat1_value + mae)
            nwe_down.append(yhat1_value - mae)

            # Set entry and exit signals
            nwe_entry.append(1 if src[i] < nwe_down[-1] else 0)
            nwe_exit.append(1 if src[i] > nwe_up[-1] else 0)

            # Trend and crossover conditions
            if i > 1:
                wasBearish.append(yhat1[i - 2] > yhat1[i - 1])
                wasBullish.append(yhat1[i - 2] < yhat1[i - 1])
            else:
                wasBearish.append(False)
                wasBullish.append(False)

            if i > 0:
                isBearish.append(yhat1[i - 1] > yhat1[i])
                isBullish.append(yhat1[i - 1] < yhat1[i])
            else:
                isBearish.append(False)
                isBullish.append(False)

            isBearishChange.append(isBearish[-1] and wasBullish[-1] if wasBullish else False)
            isBullishChange.append(isBullish[-1] and wasBearish[-1] if wasBearish else False)

            if i >= h + lag:
                window = src[i - h - lag:i - lag]
                yhat2.append(kernel_regression(window, h, h, r, x_0))
                
                # Crossover conditions with lag
                if i > 0:
                    isBullishCross.append(yhat2[-1] > yhat1[i])
                    isBearishCross.append(yhat2[-1] < yhat1[i])
                    isBullishSmooth.append(yhat2[-1] > yhat1[i])
                    isBearishSmooth.append(yhat2[-1] < yhat1[i])
                else:
                    isBullishCross.append(False)
                    isBearishCross.append(False)
                    isBullishSmooth.append(False)
                    isBearishSmooth.append(False)

                # Color and alert conditions
                colorByCross.append(1 if isBullishSmooth[-1] else -1 if isBearishSmooth[-1] else 0)
                colorByRate.append(1 if isBullish[-1] else -1 if isBearish[-1] else 0)
                plotColor.append(colorByCross[-1] if smoothColors else colorByRate[-1])
                alertBullish.append(1 if isBullishCross[-1] else 0)
                alertBearish.append(-1 if isBearishCross[-1] else 0)
            else:
                yhat2.append(0)
                isBullishCross.append(False)
                isBearishCross.append(False)
                isBullishSmooth.append(False)
                isBearishSmooth.append(False)
                colorByCross.append(0)
                colorByRate.append(0)
                plotColor.append(0)
                alertBullish.append(0)
                alertBearish.append(0)
        else:
            yhat1.append(0)
            yhat2.append(0)
            nwe_up.append(np.nan)
            nwe_down.append(np.nan)
            nwe_entry.append(0)
            nwe_exit.append(0)
            wasBearish.append(False)
            wasBullish.append(False)
            isBearish.append(False)
            isBullish.append(False)
            isBearishChange.append(False)
            isBullishChange.append(False)
            isBullishCross.append(False)
            isBearishCross.append(False)
            isBullishSmooth.append(False)
            isBearishSmooth.append(False)
            colorByCross.append(0)
            colorByRate.append(0)
            plotColor.append(0)
            alertBullish.append(0)
            alertBearish.append(0)

    # Append the new columns to the dataframe
    df[f'yhat1_{h}'] = yhat1
    df[f'yhat2_{h}'] = yhat2
    df[f'nw_up_{h}'] = nwe_up
    df[f'nw_down_{h}'] = nwe_down
    df[f'nw_entry_{h}'] = nwe_entry
    df[f'nw_exit_{h}'] = nwe_exit
    df[f'wasBearish_{h}'] = wasBearish
    df[f'wasBullish_{h}'] = wasBullish
    df[f'isBearish_{h}'] = isBearish
    df[f'isBullish_{h}'] = isBullish
    df[f'isBearishChange_{h}'] = isBearishChange
    df[f'isBullishChange_{h}'] = isBullishChange
    df[f'isBullishCross_{h}'] = isBullishCross
    df[f'isBearishCross_{h}'] = isBearishCross
    df[f'isBullishSmooth_{h}'] = isBullishSmooth
    df[f'isBearishSmooth_{h}'] = isBearishSmooth
    df[f'colorByCross_{h}'] = colorByCross
    df[f'colorByRate_{h}'] = colorByRate
    df[f'plotColor_{h}'] = plotColor
    df[f'alertBullish_{h}'] = alertBullish
    df[f'alertBearish_{h}'] = alertBearish

    return df