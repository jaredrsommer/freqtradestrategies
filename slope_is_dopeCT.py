# ==============================================================================================
# The Slope is Dope strategy
#
# Made by:
# ______         _         _      _____                      _         ______            _
# |  _  \       | |       | |    /  __ \                    | |        |  _  \          | |
# | | | | _   _ | |_  ___ | |__  | /  \/ _ __  _   _  _ __  | |_  ___  | | | | __ _   __| |
# | | | || | | || __|/ __|| '_ \ | |    | '__|| | | || '_ \ | __|/ _ \ | | | |/ _` | / _` |
# | |/ / | |_| || |_| (__ | | | || \__/\| |   | |_| || |_) || |_| (_) || |/ /| (_| || (_| |
# |___/   \__,_| \__|\___||_| |_| \____/|_|    \__, || .__/  \__|\___/ |___/  \__,_| \__,_|
#                                               __/ || |
#                                              |___/ |_|
# Version : 1.0
# Date    : 2022-10031
# Remarks :
#    As published, explained and tested in my Youtube video:
#    - https://youtu.be/UvS3ixWG2zs
#    -
# ==============================================================================================

# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
# --------------------------------

# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter)
from scipy.spatial.distance import cosine
import numpy as np
import logging
import pandas as pd
from technical import qtpylib
from pandas import DataFrame
from datetime import datetime, timezone
from typing import Optional
from functools import reduce
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from technical import qtpylib, pivots_points


class slope_is_dopeCT(IStrategy):
    use_custom_stoploss = True
    trailing_stop = True
    ignore_roi_if_entry_signal = True
    use_exit_signal = True
    minimal_roi = {
        "0":  0.10
    }
    # DCA settings
    exit_profit_only = True
    position_adjustment_enable = True
    max_entry_position_adjustment = 0
    max_dca_multiplier = 1
    stoploss = -0.25
    timeframe = '15m'

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30 

    # protections
    cooldown_lookback = IntParameter(2, 48, default=5, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # hyperopt params
    slope_length = IntParameter(5, 30, default=11, space="buy", optimize=True)
    stoploss_length = IntParameter(5, 15, default=10, space="sell", optimize=True)
    rsi_length = IntParameter(5, 14, default=14, space="buy",optimize=True)
    rsi_buy = IntParameter(30, 60, default=55, space="buy", optimize=True)
    fslope_buy = IntParameter(-5, 5, default=0, space="buy", optimize=True)
    sslope_buy = IntParameter(-5, 5, default=0, space="buy", optimize=True)
    fslope_sell = IntParameter(-5, 5, default=0, space="sell", optimize=True)

    #trailing stop loss optimiziation
    tsl_target5 = DecimalParameter(low=0.2, high=0.4, decimals=1, default=0.3, space='sell', optimize=True, load=True)
    ts5 = DecimalParameter(low=0.04, high=0.06, default=0.05, decimals=2,space='sell', optimize=True, load=True)
    tsl_target4 = DecimalParameter(low=0.15, high=0.2, default=0.2, decimals=2, space='sell', optimize=True, load=True)
    ts4 = DecimalParameter(low=0.03, high=0.05, default=0.045, decimals=2,  space='sell', optimize=True, load=True)
    tsl_target3 = DecimalParameter(low=0.10, high=0.15, default=0.15, decimals=2,  space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, decimals=3,  space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.08, high=0.10, default=0.1, decimals=3, space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, decimals=3, space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.05, high=0.08, default=0.06, decimals=3, space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, decimals=3, space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.03, high=0.05, default=0.03, decimals=3, space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.008, high=0.015, default=0.013, decimals=3, space='sell', optimize=True, load=True)


 

    @property
    def protections(self):
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 24 * 3,
                "trade_limit": 1,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": True
            })

        return prot

    ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:


        for stop5 in self.tsl_target5.range:
            if (current_profit > stop5):
                for stop5a in self.ts5.range:
                    self.dp.send_msg(f'*** {pair} *** Profit: {current_profit} - lvl5 {stop5}/{stop5a} activated')
                    return stop5a 
        for stop4 in self.tsl_target4.range:
            if (current_profit > stop4):
                for stop4a in self.ts4.range:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl4 {stop4}/{stop4a} activated')
                    return stop4a 
        for stop3 in self.tsl_target3.range:
            if (current_profit > stop3):
                for stop3a in self.ts3.range:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl3 {stop3}/{stop3a} activated')
                    return stop3a 
        for stop2 in self.tsl_target2.range:
            if (current_profit > stop2):
                for stop2a in self.ts2.range:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl2 {stop2}/{stop2a} activated')
                    return stop2a 
        for stop1 in self.tsl_target1.range:
            if (current_profit > stop1):
                for stop1a in self.ts1.range:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl1 {stop1}/{stop1a} activated')
                    return stop1a 
        for stop0 in self.tsl_target0.range:
            if (current_profit > stop0):
                for stop0a in self.ts0.range:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl0 {stop0}/{stop0a} activated')
                    return stop0a 

        return self.stoploss

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['marketMA'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['fastMA'] = ta.SMA(dataframe, timeperiod=21)
        dataframe['slowMA'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['entryMA'] = ta.SMA(dataframe, timeperiod=3)
        # Calculate slope of slowMA
        # See: https://www.wikihow.com/Find-the-Slope-of-a-Line
        dataframe['sy1'] = dataframe['slowMA'].shift(+(self.slope_length.value))
        dataframe['sy2'] = dataframe['slowMA'].shift(+1)
        sx1 = 1
        sx2 = (self.slope_length.value)
        dataframe['sy'] = dataframe['sy2'] - dataframe['sy1']
        dataframe['sx'] = sx2 - sx1
        dataframe['slow_slope'] = dataframe['sy']/dataframe['sx']
        dataframe['fy1'] = dataframe['fastMA'].shift(+(self.slope_length.value))
        dataframe['fy2'] = dataframe['fastMA'].shift(+1)
        fx1 = 1
        fx2 = (self.slope_length.value)
        dataframe['fy'] = dataframe['fy2'] - dataframe['fy1']
        dataframe['fx'] = fx2 - fx1
        dataframe['fast_slope'] = dataframe['fy']/dataframe['fx']
        # print(dataframe[['date','close', 'slow_slope','fast_slope']].tail(50))

        # ==== Trailing custom stoploss indicator ====
        dataframe['last_lowest'] = dataframe['low'].rolling((self.stoploss_length.value)).min().shift(1)

        for valb in self.rsi_length.range:
            dataframe[f'rsi_{valb}'] = ta.RSI(dataframe, timeperiod=valb)

        return dataframe

    plot_config = {
        "main_plot": {
            # Configuration for main plot indicators.
            "fastMA": {"color": "red"},
            "slowMA": {"color": "blue"},
        },
        "subplots": {
            # Additional subplots
            "rsi": {"rsi": {"color": "blue"}},
            "fast_slope": {"fast_slope": {"color": "red"}, "slow_slope": {"color": "blue"}},
        },
    }


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Only enter when market is bullish (this is a choice)
                (
                # (dataframe['close'] > dataframe['marketMA']) &
                # Only trade when the fast slope is above 0
                (dataframe['fast_slope'] > self.fslope_buy.value) &
                # Only trade when the slow slope is above 0
                (dataframe['slow_slope'] > self.sslope_buy.value) &
                # Only buy when the close price is higher than the 3day average of ten periods ago
                # (dataframe['close'] > dataframe['entryMA'].shift(+(slope_length))) &
                # Or only buy when the close price is higher than the close price of 3 days ago (this is a choice)
                # (qtpylib.crossed_above(dataframe['close'], dataframe['close'].shift(+(self.slope_length.value)))) &
                (dataframe['close'] > dataframe['close'].shift(+(self.slope_length.value))) &
                # Only enter trades when the RSI is higher than 55
                (dataframe[f'rsi_{self.rsi_length.value}'] > self.rsi_buy.value) 
                # Only trade when the fast MA is above the slow MA
                # (dataframe['fastMA'] > dataframe['slowMA'])
                # Or trade when the fase MA crosses above the slow MA (This is a choice...)
                # (qtpylib.crossed_above(dataframe['fastMA'], dataframe['slowMA']))
                )
            ),
            'buy'] = 1


        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (

                # Close or do not trade when fastMA is below slowMA # modified from strategy
                (dataframe['fast_slope'] < self.fslope_sell.value)
                # Or close position when the close price gets below the last lowest candle price configured
                # (AKA candle based (Trailing) stoploss) 
                | (dataframe['close'] < dataframe['last_lowest'])
                # | (dataframe['fastMA'] < dataframe['slowMA'])
                # | (dataframe['close'] < dataframe['fastMA'])
            ),
            'sell'] = 1
        return dataframe


# *   15/100:     51 trades. 48/0/3 Wins/Draws/Losses. Avg profit  14.37%. Median profit   3.99%. Total profit 3497.16371517 USDT ( 349.72%). Avg duration 5 days, 17:41:00 min. Objective: -3497.16372

# 15 mintue timeframe
#     # Buy hyperspace params:
#     buy_params = {
#         "fslope_buy": -1,
#         "rsi_buy": 35,
#         "rsi_length": 8,
#         "slope_length": 16,
#         "sslope_buy": -2,
#     }

#     # Sell hyperspace params:
#     sell_params = {
#         "fslope_sell": -5,
#         "stoploss_length": 12,
#         "ts0": 0.011,
#         "ts1": 0.012,
#         "ts2": 0.015,
#         "ts3": 0.037,
#         "ts4": 0.03,
#         "ts5": 0.05,
#         "tsl_target0": 0.05,
#         "tsl_target1": 0.052,
#         "tsl_target2": 0.097,
#         "tsl_target3": 0.14,
#         "tsl_target4": 0.16,
#         "tsl_target5": 0.3,
#     }

#     # Protection hyperspace params:
#     protection_params = {
#         "cooldown_lookback": 5,  # value loaded from strategy
#         "stop_duration": 5,  # value loaded from strategy
#         "use_stop_protection": True,  # value loaded from strategy
#     }

#     # ROI table:  # value loaded from strategy
#     minimal_roi = {
#         "0": 0.609,
#         "12443": 0.25,
#         "19865": 0.11,
#         "44116": 0
#     }

#     # Stoploss:
#     stoploss = -0.9  # value loaded from strategy

#     # Trailing stop:
#     trailing_stop = True  # value loaded from strategy
#     trailing_stop_positive = 0.03  # value loaded from strategy
#     trailing_stop_positive_offset = 0.28  # value loaded from strategy
#     trailing_only_offset_is_reached = True  # value loaded from strategy
