from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                               IStrategy, IntParameter, RealParameter, merge_informative_pair, informative)
from pandas_ta import ema
import pandas as pd
import numpy as np
import talib
from typing import Dict, List
from pandas import DataFrame

class GKD_FisherTransform(IStrategy):
    # Strategy parameters
    timeframe = "4h"
    minimal_roi = {}
    stoploss = -0.10
    trailing_stop = False
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03

    # Custom parameters
    fisher_period = IntParameter(10, 15, default=12, space="buy")      # Lookback period for Fisher Transform
    fisher_smooth = IntParameter(5, 10, default=8, space="buy")         # EMA smoothing period for Fisher Transform
    fisher_sell_threshold = DecimalParameter(2.0, 3.9, default=2.5, decimals=2, space="sell")
    fisher_buy_threshold = DecimalParameter(-1.0, 2.5, default=-0.5, decimals=2, space="buy")
    baseline_period = IntParameter(5, 21, default=21, space="buy")        # Period for baseline EMA
    atr_period = IntParameter(10, 21, default=14, space="buy")             # Period for ATR (volatility filter)
    goldie_locks = DecimalParameter(1.5, 3.0, default=2.0, decimals=2, space="buy")    # Max multiplier for Goldie Locks Zone

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate Fisher Transform
        dataframe["fisher"] = self.calculate_fisher(dataframe, self.fisher_period.value)
        
        # Smooth Fisher with EMA
        dataframe["fisher_smooth"] = ema(dataframe["fisher"], length=self.fisher_smooth.value)
        dataframe["fisher_trend"] = ema(dataframe["fisher_smooth"], length=21)

        # Baseline (EMA)
        dataframe["baseline"] = ema(dataframe["close"], length=self.baseline_period.value)
        dataframe["baseline_diff"] = dataframe["baseline"].diff()
        dataframe["baseline_up"] = dataframe["baseline_diff"] > 0
        dataframe["baseline_down"] = dataframe["baseline_diff"] < 0

        # Volatility (ATR for Goldie Locks Zone)
        dataframe["atr"] = talib.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=self.atr_period.value
        )
        dataframe["goldie_min"] = dataframe["baseline"] - (dataframe["atr"] * self.goldie_locks.value)
        dataframe["goldie_max"] = dataframe["baseline"] + (dataframe["atr"] * self.goldie_locks.value)

        return dataframe

    def calculate_fisher(self, dataframe: DataFrame, period: int) -> pd.Series:
        # Fisher Transform calculation
        median_price = (dataframe["high"] + dataframe["low"]) / 2
        fisher = pd.Series(0.0, index=dataframe.index)

        for i in range(period, len(dataframe)):
            # Normalize price
            price_window = median_price.iloc[i-period:i]
            price_min = price_window.min()
            price_max = price_window.max()
            if price_max != price_min:
                norm = (median_price.iloc[i] - price_min) / (price_max - price_min)
                norm = 2 * norm - 1  # Scale to [-0.999, 0.999]
                norm = max(min(norm, 0.999), -0.999)  # Prevent division by zero
                # Apply Fisher Transform
                fisher.iloc[i] = 0.5 * np.log((1 + norm) / (1 - norm))
            else:
                fisher.iloc[i] = 0.0

        return fisher

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Standard Entry Logic (GKD-C Confirmation)
        dataframe.loc[
            # (dataframe["fisher_smooth"] > self.fisher_buy_threshold.value) & 
            # (dataframe["fisher_smooth"] < self.fisher_sell_threshold.value) &
            # (dataframe["fisher_smooth"] > 0) & 
            # (dataframe["fisher_smooth"].shift() < -0) &  
            (dataframe["fisher"] < self.fisher_sell_threshold.value) & 
            (dataframe["fisher_smooth"] < dataframe['fisher']), # &

             # Fisher indicates bullish reversal
            # (dataframe["baseline_up"]) &                              # Baseline confirms uptrend
            # (dataframe["close"] >= dataframe["goldie_min"]) &         # Within Goldie Locks Zone
            # (dataframe["close"] <= dataframe["goldie_max"]),
            "enter_long"
        ] = 1


        if self.can_short == True:
            dataframe.loc[
                (dataframe["fisher_smooth"] < self.fisher_sell_threshold.value) &  # Fisher indicates bearish reversal
                (dataframe["baseline_down"]) &                              # Baseline confirms downtrend
                (dataframe["close"] >= dataframe["goldie_min"]) &           # Within Goldie Locks Zone
                (dataframe["close"] <= dataframe["goldie_max"]),
                "enter_short"
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit Logic: Fisher reverses or crosses neutral zone
        dataframe.loc[
            (
            (dataframe["fisher_smooth"].shift() > 0) & 
            (dataframe["fisher_smooth"] < 0) & 
            (dataframe["fisher_smooth"] > dataframe['fisher'])
            ),  # Fisher indicates weakening or bearish reversal
            "exit_long"
        ] = 1

        # dataframe.loc[
        #     (
        #     (dataframe["fisher_smooth"] < self.fisher_buy_threshold.value)
        #     ),  # Fisher indicates weakening or bearish reversal
        #     "exit_long"
        # ] = 1

        if self.can_short == True:
            dataframe.loc[
                (dataframe["fisher_smooth"] > 0),  # Fisher indicates weakening or bullish reversal
                "exit_short"
            ] = 1

        return dataframe


   # 193/1000:    358 trades. 190/25/143 Wins/Draws/Losses. Avg profit   0.87%. Median profit   0.17%. Total profit 31413.09084858 USDT ( 157.07%). Avg duration 23:52:00 min. Objective: -31413.09085


    # # Buy hyperspace params:
    # buy_params = {
    #     "atr_period": 21,
    #     "baseline_period": 16,
    #     "fisher_buy_threshold": -3.5,
    #     "fisher_period": 12,
    #     "fisher_smooth": 8,
    #     "goldie_locks": 1.61,
    # }

    # # Sell hyperspace params:
    # sell_params = {
    #     "fisher_sell_threshold": 3.74,
    # }

    # # ROI table:
    # minimal_roi = {
    #     "0": 1.035,
    #     "1166": 0.25,
    #     "2510": 0.176,
    #     "3562": 0
    # }

    # # Stoploss:
    # stoploss = -0.21

    # # Trailing stop:
    # trailing_stop = True
    # trailing_stop_positive = 0.011
    # trailing_stop_positive_offset = 0.045
    # trailing_only_offset_is_reached = False
    

   #  # Max Open Trades:
   #  max_open_trades = 3
   # 248/1000:    298 trades. 176/0/122 Wins/Draws/Losses. Avg profit   0.81%. Median profit   2.48%. Total profit 21364.22232610 USDT ( 106.82%). Avg duration 1 day, 2:44:00 min. Objective: -21364.22233


   #  # Buy hyperspace params:
   #  buy_params = {
   #      "atr_period": 5,
   #      "baseline_period": 9,
   #      "fisher_buy_threshold": -2.86,
   #      "fisher_period": 9,
   #      "fisher_smooth": 9,
   #      "goldie_locks": 2.38,
   #  }

   #  # Sell hyperspace params:
   #  sell_params = {
   #      "fisher_sell_threshold": 0.23,
   #  }

   #  # ROI table:
   #  minimal_roi = {
   #      "0": 1.234,
   #      "2209": 0.438,
   #      "3223": 0.166,
   #      "12718": 0
   #  }

   #  # Stoploss:
   #  stoploss = -0.22

   #  # Trailing stop:
   #  trailing_stop = True
   #  trailing_stop_positive = 0.01
   #  trailing_stop_positive_offset = 0.034
   #  trailing_only_offset_is_reached = False
    

   #  # Max Open Trades:
   #  max_open_trades = 3
