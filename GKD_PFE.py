from freqtrade.strategy import IStrategy
from pandas_ta import ema
import pandas as pd
import numpy as np
import talib
from typing import Dict, List
from pandas import DataFrame

class GKD_PFE(IStrategy):
    # Strategy parameters
    timeframe = "1h"
    minimal_roi = {"0": 0.05, "60": 0.03, "120": 0.01}
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03

    # Custom parameters
    pfe_period = 10          # Lookback period for PFE
    pfe_smooth = 5           # EMA smoothing period for PFE
    pfe_buy_threshold = 30   # PFE threshold for buy signal
    pfe_sell_threshold = -30 # PFE threshold for sell signal
    baseline_period = 20      # Period for baseline EMA
    atr_period = 14          # Period for ATR (volatility filter)
    goldie_locks_min = 0.5   # Min multiplier for Goldie Locks Zone
    goldie_locks_max = 2.0   # Max multiplier for Goldie Locks Zone

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate Polarized Fractal Efficiency (PFE)
        dataframe["pfe"] = self.calculate_pfe(dataframe, self.pfe_period)
        
        # Smooth PFE with EMA
        dataframe["pfe_smooth"] = ema(dataframe["pfe"], length=self.pfe_smooth)

        # Baseline (EMA)
        dataframe["baseline"] = ema(dataframe["close"], length=self.baseline_period)
        dataframe["baseline_diff"] = dataframe["baseline"].diff()
        dataframe["baseline_up"] = dataframe["baseline_diff"] > 0
        dataframe["baseline_down"] = dataframe["baseline_diff"] < 0

        # Volatility (ATR for Goldie Locks Zone)
        dataframe["atr"] = talib.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=self.atr_period
        )
        dataframe["goldie_min"] = dataframe["baseline"] - (dataframe["atr"] * self.goldie_locks_min)
        dataframe["goldie_max"] = dataframe["baseline"] + (dataframe["atr"] * self.goldie_locks_max)

        return dataframe

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

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Standard Entry Logic (GKD-C Confirmation)
        dataframe.loc[
            (dataframe["pfe_smooth"] > self.pfe_buy_threshold) &  # PFE indicates strong upward efficiency
            (dataframe["baseline_up"]) &                         # Baseline confirms uptrend
            (dataframe["close"] >= dataframe["goldie_min"]) &    # Within Goldie Locks Zone
            (dataframe["close"] <= dataframe["goldie_max"]),
            "enter_long"
        ] = 1

        if self.can_short == True:
            dataframe.loc[
                (dataframe["pfe_smooth"] < self.pfe_sell_threshold) &  # PFE indicates strong downward efficiency
                (dataframe["baseline_down"]) &                        # Baseline confirms downtrend
                (dataframe["close"] >= dataframe["goldie_min"]) &     # Within Goldie Locks Zone
                (dataframe["close"] <= dataframe["goldie_max"]),
                "enter_short"
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit Logic: PFE reverses or crosses neutral zone
        dataframe.loc[
            (dataframe["pfe_smooth"] < 0),  # PFE indicates weakening or downward efficiency
            "exit_long"
        ] = 1

        if self.can_short == True:
            dataframe.loc[
                (dataframe["pfe_smooth"] > 0),  # PFE indicates weakening or upward efficiency
                "exit_short"
            ] = 1

        return dataframe