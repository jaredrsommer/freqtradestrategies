from freqtrade.strategy import IStrategy
from pandas_ta import ema
import pandas as pd
import numpy as np
import talib
import logging
from typing import Dict, List
from pandas import DataFrame

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class GKD_HurstExponent(IStrategy):
    # Strategy parameters
    timeframe = "1h"
    minimal_roi = {"0": 0.05, "60": 0.03, "120": 0.01}
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03

    # Custom parameters
    hurst_period = 64           # Lookback period for Hurst Exponent
    hurst_smooth_period = 5     # EMA smoothing period for Hurst Exponent
    hurst_threshold = 0.55       # Threshold for trending market
    hurst_exit_threshold = 0.25  # Threshold for exit (random/mean-reverting)
    baseline_period = 20        # Period for baseline EMA
    atr_period = 14            # Period for ATR (Goldie Locks Zone)
    goldie_locks_min = 0.2     # Min multiplier for Goldie Locks Zone
    goldie_locks_max = 1.0     # Max multiplier for Goldie Locks Zone

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate log-returns
        dataframe["log_return"] = np.log(dataframe["close"] / dataframe["close"].shift(1))
        
        # Calculate Hurst Exponent
        dataframe["hurst"] = self.calculate_hurst(dataframe["log_return"], self.hurst_period)
        dataframe['zero'] = 0.5
        
        # Smooth Hurst Exponent
        dataframe["hurst_smooth"] = ema(dataframe["hurst"], length=self.hurst_smooth_period)

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

        # Debug logging
        logger.debug(f"Recent hurst values: {dataframe['hurst'].tail(10).to_list()}")
        logger.debug(f"Recent hurst_smooth values: {dataframe['hurst_smooth'].tail(10).to_list()}")

        return dataframe

    def calculate_hurst(self, series: pd.Series, period: int) -> pd.Series:
        # Hurst Exponent using single-window R/S analysis on log-returns
        hurst = pd.Series(np.nan, index=series.index)
        
        for i in range(period, len(series)):
            window = series.iloc[i-period:i].dropna()
            if len(window) < period:
                logger.debug(f"Insufficient data at i={i}, len={len(window)}")
                hurst.iloc[i] = 0.5
                continue
            
            # Mean-adjusted series
            mean = window.mean()
            mean_adj = window - mean
            # Cumulative deviation
            cum_dev = mean_adj.cumsum()
            # Range (R)
            r = cum_dev.max() - cum_dev.min()
            # Standard deviation (S)
            s = window.std()
            
            # R/S calculation
            if s == 0 or r == 0 or np.isnan(s) or np.isnan(r):
                logger.debug(f"Invalid R/S at i={i}: r={r}, s={s}")
                # Fallback: variance-based estimate
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
            
            # Approximate H without Anis-Lloyd correction
            h = np.log(rs) / np.log(period)
            hurst.iloc[i] = np.clip(h, 0, 1)
            logger.debug(f"H at i={i}: rs={rs}, h={h}, clipped={hurst.iloc[i]}")
        
        return hurst

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Volatility/Volume Entry Logic (GKD-V)
        dataframe.loc[
            (dataframe["hurst_smooth"] > self.hurst_threshold) &  # Hurst indicates trending market
            (dataframe["baseline_up"]) &                         # Baseline confirms uptrend
            (dataframe["close"] >= dataframe["goldie_min"]) &    # Within Goldie Locks Zone
            (dataframe["close"] <= dataframe["goldie_max"]),
            "enter_long"
        ] = 1

        dataframe.loc[
            (dataframe["hurst_smooth"] > self.hurst_threshold) &  # Hurst indicates trending market
            (dataframe["baseline_down"]) &                       # Baseline confirms downtrend
            (dataframe["close"] >= dataframe["goldie_min"]) &    # Within Goldie Locks Zone
            (dataframe["close"] <= dataframe["goldie_max"]),
            "enter_short"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit Logic: Hurst indicates random/mean-reverting market
        dataframe.loc[
            (dataframe["hurst_smooth"] < self.hurst_exit_threshold),  # Hurst below threshold
            "exit_long"
        ] = 1

        dataframe.loc[
            (dataframe["hurst_smooth"] < self.hurst_exit_threshold),  # Hurst below threshold
            "exit_short"
        ] = 1

        return dataframe