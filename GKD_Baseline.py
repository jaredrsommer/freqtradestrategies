from freqtrade.strategy import IStrategy
from pandas_ta import ema, sma, wma, hma, tema, dema, linreg, vwma
import pandas as pd
import numpy as np
import talib
from freqtrade.exchange import timeframe_to_minutes
from typing import Dict, List
from pandas import DataFrame

class GKD_Baseline(IStrategy):
    # Strategy parameters
    timeframe = "1h"
    minimal_roi = {"0": 0.05, "60": 0.03, "120": 0.01}
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03

    # Custom parameters
    ma_type = "EMA"  # Select one of the 64 MAs
    ma_period = 20   # Period for the moving average
    atr_period = 14  # Period for ATR (volatility filter)
    goldie_locks_min = 0.5  # Min multiplier for Goldie Locks Zone
    goldie_locks_max = 2.0  # Max multiplier for Goldie Locks Zone

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Moving Averages (implementing all 64 as described)
        ma_functions = {
            "AMA": lambda df, p: self.ama(df, p),  # Adaptive Moving Average (placeholder)
            "ADXvma": lambda df, p: self.adxvma(df, p),  # ADX Volatility MA (placeholder)
            "Ahrens": lambda df, p: self.ahrens(df, p),  # Ahrens MA
            "ALXMA": lambda df, p: wma(df["close"], length=7),  # Alexander MA (7-period WMA)
            "DSMA": lambda df, p: self.dsma(df, p),  # Deviation-Scaled MA (placeholder)
            "Donchian": lambda df, p: self.donchian(df, p),  # Donchian Midline
            "DEMA": lambda df, p: dema(df["close"], length=p),
            "DSEMA": lambda df, p: self.dsema(df, p),  # Double Smoothed EMA
            "DSFEMA": lambda df, p: self.dsfema(df, p),  # Double Smoothed Fast EMA
            "DSRWEMA": lambda df, p: self.dsrwema(df, p),  # Double Smoothed Range Weighted EMA
            "DSWEMA": lambda df, p: self.dswema(df, p),  # Double Smoothed Wilders EMA
            "DWMA": lambda df, p: self.dwma(df, p),  # Double Weighted MA
            "EOTF": lambda df, p: ema(df["close"], length=p),  # Ehlers Optimal Tracking Filter (placeholder)
            "EMA": lambda df, p: ema(df["close"], length=p),
            "FEMA": lambda df, p: ema(df["close"], length=int(p/2)),  # Fast EMA
            "FRAMA": lambda df, p: self.frama(df, p),  # Fractal Adaptive MA (placeholder)
            "GDEMA": lambda df, p: self.gdema(df, p),  # Generalized DEMA
            "GDDEMA": lambda df, p: self.gddema(df, p),  # Generalized Double DEMA
            "HMA1": lambda df, p: hma(df["close"], length=p),  # Hull MA (SMA-based)
            "HMA2": lambda df, p: self.hma_ema(df, p),  # Hull MA (EMA-based)
            "HMA3": lambda df, p: self.hma_wma(df, p),  # Hull MA (WMA-based)
            "HMA4": lambda df, p: self.hma_smma(df, p),  # Hull MA (SMMA-based)
            "IE2": lambda df, p: self.t3(df, p),  # Early T3 (T3 implementation)
            "ILRS": lambda df, p: self.ilrs(df, p),  # Integral of Linear Regression Slope
            "Instantaneous": lambda df, p: ema(df["close"], length=p),  # Instantaneous Trendline (placeholder)
            "Kalman": lambda df, p: self.kalman(df, p),  # Kalman Filter (placeholder)
            "KAMA": lambda df, p: self.kama(df, p),  # Kaufman Adaptive MA
            "Laguerre": lambda df, p: self.laguerre(df, p),  # Laguerre Filter
            "Leader": lambda df, p: self.leader_ema(df, p),  # Leader EMA
            "LSMA": lambda df, p: linreg(df["close"], length=p),  # Linear Regression MA
            "LWMA": lambda df, p: wma(df["close"], length=p),  # Linear Weighted MA
            "McGinley": lambda df, p: self.mcginley(df, p),  # McGinley Dynamic
            "McNicholl": lambda df, p: ema(df["close"], length=p),  # McNicholl EMA (placeholder)
            "NonLag": lambda df, p: self.nonlag(df, p),  # Non-Lag MA
            "ONMAMA": lambda df, p: ema(df["close"], length=p),  # Ocean NMA (placeholder)
            "OMA": lambda df, p: self.oma(df, p),  # One More MA
            "Parabolic": lambda df, p: self.parabolic_wma(df, p),  # Parabolic Weighted MA
            "PDFMA": lambda df, p: self.pdfma(df, p),  # Probability Density Function MA
            "QRMA": lambda df, p: self.qrma(df, p),  # Quadratic Regression MA
            "REMA": lambda df, p: self.rema(df, p),  # Regularized EMA
            "RWEMA": lambda df, p: self.rwema(df, p),  # Range Weighted EMA
            "Recursive": lambda df, p: self.recursive(df, p),  # Recursive Moving Trendline
            "SDEC": lambda df, p: self.sdec(df, p),  # Simple Decycler
            "SJMA": lambda df, p: ema(df["close"], length=p),  # Simple Jurik MA (placeholder)
            "SMA": lambda df, p: sma(df["close"], length=p),
            "Sine": lambda df, p: self.sine_wma(df, p),  # Sine Weighted MA
            "SLWMA": lambda df, p: self.slwma(df, p),  # Smoothed LWMA
            "SMMA": lambda df, p: self.smma(df, p),  # Smoothed MA
            "Smoother": lambda df, p: self.smoother(df, p),  # Smoother
            "SuperSmoother": lambda df, p: self.super_smoother(df, p),  # Super Smoother
            "T3": lambda df, p: self.t3(df, p),  # T3 MA
            "ThreePoleButterworth": lambda df, p: self.three_pole_butterworth(df, p),  # 3-Pole Butterworth
            "ThreePoleSmoother": lambda df, p: self.three_pole_smoother(df, p),  # 3-Pole Smoother
            "TMA": lambda df, p: self.tma(df, p),  # Triangular MA
            "TEMA": lambda df, p: tema(df["close"], length=p),  # Triple EMA
            "TwoPoleButterworth": lambda df, p: self.two_pole_butterworth(df, p),  # 2-Pole Butterworth
            "TwoPoleSmoother": lambda df, p: self.two_pole_smoother(df, p),  # 2-Pole Smoother
            "VIDYA": lambda df, p: self.vidya(df, p),  # Variable Index Dynamic Average
            "VMA": lambda df, p: self.vma(df, p),  # Variable MA
            "VEMA": lambda df, p: self.vema(df, p),  # Volume Weighted EMA
            "VWMA": lambda df, p: vwma(df["close"], df["volume"], length=p),  # Volume Weighted MA
            "ZeroLagDEMA": lambda df, p: self.zero_lag_dema(df, p),  # Zero-Lag DEMA
            "ZeroLagMA": lambda df, p: self.zero_lag_ma(df, p),  # Zero-Lag MA
            "ZeroLagTEMA": lambda df, p: self.zero_lag_tema(df, p),  # Zero-Lag TEMA
        }

        # Apply selected MA
        if self.ma_type in ma_functions:
            dataframe["baseline"] = ma_functions[self.ma_type](dataframe, self.ma_period)
        else:
            raise ValueError(f"MA type {self.ma_type} not implemented")

        # Volatility (ATR for Goldie Locks Zone)
        dataframe["atr"] = talib.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=self.atr_period
        )
        dataframe["goldie_min"] = dataframe["baseline"] - (dataframe["atr"] * self.goldie_locks_min)
        dataframe["goldie_max"] = dataframe["baseline"] + (dataframe["atr"] * self.goldie_locks_max)

        # Price crossover signals
        dataframe["price_above"] = dataframe["close"] > dataframe["baseline"]
        dataframe["price_below"] = dataframe["close"] < dataframe["baseline"]

        return dataframe

    # Custom MA Implementations (simplified or placeholders)
    def ama(self, df, period):
        # Placeholder: Adaptive MA based on volatility
        return ema(df["close"], length=period)

    def adxvma(self, df, period):
        # Placeholder: ADX-based volatility MA
        adx = talib.ADX(df["high"], df["low"], df["close"], timeperiod=period)
        return ema(df["close"], length=period) * (adx / 100)

    def ahrens(self, df, period):
        # Ahrens MA: Uses open and close
        return (df["open"] + df["close"]).rolling(window=period).mean() / 2

    def dsma(self, df, period):
        # Deviation-Scaled MA: Adjusts EMA based on standard deviation
        std = df["close"].rolling(period).std()
        alpha = 2 / (1 + period) * (std / std.mean())
        return df["close"].ewm(alpha=alpha, adjust=False).mean()

    def donchian(self, df, period):
        # Donchian Midline: (High + Low) / 2
        high = df["high"].rolling(window=period).max()
        low = df["low"].rolling(window=period).min()
        return (high + low) / 2

    def dsema(self, df, period):
        # Double Smoothed EMA
        ema1 = ema(df["close"], length=period)
        return ema(ema1, length=period)

    def dsfema(self, df, period):
        # Double Smoothed Fast EMA
        ema1 = ema(df["close"], length=int(period/2))
        return ema(ema1, length=int(period/2))

    def dsrwema(self, df, period):
        # Double Smoothed Range Weighted EMA
        range_weight = (df["high"] - df["low"]).rolling(window=period).mean()
        weighted_price = df["close"] * range_weight
        ema1 = ema(weighted_price, length=period)
        return ema(ema1, length=period)

    def dswema(self, df, period):
        # Double Smoothed Wilders EMA
        wilder_period = period * 2 - 1
        ema1 = ema(df["close"], length=wilder_period)
        return ema(ema1, length=wilder_period)

    def dwma(self, df, period):
        # Double Weighted MA
        wma1 = wma(df["close"], length=period)
        return wma(wma1, length=period)

    def frama(self, df, period):
        # Placeholder: Fractal Adaptive MA
        return ema(df["close"], length=period)

    def gdema(self, df, period, vol_factor=0.5):
        # Generalized DEMA
        ema1 = ema(df["close"], length=period)
        ema2 = ema(ema1, length=period)
        return (1 + vol_factor) * ema1 - vol_factor * ema2

    def gddema(self, df, period):
        # Generalized Double DEMA
        gdema1 = self.gdema(df, period)
        return self.gdema(pd.Series(gdema1), period)

    def hma_ema(self, df, period):
        # Hull MA with EMA smoothing
        wma1 = wma(df["close"], length=int(period/2)) * 2
        wma2 = wma(df["close"], length=period)
        raw_hma = wma1 - wma2
        return ema(raw_hma, length=int(np.sqrt(period)))

    def hma_wma(self, df, period):
        # Hull MA with WMA smoothing
        wma1 = wma(df["close"], length=int(period/2)) * 2
        wma2 = wma(df["close"], length=period)
        raw_hma = wma1 - wma2
        return wma(raw_hma, length=int(np.sqrt(period)))

    def hma_smma(self, df, period):
        # Hull MA with SMMA smoothing
        wma1 = wma(df["close"], length=int(period/2)) * 2
        wma2 = wma(df["close"], length=period)
        raw_hma = wma1 - wma2
        return self.smma(pd.Series(raw_hma), int(np.sqrt(period)))

    def t3(self, df, period, v_factor=0.7):
        # T3 Moving Average
        ema1 = ema(df["close"], length=period)
        ema2 = ema(ema1, length=period)
        ema3 = ema(ema2, length=period)
        c1 = -v_factor ** 3
        c2 = 3 * v_factor ** 2 * (1 + v_factor)
        c3 = -6 * v_factor ** 2 - 3 * v_factor * (1 + v_factor)
        c4 = 1 + 3 * v_factor + v_factor ** 3 + 3 * v_factor ** 2
        return c1 * ema3 + c2 * ema2 + c3 * ema1 + c4 * df["close"]

    def ilrs(self, df, period):
        # Integral of Linear Regression Slope
        lsma = linreg(df["close"], length=period)
        return lsma.cumsum() / period

    def kalman(self, df, period):
        # Placeholder: Kalman Filter
        return ema(df["close"], length=period)

    def kama(self, df, period, fast=2, slow=30):
        # Kaufman Adaptive MA
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
        # Laguerre Filter
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
        # Leader EMA (simplified)
        return ema(df["close"], length=period)

    def mcginley(self, df, period):
        # McGinley Dynamic
        mg = pd.Series(df["close"].iloc[0], index=df.index)
        for i in range(1, len(df)):
            mg.iloc[i] = mg.iloc[i-1] + (df["close"].iloc[i] - mg.iloc[i-1]) / (period * (df["close"].iloc[i] / mg.iloc[i-1]) ** 4)
        return mg

    def nonlag(self, df, period):
        # Non-Lag MA (simplified)
        return ema(df["close"], length=int(period/2))

    def oma(self, df, period):
        # One More MA (simplified Jurik-style)
        ema1 = ema(df["close"], length=period)
        ema2 = ema(ema1, length=period)
        return ema(ema2, length=period)

    def parabolic_wma(self, df, period, power=2):
        # Parabolic Weighted MA
        weights = np.array([((i+1)/period)**power for i in range(period)])
        weights = weights / weights.sum()
        return df["close"].rolling(window=period).apply(lambda x: np.sum(x * weights), raw=True)

    def pdfma(self, df, period):
        # Probability Density Function MA
        weights = np.exp(-np.linspace(-2, 2, period)**2)
        weights = weights / weights.sum()
        return df["close"].rolling(window=period).apply(lambda x: np.sum(x * weights), raw=True)

    def qrma(self, df, period):
        # Quadratic Regression MA
        x = np.arange(period)
        def quad_reg(y):
            coeffs = np.polyfit(x, y, 2)
            return np.polyval(coeffs, period-1)
        return df["close"].rolling(window=period).apply(quad_reg, raw=True)

    def rema(self, df, period):
        # Regularized EMA
        alpha = 2 / (period + 1)
        return ema(df["close"], length=period) * (1 + alpha)

    def rwema(self, df, period):
        # Range Weighted EMA
        range_weight = (df["high"] - df["low"]).rolling(window=period).mean()
        return ema(df["close"] * range_weight, length=period) / ema(range_weight, length=period)

    def recursive(self, df, period):
        # Recursive Moving Trendline (simplified)
        return linreg(df["close"], length=period)

    def sdec(self, df, period):
        # Simple Decycler
        return df["close"] - ema(df["close"], length=period)

    def slwma(self, df, period):
        # Smoothed LWMA
        lwma1 = wma(df["close"], length=period)
        return ema(lwma1, length=period)

    def smma(self, df, period):
        # Smoothed MA
        smma = pd.Series(index=df.index)
        smma.iloc[period-1] = df["close"].iloc[:period].mean()
        for i in range(period, len(df)):
            smma.iloc[i] = (smma.iloc[i-1] * (period - 1) + df["close"].iloc[i]) / period
        return smma

    def smoother(self, df, period):
        # Smoother
        return ema(df["close"], length=int(period/2))

    def super_smoother(self, df, period):
        # Super Smoother (simplified)
        a = np.exp(-1.414 * np.pi / period)
        b = 2 * a * np.cos(1.414 * np.pi / period)
        c = a * a
        d = 1 - b - c
        ss = pd.Series(0, index=df.index)
        for i in range(2, len(df)):
            ss.iloc[i] = d * df["close"].iloc[i] + b * ss.iloc[i-1] + c * ss.iloc[i-2]
        return ss

    def three_pole_butterworth(self, df, period):
        # 3-Pole Butterworth
        a = np.exp(-np.pi / period)
        b = 2 * a * np.cos(1.738 * np.pi / period)
        c = a * a
        d = 1 - b - c
        butter = pd.Series(0, index=df.index)
        for i in range(3, len(df)):
            butter.iloc[i] = d * df["close"].iloc[i] + b * butter.iloc[i-1] + c * butter.iloc[i-2]
        return butter

    def three_pole_smoother(self, df, period):
        # 3-Pole Smoother
        return self.three_pole_butterworth(df, period) * 1.1  # Slightly adjusted

    def tma(self, df, period):
        # Triangular MA
        sma1 = sma(df["close"], length=period)
        return sma(sma1, length=period)

    def two_pole_butterworth(self, df, period):
        # 2-Pole Butterworth
        a = np.exp(-np.pi / period)
        b = 2 * a * np.cos(1.414 * np.pi / period)
        c = a * a
        d = 1 - b - c
        butter = pd.Series(0, index=df.index)
        for i in range(2, len(df)):
            butter.iloc[i] = d * df["close"].iloc[i] + b * butter.iloc[i-1] + c * butter.iloc[i-2]
        return butter

    def two_pole_smoother(self, df, period):
        # 2-Pole Smoother
        return self.two_pole_butterworth(df, period) * 1.05  # Slightly adjusted

    def vidya(self, df, period):
        # Variable Index Dynamic Average
        cmo = talib.CMO(df["close"], timeperiod=period)
        alpha = 2 / (period + 1) * (cmo / 100)
        vidya = pd.Series(df["close"].iloc[0], index=df.index)
        for i in range(1, len(df)):
            vidya.iloc[i] = alpha.iloc[i] * df["close"].iloc[i] + (1 - alpha.iloc[i]) * vidya.iloc[i-1]
        return vidya

    def vma(self, df, period):
        # Variable MA (simplified)
        return ema(df["close"], length=period)

    def vema(self, df, period):
        # Volume Weighted EMA
        vol_price = df["close"] * df["volume"]
        return ema(vol_price, length=period) / ema(df["volume"], length=period)

    def zero_lag_dema(self, df, period):
        # Zero-Lag DEMA
        ema1 = ema(df["close"], length=period)
        lag = ema1.shift(period)
        return 2 * ema1 - lag

    def zero_lag_ma(self, df, period):
        # Zero-Lag MA
        ema1 = ema(df["close"], length=period)
        lag = ema1.shift(period)
        return 2 * ema1 - lag

    def zero_lag_tema(self, df, period):
        # Zero-Lag TEMA
        tema1 = tema(df["close"], length=period)
        lag = tema1.shift(period)
        return 2 * tema1 - lag

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Baseline Entry Logic
        dataframe.loc[
            (dataframe["price_above"].shift(1) == False) & 
            (dataframe["price_above"]) &
            (dataframe["close"] >= dataframe["goldie_min"]) &
            (dataframe["close"] <= dataframe["goldie_max"]),
            "enter_long"
        ] = 1

        if self.can_short == True:
            dataframe.loc[
                (dataframe["price_below"].shift(1) == False) & 
                (dataframe["price_below"]) &
                (dataframe["close"] >= dataframe["goldie_min"]) &
                (dataframe["close"] <= dataframe["goldie_max"]),
                "enter_short"
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit Logic
        dataframe.loc[dataframe["price_below"], "exit_long"] = 1

        if self.can_short == True:       
            dataframe.loc[dataframe["price_above"], "exit_short"] = 1

        return dataframe