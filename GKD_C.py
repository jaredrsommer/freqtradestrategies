from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter, merge_informative_pair,
                                informative)
from pandas_ta import ema, sma, wma, hma, tema, dema, linreg, vwma
import pandas as pd
import numpy as np
import talib
import logging
from typing import Dict, List
from pandas import DataFrame

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class GKD_C(IStrategy):

    # Strategy parameters
    timeframe = "1h"
    minimal_roi = {"0": 0.05, "60": 0.03, "120": 0.01}
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03

    # Hyperopt parameters
    ma_period = IntParameter(10, 50, default=20, space="buy")
    vis_atr = IntParameter(5, 20, default=13, space="buy")
    vis_std = IntParameter(10, 30, default=20, space="buy")
    sed_atr = IntParameter(30, 60, default=40, space="buy")
    sed_std = IntParameter(80, 120, default=100, space="buy")
    threshold_level = DecimalParameter(1.0, 2.0, default=1.4, decimals=2, space="buy")
    pfe_period = IntParameter(5, 20, default=10, space="buy")
    pfe_smooth = IntParameter(3, 10, default=5, space="buy")
    pfe_buy_threshold = IntParameter(20, 50, default=30, space="buy")
    pfe_sell_threshold = IntParameter(-50, -20, default=-30, space="buy")
    fisher_period = IntParameter(5, 20, default=10, space="buy")
    fisher_smooth = IntParameter(3, 10, default=5, space="buy")
    fisher_buy_threshold = DecimalParameter(0.5, 2.0, default=1.0, decimals=2, space="buy")
    fisher_sell_threshold = DecimalParameter(-2.0, -0.5, default=-1.0, decimals=2, space="buy")
    hurst_period = IntParameter(32, 128, default=64, space="buy")
    hurst_smooth_period = IntParameter(3, 10, default=5, space="buy")
    hurst_threshold = DecimalParameter(0.5, 0.7, default=0.55, decimals=2, space="buy")
    hurst_exit_threshold = DecimalParameter(0.2, 0.4, default=0.25, decimals=2, space="buy")
    atr_period = IntParameter(10, 20, default=14, space="buy")
    goldie_locks_min = DecimalParameter(0.1, 0.5, default=0.2, decimals=2, space="buy")
    goldie_locks_max = DecimalParameter(0.8, 2.0, default=1.0, decimals=2, space="buy")

    # Fixed parameters
    lag_suppressor = True
    lag_s_k = 0.5

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # --- Baseline Indicator ---
        # Purpose: Determines trend direction by averaging all 64 moving averages.
        # Calculation: Computes each of the 64 MAs, sums their values, and divides by 64. Trend direction via difference.
        # Role: Filters trades to align with the consensus trend, reducing noise from individual MAs.
        ma_functions = {
            "AMA": self.ama, "ADXvma": self.adxvma, "Ahrens": self.ahrens, "ALXMA": lambda df, p: wma(df["close"], length=7),
            "DSMA": self.dsma, "Donchian": self.donchian, "DEMA": lambda df, p: dema(df["close"], length=p),
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
            "ThreePoleButterworth": self.three_pole_butterworth, "ThreePoleSmoother": self.three_pole_smoother,
            "TMA": self.tma, "TEMA": lambda df, p: tema(df["close"], length=p), "TwoPoleButterworth": self.two_pole_butterworth,
            "TwoPoleSmoother": self.two_pole_smoother, "VIDYA": self.vidya, "VMA": self.vma, "VEMA": self.vema,
            "VWMA": lambda df, p: vwma(df["close"], df["volume"], length=p), "ZeroLagDEMA": self.zero_lag_dema,
            "ZeroLagMA": self.zero_lag_ma, "ZeroLagTEMA": self.zero_lag_tema
        }
        ma_values = pd.DataFrame(index=dataframe.index)
        for ma_name, ma_func in ma_functions.items():
            try:
                ma_values[ma_name] = ma_func(dataframe, self.ma_period.value)
            except Exception as e:
                logger.warning(f"Error computing MA {ma_name}: {e}")
                ma_values[ma_name] = np.nan
        dataframe["baseline"] = ma_values.mean(axis=1, skipna=True)
        dataframe["baseline_diff"] = dataframe["baseline"].diff()
        dataframe["baseline_up"] = dataframe["baseline_diff"] > 0
        dataframe["baseline_down"] = dataframe["baseline_diff"] < 0

        # --- Damiani Volatmeter Indicator ---
        # Purpose: Measures volatility breakouts by comparing short-term and long-term ATRs.
        # Calculation: Volatmeter = (short-term ATR / long-term ATR) + lag_suppressor_term; anti-threshold = threshold_level - (short_std / long_std).
        # Role: Confirms volatility/volume breakouts for trade entries.
        dataframe["short_atr"] = talib.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=self.vis_atr.value
        )
        dataframe["long_atr"] = talib.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=self.sed_atr.value
        )
        dataframe["volatmeter"] = dataframe["short_atr"] / dataframe["long_atr"]
        if self.lag_suppressor:
            dataframe["volatmeter_lag"] = dataframe["volatmeter"].shift(1) - dataframe["volatmeter"].shift(3)
            dataframe["volatmeter"] = dataframe["volatmeter"] + self.lag_s_k * dataframe["volatmeter_lag"].fillna(0)
        dataframe["short_std"] = dataframe["close"].rolling(window=self.vis_std.value).std()
        dataframe["long_std"] = dataframe["close"].rolling(window=self.sed_std.value).std()
        dataframe["anti_thres"] = dataframe["short_std"] / dataframe["long_std"]
        dataframe["t"] = self.threshold_level.value - dataframe["anti_thres"].fillna(0)

        # --- Polarized Fractal Efficiency (PFE) Indicator ---
        # Purpose: Confirms trend efficiency as Confirmation 1.
        # Calculation: PFE = 100 * (straight-line distance / path length), polarized, smoothed with EMA.
        # Role: Signals strong trend efficiency for entries.
        dataframe["pfe"] = self.calculate_pfe(dataframe, self.pfe_period.value)
        dataframe["pfe_smooth"] = ema(dataframe["pfe"], length=self.pfe_smooth.value)

        # --- Fisher Transform Indicator ---
        # Purpose: Confirms trend reversals as Confirmation 2.
        # Calculation: Fisher Transform = 0.5 * ln((1 + norm) / (1 - norm)), norm is scaled median price, smoothed with EMA.
        # Role: Enhances entry signals with reversal detection.
        dataframe["fisher"] = self.calculate_fisher(dataframe, self.fisher_period.value)
        dataframe["fisher_smooth"] = ema(dataframe["fisher"], length=self.fisher_smooth.value)

        # --- Hurst Exponent Indicator ---
        # Purpose: Confirms persistent trending behavior.
        # Calculation: H = log(R/S) / log(n) on log-returns, smoothed with EMA, with variance-based fallback.
        # Role: Ensures trades occur in persistent market conditions.
        dataframe["log_return"] = np.log(dataframe["close"] / dataframe["close"].shift(1))
        dataframe["hurst"] = self.calculate_hurst(dataframe["log_return"], self.hurst_period.value)
        dataframe["hurst_smooth"] = ema(dataframe["hurst"], length=self.hurst_smooth_period.value)
        logger.debug(f"Recent hurst values: {dataframe['hurst'].tail(10).to_list()}")
        logger.debug(f"Recent hurst_smooth values: {dataframe['hurst_smooth'].tail(10).to_list()}")

        # --- Volatility Filter (Goldie Locks Zone) ---
        # Purpose: Ensures trades occur within acceptable volatility ranges.
        # Calculation: ATR-based bands around the baseline.
        # Role: Filters out extreme volatility or stagnation.
        dataframe["atr"] = talib.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=self.atr_period.value
        )
        dataframe["goldie_min"] = dataframe["baseline"] - (dataframe["atr"] * self.goldie_locks_min.value)
        dataframe["goldie_max"] = dataframe["baseline"] + (dataframe["atr"] * self.goldie_locks_max.value)

        return dataframe

    # Baseline MA Implementations
    def ama(self, df, period):
        return ema(df["close"], length=period)

    def adxvma(self, df, period):
        adx = talib.ADX(df["high"], df["low"], df["close"], timeperiod=period)
        return ema(df["close"], length=period) * (adx / 100)

    def ahrens(self, df, period):
        return (df["open"] + df["close"]).rolling(window=period).mean() / 2

    def dsma(self, df, period):
        std = df["close"].rolling(period).std().fillna(0)
        mean_std = std.mean()
        if mean_std == 0 or np.isnan(mean_std):
            logger.debug(f"DSMA: mean_std is {mean_std}, falling back to EMA")
            return ema(df["close"], length=period)
        alpha = 2 / (1 + period) * (std / (mean_std + 1e-10))
        alpha = alpha.clip(0, 1).fillna(2 / (1 + period))  # Fallback alpha if NaN
        logger.debug(f"DSMA: std={std.tail(5).to_list()}, mean_std={mean_std}, alpha={alpha.tail(5).to_list()}")
        return df["close"].ewm(alpha=alpha, adjust=False).mean()

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

    def three_pole_butterworth(self, df, period):
        a = np.exp(-np.pi / period)
        b = 2 * a * np.cos(1.738 * np.pi / period)
        c = a * a
        d = 1 - b - c
        butter = pd.Series(0, index=df.index)
        for i in range(3, len(df)):
            butter.iloc[i] = d * df["close"].iloc[i] + b * butter.iloc[i-1] + c * butter.iloc[i-2]
        return butter

    def three_pole_smoother(self, df, period):
        return self.three_pole_butterworth(df, period) * 1.1

    def tma(self, df, period):
        sma1 = sma(df["close"], length=period)
        return sma(sma1, length=period)

    def two_pole_butterworth(self, df, period):
        a = np.exp(-np.pi / period)
        b = 2 * a * np.cos(1.414 * np.pi / period)
        c = a * a
        d = 1 - b - c
        butter = pd.Series(0, index=df.index)
        for i in range(2, len(df)):
            butter.iloc[i] = d * df["close"].iloc[i] + b * butter.iloc[i-1] + c * butter.iloc[i-2]
        return butter

    def two_pole_smoother(self, df, period):
        return self.two_pole_butterworth(df, period) * 1.05

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
        close = dataframe["close"]
        pfe = pd.Series(0.0, index=dataframe.index)
        for i in range(period, len(dataframe)):
            price_diff = close.iloc[i] - close.iloc[i - period]
            straight_dist = np.sqrt(price_diff**2 + period**2)
            path_length = 0
            for j in range(i - period + 1, i + 1):
                segment_diff = close.iloc[j] - close.iloc[j - 1]
                segment_length = np.sqrt(segment_diff**2 + 1)
                path_length += segment_length
            if path_length != 0:
                pfe.iloc[i] = 100 * straight_dist / path_length
                if price_diff < 0:
                    pfe.iloc[i] = -pfe.iloc[i]
        return pfe

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
        dataframe.loc[
            (dataframe["pfe_smooth"] > self.pfe_buy_threshold.value) &
            (dataframe["fisher_smooth"] > self.fisher_buy_threshold.value) &
            (dataframe["baseline_up"]) &
            (dataframe["volatmeter"] > dataframe["t"]) &
            (dataframe["hurst_smooth"] > self.hurst_threshold.value) &
            (dataframe["close"] >= dataframe["goldie_min"]) &
            (dataframe["close"] <= dataframe["goldie_max"]),
            "enter_long"
        ] = 1

        dataframe.loc[
            (dataframe["pfe_smooth"] < self.pfe_sell_threshold.value) &
            (dataframe["fisher_smooth"] < self.fisher_sell_threshold.value) &
            (dataframe["baseline_down"]) &
            (dataframe["volatmeter"] > dataframe["t"]) &
            (dataframe["hurst_smooth"] > self.hurst_threshold.value) &
            (dataframe["close"] >= dataframe["goldie_min"]) &
            (dataframe["close"] <= dataframe["goldie_max"]),
            "enter_short"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit Logic
        dataframe.loc[
            (dataframe["pfe_smooth"] < 0) |
            (dataframe["fisher_smooth"] < 0) |
            (dataframe["hurst_smooth"] < self.hurst_exit_threshold.value) |
            (dataframe["volatmeter"] < dataframe["t"]),
            "exit_long"
        ] = 1

        dataframe.loc[
            (dataframe["pfe_smooth"] > 0) |
            (dataframe["fisher_smooth"] > 0) |
            (dataframe["hurst_smooth"] < self.hurst_exit_threshold.value) |
            (dataframe["volatmeter"] < dataframe["t"]),
            "exit_short"
        ] = 1

        return dataframe