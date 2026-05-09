from freqtrade.strategy import (BooleanParameter, DecimalParameter, IStrategy, IntParameter)
import pandas as pd
import numpy as np
import talib.abstract as ta
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.spatial.distance import mahalanobis
from tslearn.metrics import dtw
import logging

logger = logging.getLogger(__name__)

class MKR(IStrategy):
    INTERFACE_VERSION = 3

    # General Settings
    neighbors_count = IntParameter(10, 100, default=55, space="buy")
    max_bars_back = IntParameter(500, 5000, default=629, space="buy")
    feature_count = IntParameter(3, 10, default=4, space="buy")
    color_compression = IntParameter(1, 10, default=1, space="buy")
    show_exits = BooleanParameter(default=True, space="sell")
    use_dynamic_exits = BooleanParameter(default=False, space="sell")

    # Metric Fusion Weights (sum to 1)
    lorentzian_weight = DecimalParameter(0, 1, default=0.4, decimals=2, space="buy")
    mahalanobis_weight = DecimalParameter(0, 1, default=0.3, decimals=2, space="buy")
    cosine_weight = DecimalParameter(0, 1, default=0.3, decimals=2, space="buy")

    # Define available indicators
    INDICATORS = {
        "RSI": {"length": IntParameter(5, 50, default=29, space="buy"), "smooth": IntParameter(1, 20, default=1, space="buy")},
        "WT": {"length": IntParameter(5, 14, default=14, space="buy"), "smooth": IntParameter(2, 20, default=18, space="buy")},
        "CCI": {"length": IntParameter(5, 50, default=6, space="buy"), "smooth": IntParameter(2, 20, default=7, space="buy")},
        "ADX": {"length": IntParameter(5, 50, default=38, space="buy"), "smooth": IntParameter(2, 20, default=2, space="buy")},
        "MACD": {"fast": IntParameter(5, 20, default=12, space="buy"), "slow": IntParameter(20, 40, default=26, space="buy"), "signal": IntParameter(5, 15, default=9, space="buy")},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_selections = [IntParameter(0, len(self.INDICATORS) - 1, default=i % len(self.INDICATORS), space="buy") 
                                   for i in range(self.feature_count.high)]

    # Other Settings
    norm_window = IntParameter(50, 200, default=100, space="buy")  # Reduced range
    use_volatility_filter = BooleanParameter(default=True, space="buy")
    use_regime_filter = BooleanParameter(default=True, space="buy")
    use_adx_filter = BooleanParameter(default=False, space="buy")
    regime_threshold = DecimalParameter(-10, 10, default=6.1, decimals=1, space="buy")
    adx_threshold = IntParameter(0, 100, default=9, space="buy")
    trade_with_kernel = BooleanParameter(default=True, space="buy")
    lookback_window = IntParameter(3, 50, default=26, space="buy")
    relative_weighting = DecimalParameter(0.25, 25, default=17.92, decimals=2, space="buy")
    start_regression_at_bar = IntParameter(0, 25, default=11, space="buy")
    smooth_predictions = BooleanParameter(default=True, space="buy")
    smooth_period = IntParameter(3, 20, default=10, space="buy")

    # Minimal ROI and stoploss
    minimal_roi = {"0": 0.169, "362": 0.126, "729": 0.064, "1381": 0}
    stoploss = -0.1
    timeframe = "1h"

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.013
    trailing_stop_positive_offset = 0.035
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        logger.info(f"Starting populate_indicators for {metadata['pair']}, DataFrame length: {len(dataframe)}")
        logger.info(f"Feature count: {self.feature_count.value}")
        
        dataframe["hlc3"] = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3

        indicator_names = list(self.INDICATORS.keys())
        for i in range(self.feature_count.value):
            indicator_idx = self.feature_selections[i].value
            indicator_name = indicator_names[indicator_idx]
            params = self.INDICATORS[indicator_name]

            if indicator_name == "RSI":
                dataframe[f"feature_{i}"] = ta.RSI(dataframe["close"], timeperiod=params["length"].value)
            elif indicator_name == "WT":
                dataframe[f"feature_{i}"] = self.wave_trend(dataframe, params["length"].value)
            elif indicator_name == "CCI":
                dataframe[f"feature_{i}"] = ta.CCI(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=params["length"].value)
            elif indicator_name == "ADX":
                dataframe[f"feature_{i}"] = ta.ADX(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=params["length"].value)
            elif indicator_name == "MACD":
                macd = ta.MACD(dataframe["close"], fastperiod=params["fast"].value, slowperiod=params["slow"].value, signalperiod=params["signal"].value)
                dataframe[f"feature_{i}"] = macd[0]

            if "smooth" in params and params["smooth"].value > 1:
                dataframe[f"feature_{i}"] = ta.EMA(dataframe[f"feature_{i}"], timeperiod=params["smooth"].value)

            dataframe[f"feature_{i}_norm"] = (
                (dataframe[f"feature_{i}"] - dataframe[f"feature_{i}"].rolling(self.norm_window.value).mean()) /
                dataframe[f"feature_{i}"].rolling(self.norm_window.value).std()
            )
            logger.info(f"Feature {i} ({indicator_name}) Sample: {dataframe[f'feature_{i}'].tail(5).to_list()}")
            logger.info(f"Normalized Feature {i} Sample: {dataframe[f'feature_{i}_norm'].tail(5).to_list()}")

        logger.info(f"DataFrame columns after features: {list(dataframe.columns)}")

        dataframe["atr"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14)
        dataframe["sma_short"] = ta.SMA(dataframe["close"], timeperiod=10)
        dataframe["sma_long"] = ta.SMA(dataframe["close"], timeperiod=50)
        dataframe["adx"] = ta.ADX(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14)

        if self.trade_with_kernel.value:
            dataframe["kernel"] = self.kernel_regression(dataframe["hlc3"], self.lookback_window.value, self.relative_weighting.value)

        dataframe["y_train"] = np.where(dataframe["close"].shift(-4) < dataframe["close"], -1,
                                        np.where(dataframe["close"].shift(-4) > dataframe["close"], 1, 0))
        logger.info(f"y_train sample: {dataframe['y_train'].tail(5).to_list()}")
        logger.info(f"Final DataFrame columns: {list(dataframe.columns)}")
        return dataframe

    def wave_trend(self, dataframe: pd.DataFrame, period: int) -> pd.Series:
        ap = dataframe["hlc3"]
        esa = ta.EMA(ap, timeperiod=period)
        d = ta.EMA(abs(ap - esa), timeperiod=period)
        ci = (ap - esa) / (0.015 * d)
        wt = ta.EMA(ci, timeperiod=period)
        return pd.Series(wt, index=dataframe.index)

    def lorentzian_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.sum(np.log(1 + np.abs(x - y)))

    def mahalanobis_distance(self, x: np.ndarray, y: np.ndarray, cov_matrix: np.ndarray) -> float:
        try:
            inv_cov = np.linalg.inv(cov_matrix)
            return mahalanobis(x, y, inv_cov)
        except np.linalg.LinAlgError:
            return float('inf')

    def cosine_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        return cosine(x, y)

    def kernel_regression(self, series: pd.Series, lookback: int, weight: float) -> pd.Series:
        weights = np.exp(-np.arange(lookback) / weight)
        return series.rolling(lookback).apply(lambda x: np.average(x, weights=weights[:len(x)]), raw=True)

    def classify(self, dataframe: pd.DataFrame) -> pd.Series:
        logger.info("Starting classify method")
        logger.info(f"Feature count: {self.feature_count.value}")
        feature_cols = [f"feature_{i}_norm" for i in range(self.feature_count.value)]
        logger.info(f"Feature columns expected: {feature_cols}")

        missing_cols = [col for col in feature_cols if col not in dataframe.columns]
        if missing_cols:
            logger.error(f"Missing columns in DataFrame: {missing_cols}")
            raise KeyError(f"Missing columns: {missing_cols}")

        feature_df = dataframe[feature_cols + ["y_train"]].dropna()
        features = feature_df[feature_cols].values
        y_train = feature_df["y_train"].values
        logger.info(f"Features shape: {features.shape}, y_train shape: {len(y_train)}")

        cov_matrix = np.cov(features.T) + np.eye(len(feature_cols)) * 1e-6

        predictions = np.zeros(len(dataframe))
        indicator_names = list(self.INDICATORS.keys())
        max_length = max(
            self.INDICATORS[indicator_names[self.feature_selections[i].value]]["length"].value
            if "length" in self.INDICATORS[indicator_names[self.feature_selections[i].value]]
            else self.INDICATORS[indicator_names[self.feature_selections[i].value]]["fast"].value
            for i in range(self.feature_count.value)
        )
        max_smooth = max(
            self.INDICATORS[indicator_names[self.feature_selections[i].value]].get("smooth", IntParameter(1, 1, default=1)).value
            for i in range(self.feature_count.value)
        )
        start_idx = max(max_length + max_smooth + 4, self.neighbors_count.value)  # Removed norm_window
        logger.info(f"Starting classification at index {start_idx}, total bars: {len(dataframe)}")

        feature_idx_map = feature_df.index

        for i in range(start_idx, len(dataframe)):
            if i not in dataframe.index:
                continue
            try:
                feature_idx = feature_idx_map.get_loc(dataframe.index[i])
                current = features[feature_idx]
            except KeyError:
                continue

            past_start = max(0, feature_idx - self.max_bars_back.value)
            past_features = features[past_start:feature_idx]
            past_labels = y_train[past_start:feature_idx]

            if len(past_features) < self.neighbors_count.value:
                continue

            distances = []
            labels = []
            last_distance = -1.0
            for j in range(0, len(past_features)):
                if j % 4 == 0:
                    lorentzian_d = self.lorentzian_distance(current, past_features[j])
                    mahalanobis_d = self.mahalanobis_distance(current, past_features[j], cov_matrix)
                    cosine_d = self.cosine_distance(current, past_features[j])
                    fused_d = (self.lorentzian_weight.value * lorentzian_d +
                               self.mahalanobis_weight.value * mahalanobis_d +
                               self.cosine_weight.value * cosine_d)
                    if fused_d >= last_distance:
                        distances.append(fused_d)
                        labels.append(int(past_labels[j]))
                        if len(labels) > self.neighbors_count.value:
                            idx_to_remove = 0
                            last_distance = sorted(distances)[int(self.neighbors_count.value * 3 / 4)]
                            distances.pop(idx_to_remove)
                            labels.pop(idx_to_remove)

            if len(labels) > 0:
                prediction = sum(labels)
                predictions[i] = np.clip(prediction, -self.neighbors_count.value, self.neighbors_count.value)
            
            if i % 100 == 0:
                logger.debug(f"Index {i}: Prediction {predictions[i]}, Labels: {labels}")

        if self.smooth_predictions.value:
            predictions_series = pd.Series(predictions, index=dataframe.index)
            predictions = ta.EMA(predictions_series.fillna(0), timeperiod=int(self.smooth_period.value))

        logger.info(f"Classification complete, prediction sample: {predictions[-5:].tolist()}")
        return pd.Series(predictions, index=dataframe.index)

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        logger.info("Starting populate_entry_trend")
        dataframe["prediction"] = self.classify(dataframe)
        logger.info(f"Prediction NaN count: {np.isnan(dataframe['prediction']).sum()}, Sample: {dataframe['prediction'].tail(5).to_list()}")

        conditions = [(dataframe["prediction"] > 0)]
        if self.use_volatility_filter.value:
            conditions.append(dataframe["atr"] > dataframe["atr"].shift(1))
        if self.use_regime_filter.value:
            conditions.append(dataframe["sma_short"] > dataframe["sma_long"])
        if self.use_adx_filter.value:
            conditions.append(dataframe["adx"] > self.adx_threshold.value)
        if self.trade_with_kernel.value:
            conditions.append(dataframe["kernel"] > dataframe["kernel"].shift(1))

        dataframe.loc[pd.concat(conditions, axis=1).all(axis=1), "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        logger.info("Starting populate_exit_trend")
        conditions = [(dataframe["prediction"] < 0)]
        if self.use_dynamic_exits.value:
            conditions.append(dataframe["kernel"] < dataframe["kernel"].shift(1))

        dataframe.loc[pd.concat(conditions, axis=1).all(axis=1), "exit_long"] = 1
        return dataframe
