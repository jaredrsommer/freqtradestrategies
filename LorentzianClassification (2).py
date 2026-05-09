from freqtrade.strategy import (BooleanParameter, DecimalParameter, IStrategy, IntParameter)
import pandas as pd
import numpy as np
import talib.abstract as ta
from scipy.spatial.distance import pdist, squareform
import logging

logger = logging.getLogger(__name__)

class LorentzianClassification(IStrategy):
    INTERFACE_VERSION = 3

    # General Settings
    neighbors_count = IntParameter(10, 100, default=55, space="buy")
    max_bars_back = IntParameter(500, 5000, default=629, space="buy")
    feature_count = IntParameter(3, 5, default=4, space="buy")
    color_compression = IntParameter(1, 10, default=1, space="buy")
    show_exits = BooleanParameter(default=True, space="sell")
    use_dynamic_exits = BooleanParameter(default=False, space="sell")

    # Feature Engineering Settings
    feature_1 = IntParameter(0, 3, default=2, space="buy")  # 0: RSI, 1: WT, 2: CCI, 3: ADX
    feature_2 = IntParameter(0, 3, default=0, space="buy")
    feature_3 = IntParameter(0, 3, default=1, space="buy")
    feature_4 = IntParameter(0, 3, default=0, space="buy")
    feature_5 = IntParameter(0, 3, default=3, space="buy")
    rsi_length = IntParameter(5, 30, default=29, space="buy")
    wt_length = IntParameter(5, 14, default=14, space="buy")
    cci_length = IntParameter(5, 50, default=6, space="buy")
    adx_length = IntParameter(5, 20, default=38, space="buy")
    rsi_smooth = IntParameter(1, 20, default=1, space="buy")
    wt_smooth = IntParameter(2, 20, default=18, space="buy")
    cci_smooth = IntParameter(2, 20, default=7, space="buy")
    adx_smooth = IntParameter(2, 20, default=2, space="buy")
    norm_window = IntParameter(120, 500, default=433, space="buy")

    # Filter Settings
    use_volatility_filter = BooleanParameter(default=True, space="buy")
    use_regime_filter = BooleanParameter(default=True, space="buy")
    use_adx_filter = BooleanParameter(default=False, space="buy")
    regime_threshold = DecimalParameter(-10, 10, default=6.1, decimals=1, space="buy")
    adx_threshold = IntParameter(0, 100, default=9, space="buy")

    # Kernel Regression Settings
    trade_with_kernel = BooleanParameter(default=True, space="buy")
    lookback_window = IntParameter(3, 50, default=26, space="buy")
    relative_weighting = DecimalParameter(0.25, 25, default=17.92, decimals=2, space="buy")
    start_regression_at_bar = IntParameter(0, 25, default=11, space="buy")

    # Smoothing for Predictions
    smooth_predictions = BooleanParameter(default=True, space="buy")
    smooth_period = IntParameter(3, 20, default=10, space="buy")

    # Limit
    limit = DecimalParameter(0.2, 10, default=1, decimals=1, space="buy")
    # Minimal ROI and stoploss
    minimal_roi = {
        "0": 0.169,
        "362": 0.126,
        "729": 0.064,
        "1381": 0
    }
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
        logger.debug(f"hlc3 sample: {dataframe['hlc3'].tail(5).to_list()}")

        feature_map = {0: "RSI", 1: "WT", 2: "CCI", 3: "ADX"}
        features = [
            self.feature_1.value, self.feature_2.value, self.feature_3.value,
            self.feature_4.value, self.feature_5.value
        ]
        length_map = {
            "RSI": self.rsi_length.value,
            "WT": self.wt_length.value,
            "CCI": self.cci_length.value,
            "ADX": self.adx_length.value
        }
        smooth_map = {
            "RSI": self.rsi_smooth.value,
            "WT": self.wt_smooth.value,
            "CCI": self.cci_smooth.value,
            "ADX": self.adx_smooth.value
        }
        # Generate all 5 features to handle max possible feature_count
        for i in range(5):  # Always generate up to feature_4_norm
            f = features[i]
            feature_name = feature_map[f]
            period = length_map[feature_name]
            smooth = smooth_map[feature_name]
            if feature_name == "RSI":
                dataframe[f"feature_{i}"] = ta.RSI(dataframe["close"], timeperiod=period)
            elif feature_name == "WT":
                dataframe[f"feature_{i}"] = self.wave_trend(dataframe, period)
            elif feature_name == "CCI":
                dataframe[f"feature_{i}"] = ta.CCI(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=period)
            elif feature_name == "ADX":
                dataframe[f"feature_{i}"] = ta.ADX(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=period)
            if smooth > 1:
                dataframe[f"feature_{i}"] = ta.EMA(dataframe[f"feature_{i}"], timeperiod=smooth)
            
            dataframe[f"feature_{i}_norm"] = (
                (dataframe[f"feature_{i}"] - dataframe[f"feature_{i}"].rolling(self.norm_window.value).mean()) /
                dataframe[f"feature_{i}"].rolling(self.norm_window.value).std()
            )
        #     logger.info(f"Feature {i} ({feature_name}) NaN count: {np.isnan(dataframe[f'feature_{i}']).sum()}, Sample: {dataframe[f'feature_{i}'].tail(5).to_list()}")
        #     logger.info(f"Normalized Feature {i} NaN count: {np.isnan(dataframe[f'feature_{i}_norm']).sum()}, Sample: {dataframe[f'feature_{i}_norm'].tail(5).to_list()}")

        # logger.info(f"DataFrame columns after features: {list(dataframe.columns)}")

        dataframe["atr"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14)
        dataframe["sma_short"] = ta.SMA(dataframe["close"], timeperiod=10)
        dataframe["sma_long"] = ta.SMA(dataframe["close"], timeperiod=50)
        dataframe["adx"] = ta.ADX(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14)
        dataframe["kernel"] = self.kernel_regression(dataframe["hlc3"], self.lookback_window.value, self.relative_weighting.value)

        dataframe["y_train"] = np.where(dataframe["close"].shift(-4) < dataframe["close"], -1,
                                        np.where(dataframe["close"].shift(-4) > dataframe["close"], 1, 0))
        # logger.info(f"y_train sample: {dataframe['y_train'].tail(5).to_list()}")

        # logger.info(f"Final DataFrame columns: {list(dataframe.columns)}")
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

    def kernel_regression(self, series: pd.Series, lookback: int, weight: float) -> pd.Series:
        weights = np.exp(-np.arange(lookback) / weight)
        return series.rolling(lookback).apply(lambda x: np.average(x, weights=weights[:len(x)]), raw=True)

    def classify(self, dataframe: pd.DataFrame) -> pd.Series:
        logger.info("Starting classify method")
        logger.info(f"Feature count in classify: {self.feature_count.value}")
        feature_cols = [f"feature_{i}_norm" for i in range(self.feature_count.value)]
        logger.info(f"Feature columns expected: {feature_cols}")

        # Check for missing columns
        missing_cols = [col for col in feature_cols if col not in dataframe.columns]
        if missing_cols:
            logger.error(f"Missing columns in DataFrame: {missing_cols}")
            raise KeyError(f"Missing columns: {missing_cols}")

        feature_df = dataframe[feature_cols + ["y_train"]].dropna()
        features = feature_df[feature_cols].values
        y_train = feature_df["y_train"].values
        logger.info(f"Features shape: {features.shape}, y_train shape: {len(y_train)}")

        predictions = np.zeros(len(dataframe))
        start_idx = max(max(self.rsi_length.value, self.wt_length.value, self.cci_length.value, self.adx_length.value) + 
                        max(self.rsi_smooth.value, self.wt_smooth.value, self.cci_smooth.value, self.adx_smooth.value) + 
                        self.norm_window.value + 4, self.neighbors_count.value)
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
                    d = self.lorentzian_distance(current, past_features[j])
                    if d >= last_distance:
                        distances.append(d)
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

        conditions = [((dataframe["prediction"] > self.limit.value) & (dataframe["prediction"].shift() < dataframe['prediction']))]
        # if self.use_volatility_filter.value:
        #     conditions.append(dataframe["atr"] > dataframe["atr"].shift(1))
        # if self.use_regime_filter.value:
        #     conditions.append(dataframe["sma_short"] > dataframe["sma_long"])
        # if self.use_adx_filter.value:
        #     conditions.append(dataframe["adx"] > self.adx_threshold.value)
        # if self.trade_with_kernel.value:
        #     conditions.append(dataframe["kernel"] > dataframe["kernel"].shift(1))

        dataframe.loc[pd.concat(conditions, axis=1).all(axis=1), "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        logger.info("Starting populate_exit_trend")
        conditions = [((dataframe["prediction"] < -self.limit.value) & (dataframe["prediction"].shift() > dataframe['prediction']))]
        if self.use_dynamic_exits.value:
            conditions.append(dataframe["kernel"] < dataframe["kernel"].shift(1))

        dataframe.loc[pd.concat(conditions, axis=1).all(axis=1), "exit_long"] = 1
        return dataframe