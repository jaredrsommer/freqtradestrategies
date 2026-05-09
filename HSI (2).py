from freqtrade.strategy import IStrategy
from pandas import DataFrame
import pandas_ta as ta
import numpy as np

class HSI(IStrategy):
    # Strategy parameters
    timeframe = "5m"  # 5-minute timeframe (adjustable)
    minimal_roi = {"0": 0.015}  # 1.5% ROI (1:1.5 risk-reward ratio)
    stoploss = -0.01  # 1% stop loss (adjustable)
    trailing_stop = False

    # Define custom indicators
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate Heikin Ashi candles
        heikinashi = ta.ha(dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['ha_open'] = heikinashi['HA_open']
        dataframe['ha_close'] = heikinashi['HA_close']
        dataframe['ha_high'] = heikinashi['HA_high']
        dataframe['ha_low'] = heikinashi['HA_low']

        # Slope of HA close (current close - previous close)
        dataframe['ha_close_slope'] = dataframe['ha_close'] - dataframe['ha_close'].shift(1)

        # Slope of HA open (current open - previous open)
        dataframe['ha_open_slope'] = dataframe['ha_open'] - dataframe['ha_open'].shift(1)

        # Predict next HA open based on current HA open + HA open slope
        dataframe['ha_open_pred'] = dataframe['ha_open'] + dataframe['ha_open_slope']

        # Predict next HA close based on current HA close + HA close slope
        dataframe['ha_close_pred'] = dataframe['ha_close'] + dataframe['ha_close_slope']

        # Shift predictions to compare with actual values on next candle
        dataframe['ha_open_pred_prev'] = dataframe['ha_open_pred'].shift(1)
        dataframe['ha_close_pred_prev'] = dataframe['ha_close_pred'].shift(1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Buy condition:
        # - Previous predicted HA open was below actual open
        # - Current open crosses above predicted HA open
        # - HA open slope and HA close slope are positive
        # - Current close is above previous predicted HA close (additional confirmation)
        dataframe.loc[
            (dataframe['ha_open_pred_prev'] < dataframe['open'].shift(1)) &  # Previous prediction was bearish
            (dataframe['open'] > dataframe['ha_open_pred_prev']) &          # Current open crosses above
            (dataframe['ha_open_slope'] > 0) &                              # HA open slope is bullish
            (dataframe['ha_close_slope'] > 0) &                             # HA close slope is bullish
            (dataframe['close'] > dataframe['ha_close_pred_prev']),         # Close confirms bullishness
            'enter_long'] = 1

        dataframe.loc[
      # Current open crosses above
            (dataframe['ha_close_slope'].shift() < 0) &                              # HA open slope is bullish
            (dataframe['ha_close_slope'] > 0),                         # HA close slope is bullish
            # (dataframe['close'] > dataframe['ha_close_pred_prev']),         # Close confirms bullishness
            'enter_long'] = 1

        # # Sell condition:
        # # - Previous predicted HA open was above actual open
        # # - Current open crosses below predicted HA open
        # # - HA open slope and HA close slope are negative
        # # - Current close is below previous predicted HA close (additional confirmation)
        # dataframe.loc[
        #     (dataframe['ha_open_pred_prev'] > dataframe['open'].shift(1)) &  # Previous prediction was bullish
        #     (dataframe['open'] < dataframe['ha_open_pred_prev']) &          # Current open crosses below
        #     (dataframe['ha_open_slope'] < 0) &                              # HA open slope is bearish
        #     (dataframe['ha_close_slope'] < 0) &                             # HA close slope is bearish
        #     (dataframe['close'] < dataframe['ha_close_pred_prev']),         # Close confirms bearishness
        #     'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long: Open crosses below previous predicted HA open OR close crosses below predicted HA close
        dataframe.loc[
            (dataframe['ha_close_slope'].shift() > 0) &                              # HA open slope is bullish
            (dataframe['ha_close_slope'] < 0), 
            'exit_long'] = 1

        # Sell condition:
        # - Previous predicted HA open was above actual open
        # - Current open crosses below predicted HA open
        # - HA open slope and HA close slope are negative
        # - Current close is below previous predicted HA close (additional confirmation)
        dataframe.loc[
            (dataframe['ha_open_pred_prev'] > dataframe['open'].shift(1)) &  # Previous prediction was bullish
            (dataframe['open'] < dataframe['ha_open_pred_prev']) &          # Current open crosses below
            (dataframe['ha_open_slope'] < 0) &                              # HA open slope is bearish
            (dataframe['ha_close_slope'] < 0) &                             # HA close slope is bearish
            (dataframe['close'] < dataframe['ha_close_pred_prev']),         # Close confirms bearishness
            'exit_long'] = 1

        # # Exit short: Open crosses above previous predicted HA open OR close crosses above predicted HA close
        # dataframe.loc[
        #     (dataframe['enter_short'].shift(1) == 1) &
        #     ((dataframe['open'] > dataframe['ha_open_pred_prev']) | 
        #      (dataframe['close'] > dataframe['ha_close_pred_prev'])),
        #     'exit_short'] = 1

        return dataframe