import logging
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from sqlalchemy.orm.base import RELATED_OBJECT_OK
from sqlalchemy.sql.elements import or_
import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.persistence import Trade
from technical import indicators
from datetime import datetime, timezone
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, RealParameter,
                                IStrategy, IntParameter, merge_informative_pair)


class pcb20(IStrategy):

    # Add some logging
    logger = logging.getLogger(__name__)
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # ROI1 table:
    
    minimal_roi = {
        "0": 0.309,
        "60": -0.01
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    # use_custom_stoploss = True
    stoploss = -0.015

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30 #30

    # Optimal timeframe for the strategy.
    timeframe = '6h'


    # hyperopt params
    sell_rsil = DecimalParameter(45, 90, default=70)
    sell_rsiu = DecimalParameter(65, 100, default=80)
    sell_wavetrend = DecimalParameter(0, 5, default=0)
    buy_wavetrend = DecimalParameter(-10, -1, default=0)
    buy_rsil = DecimalParameter(0, 40, default=40)
    buy_rsiu = DecimalParameter(40, 65, default=50)

    @property
    def plot_config(self):
        return {
            "main_plot": {
            },
            "subplots": {
                "obv": {
                    "OBV": {
                        "color": "#1b61ab",
                        "type": "line"
                    },
                    "OBVSlope": {
                        "color": "#f18b7a",
                        "type": "line"
                    }
                },
                "wavetrend": {
                    "wave_t1": {
                        "color": "#1b61ab",
                        "type": "line"
                    },
                    "wave_t2": {
                        "color": "#f18b7a",
                        "type": "line"
                    }
                },
            }
        }



    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # WaveTrend
        ap = (0.25 * (dataframe['high'] + dataframe['low'] + dataframe["close"] + dataframe["open"]))
        esa = ta.EMA(ap, timeperiod = 10)
        d = ta.EMA(abs(ap - esa), timeperiod = 10)

        dataframe["wave_ci"] = (ap-esa) / (0.015 * d)
        dataframe["wave_t1"] = ta.EMA(dataframe["wave_ci"], timeperiod = 21)
        dataframe["wave_t2"] = ta.SMA(dataframe["wave_t1"], timeperiod = 4)
        dataframe["wave_t1_pc"] = round((dataframe["wave_t1"] - dataframe["wave_t1"].shift()) / abs(dataframe["wave_t1"]) * 100, 2)


        # # Bollinger!
        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # dataframe['bb.lower'] = bollinger['lower']
        # dataframe['bb.middle'] = bollinger['mid']
        # dataframe['bb.upper'] = bollinger['upper']

        # Added PCB Style OBV
        dataframe['OBV'] = ta.OBV(dataframe)
        dataframe['OBVSlope'] = pta.momentum.slope(dataframe['OBV'])

        # VWMA
        # vwma_period = 13
        # dataframe['vwma'] = ((dataframe["close"] * dataframe["volume"]).rolling(vwma_period).sum() / 
                    # dataframe['volume'].rolling(vwma_period).sum())
        
        # VWAP
        # vwap_period = 20
        # dataframe['vwap'] = qtpylib.rolling_vwap(dataframe, window=vwap_period)
        
        # # VPCI
        # dataframe['vpci'] = indicators.vpci(dataframe, period_long=14)
        
        # #williamsR
        # dataframe['williamspercent'] = indicators.williams_percent(dataframe)

        # # ADX
        # dataframe['adx'] = ta.ADX(dataframe)
        # dataframe['plus.di'] = ta.PLUS_DI(dataframe)
        # dataframe['minus.di'] = ta.MINUS_DI(dataframe)
        # dataframe['plus.di.slope'] = pta.momentum.slope(dataframe['plus.di'])

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['rsi_slope'] = pta.momentum.slope(dataframe['rsi'])
        dataframe['rsi_ma'] = ta.EMA(dataframe['rsi'], timeperiod = 5)
        dataframe['rsi_ma_slope'] = pta.momentum.slope(dataframe['rsi_ma'])

        # # MACD
        # macd = ta.MACD(dataframe)
        # dataframe['macd'] = macd['macd']
        # dataframe['macdsignal'] = macd['macdsignal']
        # dataframe['macdhist'] = macd['macdhist']

        # # Stochastic Fast
        # stoch_fast = ta.STOCHF(dataframe)
        # dataframe['fastd'] = stoch_fast['fastd']
        # dataframe['fastk'] = stoch_fast['fastk']


        # # Stochastic Slow
        # stoch_slow = ta.STOCH(dataframe)
        # dataframe['slowd'] = stoch_slow['slowd']
        # dataframe['slowk'] = stoch_slow['slowk']

        # # Perc
        # dataframe['perc'] = ((dataframe['high'] - dataframe['low']) / dataframe['low']*100)
        # dataframe['avg3_perc'] = ta.EMA(dataframe['perc'], 3)
        # dataframe['perc_norm'] = (dataframe['perc'] - dataframe['perc'].rolling(50).min())/(dataframe['perc'].rolling(50).max() - dataframe['perc'].rolling(50).min())

        return dataframe
        
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
            (dataframe['wave_t1'] >= dataframe['wave_t2'])
            & (dataframe['rsi_ma'] <= self.buy_rsiu.value) 
            & (dataframe['rsi_ma'] >= self.buy_rsil.value) 
            & (dataframe['rsi_ma_slope'] > 0)
            ),
            'buy'] = 1
        
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['wave_t1_pc'] <= self.sell_wavetrend.value) 
                # (dataframe['wave_t1'] <= dataframe['wave_t2'])
                & (dataframe['rsi_ma'] >= self.sell_rsil.value)
                & (dataframe['rsi_ma'] <= self.sell_rsiu.value)
                & (dataframe['rsi_ma_slope'] <= 0)

            ),
            'sell'] = 1

        return dataframe

    
    # "All watched over by machines with loving grace..."
