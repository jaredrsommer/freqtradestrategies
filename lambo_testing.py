import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from sqlalchemy.orm.base import RELATED_OBJECT_OK
from sqlalchemy.sql.elements import or_
import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.exchange import timeframe_to_minutes
from technical import indicators
import logging
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, RealParameter,
                                IStrategy, IntParameter, merge_informative_pair)


class lambotest(IStrategy):

    # Add some logging
    logger = logging.getLogger(__name__)
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # ROI table:
    minimal_roi = {
        "0": 0.0625,
        "28": 0.05,
        "76": 0.04,
        "125": 0.03,
        "240": 0.02,
        "360": 0
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    # use_custom_stoploss = True
    stoploss = -0.0625 #-0.10

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.0069
    trailing_stop_positive_offset = 0.038
    trailing_only_offset_is_reached = False
    process_only_new_candles = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 20 #30

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = True
    # sell_profit_offset = 0.019
    ignore_roi_if_buy_signal = False


    # hyperopt params
    buy_williams_low = DecimalParameter(-100, -50, default=-58.201)
    buy_williams_high = DecimalParameter(-100, -50, default=-92.098)
    buy_rsi_low = DecimalParameter(0, 30, default=2.342)
    buy_rsi_high = DecimalParameter(30, 60, default=51.22)
    buy_volume = DecimalParameter(1000, 100000, default=5000)

    sell_rsi = DecimalParameter(60, 100, default=60.631)
    sell_williams = DecimalParameter(-40, -10, default=-19.308)
    sell_volume = DecimalParameter(1000, 100000, default=1000)


    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }
    
    @property
    def plot_config(self):
        return {
            "main_plot": {
                "bb.lower": {
                    "color": "#9c6edc",
                    "type": "line"
                },
                "bb.upper": {
                    "color": "#9c6edc",
                    "type": "line"
                },
                "vwma": {
                    "color": "#4f9f02",
                    "type": "line"
                }
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
                "vpci": {
                    "vpci": {
                        "color": "#d59a7a",
                        "type": "line"
                    }
                },
                "macd": {
                    "macd": {
                        "color": "#1c3d6a",
                        "type": "line"
                    },
                    "macdsignal": {
                        "color": "#873480",
                        "type": "line"
                    },
                    "macdhist": {
                        "color": "#478a87",
                        "type": "bar"
                    }
                },
                "wiliams": {
                    "williamspercent": {
                        "color": "#10f551",
                        "type": "line"
                    }
                },
                "stoch + rsi": {
                    "rsi": {
                        "color": "#d7affd",
                        "type": "line"
                    },
                    "slowd": {
                        "color": "#d7cc5c",
                        "type": "line"
                    },
                    "fastk": {
                        "color": "#186f86",
                        "type": "line"
                    }
                },
                "adx": {
                    "adx": {
                        "color": "#c392cd",
                        "type": "line"
                    },
                    "plus.di": {
                        "color": "#bcd6c5",
                        "type": "line"
                    },
                    "minus.di": {
                        "color": "#eb044c",
                        "type": "line"
                    }
                }
            }
        }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.timeframe) for pair in pairs]

        return informative_pairs


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Retrieve best bid and best ask from the orderbook
        # ------------------------------------
        
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]


        # Bollinger!
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb.lower'] = bollinger['lower']
        dataframe['bb.middle'] = bollinger['mid']
        dataframe['bb.upper'] = bollinger['upper']

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
        
        # VPCI
        dataframe['vpci'] = indicators.vpci(dataframe, period_long=14)
        
        #williamsR
        dataframe['williamspercent'] = indicators.williams_percent(dataframe)

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['plus.di'] = ta.PLUS_DI(dataframe)
        dataframe['minus.di'] = ta.MINUS_DI(dataframe)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Stochastic Fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']


        # Stochastic Slow
        stoch_slow = ta.STOCH(dataframe)
        dataframe['slowd'] = stoch_slow['slowd']
        dataframe['slowk'] = stoch_slow['slowd']


        return dataframe


    def do_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Bollinger!
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb.lower'] = bollinger['lower']
        dataframe['bb.middle'] = bollinger['mid']
        dataframe['bb.upper'] = bollinger['upper']

        # Added PCB Style OBV
        dataframe['OBV'] = ta.OBV(dataframe)
        dataframe['OBVSlope'] = pta.momentum.slope(dataframe['OBV'])
        

        # VWMA
        # vwma_period = 13
        # dataframe['vwma'] = ((dataframe["close"] * dataframe["volume"]).rolling(vwma_period).sum() / 
                    # dataframe['volume'].rolling(vwma_period).sum())
                
        # VPCI
        dataframe['vpci'] = indicators.vpci(dataframe, period_long=14)

        #williamsR
        dataframe['williamspercent'] = indicators.williams_percent(dataframe)

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['plus.di'] = ta.PLUS_DI(dataframe)
        dataframe['minus.di'] = ta.MINUS_DI(dataframe)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Stochastic Fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # Stochastic Slow
        stoch_slow = ta.STOCH(dataframe)
        dataframe['slowd'] = stoch_slow['slowd']
        dataframe['slowk'] = stoch_slow['slowd']

        return dataframe
        
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
            (dataframe['volume'] > self.buy_volume.value) &
            (dataframe['OBVSlope'] > 0) &
            (dataframe['williamspercent'] <= self.buy_williams_low.value) &
            (dataframe['williamspercent'] > self.buy_williams_high.value) &
            (dataframe['rsi'] > self.buy_rsi_low.value) &
            (dataframe['rsi'] <= self.buy_rsi_high.value) &
            (qtpylib.crossed_above(dataframe['fastk'], dataframe['slowd'])) &
            (dataframe['close'] < dataframe['bb.middle'])
            ),'buy'] = 1
        
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['volume'] > self.sell_volume.value) &
                (dataframe['williamspercent'] > self.sell_williams.value) &
                (dataframe['rsi'] > self.sell_rsi.value) |
                (qtpylib.crossed_below(dataframe['plus.di'], dataframe['minus.di'])) |
                (qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])) |
                (qtpylib.crossed_below(dataframe['fastk'], dataframe['slowd'])) &
                (dataframe['close'] > dataframe['bb.upper']) |
                (dataframe['vpci'] >= dataframe['bb.upper'])
            ),
            'sell'] = 1

        return dataframe
    
    # "All watched over by machines with loving grace..."
