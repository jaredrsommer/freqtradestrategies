from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
from typing import Dict, List
from pandas import DataFrame
import numpy as np
import talib.abstract as ta
import math
import datetime

class ForexRobootSuperScalper(IStrategy):
    # Hyperopt parameters
    volatility_period = IntParameter(5, 50, default=10, space="buy")
    sensitivity = DecimalParameter(1.0, 20.0, default=1.8, space="buy")
    atr_period = IntParameter(5, 30, default=14, space="sell")
    tp_multiplier = DecimalParameter(0.5, 5.0, default=1.0, space="sell")
    ema_short = IntParameter(5, 50, default=5, space="buy")
    ema_medium = IntParameter(10, 100, default=9, space="buy")
    ema_long = IntParameter(20, 200, default=21, space="buy")

    timeframe = '5m'
    stoploss = -0.99  # Dynamic SL handled in custom_stoploss

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Volatility calculations
        dataframe['range'] = dataframe['high'] - dataframe['low']
        dataframe['log_ret'] = np.log(dataframe['close'] / dataframe['close'].shift(1))
        
        # Parkinson Volatility
        dataframe['parkinson'] = self.parkinson_volatility(dataframe, int(self.volatility_period.value))
        
        # SuperTrend
        dataframe = self.supertrend(dataframe, factor=self.sensitivity.value, period=int(self.volatility_period.value))
        
        # Moving Averages
        dataframe['ema_short'] = ta.EMA(dataframe['close'], timeperiod=int(self.ema_short.value))
        dataframe['ema_medium'] = ta.EMA(dataframe['close'], timeperiod=int(self.ema_medium.value))
        dataframe['ema_long'] = ta.EMA(dataframe['close'], timeperiod=int(self.ema_long.value))
        
        # Trend Strength
        dataframe['adx'] = ta.ADX(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        
        # Volume Analysis
        dataframe['obv'] = ta.OBV(dataframe['close'], dataframe['volume'])
        dataframe['vosc'] = dataframe['obv'] - ta.EMA(dataframe['obv'], timeperiod=20)
        
        return dataframe

    def parkinson_volatility(self, dataframe, period):
        hl_log = np.log(dataframe['high'] / dataframe['low'])
        return hl_log.rolling(period).apply(lambda x: math.sqrt(1.0 / (4 * math.log(2)) * x.pow(2).sum() / period))

    def supertrend(self, dataframe, factor, period):
        hl2 = (dataframe['high'] + dataframe['low']) / 2
        atr = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=period)
        
        upper_band = hl2 + factor * atr
        lower_band = hl2 - factor * atr
        
        st = [0.0] * len(dataframe)
        direction = [1] * len(dataframe)
        
        for i in range(1, len(dataframe)):
            if dataframe['close'][i] > upper_band[i-1]:
                direction[i] = 1
            elif dataframe['close'][i] < lower_band[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
                
            st[i] = lower_band[i] if direction[i] == 1 else upper_band[i]
        
        dataframe['supertrend'] = st
        dataframe['direction'] = direction
        dataframe['atr'] = atr
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Buy when price crosses above supertrend and EMA conditions
        dataframe.loc[
            (dataframe['close'] > dataframe['supertrend']) &
            (dataframe['ema_short'] > dataframe['ema_medium']) &
            (dataframe['ema_medium'] > dataframe['ema_long']) &
            (dataframe['adx'] > 25) &
            (dataframe['vosc'] > 0),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Sell when price crosses below supertrend and EMA conditions
        dataframe.loc[
            (dataframe['close'] < dataframe['supertrend']) &
            (dataframe['ema_short'] < dataframe['ema_medium']) &
            (dataframe['ema_medium'] < dataframe['ema_long']) &
            (dataframe['adx'] > 25) &
            (dataframe['vosc'] < 0),
            'exit_long'] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # Dynamic ATR-based stoploss
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        atr = dataframe['atr'].iat[-1]
        return 3 * atr / current_rate

    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        # Multiple TP levels
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        atr = dataframe['atr'].iat[-1]
        
        if trade.is_short:
            tp1 = trade.open_rate - self.tp_multiplier.value * atr
            if current_rate <= tp1:
                return 'tp1'
        else:
            tp1 = trade.open_rate + self.tp_multiplier.value * atr
            if current_rate >= tp1:
                return 'tp1'
        
        return None

    # Hyperopt parameter space
    @staticmethod
    def generate_hyperopt_parameters():
        return {
            'volatility_period': (5, 50),
            'sensitivity': (1.0, 20.0),
            'atr_period': (5, 30),
            'tp_multiplier': (0.5, 5.0),
            'ema_short': (5, 50),
            'ema_medium': (10, 100),
            'ema_long': (20, 200)
        }