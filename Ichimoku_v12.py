from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
#from technical.indicators import accumulation_distribution
from technical.util import resample_to_interval, resampled_merge
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy
from technical.indicators import ichimoku
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)

import logging
import numpy as np
import pandas as pd
from technical import qtpylib
from pandas import DataFrame
from datetime import datetime, timezone
from typing import Optional
from functools import reduce
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from technical import qtpylib, pivots_points

class Ichimoku_v12(IStrategy):
    """

    """


    minimal_roi = {
        "0": 0.05
    }

    stoploss = -1 #-0.35
    exit_profit_only = True
    use_custom_stoploss = True
    trailing_stop = True
    ignore_roi_if_entry_signal = True
    use_exit_signal = True
    timeframe ='4h'

    cl = IntParameter(10, 30, default=20, space="buy", optimize=True)
    bl = IntParameter(40, 80, default=60, space="buy", optimize=True)
    lag = IntParameter(100, 140, default=120, space="buy", optimize=True)
    dpl = IntParameter(20, 40, default=30, space="buy", optimize=True)

    buy_offset1 = DecimalParameter(low=0.98, high=0.99, decimals=2, default=0.99, space='buy', optimize=False, load=True)
    buy_offset2 = DecimalParameter(low=0.90, high=0.95, decimals=2, default=0.94, space='buy', optimize=False, load=True)
    sell_offset1 = DecimalParameter(low=1.01, high=1.05, decimals=2, default=1.05, space='sell', optimize=True, load=True)
    sell_offset2 = DecimalParameter(low=1.05, high=1.10, decimals=2, default=1.05, space='sell', optimize=True, load=True)

    ### trailing stop loss optimiziation ###
    tsl_target5 = DecimalParameter(low=0.2, high=0.4, decimals=1, default=0.3, space='sell', optimize=True, load=True)
    ts5 = DecimalParameter(low=0.04, high=0.06, default=0.05, decimals=2,space='sell', optimize=True, load=True)
    tsl_target4 = DecimalParameter(low=0.15, high=0.2, default=0.2, decimals=2, space='sell', optimize=True, load=True)
    ts4 = DecimalParameter(low=0.03, high=0.05, default=0.045, decimals=2,  space='sell', optimize=True, load=True)
    tsl_target3 = DecimalParameter(low=0.10, high=0.15, default=0.15, decimals=2,  space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, decimals=3,  space='sell', optimize=True, load=True)


        ### protections ####
    # CooldownPeriod 
    cooldown_lookback = IntParameter(0, 48, default=5, space="protection", optimize=True)
    
    # StoplossGuard    
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    stop_protection_only_per_pair = BooleanParameter(default=False, space="protection", optimize=True)
    stop_protection_only_per_side = BooleanParameter(default=False, space="protection", optimize=True)
    stop_protection_trade_limit = IntParameter(1, 10, default=4, space="protection", optimize=True)
    stop_protection_required_profit = DecimalParameter(-1.0, 3.0, default=0.0, space="protection", optimize=True)

    # LowProfitPairs    
    use_lowprofit_protection = BooleanParameter(default=True, space="protection", optimize=True)
    lowprofit_protection_lookback = IntParameter(1, 10, default=6, space="protection", optimize=True)
    lowprofit_trade_limit = IntParameter(1, 10, default=4, space="protection", optimize=True)
    lowprofit_stop_duration = IntParameter(1, 100, default=60, space="protection", optimize=True)
    lowprofit_required_profit = DecimalParameter(-1.0, 3.0, default=0.0, space="protection", optimize=True)
    lowprofit_only_per_pair = BooleanParameter(default=False, space="protection", optimize=True)

    # MaxDrawdown    
    use_maxdrawdown_protection = BooleanParameter(default=True, space="protection", optimize=True)
    maxdrawdown_protection_lookback = IntParameter(1, 10, default=6, space="protection", optimize=True)
    maxdrawdown_trade_limit = IntParameter(1, 20, default=10, space="protection", optimize=True)
    maxdrawdown_stop_duration = IntParameter(1, 100, default=6, space="protection", optimize=True)
    maxdrawdown_allowed_drawdown = DecimalParameter(0.01, 0.10, default=0.0, decimals=2, space="protection", optimize=True)

    # startup_candle_count: int = 2

    # trailing stoploss
    #trailing_stop = True
    #trailing_stop_positive = 0.40 #0.35
    #trailing_stop_positive_offset = 0.50
    #trailing_only_offset_is_reached = False

    @property
    def protections(self):
    
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 24 * 3,
                "trade_limit": self.stop_protection_trade_limit.value,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": self.stop_protection_only_per_pair.value,
                "required_profit": self.stop_protection_required_profit.value,
            "only_per_side": self.stop_protection_only_per_side.value
        })

        if self.use_lowprofit_protection.value:
            prot.append({
                    "method": "LowProfitPairs",
                    "lookback_period_candles": self.lowprofit_protection_lookback.value,
                    "trade_limit": self.lowprofit_trade_limit.value,
                    "stop_duration_candles": self.lowprofit_stop_duration.value,
                    "required_profit": self.lowprofit_required_profit.value,
                    "only_per_pair": self.lowprofit_only_per_pair.value
        })

        if self.use_maxdrawdown_protection.value:
            prot.append({
                    "method": "MaxDrawdown",
                    "lookback_period_candles": self.maxdrawdown_protection_lookback.value,
                    "trade_limit": self.maxdrawdown_trade_limit.value,
                    "stop_duration_candles": self.maxdrawdown_stop_duration.value,
                    "max_allowed_drawdown": self.maxdrawdown_allowed_drawdown.value
        })

        return prot    

    ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:


        for stop5 in self.tsl_target5.range:
            if (current_profit > stop5):
                for stop5a in self.ts5.range:
                    self.dp.send_msg(f'*** {pair} *** Profit: {current_profit} - lvl5 {stop5}/{stop5a} activated')
                    return stop5a 
        for stop4 in self.tsl_target4.range:
            if (current_profit > stop4):
                for stop4a in self.ts4.range:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl4 {stop4}/{stop4a} activated')
                    return stop4a 
        for stop3 in self.tsl_target3.range:
            if (current_profit > stop3):
                for stop3a in self.ts3.range:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl3 {stop3}/{stop3a} activated')
                    return stop3a 

        return self.stoploss

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        for c in self.cl.range:
            _cl = self.cl.value

        for b in self.bl.range:
            _bl = self.bl.value

        for l in self.lag.range:
            _lag = self.lag.value

        for d in self.dpl.range:
            _dpl = self.dpl.value

        heikinashi = qtpylib.heikinashi(dataframe)

        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        ichi = ichimoku(dataframe, conversion_line_period=_cl, base_line_periods=_bl, laggin_span=_lag, displacement=_dpl)
        dataframe['chikou_span'] = ichi['chikou_span']
        dataframe['tenkan'] = ichi['tenkan_sen']
        dataframe['kijun'] = ichi['kijun_sen']
        dataframe['senkou_a'] = ichi['senkou_span_a']
        dataframe['senkou_b'] = ichi['senkou_span_b']
        dataframe['cloud_green'] = ichi['cloud_green']
        dataframe['cloud_red'] = ichi['cloud_red']

        dataframe['200MA'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['buy_offset1'] = dataframe['200MA'] * self.buy_offset1.value
        dataframe['buy_offset2'] = dataframe['200MA'] * self.buy_offset2.value
        dataframe['sell_offset1'] = dataframe['200MA'] * self.sell_offset1.value
        dataframe['sell_offset2'] = dataframe['200MA'] * self.sell_offset2.value

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        conditions = []


        df.loc[
            (
                (df['close'].shift(1) < df['senkou_b']) &
                (df['close'] > df['senkou_a']) &
                (df['close'] > df['senkou_b']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'close above senkous 1')

        df.loc[
            (
                (df['close'] < df['kijun']) &
                (df['tenkan'] > df['kijun']) &
                (df['close'] > df['senkou_a']) &
                (df['close'] > df['senkou_b']) &
                (df['close'] < df['open']) &
                (df['close'].shift(1) < df['open'].shift(1)) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'close above senkous 2')


        return df

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['close'], dataframe['senkou_b'])) &
                (dataframe['close'] < dataframe['senkou_a']) &
                (dataframe['close'] < dataframe['senkou_b'])
            ),
            'sell'] = 1
        
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['close'], dataframe['senkou_a'])) &
                (dataframe['close'] < dataframe['senkou_a']) &
                (dataframe['close'] < dataframe['senkou_b'])
            ),
            'sell'] = 1

        return dataframe