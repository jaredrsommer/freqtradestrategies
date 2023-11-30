from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
import talib.abstract as ta
from technical import qtpylib, pivots_points
import numpy as np
import logging
import pandas as pd
import pandas_ta as pta
import datetime
from datetime import datetime, timedelta, timezone
from typing import Optional
import talib.abstract as ta
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
from freqtrade.strategy import stoploss_from_open
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
import technical.indicators as ftt

logger = logging.getLogger('freqtrade')


### Change log ###

### Change log ###

def PC(dataframe, in1, in2):
    df = dataframe.copy()
    pc = ((in2-in1)/in1) * 100
    return pc

class dualwave(IStrategy):

    ### Strategy parameters ###
    exit_profit_only = True ### No selling at a loss
    use_custom_stoploss = True
    trailing_stop = False # True
    ignore_roi_if_entry_signal = True
    use_exit_signal = True
    stoploss = -0.25
    # DCA Parameters
    position_adjustment_enable = True
    max_entry_position_adjustment = 0
    max_dca_multiplier = 1
    market_status = 0
    minimal_roi = {
        "0": 0.215,

    }

    ### Hyperoptable parameters ###

    # protections
    cooldown_lookback = IntParameter(24, 48, default=46, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # SMA   
    filterlength = IntParameter(low=15, high=35, default=25, space='sell', optimize=True)
    max_length = CategoricalParameter([24, 48, 72, 96, 144, 192, 240], default=48, space="buy", optimize=False)
    from15 = IntParameter(low=5, high=35, default=25, space='buy', optimize=True)
    from2 = IntParameter(low=2, high=10, default=5, space='buy', optimize=True)


    # Buy Parameters
    rsi_buy = IntParameter(55, 70, default=65, space='buy', optimize=True)
    rsi_buy_safe = IntParameter(40, 55, default=50, space='buy', optimize=True)
    rsi_ma_buypc = IntParameter(-5, 5, default=0, space='buy', optimize=True)
    sma200_buy_pc = IntParameter(-5, 5, default=0, space='buy', optimize=True)
    willr_buy = IntParameter(-50, -20, default=-50, space='buy', optimize=True)
    auto_buy = IntParameter(3, 13, default=3, space='buy', optimize=True)
    auto_buy_down = IntParameter(3, 13, default=3, space='buy', optimize=True)
    auto_buy_bearzzz = IntParameter(3, 13, default=3, space='buy', optimize=True)
    auto_buy_bearzzz_down = IntParameter(3, 10, default=3, space='buy', optimize=True)
    auto_buy_2 = IntParameter(3, 10, default=3, space='buy', optimize=True)
    auto_buy_down_2 = IntParameter(3, 10, default=3, space='buy', optimize=True)
    auto_buy_bearzzz_2 = IntParameter(3, 10, default=3, space='buy', optimize=True)
    auto_buy_bearzzz_down_2 = IntParameter(3, 10, default=3, space='buy', optimize=True)
    fast_wave_buy = IntParameter(-20, 0, default=-20, space='buy', optimize=True)
    slow_wave_buy = IntParameter(-20, 0, default=-20, space='buy', optimize=True)
    fast_wave_buy_pc = IntParameter(-5, 5, default=0, space='buy', optimize=True)
    slow_wave_buy_pc = IntParameter(-5, 5, default=0, space='buy', optimize=True)
    buy_smoothing = IntParameter(low=3, high=35, default=15, space='buy', optimize=True)

    # Sell Parameters
    rsi_sell = IntParameter(55, 70, default=50, space='sell', optimize=True)
    rsi_sell_safe = IntParameter(60, 80, default=70, space='sell', optimize=True)
    rsi_ma_sellpc = IntParameter(-5, 5, default=0, space='sell', optimize=True)
    sma200_sell_pc = IntParameter(-5, 5, default=0, space='sell', optimize=True)
    willr_sell = IntParameter(-50, -20, default=-20, space='sell', optimize=True)
    auto_sell_bull = IntParameter(3, 15, default=4, space='sell', optimize=True)
    auto_sell_bear = IntParameter(3, 15, default=4, space='sell', optimize=True)
    fast_wave_sell = IntParameter(-20, 10, default=0, space='sell', optimize=True)
    slow_wave_sell = IntParameter(-20, 10, default=0, space='sell', optimize=True)
    sell_smoothing = IntParameter(low=5, high=35, default=15, space='sell', optimize=True)

    ###  Buy Weight Mulitpliers ###
    x01 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='buy', optimize=True)
    x02 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='buy', optimize=True)
    x03 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='buy', optimize=True)
    x04 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='buy', optimize=True)
    x05 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='buy', optimize=True)
    x06 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='buy', optimize=True)
    x07 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='buy', optimize=True)
    x08 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='buy', optimize=True)
    x09 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='buy', optimize=True)
    x10 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='buy', optimize=True)

    ###  Sell Weight Mulitpliers ###
    y01 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='sell', optimize=True)
    y02 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='sell', optimize=True)
    y03 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='sell', optimize=True)
    y04 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='sell', optimize=True)
    y05 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='sell', optimize=True)
    y06 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='sell', optimize=True)
    y07 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='sell', optimize=True)
    y08 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='sell', optimize=True)
    y09 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='sell', optimize=True)
    y10 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='sell', optimize=True)

    #trailing stop loss optimiziation
    tsl_target5 = DecimalParameter(low=0.25, high=0.4, decimals=1, default=0.3, space='sell', optimize=True, load=True)
    ts5 = DecimalParameter(low=0.04, high=0.06, default=0.05, space='sell', optimize=True, load=True)
    tsl_target4 = DecimalParameter(low=0.15, high=0.25, default=0.2, space='sell', optimize=True, load=True)
    ts4 = DecimalParameter(low=0.03, high=0.05, default=0.045, space='sell', optimize=True, load=True)
    tsl_target3 = DecimalParameter(low=0.10, high=0.15, default=0.15, space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.08, high=0.10, default=0.1, space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.06, high=0.08, default=0.06, space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.04, high=0.06, default=0.03, space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.008, high=0.015, default=0.01, space='sell', optimize=True, load=True)

    ## Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'ioc'
    }

    # Optimal timeframe for the strategy
    timeframe = '15m'
    informative_timeframe = '2h'


    process_only_new_candles = True
    startup_candle_count = 30
    
    ### protections ###
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
                "trade_limit": 2,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False
            })

        return prot

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def get_informative_indicators(self, metadata: dict):

        dataframe = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.informative_timeframe)

        return dataframe


    ### Dollar Cost Averaging ###
    # This is called when placing the initial order (opening trade)
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        # We need to leave most of the funds for possible further DCA orders
        # This also applies to fixed stakes
        return proposed_stake / self.max_dca_multiplier 

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:

        if current_profit > 0.10 and trade.nr_of_successful_exits == 0:
            # Take half of the profit at +10%
            return -(trade.stake_amount / 2)

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
        for stop2 in self.tsl_target2.range:
            if (current_profit > stop2):
                for stop2a in self.ts2.range:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl2 {stop2}/{stop2a} activated')
                    return stop2a 
        for stop1 in self.tsl_target1.range:
            if (current_profit > stop1):
                for stop1a in self.ts1.range:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl1 {stop1}/{stop1a} activated')
                    return stop1a 
        for stop0 in self.tsl_target0.range:
            if (current_profit > stop0):
                for stop0a in self.ts0.range:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl0 {stop0}/{stop0a} activated')
                    return stop0a 

        return self.stoploss

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.dp:
            inf_tf = '2h'
            pair = metadata['pair']
            informative = self.dp.get_pair_dataframe(pair=pair, timeframe=inf_tf)

            # RSI
            informative['rsi'] = ta.RSI(informative)
            informative['rsi_ma'] = ta.SMA(informative['rsi'], timeperiod=10)
            informative['rsi_ma_pcnt'] = PC(informative, informative['rsi_ma'], informative['rsi_ma'].shift(1))

            # WaveTrend using OHLC4 or HA close - 3/21
            ap = (0.25 * (informative['high'] + informative['low'] + informative["close"] + informative["open"]))
        
            informative['esa'] = ta.EMA(ap, timeperiod = 3)
            informative['d'] = ta.EMA(abs(ap - informative['esa']), timeperiod = 3)
            informative['wave_ci'] = (ap-informative['esa']) / (0.015 * informative['d'])
            informative['wave_t1'] = ta.EMA(informative['wave_ci'], timeperiod = 21)  
            informative['wave_t2'] = ta.SMA(informative['wave_t1'], timeperiod = 3)
            informative['t1_pc'] = PC(informative, informative['wave_t1'], informative['wave_t1'].shift(1))

            # SMA
            informative['200_SMA'] = ta.SMA(informative["close"], timeperiod = 200)
            informative['200_SMAPC'] = PC(informative, informative['200_SMA'], informative['200_SMA'].shift(1) )
            informative['from_200'] = ta.SMA(
                ((((informative['close'] + informative['open']) / 2) - informative['200_SMA']) / informative['close']) * 100, timeperiod=self.from2.value)

            ### BUYING WEIGHTS ###
            informative.loc[(informative['rsi']<self.rsi_buy.value), 'rsi_buy1'] = 1
            informative.loc[(informative['rsi']>self.rsi_buy.value), 'rsi_buy1'] = -1

            informative.loc[(informative['rsi']>informative['rsi_ma']), 'rsi_buy2'] = 1
            informative.loc[(informative['rsi']<informative['rsi_ma']), 'rsi_buy2'] = -1
        
            informative.loc[(informative['rsi_ma_pcnt']>self.rsi_ma_buypc.value), 'rsi_buy3'] = 1
            informative.loc[(informative['rsi_ma_pcnt']<self.rsi_ma_buypc.value), 'rsi_buy3'] = -1
        
            informative.loc[(informative['rsi']<self.rsi_buy_safe.value), 'rsi_buy4'] = 1
            informative.loc[(informative['rsi']>self.rsi_buy_safe.value), 'rsi_buy4'] = 0

            informative['rsi_weight'] = (
                (informative['rsi_buy1']+informative['rsi_buy2']+informative['rsi_buy3']+informative['rsi_buy4'])/4) * self.x01.value

            informative.loc[((informative['close'] > informative['200_SMA']) & (informative['200_SMAPC'] > self.sma200_buy_pc.value)), 'sma_buy1'] = 1
            informative.loc[((informative['close'] < informative['200_SMA']) & (informative['200_SMAPC'] > self.sma200_buy_pc.value)), 'sma_buy1'] = 1
            informative.loc[((informative['close'] > informative['200_SMA']) & (informative['200_SMAPC'] < self.sma200_buy_pc.value)), 'sma_buy1'] = -1
            informative.loc[((informative['close'] < informative['200_SMA']) & (informative['200_SMAPC'] < self.sma200_buy_pc.value)), 'sma_buy1'] = -1

            informative.loc[(informative['200_SMAPC'] > self.sma200_buy_pc.value), 'sma_buy2'] = 1
            informative.loc[(informative['200_SMAPC'] < self.sma200_buy_pc.value), 'sma_buy2'] = -1

            informative['200SMA_weight'] = ((informative['sma_buy1']+informative['sma_buy2'])/2) * self.x02.value

            informative['from_weight'] = 0 #-((informative['from_200']) * self.x03.value)

            informative.loc[(informative['wave_t1']<self.slow_wave_buy.value), 'wave_buy1'] = 1
            informative.loc[(informative['wave_t1']>self.slow_wave_buy.value), 'wave_buy1'] = -1

            informative.loc[(informative['wave_t1']>informative['wave_t1'].shift(1)), 'wave_buy2'] = 1
            informative.loc[(informative['wave_t1']<informative['wave_t1'].shift(1)), 'wave_buy2'] = -1

            informative.loc[(informative['wave_t1']>informative['wave_t2']), 'wave_buy3'] = 1
            informative.loc[(informative['wave_t1']<informative['wave_t2']), 'wave_buy3'] = -1

            informative['wave_weight'] = ((informative['wave_buy1']+informative['wave_buy2']+informative['wave_buy3'])/3) * self.x04.value

            informative['auto_buy'] = informative[['rsi_weight', 'wave_weight', '200SMA_weight', 'from_weight']].sum(axis=1)

            ### SELLING WEIGHTS ###
            informative.loc[(informative['rsi']>self.rsi_sell.value), 'rsi_sell1'] = 1
            informative.loc[(informative['rsi']<self.rsi_sell.value), 'rsi_sell1'] = -1

            informative.loc[(informative['rsi']<informative['rsi_ma']), 'rsi_sell2'] = 1
            informative.loc[(informative['rsi']>informative['rsi_ma']), 'rsi_sell2'] = -1
        
            informative.loc[(informative['rsi_ma_pcnt']<self.rsi_ma_sellpc.value), 'rsi_sell3'] = 1
            informative.loc[(informative['rsi_ma_pcnt']>self.rsi_ma_sellpc.value), 'rsi_sell3'] = -1
        
            informative.loc[(informative['rsi']>self.rsi_sell_safe.value), 'rsi_sell4'] = 1
            informative.loc[(informative['rsi']<self.rsi_sell_safe.value), 'rsi_sell4'] = 0

            informative['rsi_weight'] = (
                (informative['rsi_sell1']+informative['rsi_sell2']+informative['rsi_sell3']+informative['rsi_sell4'])/4) * self.y01.value

            informative.loc[((informative['close'] > informative['200_SMA']) & (informative['200_SMAPC'] > self.sma200_sell_pc.value)), 'sma_sell1'] = -1
            informative.loc[((informative['close'] < informative['200_SMA']) & (informative['200_SMAPC'] > self.sma200_sell_pc.value)), 'sma_sell1'] = -1
            informative.loc[((informative['close'] > informative['200_SMA']) & (informative['200_SMAPC'] < self.sma200_sell_pc.value)), 'sma_sell1'] = 1
            informative.loc[((informative['close'] < informative['200_SMA']) & (informative['200_SMAPC'] < self.sma200_sell_pc.value)), 'sma_sell1'] = 1

            informative.loc[(informative['200_SMAPC'] > self.sma200_sell_pc.value), 'sma_sell2'] = -1
            informative.loc[(informative['200_SMAPC'] < self.sma200_sell_pc.value), 'sma_sell2'] = 1

            informative['200SMA_weight'] = ((informative['sma_sell1']+informative['sma_sell2'])/2) * self.y02.value

            informative['from_weight_sell'] = 0 #((informative['from_200']) * self.y03.value)

            informative.loc[(informative['wave_t1']<self.slow_wave_sell.value), 'wave_sell1'] = -1
            informative.loc[(informative['wave_t1']>self.slow_wave_sell.value), 'wave_sell1'] = 1

            informative.loc[(informative['wave_t1']>informative['wave_t1'].shift(1)), 'wave_sell2'] = -1
            informative.loc[(informative['wave_t1']<informative['wave_t1'].shift(1)), 'wave_sell2'] = 1

            informative.loc[(informative['wave_t1']>informative['wave_t2']), 'wave_sell3'] = -1
            informative.loc[(informative['wave_t1']<informative['wave_t2']), 'wave_sell3'] = 1

            informative['wave_weight'] = ((informative['wave_sell1']+informative['wave_sell2']+informative['wave_sell3'])/3) * self.y04.value

            informative['auto_sell'] = informative[['rsi_weight', 'wave_weight', '200SMA_weight', 'from_weight_sell']].sum(axis=1)

            informative['auto_buy_decision'] = (informative['auto_buy'] - informative['auto_sell'])
            informative['auto_sell_decision'] = (informative['auto_sell'] - informative['auto_buy'])


        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        ### 15m indicators ###

        # WaveTrend using OHLC4 or HA close - 3/21
        ap = (0.25 * (dataframe['high'] + dataframe['low'] + dataframe["close"] + dataframe["open"]))
        
        dataframe['esa'] = ta.EMA(ap, timeperiod = 3)
        dataframe['d'] = ta.EMA(abs(ap - dataframe['esa']), timeperiod = 3)
        dataframe['wave_ci'] = (ap-dataframe['esa']) / (0.015 * dataframe['d'])
        dataframe['wave_t1'] = ta.EMA(dataframe['wave_ci'], timeperiod = 21)  
        dataframe['wave_t2'] = ta.SMA(dataframe['wave_t1'], timeperiod = 3)
        dataframe['t1_pc'] = PC(dataframe, dataframe['wave_t1'], dataframe['wave_t1'].shift(1))

        # Filter ZEMA
        for length in self.filterlength.range:
            dataframe[f'ema_1{length}'] = ta.EMA(dataframe['close'], timeperiod=length)
            dataframe[f'ema_2{length}'] = ta.EMA(dataframe[f'ema_1{length}'], timeperiod=length)
            dataframe[f'ema_dif{length}'] = dataframe[f'ema_1{length}'] - dataframe[f'ema_2{length}']
            dataframe[f'zema_{length}'] = dataframe[f'ema_1{length}'] + dataframe[f'ema_dif{length}']

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_ma'] = ta.SMA(dataframe['rsi'], timeperiod=10)
        dataframe['rsi_ma_pcnt'] = PC(dataframe, dataframe['rsi_ma'], dataframe['rsi_ma'].shift(1))

        # SMA
        dataframe['200_SMA'] = ta.SMA(dataframe["close"], timeperiod = 200)
        dataframe['200_SMAPC'] = PC(dataframe, dataframe['200_SMA'], dataframe['200_SMA'].shift(1) )

        # Plot 0
        dataframe['zero'] = 0
 
        # Williams R%
        dataframe['willr14'] = pta.willr(dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['willr14PC'] = PC(dataframe, dataframe['willr14'], dataframe['willr14'].shift(1) )

        for l in self.max_length.range:
            dataframe['min'] = dataframe['open'].rolling(l).min()
            dataframe['max'] = dataframe['close'].rolling(l).max()

        # distance from 200SMA max
        dataframe['from_200'] = ta.SMA(
            ((((dataframe['close'] + dataframe['open']) / 2) - dataframe['200_SMA']) / dataframe['close']) * 100, timeperiod=self.from15.value)

        ### Buying Weights ###
        dataframe.loc[(dataframe['rsi']<self.rsi_buy.value), 'rsi_buy1'] = 1
        dataframe.loc[(dataframe['rsi']>self.rsi_buy.value), 'rsi_buy1'] = -1

        dataframe.loc[(dataframe['rsi']>dataframe['rsi_ma']), 'rsi_buy2'] = 1
        dataframe.loc[(dataframe['rsi']<dataframe['rsi_ma']), 'rsi_buy2'] = -1
        
        dataframe.loc[(dataframe['rsi_ma_pcnt']>self.rsi_ma_buypc.value), 'rsi_buy3'] = 1
        dataframe.loc[(dataframe['rsi_ma_pcnt']<self.rsi_ma_buypc.value), 'rsi_buy3'] = -1
        
        dataframe.loc[(dataframe['rsi']<self.rsi_buy_safe.value), 'rsi_buy4'] = 1
        dataframe.loc[(dataframe['rsi']>self.rsi_buy_safe.value), 'rsi_buy4'] = 0

        dataframe['rsi_weight'] = (
            (dataframe['rsi_buy1']+dataframe['rsi_buy2']+dataframe['rsi_buy3']+dataframe['rsi_buy4'])/4) * self.x05.value

        dataframe.loc[((dataframe['close'] > dataframe['200_SMA']) & (dataframe['200_SMAPC'] > self.sma200_buy_pc.value)), 'sma_buy1'] = 1
        dataframe.loc[((dataframe['close'] < dataframe['200_SMA']) & (dataframe['200_SMAPC'] > self.sma200_buy_pc.value)), 'sma_buy1'] = 1
        dataframe.loc[((dataframe['close'] > dataframe['200_SMA']) & (dataframe['200_SMAPC'] < self.sma200_buy_pc.value)), 'sma_buy1'] = -1
        dataframe.loc[((dataframe['close'] < dataframe['200_SMA']) & (dataframe['200_SMAPC'] < self.sma200_buy_pc.value)), 'sma_buy1'] = -1

        dataframe.loc[(dataframe['200_SMAPC'] > self.sma200_buy_pc.value), 'sma_buy2'] = 1
        dataframe.loc[(dataframe['200_SMAPC'] < self.sma200_buy_pc.value), 'sma_buy2'] = -1

        dataframe['200SMA_weight'] = ((dataframe['sma_buy1']+dataframe['sma_buy2'])/2) * self.x06.value

        dataframe.loc[(dataframe['willr14'] < self.willr_buy.value), 'willr_buy1'] = 1
        dataframe.loc[(dataframe['willr14'] > self.willr_buy.value), 'willr_buy1'] = -1

        dataframe.loc[(dataframe['willr14'] > -80), 'willr_buy2'] = 1
        dataframe.loc[(dataframe['willr14'] < -80), 'willr_buy2'] = -1
        
        dataframe.loc[(dataframe['willr14PC'] > 0), 'willr_buy3'] = 1
        dataframe.loc[(dataframe['willr14PC'] < 0), 'willr_buy3'] = -1

        dataframe['willr_weight'] = ((dataframe['willr_buy1']+dataframe['willr_buy2']+dataframe['willr_buy3'])/3) * self.x07.value

        dataframe['from_weight'] = -((dataframe['from_200']) * self.x08.value)

        dataframe.loc[(dataframe['wave_t1']<self.fast_wave_buy.value), 'wave_buy1'] = 1
        dataframe.loc[(dataframe['wave_t1']>self.fast_wave_buy.value), 'wave_buy1'] = -1

        dataframe.loc[(dataframe['wave_t1']>dataframe['wave_t1'].shift(1)), 'wave_buy2'] = 1
        dataframe.loc[(dataframe['wave_t1']<dataframe['wave_t1'].shift(1)), 'wave_buy2'] = -1

        dataframe.loc[(dataframe['wave_t1']>dataframe['wave_t2']), 'wave_buy3'] = 1
        dataframe.loc[(dataframe['wave_t1']<dataframe['wave_t2']), 'wave_buy3'] = -1

        dataframe.loc[((dataframe['wave_t1']>dataframe['wave_t1_2h']) & (dataframe['wave_t1_2h']<self.slow_wave_buy.value)), 'wave_buy4'] = 1
        dataframe.loc[((dataframe['wave_t1']<dataframe['wave_t1_2h']) & (dataframe['wave_t1_2h']<self.slow_wave_buy.value)), 'wave_buy4'] = 2

        dataframe['wave_weight'] = ((dataframe['wave_buy1']+dataframe['wave_buy2']+dataframe['wave_buy3']+dataframe['wave_buy4'])/4) * self.x09.value

        dataframe['auto_buy'] = dataframe[['rsi_weight', 'willr_weight', '200SMA_weight', 'from_weight', 'wave_weight']].sum(axis=1)

        ### SELLING ###

        dataframe.loc[(dataframe['rsi']>self.rsi_sell.value), 'rsi_sell1'] = 1
        dataframe.loc[(dataframe['rsi']<self.rsi_sell.value), 'rsi_sell1'] = -1

        dataframe.loc[(dataframe['rsi']>dataframe['rsi_ma']), 'rsi_sell2'] = -1
        dataframe.loc[(dataframe['rsi']<dataframe['rsi_ma']), 'rsi_sell2'] = 1
        
        dataframe.loc[(dataframe['rsi_ma_pcnt']>self.rsi_ma_sellpc.value), 'rsi_sell3'] = -1
        dataframe.loc[(dataframe['rsi_ma_pcnt']<self.rsi_ma_sellpc.value), 'rsi_sell3'] = 1
        
        dataframe.loc[(dataframe['rsi']<self.rsi_sell_safe.value), 'rsi_sell4'] = 0
        dataframe.loc[(dataframe['rsi']>self.rsi_sell_safe.value), 'rsi_sell4'] = 1

        dataframe['rsi_weight_sell'] = (
            (dataframe['rsi_sell1']+dataframe['rsi_sell2']+dataframe['rsi_sell3']+dataframe['rsi_sell4'])/4) * self.y05.value

        dataframe.loc[((dataframe['close'] > dataframe['200_SMA']) & (dataframe['200_SMAPC'] > self.sma200_sell_pc.value)), 'sma_sell1'] = 1
        dataframe.loc[((dataframe['close'] < dataframe['200_SMA'])& (dataframe['200_SMAPC'] > self.sma200_sell_pc.value)), 'sma_sell1'] = 2
        dataframe.loc[((dataframe['close'] > dataframe['200_SMA']) & (dataframe['200_SMAPC'] < self.sma200_sell_pc.value)), 'sma_sell1'] = -2
        dataframe.loc[((dataframe['close'] < dataframe['200_SMA']) & (dataframe['200_SMAPC'] < self.sma200_sell_pc.value)), 'sma_sell1'] = -1

        dataframe.loc[(dataframe['200_SMAPC'] > self.sma200_sell_pc.value), 'sma_sell2'] = 1
        dataframe.loc[(dataframe['200_SMAPC'] < self.sma200_sell_pc.value), 'sma_sell2'] = -1

        dataframe['200SMA_weight_sell'] = ((dataframe['sma_sell1']+dataframe['sma_sell2'])/2) * self.y06.value

        dataframe.loc[(dataframe['willr14'] < self.willr_sell.value), 'willr_sell1'] = -1
        dataframe.loc[(dataframe['willr14'] > self.willr_sell.value), 'willr_sell1'] = 1

        dataframe.loc[(dataframe['willr14'] > -10), 'willr_sell2'] = 1
        dataframe.loc[(dataframe['willr14'] < -10), 'willr_sell2'] = -1
        
        dataframe.loc[(dataframe['willr14PC'] > 0), 'willr_sell3'] = -1
        dataframe.loc[(dataframe['willr14PC'] < 0), 'willr_sell3'] = 1

        dataframe['willr_weight_sell'] = ((dataframe['willr_sell1']+dataframe['willr_sell2']+dataframe['willr_sell3'])/3) * self.y07.value

        dataframe['from_weight_sell'] = ((dataframe['from_200']) * self.y08.value)

        dataframe.loc[(dataframe['wave_t1']<self.slow_wave_sell.value), 'wave_sell1'] = -1
        dataframe.loc[(dataframe['wave_t1']>self.slow_wave_sell.value), 'wave_sell1'] = 1

        dataframe.loc[(dataframe['wave_t1']>dataframe['wave_t1'].shift(1)), 'wave_sell2'] = -1
        dataframe.loc[(dataframe['wave_t1']<dataframe['wave_t1'].shift(1)), 'wave_sell2'] = 1

        dataframe.loc[(dataframe['wave_t1']>dataframe['wave_t2']), 'wave_sell3'] = -1
        dataframe.loc[(dataframe['wave_t1']<dataframe['wave_t2']), 'wave_sell3'] = 1

        dataframe.loc[((dataframe['wave_t1']>dataframe['wave_t1_2h']) & (dataframe['wave_t1_2h']>self.slow_wave_sell.value)), 'wave_sell4'] = 1
        dataframe.loc[((dataframe['wave_t1']<dataframe['wave_t1_2h']) & (dataframe['wave_t1_2h']>self.slow_wave_sell.value)), 'wave_sell4'] = 2

        dataframe['wave_weight'] = ((dataframe['wave_sell1']+dataframe['wave_sell2']+dataframe['wave_sell3']+dataframe['wave_sell4'])/4) * self.y09.value

        dataframe['auto_sell'] = dataframe[['rsi_weight_sell', 'willr_weight_sell', '200SMA_weight_sell', 'from_weight_sell']].sum(axis=1)

        # dataframe['auto_buy_decision'] = ta.SMA((dataframe['auto_buy'] - dataframe['auto_sell']), timeperiod=self.buy_smoothing.value)
        # dataframe['auto_sell_decision'] = ta.SMA((dataframe['auto_sell'] - dataframe['auto_buy']), timeperiod=self.sell_smoothing.value)

        dataframe['auto_buy_decision'] = (dataframe['auto_buy'] - dataframe['auto_sell'])
        dataframe['auto_sell_decision'] = (dataframe['auto_sell'] - dataframe['auto_buy'])

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:



        dataframe.loc[
            (
                (dataframe['auto_buy_decision'] >= self.auto_buy.value) &
                (dataframe['auto_buy_decision_2h'] >= self.auto_buy_2.value) &
                (dataframe['auto_buy_decision'] >= dataframe['auto_buy_decision'].shift(1)) &
                (dataframe['200_SMA_2h'] > dataframe['200_SMA_2h'].shift(8)) &
                (dataframe['200_SMA'] > dataframe['200_SMA'].shift(1)) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'auto buy bullzzz up')

        dataframe.loc[
            (
                (dataframe['auto_buy_decision'] >= (self.auto_buy.value + self.auto_buy_down.value)) &
                (dataframe['auto_buy_decision_2h'] >= (self.auto_buy_2.value + self.auto_buy_down_2.value)) &
                (dataframe['auto_buy_decision'] >= dataframe['auto_buy_decision'].shift(1)) &
                (dataframe['200_SMA_2h'] > dataframe['200_SMA_2h'].shift(8)) &
                (dataframe['200_SMA'] < dataframe['200_SMA'].shift(1)) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'auto buy bullzzz down')

        dataframe.loc[
            (
                (dataframe['auto_buy_decision'] >= (self.auto_buy.value + self.auto_buy_bearzzz.value)) &
                (dataframe['auto_buy_decision_2h'] >= (self.auto_buy_2.value + self.auto_buy_bearzzz_2.value)) &
                (dataframe['auto_buy_decision'] >= dataframe['auto_buy_decision'].shift(1)) &
                (dataframe['200_SMA_2h'] < dataframe['200_SMA_2h'].shift(8)) &
                (dataframe['200_SMA'] > dataframe['200_SMA'].shift(1)) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'auto buy bearzzz up')

        dataframe.loc[
            (
                (dataframe['auto_buy_decision'] >= (self.auto_buy.value + self.auto_buy_bearzzz.value + self.auto_buy_bearzzz_down.value)) &
                (dataframe['auto_buy_decision_2h'] >= (self.auto_buy_2.value + self.auto_buy_bearzzz_down_2.value)) &
                (dataframe['auto_buy_decision'] >= dataframe['auto_buy_decision'].shift(1)) &
                (dataframe['200_SMA_2h'] < dataframe['200_SMA_2h'].shift(8)) &
                (dataframe['200_SMA'] < dataframe['200_SMA'].shift(1)) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'auto buy bearzzz down')

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
            
                (dataframe['auto_sell_decision'] >= (self.auto_sell_bull.value + self.auto_sell_bear.value)) &
                (dataframe['200_SMA_2h'] > dataframe['200_SMA_2h'].shift(8)) &
                (dataframe['volume'] > 0)

            ),
            ['exit_long', 'exit_tag']] = (1, 'auto_sell_bull')


        dataframe.loc[
            (
            
                (dataframe['auto_sell_decision'] >= (self.auto_sell_bear.value)) &
                (dataframe['200_SMA_2h'] < dataframe['200_SMA_2h'].shift(8)) &
                (dataframe['volume'] > 0)

            ),
            ['exit_long', 'exit_tag']] = (1, 'auto_sell_bear')

        return dataframe

