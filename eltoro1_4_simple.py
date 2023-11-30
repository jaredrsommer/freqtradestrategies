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
# C.T. 3-9-23
# adding bull/bear detect of 1hr fast ewo
### Change log ###

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif


def PC(dataframe, in1, in2):
    df = dataframe.copy()
    pc = ((in2-in1)/in1) * 100
    return pc

class eltoro1_4_simple(IStrategy):

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

    # fast ewo
    fastest_ewo = 5
    faster_ewo = 35

    # slow ewo
    fast_ewo = 35
    slow_ewo = 200

    ### Hyperoptable parameters ###

    # protections
    cooldown_lookback = IntParameter(24, 48, default=46, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # SMAOffset
    base_nb_candles_buy = IntParameter(25, 60, default=25, space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(30, 60, default=49, space='sell', optimize=True)
    low_offset = DecimalParameter(0.97, 0.99, default=0.97, decimals=2, space='buy', optimize=True)
    high_offset = DecimalParameter(1.01, 1.05, default=1.01,  decimals=2, space='sell', optimize=True)
    high_offset_2 = DecimalParameter(1.05, 1.1, default=1.05, decimals=2, space='sell', optimize=True)   
    max_length = CategoricalParameter([24, 48, 72, 96, 144, 192, 240], default=48, space="buy", optimize=False)
    decision_length = IntParameter(30, 60, default=25, space='buy', optimize=True)

    # Buy Parameters
    rsi_buy = IntParameter(55, 70, default=65, space='buy', optimize=True)
    rsi_buy_safe = IntParameter(40, 55, default=50, space='buy', optimize=True)
    rsi_ma_buypc = IntParameter(-5, 5, default=0, space='buy', optimize=True)
    sma200_buy_pc = IntParameter(-5, 5, default=0, space='buy', optimize=True)
    willr_buy = IntParameter(-50, -20, default=-50, space='buy', optimize=True)
    hma_buy_pc = IntParameter(-5, 5, default=0, space='buy', optimize=True)
    macdl_buy_range = DecimalParameter(0.01, 0.03, default=0.01, decimals=2, space='buy', optimize=True)
    macdl_buy_pc = IntParameter(-5, 5, default=0, space='buy', optimize=True)
    auto_buy = IntParameter(5, 15, default=10, space='buy', optimize=True)
    auto_buy_down = IntParameter(5, 15, default=10, space='buy', optimize=True)
    auto_buy_bearzzz = IntParameter(5, 15, default=5, space='buy', optimize=True)
    auto_buy_bearzzz_down = IntParameter(5, 15, default=5, space='buy', optimize=True)
    auto_buy_EWO = IntParameter(8, 15, default=10, space='buy', optimize=True)


    # Buy Parameters
    rsi_sell = IntParameter(60, 70, default=64, space='sell', optimize=True)
    rsi_sell_safe = IntParameter(60, 80, default=70, space='sell', optimize=True)
    rsi_ma_sellpc = IntParameter(-5, 5, default=0, space='sell', optimize=True)
    sma200_sell_pc = IntParameter(-5, 5, default=0, space='sell', optimize=True)
    willr_sell = IntParameter(-50, -20, default=-20, space='sell', optimize=True)
    hma_sell_pc = IntParameter(-5, 5, default=0, space='sell', optimize=True)
    macdl_sell_range = DecimalParameter(0.01, 0.04, default=0.01, decimals=2, space='sell', optimize=True)
    macdl_sell_pc = IntParameter(-5, 5, default=0, space='sell', optimize=True)
    auto_sell_bull = IntParameter(3, 15, default=4, space='sell', optimize=True)
    auto_sell_bear = IntParameter(3, 15, default=4, space='sell', optimize=True)

    ### BTC and Pair EWO values
    bull = DecimalParameter(-0.25, 0.25, default=0, space='buy',decimals=2, optimize=True)
    estop = DecimalParameter(-0.5, 0, default=-0.5, space='sell',decimals=2, optimize=True)

    ###  Buy Weight Mulitpliers ###
    x1 = DecimalParameter(0.3, 5.0, default=1, decimals=1, space='buy', optimize=True)
    x2 = DecimalParameter(0.3, 5.0, default=1, decimals=1, space='buy', optimize=True)
    x3 = DecimalParameter(0.3, 5.0, default=1, decimals=1, space='buy', optimize=True)
    x4 = DecimalParameter(0.3, 5.0, default=1, decimals=1, space='buy', optimize=True)
    x5 = DecimalParameter(0.3, 5.0, default=1, decimals=1, space='buy', optimize=True)
    x6 = DecimalParameter(0.3, 5.0, default=1, decimals=1, space='buy', optimize=True)
    x7 = DecimalParameter(0.3, 5.0, default=1, decimals=1, space='buy', optimize=True)
    x8 = DecimalParameter(0.3, 5.0, default=1, decimals=1, space='buy', optimize=True)
    x9 = DecimalParameter(0.3, 5.0, default=1, decimals=1, space='buy', optimize=True)
    x10 = DecimalParameter(0.3, 5.0, default=1, decimals=1, space='buy', optimize=True)

    ###  Sell Weight Mulitpliers ###
    y1 = DecimalParameter(0.3, 5.0, default=1, decimals=1, space='sell', optimize=True)
    y2 = DecimalParameter(0.3, 5.0, default=1, decimals=1, space='sell', optimize=True)
    y3 = DecimalParameter(0.3, 5.0, default=1, decimals=1, space='sell', optimize=True)
    y4 = DecimalParameter(0.3, 5.0, default=1, decimals=1, space='sell', optimize=True)
    y5 = DecimalParameter(0.3, 5.0, default=1, decimals=1, space='sell', optimize=True)
    y6 = DecimalParameter(0.3, 5.0, default=1, decimals=1, space='sell', optimize=True)
    y7 = DecimalParameter(0.3, 5.0, default=1, decimals=1, space='sell', optimize=True)
    y8 = DecimalParameter(0.3, 5.0, default=1, decimals=1, space='sell', optimize=True)
    y9 = DecimalParameter(0.3, 5.0, default=1, decimals=1, space='sell', optimize=True)
    y10 = DecimalParameter(0.3, 5.0, default=1, decimals=1, space='sell', optimize=True)

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
    informative_timeframe = '1h'

    process_only_new_candles = True
    startup_candle_count = 79
    
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
        pairs += ['BTC/USDT']
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
            # Take half of the profit at +5%
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
        # for stop0 in self.tsl_target0.range:
        #     if (current_profit > stop0):
        #         for stop0a in self.ts0.range:
        #             self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl0 {stop0}/{stop0a} activated')
        #             return stop0a 

        return self.stoploss

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.dp:
            inf_tf = '1h'
            pair = metadata['pair']
            # print(pair)
            informative = self.dp.get_pair_dataframe(pair=f"BTC/USDT", timeframe=inf_tf)
            informative_pair = self.dp.get_pair_dataframe(pair=pair, timeframe=inf_tf)
            informative['INFEWO'] = EWO(informative_pair, 5, 35)
        # BTC EWO 5/35
            informative['BTC_EWO_Fast'] = EWO(informative, 5, 35)
            informative['BTC_EWO_ PC'] = PC(informative, informative['BTC_EWO_Fast'], informative['BTC_EWO_Fast'].shift(1))

       ### Changed this part ###
        # if np.where(informative['BTC_EWO_Fast'] > self.bull.value and informative['BTC_EWO_Fast'].shift(1) < self.bull.value, 1, 0) == 1:
        #     self.dp.send_msg(f"MARKET STATUS: Bear is gone! Lets F00kInG GOOOOO!!!", always_send=True)
        #     print("MARKET STATUS: Bear is gone! Lets F00kInG GOOOOO!!!")

        # elif np.where(informative['BTC_EWO_Fast'] < self.bull.value and informative['BTC_EWO_Fast'].shift(1) > self.bull.value, 1, 0) == 1:
        #     self.dp.send_msg(f"MARKET STATUS: Bear Lurking! Grab the Lube, This could hurt...", always_send=True)
        #     print("MARKET STATUS: Bear Lurking! Grab the Lube, This could hurt...")

        # elif np.where(informative['BTC_EWO_Fast'] < self.estop.value and informative['BTC_EWO_Fast'].shift(1) > self.estop.value, 1, 0) == 1:
        #     self.dp.send_msg(f"MARKET STATUS: ABANDON SHIP!!!", always_send=True)
        #     print("MARKET STATUS: ABANDON SHIP!!!")
        
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        ### 5m indicators ###

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_ma'] = ta.SMA(dataframe['rsi'], timeperiod=10)
        dataframe['rsi_ma_pcnt'] = PC(dataframe, dataframe['rsi_ma'], dataframe['rsi_ma'].shift(1))

        # HMA
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['hma_50_pc'] = PC(dataframe, dataframe['hma_50'], dataframe['hma_50'].shift(1))

        # SMA
        dataframe['200_SMA'] = ta.SMA(dataframe["close"], timeperiod = 200)
        dataframe['200_SMAPC'] = PC(dataframe, dataframe['200_SMA'], dataframe['200_SMA'].shift(1) )

        # Plot 0
        dataframe['zero'] = 0
 
        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Lazy Bear's Macd Lead
        dataframe['sema'] = ta.EMA(dataframe['close'], timeperiod=8)
        dataframe['lema'] = ta.EMA(dataframe['close'], timeperiod=18)
        dataframe['i1'] = dataframe['sema'] + ta.EMA(dataframe['close'] - dataframe['sema'], timeperiod=8)
        dataframe['i2'] = dataframe['lema'] + ta.EMA(dataframe['close']  - dataframe['lema'], timeperiod=18)
        dataframe['macdlead'] = dataframe['i1'] - dataframe['i2']
        dataframe['macdl'] = dataframe['sema'] - dataframe['lema']
        dataframe['macdl_sig'] = ta.SMA(dataframe['macdl'], period=5)
        dataframe["macdlead_pc"] = round((dataframe["macdlead"].shift() - dataframe["macdlead"]) / abs(dataframe["macdlead"].shift()) * -100, 2)
        
        # Williams R%
        dataframe['willr14'] = pta.willr(dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['willr14PC'] = PC(dataframe, dataframe['willr14'], dataframe['willr14'].shift(1) )

        for l in self.max_length.range:
            dataframe['min'] = dataframe['open'].rolling(l).min()
            dataframe['max'] = dataframe['close'].rolling(l).max()

        # distance from the rolling max in percent
        dataframe['from_200'] = ta.SMA(((((dataframe['close'] + dataframe['open']) / 2) - dataframe['200_SMA']) / dataframe['close']) * 100, timeperiod=5)

        ### Buying Weights ###
        dataframe.loc[(dataframe['rsi']<self.rsi_buy.value), 'rsi_buy1'] = 1
        dataframe.loc[(dataframe['rsi']>self.rsi_buy.value), 'rsi_buy1'] = -1

        dataframe.loc[(dataframe['rsi']>dataframe['rsi_ma']), 'rsi_buy2'] = 1
        dataframe.loc[(dataframe['rsi']<dataframe['rsi_ma']), 'rsi_buy2'] = -1
        
        dataframe.loc[(dataframe['rsi_ma_pcnt']>self.rsi_ma_buypc.value), 'rsi_buy3'] = 1
        dataframe.loc[(dataframe['rsi_ma_pcnt']<self.rsi_ma_buypc.value), 'rsi_buy3'] = -1
        
        dataframe.loc[(dataframe['rsi']<self.rsi_buy_safe.value), 'rsi_buy4'] = 2
        dataframe.loc[(dataframe['rsi']>self.rsi_buy_safe.value), 'rsi_buy4'] = -2

        dataframe['rsi_weight'] = (
            (dataframe['rsi_buy1']+dataframe['rsi_buy2']+dataframe['rsi_buy3']+dataframe['rsi_buy4'])/4) * self.x1.value

        dataframe.loc[((dataframe['close'] > dataframe['200_SMA']) & (dataframe['200_SMAPC'] > self.sma200_buy_pc.value)), 'sma_buy1'] = 1
        dataframe.loc[((dataframe['close'] < dataframe['200_SMA'])& (dataframe['200_SMAPC'] > self.sma200_buy_pc.value)), 'sma_buy1'] = 2
        dataframe.loc[((dataframe['close'] > dataframe['200_SMA']) & (dataframe['200_SMAPC'] < self.sma200_buy_pc.value)), 'sma_buy1'] = -1
        dataframe.loc[((dataframe['close'] < dataframe['200_SMA']) & (dataframe['200_SMAPC'] < self.sma200_buy_pc.value)), 'sma_buy1'] = -1

        dataframe.loc[(dataframe['200_SMAPC'] > self.sma200_buy_pc.value), 'sma_buy2'] = 1
        dataframe.loc[(dataframe['200_SMAPC'] < self.sma200_buy_pc.value), 'sma_buy2'] = -1
        
        dataframe.loc[(dataframe['hma_50'] > dataframe['200_SMA']) & (dataframe['hma_50'].shift(1) < dataframe['200_SMA'].shift(1)), 'sma_buy3'] = 2
        dataframe.loc[(dataframe['hma_50'] > dataframe['200_SMA']) & (dataframe['hma_50'] > self.hma_buy_pc.value) , 'sma_buy3'] = 1

        dataframe['200SMA_weight'] = ((dataframe['sma_buy1']+dataframe['sma_buy2']+dataframe['sma_buy3'])/3) * self.x4.value

        dataframe.loc[(dataframe['willr14'] < self.willr_buy.value), 'willr_buy1'] = 1
        dataframe.loc[(dataframe['willr14'] > self.willr_buy.value), 'willr_buy1'] = -1

        dataframe.loc[(dataframe['willr14'] > -80), 'willr_buy2'] = 1
        dataframe.loc[(dataframe['willr14'] < -80), 'willr_buy2'] = -1
        
        dataframe.loc[(dataframe['willr14PC'] > 0), 'willr_buy3'] = 1
        dataframe.loc[(dataframe['willr14PC'] < 0), 'willr_buy3'] = -1

        dataframe['willr_weight'] = ((dataframe['willr_buy1']+dataframe['willr_buy2']+dataframe['willr_buy3'])/3) * self.x5.value

        dataframe.loc[(dataframe['close'] > dataframe['hma_50']), 'hma_buy1'] = 2
        dataframe.loc[(dataframe['close'] < dataframe['hma_50']), 'hma_buy1'] = -1

        dataframe.loc[(dataframe['hma_50_pc'] > self.hma_buy_pc.value) & (dataframe['hma_50'] > dataframe['200_SMA']), 'hma_buy2'] = 1
        dataframe.loc[(dataframe['hma_50_pc'] < self.hma_buy_pc.value) & (dataframe['hma_50'] > dataframe['200_SMA']), 'hma_buy2'] = -1

        dataframe['hma_weight'] = ((dataframe['hma_buy1']+dataframe['hma_buy2'])/2) * self.x6.value

        dataframe.loc[(dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)), 'base_ma_buy1'] = 2
        dataframe.loc[(dataframe['close'] > (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)), 'base_ma_buy'] = 0

        dataframe.loc[(dataframe['close'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}']), 'base_ma_buy2'] = 1
        dataframe.loc[(dataframe['close'] > dataframe[f'ma_buy_{self.base_nb_candles_buy.value}']), 'base_ma_buy2'] = -1

        dataframe['base_ma_buy_weight'] = ((dataframe['base_ma_buy1'] + dataframe['base_ma_buy2'])/2) * self.x7.value

        dataframe.loc[(dataframe['macdl'] > dataframe['macdl_sig']), 'macdl_buy1'] = 1
        dataframe.loc[(dataframe['macdl'] < dataframe['macdl_sig']), 'macdl_buy1'] = -1

        dataframe.loc[(dataframe['macdlead'] > -(self.macdl_buy_range.value * dataframe['close'])), 'macdl_buy2'] = 1
        dataframe.loc[(dataframe['macdlead'] < -(self.macdl_buy_range.value * dataframe['close'])), 'macdl_buy2'] = -1

        dataframe.loc[(dataframe['macdlead'] < (self.macdl_buy_range.value * dataframe['close'])), 'macdl_buy3'] = 1
        dataframe.loc[(dataframe['macdlead'] > (self.macdl_buy_range.value * dataframe['close'])), 'macdl_buy3'] = -1

        dataframe.loc[(dataframe['macdlead_pc'] > self.macdl_buy_pc.value), 'macdl_buy4'] = 1
        dataframe.loc[(dataframe['macdlead_pc'] < self.macdl_buy_pc.value), 'macdl_buy4'] = -1

        dataframe['macdl_weight'] = ((dataframe['macdl_buy1']+dataframe['macdl_buy2']+dataframe['macdl_buy3']+dataframe['macdl_buy4'])/4) * self.x8.value

        dataframe['from_weight'] = -(dataframe['from_200'] * self.x10.value)

        dataframe['auto_buy'] = dataframe[['rsi_weight', 'willr_weight', 'hma_weight', 'base_ma_buy_weight', 'macdl_weight','200SMA_weight', 'from_weight']].sum(axis=1)

        ### SELLING ###

        dataframe.loc[(dataframe['rsi']<self.rsi_sell.value), 'rsi_sell1'] = 1
        dataframe.loc[(dataframe['rsi']>self.rsi_sell.value), 'rsi_sell1'] = -1

        dataframe.loc[(dataframe['rsi']>dataframe['rsi_ma']), 'rsi_sell2'] = -1
        dataframe.loc[(dataframe['rsi']<dataframe['rsi_ma']), 'rsi_sell2'] = 1
        
        dataframe.loc[(dataframe['rsi_ma_pcnt']>self.rsi_ma_sellpc.value), 'rsi_sell3'] = -1
        dataframe.loc[(dataframe['rsi_ma_pcnt']<self.rsi_ma_sellpc.value), 'rsi_sell3'] = 1
        
        dataframe.loc[(dataframe['rsi']<self.rsi_sell_safe.value), 'rsi_sell4'] = -1
        dataframe.loc[(dataframe['rsi']>self.rsi_sell_safe.value), 'rsi_sell4'] = 1

        dataframe['rsi_weight_sell'] = (
            (dataframe['rsi_sell1']+dataframe['rsi_sell2']+dataframe['rsi_sell3']+dataframe['rsi_sell4'])/4) * self.y1.value

        dataframe.loc[((dataframe['close'] > dataframe['200_SMA']) & (dataframe['200_SMAPC'] > self.sma200_sell_pc.value)), 'sma_sell1'] = -1
        dataframe.loc[((dataframe['close'] < dataframe['200_SMA'])& (dataframe['200_SMAPC'] > self.sma200_sell_pc.value)), 'sma_sell1'] = -2
        dataframe.loc[((dataframe['close'] > dataframe['200_SMA']) & (dataframe['200_SMAPC'] < self.sma200_sell_pc.value)), 'sma_sell1'] = 2
        dataframe.loc[((dataframe['close'] < dataframe['200_SMA']) & (dataframe['200_SMAPC'] < self.sma200_sell_pc.value)), 'sma_sell1'] = 1

        dataframe.loc[(dataframe['200_SMAPC'] > self.sma200_sell_pc.value), 'sma_sell2'] = -1
        dataframe.loc[(dataframe['200_SMAPC'] < self.sma200_sell_pc.value), 'sma_sell2'] = 1
        
        dataframe.loc[(dataframe['hma_50'] < dataframe['200_SMA']) & (dataframe['hma_50'].shift(1) > dataframe['200_SMA'].shift(1)), 'sma_sell3'] = 1
        dataframe.loc[(dataframe['hma_50'] > dataframe['200_SMA']) & (dataframe['hma_50'] < self.hma_sell_pc.value) , 'sma_sell3'] = 2

        dataframe['200SMA_weight_sell'] = ((dataframe['sma_sell1']+dataframe['sma_sell2']+dataframe['sma_sell3'])/3) * self.y4.value

        dataframe.loc[(dataframe['willr14'] < self.willr_sell.value), 'willr_sell1'] = -1
        dataframe.loc[(dataframe['willr14'] > self.willr_sell.value), 'willr_sell1'] = 1

        dataframe.loc[(dataframe['willr14'] > -10), 'willr_sell2'] = 1
        dataframe.loc[(dataframe['willr14'] < -10), 'willr_sell2'] = -1
        
        dataframe.loc[(dataframe['willr14PC'] > 0), 'willr_sell3'] = -1
        dataframe.loc[(dataframe['willr14PC'] < 0), 'willr_sell3'] = 1

        dataframe['willr_weight_sell'] = ((dataframe['willr_sell1']+dataframe['willr_sell2']+dataframe['willr_sell3'])/3) * self.y5.value

        dataframe.loc[(dataframe['close'] > dataframe['hma_50']), 'hma_sell1'] = -2
        dataframe.loc[(dataframe['close'] < dataframe['hma_50']), 'hma_sell1'] = 2

        dataframe.loc[(dataframe['hma_50_pc'] > self.hma_sell_pc.value), 'hma_sell2'] = -1
        dataframe.loc[(dataframe['hma_50_pc'] < self.hma_sell_pc.value), 'hma_sell2'] = 1

        dataframe['hma_weight_sell'] = ((dataframe['hma_sell1']+dataframe['hma_sell2'])/2) * self.y6.value

        dataframe.loc[(dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)), 'base_ma_sell1'] = -1
        dataframe.loc[(dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)), 'base_ma_sell'] = 1

        dataframe.loc[(dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)), 'base_ma_sell2'] = -1
        dataframe.loc[(dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)), 'base_ma_sell2'] = 2

        dataframe['base_ma_sell_weight'] = ((dataframe['base_ma_sell1'] + dataframe['base_ma_sell2'])/2) * self.y7.value

        dataframe.loc[(dataframe['macdl'] > dataframe['macdl_sig']), 'macdl_sell1'] = -1
        dataframe.loc[(dataframe['macdl'] < dataframe['macdl_sig']), 'macdl_sell1'] = 1

        dataframe.loc[(dataframe['macdlead'] > -(self.macdl_sell_range.value * dataframe['close'])), 'macdl_sell2'] = 1
        dataframe.loc[(dataframe['macdlead'] < -(self.macdl_sell_range.value * dataframe['close'])), 'macdl_sell2'] = -1

        dataframe.loc[(dataframe['macdlead'] < (self.macdl_sell_range.value * dataframe['close'])), 'macdl_sell3'] = 1
        dataframe.loc[(dataframe['macdlead'] > (self.macdl_sell_range.value * dataframe['close'])), 'macdl_sell3'] = -1

        dataframe.loc[(dataframe['macdlead_pc'] > self.macdl_sell_pc.value), 'macdl_sell4'] = -1
        dataframe.loc[(dataframe['macdlead_pc'] < self.macdl_sell_pc.value), 'macdl_sell4'] = 1

        dataframe['macdl_weight_sell'] = ((dataframe['macdl_sell1']+dataframe['macdl_sell2']+dataframe['macdl_sell3']+dataframe['macdl_sell4'])/4) * self.y8.value

        dataframe['from_weight_sell'] = (dataframe['from_200'] * self.y10.value)

        dataframe['auto_sell'] = dataframe[['rsi_weight_sell', 'willr_weight_sell', 'hma_weight_sell', 'base_ma_sell_weight', 'macdl_weight_sell', '200SMA_weight_sell', 'from_weight_sell']].sum(axis=1)

        dataframe['auto_buy_decision'] = ta.SMA((dataframe['auto_buy'] - dataframe['auto_sell']), timeperiod=self.decision_length.value)
        dataframe['auto_sell_decision'] = ta.SMA((dataframe['auto_sell'] - dataframe['auto_buy']), timeperiod=self.decision_length.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe.loc[
            (
                (dataframe['auto_buy_decision'] >= self.auto_buy.value) &
                (dataframe['auto_buy_decision'] < dataframe['auto_buy']) &
                (dataframe['INFEWO_1h'] >= self.bull.value) & 
                (dataframe['INFEWO_1h'] > dataframe['INFEWO_1h'].shift(4)) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'auto buy bullzzz up')

        dataframe.loc[
            (
                (dataframe['auto_buy_decision'] >= (self.auto_buy.value + self.auto_buy_down.value)) &
                (dataframe['auto_buy_decision'] < dataframe['auto_buy']) &
                (dataframe['INFEWO_1h'] >= self.bull.value) &
                (dataframe['INFEWO_1h'] <= dataframe['INFEWO_1h'].shift(4)) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'auto buy bullzzz down')

        dataframe.loc[
            (
                (dataframe['auto_buy_decision'] >= (self.auto_buy.value + self.auto_buy_bearzzz.value)) &
                (dataframe['auto_buy_decision'] < dataframe['auto_buy']) &
                (dataframe['INFEWO_1h'] < self.bull.value) &
                (dataframe['INFEWO_1h'] > dataframe['INFEWO_1h'].shift(4)) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'auto buy bearzzz up')

        dataframe.loc[
            (
                (dataframe['auto_buy_decision'] >= (self.auto_buy.value + self.auto_buy_bearzzz.value + self.auto_buy_bearzzz_down.value)) &
                (dataframe['auto_buy_decision'] < dataframe['auto_buy']) &
                (dataframe['INFEWO_1h'] < self.bull.value) &
                (dataframe['INFEWO_1h'] <= dataframe['INFEWO_1h'].shift(4)) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'auto buy bearzzz down')

        dataframe.loc[
            (
                (dataframe['auto_buy_decision'] > self.auto_buy_EWO.value) &
                (dataframe['auto_buy_decision'] > dataframe['auto_buy_decision'].shift(1)) &
                (dataframe['INFEWO_1h'] > dataframe['INFEWO_1h'].shift(4)) &
                (dataframe['INFEWO_1h'].shift(4) < dataframe['INFEWO_1h'].shift(8)) &
                # (dataframe['INFEWO_1h'].shift(48) > dataframe['INFEWO_1h'].shift(32)) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, '1h EWO Transition')

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
            
                (dataframe['auto_sell_decision'] >= (self.auto_sell_bull.value + self.auto_sell_bear.value)) &
                (dataframe['close'] < dataframe['hma_50']) & 
                (dataframe['INFEWO_1h'] < self.bull.value) &
                (dataframe['INFEWO_1h'] < dataframe['BTC_EWO_Fast_1h']) &
                (dataframe['INFEWO_1h'] <= dataframe['INFEWO_1h'].shift(2)) &
                (dataframe['volume'] > 0)

            ),
            ['exit_long', 'exit_tag']] = (1, 'auto_sell_bull')


        dataframe.loc[
            (
            
                (dataframe['auto_sell_decision'] >= (self.auto_sell_bear.value)) &
                (dataframe['close'] < dataframe['hma_50']) & 
                (dataframe['INFEWO_1h'] > self.bull.value) &
                (dataframe['INFEWO_1h'] < dataframe['BTC_EWO_Fast_1h']) &
                (dataframe['INFEWO_1h'] <= dataframe['INFEWO_1h'].shift(4)) &
                (dataframe['volume'] > 0)

            ),
            ['exit_long', 'exit_tag']] = (1, 'auto_sell_bear')

        return dataframe

