
import logging
import numpy as np
import pandas as pd
from technical import qtpylib, pivots_points
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

logger = logging.getLogger(__name__)


class ultratank(IStrategy):


    ### Strategy parameters ###
    exit_profit_only = False### No selling at a loss
    use_custom_stoploss = True
    trailing_stop = True
    position_adjustment_enable = True
    ignore_roi_if_entry_signal = True
    use_exit_signal = True
    stoploss = -0.25
    startup_candle_count: int = 30
    timeframe = '1h'
    # DCA Parameters
    position_adjustment_enable = True
    max_entry_position_adjustment = 0
    max_dca_multiplier = 1
    minimal_roi = {

        "12000": 0.01,
        "2400": 0.10,
        "300": 0.15,
        "180": 0.30,
        "120":0.40,
        "60": 0.45,
        "0": 0.50
    }


    ### Hyperoptable parameters ###
    # entry optizimation
    # max_epa = CategoricalParameter([0, 1, 2, 3], default=1, space="buy", optimize=True)

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

    # indicators
    buy_offset1 = DecimalParameter(low=0.95, high=0.99, decimals=3, default=0.985, space='buy', optimize=True, load=True)
    buy_offset2 = DecimalParameter(low=0.90, high=0.95, decimals=3, default=0.950, space='buy', optimize=True, load=True)
    sell_offset1 = DecimalParameter(low=1.01, high=1.05, decimals=3, default=1.035, space='sell', optimize=True, load=True)
    max_length = CategoricalParameter([24, 48, 72, 96, 144, 192, 240], default=48, space="buy", optimize=False)
    filterlength = IntParameter(low=30, high=40, default=35, space='sell', optimize=True)
    smoothing_length = IntParameter(15, 50, default=30, space="buy", optimize=True)
    atr_buy = DecimalParameter(0.2, 4, default=1, decimals=1, space='buy', optimize=True)
    atr_sell = DecimalParameter(0.2, 4, default=1, decimals=1, space='sell', optimize=True)


    # trading
    buy_rsi = IntParameter(low=10, high=30, default=25, space='buy', optimize=True, load=True)
    buy_rsi_bear = IntParameter(low=40, high=55, default=45, space='buy', optimize=True, load=True)
    ema_buy = IntParameter(low=30, high=60, default=40, space='buy', optimize=True, load=True)
    buy_rsi_bull = IntParameter(low=55, high=75, default=65, space='buy', optimize=True, load=True)
    buy_wt_bear = IntParameter(low=-30, high=0, default=0, space='buy', optimize=True, load=True)
    buy_wt_bull = IntParameter(low=-10, high=30, default=15, space='buy', optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=80, default=55, space='sell', optimize=True, load=True)
    ema_sell = IntParameter(low=30, high=60, default=40, space='sell', optimize=True, load=True)
    ### selling values ###
    sell_slope = DecimalParameter(-0.05, 0.10, default=0.04, decimals=2, space="sell", optimize=True)

    # pick what works
    buy01 = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    buy02 = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    buy03 = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    buy04 = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    buy05 = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    buy06 = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    buy07 = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    buy08 = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    buy09 = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    buy10 = CategoricalParameter([True, False], default=True, space="buy", optimize=True)

    sell01 = CategoricalParameter([True, False], default=True, space="sell")
    sell02 = CategoricalParameter([True, False], default=True, space="sell")
    sell03 = CategoricalParameter([True, False], default=True, space="sell")
    sell04 = CategoricalParameter([True, False], default=True, space="sell")
    sell05 = CategoricalParameter([True, False], default=True, space="sell")
    sell06 = CategoricalParameter([True, False], default=True, space="sell")
    sell07 = CategoricalParameter([True, False], default=True, space="sell")
    sell08 = CategoricalParameter([True, False], default=True, space="sell")
    sell09 = CategoricalParameter([True, False], default=True, space="sell")
    sell10 = CategoricalParameter([True, False], default=True, space="sell")


    # dca level optimization
    dca1 = DecimalParameter(low=0.01, high=0.08, decimals=2, default=0.05, space='buy', optimize=True, load=True)
    dca2 = DecimalParameter(low=0.08, high=0.15, decimals=2, default=0.10, space='buy', optimize=True, load=True)
    dca3 = DecimalParameter(low=0.15, high=0.25, decimals=2, default=0.15, space='buy', optimize=True, load=True)

    #trailing stop loss optimiziation
    tsl_target5 = DecimalParameter(low=0.2, high=0.4, decimals=1, default=0.3, space='sell', optimize=True, load=True)
    ts5 = DecimalParameter(low=0.04, high=0.06, default=0.05, space='sell', optimize=True, load=True)
    tsl_target4 = DecimalParameter(low=0.18, high=0.3, default=0.2, space='sell', optimize=True, load=True)
    ts4 = DecimalParameter(low=0.03, high=0.05, default=0.045, space='sell', optimize=True, load=True)
    tsl_target3 = DecimalParameter(low=0.10, high=0.15, default=0.15, space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.07, high=0.12, default=0.1, space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.04, high=0.06, default=0.06, space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.02, high=0.05, default=0.03, space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.008, high=0.015, default=0.013, space='sell', optimize=True, load=True)



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


    ### Dollar Cost Averaging ###
    # This is called when placing the initial order (opening trade)
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
    #     # if self.max_epa.value == 0:
    #     #     self.max_dca_multiplier = 1
    #     # elif self.max_epa.value == 1:
    #     #     self.max_dca_multiplier = 2
    #     # elif self.max_epa.value == 2:
    #     #     self.max_dca_multiplier = 3
    #     # else:
    #     #     self.max_dca_multiplier = 4

        # We need to leave most of the funds for possible further DCA orders
        # This also applies to fixed stakes
        return proposed_stake / self.max_dca_multiplier

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be
        increased or decreased.
        This means extra buy or sell orders with additional fees.
        Only called when `position_adjustment_enable` is set to True.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns None

        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Current buy rate.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param min_stake: Minimal stake size allowed by exchange (for both entries and exits)
        :param max_stake: Maximum stake allowed (either through balance, or by exchange limits).
        :param current_entry_rate: Current rate using entry pricing.
        :param current_exit_rate: Current rate using exit pricing.
        :param current_entry_profit: Current profit using entry pricing.
        :param current_exit_profit: Current profit using exit pricing.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: Stake amount to adjust your trade,
                       Positive values to increase position, Negative values to decrease position.
                       Return None for no action.
        """

        if current_profit > 0.10 and trade.nr_of_successful_exits == 0:
            # Take half of the profit at +5%
            return -(trade.stake_amount / 2)

        if current_profit > -(self.dca1.value) and trade.nr_of_successful_entries == 1:
            return None

        if current_profit > -(self.dca2.value) and trade.nr_of_successful_entries == 2:
            return None

        if current_profit > -(self.dca3.value) and trade.nr_of_successful_entries == 3:
            return None


        # Obtain pair dataframe (just to show how to access it)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries

        # Allow up to 3 additional increasingly larger buys (4 in total)
        # Initial buy is 1x
        # If that falls to -2.5% profit, we buy more, 
        # If that falls down to -5% again, we buy 1.5x more
        # If that falls once again down to -5%, we buy  more
        # Total stake for this trade would be 1 + 1.5 + 2 + 2.5 = 7x of the initial allowed stake.
        # That is why max_dca_multiplier is 7
        # Hope you have a deep wallet!

        try:
            # This returns first order stake size
            stake_amount = filled_entries[0].cost
            # This then calculates current safety order size
            if count_of_entries == 1: 
                stake_amount = stake_amount * 1
            elif count_of_entries == 2:
                stake_amount = stake_amount * 1
            elif count_of_entries == 3:
                stake_amount = stake_amount * 1
            else:
                stake_amount = stake_amount

            return stake_amount
        except Exception as exception:
            return None

        return None


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


    ### NORMAL INDICATORS ###
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Pivot Points
        pivots = pivots_points.pivots_points(dataframe, timeperiod=32, levels=5) #50
        dataframe['pivot'] = pivots['pivot']
        dataframe['s5'] = pivots['s5']
        dataframe['r5'] = pivots['r5']
        dataframe['s3'] = pivots['s3']
        dataframe['r3'] = pivots['r3']
        dataframe['s2'] = pivots['s2']
        dataframe['r2'] = pivots['r2']      
        dataframe['r3-dif'] = (dataframe['r3'] - dataframe['r2']) / 4 
        dataframe['r2.25'] = dataframe['r2'] + dataframe['r3-dif'] 
        dataframe['r2.50'] = dataframe['r2'] + (dataframe['r3-dif'] * 2) 
        dataframe['r2.75'] = dataframe['r2'] + (dataframe['r3-dif'] * 3)

        # Filter ZEMA for selling
        for length in self.filterlength.range:
            dataframe[f'ema_1{length}'] = ta.EMA(dataframe['close'], timeperiod=length)
            dataframe[f'ema_2{length}'] = ta.EMA(dataframe[f'ema_1{length}'], timeperiod=length)
            dataframe[f'ema_dif{length}'] = dataframe[f'ema_1{length}'] - dataframe[f'ema_2{length}']
            dataframe[f'zema_{length}'] = dataframe[f'ema_1{length}'] + dataframe[f'ema_dif{length}']

        for x in self.ema_buy.range:
            dataframe[f'ema_buy{x}'] = ta.EMA(dataframe['close'], timeperiod=x)    

        for y in self.ema_sell.range:
            dataframe[f'ema_sell{y}'] = ta.EMA(dataframe['close'], timeperiod=y)    

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['rsi_ma'] = ta.SMA(dataframe['rsi'], timeperiod=10)

        # WaveTrend using OHLC4 or HA close - 3/21
        ap = (0.25 * (dataframe['high'] + dataframe['low'] + dataframe["close"] + dataframe["open"]))
        
        dataframe['esa'] = ta.EMA(ap, timeperiod = 3)
        dataframe['d'] = ta.EMA(abs(ap - dataframe['esa']), timeperiod = 3)
        dataframe['wave_ci'] = (ap-dataframe['esa']) / (0.015 * dataframe['d'])
        dataframe['wave_t1'] = ta.EMA(dataframe['wave_ci'], timeperiod = 21)  
        dataframe['wave_t2'] = ta.SMA(dataframe['wave_t1'], timeperiod = 3)

        # SMA
        dataframe['200_SMA'] = ta.SMA(dataframe["close"], timeperiod = 200)
        dataframe['30_SMA'] = ta.SMA(dataframe["close"], timeperiod = 30)

        # TTM Squeeze
        ttm_Squeeze = pta.squeeze(high = dataframe['high'], low = dataframe['low'], close = dataframe["close"], lazybear = True)
        dataframe['ttm_Squeeze'] = ttm_Squeeze['SQZ_20_2.0_20_1.5_LB']
        dataframe['ttm_ema'] = ta.EMA(dataframe['ttm_Squeeze'], timeperiod = 4)
        dataframe['squeeze_ON'] = ttm_Squeeze['SQZ_ON']
        dataframe['squeeze_OFF'] = ttm_Squeeze['SQZ_OFF']
        dataframe['NO_squeeze'] = ttm_Squeeze['SQZ_NO']

        #ATR active MA Offset adjustments
        dataframe['atr_pcnt'] = (qtpylib.atr(dataframe) / (dataframe['close'] + dataframe['open'] / 2))
        dataframe['atr_buy'] = dataframe['atr_pcnt'] / self.atr_buy.value
        dataframe['atr_sell'] = dataframe['atr_pcnt'] / self.atr_sell.value

        dataframe['buy_offset1'] = dataframe[f'ema_buy{self.ema_buy.value}'] * (self.buy_offset1.value - dataframe['atr_buy'])
        dataframe['buy_offset2'] = dataframe[f'ema_buy{self.ema_buy.value}'] * (self.buy_offset2.value - dataframe['atr_buy'])
        dataframe['sell_offset1'] = dataframe[f'ema_sell{self.ema_sell.value}'] * (self.sell_offset1.value + dataframe['atr_sell'])

        ### I.ntelligent B.uying S.ystem ###

        # distance from reference ma to current close
        dataframe['change'] = ((dataframe['close'] - dataframe['200_SMA']) / dataframe['close']) * 100

        # smoothing the change and offsets
        for valsma in self.smoothing_length.range:
            dataframe[f'smooth_change_{valsma}'] = ta.SMA(dataframe['change'], timeperiod=valsma)

        dataframe['smooth_ma_slope'] = pta.momentum.slope(dataframe[f'smooth_change_{valsma}'])
        dataframe['smooth_slope_sma'] = ta.SMA(dataframe['smooth_ma_slope'], timeperiod=5)


        # 300 Candle Rolling Min-Max
        for l in self.max_length.range:
            dataframe['min'] = dataframe['open'].rolling(l).min()
            dataframe['max'] = dataframe['close'].rolling(l).max()

        # distance from the rolling max in percent
        dataframe['from_max'] = ((dataframe['close'] - dataframe['max']) / dataframe['close']) * 100
        # distance from the rolling min in percent
        dataframe['from_min'] = ((dataframe['open'] - dataframe['min']) / dataframe['open']) * 100
 
        return dataframe

    ### ENTRY CONDITIONS ###
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                
                (df['squeeze_ON'] == 1) &
                (self.buy01.value == True) &
                (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                (df['200_SMA'] < df['200_SMA'].shift(1)) &
                (df['rsi'] < self.buy_rsi_bear.value) &
                (df['wave_t1'] < self.buy_wt_bear.value)&
                (df['wave_t1'] > df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
                (qtpylib.crossed_above(df['wave_t1'], df['wave_t2'])) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'ttm_Squeeze/WT - bear')

        df.loc[
            (
                (df['squeeze_ON'] == 1) &
                (self.buy02.value == True) &
                (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                (df['rsi'] < self.buy_rsi_bear.value) &
                (df['200_SMA'] < df['200_SMA'].shift(1)) &
                (df['wave_t1'] < self.buy_wt_bear.value)&
                (df['wave_t1'] > df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
                (df['wave_t1'].shift(2) > df['wave_t1'].shift(1)) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'ttm_Squeeze/WTT - bear')

        df.loc[
            (
                (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                (self.buy03.value == True) &
                (df['rsi'] < self.buy_rsi_bear.value) &
                (df['200_SMA'] < df['200_SMA'].shift(1)) &
                (df['wave_t1'] < self.buy_wt_bear.value)&
                (df['wave_t1'] > df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
                (df['wave_t1'].shift(2) > df['wave_t1'].shift(1)) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'WT transition - bear')

        df.loc[
            (
                (df['squeeze_ON'] == 1) &
                (self.buy04.value == True) &
                (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                (df['rsi'] < self.buy_rsi_bull.value) &
                (df['30_SMA'] > df['close']) &
                (df['200_SMA'] > df['200_SMA'].shift(1)) &
                (df['wave_t1'] > df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
                (qtpylib.crossed_above(df['wave_t1'], df['wave_t2'])) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'ttm_Squeeze/WT - bull')

        df.loc[
            (
                (df['squeeze_ON'] == 1) &
                (self.buy05.value == True) &
                (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                (df['rsi'] < self.buy_rsi_bull.value) &
                (df['30_SMA'] > df['close']) &
                (df['200_SMA'] > df['200_SMA'].shift(1)) &
                (df['wave_t1'] > df['wave_t1'].shift(1)) & 
                (df['wave_t1'] < self.buy_wt_bull.value) & 
                (df['wave_t1'].shift(2) > df['wave_t1'].shift(1)) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'ttm_Squeeze/WTT - bull')

        df.loc[
            (
                (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                (self.buy06.value == True) &
                (df['rsi'] < self.buy_rsi_bull.value) &
                (df['30_SMA'] > df['close']) &
                (df['200_SMA'] > df['200_SMA'].shift(1)) &
                (df['wave_t1'] > df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
                (df['wave_t1'] < self.buy_wt_bull.value) & 
                (df['wave_t1'].shift(2) > df['wave_t1'].shift(1)) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'WT transition - bull')


        df.loc[
            (
                # (df['rsi'] >  self.buy_rsi.value) &
                (self.buy07.value == True) &
                # (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                (df['close'] < df['buy_offset1']) &
                (df['rsi'] < self.buy_rsi_bear.value) &
                # (df['200_SMA'] < df['200_SMA'].shift(1)) &
                (qtpylib.crossed_above(df['rsi'], df['rsi_ma'])) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'RSI-XO < Buy Offset1')

        df.loc[
            (
                # Signal: RSI crosses above 30
                (self.buy08.value == True) &
                (df['rsi'] >  self.buy_rsi.value) &
                (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                (df['rsi'] < self.buy_rsi_bull.value) &
                (df['30_SMA'] > df['close']) &
                (df['30_SMA'] > df['200_SMA']) &
                (df['200_SMA'] > df['200_SMA'].shift(1)) &
                (qtpylib.crossed_above(df['rsi'], df['rsi_ma'])) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'RSI-XO bull')

        return df


    ### EXIT CONDITIONS ###
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r5'])) &
                (df['close'] > (df['200_SMA'] * self.sell_offset1.value)) &
                (self.sell01.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R5 - XO')
      
        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r3'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (df['rsi'] > self.sell_rsi.value) &
                (self.sell02.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ), 
            ['exit_long', 'exit_tag']] = (1, 'R3 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r2.75'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (df['rsi'] > self.sell_rsi.value) &
                (self.sell03.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R2.75 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r2.50'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (df['rsi'] > self.sell_rsi.value) &
                (self.sell04.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R2.5 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r2.25'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (df['rsi'] > self.sell_rsi.value) &
                (self.sell05.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R2.25 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r2'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (df['rsi'] > self.sell_rsi.value) &
                (self.sell06.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R2 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['pivot'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (df['rsi'] > self.sell_rsi.value) &
                (self.sell07.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'Pivot - XO')

        return df   

# Best result:

#     40/250:   1027 trades. 699/290/38 Wins/Draws/Losses. Avg profit   0.91%. Median profit   0.25%. Total profit 6051.24032389 USDT (  60.51%). Avg duration 11:33:00 min. Objective: -6051.24032


#     # Buy hyperspace params:
#     buy_params = {
#         "buy_rsi": 21,
#         "buy_rsi_bear": 48,
#         "buy_rsi_bull": 75,
#         "buy_wt_bear": 46,
#         "buy_wt_bull": 70,
#         "dca1": 0.01,
#         "dca2": 0.11,
#         "dca3": 0.23,
#         "max_epa": 2,
#         "wavelength": 10,
#     }

#     # Sell hyperspace params:
#     sell_params = {
#         "sell_rsi": 71,
#         "ts0": 0.012,
#         "ts1": 0.011,
#         "ts2": 0.022,
#         "ts3": 0.03,
#         "ts4": 0.038,
#         "ts5": 0.045,
#         "tsl_target0": 0.029,
#         "tsl_target1": 0.058,
#         "tsl_target2": 0.091,
#         "tsl_target3": 0.143,
#         "tsl_target4": 0.183,
#         "tsl_target5": 0.2,
#     }

#     # Protection hyperspace params:
#     protection_params = {
#         "cooldown_lookback": 38,  # value loaded from strategy
#         "stop_duration": 47,  # value loaded from strategy
#         "use_stop_protection": False,  # value loaded from strategy
#     }

#     # ROI table:
#     minimal_roi = {
#         "0": 0.364,
#         "45": 0.145,
#         "160": 0.048,
#         "520": 0
#     }

#     # Stoploss:
#     stoploss = -0.286

#     # Trailing stop:
#     trailing_stop = True  # value loaded from strategy
#     trailing_stop_positive = 0.269  # value loaded from strategy
#     trailing_stop_positive_offset = 0.317  # value loaded from strategy
#     trailing_only_offset_is_reached = True  # value loaded from strategy
    

#     # Max Open Trades:
#     max_open_trades = 12  # value loaded from strategy
# Best result:

#     53/250:    886 trades. 608/262/16 Wins/Draws/Losses. Avg profit   0.90%. Median profit   0.23%. Total profit 5306.73713390 USDT (  53.07%). Avg duration 12:10:00 min. Objective: -5306.73713


#     # Buy hyperspace params:
#     buy_params = {
#         "buy_rsi": 22,
#         "buy_rsi_bear": 40,
#         "buy_rsi_bull": 59,
#         "buy_wt_bear": 48,
#         "buy_wt_bull": 56,
#         "dca1": 0.01,
#         "dca2": 0.14,
#         "dca3": 0.21,
#         "max_epa": 1,
#         "wavelength": 9,
#     }

#     # Sell hyperspace params:
#     sell_params = {
#         "sell_rsi": 68,
#         "ts0": 0.009,
#         "ts1": 0.013,
#         "ts2": 0.025,
#         "ts3": 0.028,
#         "ts4": 0.04,
#         "ts5": 0.047,
#         "tsl_target0": 0.047,
#         "tsl_target1": 0.053,
#         "tsl_target2": 0.099,
#         "tsl_target3": 0.135,
#         "tsl_target4": 0.25,
#         "tsl_target5": 0.4,
#     }

#     # Protection hyperspace params:
#     protection_params = {
#         "cooldown_lookback": 26,
#         "stop_duration": 83,
#         "use_stop_protection": False,
#     }

#     # ROI table:
#     minimal_roi = {
#         "0": 0.154,
#         "120": 0.09,
#         "300": 0.044,
#         "554": 0
#     }

#     # Stoploss:
#     stoploss = -0.205

#     # Trailing stop:
#     trailing_stop = True
#     trailing_stop_positive = 0.342
#     trailing_stop_positive_offset = 0.437
#     trailing_only_offset_is_reached = True
    

#     # Max Open Trades:
#     max_open_trades = 12  # value loaded from strategy
# (.env) core@core:~/freqtrade$ freqtrade hyperopt --hyperopt-loss OnlyProfitHyperOptLoss --strategy thetank2 --config ho.json -e 250 --timerange 20230101-20230215 --spaces buy sell roi stoploss trailing protection
