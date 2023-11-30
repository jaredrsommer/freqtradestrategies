
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

class CTIBS(IStrategy):
    stoploss = -0.20
    timeframe ='15m'
    use_custom_stoploss = True
    trailing_stop = True
    ignore_roi_if_entry_signal = True
    use_exit_signal = True
    minimal_roi = {
        "0":  0.10
    }
    # DCA settings
    exit_profit_only = True
    position_adjustment_enable = True
    max_entry_position_adjustment = 0
    max_dca_multiplier = 1

    ## Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'ioc'
    }


    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    ### ------------------ HYPER-OPT PARAMETERS ----------------------- ###

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
    maxdrawdown_allowed_drawdown = DecimalParameter(0.01, 0.10, default=0.0, space="protection", optimize=True)

  
    ### DCA ###
    # max_epa = CategoricalParameter([0, 1, 2, 3], default=0, space="buy", optimize=True)

  
    ### trailing stop loss optimiziation ###
    tsl_target5 = DecimalParameter(low=0.2, high=0.4, decimals=1, default=0.3, space='sell', optimize=True, load=True)
    ts5 = DecimalParameter(low=0.04, high=0.06, default=0.05, decimals=2,space='sell', optimize=True, load=True)
    tsl_target4 = DecimalParameter(low=0.15, high=0.2, default=0.2, decimals=2, space='sell', optimize=True, load=True)
    ts4 = DecimalParameter(low=0.03, high=0.05, default=0.045, decimals=2,  space='sell', optimize=True, load=True)
    tsl_target3 = DecimalParameter(low=0.10, high=0.15, default=0.15, decimals=2,  space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, decimals=3,  space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.08, high=0.10, default=0.1, decimals=3, space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, decimals=3, space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.05, high=0.08, default=0.06, decimals=3, space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, decimals=3, space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.02, high=0.045, default=0.03, decimals=3, space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.008, high=0.015, default=0.013, decimals=3, space='sell', optimize=True, load=True)


    ### indicators ###
    #buy
    reference_ma_length = IntParameter(185, 195, default=200, space="buy" ,optimize=True)
    smoothing_length = IntParameter(15, 30, default=30, space="buy", optimize=True)
    buy_offset1 = DecimalParameter(low=0.97, high=0.99, decimals=2, default=0.98, space='buy', optimize=True, load=True)
    buy_offset2 = DecimalParameter(low=0.92, high=0.97, decimals=2, default=0.96, space='buy', optimize=True, load=True)
    buy_change = DecimalParameter(1.2, 2.0, default=1.5, decimals=1, space="buy", optimize=True)
    sell_change = DecimalParameter(1.2, 2.0, default=1.5, decimals=1, space="sell", optimize=True)
    max_length = CategoricalParameter([24, 48, 72, 96, 144, 192, 240], default=48, space="buy")

    # selling 
    filterlength = IntParameter(low=30, high=40, default=35, space='sell', optimize=True)
    sell_offset1 = DecimalParameter(low=1.01, high=1.05, decimals=2, default=1.03, space='sell', optimize=True, load=True)
    sell_change = DecimalParameter(1.2, 2.0, default=1.5, decimals=1, space="sell", optimize=True)

    ### buying values ###
    buy_change = DecimalParameter(1.2, 2.0, default=1.5, decimals=1, space="buy", optimize=True)

    ### selling values ###
    sell_slope = DecimalParameter(-0.05, 0.10, default=0.04, decimals=2, space="sell", optimize=True)

    ### I.B.S. ###
    # Reference Ma - Long Term Direction
    ref_bull = DecimalParameter(0.075, 0.125, default=0.010, decimals=3, space="buy", optimize=True)
    ref_up = DecimalParameter(0.01, 0.075, default=0.01, decimals=3, space="buy", optimize=True)
    ref_down = DecimalParameter(-0.075, -0.01, default=-0.01, decimals=3, space="buy", optimize=True)
    ref_bear = DecimalParameter(-0.125, -0.075, default=-0.1, decimals=3, space="buy", optimize=True)

    # Smooth Change Ma - Short Term Direction
    smooth_bull = DecimalParameter(0.075, 0.125, default=0.010, decimals=3, space="buy", optimize=True)
    smooth_up = DecimalParameter(0.01, 0.075, default=0.02, decimals=3, space="buy", optimize=True)
    smooth_down = DecimalParameter(-0.075, -0.01, default=-0.02, decimals=3, space="buy", optimize=True)
    smooth_bear = DecimalParameter(-0.125, -0.075, default=-0.1, decimals=3, space="buy", optimize=True)

    # Distance from Long Term High
    from_bull = DecimalParameter(-5, -1, default=-1, decimals=1, space="buy", optimize=True)
    from_up = DecimalParameter(-5, -2, default=-3.5, decimals=1, space="buy", optimize=True)
    from_ranging = DecimalParameter(-6.5, -2, default=-6.5, decimals=1, space="buy", optimize=True)
    from_down = DecimalParameter(-7.5, -5.0, default=-7.5, decimals=1, space="buy", optimize=True)
    from_bear = DecimalParameter(-12.5, -7.5, default=-10.0, decimals=1, space="buy", optimize=True)

    # Selecting what works
    buy01 = CategoricalParameter([True, False], default=True, space="buy")
    buy02 = CategoricalParameter([True, False], default=True, space="buy")
    buy03 = CategoricalParameter([True, False], default=True, space="buy")
    buy04 = CategoricalParameter([True, False], default=True, space="buy")
    buy05 = CategoricalParameter([True, False], default=True, space="buy")
    buy06 = CategoricalParameter([True, False], default=True, space="buy")
    buy07 = CategoricalParameter([True, False], default=True, space="buy")
    buy08 = CategoricalParameter([True, False], default=True, space="buy")
    buy09 = CategoricalParameter([True, False], default=True, space="buy")
    buy10 = CategoricalParameter([True, False], default=True, space="buy")
    buy11 = CategoricalParameter([True, False], default=True, space="buy")
    buy12 = CategoricalParameter([True, False], default=True, space="buy")
    buy13 = CategoricalParameter([True, False], default=True, space="buy")
    buy14 = CategoricalParameter([True, False], default=True, space="buy")
    buy15 = CategoricalParameter([True, False], default=True, space="buy")
    buy16 = CategoricalParameter([True, False], default=True, space="buy")
    buy17 = CategoricalParameter([True, False], default=True, space="buy")
    buy18 = CategoricalParameter([True, False], default=True, space="buy")
    buy19 = CategoricalParameter([True, False], default=True, space="buy")
    buy20 = CategoricalParameter([True, False], default=True, space="buy")
    buy21 = CategoricalParameter([True, False], default=True, space="buy")
    buy22 = CategoricalParameter([True, False], default=True, space="buy")
    buy23 = CategoricalParameter([True, False], default=True, space="buy")
    buy24 = CategoricalParameter([True, False], default=True, space="buy")
    buy25 = CategoricalParameter([True, False], default=True, space="buy")
    buy26 = CategoricalParameter([True, False], default=True, space="buy")
    buy27 = CategoricalParameter([True, False], default=True, space="buy")
    buy28 = CategoricalParameter([True, False], default=True, space="buy")
    buy29 = CategoricalParameter([True, False], default=True, space="buy")
    buy30 = CategoricalParameter([True, False], default=True, space="buy")

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



    #----------------------- END OF HYPER-OPT PARAMETERS -------------------------#

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


    # @property
    # def max_entry_position_adjustment(self):
    #     return self.max_epa.value

    ### Dollar Cost Averaging ### This can be Turned on ###
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

        if current_profit > 0.1 and current_profit < 0.15 and trade.nr_of_successful_exits == 0:
            # Take 50% of the profit at +10%
            return -(trade.stake_amount / 2)

        if current_profit > -0.03 and trade.nr_of_successful_entries == 1:
            return None

        if current_profit > -0.1 and trade.nr_of_successful_entries == 2:
            return None

        if current_profit > -0.16 and trade.nr_of_successful_entries == 3:
            return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        # Allow up to 3 additional increasingly larger buys (4 in total)
        # Initial buy is 1x
        # If that falls to -5% profit, we buy 1.25x more, average profit should increase to roughly -2.2%
        # If that falls down to -5% again, we buy 1.5x more
        # If that falls once again down to -5%, we buy 1.75x more
        # Total stake for this trade would be 1 + 1.25 + 1.5 + 1.75 = 5.5x of the initial allowed stake.
        # Total stake for this trade would be 1 + 1.5 + 2 + 2.5 = 5.5x of the initial allowed stake.
        # That is why max_dca_multiplier is 5.5
        # Hope you have a deep wallet!
        try:
            # This returns first order stake size
            stake_amount = filled_entries[0].cost
            # This then calculates current safety order size
            stake_amount = stake_amount * (1 + (count_of_entries * 0.5))
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


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate all indicators used by the strategy"""

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

        # Reference MA and offsets
        for valma in self.reference_ma_length.range:
            dataframe[f'reference_ma_{valma}'] = ta.SMA(dataframe['close'], timeperiod=valma)

        dataframe['buy_offset1'] = dataframe[f'reference_ma_{valma}'] * self.buy_offset1.value
        dataframe['buy_offset2'] = dataframe[f'reference_ma_{valma}'] * self.buy_offset2.value
        dataframe['sell_offset1'] = dataframe[f'reference_ma_{valma}'] * self.sell_offset1.value
        dataframe['ref_slope'] =((dataframe[f'reference_ma_{valma}'] - dataframe[f'reference_ma_{valma}'].shift(1)) / dataframe[f'reference_ma_{valma}'].shift(1)) 
        dataframe['ref_slope_sma'] = ta.SMA((dataframe['ref_slope'] *100), timeperiod=5)
        
        # distance from reference ma to current close
        dataframe['change'] = ((dataframe['close'] - dataframe[f'reference_ma_{valma}']) / dataframe['close']) * 100

        # smoothing the change and offsets
        for valsma in self.smoothing_length.range:
            dataframe[f'smooth_change_{valsma}'] = ta.SMA(dataframe['change'], timeperiod=valsma)

        dataframe['buy_offset3'] = dataframe[f'smooth_change_{valsma}'] - self.buy_change.value
        dataframe['sell_offset2'] = dataframe[f'smooth_change_{valsma}'] + self.sell_change.value
        dataframe['smooth_ma_slope'] = pta.momentum.slope(dataframe[f'smooth_change_{valsma}'])
        dataframe['smooth_slope_sma'] = ta.SMA(dataframe['smooth_ma_slope'], timeperiod=5)

        ### I.ntelligent B.uying S.ystem ###
        # 300 Candle Rolling Min-Max
        for l in self.max_length.range:
            dataframe['min'] = dataframe['open'].rolling(l).min()
            dataframe['max'] = dataframe['close'].rolling(l).max()

        # distance from the rolling max in percent
        dataframe['from_max'] = ((dataframe['close'] - dataframe['max']) / dataframe['close']) * 100
        # distance from the rolling min in percent
        dataframe['from_min'] = ((dataframe['open'] - dataframe['max']) / dataframe['open']) * 100


        return dataframe


    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_bull.value)) &
                (df['ref_slope_sma'] > self.ref_bull.value) &
                (df['from_max'] > self.from_bull.value) &
                (self.buy01.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '1 Smooth Bull - Ref Bull')

#not using
        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_up.value)) &
                (df['smooth_slope_sma'] < self.smooth_bull.value) &
                (df['from_max'] < self.from_bull.value) &
                (df['ref_slope_sma'] > self.ref_bull.value) &
                (self.buy02.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '2 Smooth Up - Ref Bull')


        df.loc[
            (
                # (qtpylib.crossed_above(df['smooth_slope_sma'],  -self.smooth_ranging.value)) &
                (df['smooth_slope_sma'] < self.smooth_up.value) &
                (df['smooth_slope_sma'] > self.smooth_down.value) &
                (df['from_max'] < self.from_ranging.value) &
                (df['ref_slope_sma'] > self.ref_bull.value) &
                (self.buy03.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '3 Smooth Range - Ref Bull')

        df.loc[
            (
                (qtpylib.crossed_below(df['smooth_slope_sma'],  self.smooth_down.value)) &
                (df['smooth_slope_sma'] > self.smooth_bear.value) &
                (df['ref_slope_sma'] > self.ref_bull.value) &
                (df['from_max'] < self.from_down.value) &
                (self.buy04.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '4Smooth Down - Ref Bull')

#not using
        df.loc[
            (
                (qtpylib.crossed_below(df['smooth_slope_sma'],  self.smooth_bear.value)) &
                (df['close'] < df['buy_offset2']) & # changed from 1 - 2
                (df['smooth_slope_sma'] < self.smooth_bear.value) &
                (df['ref_slope_sma'] > self.ref_bull.value) &
                (df['from_max'] < self.from_bear.value) &
                (self.buy05.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '5 Smooth Bear - Ref Bull')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_bull.value)) &
                (df['ref_slope_sma'] < self.ref_bull.value) &
                (df['ref_slope_sma'] > self.ref_up.value) &
                (df['from_max'] < self.from_up.value) &
                (self.buy06.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '6 Smooth Bull - Ref Up')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_up.value)) &
                (df['smooth_slope_sma'] < self.smooth_bull.value) &
                (df['ref_slope_sma'] < self.ref_bull.value) &
                (df['ref_slope_sma'] > self.ref_up.value) &
                (df['from_max'] < self.from_up.value) &
                (self.buy07.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '7 Smooth Up - Ref Up')

# not using
        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_down.value)) &
                (df['smooth_slope_sma'] < self.smooth_up.value) &
                (df['smooth_slope_sma'] > self.smooth_down.value) &
                (df['ref_slope_sma'] < self.ref_bull.value) &
                (df['ref_slope_sma'] > self.ref_up.value) &
                (df['from_max'] < self.from_ranging.value) &
                (self.buy08.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '8 Smooth Range - Ref Up')
# not using
        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_down.value)) &
                (df['smooth_slope_sma'] > self.smooth_bear.value) &
                (df['ref_slope_sma'] < self.ref_bull.value) &
                (df['ref_slope_sma'] > self.ref_up.value) &
                (df['from_max'] < self.from_down.value) &
                (self.buy09.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '9 Smooth Down - Ref Up')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_bear.value)) &
                (df['close'] < df['buy_offset1']) &
                (df['smooth_slope_sma'] < self.smooth_bear.value) &
                (df['ref_slope_sma'] < self.ref_bull.value) &
                (df['ref_slope_sma'] > self.ref_up.value) &
                (df['from_max'] < self.from_bear.value) &
                (self.buy10.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '10 Smooth Bear - Ref Up')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_bull.value)) &
                (df['ref_slope_sma'] < self.ref_up.value) &
                (df['ref_slope_sma'] > self.ref_down.value) &
                (df['from_max'] < self.from_ranging.value) &
                (self.buy11.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '11 Smooth Bull - Ref Range')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_up.value)) &
                (df['smooth_slope_sma'] < self.smooth_bull.value) &
                (df['ref_slope_sma'] < self.ref_up.value) &
                (df['ref_slope_sma'] > self.ref_down.value) &
                (df['from_max'] < self.from_ranging.value) &
                (self.buy12.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '12 Smooth Up - Ref Range')


        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'], self.smooth_down.value)) &
                (df['smooth_slope_sma'] < self.smooth_up.value) &
                (df['smooth_slope_sma'] > self.smooth_down.value) &
                (df['ref_slope_sma'] < self.ref_up.value) &
                (df['ref_slope_sma'] > self.ref_down.value) &
                (df['from_max'] < self.from_ranging.value) &
                (self.buy13.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '13 Smooth Range - Ref Range')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_down.value)) &
                (df['smooth_slope_sma'] > self.smooth_bear.value) &
                (df['ref_slope_sma'] < self.ref_up.value) &
                (df['ref_slope_sma'] > self.ref_down.value) &
                (df['from_max'] < self.from_down.value) &
                (self.buy14.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '14 Smooth Down - Ref Range')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_bear.value)) &
                (df['close'] < df['buy_offset1']) &
                (df['smooth_slope_sma'] < self.smooth_bear.value) &
                (df['ref_slope_sma'] < self.ref_up.value) &
                (df['ref_slope_sma'] > self.ref_down.value) &
                (df['from_max'] < self.from_bear.value) &
                (self.buy15.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '15 Smooth Bear - Ref Range')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_bull.value)) &
                (df['ref_slope_sma'] < self.ref_down.value) &
                (df['ref_slope_sma'] > self.ref_bear.value) &
                (df['from_max'] < self.from_down.value) &
                (self.buy16.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '16 Smooth Bull - Ref Down')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_up.value)) &
                (df['smooth_slope_sma'] > self.smooth_bull.value) &
                (df['ref_slope_sma'] < self.ref_down.value) &
                (df['ref_slope_sma'] > self.ref_bear.value) &
                (df['from_max'] < self.from_down.value) &
                (self.buy17.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '17 Smooth Up - Ref Down')


        df.loc[
            (
                (qtpylib.crossed_above(df['change'], df[f'smooth_change_{self.smoothing_length.value}'])) &
                (df['close'] < df['buy_offset1']) &
                (df['smooth_slope_sma'] < self.smooth_up.value) &
                (df['smooth_slope_sma'] > self.smooth_down.value) &
                (df['ref_slope_sma'] < self.ref_down.value) &
                (df['ref_slope_sma'] > self.ref_bear.value) &
                (df['from_max'] < self.from_down.value) &
                (self.buy18.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '18 Smooth Range - Ref Down')

        df.loc[
            (
                (qtpylib.crossed_above(df['change'], df[f'smooth_change_{self.smoothing_length.value}'])) &
                (df['smooth_slope_sma'] > self.smooth_bear.value) &
                (df['ref_slope_sma'] < self.ref_down.value) &
                (df['ref_slope_sma'] > self.ref_bear.value) &
                (df['from_max'] < self.from_down.value) &
                (self.buy19.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '19 Smooth Down - Ref Down - % XO')

        df.loc[
            (
                (df['min'] > df['close']) &
                (df['smooth_slope_sma'] > self.smooth_bear.value) &
                (df['ref_slope_sma'] < self.ref_down.value) &
                (df['ref_slope_sma'] > self.ref_bear.value) &
                (df['from_max'] < self.from_down.value) &
                (self.buy20.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '20 Smooth Down - Ref Down - Open < Min')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_bear.value)) &
                (df['smooth_slope_sma'] < self.smooth_bear.value) &
                (df['ref_slope_sma'] < self.ref_down.value) &
                (df['ref_slope_sma'] > self.ref_bear.value) &
                (df['from_max'] < self.from_down.value) &
                (self.buy21.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '21 Smooth Bear - Ref Down')


        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_bull.value)) &
                (df['ref_slope_sma'] < self.ref_bear.value) &
                (df['from_max'] < self.from_bear.value) &
                (self.buy22.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '22 Smooth Bull - Ref Bear')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_up.value)) &
                (df['smooth_slope_sma'] < self.smooth_bull.value) &
                (df['ref_slope_sma'] < self.ref_bear.value) &
                (df['from_max'] < self.from_bear.value) &
                (self.buy23.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '23 Smooth Up - Ref Bear')


        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_down.value)) &
                (df['smooth_slope_sma'] < self.smooth_up.value) &
                (df['smooth_slope_sma'] > self.smooth_down.value) &
                (df['ref_slope_sma'] < self.ref_bear.value) &
                (df['from_max'] < self.from_bear.value) &
                (self.buy24.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '24 Smooth Range - Ref Bear')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_down.value)) &
                (df['smooth_slope_sma'] > self.smooth_bear.value) &
                (df['ref_slope_sma'] < self.ref_bear.value) &
                (df['from_max'] < self.from_bear.value) &
                (self.buy25.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '25 Smooth Down - Ref Bear')

        df.loc[
            (
                (qtpylib.crossed_above(df['smooth_slope_sma'],  self.smooth_bear.value)) &
                # (df['close'] < df['buy_offset2']) &
                (df['smooth_slope_sma'] < self.smooth_bear.value) &
                (df['ref_slope_sma'] < self.ref_bear.value) &
                (df['from_max'] < self.from_bear.value) &
                (self.buy26.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '26 Smooth Bear - Ref Bear')


        df.loc[
            (
                (qtpylib.crossed_above(df['change'], df['buy_offset3'])) &
                (df['pivot'] < df['sell_offset1']) &
                (df['close'] < df['buy_offset1']) &
                (df['smooth_slope_sma'] > self.smooth_down.value) &
                (df['from_max'] < self.from_ranging.value) &
                (self.buy27.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '27 XO above buy_offset3 | Range - Bull')

        df.loc[
            (
                (df['min'] < df['buy_offset1']) &
                ((df['from_max'] - df['from_max'].shift(8)) < self.from_down.value) &
                (df['close'] < df['buy_offset1']) &
                (self.buy28.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '28 Low XB Min < buy_offset1')


        df.loc[
            (
                (qtpylib.crossed_above(df['change'], df['buy_offset3'])) &
                (df['pivot'] < df['sell_offset1']) &
                (df['close'] < df['buy_offset1']) &
                (df['smooth_slope_sma'] < self.smooth_down.value) &
                (df['from_max'] < self.from_down.value) &
                (self.buy29.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '29 XO above buy_offset3 | Down - Bear')

        # df.loc[
        #     (
        #         (qtpylib.crossed_above(df['pivot'], df[f'reference_ma_{self.reference_ma_length.value}'])) &
        #         # (df['close'] < df[f'reference_ma_{self.reference_ma_length.value}']) &
        #         # (df[f'reference_ma_{self.reference_ma_length.value}'] > df[f'reference_ma_{self.reference_ma_length.value}'].shift(1)) &
        #         (df['smooth_ma_slope'] > df['ref_slope_sma']) &
        #         (df['ref_slope_sma'] > 0) &
        #         (self.buy30.value == True) &
        #         (df['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, '30 Golden XO')

        # df.loc[
        #     (
        #         (qtpylib.crossed_above(df['change'], df['buy_offset3'])) &
        #         (df['smooth_ma_slope'] < 0) &
        #         (df['close'] < df[f'reference_ma_{self.reference_ma_length.value}']) &
        #         (df['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'XO above buy_offset|slope down')

        # df.loc[
        #     (
        #         (qtpylib.crossed_above(df['change'], df['buy_offset3'])) &
        #         (df['smooth_ma_slope'] > -0.005) &
        #         (df['close'] < df[f'reference_ma_{self.reference_ma_length.value}']) &
        #         (df['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'XO above buy_offset|slope up')

        # df.loc[
        #     (
        #         (df['min'] < df['buy_offset1']) &
        #         ((df['from_max'] - df['from_max'].shift(8)) < self.bear_from_max.value) &
        #         (df['close'] < df['buy_offset1']) &
        #         (df['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'Low XB Min < buy_offset1')

        # df.loc[
        #     (
        #         (qtpylib.crossed_above(df['close'], df['buy_offset2'])) &
        #         (df['change'] < df['buy_offset3']) &
        #         (df['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'XO above buy_offset2')

        # df.loc[
        #     (
        #         (qtpylib.crossed_above(df['change'], df[f'smooth_change_{self.smoothing_length.value}'])) &
        #         (df['close'] < df[f'reference_ma_{self.reference_ma_length.value}']) &
        #         (df[f'reference_ma_{self.reference_ma_length.value}'] > df[f'reference_ma_{self.reference_ma_length.value}'].shift(1)) &
        #         (df['close'] > df['buy_offset1']) &
        #         (df['smooth_ma_slope'] < 0) &
        #         (df['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'XO above Ref')

        return df


    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r5'])) &
                (df['close'] > (df[f'reference_ma_{self.reference_ma_length.value}'] * self.sell_offset1.value)) &
                (self.sell01.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R5 - XO')
      
        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r3'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (self.sell02.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R3 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r2.75'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (self.sell03.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R2.75 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r2.50'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (self.sell04.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R2.5 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r2.25'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (self.sell05.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R2.25 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r2'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (self.sell06.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R2 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['pivot'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (self.sell07.value == True) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'Pivot - XO')

        return df


        # Best result:

        #    585/1000:    279 trades. 259/0/20 Wins/Draws/Losses. Avg profit   2.54%. Median profit   3.90%. Total profit 2061.56810074 USDT ( 206.16%). Avg duration 1 day, 23:57:00 min. Objective: -2061.56810


        #     # Buy hyperspace params:
        #     buy_params = {
        #         "buy1": True,  # value loaded from strategy
        #         "buy10": True,  # value loaded from strategy
        #         "buy11": False,  # value loaded from strategy
        #         "buy12": False,  # value loaded from strategy
        #         "buy13": False,  # value loaded from strategy
        #         "buy14": False,  # value loaded from strategy
        #         "buy15": False,  # value loaded from strategy
        #         "buy16": True,  # value loaded from strategy
        #         "buy17": True,  # value loaded from strategy
        #         "buy18": True,  # value loaded from strategy
        #         "buy19": False,  # value loaded from strategy
        #         "buy2": False,  # value loaded from strategy
        #         "buy20": True,  # value loaded from strategy
        #         "buy21": True,  # value loaded from strategy
        #         "buy22": True,  # value loaded from strategy
        #         "buy23": True,  # value loaded from strategy
        #         "buy24": False,  # value loaded from strategy
        #         "buy25": True,  # value loaded from strategy
        #         "buy26": True,  # value loaded from strategy
        #         "buy27": True,  # value loaded from strategy
        #         "buy28": False,  # value loaded from strategy
        #         "buy29": True,  # value loaded from strategy
        #         "buy3": True,  # value loaded from strategy
        #         "buy4": True,  # value loaded from strategy
        #         "buy5": False,  # value loaded from strategy
        #         "buy6": True,  # value loaded from strategy
        #         "buy7": True,  # value loaded from strategy
        #         "buy8": False,  # value loaded from strategy
        #         "buy9": False,  # value loaded from strategy
        #         "buy_change": 1.6,  # value loaded from strategy
        #         "buy_offset1": 0.99,  # value loaded from strategy
        #         "buy_offset2": 0.92,  # value loaded from strategy
        #         "from_bear": -11.7,  # value loaded from strategy
        #         "from_bull": -3.1,  # value loaded from strategy
        #         "from_down": -5.4,  # value loaded from strategy
        #         "from_ranging": -3.5,  # value loaded from strategy
        #         "from_up": -4.7,  # value loaded from strategy
        #         "ref_bear": -0.113,  # value loaded from strategy
        #         "ref_bull": 0.098,  # value loaded from strategy
        #         "ref_down": -0.067,  # value loaded from strategy
        #         "ref_up": 0.027,  # value loaded from strategy
        #         "reference_ma_length": 192,  # value loaded from strategy
        #         "smooth_bear": -0.117,  # value loaded from strategy
        #         "smooth_bull": 0.086,  # value loaded from strategy
        #         "smooth_down": -0.013,  # value loaded from strategy
        #         "smooth_up": 0.073,  # value loaded from strategy
        #         "smoothing_length": 23,  # value loaded from strategy
        #     }

        #     # Sell hyperspace params:
        #     sell_params = {
        #         "filterlength": 35,
        #         "sell_change": 1.8,
        #         "sell_offset1": 1.03,
        #         "sell_slope": -0.04,
        #         "ts0": 0.008,
        #         "ts1": 0.011,
        #         "ts2": 0.021,
        #         "ts3": 0.031,
        #         "ts4": 0.03,
        #         "ts5": 0.05,
        #         "tsl_target0": 0.044,
        #         "tsl_target1": 0.055,
        #         "tsl_target2": 0.083,
        #         "tsl_target3": 0.12,
        #         "tsl_target4": 0.2,
        #         "tsl_target5": 0.4,
        #     }

        #     # Protection hyperspace params:
        #     protection_params = {
        #         "cooldown_lookback": 1,
        #         "lowprofit_only_per_pair": True,
        #         "lowprofit_protection_lookback": 1,
        #         "lowprofit_required_profit": -0.829,
        #         "lowprofit_stop_duration": 83,
        #         "lowprofit_trade_limit": 1,
        #         "maxdrawdown_allowed_drawdown": 0.1,
        #         "maxdrawdown_protection_lookback": 3,
        #         "maxdrawdown_stop_duration": 74,
        #         "maxdrawdown_trade_limit": 1,
        #         "stop_duration": 102,
        #         "stop_protection_only_per_pair": True,
        #         "stop_protection_only_per_side": True,
        #         "stop_protection_required_profit": -0.924,
        #         "stop_protection_trade_limit": 9,
        #         "use_lowprofit_protection": False,
        #         "use_maxdrawdown_protection": True,
        #         "use_stop_protection": False,
        #     }

        #     # ROI table:  # value loaded from strategy
        #     minimal_roi = {
        #         "0": 0.1
        #     }

        #     # Stoploss:
        #     stoploss = -0.2  # value loaded from strategy

        #     # Trailing stop:
        #     trailing_stop = True  # value loaded from strategy
        #     trailing_stop_positive = None  # value loaded from strategy
        #     trailing_stop_positive_offset = 0.0  # value loaded from strategy
        #     trailing_only_offset_is_reached = False  # value loaded from strategy


# {
#   "strategy_name": "CTIBS",
#   "params": {
#     "roi": {
#       "0": 0.14300000000000002,
#       "736": 0.099,
#       "2369": 0.024,
#       "5731": 0
#     },
#     "stoploss": {
#       "stoploss": -0.28
#     },
#     "trailing": {
#       "trailing_stop": true,
#       "trailing_stop_positive": null,
#       "trailing_stop_positive_offset": 0.0,
#       "trailing_only_offset_is_reached": false
#     },
#     "max_open_trades": {
#       "max_open_trades": 6
#     },
#     "buy": {
#       "buy1": true,
#       "buy10": false,
#       "buy11": true,
#       "buy12": false,
#       "buy13": false,
#       "buy14": true,
#       "buy15": false,
#       "buy16": true,
#       "buy17": false,
#       "buy18": false,
#       "buy19": false,
#       "buy2": false,
#       "buy20": false,
#       "buy21": false,
#       "buy22": false,
#       "buy23": true,
#       "buy24": false,
#       "buy25": true,
#       "buy26": false,
#       "buy27": true,
#       "buy28": true,
#       "buy29": true,
#       "buy3": true,
#       "buy4": false,
#       "buy5": false,
#       "buy6": true,
#       "buy7": false,
#       "buy8": false,
#       "buy9": true,
#       "buy_change": 1.5,
#       "buy_offset1": 0.98,
#       "buy_offset2": 0.94,
#       "from_bear": -12.4,
#       "from_bull": -3.1,
#       "from_down": -7.1,
#       "from_ranging": -2.7,
#       "from_up": -2.0,
#       "max_length": 48,
#       "ref_bear": -0.088,
#       "ref_bull": 0.087,
#       "ref_down": -0.075,
#       "ref_up": 0.062,
#       "reference_ma_length": 187,
#       "smooth_bear": -0.098,
#       "smooth_bull": 0.085,
#       "smooth_down": -0.058,
#       "smooth_up": 0.052,
#       "smoothing_length": 15
#     },
#     "sell": {
#       "filterlength": 40,
#       "sell_change": 1.5,
#       "sell_offset1": 1.04,
#       "sell_slope": 0.01,
#       "ts0": 0.008,
#       "ts1": 0.01,
#       "ts2": 0.018,
#       "ts3": 0.031,
#       "ts4": 0.05,
#       "ts5": 0.04,
#       "tsl_target0": 0.041,
#       "tsl_target1": 0.079,
#       "tsl_target2": 0.099,
#       "tsl_target3": 0.14,
#       "tsl_target4": 0.17,
#       "tsl_target5": 0.3
#     },
#     "protection": {
#       "cooldown_lookback": 48,
#       "lowprofit_only_per_pair": true,
#       "lowprofit_protection_lookback": 5,
#       "lowprofit_required_profit": -0.704,
#       "lowprofit_stop_duration": 100,
#       "lowprofit_trade_limit": 2,
#       "maxdrawdown_allowed_drawdown": 0.045,
#       "maxdrawdown_protection_lookback": 3,
#       "maxdrawdown_stop_duration": 15,
#       "maxdrawdown_trade_limit": 5,
#       "stop_duration": 19,
#       "stop_protection_only_per_pair": true,
#       "stop_protection_only_per_side": true,
#       "stop_protection_required_profit": -0.126,
#       "stop_protection_trade_limit": 9,
#       "use_lowprofit_protection": true,
#       "use_maxdrawdown_protection": false,
#       "use_stop_protection": false
#     }
#   },
#   "ft_stratparam_v": 1,
#   "export_time": "2023-05-02 22:49:05.899725+00:00"
# }

# Best result:

#   1425/1500:    365 trades. 318/20/27 Wins/Draws/Losses. Avg profit   2.60%. Median profit   3.93%. Total profit 3495.92254012 USDT ( 349.59%). Avg duration 1 day, 18:44:00 min. Objective: -3495.92254


#     # Buy hyperspace params:
#     buy_params = {
#         "buy01": True,
#         "buy02": False,
#         "buy03": True,
#         "buy04": True,
#         "buy05": True,
#         "buy06": False,
#         "buy07": True,
#         "buy08": True,
#         "buy09": False,
#         "buy10": False,
#         "buy11": False,
#         "buy12": False,
#         "buy13": False,
#         "buy14": False,
#         "buy15": False,
#         "buy16": False,
#         "buy17": False,
#         "buy18": False,
#         "buy19": False,
#         "buy20": False,
#         "buy21": True,
#         "buy22": True,
#         "buy23": True,
#         "buy24": True,
#         "buy25": True,
#         "buy26": False,
#         "buy27": False,
#         "buy28": False,
#         "buy29": True,
#         "buy30": False,
#         "buy_change": 1.2,
#         "buy_offset1": 0.99,
#         "buy_offset2": 0.92,
#         "from_bear": -10.5,
#         "from_bull": -3.1,
#         "from_down": -6.3,
#         "from_ranging": -2.5,
#         "from_up": -2.2,
#         "max_length": 192,
#         "ref_bear": -0.097,
#         "ref_bull": 0.095,
#         "ref_down": -0.07,
#         "ref_up": 0.043,
#         "reference_ma_length": 191,
#         "smooth_bear": -0.095,
#         "smooth_bull": 0.076,
#         "smooth_down": -0.056,
#         "smooth_up": 0.04,
#         "smoothing_length": 23,
#     }

#     # Sell hyperspace params:
#     sell_params = {
#         "filterlength": 30,
#         "sell01": False,
#         "sell02": True,
#         "sell03": True,
#         "sell04": True,
#         "sell05": False,
#         "sell06": False,
#         "sell07": True,
#         "sell08": True,
#         "sell09": False,
#         "sell10": True,
#         "sell_change": 1.7,
#         "sell_offset1": 1.04,
#         "sell_slope": 0.0,
#         "ts0": 0.008,
#         "ts1": 0.01,
#         "ts2": 0.022,
#         "ts3": 0.038,
#         "ts4": 0.03,
#         "ts5": 0.04,
#         "tsl_target0": 0.044,
#         "tsl_target1": 0.065,
#         "tsl_target2": 0.097,
#         "tsl_target3": 0.12,
#         "tsl_target4": 0.16,
#         "tsl_target5": 0.3,
#     }

#     # Protection hyperspace params:
#     protection_params = {
#         "cooldown_lookback": 48,  # value loaded from strategy
#         "lowprofit_only_per_pair": True,  # value loaded from strategy
#         "lowprofit_protection_lookback": 5,  # value loaded from strategy
#         "lowprofit_required_profit": -0.704,  # value loaded from strategy
#         "lowprofit_stop_duration": 100,  # value loaded from strategy
#         "lowprofit_trade_limit": 2,  # value loaded from strategy
#         "maxdrawdown_allowed_drawdown": 0.045,  # value loaded from strategy
#         "maxdrawdown_protection_lookback": 3,  # value loaded from strategy
#         "maxdrawdown_stop_duration": 15,  # value loaded from strategy
#         "maxdrawdown_trade_limit": 5,  # value loaded from strategy
#         "stop_duration": 19,  # value loaded from strategy
#         "stop_protection_only_per_pair": True,  # value loaded from strategy
#         "stop_protection_only_per_side": True,  # value loaded from strategy
#         "stop_protection_required_profit": -0.126,  # value loaded from strategy
#         "stop_protection_trade_limit": 9,  # value loaded from strategy
#         "use_lowprofit_protection": True,  # value loaded from strategy
#         "use_maxdrawdown_protection": False,  # value loaded from strategy
#         "use_stop_protection": False,  # value loaded from strategy
#     }

#     # ROI table:  # value loaded from strategy
#     minimal_roi = {
#         "0": 0.143,
#         "736": 0.099,
#         "2369": 0.024,
#         "5731": 0
#     }

#     # Stoploss:
#     stoploss = -0.28  # value loaded from strategy

#     # Trailing stop:
#     trailing_stop = True  # value loaded from strategy
#     trailing_stop_positive = None  # value loaded from strategy
#     trailing_stop_positive_offset = 0.0  # value loaded from strategy
#     trailing_only_offset_is_reached = False  # value loaded from strategy
    

#     # Max Open Trades:
#     max_open_trades = 6  # value loaded from strategy

# Result for strategy CTIBS
# ============================================================= BACKTESTING REPORT =============================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# |  OPUL/USDT |        30 |           4.64 |         139.20 |           708.395 |          70.84 |         14:03:00 |    28     1     1  93.3 |
# |  AGIX/USDT |        37 |           3.02 |         111.78 |           342.690 |          34.27 |         14:09:00 |    32     0     5  86.5 |
# |   INJ/USDT |        21 |           2.79 |          58.60 |           332.616 |          33.26 |  1 day, 16:47:00 |    18     1     2  85.7 |
# |   LDO/USDT |        20 |           5.12 |         102.36 |           302.659 |          30.27 |         23:04:00 |    19     1     0   100 |
# |  RNDR/USDT |        23 |           2.46 |          56.48 |           292.762 |          29.28 |   1 day, 1:40:00 |    22     0     1  95.7 |
# |    OP/USDT |        11 |           4.43 |          48.75 |           225.127 |          22.51 |         10:14:00 |    11     0     0   100 |
# |   IMX/USDT |        18 |           2.33 |          41.89 |           218.038 |          21.80 |  1 day, 13:00:00 |    17     0     1  94.4 |
# |  ORAI/USDT |        19 |           2.63 |          49.93 |           156.959 |          15.70 |         14:02:00 |    18     0     1  94.7 |
# |   APT/USDT |        10 |           4.28 |          42.77 |           144.811 |          14.48 |   1 day, 1:15:00 |    10     0     0   100 |
# |  KLAY/USDT |         6 |           3.28 |          19.71 |           132.670 |          13.27 |  1 day, 12:40:00 |     4     1     1  66.7 |
# |  DYDX/USDT |        16 |           2.16 |          34.52 |           125.692 |          12.57 | 2 days, 13:57:00 |    14     1     1  87.5 |
# |   FIL/USDT |         6 |           4.08 |          24.47 |           125.056 |          12.51 |         20:08:00 |     6     0     0   100 |
# |   ARB/USDT |         5 |           3.15 |          15.75 |           112.833 |          11.28 | 3 days, 16:30:00 |     4     1     0   100 |
# |  CSPR/USDT |         7 |           2.80 |          19.58 |           108.527 |          10.85 |   1 day, 7:24:00 |     7     0     0   100 |
# |  ROSE/USDT |         4 |           4.21 |          16.85 |           106.125 |          10.61 |         18:41:00 |     4     0     0   100 |
# |   FLR/USDT |         4 |           5.50 |          21.98 |            99.699 |           9.97 |         15:41:00 |     4     0     0   100 |
# |   GRT/USDT |         5 |           3.21 |          16.05 |            96.898 |           9.69 |   1 day, 0:21:00 |     5     0     0   100 |
# | GALAX/USDT |         8 |           5.04 |          40.31 |            84.824 |           8.48 |          8:26:00 |     7     0     1  87.5 |
# |  HBAR/USDT |         6 |           2.17 |          13.02 |            80.588 |           8.06 | 9 days, 16:55:00 |     3     3     0   100 |
# | JASMY/USDT |        11 |           2.85 |          31.36 |            78.855 |           7.89 |  2 days, 1:59:00 |     9     0     2  81.8 |
# |   CRO/USDT |         1 |           8.57 |           8.57 |            67.589 |           6.76 |         16:30:00 |     1     0     0   100 |
# |   EWT/USDT |         5 |           2.53 |          12.67 |            64.415 |           6.44 |   1 day, 3:06:00 |     5     0     0   100 |
# |   FET/USDT |         2 |           3.41 |           6.83 |            49.115 |           4.91 |   1 day, 2:38:00 |     2     0     0   100 |
# |   XRP/USDT |         2 |           3.26 |           6.53 |            41.612 |           4.16 |         17:22:00 |     2     0     0   100 |
# |  SAND/USDT |         3 |           4.62 |          13.86 |            38.537 |           3.85 |         12:20:00 |     3     0     0   100 |
# |  NEAR/USDT |         4 |           2.51 |          10.03 |            36.898 |           3.69 |   1 day, 3:08:00 |     4     0     0   100 |
# |   XLM/USDT |         2 |           3.03 |           6.07 |            28.982 |           2.90 |         16:08:00 |     2     0     0   100 |
# |   TRX/USDT |         2 |           4.11 |           8.23 |            27.729 |           2.77 |         12:08:00 |     2     0     0   100 |
# |  KAVA/USDT |         2 |           4.18 |           8.36 |            27.547 |           2.75 |   1 day, 1:38:00 |     2     0     0   100 |
# |   SOL/USDT |         7 |           1.90 |          13.32 |            27.384 |           2.74 |  2 days, 6:04:00 |     4     3     0   100 |
# |   QNT/USDT |         2 |           2.24 |           4.48 |            25.599 |           2.56 |  2 days, 8:00:00 |     2     0     0   100 |
# |   APE/USDT |         3 |           3.18 |           9.54 |            24.531 |           2.45 |         13:15:00 |     3     0     0   100 |
# |   ETC/USDT |         2 |           4.58 |           9.16 |            16.738 |           1.67 |         22:45:00 |     2     0     0   100 |
# |   ENJ/USDT |         2 |           2.21 |           4.42 |            14.311 |           1.43 |  2 days, 2:22:00 |     1     1     0   100 |
# |   BTC/USDT |         1 |           2.40 |           2.40 |            14.117 |           1.41 |  2 days, 6:45:00 |     1     0     0   100 |
# |   ZEC/USDT |         1 |           2.40 |           2.40 |            11.422 |           1.14 |  2 days, 8:00:00 |     1     0     0   100 |
# |   CRV/USDT |         3 |           1.03 |           3.09 |            10.368 |           1.04 | 2 days, 20:15:00 |     2     1     0   100 |
# |  ATOM/USDT |         3 |           0.65 |           1.94 |             8.839 |           0.88 | 3 days, 12:30:00 |     2     1     0   100 |
# |   XTZ/USDT |         1 |           1.79 |           1.79 |             8.540 |           0.85 | 2 days, 22:30:00 |     1     0     0   100 |
# |   KSM/USDT |         1 |           2.40 |           2.40 |             8.217 |           0.82 | 2 days, 14:15:00 |     1     0     0   100 |
# |  AVAX/USDT |         3 |           0.90 |           2.71 |             8.148 |           0.81 | 6 days, 15:15:00 |     2     1     0   100 |
# |  SCRT/USDT |         1 |           2.40 |           2.40 |             5.970 |           0.60 |  1 day, 21:45:00 |     1     0     0   100 |
# | THETA/USDT |         2 |           0.65 |           1.29 |             4.826 |           0.48 | 9 days, 15:52:00 |     1     1     0   100 |
# |  IOTA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |  OSMO/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |   ETH/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |   XDC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |  LINK/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |   DOT/USDT |         1 |           0.00 |           0.00 |             0.000 |           0.00 |  6 days, 8:00:00 |     0     1     0     0 |
# |   UNI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |   VET/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |   EOS/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |  DASH/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# | OCEAN/USDT |        12 |           1.49 |          17.92 |            -8.904 |          -0.89 | 2 days, 18:38:00 |    10     1     1  83.3 |
# |  DOGE/USDT |         2 |          -1.34 |          -2.68 |           -21.435 |          -2.14 | 12 days, 9:15:00 |     1     0     1  50.0 |
# |   ADA/USDT |         2 |          -1.69 |          -3.39 |           -28.639 |          -2.86 | 5 days, 17:38:00 |     1     0     1  50.0 |
# |  EGLD/USDT |         2 |          -2.38 |          -4.76 |           -47.439 |          -4.74 |   1 day, 2:45:00 |     1     0     1  50.0 |
# |   AKT/USDT |         7 |          -1.86 |         -13.02 |          -113.644 |         -11.36 |  2 days, 7:26:00 |     6     0     1  85.7 |
# |  LUNC/USDT |         2 |         -12.83 |         -25.66 |          -140.852 |         -14.09 | 17 days, 5:38:00 |     0     1     1     0 |
# | MATIC/USDT |         3 |          -6.91 |         -20.73 |          -158.044 |         -15.80 | 5 days, 14:05:00 |     2     0     1  66.7 |
# |  ALGO/USDT |         2 |         -12.97 |         -25.95 |          -182.961 |         -18.30 |  8 days, 4:00:00 |     1     0     1  50.0 |
# |  ANKR/USDT |        10 |          -2.20 |         -21.96 |          -249.415 |         -24.94 |  1 day, 19:15:00 |     7     0     3  70.0 |
# |      TOTAL |       365 |           2.60 |         947.62 |          3495.923 |         349.59 |  1 day, 18:44:00 |   318    20    27  87.1 |
# =========================================================== LEFT OPEN TRADES REPORT ===========================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |      Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+-------------------+-------------------------|
# |   INJ/USDT |         1 |          -1.17 |          -1.17 |            -9.476 |          -0.95 |           4:30:00 |     0     0     1     0 |
# |   ADA/USDT |         1 |          -4.28 |          -4.28 |           -32.818 |          -3.28 | 10 days, 12:15:00 |     0     0     1     0 |
# | JASMY/USDT |         1 |          -5.83 |          -5.83 |           -42.769 |          -4.28 | 13 days, 20:45:00 |     0     0     1     0 |
# |  EGLD/USDT |         1 |          -9.01 |          -9.01 |           -73.187 |          -7.32 |   1 day, 19:15:00 |     0     0     1     0 |
# |  DOGE/USDT |         1 |         -16.97 |         -16.97 |          -116.795 |         -11.68 | 24 days, 18:15:00 |     0     0     1     0 |
# |   AKT/USDT |         1 |         -19.49 |         -19.49 |          -148.353 |         -14.84 |  11 days, 4:30:00 |     0     0     1     0 |
# |      TOTAL |         6 |          -9.46 |         -56.73 |          -423.397 |         -42.34 |  10 days, 9:15:00 |     0     0     6     0 |
# ============================================================================ ENTER TAG STATS ============================================================================
# |                                   TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |---------------------------------------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# |               8 Smooth Range - Ref Up |        71 |           3.87 |         275.08 |          1176.933 |         117.69 |  1 day, 10:28:00 |    66     4     1  93.0 |
# |              1 Smooth Bull - Ref Bull |        62 |           3.90 |         241.71 |          1018.094 |         101.81 |   1 day, 3:00:00 |    57     2     3  91.9 |
# |             3 Smooth Range - Ref Bull |       126 |           2.14 |         269.55 |           773.706 |          77.37 |  1 day, 13:27:00 |   107     6    13  84.9 |
# | 29 XO above buy_offset3 | Down - Bear |        77 |           1.05 |          80.76 |           252.870 |          25.29 | 2 days, 19:45:00 |    64     6     7  83.1 |
# |                  7 Smooth Up - Ref Up |        27 |           2.81 |          75.83 |           246.711 |          24.67 |  2 days, 8:07:00 |    22     2     3  81.5 |
# |             22 Smooth Bull - Ref Bear |         1 |           3.81 |           3.81 |            23.052 |           2.31 |         21:30:00 |     1     0     0   100 |
# |               4Smooth Down - Ref Bull |         1 |           0.87 |           0.87 |             4.556 |           0.46 |          0:15:00 |     1     0     0   100 |
# |                                 TOTAL |       365 |           2.60 |         947.62 |          3495.923 |         349.59 |  1 day, 18:44:00 |   318    20    27  87.1 |
# ======================================================= EXIT REASON STATS ========================================================
# |        Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
# |--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
# | trailing_stop_loss |     249 |    228     0    21  91.6 |           3.09 |         770.24 |          2813.35  |         128.37 |
# |                roi |      74 |     54    20     0   100 |           2.71 |         200.51 |           922.846 |          33.42 |
# |         Pivot - XO |      21 |     21     0     0   100 |           0.8  |          16.75 |            99.357 |           2.79 |
# |          R2.5 - XO |       8 |      8     0     0   100 |           1.3  |          10.37 |            54.745 |           1.73 |
# |         force_exit |       6 |      0     0     6     0 |          -9.46 |         -56.73 |          -423.397 |          -9.46 |
# |         R2.75 - XO |       4 |      4     0     0   100 |           1    |           4.01 |            17.628 |           0.67 |
# |            R3 - XO |       3 |      3     0     0   100 |           0.82 |           2.47 |            11.393 |           0.41 |
# ================== SUMMARY METRICS ==================
# | Metric                      | Value               |
# |-----------------------------+---------------------|
# | Backtesting from            | 2023-01-01 00:00:00 |
# | Backtesting to              | 2023-04-30 00:00:00 |
# | Max open trades             | 6                   |
# |                             |                     |
# | Total/Daily Avg Trades      | 365 / 3.07          |
# | Starting balance            | 1000 USDT           |
# | Final balance               | 4495.923 USDT       |
# | Absolute profit             | 3495.923 USDT       |
# | Total profit %              | 349.59%             |
# | CAGR %                      | 9954.11%            |
# | Sortino                     | 7.32                |
# | Sharpe                      | 14.53               |
# | Calmar                      | 185.66              |
# | Profit factor               | 2.33                |
# | Expectancy                  | 0.04                |
# | Trades per day              | 3.07                |
# | Avg. daily profit %         | 2.94%               |
# | Avg. stake amount           | 486.735 USDT        |
# | Total trade volume          | 177658.245 USDT     |
# |                             |                     |
# | Best Pair                   | OPUL/USDT 139.20%   |
# | Worst Pair                  | ALGO/USDT -25.95%   |
# | Best trade                  | LDO/USDT 14.29%     |
# | Worst trade                 | MATIC/USDT -27.97%  |
# | Best day                    | 316.573 USDT        |
# | Worst day                   | -423.397 USDT       |
# | Days win/draw/lose          | 87 / 23 / 9         |
# | Avg. Duration Winners       | 18:38:00            |
# | Avg. Duration Loser         | 6 days, 21:24:00    |
# | Rejected Entry signals      | 455976              |
# | Entry/Exit Timeouts         | 0 / 0               |
# |                             |                     |
# | Min balance                 | 1001.773 USDT       |
# | Max balance                 | 4919.32 USDT        |
# | Max % of account underwater | 30.23%              |
# | Absolute Drawdown (Account) | 30.23%              |
# | Absolute Drawdown           | 1316.127 USDT       |
# | Drawdown high               | 3353.592 USDT       |
# | Drawdown low                | 2037.465 USDT       |
# | Drawdown Start              | 2023-02-23 01:45:00 |
# | Drawdown End                | 2023-03-10 10:00:00 |
# | Market change               | 84.63%              |
# =====================================================
