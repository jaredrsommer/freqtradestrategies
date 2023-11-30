
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

class cryptotankV5(IStrategy):

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
    stoploss = -0.25
    timeframe ='15m'

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30 

    ### HYPER-OPT PARAMETERS ###

    # protections
    cooldown_lookback = IntParameter(2, 48, default=5, space="protection", optimize=True)
    stop_duration = IntParameter(1, 36, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # entry optimization
    # max_epa = CategoricalParameter([0, 1, 2, 3], default=0, space="buy", optimize=True)

    # indicators
    # buy_rsi_length = IntParameter(14, 16, default=14, space="buy",optimize=True)
    # buy_rsi_ma_length = IntParameter(5, 14, default=14, space="buy",optimize=True)
    reference_ma_length = IntParameter(185, 195, default=200, space="buy" ,optimize=True)
    smoothing_length = IntParameter(15, 30, default=30, space="buy", optimize=True)
    filterlength = IntParameter(low=30, high=40, default=35, space='sell', optimize=True)
    min_length = IntParameter(5, 300, default=300, space="buy", optimize=True)
    max_length = IntParameter(5, 300, default=300, space="buy", optimize=True)
    buy_offset1 = DecimalParameter(low=0.97, high=0.99, decimals=2, default=0.99, space='buy', optimize=True, load=True)
    buy_offset2 = DecimalParameter(low=0.92, high=0.97, decimals=2, default=0.96, space='buy', optimize=True, load=True)
    sell_offset = DecimalParameter(low=1.01, high=1.05, decimals=2, default=1.02, space='sell', optimize=True, load=True)

    #trailing stop loss optimiziation
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

    ### buying 
    # # rsi
    # buy_rsi_upper = IntParameter(55, 65, default=45, space="buy", optimize=True)
    # buy_rsi_lower = IntParameter(20, 35, default=35, space="buy", optimize=True)
    # buy_rsi_ma_slope = DecimalParameter(-0.01, 0.10, default=0, decimals=2, space="buy", optimize=True)
    # buy_ma_slope = DecimalParameter(-0.10, 0.10, default=0, decimals=2, space="buy", optimize=True)
    # buy_lower = DecimalParameter(-10, 0, default=-2, decimals=1, space="buy", optimize=True)
    # buy_rsi_dip = IntParameter(30, 40, default=35, space="buy", optimize=True)
    buy_change = DecimalParameter(1.2, 2.0, default=1.5, decimals=1, space="buy", optimize=True)
    sell_change = DecimalParameter(1.2, 2.0, default=1.5, decimals=1, space="sell", optimize=True)
    sell_slope = DecimalParameter(-0.05, 0.10, default=0.04, decimals=2, space="sell", optimize=True)
    bear_from_max = DecimalParameter(-10.0, -5.0, default=-6.5, decimals=1, space="buy", optimize=True)



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
                "trade_limit": 1,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False
            })

        return prot
        

    # @property
    # def max_entry_position_adjustment(self):
    #     return self.max_epa.value

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

        if current_profit > 0.1 and current_profit < 0.15 and trade.nr_of_successful_exits == 0:
            # Take 50% of the profit at +5%
            return -(trade.stake_amount / 2)



        if current_profit > -0.03 and trade.nr_of_successful_entries == 1:
            return None

        if current_profit > -0.1 and trade.nr_of_successful_entries == 2:
            return None

        if current_profit > -0.16 and trade.nr_of_successful_entries == 3:
            return None

        # # Obtain pair dataframe (just to show how to access it)
        # dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        # # Only buy when not actively falling price.
        # last_candle = dataframe.iloc[-1].squeeze()
        # previous_candle = dataframe.iloc[-2].squeeze()
        # if last_candle['close'] < previous_candle['close']:
        #     return None

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

        # Filter ZEMA
        for length in self.filterlength.range:
            dataframe[f'ema_1{length}'] = ta.EMA(dataframe['close'], timeperiod=length)
            dataframe[f'ema_2{length}'] = ta.EMA(dataframe[f'ema_1{length}'], timeperiod=length)
            dataframe[f'ema_dif{length}'] = dataframe[f'ema_1{length}'] - dataframe[f'ema_2{length}']
            dataframe[f'zema_{length}'] = dataframe[f'ema_1{length}'] + dataframe[f'ema_dif{length}']

        # # RSI
        # # Calculate all rsi_buy values
        # for valb in self.buy_rsi_length.range:
        #     dataframe[f'buy_rsi_{valb}'] = ta.RSI(dataframe, timeperiod=valb)

        # # Calculate all rsi_buy ma values
        # for valbm in self.buy_rsi_ma_length.range:
        #     dataframe[f'buy_rsi_ma_{valbm}'] = ta.SMA(dataframe[f'buy_rsi_{valb}'], timeperiod=valbm)

        # dataframe['rsi_ma_slope'] = pta.momentum.slope(dataframe[f'buy_rsi_ma_{valbm}'])


        # % from reference MA
        for valma in self.reference_ma_length.range:
            dataframe[f'reference_ma_{valma}'] = ta.SMA(dataframe['close'], timeperiod=valma)
        # distance from reference ma to current close
        dataframe['change'] = ((dataframe['close'] - dataframe[f'reference_ma_{valma}']) / dataframe['close']) * 100

        dataframe['buy_offset1'] = dataframe[f'reference_ma_{valma}'] * self.buy_offset1.value

        dataframe['buy_offset2'] = dataframe[f'reference_ma_{valma}'] * self.buy_offset2.value

        dataframe['sell_offset'] = dataframe[f'reference_ma_{valma}'] * self.sell_offset.value
        
        for valsma in self.smoothing_length.range:
            dataframe[f'smooth_change_{valsma}'] = ta.SMA(dataframe['change'], timeperiod=valsma)

        # dataframe['ATR_change'] = pta.volatility.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=14)

        dataframe['smooth_ma_slope'] = pta.momentum.slope(dataframe[f'smooth_change_{valsma}'])

        dataframe['buy_offset'] = dataframe[f'smooth_change_{valsma}'] - self.buy_change.value

        dataframe['sell_offset1'] = dataframe[f'smooth_change_{valsma}'] + self.sell_change.value

        for minl in self.min_length.range:
            dataframe['min'] = dataframe['open'].rolling(minl).min()
        for maxl in self.max_length.range:
            dataframe['max'] = dataframe['close'].rolling(maxl).max()

        dataframe['from_max'] = ((dataframe['close'] - dataframe['max']) / dataframe['close']) * 100

        return dataframe


    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        df.loc[
            (
                (qtpylib.crossed_above(df['change'], df['buy_offset'])) &
                (df['smooth_ma_slope'] < 0) &
                (df['close'] < df[f'reference_ma_{self.reference_ma_length.value}']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'XO above buy_offset|slope down')

        df.loc[
            (
                (qtpylib.crossed_above(df['change'], df['buy_offset'])) &
                (df['smooth_ma_slope'] > -0.005) &
                (df['close'] < df[f'reference_ma_{self.reference_ma_length.value}']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'XO above buy_offset|slope up')

        df.loc[
            (
                (df['min'] < df['buy_offset1']) &
                ((df['from_max'] - df['from_max'].shift(8)) < self.bear_from_max.value) &
                (df['close'] < df['buy_offset1']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'Low XB Min < buy_offset1')

        df.loc[
            (
                (qtpylib.crossed_above(df['close'], df['buy_offset2'])) &
                (df['change'] < df['buy_offset']) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'XO above buy_offset2')

        df.loc[
            (
                (qtpylib.crossed_above(df['change'], df[f'smooth_change_{self.smoothing_length.value}'])) &
                (df['close'] < df[f'reference_ma_{self.reference_ma_length.value}']) &
                (df[f'reference_ma_{self.reference_ma_length.value}'] > df[f'reference_ma_{self.reference_ma_length.value}'].shift(1)) &
                (df['close'] > df['buy_offset1']) &
                (df['smooth_ma_slope'] < 0) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'XO above Ref')

        return df


    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r5'])) &
                (df['close'] > (df[f'reference_ma_{self.reference_ma_length.value}'] * self.sell_offset.value)) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R5 - XO')
      
        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r3'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R3 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r2.75'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R2.75 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r2.50'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R2.5 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r2.25'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R2.25 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r2'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R2 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['pivot'])) &
                (df['smooth_ma_slope'] < self.sell_slope.value) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'Pivot - XO')

        return df


   # 12/1000:    140 trades. 135/0/5 Wins/Draws/Losses. Avg profit   2.29%. Median profit   2.30%. Total profit 1744.78452990 USDT ( 174.48%). Avg duration 2 days, 0:12:00 min. Objective: -1744.78453


   #  # Buy hyperspace params:
   #  buy_params = {
   #      "buy_dip_x": 2.0,
   #      "buy_offset1": 0.98,
   #      "buy_offset2": 0.96,
   #      "reference_ma_length": 192,
   #      "smoothing_length": 25,
   #  }

   #  # Sell hyperspace params:
   #  sell_params = {
   #      "filterlength": 34,
   #      "sell_offset": 1.01,
   #      "ts0": 0.011,
   #      "ts1": 0.013,
   #      "ts2": 0.015,
   #      "ts3": 0.033,
   #      "ts4": 0.05,
   #      "ts5": 0.06,
   #      "tsl_target0": 0.047,
   #      "tsl_target1": 0.067,
   #      "tsl_target2": 0.097,
   #      "tsl_target3": 0.13,
   #      "tsl_target4": 0.15,
   #      "tsl_target5": 0.2,
   #  }

   #  # Protection hyperspace params:
   #  protection_params = {
   #      "cooldown_lookback": 5,  # value loaded from strategy
   #      "stop_duration": 5,  # value loaded from strategy
   #      "use_stop_protection": True,  # value loaded from strategy
   #  }

   #  # ROI table:  # value loaded from strategy
   #  minimal_roi = {
   #      "0": 0.1
   #  }

   #  # Stoploss:
   #  stoploss = -0.296  # value loaded from strategy

#    {
#   "strategy_name": "cryptotankV5",
#   "params": {
#     "roi": {
#       "0": 0.1
#     },
#     "stoploss": {
#       "stoploss": -0.345
#     },
#     "trailing": {
#       "trailing_stop": true,
#       "trailing_stop_positive": null,
#       "trailing_stop_positive_offset": 0.0,
#       "trailing_only_offset_is_reached": false
#     },
#     "buy": {
#       "buy_dip_x": 1.4,
#       "buy_offset1": 0.98,
#       "buy_offset2": 0.95,
#       "max_epa": 2,
#       "reference_ma_length": 185,
#       "smoothing_length": 30
#     },
#     "sell": {
#       "filterlength": 35,
#       "sell_offset": 1.04,
#       "ts0": 0.008,
#       "ts1": 0.016,
#       "ts2": 0.028,
#       "ts3": 0.029,
#       "ts4": 0.05,
#       "ts5": 0.05,
#       "tsl_target0": 0.035,
#       "tsl_target1": 0.051,
#       "tsl_target2": 0.083,
#       "tsl_target3": 0.14,
#       "tsl_target4": 0.15,
#       "tsl_target5": 0.4
#     },
#     "protection": {
#       "cooldown_lookback": 5,
#       "stop_duration": 5,
#       "use_stop_protection": true
#     }
#   },
#   "ft_stratparam_v": 1,
#   "export_time": "2023-04-20 13:57:43.790306+00:00"
# }
