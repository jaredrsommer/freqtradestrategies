
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

class cryptotankV2(IStrategy):

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
    timeframe ='5m'

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30 

    ### HYPER-OPT PARAMETERS ###

    # protections
    cooldown_lookback = IntParameter(2, 48, default=5, space="protection", optimize=True)
    stop_duration = IntParameter(1, 36, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # entry optimization
    # max_epa = CategoricalParameter([0, 1], default=0, space="buy", optimize=True)

    # indicators
    buy_rsi_length = IntParameter(14, 16, default=14, space="buy",optimize=True)
    buy_rsi_ma_length = IntParameter(10, 14, default=14, space="buy",optimize=True)
    reference_ma_length = IntParameter(180, 200, default=200, space="buy" ,optimize=True)
    smoothing_length = IntParameter(25, 50, default=35, space="buy", optimize=True)
    filterlength = IntParameter(low=8, high=40, default=35, space='sell', optimize=True)
    buy_offset1 = DecimalParameter(low=0.90, high=0.99, decimals=2, default=0.96, space='buy', optimize=True, load=True)
    buy_offset2 = DecimalParameter(low=0.90, high=0.96, decimals=2, default=0.94, space='buy', optimize=True, load=True)
    sell_offset = DecimalParameter(low=1.01, high=1.10, decimals=2, default=1.08, space='sell', optimize=True, load=True)

    #trailing stop loss optimiziation
    tsl_target5 = DecimalParameter(low=0.15, high=0.20, decimals=2, default=0.3, space='sell', optimize=True, load=True)
    ts5 = DecimalParameter(low=0.04, high=0.06, default=0.05, decimals=2,space='sell', optimize=True, load=True)
    tsl_target4 = DecimalParameter(low=0.10, high=0.15, default=0.2, decimals=2, space='sell', optimize=True, load=True)
    ts4 = DecimalParameter(low=0.03, high=0.05, default=0.045, decimals=2,  space='sell', optimize=True, load=True)
    tsl_target3 = DecimalParameter(low=0.08, high=0.10, default=0.15, decimals=2,  space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, decimals=3,  space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.06, high=0.8, default=0.1, decimals=3, space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, decimals=3, space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.04, high=0.06, default=0.06, decimals=3, space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, decimals=3, space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.02, high=0.04, default=0.03, decimals=3, space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.005, high=0.015, default=0.013, decimals=3, space='sell', optimize=True, load=True)

    ### buying 
    # rsi
    buy_rsi_upper = IntParameter(55, 65, default=55, space="buy", optimize=True)
    buy_rsi_lower = IntParameter(20, 35, default=35, space="buy", optimize=True)
    buy_rsi_ma_slope = DecimalParameter(-0.01, 0.10, default=0, decimals=2, space="buy", optimize=True)
    buy_ma_slope = DecimalParameter(-0.10, 0.10, default=0, decimals=2, space="buy", optimize=True)
    buy_lower = DecimalParameter(-10, 0, default=-2, decimals=1, space="buy", optimize=True)
    buy_rsi_dip = IntParameter(30, 40, default=35, space="buy", optimize=True)
    buy_dip_x = DecimalParameter(0.5, 2.0, default=1.5, decimals=1, space="buy", optimize=True)

    ### selling
    # # rsi
    # sell_rsi_upper = IntParameter(80, 100, default=80, space="sell", optimize=True)
    # sell_rsi_lower = IntParameter(50, 79, default=55, space="sell", optimize=True)
    # sell_ma_slope = IntParameter(-10, 10, default=0, space="sell", optimize=True)

# {
#   "main_plot": {
#     "enter_tag": {
#       "color": "#6fc96b",
#       "type": "line"
#     },
#     "exit_tag": {
#       "color": "#d51a44",
#       "type": "line"
#     },
#     "reference_ma_198": {
#       "color": "#dacd51",
#       "type": "line"
#     },
#     "buy_offset2": {
#       "color": "#289531",
#       "type": "line"
#     },
#     "buy_offset1": {
#       "color": "#289531",
#       "type": "line"
#     },
#     "sell_offset": {
#       "color": "#e93c22",
#       "type": "line"
#     }
#   },
#   "subplots": {
#     "RSI": {
#       "buy_rsi_18": {
#         "color": "#9262a5",
#         "type": "line"
#       },
#       "buy_rsi_ma_5": {
#         "color": "#bddc16",
#         "type": "line"
#       }
#     },
#     "Change": {
#       "change": {
#         "color": "#ec612c",
#         "type": "line"
#       },
#       "smooth_change_11": {
#         "color": "#357cf9",
#         "type": "line"
#       }
#     },
#     "SLOPE": {
#       "smooth_ma_slope": {
#         "color": "#5fc631",
#         "type": "line"
#       }
#     }
#   }
# }

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

        if current_profit > -0.05 and trade.nr_of_successful_entries == 2:
            return None

        if current_profit > -0.6 and trade.nr_of_successful_entries == 3:
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
        pivots = pivots_points.pivots_points(dataframe, timeperiod=50, levels=5)
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

        # RSI
        # Calculate all rsi_buy values
        for valb in self.buy_rsi_length.range:
            dataframe[f'buy_rsi_{valb}'] = ta.RSI(dataframe, timeperiod=valb)

        # Calculate all rsi_buy ma values
        for valbm in self.buy_rsi_ma_length.range:
            dataframe[f'buy_rsi_ma_{valbm}'] = ta.SMA(dataframe[f'buy_rsi_{valb}'], timeperiod=valbm)

        dataframe['rsi_ma_slope'] = pta.momentum.slope(dataframe[f'buy_rsi_ma_{valbm}'])


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

        dataframe['smooth_ma_slope'] = pta.momentum.slope(dataframe[f'smooth_change_{valsma}'])



        return dataframe


    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        conditions = []



        # df.loc[
        #     (
        #         (qtpylib.crossed_above(df['change'], df[f'smooth_change_{self.smoothing_length.value}'])) &
        #         (df[f'buy_rsi_{self.buy_rsi_length.value}'] < self.buy_rsi_upper.value) &
        #         (df['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'XO ')

        df.loc[
            (
                (qtpylib.crossed_above(df['change'], df[f'smooth_change_{self.smoothing_length.value}'])) &
                (df['smooth_ma_slope'] > self.buy_ma_slope.value) &
                (df['close'] < (df[f'reference_ma_{self.reference_ma_length.value}'] * self.sell_offset.value)) &
                (df[f'smooth_change_{self.smoothing_length.value}'] > 0) &
                (df[f'rsi_ma_slope'] > self.buy_rsi_ma_slope.value) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'XO Above 0')

        #WIP
        df.loc[
            (
                (qtpylib.crossed_above(df['change'], df[f'smooth_change_{self.smoothing_length.value}'])) &
                (df['close'] < (df[f'reference_ma_{self.reference_ma_length.value}'])) &
                (df[f'smooth_change_{self.smoothing_length.value}'] < (self.buy_lower.value * 2)) &
                (df[f'buy_rsi_{self.buy_rsi_length.value}'] < self.buy_rsi_dip.value) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'XO Below Lower Entry * 2')

        # df.loc[
        #     (
        #         (qtpylib.crossed_above(df['change'], df[f'smooth_change_{self.smoothing_length.value}'])) &
        #         (df['smooth_ma_slope'] > self.buy_ma_slope.value) &
        #         (df[f'smooth_change_{self.smoothing_length.value}'] < self.buy_lower.value) &
        #         (df['close'] < df[f'reference_ma_{self.reference_ma_length.value}'] * self.buy_offset1.value) &
        #         (df['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'XO Below Lower and OFFSET1')


        # df.loc[
        #     (
        #         (qtpylib.crossed_above(df['change'], df[f'smooth_change_{self.smoothing_length.value}'])) &
        #         (df['smooth_ma_slope'] > self.buy_ma_slope.value) &
        #         (df[f'smooth_change_{self.smoothing_length.value}'] < self.buy_lower.value) &
        #         (df['close'] < df[f'reference_ma_{self.reference_ma_length.value}'] * self.buy_offset2.value) &
        #         (df['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'XO Below Lower and OFFSET2')


        df.loc[
            (
                (qtpylib.crossed_above(df[f'buy_rsi_{self.buy_rsi_length.value}'], df[f'buy_rsi_ma_{self.buy_rsi_ma_length.value}'])) &
                (df[f'buy_rsi_{self.buy_rsi_length.value}'] < self.buy_rsi_dip.value) &
                (df['change'] < (df[f'smooth_change_{self.smoothing_length.value}'] - self.buy_dip_x.value)) &
                (df['change'] < self.buy_lower.value) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'RSI XO | Change < Smooth + Dip < 0')

        df.loc[
            (
                (qtpylib.crossed_above(df['close'], df['r3'])) & # r3..
                (df['change'] > df[f'smooth_change_{self.smoothing_length.value}']) &
                (df[f'buy_rsi_{self.buy_rsi_length.value}'] < self.buy_rsi_upper.value) &
                (df['close'] > df[f'reference_ma_{self.reference_ma_length.value}']) &
                (df['change'] > 0) &
                (df['change'] > df['change'].shift(1)) &
                (df[f'buy_rsi_{self.buy_rsi_length.value}'] > df[f'buy_rsi_{self.buy_rsi_length.value}'].shift(1)) &
                (df[f'buy_rsi_ma_{self.buy_rsi_ma_length.value}'] > df[f'buy_rsi_ma_{self.buy_rsi_ma_length.value}'].shift(1)) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'MEGA PUMP')

        # df.loc[
        #     (
        #         (df[f'buy_rsi_{self.buy_rsi_length.value}'] < self.buy_rsi_dip.value) &
        #         (df['close'] < (df[f'reference_ma_{self.reference_ma_length.value}'] * self.buy_offset2.value)) &
        #         (df['change'] < self.buy_lower.value) &
        #         (df['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'MEGA DUMP')

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
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R3 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r2.75'])) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R2.75 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r2.50'])) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R2.5 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r2.25'])) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R2.25 - XO')

        df.loc[
            (
                (qtpylib.crossed_below(df[f'zema_{self.filterlength.value}'], df['r2'])) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'R2 - XO')

        return df


# {
#   "strategy_name": "cryptotankV2",
#   "params": {
#     "trailing": {
#       "trailing_stop": true,
#       "trailing_stop_positive": null,
#       "trailing_stop_positive_offset": 0.0,
#       "trailing_only_offset_is_reached": false
#     },
#     "buy": {
#       "buy_ma_slope": -10,
#       "buy_offset1": 0.92,
#       "buy_offset2": 0.92,
#       "buy_rsi_length": 5,
#       "buy_rsi_lower": 33,
#       "buy_rsi_ma_length": 24,
#       "buy_rsi_upper": 64,
#       "filterlength": 30,
#       "max_epa": 1,
#       "reference_ma_length": 194,
#       "smoothing_length": 6
#     },
#     "sell": {
#       "sell_ma_slope": -8,
#       "sell_offset": 1.05,
#       "sell_rsi_lower": 77,
#       "sell_rsi_upper": 98,
#       "ts0": 0.01,
#       "ts1": 0.015,
#       "ts2": 0.026,
#       "ts3": 0.03,
#       "ts4": 0.05,
#       "ts5": 0.04,
#       "tsl_target0": 0.042,
#       "tsl_target1": 0.073,
#       "tsl_target2": 0.087,
#       "tsl_target3": 0.14,
#       "tsl_target4": 0.2,
#       "tsl_target5": 0.3
#     },
#     "protection": {
#       "cooldown_lookback": 27,
#       "stop_duration": 32,
#       "use_stop_protection": true
#     },
#     "roi": {
#       "0": 0.45299999999999996,
#       "1268": 0.34199999999999997,
#       "7051": 0.13,
#       "21126": 0
#     },
#     "stoploss": {
#       "stoploss": -0.261
#     }
#   },
#   "ft_stratparam_v": 1,
#   "export_time": "2023-04-11 21:35:29.779280+00:00"
# }
# ================== SUMMARY METRICS ==================
# | Metric                      | Value               |
# |-----------------------------+---------------------|
# | Backtesting from            | 2023-01-01 00:00:00 |
# | Backtesting to              | 2023-04-17 00:00:00 |
# | Max open trades             | 3                   |
# |                             |                     |
# | Total/Daily Avg Trades      | 164 / 1.55          |
# | Starting balance            | 1000 USDT           |
# | Final balance               | 3817.038 USDT       |
# | Absolute profit             | 2817.038 USDT       |
# | Total profit %              | 281.70%             |
# | CAGR %                      | 9971.97%            |
# | Profit factor               | 4.70                |
# | Trades per day              | 1.55                |
# | Avg. daily profit %         | 2.66%               |
# | Avg. stake amount           | 722.729 USDT        |
# | Total trade volume          | 118527.56 USDT      |
# |                             |                     |
# | Best Pair                   | AKT/USDT 54.28%     |
# | Worst Pair                  | ATOM/USDT -18.78%   |
# | Best trade                  | RNDR/USDT 9.99%     |
# | Worst trade                 | ATOM/USDT -23.16%   |
# | Best day                    | 184.782 USDT        |
# | Worst day                   | -237.53 USDT        |
# | Days win/draw/lose          | 66 / 34 / 4         |
# | Avg. Duration Winners       | 1 day, 5:50:00      |
# | Avg. Duration Loser         | 14 days, 19:38:00   |
# | Rejected Entry signals      | 270163              |
# | Entry/Exit Timeouts         | 0 / 0               |
# |                             |                     |
# | Min balance                 | 1000.968 USDT       |
# | Max balance                 | 3876.021 USDT       |
# | Max % of account underwater | 15.71%              |
# | Absolute Drawdown (Account) | 15.71%              |
# | Absolute Drawdown           | 534.177 USDT        |
# | Drawdown high               | 2401.028 USDT       |
# | Drawdown low                | 1866.851 USDT       |
# | Drawdown Start              | 2023-03-01 13:45:00 |
# | Drawdown End                | 2023-03-11 15:45:00 |
# | Market change               | 114.36%             |
# =====================================================
