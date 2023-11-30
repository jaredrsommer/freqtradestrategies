
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

logger = logging.getLogger(__name__)


class ImpulseV1(IStrategy):


    ### Strategy parameters ###
    exit_profit_only = True ### No selling at a loss
    use_custom_stoploss = True
    trailing_stop = True
    position_adjustment_enable = True
    ignore_roi_if_entry_signal = True
    use_exit_signal = True
    stoploss = -0.09
    startup_candle_count: int = 30
    timeframe = '5m'
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
    max_epa = CategoricalParameter([0, 1, 2 ], default=0, space="buy", optimize=True)

    # protections
    cooldown_lookback = IntParameter(2, 48, default=5, space="protection", optimize=True)
    stop_duration = IntParameter(12, 120, default=72, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # indicators
    ma_length = IntParameter(8 ,30, default=15, space='buy', optimize=True)
    #trailing stop loss optimiziation
    tsl_target5 = DecimalParameter(low=0.2, high=0.4, decimals=1, default=0.3, space='sell', optimize=True, load=True)
    ts5 = DecimalParameter(low=0.04, high=0.06, default=0.05, decimals=2,space='sell', optimize=True, load=True)
    tsl_target4 = DecimalParameter(low=0.15, high=0.2, default=0.2, decimals=2, space='sell', optimize=True, load=True)
    ts4 = DecimalParameter(low=0.03, high=0.05, default=0.045, decimals=2,  space='sell', optimize=True, load=True)
    tsl_target3 = DecimalParameter(low=0.10, high=0.15, default=0.15, decimals=2,  space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, decimals=3,  space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.06, high=0.10, default=0.1, decimals=3, space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, decimals=3, space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.03, high=0.06, default=0.06, decimals=3, space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, decimals=3, space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.02, high=0.03, default=0.03, decimals=3, space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.008, high=0.015, default=0.013, decimals=3, space='sell', optimize=True, load=True)


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
                "trade_limit": 1,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False
            })

        return prot


    ### Dollar Cost Averaging ###
    # This is called when placing the initial order (opening trade)
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        # if self.max_epa.value == 0:
        #     self.max_dca_multiplier = 1
        # elif self.max_epa.value == 1:
        #     self.max_dca_multiplier = 2
        # elif self.max_epa.value == 2:
        #     self.max_dca_multiplier = 3
        # else:
        #     self.max_dca_multiplier = 4

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

        if current_profit > -0.05 and trade.nr_of_successful_entries == 1:
            return None

        if current_profit > -0.0585 and trade.nr_of_successful_entries == 2:
            return None

        if current_profit > -0.109 and trade.nr_of_successful_entries == 3:
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
                    return stop5a 
        for stop4 in self.tsl_target4.range:
            if (current_profit > stop4):
                for stop4a in self.ts4.range:
                    return stop4a 
        for stop3 in self.tsl_target3.range:
            if (current_profit > stop3):
                for stop3a in self.ts3.range:
                    return stop3a 
        for stop2 in self.tsl_target2.range:
            if (current_profit > stop2):
                for stop2a in self.ts2.range:
                    return stop2a 
        for stop1 in self.tsl_target1.range:
            if (current_profit > stop1):
                for stop1a in self.ts1.range:
                    return stop1a 
        for stop0 in self.tsl_target0.range:
            if (current_profit > stop0):
                for stop0a in self.ts0.range:
                    return stop0a 


        return self.stoploss


    ### NORMAL INDICATORS ###
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # SMA
        dataframe['200_SMA'] = ta.SMA(dataframe["close"], timeperiod = 200)
        dataframe['30_SMA'] = ta.SMA(dataframe["close"], timeperiod = 30)

        # HLC3
        dataframe['HLC3'] = (dataframe['high'] + dataframe['low'] + dataframe['close'])/3

        for ma_len in self.ma_length.range:
            dataframe[f'hi_{ma_len}'] = ta.SMA(dataframe['high'], timeperiod = ma_len)
            dataframe[f'lo_{ma_len}'] = ta.SMA(dataframe['low'], timeperiod = ma_len)
            dataframe[f'ema1_{ma_len}'] = ta.EMA(dataframe['HLC3'], timeperiod = ma_len)
            dataframe[f'ema2_{ma_len}'] = ta.EMA(dataframe[f'ema1_{ma_len}'], timeperiod = ma_len)
            dataframe[f'd_{ma_len}'] = dataframe[f'ema1_{ma_len}'] - dataframe[f'ema2_{ma_len}']
            dataframe[f'mi_{ma_len}'] = dataframe[f'ema1_{ma_len}'] + dataframe[f'd_{ma_len}']
            dataframe[f'md_{ma_len}'] = np.where(dataframe[f'mi_{ma_len}'] > dataframe[f'hi_{ma_len}'], 
                dataframe[f'mi_{ma_len}'] - dataframe[f'hi_{ma_len}'], 
                np.where(dataframe[f'mi_{ma_len}'] < dataframe[f'lo_{ma_len}'], 
                dataframe[f'mi_{ma_len}'] - dataframe[f'lo_{ma_len}'], 0))
            dataframe[f'sb_{ma_len}'] = ta.SMA(dataframe[f'md_{ma_len}'], timeperiod = 7)
            dataframe[f'sh_{ma_len}'] = dataframe[f'md_{ma_len}'] - dataframe[f'sb_{ma_len}']


        # if self.dp.runmode.value in ('live', 'dry_run'):
        #     ticker = self.dp.ticker(metadata['pair'])
        #     dataframe['last_price'] = ticker['last']
        #     dataframe['volume24h'] = ticker['quoteVolume']
        #     dataframe['vwap'] = ticker['vwap']
        # # Trading pair: {'symbol': 'KAVA/USDT', 'timestamp': 1675451904532, 'datetime': '2023-02-03T19:18:24.532Z', 
        # 'high': 1.0652, 'low': 1.0031, 'bid': 1.0382, 'bidVolume': None, 'ask': 1.0395, 'askVolume': None, 
        # 'vwap': 1.035200204815235, 'open': 1.0558, 'close': 1.0403, 'last': 1.0403, 'previousClose': None, 'change': -0.0155, 
        # 'percentage': -1.46, 'average': 1.04130553, 'baseVolume': 457210.7641, 'quoteVolume': 473304.67664005, 
        # 'info':
        # {'time': 1675451904532, 'symbol': 'KAVA-USDT', 'buy': '1.0382', 'sell': '1.0395', 'changeRate': '-0.0146', 'changePrice': '-0.0155', 'high': '1.0652', 'low': '1.0031', 'vol': '457210.7641', 'volValue': '473304.67664005', 'last': '1.0403', 'averagePrice': '1.04130553', 'takerFeeRate': '0.001', 'makerFeeRate': '0.001', 'takerCoefficient': '1', 'makerCoefficient': '1'}}, self.max_epa.value: 1, self.max_dca_multiplier: 2


        # if (dataframe['30_SMA'].iloc[-1] > dataframe['200_SMA'].iloc[-1] 
        #     and dataframe['30_SMA'].iloc[-1] > dataframe['30_SMA'].iloc[-2]
        #     and dataframe['200_SMA'].iloc[-1] > dataframe['200_SMA'].iloc[-2]).all():
        #     self.max_epa.value = 2
        # elif (dataframe['30_SMA'].iloc[-1] > dataframe['200_SMA'].iloc[-1] 
        #     and dataframe['30_SMA'].iloc[-1] > dataframe['30_SMA'].iloc[-2].all()):
        #     self.max_epa.value = 1
        # elif (dataframe['30_SMA'].iloc[-1] < dataframe['200_SMA'].iloc[-1] 
        #     and dataframe['30_SMA'].iloc[-1] > dataframe['30_SMA'].iloc[-2].all()):
        #     self.max_epa.value = 1
        # else:
        #     self.max_epa.value = 3
        # # print(f"Trading Pair: {dataframe.name}")
        # print(f"self.max_epa.value: {self.max_epa.value}")
        # print(f"self.max_dca_multiplier: {self.max_dca_multiplier}")
        # print(f"Trading pair: {ticker['symbol']}, max_epa: {self.max_epa.value}, dca_multiplier: {self.max_dca_multiplier}")
 
        return dataframe

    ### ENTRY CONDITIONS ###
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:


        df.loc[ # Bullish 0 Cross - Lime Color on TV
            (
                (df[f'sh_{self.ma_length.value}'] > df[f'sh_{self.ma_length.value}'].shift(1)) &  
                (df['HLC3'] > df[f'hi_{self.ma_length.value}']) &
                (qtpylib.crossed_above(df[f'sh_{self.ma_length.value}'], 0)) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'Impulse Bullish Buy')

        df.loc[ # Bearish 0 Cross - Green Color on TV
            (
                (df[f'sh_{self.ma_length.value}'] > df[f'sh_{self.ma_length.value}'].shift(1)) &  
                (df['HLC3'] < df[f'hi_{self.ma_length.value}']) &
                (df['HLC3'] > df[f'mi_{self.ma_length.value}']) &
                (df['HLC3'].shift(1) < df[f'hi_{self.ma_length.value}'].shift(1)) &
                (df['HLC3'].shift(1) > df[f'mi_{self.ma_length.value}'].shift(1)) &
                (qtpylib.crossed_above(df[f'sh_{self.ma_length.value}'], 0)) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'Impulse Bearish Buy')

        return df


    ### EXIT CONDITIONS ###
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[ # Bullish Exit - Orange Color on TV
            (
                (df[f'sh_{self.ma_length.value}'] > df[f'sh_{self.ma_length.value}'].shift(1)) &  
                (df['HLC3'] < df[f'mi_{self.ma_length.value}']) &
                (df['HLC3'] > df[f'lo_{self.ma_length.value}']) &
                (df['HLC3'].shift(1) < df[f'mi_{self.ma_length.value}'].shift(1)) &
                (df['HLC3'].shift(1) > df[f'lo_{self.ma_length.value}'].shift(1)) &
                # (qtpylib.crossed_below(df[f'sh_{self.ma_length.value}'], 0)) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'Impulse Bullish Sell')

        df.loc[ # Bearish Exit - Red Color on TV
            (
                (df[f'sh_{self.ma_length.value}'] > df[f'sh_{self.ma_length.value}'].shift(1)) &  
                (df['HLC3'] < df[f'lo_{self.ma_length.value}']) &
                (df['30_SMA'] > df['200_SMA']) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'Impulse Bearish Sell')

        return df


# # 1HR
# 2023-02-07 19:18:33,386 - freqtrade.optimize.backtesting - INFO - Running backtesting for Strategy ImpulseV1
# 2023-02-07 19:18:33,387 - freqtrade.strategy.hyper - INFO - No params for buy found, using default values.
# 2023-02-07 19:18:33,387 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): buy_rsi = 25
# 2023-02-07 19:18:33,387 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): ma_length = 34
# 2023-02-07 19:18:33,388 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): max_epa = 3
# 2023-02-07 19:18:33,388 - freqtrade.strategy.hyper - INFO - No params for sell found, using default values.
# 2023-02-07 19:18:33,388 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): sell_rsi = 55
# 2023-02-07 19:18:33,388 - freqtrade.strategy.hyper - INFO - No params for protection found, using default values.
# 2023-02-07 19:18:33,388 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): cooldown_lookback = 5
# 2023-02-07 19:18:33,389 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): stop_duration = 72
# 2023-02-07 19:18:33,389 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): use_stop_protection = True
# 2023-02-07 19:18:33,608 - freqtrade.optimize.backtesting - INFO - Backtesting with data from 2023-01-01 00:00:00 up to 2023-02-05 00:00:00 (35 days).
# 2023-02-07 19:18:42,739 - freqtrade.misc - INFO - dumping json to "/home/jared/freq/user_data/backtest_results/backtest-result-2023-02-07_19-18-42.meta.json"
# 2023-02-07 19:18:42,740 - freqtrade.misc - INFO - dumping json to "/home/jared/freq/user_data/backtest_results/backtest-result-2023-02-07_19-18-42.json"
# 2023-02-07 19:18:42,747 - freqtrade.misc - INFO - dumping json to "/home/jared/freq/user_data/backtest_results/.last_result.json"
# Result for strategy ImpulseV1
# ============================================================= BACKTESTING REPORT =============================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# |  KAVA/USDT |         8 |           4.77 |          38.13 |            16.284 |           1.63 |   1 day, 9:38:00 |     8     0     0   100 |
# | THETA/USDT |         8 |           3.17 |          25.39 |            14.718 |           1.47 |  2 days, 1:45:00 |     7     0     1  87.5 |
# |  CSPR/USDT |         8 |           3.13 |          25.03 |            13.812 |           1.38 |  1 day, 14:52:00 |     8     0     0   100 |
# |   FTM/USDT |        12 |           2.71 |          32.52 |            13.694 |           1.37 |         22:00:00 |    11     0     1  91.7 |
# |  AVAX/USDT |        11 |           3.03 |          33.33 |            13.058 |           1.31 |         16:27:00 |    11     0     0   100 |
# |  SCRT/USDT |        10 |           4.01 |          40.08 |            11.366 |           1.14 |  1 day, 12:42:00 |     9     0     1  90.0 |
# |   ETH/USDT |         6 |           5.26 |          31.57 |            11.174 |           1.12 | 2 days, 12:30:00 |     6     0     0   100 |
# |  ATOM/USDT |         7 |           2.62 |          18.36 |            10.494 |           1.05 |  1 day, 19:43:00 |     6     0     1  85.7 |
# |  LUNC/USDT |         5 |           4.32 |          21.60 |             9.573 |           0.96 | 2 days, 23:12:00 |     5     0     0   100 |
# |   DOT/USDT |         7 |           2.92 |          20.46 |             7.714 |           0.77 |  1 day, 10:43:00 |     7     0     0   100 |
# | MATIC/USDT |        10 |           2.04 |          20.45 |             7.475 |           0.75 |   1 day, 0:42:00 |     9     0     1  90.0 |
# |   QNT/USDT |         9 |           2.48 |          22.34 |             7.370 |           0.74 |  1 day, 13:20:00 |     9     0     0   100 |
# |   UNI/USDT |         6 |           2.39 |          14.33 |             7.236 |           0.72 |   1 day, 7:10:00 |     6     0     0   100 |
# |   BTC/USDT |         8 |           2.18 |          17.44 |             7.099 |           0.71 |  1 day, 19:00:00 |     7     0     1  87.5 |
# |  ALGO/USDT |         8 |           2.13 |          17.03 |             6.635 |           0.66 |   1 day, 4:00:00 |     8     0     0   100 |
# |  LINK/USDT |         8 |           2.44 |          19.52 |             6.312 |           0.63 |  1 day, 16:15:00 |     7     0     1  87.5 |
# |  IOTA/USDT |         9 |           1.92 |          17.26 |             6.054 |           0.61 |   1 day, 0:13:00 |     9     0     0   100 |
# |   XTZ/USDT |         6 |           2.64 |          15.85 |             5.715 |           0.57 |   1 day, 6:40:00 |     6     0     0   100 |
# |   XDC/USDT |         6 |           2.34 |          14.03 |             5.113 |           0.51 |  3 days, 5:10:00 |     6     0     0   100 |
# |  NEAR/USDT |         7 |           1.12 |           7.86 |             4.943 |           0.49 |         10:09:00 |     3     0     4  42.9 |
# |   XRP/USDT |         8 |           1.27 |          10.12 |             3.360 |           0.34 |  2 days, 7:38:00 |     7     0     1  87.5 |
# |   ADA/USDT |         5 |           1.93 |           9.65 |             2.448 |           0.24 |         22:00:00 |     5     0     0   100 |
# |      TOTAL |       172 |           2.75 |         472.35 |           191.648 |          19.16 |  1 day, 12:06:00 |   160     0    12  93.0 |
# ================================================================== ENTER TAG STATS ===================================================================
# |                 TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |
# |---------------------+-----------+----------------+----------------+-------------------+----------------+-----------------+-------------------------|
# | Impulse Bullish Buy |       161 |           2.74 |         441.36 |           177.200 |          17.72 | 1 day, 11:15:00 |   149     0    12  92.5 |
# | Impulse Bearish Buy |        11 |           2.82 |          30.99 |            14.449 |           1.44 | 2 days, 0:33:00 |    11     0     0   100 |
# |               TOTAL |       172 |           2.75 |         472.35 |           191.648 |          19.16 | 1 day, 12:06:00 |   160     0    12  93.0 |
# ======================================================== EXIT REASON STATS =========================================================
# |          Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
# |----------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
# |   trailing_stop_loss |      82 |     77     0     5  93.9 |           4.99 |         409.09 |           167.97  |          40.91 |
# | Impulse Bullish Sell |      68 |     68     0     0   100 |           0.88 |          59.62 |            21.634 |           5.96 |
# | Impulse Bearish Sell |      12 |     12     0     0   100 |           0.45 |           5.39 |             1.999 |           0.54 |
# |           force_exit |       8 |      1     0     7  12.5 |          -0.47 |          -3.74 |            -1.613 |          -0.37 |
# |                  roi |       2 |      2     0     0   100 |           1    |           2    |             1.659 |           0.2  |
# ========================================================== LEFT OPEN TRADES REPORT ==========================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+-----------------+-------------------------|
# |  CSPR/USDT |         1 |           0.76 |           0.76 |             0.225 |           0.02 |         5:00:00 |     1     0     0   100 |
# | THETA/USDT |         1 |          -0.13 |          -0.13 |            -0.038 |          -0.00 | 1 day, 11:00:00 |     0     0     1     0 |
# |  NEAR/USDT |         1 |          -0.32 |          -0.32 |            -0.095 |          -0.01 |         9:00:00 |     0     0     1     0 |
# | MATIC/USDT |         1 |          -0.51 |          -0.51 |            -0.150 |          -0.02 |         7:00:00 |     0     0     1     0 |
# |  LINK/USDT |         1 |          -0.38 |          -0.38 |            -0.223 |          -0.02 | 2 days, 5:00:00 |     0     0     1     0 |
# |   XRP/USDT |         1 |          -0.82 |          -0.82 |            -0.235 |          -0.02 | 3 days, 0:00:00 |     0     0     1     0 |
# |   BTC/USDT |         1 |          -0.95 |          -0.95 |            -0.281 |          -0.03 |        10:00:00 |     0     0     1     0 |
# |  ATOM/USDT |         1 |          -1.39 |          -1.39 |            -0.816 |          -0.08 | 1 day, 12:00:00 |     0     0     1     0 |
# |      TOTAL |         8 |          -0.47 |          -3.74 |            -1.613 |          -0.16 |  1 day, 4:22:00 |     1     0     7  12.5 |
# ================== SUMMARY METRICS ==================
# | Metric                      | Value               |
# |-----------------------------+---------------------|
# | Backtesting from            | 2023-01-01 00:00:00 |
# | Backtesting to              | 2023-02-05 00:00:00 |
# | Max open trades             | 10                  |
# |                             |                     |
# | Total/Daily Avg Trades      | 172 / 4.91          |
# | Starting balance            | 1000 USDT           |
# | Final balance               | 1191.648 USDT       |
# | Absolute profit             | 191.648 USDT        |
# | Total profit %              | 19.16%              |
# | CAGR %                      | 522.47%             |
# | Profit factor               | 69.74               |
# | Trades per day              | 4.91                |
# | Avg. daily profit %         | 0.55%               |
# | Avg. stake amount           | 39.968 USDT         |
# | Total trade volume          | 6874.552 USDT       |
# |                             |                     |
# | Best Pair                   | SCRT/USDT 40.08%    |
# | Worst Pair                  | NEAR/USDT 7.86%     |
# | Best trade                  | CSPR/USDT 9.68%     |
# | Worst trade                 | ATOM/USDT -1.39%    |
# | Best day                    | 21.505 USDT         |
# | Worst day                   | -1.613 USDT         |
# | Days win/draw/lose          | 33 / 0 / 1          |
# | Avg. Duration Winners       | 1 day, 13:25:00     |
# | Avg. Duration Loser         | 18:30:00            |
# | Rejected Entry signals      | 3284                |
# | Entry/Exit Timeouts         | 0 / 0               |
# |                             |                     |
# | Min balance                 | 1000.019 USDT       |
# | Max balance                 | 1193.262 USDT       |
# | Max % of account underwater | 0.15%               |
# | Absolute Drawdown (Account) | 0.15%               |
# | Absolute Drawdown           | 1.8 USDT            |
# | Drawdown high               | 193.262 USDT        |
# | Drawdown low                | 191.462 USDT        |
# | Drawdown Start              | 2023-02-04 23:00:00 |
# | Drawdown End                | 2023-02-05 00:00:00 |
# | Market change               | 56.93%              |
# =====================================================



# 2023-02-07 19:20:48,997 - freqtrade.optimize.backtesting - INFO - Running backtesting for Strategy ImpulseV1
# 2023-02-07 19:20:48,997 - freqtrade.strategy.hyper - INFO - No params for buy found, using default values.
# 2023-02-07 19:20:48,998 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): buy_rsi = 25
# 2023-02-07 19:20:48,998 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): ma_length = 34
# 2023-02-07 19:20:48,998 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): max_epa = 1
# 2023-02-07 19:20:48,998 - freqtrade.strategy.hyper - INFO - No params for sell found, using default values.
# 2023-02-07 19:20:48,999 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): sell_rsi = 55
# 2023-02-07 19:20:48,999 - freqtrade.strategy.hyper - INFO - No params for protection found, using default values.
# 2023-02-07 19:20:48,999 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): cooldown_lookback = 5
# 2023-02-07 19:20:48,999 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): stop_duration = 72
# 2023-02-07 19:20:48,999 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): use_stop_protection = True
# 2023-02-07 19:20:49,203 - freqtrade.optimize.backtesting - INFO - Backtesting with data from 2023-01-01 00:00:00 up to 2023-02-05 00:00:00 (35 days).
# 2023-02-07 19:20:58,471 - freqtrade.misc - INFO - dumping json to "/home/jared/freq/user_data/backtest_results/backtest-result-2023-02-07_19-20-58.meta.json"
# 2023-02-07 19:20:58,471 - freqtrade.misc - INFO - dumping json to "/home/jared/freq/user_data/backtest_results/backtest-result-2023-02-07_19-20-58.json"
# 2023-02-07 19:20:58,477 - freqtrade.misc - INFO - dumping json to "/home/jared/freq/user_data/backtest_results/.last_result.json"
# Result for strategy ImpulseV1
# ============================================================= BACKTESTING REPORT =============================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# |  CSPR/USDT |        10 |           4.78 |          47.76 |            49.332 |           4.93 |  1 day, 13:48:00 |    10     0     0   100 |
# |   FTM/USDT |         8 |           4.12 |          32.97 |            27.464 |           2.75 |   1 day, 7:22:00 |     8     0     0   100 |
# |  AVAX/USDT |        11 |           3.01 |          33.13 |            24.728 |           2.47 |         13:05:00 |    11     0     0   100 |
# |   ETH/USDT |         6 |           5.26 |          31.57 |            23.840 |           2.38 | 2 days, 12:30:00 |     6     0     0   100 |
# |  LUNC/USDT |         6 |           4.28 |          25.69 |            22.700 |           2.27 | 2 days, 12:00:00 |     6     0     0   100 |
# |  KAVA/USDT |         8 |           4.06 |          32.47 |            20.833 |           2.08 |  1 day, 13:38:00 |     8     0     0   100 |
# | MATIC/USDT |        10 |           2.10 |          21.01 |            19.934 |           1.99 |  1 day, 12:30:00 |     9     0     1  90.0 |
# |  SCRT/USDT |         6 |           4.70 |          28.18 |            19.933 |           1.99 | 2 days, 14:00:00 |     5     0     1  83.3 |
# | THETA/USDT |         6 |           3.21 |          19.26 |            17.163 |           1.72 | 2 days, 18:30:00 |     6     0     0   100 |
# |   UNI/USDT |         7 |           2.49 |          17.44 |            14.120 |           1.41 |   1 day, 3:51:00 |     7     0     0   100 |
# |  ATOM/USDT |         6 |           2.75 |          16.52 |            13.037 |           1.30 | 2 days, 17:30:00 |     6     0     0   100 |
# |  IOTA/USDT |         9 |           1.92 |          17.26 |            12.901 |           1.29 |   1 day, 0:13:00 |     9     0     0   100 |
# |  LINK/USDT |         8 |           2.36 |          18.87 |            12.352 |           1.24 |  2 days, 5:08:00 |     7     0     1  87.5 |
# |   XTZ/USDT |         4 |           3.89 |          15.55 |            12.060 |           1.21 |  1 day, 22:00:00 |     4     0     0   100 |
# |   DOT/USDT |         6 |           2.14 |          12.87 |            10.296 |           1.03 |   1 day, 4:40:00 |     6     0     0   100 |
# |  ALGO/USDT |         7 |           2.15 |          15.04 |             9.878 |           0.99 |   1 day, 7:43:00 |     7     0     0   100 |
# |   XDC/USDT |         5 |           2.60 |          12.99 |             9.701 |           0.97 |  3 days, 9:36:00 |     5     0     0   100 |
# |   QNT/USDT |         9 |           1.98 |          17.79 |             9.560 |           0.96 |  1 day, 18:33:00 |     8     0     1  88.9 |
# |   BTC/USDT |         7 |           1.78 |          12.45 |             9.501 |           0.95 |  1 day, 21:17:00 |     6     0     1  85.7 |
# |  NEAR/USDT |         8 |           0.98 |           7.87 |             7.447 |           0.74 |         10:38:00 |     4     0     4  50.0 |
# |   ADA/USDT |         5 |           1.93 |           9.65 |             5.015 |           0.50 |         22:00:00 |     5     0     0   100 |
# |   XRP/USDT |         7 |           0.88 |           6.15 |             4.663 |           0.47 |  2 days, 2:26:00 |     6     0     1  85.7 |
# |      TOTAL |       159 |           2.85 |         452.48 |           356.460 |          35.65 |  1 day, 16:14:00 |   149     0    10  93.7 |
# ================================================================== ENTER TAG STATS ===================================================================
# |                 TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |
# |---------------------+-----------+----------------+----------------+-------------------+----------------+-----------------+-------------------------|
# | Impulse Bullish Buy |       150 |           2.89 |         433.05 |           343.277 |          34.33 | 1 day, 15:43:00 |   141     0     9  94.0 |
# | Impulse Bearish Buy |         9 |           2.16 |          19.43 |            13.183 |           1.32 | 2 days, 0:53:00 |     8     0     1  88.9 |
# |               TOTAL |       159 |           2.85 |         452.48 |           356.460 |          35.65 | 1 day, 16:14:00 |   149     0    10  93.7 |
# ======================================================== EXIT REASON STATS =========================================================
# |          Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
# |----------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
# |   trailing_stop_loss |      78 |     74     0     4  94.9 |           5.08 |         396.51 |           312.719 |          39.65 |
# | Impulse Bullish Sell |      61 |     61     0     0   100 |           0.9  |          54.99 |            43.825 |           5.5  |
# | Impulse Bearish Sell |      10 |     10     0     0   100 |           0.51 |           5.12 |             4.078 |           0.51 |
# |           force_exit |       7 |      1     0     6  14.3 |          -1.19 |          -8.31 |            -8.832 |          -0.83 |
# |                  roi |       3 |      3     0     0   100 |           1.4  |           4.19 |             4.669 |           0.42 |
# ========================================================== LEFT OPEN TRADES REPORT ===========================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# |  CSPR/USDT |         1 |           0.76 |           0.76 |             0.511 |           0.05 |          5:00:00 |     1     0     0   100 |
# |  NEAR/USDT |         1 |          -0.32 |          -0.32 |            -0.217 |          -0.02 |          9:00:00 |     0     0     1     0 |
# | MATIC/USDT |         1 |          -0.51 |          -0.51 |            -0.342 |          -0.03 |          7:00:00 |     0     0     1     0 |
# |   XRP/USDT |         1 |          -1.01 |          -1.01 |            -0.677 |          -0.07 |          5:00:00 |     0     0     1     0 |
# |   BTC/USDT |         1 |          -1.82 |          -1.82 |            -1.188 |          -0.12 |  3 days, 1:00:00 |     0     0     1     0 |
# |  LINK/USDT |         1 |          -1.04 |          -1.04 |            -1.332 |          -0.13 | 6 days, 12:00:00 |     0     0     1     0 |
# |   QNT/USDT |         1 |          -4.38 |          -4.38 |            -5.588 |          -0.56 | 6 days, 21:00:00 |     0     0     1     0 |
# |      TOTAL |         7 |          -1.19 |          -8.31 |            -8.832 |          -0.88 | 2 days, 12:00:00 |     1     0     6  14.3 |
# ================== SUMMARY METRICS ==================
# | Metric                      | Value               |
# |-----------------------------+---------------------|
# | Backtesting from            | 2023-01-01 00:00:00 |
# | Backtesting to              | 2023-02-05 00:00:00 |
# | Max open trades             | 10                  |
# |                             |                     |
# | Total/Daily Avg Trades      | 159 / 4.54          |
# | Starting balance            | 1000 USDT           |
# | Final balance               | 1356.46 USDT        |
# | Absolute profit             | 356.46 USDT         |
# | Total profit %              | 35.65%              |
# | CAGR %                      | 2303.34%            |
# | Profit factor               | 33.53               |
# | Trades per day              | 4.54                |
# | Avg. daily profit %         | 1.02%               |
# | Avg. stake amount           | 77.949 USDT         |
# | Total trade volume          | 12393.82 USDT       |
# |                             |                     |
# | Best Pair                   | CSPR/USDT 47.76%    |
# | Worst Pair                  | XRP/USDT 6.15%      |
# | Best trade                  | CSPR/USDT 9.68%     |
# | Worst trade                 | QNT/USDT -4.38%     |
# | Best day                    | 36.115 USDT         |
# | Worst day                   | -8.832 USDT         |
# | Days win/draw/lose          | 31 / 2 / 1          |
# | Avg. Duration Winners       | 1 day, 16:09:00     |
# | Avg. Duration Loser         | 1 day, 17:30:00     |
# | Rejected Entry signals      | 3736                |
# | Entry/Exit Timeouts         | 0 / 0               |
# |                             |                     |
# | Min balance                 | 1000.038 USDT       |
# | Max balance                 | 1365.292 USDT       |
# | Max % of account underwater | 0.65%               |
# | Absolute Drawdown (Account) | 0.65%               |
# | Absolute Drawdown           | 8.832 USDT          |
# | Drawdown high               | 365.292 USDT        |
# | Drawdown low                | 356.46 USDT         |
# | Drawdown Start              | 2023-02-04 23:00:00 |
# | Drawdown End                | 2023-02-05 00:00:00 |
# | Market change               | 56.93%              |
# =====================================================
# 2023-02-07 19:24:10,531 - freqtrade.optimize.backtesting - INFO - Running backtesting for Strategy ImpulseV1
# 2023-02-07 19:24:10,531 - freqtrade.strategy.hyper - INFO - No params for buy found, using default values.
# 2023-02-07 19:24:10,532 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): buy_rsi = 25
# 2023-02-07 19:24:10,532 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): ma_length = 34
# 2023-02-07 19:24:10,532 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): max_epa = 0
# 2023-02-07 19:24:10,533 - freqtrade.strategy.hyper - INFO - No params for sell found, using default values.
# 2023-02-07 19:24:10,533 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): sell_rsi = 55
# 2023-02-07 19:24:10,533 - freqtrade.strategy.hyper - INFO - No params for protection found, using default values.
# 2023-02-07 19:24:10,533 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): cooldown_lookback = 5
# 2023-02-07 19:24:10,533 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): stop_duration = 72
# 2023-02-07 19:24:10,534 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): use_stop_protection = True
# 2023-02-07 19:24:10,751 - freqtrade.optimize.backtesting - INFO - Backtesting with data from 2023-01-01 00:00:00 up to 2023-02-05 00:00:00 (35 days).
# 2023-02-07 19:24:19,001 - freqtrade.misc - INFO - dumping json to "/home/jared/freq/user_data/backtest_results/backtest-result-2023-02-07_19-24-19.meta.json"
# 2023-02-07 19:24:19,001 - freqtrade.misc - INFO - dumping json to "/home/jared/freq/user_data/backtest_results/backtest-result-2023-02-07_19-24-19.json"
# 2023-02-07 19:24:19,011 - freqtrade.misc - INFO - dumping json to "/home/jared/freq/user_data/backtest_results/.last_result.json"
# Result for strategy ImpulseV1
# ============================================================= BACKTESTING REPORT =============================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# |  CSPR/USDT |         8 |           4.68 |          37.45 |            46.950 |           4.69 |  1 day, 22:30:00 |     8     0     0   100 |
# |  AVAX/USDT |         9 |           3.96 |          35.66 |            41.050 |           4.10 |   1 day, 1:33:00 |     9     0     0   100 |
# |  KAVA/USDT |         7 |           4.39 |          30.76 |            35.217 |           3.52 |  1 day, 23:17:00 |     6     0     1  85.7 |
# |   ETH/USDT |         6 |           4.88 |          29.25 |            33.381 |           3.34 | 2 days, 12:50:00 |     6     0     0   100 |
# |  SCRT/USDT |         5 |           5.46 |          27.29 |            30.290 |           3.03 |  3 days, 7:24:00 |     4     0     1  80.0 |
# |   FTM/USDT |         6 |           3.98 |          23.90 |            29.147 |           2.91 |  1 day, 21:00:00 |     6     0     0   100 |
# |   DOT/USDT |         6 |           3.04 |          18.25 |            23.434 |           2.34 |  2 days, 2:20:00 |     6     0     0   100 |
# |  ATOM/USDT |         6 |           3.02 |          18.10 |            22.201 |           2.22 | 2 days, 23:20:00 |     6     0     0   100 |
# |  LUNC/USDT |         4 |           4.37 |          17.49 |            21.670 |           2.17 |  1 day, 23:15:00 |     4     0     0   100 |
# | THETA/USDT |         5 |           3.37 |          16.85 |            21.016 |           2.10 | 3 days, 18:36:00 |     5     0     0   100 |
# |   UNI/USDT |         5 |           2.94 |          14.72 |            17.907 |           1.79 |  1 day, 17:48:00 |     5     0     0   100 |
# |  IOTA/USDT |         6 |           2.40 |          14.38 |            16.916 |           1.69 |  1 day, 16:50:00 |     6     0     0   100 |
# |  LINK/USDT |         7 |           2.16 |          15.11 |            16.142 |           1.61 | 3 days, 16:17:00 |     6     0     1  85.7 |
# | MATIC/USDT |         8 |           1.58 |          12.63 |            15.556 |           1.56 |  1 day, 23:00:00 |     7     0     1  87.5 |
# |  ALGO/USDT |         6 |           2.27 |          13.63 |            15.318 |           1.53 |  1 day, 13:20:00 |     6     0     0   100 |
# |   BTC/USDT |         7 |           1.80 |          12.60 |            14.117 |           1.41 |  1 day, 21:34:00 |     6     0     1  85.7 |
# |   QNT/USDT |         7 |           1.97 |          13.76 |            13.948 |           1.39 | 2 days, 18:43:00 |     6     0     1  85.7 |
# |   XTZ/USDT |         3 |           2.89 |           8.66 |            12.116 |           1.21 |         15:20:00 |     3     0     0   100 |
# |   ADA/USDT |         6 |           1.83 |          10.99 |            11.983 |           1.20 |  1 day, 12:10:00 |     6     0     0   100 |
# |   XRP/USDT |         5 |           1.83 |           9.14 |            10.789 |           1.08 |  2 days, 6:24:00 |     5     0     0   100 |
# |   XDC/USDT |         5 |           1.75 |           8.75 |            10.244 |           1.02 | 2 days, 10:12:00 |     4     0     1  80.0 |
# |  NEAR/USDT |         5 |          -0.61 |          -3.03 |            -3.568 |          -0.36 |          2:24:00 |     1     0     4  20.0 |
# |      TOTAL |       132 |           2.93 |         386.36 |           455.823 |          45.58 |  2 days, 2:15:00 |   121     0    11  91.7 |
# ================================================================== ENTER TAG STATS ===================================================================
# |                 TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |
# |---------------------+-----------+----------------+----------------+-------------------+----------------+-----------------+-------------------------|
# | Impulse Bullish Buy |       125 |           2.88 |         359.43 |           423.360 |          42.34 | 2 days, 3:03:00 |   114     0    11  91.2 |
# | Impulse Bearish Buy |         7 |           3.85 |          26.93 |            32.463 |           3.25 | 1 day, 12:00:00 |     7     0     0   100 |
# |               TOTAL |       132 |           2.93 |         386.36 |           455.823 |          45.58 | 2 days, 2:15:00 |   121     0    11  91.7 |
# ======================================================== EXIT REASON STATS =========================================================
# |          Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
# |----------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
# |   trailing_stop_loss |      73 |     69     0     4  94.5 |           4.88 |         356.54 |           426.788 |          35.65 |
# | Impulse Bullish Sell |      45 |     45     0     0   100 |           1.01 |          45.39 |            51.121 |           4.54 |
# |           force_exit |       7 |      0     0     7     0 |          -2.98 |         -20.89 |           -28.48  |          -2.09 |
# | Impulse Bearish Sell |       4 |      4     0     0   100 |           0.58 |           2.32 |             2.716 |           0.23 |
# |                  roi |       3 |      3     0     0   100 |           1    |           3    |             3.678 |           0.3  |
# ========================================================== LEFT OPEN TRADES REPORT ===========================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# |  NEAR/USDT |         1 |          -0.32 |          -0.32 |            -0.470 |          -0.05 |          9:00:00 |     0     0     1     0 |
# | MATIC/USDT |         1 |          -0.51 |          -0.51 |            -0.743 |          -0.07 |          7:00:00 |     0     0     1     0 |
# |   XDC/USDT |         1 |          -0.88 |          -0.88 |            -1.261 |          -0.13 |  2 days, 7:00:00 |     0     0     1     0 |
# |   BTC/USDT |         1 |          -1.82 |          -1.82 |            -2.568 |          -0.26 |  3 days, 1:00:00 |     0     0     1     0 |
# |  LINK/USDT |         1 |          -2.19 |          -2.19 |            -3.058 |          -0.31 | 6 days, 12:00:00 |     0     0     1     0 |
# |  KAVA/USDT |         1 |          -4.79 |          -4.79 |            -6.498 |          -0.65 |  9 days, 5:00:00 |     0     0     1     0 |
# |   QNT/USDT |         1 |         -10.37 |         -10.37 |           -13.882 |          -1.39 | 10 days, 4:00:00 |     0     0     1     0 |
# |      TOTAL |         7 |          -2.98 |         -20.89 |           -28.480 |          -2.85 | 4 days, 13:17:00 |     0     0     7     0 |
# ================== SUMMARY METRICS ==================
# | Metric                      | Value               |
# |-----------------------------+---------------------|
# | Backtesting from            | 2023-01-01 00:00:00 |
# | Backtesting to              | 2023-02-05 00:00:00 |
# | Max open trades             | 10                  |
# |                             |                     |
# | Total/Daily Avg Trades      | 132 / 3.77          |
# | Starting balance            | 1000 USDT           |
# | Final balance               | 1455.823 USDT       |
# | Absolute profit             | 455.823 USDT        |
# | Total profit %              | 45.58%              |
# | CAGR %                      | 4923.31%            |
# | Profit factor               | 15.32               |
# | Trades per day              | 3.77                |
# | Avg. daily profit %         | 1.30%               |
# | Avg. stake amount           | 117.822 USDT        |
# | Total trade volume          | 15552.564 USDT      |
# |                             |                     |
# | Best Pair                   | CSPR/USDT 37.45%    |
# | Worst Pair                  | NEAR/USDT -3.03%    |
# | Best trade                  | SCRT/USDT 9.63%     |
# | Worst trade                 | QNT/USDT -10.37%    |
# | Best day                    | 71.439 USDT         |
# | Worst day                   | -28.48 USDT         |
# | Days win/draw/lose          | 29 / 4 / 1          |
# | Avg. Duration Winners       | 2 days, 0:30:00     |
# | Avg. Duration Loser         | 2 days, 21:33:00    |
# | Rejected Entry signals      | 4757                |
# | Entry/Exit Timeouts         | 0 / 0               |
# |                             |                     |
# | Min balance                 | 1000.077 USDT       |
# | Max balance                 | 1484.304 USDT       |
# | Max % of account underwater | 1.92%               |
# | Absolute Drawdown (Account) | 1.92%               |
# | Absolute Drawdown           | 28.48 USDT          |
# | Drawdown high               | 484.304 USDT        |
# | Drawdown low                | 455.823 USDT        |
# | Drawdown Start              | 2023-02-04 23:00:00 |
# | Drawdown End                | 2023-02-05 00:00:00 |
# | Market change               | 56.93%              |
# =====================================================
# 15M
# 2023-02-07 19:24:10,531 - freqtrade.optimize.backtesting - INFO - Running backtesting for Strategy ImpulseV1
# 2023-02-07 19:24:10,531 - freqtrade.strategy.hyper - INFO - No params for buy found, using default values.
# 2023-02-07 19:24:10,532 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): buy_rsi = 25
# 2023-02-07 19:24:10,532 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): ma_length = 34
# 2023-02-07 19:24:10,532 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): max_epa = 0
# 2023-02-07 19:24:10,533 - freqtrade.strategy.hyper - INFO - No params for sell found, using default values.
# 2023-02-07 19:24:10,533 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): sell_rsi = 55
# 2023-02-07 19:24:10,533 - freqtrade.strategy.hyper - INFO - No params for protection found, using default values.
# 2023-02-07 19:24:10,533 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): cooldown_lookback = 5
# 2023-02-07 19:24:10,533 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): stop_duration = 72
# 2023-02-07 19:24:10,534 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): use_stop_protection = True
# 2023-02-07 19:24:10,751 - freqtrade.optimize.backtesting - INFO - Backtesting with data from 2023-01-01 00:00:00 up to 2023-02-05 00:00:00 (35 days).
# 2023-02-07 19:24:19,001 - freqtrade.misc - INFO - dumping json to "/home/jared/freq/user_data/backtest_results/backtest-result-2023-02-07_19-24-19.meta.json"
# 2023-02-07 19:24:19,001 - freqtrade.misc - INFO - dumping json to "/home/jared/freq/user_data/backtest_results/backtest-result-2023-02-07_19-24-19.json"
# 2023-02-07 19:24:19,011 - freqtrade.misc - INFO - dumping json to "/home/jared/freq/user_data/backtest_results/.last_result.json"
# Result for strategy ImpulseV1
# ============================================================= BACKTESTING REPORT =============================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# |  CSPR/USDT |         8 |           4.68 |          37.45 |            46.950 |           4.69 |  1 day, 22:30:00 |     8     0     0   100 |
# |  AVAX/USDT |         9 |           3.96 |          35.66 |            41.050 |           4.10 |   1 day, 1:33:00 |     9     0     0   100 |
# |  KAVA/USDT |         7 |           4.39 |          30.76 |            35.217 |           3.52 |  1 day, 23:17:00 |     6     0     1  85.7 |
# |   ETH/USDT |         6 |           4.88 |          29.25 |            33.381 |           3.34 | 2 days, 12:50:00 |     6     0     0   100 |
# |  SCRT/USDT |         5 |           5.46 |          27.29 |            30.290 |           3.03 |  3 days, 7:24:00 |     4     0     1  80.0 |
# |   FTM/USDT |         6 |           3.98 |          23.90 |            29.147 |           2.91 |  1 day, 21:00:00 |     6     0     0   100 |
# |   DOT/USDT |         6 |           3.04 |          18.25 |            23.434 |           2.34 |  2 days, 2:20:00 |     6     0     0   100 |
# |  ATOM/USDT |         6 |           3.02 |          18.10 |            22.201 |           2.22 | 2 days, 23:20:00 |     6     0     0   100 |
# |  LUNC/USDT |         4 |           4.37 |          17.49 |            21.670 |           2.17 |  1 day, 23:15:00 |     4     0     0   100 |
# | THETA/USDT |         5 |           3.37 |          16.85 |            21.016 |           2.10 | 3 days, 18:36:00 |     5     0     0   100 |
# |   UNI/USDT |         5 |           2.94 |          14.72 |            17.907 |           1.79 |  1 day, 17:48:00 |     5     0     0   100 |
# |  IOTA/USDT |         6 |           2.40 |          14.38 |            16.916 |           1.69 |  1 day, 16:50:00 |     6     0     0   100 |
# |  LINK/USDT |         7 |           2.16 |          15.11 |            16.142 |           1.61 | 3 days, 16:17:00 |     6     0     1  85.7 |
# | MATIC/USDT |         8 |           1.58 |          12.63 |            15.556 |           1.56 |  1 day, 23:00:00 |     7     0     1  87.5 |
# |  ALGO/USDT |         6 |           2.27 |          13.63 |            15.318 |           1.53 |  1 day, 13:20:00 |     6     0     0   100 |
# |   BTC/USDT |         7 |           1.80 |          12.60 |            14.117 |           1.41 |  1 day, 21:34:00 |     6     0     1  85.7 |
# |   QNT/USDT |         7 |           1.97 |          13.76 |            13.948 |           1.39 | 2 days, 18:43:00 |     6     0     1  85.7 |
# |   XTZ/USDT |         3 |           2.89 |           8.66 |            12.116 |           1.21 |         15:20:00 |     3     0     0   100 |
# |   ADA/USDT |         6 |           1.83 |          10.99 |            11.983 |           1.20 |  1 day, 12:10:00 |     6     0     0   100 |
# |   XRP/USDT |         5 |           1.83 |           9.14 |            10.789 |           1.08 |  2 days, 6:24:00 |     5     0     0   100 |
# |   XDC/USDT |         5 |           1.75 |           8.75 |            10.244 |           1.02 | 2 days, 10:12:00 |     4     0     1  80.0 |
# |  NEAR/USDT |         5 |          -0.61 |          -3.03 |            -3.568 |          -0.36 |          2:24:00 |     1     0     4  20.0 |
# |      TOTAL |       132 |           2.93 |         386.36 |           455.823 |          45.58 |  2 days, 2:15:00 |   121     0    11  91.7 |
# ================================================================== ENTER TAG STATS ===================================================================
# |                 TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |
# |---------------------+-----------+----------------+----------------+-------------------+----------------+-----------------+-------------------------|
# | Impulse Bullish Buy |       125 |           2.88 |         359.43 |           423.360 |          42.34 | 2 days, 3:03:00 |   114     0    11  91.2 |
# | Impulse Bearish Buy |         7 |           3.85 |          26.93 |            32.463 |           3.25 | 1 day, 12:00:00 |     7     0     0   100 |
# |               TOTAL |       132 |           2.93 |         386.36 |           455.823 |          45.58 | 2 days, 2:15:00 |   121     0    11  91.7 |
# ======================================================== EXIT REASON STATS =========================================================
# |          Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
# |----------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
# |   trailing_stop_loss |      73 |     69     0     4  94.5 |           4.88 |         356.54 |           426.788 |          35.65 |
# | Impulse Bullish Sell |      45 |     45     0     0   100 |           1.01 |          45.39 |            51.121 |           4.54 |
# |           force_exit |       7 |      0     0     7     0 |          -2.98 |         -20.89 |           -28.48  |          -2.09 |
# | Impulse Bearish Sell |       4 |      4     0     0   100 |           0.58 |           2.32 |             2.716 |           0.23 |
# |                  roi |       3 |      3     0     0   100 |           1    |           3    |             3.678 |           0.3  |
# ========================================================== LEFT OPEN TRADES REPORT ===========================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# |  NEAR/USDT |         1 |          -0.32 |          -0.32 |            -0.470 |          -0.05 |          9:00:00 |     0     0     1     0 |
# | MATIC/USDT |         1 |          -0.51 |          -0.51 |            -0.743 |          -0.07 |          7:00:00 |     0     0     1     0 |
# |   XDC/USDT |         1 |          -0.88 |          -0.88 |            -1.261 |          -0.13 |  2 days, 7:00:00 |     0     0     1     0 |
# |   BTC/USDT |         1 |          -1.82 |          -1.82 |            -2.568 |          -0.26 |  3 days, 1:00:00 |     0     0     1     0 |
# |  LINK/USDT |         1 |          -2.19 |          -2.19 |            -3.058 |          -0.31 | 6 days, 12:00:00 |     0     0     1     0 |
# |  KAVA/USDT |         1 |          -4.79 |          -4.79 |            -6.498 |          -0.65 |  9 days, 5:00:00 |     0     0     1     0 |
# |   QNT/USDT |         1 |         -10.37 |         -10.37 |           -13.882 |          -1.39 | 10 days, 4:00:00 |     0     0     1     0 |
# |      TOTAL |         7 |          -2.98 |         -20.89 |           -28.480 |          -2.85 | 4 days, 13:17:00 |     0     0     7     0 |
# ================== SUMMARY METRICS ==================
# | Metric                      | Value               |
# |-----------------------------+---------------------|
# | Backtesting from            | 2023-01-01 00:00:00 |
# | Backtesting to              | 2023-02-05 00:00:00 |
# | Max open trades             | 10                  |
# |                             |                     |
# | Total/Daily Avg Trades      | 132 / 3.77          |
# | Starting balance            | 1000 USDT           |
# | Final balance               | 1455.823 USDT       |
# | Absolute profit             | 455.823 USDT        |
# | Total profit %              | 45.58%              |
# | CAGR %                      | 4923.31%            |
# | Profit factor               | 15.32               |
# | Trades per day              | 3.77                |
# | Avg. daily profit %         | 1.30%               |
# | Avg. stake amount           | 117.822 USDT        |
# | Total trade volume          | 15552.564 USDT      |
# |                             |                     |
# | Best Pair                   | CSPR/USDT 37.45%    |
# | Worst Pair                  | NEAR/USDT -3.03%    |
# | Best trade                  | SCRT/USDT 9.63%     |
# | Worst trade                 | QNT/USDT -10.37%    |
# | Best day                    | 71.439 USDT         |
# | Worst day                   | -28.48 USDT         |
# | Days win/draw/lose          | 29 / 4 / 1          |
# | Avg. Duration Winners       | 2 days, 0:30:00     |
# | Avg. Duration Loser         | 2 days, 21:33:00    |
# | Rejected Entry signals      | 4757                |
# | Entry/Exit Timeouts         | 0 / 0               |
# |                             |                     |
# | Min balance                 | 1000.077 USDT       |
# | Max balance                 | 1484.304 USDT       |
# | Max % of account underwater | 1.92%               |
# | Absolute Drawdown (Account) | 1.92%               |
# | Absolute Drawdown           | 28.48 USDT          |
# | Drawdown high               | 484.304 USDT        |
# | Drawdown low                | 455.823 USDT        |
# | Drawdown Start              | 2023-02-04 23:00:00 |
# | Drawdown End                | 2023-02-05 00:00:00 |
# | Market change               | 56.93%              |
# =====================================================
# 2 trades unlimited stake
    # 46/1000:    172 trades. 140/27/5 Wins/Draws/Losses. Avg profit   1.35%. Median profit   1.90%. Total profit 1998.82983205 USDT ( 199.88%). Avg duration 23:59:00 min. Objective: -1998.82983


    # # Buy hyperspace params:
    # buy_params = {
    #     "ma_length": 23,  # value loaded from strategy
    #     "max_epa": 2,  # value loaded from strategy
    # }

    # # Sell hyperspace params:
    # sell_params = {
    #     "ts0": 0.008,  # value loaded from strategy
    #     "ts1": 0.01,  # value loaded from strategy
    #     "ts2": 0.018,  # value loaded from strategy
    #     "ts3": 0.026,  # value loaded from strategy
    #     "ts4": 0.05,  # value loaded from strategy
    #     "ts5": 0.04,  # value loaded from strategy
    #     "tsl_target0": 0.03,  # value loaded from strategy
    #     "tsl_target1": 0.06,  # value loaded from strategy
    #     "tsl_target2": 0.087,  # value loaded from strategy
    #     "tsl_target3": 0.14,  # value loaded from strategy
    #     "tsl_target4": 0.18,  # value loaded from strategy
    #     "tsl_target5": 0.2,  # value loaded from strategy
    # }

    # # Protection hyperspace params:
    # protection_params = {
    #     "cooldown_lookback": 47,
    #     "stop_duration": 82,
    #     "use_stop_protection": False,
    # }

    # # ROI table:
    # minimal_roi = {
    #     "0": 0.304,
    #     "367": 0.111,
    #     "949": 0.045,
    #     "1365": 0
    # }

    # # Stoploss:
    # stoploss = -0.304

    # # Trailing stop:
    # trailing_stop = True  # value loaded from strategy
    # trailing_stop_positive = None  # value loaded from strategy
    # trailing_stop_positive_offset = 0.0  # value loaded from strategy
    # trailing_only_offset_is_reached = False  # value loaded from strategy
