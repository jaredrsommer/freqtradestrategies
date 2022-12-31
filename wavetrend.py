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


class wavetrend(IStrategy):


    ### Strategy parameters ###
    exit_profit_only = True ### No selling at a loss
    use_custom_stoploss = True
    trailing_stop = True
    position_adjustment_enable = True
    ignore_roi_if_entry_signal = True
    use_exit_signal = True
    stoploss = -0.40
    startup_candle_count: int = 30
    timeframe = '15m'
    # DCA Parameters
    position_adjustment_enable = True
    max_entry_position_adjustment = 3
    max_dca_multiplier = 7
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }


    ### Hyperoptable parameters ###
    # entry optizimation
    max_epa = CategoricalParameter([-1, 0, 1, 3, 5, 10], default=3, space="buy", optimize=True)

    # protections
    cooldown_lookback = IntParameter(2, 48, default=5, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # indicators
    mfi_length = IntParameter(3,60, default=53, space='buy', optimize=True)

    # trading
    buy_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)
    mfi_buy_slope = IntParameter(-30, 30 , default=0, space='buy', optimize=True)
    sell_rsi = IntParameter(low=50, high=100, default=55, space='sell', optimize=True, load=True)
    mfi_sell_slope = IntParameter(-30, 30 , default=0, space='sell', optimize=True)


    ### entry opt. ###
    @property
    def max_entry_position_adjustment(self):
        return self.max_epa.value


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
                "trade_limit": 4,
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

        if current_profit > -0.025 and trade.nr_of_successful_entries == 1:
            return None

        if current_profit > -0.05 and trade.nr_of_successful_entries == 2:
            return None

        if current_profit > -0.10 and trade.nr_of_successful_entries == 3:
            return None


        # Obtain pair dataframe (just to show how to access it)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries

        # Allow up to 3 additional increasingly larger buys (4 in total)
        # Initial buy is 1x
        # If that falls to -5% profit, we buy more, 
        # If that falls down to -5% again, we buy 1.5x more
        # If that falls once again down to -5%, we buy  more
        # Total stake for this trade would be 1 + 1.5 + 2 + 2.5 = 7x of the initial allowed stake.
        # That is why max_dca_multiplier is 7
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

        if (current_profit > 0.3):
            return 0.05
        elif (current_profit > 0.1):
            return 0.025
        elif (current_profit > 0.06):
            return 0.01
        elif (current_profit > 0.04):
            return 0.01
        elif (current_profit > 0.025):
            return 0.005

        return self.stoploss


    ### NORMAL INDICATORS ###
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['rsi_ma'] = ta.SMA(dataframe['rsi'], timeperiod=7)

        # TEMA - Triple Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        # WaveTrend using OHLC4 or HA close - 3/21
        ap = (0.25 * (dataframe['high'] + dataframe['low'] + dataframe["close"] + dataframe["open"]))
        
        dataframe['esa'] = ta.EMA(ap, timeperiod = 3)
        dataframe['d'] = ta.EMA(abs(ap - dataframe['esa']), timeperiod = 3)
        dataframe['wave_ci'] = (ap-dataframe['esa']) / (0.015 * dataframe['d'])
        dataframe['wave_t1'] = ta.EMA(dataframe['wave_ci'], timeperiod = 21)  
        dataframe['wave_t2'] = ta.SMA(dataframe['wave_t1'], timeperiod = 4)

        # Money Flow Index

        # find mfi optimum length
        for valmfi in self.mfi_length.range:
            dataframe[f'mfi_{valmfi}'] = ta.MFI(dataframe, timeperiod = valmfi)

        dataframe['mfi_slope'] = pta.momentum.slope(dataframe[f'mfi_{valmfi}'])

        return dataframe

    ### ENTRY CONDITIONS ###
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:


        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] >  self.buy_rsi.value) &
                (df['rsi'] < 65) &
                (qtpylib.crossed_above(df['rsi'], df['rsi_ma'])) &
                (df['tema'] > df['tema'].shift(1)) &  # Guard: tema is raising
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['mfi_slope'] >= self.mfi_buy_slope.value)  # Money flow index
            ),
            ['enter_long', 'enter_tag']] = (1, 'TEMA/RSI/MF')

        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] >  self.buy_rsi.value) &
                (df['rsi'] < 55) &
                (qtpylib.crossed_above(df['rsi'], df['rsi_ma'])) &
                (df['tema'] > df['tema'].shift(1)) &  # Guard: tema is raising
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'TEMA/RSI')

        # df.loc[
        #     (
        #         (df['tema'] > df['tema'].shift(1)) &  # Guard: tema is raising
        #         (df['volume'] > 0) &  # Make sure Volume is not 0
        #         (df['mfi_slope'] >= self.mfi_buy_slope.value)  # Money flow index
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'TEMA/MF')


        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] >  self.buy_rsi.value) &
                (df['rsi'] < 60) &
                (df['wave_t1'] > df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
                (df['wave_t1'] > df['wave_t2']) &
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['mfi_slope'] >= self.mfi_buy_slope.value) # Money flow index 
            ),
            ['enter_long', 'enter_tag']] = (1, 'WT/RSI/MF')

        df.loc[
            (
                (df['wave_t1'] > df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
                (qtpylib.crossed_above(df['wave_t1'], df['wave_t2'])) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'WT')

        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] >  self.buy_rsi.value) &
                (df['rsi'] < 55) &
                (qtpylib.crossed_above(df['rsi'], df['rsi_ma'])) &
                (df['wave_t1'] > df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
                (qtpylib.crossed_above(df['wave_t1'], df['wave_t2'])) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'WT/RSI')

        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] >  self.buy_rsi.value) &
                (df['rsi'] < 55) &
                (qtpylib.crossed_above(df['rsi'], df['rsi_ma'])) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'RSI-XO')

        return df


    ### EXIT CONDITIONS ###
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df['rsi'] > self.sell_rsi.value) &
                (df['tema'] < df['tema'].shift(1)) &  # Guard: tema is falling
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),

            ['exit_long', 'exit_tag']] = (1, 'TEMA/RSI')


        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] > self.sell_rsi.value) &
                (df['wave_t1'] < df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
                (df['wave_t1'] < df['wave_t2']) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'WT/RSI')

        df.loc[
            (
                (df['rsi'] > self.sell_rsi.value) &
                (qtpylib.crossed_above(df['rsi_ma'], df['rsi'])) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'RSI-XO')

        df.loc[
            (
                (df['wave_t1'] < df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
                (df['wave_t1'] < df['wave_t2']) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'WT')

        return df
