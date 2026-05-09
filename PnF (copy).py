import logging
from functools import reduce
import datetime
import talib.abstract as ta
import pandas_ta as pta
import logging
import numpy as np
import pandas as pd
import time
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical import qtpylib
from datetime import timedelta, datetime, timezone
from pandas import DataFrame, Series
from technical import qtpylib
from typing import Optional
from freqtrade.strategy.interface import IStrategy
from technical.pivots_points import pivots_points
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_minutes
from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
from scipy.signal import argrelextrema
from typing import Optional, Union, Tuple
from functools import reduce


logger = logging.getLogger(__name__)

class PnF(IStrategy):
    exit_profit_only = True ### No selling at a loss
    use_custom_stoploss = False
    trailing_stop = False
    position_adjustment_enable = False
    ignore_roi_if_entry_signal = True
    max_entry_position_adjustment = 3
    max_dca_multiplier = 3.5
    process_only_new_candles = True
    can_short = False
    use_exit_signal = True
    startup_candle_count = 200
    stoploss = -0.99
    timeframe = '15m'

    locked_stoploss = {}
    minimal_roi = {}

    plot_config = {}
    # DCA
    position_adjustment_enable = True
    # Example specific variables
    max_entry_position_adjustment = 3
    # This number is explained a bit further down
    # max_dca_multiplier = 3
    max_dca_multiplier = DecimalParameter(low=2.0, high=4.0, default=3, decimals=1 ,space='buy', optimize=True, load=True)
    box_size = DecimalParameter(low=0.5, high=10.0, default=1.5, decimals=1 ,space='buy', optimize=True, load=True)
    reversal = DecimalParameter(low=0.6, high=1.5, default=1.1, decimals=1 ,space='sell', optimize=True, load=True)
    filldelay = IntParameter(10, 120, default = 17 ,space='buy', optimize=True, load=True)
    roll = IntParameter(10, 40, default = 17 ,space='buy', optimize=True, load=True)

    # protections
    cooldown_lookback = IntParameter(12, 48, default=12, space="protection", optimize=True)
    stop_duration = IntParameter(12, 48, default=12, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

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


    # This is called when placing the initial order (opening trade)
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        # We need to leave most of the funds for possible further DCA orders
        # This also applies to fixed stakes
        return proposed_stake / self.max_dca_multiplier.value

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs
                              ) -> Union[Optional[float], Tuple[Optional[float], Optional[str]]]:
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be
        increased or decreased.
        This means extra entry or exit orders with additional fees.
        Only called when `position_adjustment_enable` is set to True.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns None

        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Current entry rate (same as current_entry_profit)
        :param current_profit: Current profit (as ratio), calculated based on current_rate 
                               (same as current_entry_profit).
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
                       Optionally, return a tuple with a 2nd element with an order reason
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
                # Only buy when not actively falling price.
        last_candle = dataframe.iloc[-1].squeeze()
        filled_entries = trade.select_filled_orders(trade.entry_side)
        last_fill = (current_time - trade.date_last_filled_utc).seconds / 60 
        count_of_entries = trade.nr_of_successful_entries
        count_of_exits = trade.nr_of_successful_exits
        buy_sig = last_candle['enter_long']
        sell_sig = last_candle['exit_long']
        if sell_sig ==1:
            if current_profit > 0.003 and count_of_entries == 1:
                # Take half of the profit at +5%
                return -(trade.stake_amount / count_of_entries)
            if current_profit > 0.003 and count_of_entries == 2 and count_of_exits == 0:
                # Take half of the profit at +5%
                return -(trade.stake_amount / count_of_entries)
            if current_profit > 0.003 and count_of_entries == 2 and count_of_exits == 1:
                # Take half of the profit at +5%
                return -(trade.stake_amount)
            if current_profit > 0.003 and count_of_entries == 3 and count_of_exits == 0:
                # Take half of the profit at +5%
                return -(trade.stake_amount / count_of_entries)
            if current_profit > 0.003 and count_of_entries == 3 and count_of_exits == 1:
                # Take half of the profit at +5%
                return -(trade.stake_amount)


        if buy_sig == 1 and (last_fill > self.filldelay.value):
            try:
                # This returns first order stake size
                stake_amount = filled_entries[0].stake_amount

                return stake_amount, "Grid Order"
            except Exception as exception:
                return None

            return None

    # def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
    #                    rate: float, time_in_force: str, exit_reason: str,
    #                    current_time: datetime, **kwargs) -> bool:
        
    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #     last_fill = (current_time - trade.date_last_filled_utc).seconds / 60 
    #     count_of_entries = trade.nr_of_successful_entries
    #     count_of_exits = trade.nr_of_successful_exits
    #     buy_sig = last_candle['enter_long']
    #     sell_sig = last_candle['exit_long']

    #     if exit_reason == 'roi' and trade.calc_profit_ratio(rate) < 0.003:
    #         logger.info(f"{trade.pair} ROI is below 0%")
    #         # self.dp.send_msg(f'{trade.pair} ROI is below 0')
    #         return False

    #     if exit_reason == 'partial_exit' and trade.calc_profit_ratio(rate) < 0.005:
    #         logger.info(f"{trade.pair} partial exit is below 0%")
    #         # self.dp.send_msg(f'{trade.pair} partial exit is below 0')
    #         return False

    #     if exit_reason == 'exit_long' and trade.calc_profit_ratio(rate) < 0.01:
    #         logger.info(f"{trade.pair} preparing trade adjustment")
    #         # self.dp.send_msg(f'{trade.pair} partial exit is below 0')
    #         return False

    #     return True


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Apply rolling window operation to the 'OHLC4' column
        rolling_window = dataframe['close'].rolling(self.roll.value) 

        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_value = rolling_window.apply(lambda x: np.ptp(x))

        # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['cycle_move'] = ptp_value / dataframe['close']
        dataframe['cycle_move_mean'] = dataframe['cycle_move'].rolling(self.roll.value).mean()  
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=12)
        dataframe["atrpercent"] = (dataframe["atr"] / dataframe['close']) * 100
        box_size_percentage = self.box_size.value
        # box_size_percentage = dataframe['cycle_move_mean'].iloc[-1] * 100
        reversal_size = box_size_percentage * self.reversal.value
        calculate_pnf_with_percentage(dataframe, box_size_percentage, reversal_size)

        return dataframe


    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df["PnF_Column"] == 1) & 
                (df["PnF_Column"].shift() != 1) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['enter_long', 'enter_tag']] = (1, 'downtrend ends')

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df["PnF_Column"] == -1) & 
                (df["PnF_Column"].shift() != -1) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['exit_long', 'exit_tag']] = (1, 'uptrend ends')

        return df


def calculate_pnf_with_percentage(dataframe, box_size_percentage, reversal_size):
    # Initialize columns
    dataframe["PnF_Column"] = ""
    dataframe["PnF_Direction"] = ""
    dataframe["PnF_Value"] = None
    
    current_column = None
    current_direction = None
    current_value = None
    
    for index, row in dataframe.iterrows():
        close = row["close"]
        box_size = close * (box_size_percentage / 100)  # Calculate box size as a percentage
        
        # Default values for the current row
        dataframe.loc[index, "PnF_Column"] = current_column
        dataframe.loc[index, "PnF_Direction"] = current_direction
        dataframe.loc[index, "PnF_Value"] = current_value
        
        # First row initialization
        if current_value is None:
            current_value = close
            current_column = 1  # Assume first column is an uptrend (X)
            current_direction = "Up"
            dataframe.loc[index, "PnF_Column"] = current_column
            dataframe.loc[index, "PnF_Direction"] = current_direction
            dataframe.loc[index, "PnF_Value"] = current_value
            continue
        
        # Determine the price change
        price_change = close - current_value
        
        if current_direction == "Up":
            # Continue uptrend or reversal
            if price_change >= box_size:
                current_value += box_size
                dataframe.loc[index, "PnF_Column"] = 1
                dataframe.loc[index, "PnF_Direction"] = "Up"
                dataframe.loc[index, "PnF_Value"] = current_value
            elif price_change <= -reversal_size * box_size:
                current_value -= box_size
                current_column = -1
                current_direction = "Down"
                dataframe.loc[index, "PnF_Column"] = current_column
                dataframe.loc[index, "PnF_Direction"] = current_direction
                dataframe.loc[index, "PnF_Value"] = current_value
        elif current_direction == "Down":
            # Continue downtrend or reversal
            if price_change <= -box_size:
                current_value -= box_size
                dataframe.loc[index, "PnF_Column"] = -1
                dataframe.loc[index, "PnF_Direction"] = "Down"
                dataframe.loc[index, "PnF_Value"] = current_value
            elif price_change >= reversal_size * box_size:
                current_value += box_size
                current_column = 1
                current_direction = "Up"
                dataframe.loc[index, "PnF_Column"] = current_column
                dataframe.loc[index, "PnF_Direction"] = current_direction
                dataframe.loc[index, "PnF_Value"] = current_value

    return dataframe
