# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import numpy as np
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
import technical.indicators as ftt
from technical import qtpylib
import freqtrade.vendor.qtpylib.indicators as qtpylib
import logging
import pandas as pd
import pandas_ta as pta
import datetime
from datetime import datetime, timedelta, timezone
from typing import Optional
import talib.abstract as ta
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter, merge_informative_pair,
                                stoploss_from_open)


# @Rallipanos # changes by IcHiAT

# Buy hyperspace params:
buy_params = {
        "base_nb_candles_buy": 12,
        "ewo_high": 3.147,
        "ewo_low": -17.145,
        "low_offset": 0.987,
        "rsi_buy": 57,
    }

# Sell hyperspace params:
sell_params = {
        "base_nb_candles_sell": 22,
        "high_offset": 1.008,
        "high_offset_2": 1.016,
    }

def EWO(dataframe, ema_length=5, ema2_length=3):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif



class turbov8(IStrategy):
    INTERFACE_VERSION = 2

    ### Strategy parameters ###
    exit_profit_only = True ### No selling at a loss
    use_custom_stoploss = True
    trailing_stop = False # True
    ignore_roi_if_entry_signal = True
    use_exit_signal = True
    startup_candle_count = 400
    run_from_bear = 0
    # Stoploss:
    stoploss = -0.20

    # DCA Parameters
    position_adjustment_enable = True
    max_entry_position_adjustment = 3
    max_dca_multiplier = 1.5

    ## Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    # ROI table:
    minimal_roi = {
        "0": 0.99,
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    informative_timeframe = '1h'
    
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


    ### Hyperoptable parameters ###
    # entry optizimation
    max_epa = CategoricalParameter([0, 1, 2, 3], default=3, space="buy", optimize=True)

    # protections
    cooldown_lookback = IntParameter(24, 48, default=46, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # SMAOffset
    base_nb_candles_buy = IntParameter(5, 80, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(0.95, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(0.99, 1.5, default=sell_params['high_offset_2'], space='sell', optimize=True)   

    # Protection
    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -8.0, default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)
    ### BTC 
    fast_1h = IntParameter(3, 8, default=5, space='buy', optimize=True)
    slow_1h = IntParameter(25, 40, default=30, space='buy', optimize=True)


    # dca level optimization
    dca1 = DecimalParameter(low=0.01, high=0.03, decimals=2, default=0.02, space='buy', optimize=True, load=True)
    dca2 = DecimalParameter(low=0.03, high=0.05, decimals=2, default=0.04, space='buy', optimize=True, load=True)
    dca3 = DecimalParameter(low=0.05, high=0.07, decimals=2, default=0.06, space='buy', optimize=True, load=True)

    #trailing stop loss optimiziation
    tsl_target5 = DecimalParameter(low=0.3, high=0.4, decimals=1, default=0.3, space='sell', optimize=True, load=True)
    ts5 = DecimalParameter(low=0.04, high=0.06, default=0.05, space='sell', optimize=True, load=True)
    tsl_target4 = DecimalParameter(low=0.18, high=0.3, default=0.2, space='sell', optimize=True, load=True)
    ts4 = DecimalParameter(low=0.03, high=0.05, default=0.045, space='sell', optimize=True, load=True)
    tsl_target3 = DecimalParameter(low=0.12, high=0.18, default=0.15, space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.07, high=0.12, default=0.1, space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.04, high=0.07, default=0.06, space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.02, high=0.05, default=0.03, space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.008, high=0.015, default=0.013, space='sell', optimize=True, load=True)


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

        # Grid Buy down every 2% with a small amount of capital
        try:
            # This returns first order stake size
            stake_amount = filled_entries[0].cost
            # This then calculates current safety order size
            if count_of_entries == 1: 
                stake_amount = stake_amount * 0.166
            elif count_of_entries == 2:
                stake_amount = stake_amount * 0.166
            elif count_of_entries == 3:
                stake_amount = stake_amount * 0.166
            else:
                stake_amount = stake_amount

            return stake_amount
        except Exception as exception:
            return None

        return None

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        for stop5 in self.tsl_target5.range:
            if (current_profit > stop5):
                for stop5a in self.ts5.range:
                    self.dp.send_msg(f'*** FEEDBACK *** Pair {pair} profit {current_profit} - lvl5 {stop5a} stoploss activated')
                    return stop5a 
        for stop4 in self.tsl_target4.range:
            if (current_profit > stop4):
                for stop4a in self.ts4.range:
                    self.dp.send_msg(f'*** FEEDBACK *** Pair {pair} profit {current_profit} - lvl4 {stop4a} stoploss activated')
                    return stop4a 
        for stop3 in self.tsl_target3.range:
            if (current_profit > stop3):
                for stop3a in self.ts3.range:
                    self.dp.send_msg(f'*** FEEDBACK *** Pair {pair} profit {current_profit} - lvl3 {stop3a} stoploss activated')
                    return stop3a 
        for stop2 in self.tsl_target2.range:
            if (current_profit > stop2):
                for stop2a in self.ts2.range:
                    self.dp.send_msg(f'*** FEEDBACK *** Pair {pair} profit {current_profit} - lvl2 {stop2a} stoploss activated')
                    return stop2a 
        for stop1 in self.tsl_target1.range:
            if (current_profit > stop1):
                for stop1a in self.ts1.range:
                    self.dp.send_msg(f'*** FEEDBACK *** Pair {pair} profit {current_profit} - lvl1 {stop1a} stoploss activated')
                    return stop1a 
        for stop0 in self.tsl_target0.range:
            if (current_profit > stop0):
                for stop0a in self.ts0.range:
                    self.dp.send_msg(f'*** FEEDBACK *** Pair {pair} profit {current_profit} - lvl0 {stop0a} stoploss activated')
                    return stop0a 

        return self.stoploss


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # BTC SMA 5/30
        if self.dp:
            inf_tf = '1h'
            informative = self.dp.get_pair_dataframe(pair=f"BTC/USDT", timeframe=inf_tf)
            for fast in self.fast_1h.range:
                informative[f'sma_{fast}'] = ta.SMA(informative["close"], timeperiod = fast)

            for slow in self.slow_1h.range:
                informative[f'sma_{slow}'] = ta.SMA(informative["close"], timeperiod = slow)

        
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        
        # SMA
        dataframe['200_SMA'] = ta.SMA(dataframe["close"], timeperiod = 200)
        dataframe['30_SMA'] = ta.SMA(dataframe["close"], timeperiod = 30)
        dataframe['5_SMA'] = ta.SMA(dataframe["close"], timeperiod = 5)
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)


        ### Changed this part ###
        if (informative[f'sma_{fast}'].iloc[-1] > informative[f'sma_{slow}'].iloc[-1]):
            dataframe['Run_From_Bear'] = 0
            if (self.run_from_bear == 1):
                self.dp.send_msg(f"MARKET STATUS: Bear is gone! Time to wake up from hibernations...", always_send=True)
                self.run_from_bear = 0
        elif (informative[f'sma_{fast}'].iloc[-1] < informative[f'sma_{slow}'].iloc[-1]
            and informative[f'sma_{fast}'].iloc[-1] < informative[f'sma_{fast}'].iloc[-2]).all():
            dataframe['Run_From_Bear'] = 1
            if (self.run_from_bear == 0):
                self.dp.send_msg(f"MARKET STATUS: Bear sighted! Selling off and going into hibernation...", always_send=True)
                self.run_from_bear = 1    
        elif (informative[f'sma_{fast}'].iloc[-1] < informative[f'sma_{slow}'].iloc[-1]
            and informative[f'sma_{fast}'].iloc[-1] > informative[f'sma_{fast}'].iloc[-2]).all():
            dataframe['Run_From_Bear'] = -1
            self.dp.send_msg(f"MARKET STATUS: Bear getting sleepy! w3n m00n?...", always_send=True)
        else:
            dataframe['Run_From_Bear'] = -1
            self.dp.send_msg(f"MARKET STATUS: Bear Lurking! Grab the Lube, This could hurt...", always_send=True)

             
        if (dataframe['30_SMA'].iloc[-1] > dataframe['200_SMA'].iloc[-1] 
            and dataframe['30_SMA'].iloc[-1] > dataframe['30_SMA'].iloc[-2]
            and dataframe['200_SMA'].iloc[-1] > dataframe['200_SMA'].iloc[-2]).all():
            self.max_epa.value = 1
        elif (dataframe['30_SMA'].iloc[-1] > dataframe['200_SMA'].iloc[-1] 
            and dataframe['30_SMA'].iloc[-1] > dataframe['30_SMA'].iloc[-2].all()):
            self.max_epa.value = 1
        elif (dataframe['30_SMA'].iloc[-1] < dataframe['200_SMA'].iloc[-1] 
            and dataframe['30_SMA'].iloc[-1] > dataframe['30_SMA'].iloc[-2].all()):
            self.max_epa.value = 2
        else:
            self.max_epa.value = 2

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['rsi_fast'] <35) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['Run_From_Bear'] == 0) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO above high')


        dataframe.loc[
            (
                (dataframe['rsi_fast'] <35)&
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['Run_From_Bear'] == 0) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO below low')

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        dataframe.loc[
            (
                (dataframe['close'] > dataframe['hma_50'])&
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) &
                (dataframe['rsi']>50)&
                (dataframe['volume'] > 0)&
                (dataframe['rsi_fast']>dataframe['rsi_slow'])

            ),
            ['exit_long', 'exit_tag']] = (1, 'Close > Offset Hi')

        dataframe.loc[
            (
                (dataframe['close']<dataframe['hma_50'])&
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0)&
                (dataframe['rsi_fast']>dataframe['rsi_slow'])

            ),
            ['exit_long', 'exit_tag']] = (1, 'Close > Offset Lo')

        dataframe.loc[
            (
                (dataframe['Run_From_Bear'] == 1) &
                (dataframe['volume'] > 0)
            ),
            ['exit_long', 'exit_tag']] = (1, 'fucking bearzzz')

        return dataframe