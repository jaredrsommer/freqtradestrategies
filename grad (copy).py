
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
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)


class grad(IStrategy):

    '''
          ______   __          __              __    __   ______   __    __        __     __    __             ______            
     /      \ /  |       _/  |            /  |  /  | /      \ /  \  /  |      /  |   /  |  /  |           /      \           
    /$$$$$$  |$$ |____  / $$ |    _______ $$ | /$$/ /$$$$$$  |$$  \ $$ |     _$$ |_  $$ |  $$ |  _______ /$$$$$$  |  _______ 
    $$ |  $$/ $$      \ $$$$ |   /       |$$ |/$$/  $$ ___$$ |$$$  \$$ |    / $$   | $$ |__$$ | /       |$$$  \$$ | /       |
    $$ |      $$$$$$$  |  $$ |  /$$$$$$$/ $$  $$<     /   $$< $$$$  $$ |    $$$$$$/  $$    $$ |/$$$$$$$/ $$$$  $$ |/$$$$$$$/ 
    $$ |   __ $$ |  $$ |  $$ |  $$ |      $$$$$  \   _$$$$$  |$$ $$ $$ |      $$ | __$$$$$$$$ |$$ |      $$ $$ $$ |$$      \ 
    $$ \__/  |$$ |  $$ | _$$ |_ $$ \_____ $$ |$$  \ /  \__$$ |$$ |$$$$ |      $$ |/  |     $$ |$$ \_____ $$ \$$$$ | $$$$$$  |
    $$    $$/ $$ |  $$ |/ $$   |$$       |$$ | $$  |$$    $$/ $$ | $$$ |______$$  $$/      $$ |$$       |$$   $$$/ /     $$/ 
     $$$$$$/  $$/   $$/ $$$$$$/  $$$$$$$/ $$/   $$/  $$$$$$/  $$/   $$//      |$$$$/       $$/  $$$$$$$/  $$$$$$/  $$$$$$$/  
                                                                       $$$$$$/                                               
                                                                                                                             
    '''          

    ### Strategy parameters ###
    timeframe = '1h'
    exit_profit_only = False ### No selling at a loss
    ignore_roi_if_entry_signal = True
    process_only_new_candles = True
    can_short = False
    use_exit_signal = True
    startup_candle_count = 20
    use_custom_stoploss = False
    trailing_stop = False

    locked_stoploss = {}
    minimal_roi = {}
    

    # Stoploss:
    stoploss = -0.35  # Fail Safe Default do not hyper opt Stoploss Space

    # Trailing stop:
    trailing_stop = True  # value loaded from strategy
    trailing_stop_positive = 0.015 # value loaded from strategy
    trailing_stop_positive_offset = 0.15 # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy

    # CooldownPeriod 
    cooldown_lookback = IntParameter(0, 12, default=5, space="protection", optimize=True, load=True)
    
    # StoplossGuard    
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True, load=True)
    stop_duration = IntParameter(6, 40, default=39, space="protection", optimize=True, load=True)
    stop_protection_only_per_pair = BooleanParameter(default=True, space="protection", optimize=True, load=True)
    stop_protection_only_per_side = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    stop_protection_trade_limit = IntParameter(1, 10, default=4, space="protection", optimize=True, load=True)
    stop_protection_required_profit = DecimalParameter(-0.10, 0.01, default=-0.04, decimals=2, space="protection", optimize=True, load=True)

    # LowProfitPairs    
    use_lowprofit_protection = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    lowprofit_protection_lookback = IntParameter(1, 24, default=10, space="protection", optimize=True, load=True)
    lowprofit_trade_limit = IntParameter(1, 10, default=6, space="protection", optimize=True, load=True)
    lowprofit_stop_duration = IntParameter(1, 70, default=65, space="protection", optimize=True, load=True)
    lowprofit_required_profit = DecimalParameter(-0.10, 0.00, default=-0.04, decimals=2, space="protection", optimize=True, load=True)
    lowprofit_only_per_pair = BooleanParameter(default=True, space="protection", optimize=True, load=True)

    # MaxDrawdown    
    use_maxdrawdown_protection = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    maxdrawdown_protection_lookback = IntParameter(1, 10, default=6, space="protection", optimize=True, load=True)
    maxdrawdown_trade_limit = IntParameter(1, 20, default=10, space="protection", optimize=True, load=True)
    maxdrawdown_stop_duration = IntParameter(1, 40, default=6, space="protection", optimize=True, load=True)
    maxdrawdown_allowed_drawdown = DecimalParameter(-0.10, 0.00, default=-0.04, decimals=2, space="protection", optimize=True, load=True)

    # Custom Entry
    increment = DecimalParameter(low=1.0005, high=1.002, default=1.001, decimals=4 ,space='buy', optimize=True, load=True)
    entryX = DecimalParameter(low=0.98, high=1.00, default=1.00, decimals=3 ,space='buy', optimize=True, load=True)
    last_entry_price = None
    
    # Gradient Smoothing
    smooth = IntParameter(2, 10, default=3, space="buy", optimize=True, load=True)

    # Position Management
    base_trades = IntParameter(4, 10, default=6, space="protection", optimize=True)
    stepSize = IntParameter(1, 4, default=2, space="protection", optimize=True)

    def custom_entry_price(self, pair: str, trade: Optional['Trade'], current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)

        entry_price = ((dataframe['close'].iat[-1] + dataframe['open'].iat[-1] + dataframe['low'].iat[-1] + proposed_rate) / 4) * self.entryX.value
        if (self.dp.runmode.value in ('live', 'dry_run')):
            logger.info(f"{pair} Using Entry Price: {entry_price} | close: {dataframe['close'].iat[-1]} open: {dataframe['open'].iat[-1]} proposed_rate: {proposed_rate}")
            self.dp.send_msg(f"{pair} Using Entry Price: {entry_price} | close: {dataframe['close'].iat[-1]} open: {dataframe['open'].iat[-1]} proposed_rate: {proposed_rate}")

        # Check if there is a stored last entry price and if it matches the proposed entry price
        if self.last_entry_price is not None and abs(entry_price - self.last_entry_price) < 0.0001:  # Tolerance for floating-point comparison
            entry_price *= self.increment.value # Increment by 0.2%%
            if (self.dp.runmode.value in ('live', 'dry_run')):
                logger.info(f"{pair} Incremented entry price: {entry_price} based on previous entry price : {self.last_entry_price}.")

        # Update the last entry price
        self.last_entry_price = entry_price

        # Define balance thresholds and stake multipliers
        # EDIT These to match your funding level <--- !!!!!
        balance = self.wallets.get_total_stake_amount()
        max_balance = 40000
        if balance > 10000:
            max_open_trades_at_max_balance = self.base_trades.value + int((max_balance - 10000) / 10000) * self.stepSize.value
            self.config['max_open_trades'] = min(self.base_trades.value + int((balance - 10000) / 10000) * self.stepSize.value, max_open_trades_at_max_balance)
        else:
            self.config['max_open_trades'] = self.base_trades.value

        return entry_price

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

    ### NORMAL INDICATORS ###
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        heikinashi = qtpylib.heikinashi(dataframe)

        # Heikin-Ashi calculations
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        ta
        # Gradient of ha_close
        dataframe['ha_close_gradient'] = np.gradient(dataframe['ha_close'])

        # Calculate directional volume based on ha_close and its gradient
        dataframe['ha_close_change'] = dataframe['ha_close'].diff()
        dataframe['directional_volume'] = dataframe['volume'] * np.sign(dataframe['ha_close_change'] + dataframe['ha_close_gradient'])
        dataframe['dir_volume_grad'] = np.gradient(dataframe['directional_volume'])
        dataframe['volume_grad'] = np.gradient(dataframe['volume'])

        # Include gradients for trend analysis
        dataframe['frac_diff'] = np.gradient(ta.SMA(dataframe['ha_close'], self.smooth.value))
        dataframe['fracdiff'] = dataframe['ha_close'] + dataframe['frac_diff']
        dataframe['zero'] = 0

        return dataframe



    ### ENTRY CONDITIONS ###
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:


        df.loc[
            (
                (df['frac_diff'] > 0) &
                (df['frac_diff'].shift() < 0) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'Gradient X Up')

        return df


    ### EXIT CONDITIONS ###
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df['frac_diff'] < 0) &
                (df['frac_diff'].shift() > 0) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'Gradient X Down')


        return df




