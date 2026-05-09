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
from typing import Optional
from functools import reduce


logger = logging.getLogger(__name__)

class TankAiRevival(IStrategy):


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


    exit_profit_only = True ### No selling at a loss
    use_custom_stoploss = True
    trailing_stop = False
    position_adjustment_enable = True
    ignore_roi_if_entry_signal = True
    position_adjustment_enable = True
    max_entry_position_adjustment = 2
    max_dca_multiplier = 2.5
    process_only_new_candles = True
    can_short = False
    use_exit_signal = True
    startup_candle_count = 200
    stoploss = -0.99
    timeframe = '30m'

    locked_stoploss = {}
    minimal_roi = {}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "extrema": {
                "&s-extrema": {
                    "color": "#f53580",
                    "type": "line"
                },
                "&s-minima_sort_threshold": {
                    "color": "#4ae747",
                    "type": "line"
                },
                "&s-maxima_sort_threshold": {
                    "color": "#5b5e4b",
                    "type": "line"
                }
            },
            "min_max": {
                "maxima": {
                    "color": "#a29db9",
                    "type": "line"
                },
                "minima": {
                    "color": "#ac7fc",
                    "type": "line"
                },
                "maxima_check": {
                    "color": "#a29db9",
                    "type": "line"
                },
                "minima_check": {
                    "color": "#ac7fc",
                    "type": "line"
                }
            }
        }
    }

    # DCA
    position_adjustment_enable = True
    max_epa = IntParameter(0, 6, default = 1 ,space='buy', optimize=True, load=True) # of additional buys.
    max_dca_multiplier = DecimalParameter(low=1.0, high=15, default=3, decimals=1 ,space='buy', optimize=True, load=True)
    use_static = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    filldelay = IntParameter(100, 300, default = 100 ,space='buy', optimize=True, load=True)
    max_entry_position_adjustment = max_epa.value

    # Risk Level Stake size adjustments based on Location to  
    stake0 = DecimalParameter(low=0.7, high=1.5, default=1.0, decimals=1 ,space='buy', optimize=True, load=True)
    stake1 = DecimalParameter(low=0.7, high=1.5, default=1.0, decimals=1 ,space='buy', optimize=True, load=True)
    stake2 = DecimalParameter(low=0.7, high=1.5, default=1.0, decimals=1 ,space='buy', optimize=True, load=True)
    stake3 = DecimalParameter(low=0.7, high=1.5, default=1.0, decimals=1 ,space='buy', optimize=True, load=True)

    ### Custom Functions
    # Threshold and Limits
    u_window_size = IntParameter(100, 160, default=100, space='buy', optimize=True)
    l_window_size = IntParameter(5, 40, default=20, space='buy', optimize=True)
    dc_x = DecimalParameter(low=3.0, high=5.0, default=4.5, decimals=1 ,space='buy', optimize=True, load=True)

    # Custom Entry
    increment = DecimalParameter(low=1.0005, high=1.002, default=1.001, decimals=4 ,space='buy', optimize=True, load=True)
    last_entry_price = None

    # protections
    cooldown_lookback = IntParameter(24, 48, default=12, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # negative stoploss
    use_stop1 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    use_stop2 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    use_stop3 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    use_stop4 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    # roi
    time0 = IntParameter(low=1440, high=2600, default=1440, space='sell', optimize=True, load=True)
    time1 = IntParameter(low=1440, high=2600, default=2000, space='sell', optimize=True, load=True)
    time2 = IntParameter(low=2600, high=4000, default=3200, space='sell', optimize=True, load=True)
    time3 = IntParameter(low=2500, high=5000, default=4500, space='sell', optimize=True, load=True)  
    # block specific exit timer
    time4 = IntParameter(low=240, high=480, default=240, space='sell', optimize=True, load=True)  

    # Logic Selection
    use0 = BooleanParameter(default=False, space="sell", optimize=True, load=True)
    use1 = BooleanParameter(default=False, space="sell", optimize=True, load=True)
    use2 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    # use3 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use4 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use5 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use6 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use7 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use8 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    # use9 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    # use10 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    # use11 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    # use12 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    # use13 = BooleanParameter(default=True, space="sell", optimize=True, load=True)

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

    ### Dollar Cost Averaging ###
    ### Custom Functions ###
    # This is called when placing the initial order (opening trade)
    # Let unlimited stakes leave funds open for DCA orders
    def custom_stake_amount(
            self, pair: str, current_time: datetime, current_rate: float,
            proposed_stake: float, min_stake: Optional[float], max_stake: float,
            leverage: float, entry_tag: Optional[str], side: str,
            **kwargs
        ) -> float:
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        EP0 = current_candle['lower_envelope'] 
        EP1 = current_candle['lower_envelope_h0'] 
        EP2 = current_candle['lower_envelope_h1'] 
        EP3 = current_candle['lower_envelope_h2']

        # Define balance thresholds and stake multipliers
        # EDIT These to match your funding level <--- !!!!!
        balance = self.wallets.get_total_stake_amount()
        if balance < 3000:
            balance_multiplier = 0.9
            self.config['max_open_trades'] = 5
        elif balance < 7500:
            balance_multiplier = 0.95  # Increase stake by 20%
            self.config['max_open_trades'] = 6
        elif balance < 10000:
            balance_multiplier = 1.0  # No increase
            self.config['max_open_trades'] = 7
        elif balance < 15000:
            balance_multiplier = 1.05  # Increase stake by 20%
            self.config['max_open_trades'] = 8
        elif balance < 20000:
            balance_multiplier = 1.1  # Increase stake by 50%
            self.config['max_open_trades'] = 10
        else:
            balance_multiplier = 1.0  # Double stake for larger balances

        # Static or calculated staking
        if self.use_static.value == 'True':
            return (balance / self.config["max_open_trades"]) * balance_multiplier

        num_tr = self.config['max_open_trades']
        # Adjust stake based on DCA, balance multiplier, and Location
        if current_rate < EP0: # Longest envelope typically the least frequent to be crossed
            calculated_stake = (proposed_stake / self.max_dca_multiplier.value) * balance_multiplier
        elif EP0 <= current_rate < EP1:
            calculated_stake = (proposed_stake / self.max_dca_multiplier.value) * self.stake3.value * balance_multiplier
        elif EP1 <= current_rate < EP2:
            calculated_stake = (proposed_stake / self.max_dca_multiplier.value) * self.stake2.value * balance_multiplier
        elif EP2 <= current_rate < EP3: # Shortest envelope typically the most frequent to be crossed
            calculated_stake = (proposed_stake / self.max_dca_multiplier.value) * self.stake1.value * balance_multiplier
        else:
            calculated_stake = (proposed_stake / self.max_dca_multiplier.value) * self.stake0.value * balance_multiplier
        logger.info(f'{pair} using {calculated_stake} instead of {proposed_stake} | Trade Slots: {num_tr} Balance X:{balance_multiplier}')
        self.dp.send_msg(f'{pair} using {calculated_stake} instead of {proposed_stake} | Trade Slots: {num_tr} Balance X:{balance_multiplier}')

        '''ICP/USDT using 558.49607684277 instead of 1116.99215368554 | Trade Slots: 15 Balance X:1.5'''

        return min(max(calculated_stake, min_stake or 0), max_stake)



    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        trade_duration = (current_time - trade.open_date_utc).seconds / 60
        last_fill = (current_time - trade.date_last_filled_utc).seconds / 60 

        current_candle = dataframe.iloc[-1].squeeze()

        TP0 = current_candle['h2_move_mean'] 
        TP1 = current_candle['h1_move_mean'] 
        TP2 = current_candle['h0_move_mean'] 
        TP3 = current_candle['cycle_move_mean']
        display_profit = current_profit * 100

        tp0 = TP0 * 100
        tp1 = TP1 * 100
        tp2 = TP2 * 100
        tp3 = TP3 * 100
        dca_str = ''
        dcaDisp = display_profit+tp1
        if display_profit < 0:
            dca_str = f'| DCA Dist: {dcaDisp:.3}% | Last Fill: {last_fill:.5}m"' 

        if current_candle['enter_long'] is not None:
            signal = current_candle['enter_long']

        if current_profit is not None:
            logger.info(f"{trade.pair} - Current Profit: {display_profit:.3}% # of Entries: {trade.nr_of_successful_entries} {dca_str}")
            logger.info(f"{trade.pair} - TP0: {tp0:.3}% | TP1: {tp1:.3}% | TP2: {tp2:.3}% | TP3: {tp3:.3}%")
        # Take Profit if m00n
        if current_profit > TP2 and trade.nr_of_successful_exits == 0:
            # Take quarter of the profit at next fib%%
            return -(trade.stake_amount / 2)
        if current_profit > TP3 and trade.nr_of_successful_exits == 1:
            # Take half of the profit at last fib%%
            return -(trade.stake_amount / 2)
        if current_profit > (TP3 * 1.5) and trade.nr_of_successful_exits == 2:
            # Take half of the profit at last fib%%
            return -(trade.stake_amount / 2)
        if current_profit > (TP3 * 2.0) and trade.nr_of_successful_exits == 3:
            # Take profit at last fib%%
            return -(trade.stake_amount)
        # Take Profit Early if DCA was used 
        if current_profit > TP0 and trade.nr_of_successful_exits == 0 and count_of_entries == 2:
            # Take half of the profit at next fib%%
            return -(trade.stake_amount / 2)
        if current_profit > TP1 and trade.nr_of_successful_exits == 1 and count_of_entries == 2:
            # Take profit at last fib%%
            return -(trade.stake_amount)
            
        if trade.nr_of_successful_entries == self.max_epa.value + 1:
            return None 
        # Block concurrent buys and Hold otherwise if the dip is not large enough
        if current_profit > -TP1:
            return None

        try:
            # This returns first order stake size 
            # Modify the following parameters to enable more levels or different buy size:
            # max_entry_position_adjustment = 3 
            # max_dca_multiplier = 3.5 

            stake_amount = filled_entries[0].cost
            # This then calculates current safety order size when a secondary buy signal is generated.
            if (last_fill > self.filldelay.value):
                if (signal == 1 and current_profit < -TP0):
                    if count_of_entries >= 1: 
                        stake_amount = stake_amount * 2
                    else:
                        stake_amount = stake_amount

                    return stake_amount

            # This then calculates current safety order size when below -Take Profit 1.
            if (last_fill > self.filldelay.value):
                if current_profit < -TP0:
                    if count_of_entries >= 1: 
                        stake_amount = stake_amount * 1.5
                    else:
                        stake_amount = stake_amount

                    return stake_amount

            # This accommadates a one shot at buying the dip on a big wick with one 
            # large buy if the funds are available...
            if (last_fill > self.filldelay.value):
                if (current_profit < -TP3):
                    if count_of_entries == 1: 
                        stake_amount = stake_amount * 4
                    else:
                        stake_amount = stake_amount

                    return stake_amount

        except Exception as exception:
            return None

        return None


    ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        SLT0 = current_candle['h2_move_mean']
        SLT1 = current_candle['h1_move_mean']
        SLT2 = current_candle['h0_move_mean']
        SLT3 = current_candle['cycle_move_mean']
        sortBop = current_candle['sort_bop']
        
        display_profit = current_profit * 100

        if current_profit < -0.01:
            if pair in self.locked_stoploss:
                del self.locked_stoploss[pair]
                self.dp.send_msg(f'*** {pair} *** Stoploss reset.')
                logger.info(f'*** {pair} *** Stoploss reset.')
            return self.stoploss

        new_stoploss = None
        if SLT3 is not None and current_profit > SLT3:
            new_stoploss = (SLT2 - SLT1)
            level = 4
        elif SLT2 is not None and current_profit > SLT2:
            new_stoploss = (SLT2 - SLT1)
            level = 3

        # in the future toggle these on certain conditions with indicators.
        elif SLT1 is not None and current_profit > SLT1 and sortBop < -0.2:
            new_stoploss = (SLT1 - SLT0)
            level = 2
        elif SLT0 is not None and current_profit > SLT0 and sortBop < -0.3:
            new_stoploss = (SLT1 - SLT0)
            level = 1

        if new_stoploss is not None:
            if pair not in self.locked_stoploss or new_stoploss > self.locked_stoploss[pair]:
                self.locked_stoploss[pair] = new_stoploss
                self.dp.send_msg(f'*** {pair} *** Profit {level} {display_profit:.3f}%% - New stoploss: {new_stoploss:.4f} activated')
                logger.info(f'*** {pair} *** Profit {level} {display_profit:.3f}%% - New stoploss: {new_stoploss:.4f} activated')
            return self.locked_stoploss[pair]

        return self.stoploss


    def custom_entry_price(self, pair: str, trade: Optional['Trade'], current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)

        entry_price = (dataframe['close'].iat[-1] + dataframe['open'].iat[-1] + dataframe['low'].iat[-1] + proposed_rate) / 4
        logger.info(f"{pair} Using Entry Price: {entry_price} | close: {dataframe['close'].iat[-1]} open: {dataframe['open'].iat[-1]} proposed_rate: {proposed_rate}")
        self.dp.send_msg(f"{pair} Using Entry Price: {entry_price} | close: {dataframe['close'].iat[-1]} open: {dataframe['open'].iat[-1]} proposed_rate: {proposed_rate}")

        # Check if there is a stored last entry price and if it matches the proposed entry price
        if self.last_entry_price is not None and abs(entry_price - self.last_entry_price) < 0.0001:  # Tolerance for floating-point comparison
            entry_price *= self.increment.value # Increment by 0.2%%
            logger.info(f"{pair} Incremented entry price: {entry_price} based on previous entry price : {self.last_entry_price}.")

        # Update the last entry price
        self.last_entry_price = entry_price

        return entry_price

    # Custom_Exits
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
    
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        trade_duration = (current_time - trade.open_date_utc).seconds / 60
        last_fill = (current_time - trade.date_last_filled_utc).seconds / 60 

        current_candle = dataframe.iloc[-1].squeeze()

        TP0 = current_candle['h2_move_mean'] 
        TP1 = current_candle['h1_move_mean']
        TP2 = current_candle['h0_move_mean']
        TP3 = current_candle['cycle_move_mean']

        ### roi ###
        if current_profit > TP3 and trade_duration > self.time0.value:
            return 'Roi 0 - Easy $$$'
        if current_profit > TP2 and trade_duration > self.time1.value:
            return 'Roi 1 - ol reliable' 
        if current_profit > TP1 and trade_duration > self.time2.value:
            return 'Roi 2 - Avg Joe'
        if current_profit > TP0 and trade_duration > self.time3.value:
            return 'Roi 3 - Better than Nothing'

        ### negative stoploss ###
        if current_profit < -TP3 and self.use_stop1.value == True:
            return 'Failsafe 3 - REKTd'
        if current_profit < -TP2 and self.use_stop2.value == True:
            return 'Failsafe 2 - Ooo that hurts'
        if current_profit < -TP1 and self.use_stop3.value == True:
            return 'Failsafe 1 - Leverage is risky'
        if current_profit < -TP0 and self.use_stop4.value == True:
            return 'Failsafe 0 - Wasnt a good idea...'

        return False

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                       rate: float, time_in_force: str, exit_reason: str,
                       current_time: datetime, **kwargs) -> bool:
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        trade_duration = (current_time - trade.open_date_utc).seconds / 60 
        last_candle = dataframe.iloc[-1].squeeze()
        heikinashi = qtpylib.heikinashi(dataframe)

        dataframe['%%-ha_open'] = heikinashi['open']
        dataframe['%%-ha_close'] = heikinashi['close']
        hodl = last_candle['frac_HODL'] 
        TP2 = current_candle['h0_move_mean']

        # Handle freak events and stoploss engagement
        if hodl == 1 and trade.calc_profit_ratio(rate) > TP2:
            logger.info(f"{trade.pair} HODL!!!")
            return False

        if exit_reason == 'roi' and trade.calc_profit_ratio(rate) < 0.003:
            logger.info(f"{trade.pair} ROI is below 0%")
            # self.dp.send_msg(f'{trade.pair} ROI is below 0')
            return False

        if exit_reason == 'partial_exit' and trade.calc_profit_ratio(rate) < 0.005:
            logger.info(f"{trade.pair} partial exit is below 0%")
            # self.dp.send_msg(f'{trade.pair} partial exit is below 0')
            return False

        if exit_reason == 'trailing_stop_loss' and trade.calc_profit_ratio(rate) < 0.005:
            logger.info(f"{trade.pair} trailing stop price is below 0%")
            # self.dp.send_msg(f'{trade.pair} trailing stop price is below 0')
            return False

        return True


    def feature_engineering_expand_all(self, dataframe, period, **kwargs):
        # We do this in expand basic using Hurst Cycle Theory and FFT
        return dataframe


    def feature_engineering_expand_basic(self, dataframe, metadata, **kwargs):
        start_time = time.time()
        pair = metadata['pair']
        dataframe["%%-obv"] = ta.OBV(dataframe)
        # Williams R%%
        dataframe['%%-willr14'] = pta.willr(dataframe['high'], dataframe['low'], dataframe['close'])

        # VWAP
        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe['%%-vwap_upperband'] = vwap_high
        dataframe['%%-vwap_middleband'] = vwap
        dataframe['%%-vwap_lowerband'] = vwap_low
        dataframe['%%-vwap_width'] = ((dataframe['%%-vwap_upperband'] -
                                     dataframe['%%-vwap_lowerband']) / dataframe['%%-vwap_middleband']) * 100
        dataframe = dataframe.copy()
        dataframe['%%-dist_to_vwap_upperband'] = get_distance(dataframe['close'], dataframe['%%-vwap_upperband'])
        dataframe['%%-dist_to_vwap_middleband'] = get_distance(dataframe['close'], dataframe['%%-vwap_middleband'])
        dataframe['%%-dist_to_vwap_lowerband'] = get_distance(dataframe['close'], dataframe['%%-vwap_lowerband'])
        dataframe['%%-tail'] = (dataframe['close'] - dataframe['low']).abs()
        dataframe['%%-wick'] = (dataframe['high'] - dataframe['close']).abs()
        dataframe['%%-rawclose'] = dataframe['close']
        dataframe["%%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%%-raw_volume"] = dataframe["volume"]
        dataframe["%%-raw_price"] = dataframe["close"]
        dataframe["%%-raw_open"] = dataframe["open"]
        dataframe["%%-raw_low"] = dataframe["low"]
        dataframe["%%-raw_high"] = dataframe["high"]

        heikinashi = qtpylib.heikinashi(dataframe)

        dataframe['%%-ha_open'] = heikinashi['open']
        dataframe['%%-ha_close'] = heikinashi['close']
        dataframe['%%-ha_high'] = heikinashi['high']
        dataframe['%%-ha_low'] = heikinashi['low']
        dataframe['%%-ha_closedelta'] = (heikinashi['close'] - heikinashi['close'].shift())
        dataframe['%%-ha_tail'] = (heikinashi['close'] - heikinashi['low'])
        dataframe['%%-ha_wick'] = (heikinashi['high'] - heikinashi['close'])

        dataframe['%%-HLC3'] = (heikinashi['high'] + heikinashi['low'] + heikinashi['close'])/3

        dataframe['%%-body_size'] = abs(dataframe['open'] - dataframe['close'])
        dataframe['%%-range_size'] = dataframe['high'] - dataframe['low']
        dataframe['%%-body_range_ratio'] = dataframe['%%-body_size'] / dataframe['%%-range_size']

        dataframe['%%-upper_wick_size'] = dataframe['high'] - dataframe[['open', 'close']].max(axis=1)
        dataframe['%%-upper_wick_range_ratio'] = dataframe['%%-upper_wick_size'] / dataframe['%%-range_size']
        
        dataframe['%%-max_high_24'] = dataframe['high'].rolling(24).max()
        dataframe['%%-min_low_24'] = dataframe['low'].rolling(24).min()
        dataframe['%%-close_position'] = (dataframe['close'] - dataframe['%%-min_low_24']) / (dataframe['%%-max_high_24'] - dataframe['%%-min_low_24'])
        dataframe['%%-current_candle_perc_change'] = (dataframe['high'] / dataframe['open'] - 1) * 100

        # Lazy Bear Impulse Macd
        dataframe['%%-hi'] = ta.SMA(dataframe['high'], timeperiod = 28)
        dataframe['%%-lo'] = ta.SMA(dataframe['low'], timeperiod = 28)
        dataframe['%%-ema1'] = ta.EMA(dataframe['%%-HLC3'], timeperiod = 28)
        dataframe['%%-ema2'] = ta.EMA(dataframe['%%-ema1'], timeperiod = 28)
        dataframe['%%-d'] = dataframe['%%-ema1'] - dataframe['%%-ema2']
        dataframe['%%-mi'] = dataframe['%%-ema1'] + dataframe['%%-d']
        dataframe['%%-md'] = np.where(dataframe['%%-mi'] > dataframe['%%-hi'], 
            dataframe['%%-mi'] - dataframe['%%-hi'], 
            np.where(dataframe['%%-mi'] < dataframe['%%-lo'], 
            dataframe['%%-mi'] - dataframe['%%-lo'], 0))
        dataframe['%%-sb'] = ta.SMA(dataframe['%%-md'], timeperiod = 8)
        dataframe['%%-sh'] = dataframe['%%-md'] - dataframe['%%-sb']
        
        # WaveTrend using OHLC4 or HA close - 3/21
        ap = dataframe['%%-HLC3']
        
        dataframe['esa'] = ta.EMA(ap, timeperiod = 9)
        dataframe['d'] = ta.EMA(abs(ap - dataframe['esa']), timeperiod = 9)
        dataframe['%%-wave_ci'] = (ap-dataframe['esa']) / (0.015 * dataframe['d'])
        dataframe['%%-wave_t1'] = ta.EMA(dataframe['%%-wave_ci'], timeperiod = 12)  
        dataframe['%%-wave_t2'] = ta.SMA(dataframe['%%-wave_t1'], timeperiod = 4)

        # 200 SMA and distance
        dataframe['%%-200sma'] = ta.SMA(dataframe, timeperiod = 200)
        dataframe['%%-200sma_dist'] = get_distance(heikinashi["close"], dataframe['%%-200sma'])

        # h = self.h2
        # r = 8.0
        # x_0 = self.cp
        # smoothColors = False
        # lag = 0

        # nadaraya_watson(dataframe, h, r, x_0, smoothColors, lag, mult = 2.5)

        # box_size_percentage = dataframe['h2_move_mean'].iloc[-1] * 0.5
        # reversal_size = box_size_percentage * 0.25
        # calculate_pnf_with_percentage(dataframe, box_size_percentage, reversal_size)

        if not self.dp.runmode.value in ("backtest", "plot", "hyperopt"):
            
            end_time = time.time()
            logger.info(f"Expanded Indicators done for {pair} in {end_time - start_time:.2f} secs")

        return dataframe

    def feature_engineering_standard(self, dataframe, metadata, **kwargs):
        start_time = time.time()
        pair = metadata['pair']

       # Initialize Hurst Cycles for startup errors
        cycle_period = 80
        harmonics = [0, 0, 0]
        harmonics[0] = 40
        harmonics[1] = 27
        harmonics[2] = 20

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['%%-ha_close'] = heikinashi['close']

        if len(dataframe) < self.u_window_size.value:
            raise ValueError(f"Insufficient data points for FFT: {len(dataframe)}. Need at least {self.u_window_size.value} data points.")

        # Perform FFT to identify cycles with a rolling window
        freq, power = perform_fft(dataframe[f'%%-ha_close'], window_size=self.u_window_size.value)

        if len(freq) == 0 or len(power) == 0:
            raise ValueError("FFT resulted in zero or invalid frequencies. Check the data or the FFT implementation.")

        # Filter out the zero-frequency component and limit the frequency to below 500
        # positive_mask = (freq > 0) & (1 / freq < self.u_window_size.value)
        positive_mask = (1 / freq > self.l_window_size.value) & (1 / freq < self.u_window_size.value)
        positive_freqs = freq[positive_mask]
        positive_power = power[positive_mask]

        # Convert frequencies to periods
        cycle_periods = 1 / positive_freqs

        # Set a threshold to filter out insignificant cycles based on power
        power_threshold = 0.01 * np.max(positive_power)
        significant_indices = positive_power > power_threshold
        significant_periods = cycle_periods[significant_indices]
        significant_power = positive_power[significant_indices]

        # Identify the dominant cycle
        dominant_freq_index = np.argmax(significant_power)
        dominant_freq = positive_freqs[dominant_freq_index]
        # logger.info(f'{pair} Hurst Exponent: {dominant_freq}')
        cycle_period = int(np.abs(1 / dominant_freq)) if dominant_freq != 0 else 100

        if cycle_period == np.inf:
            raise ValueError("No dominant frequency found. Check the data or the method used.")

        # Calculate harmonics for the dominant cycle
        harmonics = [cycle_period / (i + 1) for i in range(1, 4)]
        # print(cycle_period, harmonics)
        self.cp = int(cycle_period)
        self.h0 = int(harmonics[0])
        self.h1 = int(harmonics[1])
        self.h2 = int(harmonics[2])

        dataframe['%%-dc_EWM'] = dataframe['%%-ha_close'].ewm(span=int(cycle_period)).mean()
        dataframe['%%-dc_1/2'] = dataframe['%%-ha_close'].ewm(span=int(harmonics[0])).mean()
        dataframe['%%-dc_1/3'] = dataframe['%%-ha_close'].ewm(span=int(harmonics[1])).mean()
        dataframe['%%-dc_1/4'] = dataframe['%%-ha_close'].ewm(span=int(harmonics[2])).mean()

        # Fractional Brownian Motion (fBm)
        n = len(dataframe['%%-dc_EWM'])
        h = dominant_freq #0.5
        t = 1
        dt = t / n
        fBm = np.zeros(n)
        for i in range(1, n):
            fBm[i] = fBm[i-1] + np.sqrt(dt) * np.random.normal(0, 1)
        dataframe['%%-fBm'] = fBm * (dt ** h)

        # Fractional differentiation
        dataframe['%%-frac_diff'] = np.gradient(dataframe['%%-ha_close'])
        dataframe['%%-fBm_mean'] = np.mean(dataframe['%%-fBm']) + 2 * np.std(dataframe['%%-fBm'])
        dataframe['%%-frac_diff_norm'] = indicator_normalization(dataframe, col='%%-frac_diff', length=int(cycle_period), norm_range='minus_one_to_one')


        dataframe['%%-frac_sma_dc'] = dataframe['%%-frac_diff'].ewm(span=int(harmonics[1])).mean()
        dataframe['%%-signal_UP_dc'] = np.where(dataframe['%%-frac_sma_dc'] > 0, dataframe['%%-frac_sma_dc'], 0)
        dataframe['%%-signal_DN_dc'] = np.where(dataframe['%%-frac_sma_dc'] < 0, dataframe['%%-frac_sma_dc'], 0)
        dataframe['%%-signal_UP_dc'] = dataframe['%%-signal_UP_dc'].ffill()
        dataframe['%%-signal_DN_dc'] = dataframe['%%-signal_DN_dc'].ffill()
        dataframe['%%-signal_MEAN_UP_dc'] = dataframe['%%-signal_UP_dc'].rolling(int(cycle_period)).mean() * self.dc_x.value
        dataframe['%%-signal_MEAN_DN_dc'] = dataframe['%%-signal_DN_dc'].rolling(int(cycle_period)).mean() * self.dc_x.value
        dataframe['%%-signal_buy'] = np.where((dataframe['%%-frac_sma_dc'] > 0) & (dataframe['%%-frac_sma_dc'].shift() < 0), 1, 0)
        dataframe['%%-signal_sell'] = np.where((dataframe['%%-frac_sma_dc'] < 0) & (dataframe['%%-frac_sma_dc'].shift() > 0), -1, 0)

        dataframe['frac_HODL'] = np.where(dataframe['%%-frac_sma_dc'] > dataframe['%%-signal_MEAN_UP_dc'], 1, 0)
        dataframe['frac_y33t'] = np.where(dataframe['%%-frac_sma_dc'] < dataframe['%%-signal_MEAN_DN_dc'], -1, 0)


        # Apply rolling window operation to the 'OHLC4' column
        rolling_windowc = dataframe['%%-ha_close'].rolling(cycle_period) 
        rolling_windowh0 = dataframe['%%-ha_close'].rolling(int(harmonics[0]))
        rolling_windowh1 = dataframe['%%-ha_close'].rolling(int(harmonics[1])) 
        rolling_windowh2 = dataframe['%%-ha_close'].rolling(int(harmonics[2])) 

        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_valuec = rolling_windowc.apply(lambda x: np.ptp(x))
        ptp_valueh0 = rolling_windowh0.apply(lambda x: np.ptp(x))
        ptp_valueh1 = rolling_windowh1.apply(lambda x: np.ptp(x))
        ptp_valueh2 = rolling_windowh2.apply(lambda x: np.ptp(x))

        # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['cycle_move'] = ptp_valuec / dataframe['%%-ha_close']
        dataframe['h0_move'] = ptp_valueh0 / dataframe['%%-ha_close']
        dataframe['h1_move'] = ptp_valueh1 / dataframe['%%-ha_close']
        dataframe['h2_move'] = ptp_valueh2 / dataframe['%%-ha_close']

        dataframe['%%-cycle_move'] = dataframe['cycle_move']
        dataframe['%%-h0_move'] = dataframe['h0_move']
        dataframe['%%-h1_move'] = dataframe['h1_move']
        dataframe['%%-h2_move'] = dataframe['h2_move']


        dataframe['cycle_move_mean'] = dataframe['cycle_move'].rolling(self.h2).mean()        
        dataframe['h0_move_mean'] = dataframe['h0_move'].rolling(self.h2).mean()
        dataframe['h1_move_mean'] = dataframe['h1_move'].rolling(self.h2).mean() 
        dataframe['h2_move_mean'] = dataframe['h2_move'].rolling(self.h2).mean()
        dataframe['h2_move_min'] = dataframe['h2_move'].rolling(self.h2).min()

        dataframe['%%-cycle_move_mean'] = dataframe['cycle_move_mean']       
        dataframe['%%-h0_move_mean'] = dataframe['h0_move_mean']
        dataframe['%%-h1_move_mean'] = dataframe['h1_move_mean'] 
        dataframe['%%-h2_move_mean'] = dataframe['h2_move_mean']


        # Add envelopes for the dominant cycle
        dataframe['%%-upper_envelope'] = dataframe['%%-dc_EWM'] * (1 + dataframe['%%-cycle_move_mean'])
        dataframe['%%-lower_envelope'] = dataframe['%%-dc_EWM'] * (1 - dataframe['%%-cycle_move_mean'])
        dataframe['%%-upper_envelope_h0'] = dataframe['%%-dc_1/2'] * (1 + dataframe['%%-h0_move_mean'])
        dataframe['%%-lower_envelope_h0'] = dataframe['%%-dc_1/2'] * (1 - dataframe['%%-h0_move_mean'])
        dataframe['%%-upper_envelope_h1'] = dataframe['%%-dc_1/3'] * (1 + dataframe['%%-h1_move_mean'])
        dataframe['%%-lower_envelope_h1'] = dataframe['%%-dc_1/3'] * (1 - dataframe['%%-h1_move_mean'])
        dataframe['%%-upper_envelope_h2'] = dataframe['%%-dc_1/4'] * (1 + dataframe['%%-h2_move_mean'])
        dataframe['%%-lower_envelope_h2'] = dataframe['%%-dc_1/4'] * (1 - dataframe['%%-h2_move_mean'])

        dataframe['upper_envelope'] =dataframe['%%-upper_envelope'] 
        dataframe['lower_envelope'] = dataframe['%%-lower_envelope']
        dataframe['upper_envelope_h0'] = dataframe['%%-upper_envelope_h0']
        dataframe['lower_envelope_h0'] = dataframe['%%-lower_envelope_h0']
        dataframe['upper_envelope_h1'] = dataframe['%%-upper_envelope_h1']
        dataframe['lower_envelope_h1'] = dataframe['%%-lower_envelope_h1']
        dataframe['upper_envelope_h2'] = dataframe['%%-upper_envelope_h2']
        dataframe['lower_envelope_h2'] = dataframe['%%-lower_envelope_h2']

        dataframe['%%-candle_size'] = abs((dataframe['high'] - dataframe['low']) / dataframe['low'])
        dataframe['%%-candle_size_lower'] = dataframe['%%-candle_size'].rolling(window=int(harmonics[2])).mean()
        dataframe['%%-candle_size_upper'] = (dataframe['%%-candle_size'].rolling(window=int(harmonics[2])).mean()) * 2

        dataframe['%%-max_high'] = dataframe['high'].rolling(self.cp).max()
        dataframe['%%-min_low'] = dataframe['low'].rolling(self.cp).min()

        dataframe["%%-rsi"] = ta.RSI(dataframe, timeperiod=int(harmonics[2]))
        dataframe["%%-mfi-period"] = ta.MFI(dataframe, timeperiod=int(cycle_period))
        dataframe["%%-rocr-period"] = ta.ROCR(dataframe, timeperiod=int(harmonics[2]))
        dataframe["%%-cmf-period"] = chaikin_mf(dataframe, periods=int(cycle_period))
        dataframe["%%-chop-period"] = qtpylib.chopiness(dataframe, int(harmonics[2]))
        dataframe["%%-linear-period"] = ta.LINEARREG_ANGLE(
            dataframe['close'], timeperiod=int(harmonics[2]))
        dataframe["%%-atr-period"] = ta.ATR(dataframe, timeperiod=int(harmonics[2]))
        dataframe["%%-atr-periodp"] = dataframe["%%-atr-period"] / \
            dataframe['close'] * 1000


        h = self.h2
        r = 8.0
        x_0 = self.cp
        smoothColors = False
        lag = 0

        nadaraya_watson(dataframe, h, r, x_0, smoothColors, lag, mult = 2.5)

        dataframe["%%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        dataframe["%%-hour_of_day"] = (dataframe["date"].dt.hour + 1) / 25

        if not self.dp.runmode.value in ("backtest", "plot", "hyperopt"):
            logger.info(f'{pair} - DC: {cycle_period:.2f} | 1/2: {harmonics[0]:.2f} | 1/3: {harmonics[1]:.2f} | 1/4: {harmonics[2]:.2f}')
            end_time = time.time()


        return dataframe


    def set_freqai_targets(self, dataframe, **kwargs):
        dataframe["&s-extrema"] = 0
        order = self.cp
        min_peaks = argrelextrema(
            dataframe["close"].values, np.less,
            order=order
        )
        max_peaks = argrelextrema(
            dataframe["close"].values, np.greater,
            order=order
        )
        for mp in min_peaks[0]:
            dataframe.at[mp, "&s-extrema"] = -1
        for mp in max_peaks[0]:
            dataframe.at[mp, "&s-extrema"] = 1
        dataframe["minima"] = np.where(dataframe["&s-extrema"] == -1, 1, 0)
        dataframe["maxima"] = np.where(dataframe["&s-extrema"] == 1, 1, 0)
        dataframe['&s-extrema'] = dataframe['&s-extrema'].rolling(
            window=5, win_type='gaussian', center=True).mean(std=0.5)

        # predict the expected range
        dataframe['&-s_max'] = dataframe["close"].shift(-order).rolling(
            order).max()/dataframe["close"] - 1
        dataframe['&-s_min'] = dataframe["close"].shift(-order).rolling(
            order).min()/dataframe["close"] - 1

        return dataframe


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # dataframe["&s-extrema"] = 0

        pair = metadata['pair']
        pair_tag = f'_gen_{pair}_{self.timeframe}'
        dataframe = self.freqai.start(dataframe, metadata, self)

        dataframe["DI_catch"] = np.where(
            dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1,
        )
        
        dataframe["&s-extrema"] = dataframe['&s-extrema'].fillna(0)
        dataframe["minima_sort_threshold"] = dataframe["&s-minima_sort_threshold"].fillna(0)
        dataframe["maxima_sort_threshold"] = dataframe["&s-maxima_sort_threshold"].fillna(0)
        dataframe["sort_bop"] = dataframe["maxima_sort_threshold"] + dataframe["minima_sort_threshold"].fillna(0)
        dataframe['sort_bop'] = dataframe['sort_bop'].astype(float)

        dataframe["min_sort_threshold"] = dataframe["&-s_min"].mean()
        dataframe["max_sort_threshold"] = dataframe["&-s_max"].mean()

        dataframe['maxima_check'] = dataframe['maxima'].rolling(self.cp).apply(lambda x: int((x != 1).all()), raw=True).fillna(0)
        dataframe['minima_check'] = dataframe['minima'].rolling(self.cp).apply(lambda x: int((x != 1).all()), raw=True).fillna(0)

        dataframe["uptrend"] = np.where(dataframe["&s-extrema"] < 0, 1, -1)
        dataframe[["&s-extrema", "sort_bop"]] = dataframe[["&s-extrema", "sort_bop"]].fillna(0)
        # dataframe["extrema_cross_up"] = np.where(((dataframe["&s-extrema"].shift() <= dataframe["sort_bop"].shift()) & (dataframe["&s-extrema"] >= dataframe["sort_bop"])), 1, 0)
        # dataframe["extrema_cross_dn"] = np.where(((dataframe["&s-extrema"].shift() >= dataframe["sort_bop"].shift()) & (dataframe["&s-extrema"] <= dataframe["sort_bop"])), -1, 0)

        dataframe["extrema_cross_up"] = np.where(
            (
                (dataframe["&s-extrema"].shift().notnull() & dataframe["sort_bop"].shift().notnull() & (dataframe["&s-extrema"].shift() <= dataframe["sort_bop"].shift())) &
                (dataframe["&s-extrema"].notnull() & dataframe["sort_bop"].notnull() & (dataframe["&s-extrema"] >= dataframe["sort_bop"]))
            ), 1, 0)

        dataframe["extrema_cross_dn"] = np.where(
            (
                (dataframe["&s-extrema"].shift().notnull() & dataframe["sort_bop"].shift().notnull() & (dataframe["&s-extrema"].shift() >= dataframe["sort_bop"].shift())) &
                (dataframe["&s-extrema"].notnull() & dataframe["sort_bop"].notnull() & (dataframe["&s-extrema"] <= dataframe["sort_bop"]))
            ), 1,0)

        dataframe['BOP'] = (ta.SMA(dataframe['close'], self.cp) - ta.SMA(dataframe['open'], self.cp)) / (ta.SMA(dataframe['high'], self.cp) - ta.SMA(dataframe['low'], self.cp))
        dataframe['BOP_SMA'] = ta.SMA(dataframe['BOP'], self.h0)
        dataframe['zero'] = 0
        # print(dataframe.columns.tolist())

        dataframe['Nadaraya'] = dataframe[f"%%-yhat1"]
        dataframe['nw_up'] = dataframe[f"%%-nw_up"]
        dataframe['nw_down'] = dataframe[f"%%-nw_down"]
        dataframe['nw_width'] = dataframe["nw_up"] - dataframe["nw_down"]
        dataframe['nw_width_mean'] = dataframe['nw_width'].mean()
        dataframe['nw_width_up'] = (dataframe['nw_width_mean'] + (dataframe['nw_width_mean']/2))
        dataframe['nw_width_dn'] = (dataframe['nw_width_mean'] - (dataframe['nw_width_mean']/2))
        # Calculate slope and angles
        dataframe['slope'] = dataframe['Nadaraya'].diff()
        dataframe['slope_UP'] = np.where(dataframe['slope'] > 0, dataframe['slope'], 0)
        dataframe['slope_DN'] = np.where(dataframe['slope'] < 0, dataframe['slope'], 0)
        dataframe['slope_UP'] = dataframe['slope_UP'].ffill()
        dataframe['slope_DN'] = dataframe['slope_DN'].ffill()
        dataframe['slope_MEAN_UP'] = dataframe['slope_UP'].mean()
        dataframe['slope_MEAN_DN'] = dataframe['slope_DN'].mean()

        return dataframe


    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        pair = metadata['pair']
        pair_tag = f'_gen_{pair}_{self.timeframe}'

        df.loc[
            (
                (df["do_predict"] == 1) & 
                (df["DI_catch"] == 1) & 
                (df[f"%%-nw_down"] > df['close']) & 
                (df["&s-extrema"] < df["minima_sort_threshold"]) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['enter_long', 'enter_tag']] = (1, 'Extrema NW Close')

        df.loc[
            (
                (df["do_predict"] == 1) & 
                (df["DI_catch"] == 1) & 
                (df[f"%%-nw_down"] > df['open']) & 
                (df["&s-extrema"] < df["minima_sort_threshold"]) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['enter_long', 'enter_tag']] = (1, 'Extrema NW Open')

        df.loc[
            (
                (df[f"%%-nw_down"] > df['close']) & 
                (df["extrema_cross_up"] == 1) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['enter_long', 'enter_tag']] = (1, 'Extrema BOP X Up Close')

        df.loc[
            (
                (df[f"%%-nw_down"] > df['open']) & 
                (df["extrema_cross_up"] == 1) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['enter_long', 'enter_tag']] = (1, 'Extrema BOP X Up Open')

        # df.loc[
        #     (
        #         # (df["do_predict"] == 1) & 
        #         (df["DI_catch"] == 1) & 
        #         (0 < df[f'%%-frac_diff{pair_tag}']) & 
        #         (0 > df[f'%%-frac_diff{pair_tag}'].shift()) & 
        #         (df[f"%%-nw_down{pair_tag}"] > df['close']) & 
        #         # (df["&s-extrema"] < df["minima_sort_threshold"]) & 
        #         (df['volume'] > 0)   # Make sure Volume is not 0

        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'frac_diff close')

        # df.loc[
        #     (
        #         # (df["do_predict"] == 1) & 
        #         (df["DI_catch"] == 1) & 
        #         (0 < df[f'%%-frac_diff{pair_tag}']) & 
        #         (0 > df[f'%%-frac_diff{pair_tag}'].shift()) & 
        #         (df[f"%%-nw_down{pair_tag}"] > df['open']) & 
        #         # (df["&s-extrema"] < df["minima_sort_threshold"]) & 
        #         (df['volume'] > 0)   # Make sure Volume is not 0

        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'frac_diff close')

        # df.loc[
        #     (
        #         (df["do_predict"] == 0) & 
        #         (df[f"%%-nw_down{pair_tag}"] > df['close']) & 
        #         (df["&s-extrema"] < df["minima_sort_threshold"]) & 
        #         # (df["minima"].shift(1) == 1) & 
        #         (df['volume'] > 0)   # Make sure Volume is not 0

        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'Extrema Full Send')

        df.loc[
            (
                (df["do_predict"] == 1) & 
                (df["DI_catch"] == 1) & 
                (df["&s-extrema"] < df["minima_sort_threshold"]) & 
                (df["minima"].shift(1) == 1) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['enter_long', 'enter_tag']] = (1, 'Minima')

        df.loc[
            (
                (df["do_predict"] == 0) & 
                (df["&s-extrema"] < df["minima_sort_threshold"]) & 
                (df["minima"].shift(1) == 1) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['enter_long', 'enter_tag']] = (1, 'Minima Full Send')

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        pair = metadata['pair']
        pair_tag = f'_gen_{pair}_{self.timeframe}'

        # df.loc[
        #     (
        #         (df["do_predict"] == 1) &  
        #         (df["DI_catch"] == 1) & 
        #         (df["&s-extrema"] > df["maxima_sort_threshold"]) & 
        #         (df["minima_check"] == 0) & 
        #         (df['volume'] > 0)   # Make sure Volume is not 0

        #     ),
        #     ['exit_long', 'exit_tag']] = (1, 'Extrema')

        # df.loc[
        #     (
        #         (df["do_predict"] == 0) &  
        #         # (df["DI_catch"] == 1) & 
        #         (df["&s-extrema"] > df["maxima_sort_threshold"]) & 
        #         (df["minima_check"] == 0) & 
        #         (df['volume'] > 0)   # Make sure Volume is not 0

        #     ),
        #     ['exit_long', 'exit_tag']] = (1, 'Extrema Full Send')

        df.loc[
            (
                (df["do_predict"] == 1) &  
                (df["DI_catch"] == 1) & 
                (df[f"%%-nw_up"] < df['close']) & 
                (df["&s-extrema"] > df["maxima_sort_threshold"]) & 
                (df["maxima"].shift(1) == 1) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['exit_long', 'exit_tag']] = (1, 'Maxima Close')

        df.loc[
            (
                (df["do_predict"] == 1) &  
                (df["DI_catch"] == 1) & 
                (df[f"%%-nw_up"] < df['open']) & 
                (df["&s-extrema"] > df["maxima_sort_threshold"]) & 
                (df["maxima"].shift(1) == 1) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['exit_long', 'exit_tag']] = (1, 'Maxima Open')

        # df.loc[
        #     (
        #         (df["do_predict"] == 0) &  
        #         # (df["DI_catch"] == 1) & 
        #         (df["minima_check"] == 0) & 
        #         (df["maxima"].shift(1) == 1) & 
        #         (df['volume'] > 0)   # Make sure Volume is not 0

        #     ),
        #     ['exit_long', 'exit_tag']] = (1, 'Maxima Full Send')

        return df


def top_percent_change(dataframe: DataFrame, length: int) -> float:
    """
    Percentage change of the current close from the range maximum Open price
    :param dataframe: DataFrame The original OHLC dataframe
    :param length: int The length to look back
    """
    if length == 0:
        return (dataframe['open'] - dataframe['close']) / dataframe['close']
    else:
        return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']

def chaikin_mf(df, periods=20):
    close = df['close']
    low = df['low']
    high = df['high']
    volume = df['volume']
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0)
    mfv *= volume
    cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()
    return Series(cmf, name='cmf')

def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df, window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']

def get_distance(p1, p2):
    return abs((p1) - (p2))

def PC(dataframe, in1, in2):
    df = dataframe.copy()
    pc = ((in2-in1)/in1) * 100
    return pc

def perform_fft(price_data, window_size=None):
    if window_size is not None:
        # Apply rolling window to smooth the data
        price_data = price_data.rolling(window=window_size, center=True).mean().dropna()

    normalized_data = (price_data - np.mean(price_data)) / np.std(price_data)
    n = len(normalized_data)
    fft_data = np.fft.fft(normalized_data)
    freq = np.fft.fftfreq(n)
    power = np.abs(fft_data) ** 2
    power[np.isinf(power)] = 0
    return freq, power

def indicator_normalization(df: DataFrame, col: str, length: int, norm_range: str) -> Series:

    rolling_min = df[col].rolling(window=length, min_periods=1).min()
    rolling_max = df[col].rolling(window=length, min_periods=1).max()

    # Apply [0,1] scale
    if norm_range == 'zero_to_one':
        # df[f'%%-normalized_{col}'] = (df[col] - rolling_min) / (rolling_max - rolling_min)
        normalized_indicator = (df[col] - rolling_min) / (rolling_max - rolling_min)

    # Apply [-1,1] scale
    elif norm_range == 'minus_one_to_one':
        # df[f'%%-normalized_{col}'] = 2 * ((df[col] - rolling_min) / (rolling_max - rolling_min)) - 1
        normalized_indicator = 2 * ((df[col] - rolling_min) / (rolling_max - rolling_min)) - 1

    return normalized_indicator

def kernel_regression(src, size, h, r, x_0):
    _currentWeight = 0.0
    _cumulativeWeight = 0.0000000001
    for i in range(len(src)):
        y = src.iloc[i]
        w = np.power(1 + (np.power(i, 2) / ((np.power(h, 2) * 2 * r))), -r)
        _currentWeight += y * w
        _cumulativeWeight += w
    if _cumulativeWeight == 0:
        return 0
    return _currentWeight / _cumulativeWeight

def nadaraya_watson(df, h, r, x_0, smoothColors, lag, mult=2):
    src = df['%%-ha_close']
    size = len(src)
    yhat1 = []
    yhat2 = []
    nwe_up = []
    nwe_down = []
    nwe_entry = []
    nwe_exit = []
    wasBearish = []
    wasBullish = []
    isBearish = []
    isBullish = []
    isBearishChange = []
    isBullishChange = []
    isBullishCross = []
    isBearishCross = []
    isBullishSmooth = []
    isBearishSmooth = []
    colorByCross = []
    colorByRate = []
    plotColor = []
    alertBullish = []
    alertBearish = []

    for i in range(size):
        if i >= h:
            window = src[i - h:i]
            yhat1_value = kernel_regression(window, h, h, r, x_0)
            yhat1.append(yhat1_value)

            # Compute envelopes
            mae = np.mean(np.abs(src[i - h:i] - yhat1_value)) * mult
            nwe_up.append(yhat1_value + mae)
            nwe_down.append(yhat1_value - mae)

            # Set entry and exit signals
            nwe_entry.append(1 if src[i] < nwe_down[-1] else 0)
            nwe_exit.append(1 if src[i] > nwe_up[-1] else 0)

            # Trend and crossover conditions
            if i > 1:
                wasBearish.append(yhat1[i - 2] > yhat1[i - 1])
                wasBullish.append(yhat1[i - 2] < yhat1[i - 1])
            else:
                wasBearish.append(False)
                wasBullish.append(False)

            if i > 0:
                isBearish.append(yhat1[i - 1] > yhat1[i])
                isBullish.append(yhat1[i - 1] < yhat1[i])
            else:
                isBearish.append(False)
                isBullish.append(False)

            isBearishChange.append(isBearish[-1] and wasBullish[-1] if wasBullish else False)
            isBullishChange.append(isBullish[-1] and wasBearish[-1] if wasBearish else False)

            if i >= h + lag:
                window = src[i - h - lag:i - lag]
                yhat2.append(kernel_regression(window, h, h, r, x_0))
                
                # Crossover conditions with lag
                if i > 0:
                    isBullishCross.append(yhat2[-1] > yhat1[i])
                    isBearishCross.append(yhat2[-1] < yhat1[i])
                    isBullishSmooth.append(yhat2[-1] > yhat1[i])
                    isBearishSmooth.append(yhat2[-1] < yhat1[i])
                else:
                    isBullishCross.append(False)
                    isBearishCross.append(False)
                    isBullishSmooth.append(False)
                    isBearishSmooth.append(False)

                # Color and alert conditions
                colorByCross.append(1 if isBullishSmooth[-1] else -1 if isBearishSmooth[-1] else 0)
                colorByRate.append(1 if isBullish[-1] else -1 if isBearish[-1] else 0)
                plotColor.append(colorByCross[-1] if smoothColors else colorByRate[-1])
                alertBullish.append(1 if isBullishCross[-1] else 0)
                alertBearish.append(-1 if isBearishCross[-1] else 0)
            else:
                yhat2.append(0)
                isBullishCross.append(False)
                isBearishCross.append(False)
                isBullishSmooth.append(False)
                isBearishSmooth.append(False)
                colorByCross.append(0)
                colorByRate.append(0)
                plotColor.append(0)
                alertBullish.append(0)
                alertBearish.append(0)
        else:
            yhat1.append(0)
            yhat2.append(0)
            nwe_up.append(np.nan)
            nwe_down.append(np.nan)
            nwe_entry.append(0)
            nwe_exit.append(0)
            wasBearish.append(False)
            wasBullish.append(False)
            isBearish.append(False)
            isBullish.append(False)
            isBearishChange.append(False)
            isBullishChange.append(False)
            isBullishCross.append(False)
            isBearishCross.append(False)
            isBullishSmooth.append(False)
            isBearishSmooth.append(False)
            colorByCross.append(0)
            colorByRate.append(0)
            plotColor.append(0)
            alertBullish.append(0)
            alertBearish.append(0)

    # Append the new columns to the dataframe
    df['%%-yhat1'] = yhat1
    df['%%-yhat2'] = yhat2
    df['%%-nw_up'] = nwe_up
    df['%%-nw_down'] = nwe_down
    df['%%-nw_entry'] = nwe_entry
    df['%%-nw_exit'] = nwe_exit
    df['%%-wasBearish'] = wasBearish
    df['%%-wasBullish'] = wasBullish
    df['%%-isBearish'] = isBearish
    df['%%-isBullish'] = isBullish
    df['%%-isBearishChange'] = isBearishChange
    df['%%-isBullishChange'] = isBullishChange
    df['%%-isBullishCross'] = isBullishCross
    df['%%-isBearishCross'] = isBearishCross
    df['%%-isBullishSmooth'] = isBullishSmooth
    df['%%-isBearishSmooth'] = isBearishSmooth
    df['colorByCross'] = colorByCross
    df['colorByRate'] = colorByRate
    df['plotColor'] = plotColor
    df['%%-alertBullish'] = alertBullish
    df['%%-alertBearish'] = alertBearish

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
