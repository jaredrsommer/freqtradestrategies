
import logging
import numpy as np
import pandas as pd
from technical import qtpylib
from pandas import DataFrame
from datetime import datetime, timezone
from typing import Optional
from functools import reduce
import talib.abstract as ta
import talib
import pandas_ta as pta
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
import warnings
from pandas.errors import PerformanceWarning

# Suppress PerformanceWarning
warnings.simplefilter(action="ignore", category=PerformanceWarning)

logger = logging.getLogger(__name__)


class kalthetank(IStrategy):

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
    exit_profit_only = True### No selling at a loss
    ignore_roi_if_entry_signal = True
    process_only_new_candles = True
    can_short = False
    use_exit_signal = True
    startup_candle_count = 200
    use_custom_stoploss = True
    trailing_stop = False

    locked_stoploss = {}
    minimal_roi = {}


    # Stoploss:
    stoploss = -0.20  # 15Fail Safe Default do not hyper opt Stoploss Space

    # Trailing stop:
    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = None  # value loaded from strategy
    trailing_stop_positive_offset = 0.0  # value loaded from strategy
    trailing_only_offset_is_reached = False  # value loaded from strategy


    ### Hyperoptable parameters ###
    # DCA
    position_adjustment_enable = True
    max_epa = IntParameter(0, 1, default = 1 ,space='buy', optimize=True, load=True) # of additional buys.
    max_dca_multiplier = DecimalParameter(low=1.0, high=5.0, default=3, decimals=1 ,space='buy', optimize=True, load=True)
    use_static = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    filldelay = IntParameter(100, 300, default = 122 ,space='buy', optimize=True, load=True)
    max_entry_position_adjustment = max_epa.value

    # indicators
    u_window_size = IntParameter(120, 300, default=282, space='buy', optimize=True)
    l_window_size = IntParameter(20, 70, default=42, space='buy', optimize=True)
    Q = DecimalParameter(low=0.0001, high=0.5, default=0.001, decimals=4 ,space='buy', optimize=True, load=True)
    R = DecimalParameter(low=0.001, high=0.5, default=0.1, decimals=3 ,space='buy', optimize=True, load=True)

    # Custom Entry
    increment = DecimalParameter(low=1.0005, high=1.002, default=1.001, decimals=4 ,space='buy', optimize=True, load=True)
    entryX = DecimalParameter(low=0.995, high=1.00, default=1.00, decimals=3 ,space='buy', optimize=True, load=True)
    last_entry_price = None

    # protections
    cooldown_lookback = IntParameter(4, 12, default=12, space="protection", optimize=True)
    stop_duration = IntParameter(4, 12, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # Risk Level Stake size adjustments based on Location to  
    stake0 = DecimalParameter(low=0.9, high=1.1, default=1.0, decimals=1 ,space='buy', optimize=True, load=True)
    stake1 = DecimalParameter(low=0.9, high=1.1, default=1.0, decimals=1 ,space='buy', optimize=True, load=True)
    stake2 = DecimalParameter(low=0.9, high=1.1, default=1.0, decimals=1 ,space='buy', optimize=True, load=True)
    stake3 = DecimalParameter(low=0.9, high=1.1, default=1.0, decimals=1 ,space='buy', optimize=True, load=True)

    # negative stoploss
    use_stop1 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    use_stop2 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    use_stop3 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    use_stop4 = BooleanParameter(default=False, space="protection", optimize=True, load=True)

    # roi
    # negative stoploss
    use_roi1 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    use_roi2 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    use_roi3 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    use_roi4 = BooleanParameter(default=False, space="protection", optimize=True, load=True)

    time0 = IntParameter(low=1440, high=2400, default=1638, space='protection', optimize=True, load=True)
    time1 = IntParameter(low=2400, high=5000, default=2901, space='protection', optimize=True, load=True)
    time2 = IntParameter(low=5000, high=10000, default=4500, space='protection', optimize=True, load=True)
    time3 = IntParameter(low=5000, high=10000, default=6981, space='protection', optimize=True, load=True)  
    # block specific exit timer
    time4 = IntParameter(low=240, high=480, default=343, space='sell', optimize=True, load=True)  

    # trading
    buy_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)
    buy_cl1 = DecimalParameter(low=0.15, high=0.4, default=0.37, decimals=2, space='buy', optimize=True, load=True)
    buy_cl2 = DecimalParameter(low=0.15, high=0.4, default=0.16, decimals=2, space='buy', optimize=True, load=True)
    buy_cl3 = DecimalParameter(low=0.15, high=0.4, default=0.16, decimals=2, space='buy', optimize=True, load=True)
    buy_cl4 = DecimalParameter(low=0.15, high=0.4, default=0.15, decimals=2, space='buy', optimize=True, load=True)
    buy_cls = DecimalParameter(low=0.15, high=0.4, default=0.25, decimals=2, space='buy', optimize=True, load=True)

    sell_rsi = IntParameter(low=50, high=100, default=55, space='sell', optimize=True, load=True)
    sell_cl1 = DecimalParameter(low=0.75, high=0.95, default=0.9, decimals=2, space='sell', optimize=True, load=True)
    sell_cl2 = DecimalParameter(low=0.75, high=0.95, default=0.77, decimals=2, space='sell', optimize=True, load=True)
    sell_cl3 = DecimalParameter(low=0.75, high=0.95, default=0.9, decimals=2, space='sell', optimize=True, load=True)
    sell_cl4 = DecimalParameter(low=0.75, high=0.95, default=0.77, decimals=2, space='sell', optimize=True, load=True)
    sell_cls = DecimalParameter(low=0.75, high=0.95, default=0.93, decimals=2, space='sell', optimize=True, load=True)


    # Entry Exit Logic Selection
    use0 = BooleanParameter(default=False, space="buy", optimize=True, load=True)
    use1 = BooleanParameter(default=False, space="buy", optimize=True, load=True)
    use2 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use3 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use4 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use5 = BooleanParameter(default=False, space="buy", optimize=True, load=True)
    use6 = BooleanParameter(default=False, space="buy", optimize=True, load=True)
    use7 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use8 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use9 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use10 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use11 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use12 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use13 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use14 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use15 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use16 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use17 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use18 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use19 = BooleanParameter(default=True, space="sell", optimize=True, load=True)



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

        # EP0 = current_candle['lower_envelope'] 
        # EP1 = current_candle['lower_envelope_h0'] 
        # EP2 = current_candle['lower_envelope_h1'] 
        # EP3 = current_candle['lower_envelope_h2']

        # Define balance thresholds and stake multipliers
        # EDIT These to match your funding level <--- !!!!!
        # balance = self.wallets.get_total_stake_amount()
        # if balance < 3000:
        #     balance_multiplier = 0.9
        #     self.config['max_open_trades'] = 5
        # elif balance < 7500:
        #     balance_multiplier = 0.95  # Increase stake by 20%
        #     self.config['max_open_trades'] = 6
        # elif balance < 10000:
        #     balance_multiplier = 1.0  # No increase
        #     self.config['max_open_trades'] = 5
        # elif balance < 15000:
        #     balance_multiplier = 1.05  # Increase stake by 20%
        #     self.config['max_open_trades'] = 8
        # elif balance < 20000:
        #     balance_multiplier = 1.1  # Increase stake by 50%
        #     self.config['max_open_trades'] = 10
        # else:
        #     balance_multiplier = 1.0  # Double stake for larger balances

        # Static or calculated staking
        if self.use_static.value == 'True':
            return (balance / (self.config["max_open_trades"] * self.max_dca_multiplier.value)) #* balance_multiplier

        # num_tr = self.config['max_open_trades']
        # # Adjust stake based on DCA, balance multiplier, and Location
        # if current_rate < EP0: # Longest envelope typically the least frequent to be crossed
        #     calculated_stake = (proposed_stake / self.max_dca_multiplier.value) * balance_multiplier
        # elif EP0 <= current_rate < EP1:
        #     calculated_stake = (proposed_stake / self.max_dca_multiplier.value) * self.stake3.value * balance_multiplier
        # elif EP1 <= current_rate < EP2:
        #     calculated_stake = (proposed_stake / self.max_dca_multiplier.value) * self.stake2.value * balance_multiplier
        # elif EP2 <= current_rate < EP3: # Shortest envelope typically the most frequent to be crossed
        #     calculated_stake = (proposed_stake / self.max_dca_multiplier.value) * self.stake1.value * balance_multiplier
        # else:
        calculated_stake = (proposed_stake / self.max_dca_multiplier.value) * self.stake0.value # * balance_multiplier
        # logger.info(f'{pair} using {calculated_stake} instead of {proposed_stake} | Trade Slots: {num_tr} Balance X:{balance_multiplier}')
        # self.dp.send_msg(f'{pair} using {calculated_stake} instead of {proposed_stake} | Trade Slots: {num_tr} Balance X:{balance_multiplier}')



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

        if (self.dp.runmode.value in ('live', 'dry_run')):
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
                if (signal == 1 and current_profit < -TP1):
                    if count_of_entries >= 1: 
                        stake_amount = stake_amount * 2
                    else:
                        stake_amount = stake_amount

                    return stake_amount

            # This then calculates current safety order size when below -Take Profit 2.
            if (last_fill > self.filldelay.value):
                if (current_profit < -TP2) and (current_profit > -TP3):
                    if count_of_entries >= 1: 
                        stake_amount = stake_amount * 2.5
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
        # sortBop = current_candle['sort_bop']
        
        display_profit = current_profit * 100

        if current_profit < -0.01:
            if pair in self.locked_stoploss:
                del self.locked_stoploss[pair]
                if (self.dp.runmode.value in ('live', 'dry_run')):
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

        # # in the future toggle these on certain conditions with indicators.
        # elif SLT1 is not None and current_profit > SLT1 and sortBop < -0.2:
        #     new_stoploss = (SLT1 - SLT0)
        #     level = 2
        # elif SLT0 is not None and current_profit > SLT0 and sortBop < -0.3:
        #     new_stoploss = (SLT1 - SLT0)
        #     level = 1

        if new_stoploss is not None:
            if pair not in self.locked_stoploss or new_stoploss > self.locked_stoploss[pair]:
                self.locked_stoploss[pair] = new_stoploss
                if (self.dp.runmode.value in ('live', 'dry_run')):
                    self.dp.send_msg(f'*** {pair} *** Profit {level} {display_profit:.3f}%% - New stoploss: {new_stoploss:.4f} activated')
                    logger.info(f'*** {pair} *** Profit {level} {display_profit:.3f}%% - New stoploss: {new_stoploss:.4f} activated')
            return self.locked_stoploss[pair]

        return self.stoploss


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
        if current_profit > TP3 and trade_duration > self.time0.value and self.use_roi1.value == True:
            return 'Roi 0 - Easy $$$'
        if current_profit > TP2 and trade_duration > self.time1.value and self.use_roi2.value == True:
            return 'Roi 1 - ol reliable' 
        if current_profit > TP1 and trade_duration > self.time2.value and self.use_roi3.value == True:
            return 'Roi 2 - Avg Joe'
        if current_profit > TP0 and trade_duration > self.time3.value and self.use_roi4.value == True:
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

    ### NORMAL INDICATORS ###
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        pair = metadata['pair']
        heikinashi = qtpylib.heikinashi(dataframe)

        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        dataframe['kalman'] = kalman_filter(dataframe['ha_close'], self.Q.value, self.R.value)

        # Initialize Hurst Cycles for startup errors
        cycle_period = 80
        harmonics = [0, 0, 0]
        harmonics[0] = 40
        harmonics[1] = 27
        harmonics[2] = 20

        if len(dataframe) < self.u_window_size.value:
            raise ValueError(f"Insufficient data points for FFT: {len(dataframe)}. Need at least {self.u_window_size.value} data points.")

        # Perform FFT to identify cycles with a rolling window
        freq, power = perform_fft(dataframe['kalman'], window_size=self.u_window_size.value)

        if len(freq) == 0 or len(power) == 0:
            raise ValueError("FFT resulted in zero or invalid frequencies. Check the data or the FFT implementation.")

        # Filter out the zero-frequency component and limit the frequency to below 500
        positive_mask = (1 / freq > self.l_window_size.value) & (1 / freq < self.u_window_size.value)
        positive_freqs = freq[positive_mask]
        positive_power = power[positive_mask]

        # Check if there are valid frequencies after filtering
        if len(positive_power) == 0:
            raise ValueError("No positive frequencies meet the filtering criteria. Adjust window sizes or check the data.")

        # Convert frequencies to periods
        cycle_periods = 1 / positive_freqs

        # Set a threshold to filter out insignificant cycles based on power
        power_threshold = 0 if len(positive_power) == 0 else 0.01 * np.max(positive_power)
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

        dataframe['dc_EWM'] = dataframe['kalman'].ewm(span=int(cycle_period)).mean()
        dataframe['dc_1/2'] = dataframe['kalman'].ewm(span=int(harmonics[0])).mean()
        dataframe['dc_1/3'] = dataframe['kalman'].ewm(span=int(harmonics[1])).mean()
        dataframe['dc_1/4'] = dataframe['kalman'].ewm(span=int(harmonics[2])).mean()

        # Apply rolling window operation to the 'OHLC4' column
        rolling_windowc = dataframe['kalman'].rolling(cycle_period) 
        rolling_windowh0 = dataframe['kalman'].rolling(int(harmonics[0]))
        rolling_windowh1 = dataframe['kalman'].rolling(int(harmonics[1])) 
        rolling_windowh2 = dataframe['kalman'].rolling(int(harmonics[2])) 

        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_valuec = rolling_windowc.apply(lambda x: np.ptp(x))
        ptp_valueh0 = rolling_windowh0.apply(lambda x: np.ptp(x))
        ptp_valueh1 = rolling_windowh1.apply(lambda x: np.ptp(x))
        ptp_valueh2 = rolling_windowh2.apply(lambda x: np.ptp(x))

        # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['cycle_move'] = ptp_valuec / dataframe['kalman']
        dataframe['h0_move'] = ptp_valueh0 / dataframe['kalman']
        dataframe['h1_move'] = ptp_valueh1 / dataframe['kalman']
        dataframe['h2_move'] = ptp_valueh2 / dataframe['kalman']

        dataframe['cycle_move_mean'] = dataframe['cycle_move'].rolling(self.cp).mean()        
        dataframe['h0_move_mean'] = dataframe['h0_move'].rolling(self.cp).mean()
        dataframe['h1_move_mean'] = dataframe['h1_move'].rolling(self.cp).mean() 
        dataframe['h2_move_mean'] = dataframe['h2_move'].rolling(self.cp).mean()

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['rsi_ma'] = ta.SMA(dataframe['rsi'], timeperiod=10)

        # WaveTrend using OHLC4 or HA close - 3/21
        ap = (0.25 * (dataframe['high'] + dataframe['low'] + dataframe["close"] + dataframe["open"]))
        
        dataframe['esa'] = ta.EMA(ap, timeperiod = 10)
        dataframe['d'] = ta.EMA(abs(ap - dataframe['esa']), timeperiod = 10)
        dataframe['wave_ci'] = (ap-dataframe['esa']) / (0.015 * dataframe['d'])
        dataframe['wave_t1'] = ta.EMA(dataframe['wave_ci'], timeperiod = 21)  
        dataframe['wave_t2'] = ta.SMA(dataframe['wave_t1'], timeperiod = 4)

        dataframe['max_high_h2'] = dataframe['kalman'].rolling(self.h2).max()
        dataframe['min_low_h2'] = dataframe['kalman'].rolling(self.h2).min()
        dataframe['closePos_h2'] = (dataframe['kalman'] - dataframe['min_low_h2']) / (dataframe['max_high_h2'] - dataframe['min_low_h2'])
        dataframe['closePos_h2_cod'] = detect_change_of_direction(dataframe['closePos_h2'])

        dataframe['max_high_h1'] = dataframe['kalman'].rolling(self.h1).max()
        dataframe['min_low_h1'] = dataframe['kalman'].rolling(self.h1).min()
        dataframe['closePos_h1'] = (dataframe['kalman'] - dataframe['min_low_h1']) / (dataframe['max_high_h1'] - dataframe['min_low_h1'])
        dataframe['closePos_h1_cod'] = detect_change_of_direction(dataframe['closePos_h1'])

        dataframe['max_high_h0'] = dataframe['kalman'].rolling(self.h0).max()
        dataframe['min_low_h0'] = dataframe['kalman'].rolling(self.h0).min()
        dataframe['closePos_h0'] = (dataframe['kalman'] - dataframe['min_low_h0']) / (dataframe['max_high_h0'] - dataframe['min_low_h0'])
        dataframe['closePos_h0_cod'] = detect_change_of_direction(dataframe['closePos_h0'])

        dataframe['max_high_cp'] = dataframe['kalman'].rolling(self.cp).max()
        dataframe['min_low_cp'] = dataframe['kalman'].rolling(self.cp).min()
        dataframe['closePos_cp'] = (dataframe['kalman'] - dataframe['min_low_cp']) / (dataframe['max_high_cp'] - dataframe['min_low_cp'])
        dataframe['closePos_cp_cod'] = detect_change_of_direction(dataframe['closePos_cp'])

        dataframe['closePos_cod_sum'] = dataframe['closePos_h2_cod'] + dataframe['closePos_h1_cod'] + dataframe['closePos_h0_cod'] + dataframe['closePos_cp_cod']

        dataframe['closePosSum'] = (dataframe['closePos_cp'] + dataframe['closePos_h0'] + dataframe['closePos_h1'] + dataframe['closePos_h2']) / 4
        dataframe['closePosSumSmoo'] = ta.SMA(dataframe['closePosSum'], self.h2)
        dataframe['closePosSumSmooDir'] = np.where(dataframe['closePosSumSmoo'] > dataframe['closePosSumSmoo'].shift(), self.sell_cls.value, self.buy_cls.value)
        dataframe['closePosDiff'] = abs(dataframe['closePosSum'] - dataframe['closePos_h2'])
        dataframe['closePosDiffMeanHalf'] = (dataframe['closePosDiff'].rolling(self.cp).mean()) / 2
        dataframe['closePosDiffMean'] = (dataframe['closePosDiff'].rolling(self.cp).mean()) 
        dataframe['closePosDiffMeanFast'] = (dataframe['closePosDiff'].rolling(self.h2).mean()) 
        dataframe['closePosMarket'] = np.where(dataframe['closePosDiff'] > 0, 0.1, -0.1)
        dataframe['closePosV'] = np.where(dataframe['closePosSum'] > dataframe['closePosSumSmoo'] , 0.15, -0.15)


        dataframe['candle_size'] = abs((dataframe['high'] - dataframe['low']) / dataframe['low'])
        dataframe['candle_size_sum'] = dataframe['candle_size'].rolling(self.cp).sum()
        dataframe['candle_size_lower'] = dataframe['candle_size'].rolling(self.cp).mean()
        dataframe['candle_size_upper'] = dataframe['candle_size_lower'] * 2

        dataframe['BOP'] = (ta.SMA(dataframe['close'], self.cp) - ta.SMA(dataframe['open'], self.cp)) / (ta.SMA(dataframe['high'], self.cp) - ta.SMA(dataframe['low'], self.cp))
        dataframe['BOP_SMA'] = ta.SMA(dataframe['BOP'], self.h0)
        dataframe['BOP_SMA_2'] = ta.SMA(dataframe['BOP'], self.h2)
        dataframe['zero'] = 0


        h = self.h2
        r = 8.0
        x_0 = self.cp
        smoothColors = False
        lag = 0
        nadaraya_watson(dataframe, h, r, x_0, smoothColors, lag, mult = 2.5)

        dataframe['nw_width'] = dataframe['nw_up'] - dataframe['nw_down']
        dataframe['nw_width_mean'] = dataframe['nw_width'].rolling(self.h2).mean()
        dataframe['signal_UP'] = np.where(dataframe['nw_width'] > 0, dataframe['nw_width'], 0)
        dataframe['signal_DN'] = np.where(dataframe['nw_width'] < 0, dataframe['nw_width'], 0)
        dataframe['signal_UP'] = dataframe['signal_UP'].ffill()
        dataframe['signal_DN'] = dataframe['signal_DN'].ffill()
        dataframe['nw_width_up'] = dataframe['signal_UP'].rolling(self.h2).mean()
        dataframe['nw_width_dn'] = dataframe['signal_DN'].rolling(self.h2).mean()
        dataframe['channellowermid'] = dataframe['yhat1'] * (1 - (dataframe['h2_move_mean']/2))
        dataframe['channeluppermid'] = dataframe['yhat1'] * (1 + (dataframe['h2_move_mean']/2))
        dataframe['channellower'] = dataframe['yhat1'] * (1 - dataframe['h2_move_mean'])
        dataframe['channelupper'] = dataframe['yhat1'] * (1 + dataframe['h2_move_mean'])

        dataframe = detect_all_bullish_patterns(dataframe)

        dataframe['slope'] = dataframe['yhat1'].diff()
        dataframe['angle_degrees'] = dataframe['slope'].apply(lambda x: float(np.degrees(np.arctan(float(x)))) if x is not None else None)
        if not self.dp.runmode.value in ("backtest", "plot", "hyperopt"):
            logger.info(f'{pair} - DC: {self.cp:.2f} | 1/2: {self.h0:.2f} | 1/3: {self.h1:.2f} | 1/4: {self.h2:.2f}')

        return dataframe


    ### ENTRY CONDITIONS ###
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:


        df.loc[
            (
                (self.use0.value == True) &
                (df['closePos_h2'] > df['closePos_h2'].shift()) &
                (df['closePos_h2'] < self.buy_cl1.value) &
                (df['closePos_h1'] < self.buy_cl2.value) &
                (df['closePos_h0'] < self.buy_cl3.value) &
                (df['closePosDiffMeanHalf'] > df['closePosDiff']) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'WT123')

        df.loc[
            (
                (self.use1.value == True)&
                (df['closePos_h2'] < self.buy_cl1.value) &
                (df['closePos_h2'] > df['closePos_h2'].shift()) &
                (df['closePosDiffMeanHalf'] > df['closePosDiff']) &
                (df['closePosDiffMeanFast'] < df['closePosDiffMean']) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'WT1')

   
        df.loc[
            (
                (self.use2.value == True)&
                (df['closePos_h2'] < self.buy_cl1.value) &
                (df['closePos_h1'] < self.buy_cl2.value) &
                (df['closePos_h2'] > df['closePos_h2'].shift()) &
                (df['closePosDiffMeanHalf'] > df['closePosDiff']) &
                (df['closePosDiffMeanFast'] < df['closePosDiffMean']) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'WT12')
        df.loc[
            (
                (self.use3.value == True) &
                # (df['kalman'] < df['ma_hi']) &
                (df['closePos_h1'] < self.buy_cl2.value) &
                (df['closePos_h0'] < self.buy_cl3.value) &
                (df['closePos_h2'] > df['closePos_h2'].shift()) &
                (df['closePosDiffMeanHalf'] > df['closePosDiff']) &
                (df['closePosDiffMeanFast'] < df['closePosDiffMean']) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'W23')

        df.loc[
            (
                (self.use4.value == True) &
                (df['closePosSum'] < df['closePosSumSmooDir']) &
                (df['closePosSumSmoo'] < df['closePosSumSmooDir']) &
                (self.buy_cls.value > df['closePosSumSmoo']) &
                (qtpylib.crossed_above(df['closePosSum'], df['closePosSumSmoo'])) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'cp x')

        df.loc[
            (
                (self.use5.value == True) &
                # (df['kalman'] < df['yhat1']) &
                (df['closePos_h1'] < self.buy_cl2.value) &
                (df['BOP_SMA'] < 0) &
                (qtpylib.crossed_above(df['BOP'], df['BOP_SMA'])) &
                (df['closePosDiffMeanHalf'] > df['closePosDiff']) &
                (df['closePosDiffMeanFast'] < df['closePosDiffMean']) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'BOP X')

        df.loc[
            (
                (self.use6.value == True) &
                (df['closePosSum'] < self.buy_cls.value) &
                (df['closePosSum'] < df['closePosSumSmoo']) &
                (df['closePos_h2'] > df['closePos_h2'].shift()) &
                (df['closePosDiffMeanHalf'] > df['closePosDiff']) &
                (df['closePosDiffMeanFast'] < df['closePosDiffMean']) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'Sum')

        return df


    ### EXIT CONDITIONS ###
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (self.use10.value == True) &
                (df['rsi'] > self.sell_rsi.value) &
                (df['closePos_h2'] > self.sell_cl1.value) &
                (df['closePos_h1'] > self.sell_cl2.value) &
                (df['closePos_h0'] > self.sell_cl3.value) &
                (df['closePos_h2'] < df['closePos_h2'].shift()) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'WT 123')


        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] > self.sell_rsi.value) &
                (df['closePos_h2'] > self.sell_cl1.value) &
                (df['closePos_h1'] > self.sell_cl2.value) &
                (self.use11.value == True) &
                (df['closePos_h2'] < df['closePos_h2'].shift()) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'WT 12')

        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] > self.sell_rsi.value) &
                (self.use12.value == True) &
                (df['closePos_h1'] > self.sell_cl2.value) &
                (df['closePos_h0'] > self.sell_cl3.value) &
                (df['closePos_h2'] < df['closePos_h2'].shift()) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'WT 23')

        df.loc[
            (
                # Signal: RSI crosses above 30
                # (df['rsi'] > self.sell_rsi.value) &
                (self.use13.value == True) &
                (df['closePosSum'] > df['closePosSumSmooDir']) &
                (df['closePosSum'] > df['closePosSumSmoo']) &
                (df['closePos_h2'] < df['closePos_h2'].shift()) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'cp x')

  
        return df


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
    src = df['kalman']
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
    df['yhat1'] = yhat1
    df['yhat2'] = yhat2
    df['nw_up'] = nwe_up
    df['nw_down'] = nwe_down
    df['nw_entry'] = nwe_entry
    df['nw_exit'] = nwe_exit
    df['wasBearish'] = wasBearish
    df['wasBullish'] = wasBullish
    df['isBearish'] = isBearish
    df['isBullish'] = isBullish
    df['isBearishChange'] = isBearishChange
    df['isBullishChange'] = isBullishChange
    df['isBullishCross'] = isBullishCross
    df['isBearishCross'] = isBearishCross
    df['isBullishSmooth'] = isBullishSmooth
    df['isBearishSmooth'] = isBearishSmooth
    df['colorByCross'] = colorByCross
    df['colorByRate'] = colorByRate
    df['plotColor'] = plotColor
    df['alertBullish'] = alertBullish
    df['alertBearish'] = alertBearish

    return df

def detect_change_of_direction(indicator_values):
    """
    Detects changes in direction in a trading indicator.

    Args:
        indicator_values (list or np.ndarray or pd.Series): The trading indicator values.

    Returns:
        np.ndarray: An array of the same length with 1 for upward change, -1 for downward change, and 0 otherwise.
    """
    # Ensure the input is a NumPy array for efficient computation
    indicator_values = np.asarray(indicator_values)
    
    # Compute the difference between consecutive values
    diff = np.diff(indicator_values)
    
    # Create an array to hold the results (same length as input, pad with 0)
    change_direction = np.zeros_like(indicator_values, dtype=int)
    
    # Identify upward changes (1) and downward changes (-1)
    change_direction[1:] = np.sign(diff)
    
    # Detect changes of direction
    result = np.zeros_like(change_direction, dtype=int)
    result[1:] = np.where(np.diff(change_direction) != 0, change_direction[1:], 0)
    
    return result

def calculate_sine_wave(src, hp_period):

    pi = 2*np.arcsin(1)
    alpha1 = (np.cos(0.707*2*pi/hp_period) + np.sin(0.707*2*pi/hp_period)-1)/np.cos(0.707*2*pi/hp_period)
    
    hp = np.zeros(len(src))
    for i in range(2, len(src)):
        hp[i] = (1-alpha1/2)*(1-alpha1/2)*(src[i]-2*src[i-1]+src[i-2])+2*(1-alpha1)*hp[i-1]-(1-alpha1)*(1-alpha1)*hp[i-2]
    
    filt = np.zeros(len(src))
    for i in range(6, len(src)):
        filt[i] = (7*hp[i]+6*hp[i-1]+5*hp[i-2]+4*hp[i-3]+3*hp[i-4]+2*hp[i-5]+hp[i-6])/28
    
    wave = np.zeros(len(src))
    for i in range(2, len(src)):
        wave[i] = (filt[i]+filt[i-1]+filt[i-2])/3
    
    pwr = np.zeros(len(src))
    for i in range(2, len(src)):
        pwr[i] = (filt[i]**2+filt[i-1]**2+filt[i-2]**2)/3
    
    for i in range(2, len(src)):
        wave[i] = wave[i]/np.sqrt(pwr[i])

    return wave

def detect_all_bullish_patterns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Detects all bullish candlestick patterns and updates the dataframe with a 'bullish_pattern' column.

    Parameters:
        dataframe (pd.DataFrame): Dataframe containing OHLC data with columns ['open', 'high', 'low', 'close'].

    Returns:
        pd.DataFrame: Updated dataframe with a new column 'bullish_pattern' indicating bullish candlestick patterns.
    """
    # Ensure necessary columns are present
    required_columns = ['open', 'high', 'low', 'close']
    for col in required_columns:
        if col not in dataframe.columns:
            raise ValueError(f"Missing required column: {col}")

    # List of bullish candlestick patterns
    bullish_patterns = {
        'CDL_ENGULFING': talib.CDLENGULFING,            # Bullish Engulfing
        'CDL_HAMMER': talib.CDLHAMMER,                  # Hammer
        'CDL_INVERTED_HAMMER': talib.CDLINVERTEDHAMMER, # Inverted Hammer
        'CDL_MORNING_STAR': talib.CDLMORNINGSTAR,       # Morning Star
        'CDL_PIERCING': talib.CDLPIERCING,              # Piercing Line
        'CDL_THREE_INSIDE_UP': talib.CDL3INSIDE,        # Three Inside Up
        'CDL_THREE_WHITE_SOLDIERS': talib.CDL3WHITESOLDIERS, # Three White Soldiers
        'CDL_KICKING': talib.CDLKICKING,                # Kicking
    }

    # Initialize the 'bullish_pattern' column to False
    dataframe['bullish_pattern'] = False

    # Apply each pattern and update the 'bullish_pattern' column
    for pattern_name, pattern_func in bullish_patterns.items():
        dataframe[pattern_name] = pattern_func(
            dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close']
        )
        dataframe[pattern_name] = dataframe[pattern_name] > 0  # Convert to boolean (True for bullish)
        dataframe['bullish_pattern'] |= dataframe[pattern_name]  # Combine all bullish patterns

    # Optionally, drop individual pattern columns
    # dataframe.drop(columns=bullish_patterns.keys(), inplace=True)

    return dataframe

# Function to implement a simple 1D Kalman Filter
def kalman_filter(data, Q=0.001, R=0.1):
    """
    Applies a simple Kalman Filter to a time series.
    Args:
        data: numpy array of observations (closing prices)
        Q: Process noise covariance (higher = more uncertain about dynamics)
        R: Measurement noise covariance (higher = less trust in observations)
    Returns:
        xhat: Array of filtered estimates
    """
    n = len(data)
    xhat = np.zeros(n)  # a posteriori estimate of x
    P = np.zeros(n)  # a posteriori error estimate
    xhat_pred = np.zeros(n)  # a priori estimate of x
    P_pred = np.zeros(n)  # a priori error estimate
    K = np.zeros(n)  # Kalman gain

    # Initialize
    xhat[0] = data[0]
    P[0] = 1.0

    for k in range(1, n):
        # Prediction step
        xhat_pred[k] = xhat[k - 1]
        P_pred[k] = P[k - 1] + Q

        # Update step
        K[k] = P_pred[k] / (P_pred[k] + R)
        xhat[k] = xhat_pred[k] + K[k] * (data[k] - xhat_pred[k])
        P[k] = (1 - K[k]) * P_pred[k]

    return xhat