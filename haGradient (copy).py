
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
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)


class haGradient(IStrategy):

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
    timeframe = '15m'
    exit_profit_only = True ### No selling at a loss
    ignore_roi_if_entry_signal = True
    process_only_new_candles = True
    can_short = True
    use_exit_signal = True
    startup_candle_count = 20
    use_custom_stoploss = True


    locked_stoploss = {}
    minimal_roi = {
      "0": 0.214,
      "81": 0.105,
      "346": 0.053,
      "789": 0}
    

    # Stoploss:
    stoploss = -0.12  # Fail Safe Default do not hyper opt Stoploss Space

    # Trailing stop:
    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = 0.005 # value loaded from strategy
    trailing_stop_positive_offset = 0.035 # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy

    # CooldownPeriod 
    cooldown_lookback = IntParameter(0, 12, default=5, space="protection", optimize=True, load=True)
    
    # StoplossGuard    
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True, load=True)
    stop_duration = IntParameter(1, 24, default=12, space="protection", optimize=True, load=True)
    stop_protection_only_per_pair = BooleanParameter(default=True, space="protection", optimize=True, load=True)
    stop_protection_only_per_side = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    stop_protection_trade_limit = IntParameter(1, 10, default=4, space="protection", optimize=True, load=True)
    stop_protection_required_profit = DecimalParameter(-0.10, 0.01, default=-0.04, decimals=2, space="protection", optimize=True, load=True)

    # LowProfitPairs    
    use_lowprofit_protection = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    lowprofit_protection_lookback = IntParameter(1, 24, default=10, space="protection", optimize=True, load=True)
    lowprofit_trade_limit = IntParameter(1, 10, default=6, space="protection", optimize=True, load=True)
    lowprofit_stop_duration = IntParameter(1, 24, default=24, space="protection", optimize=True, load=True)
    lowprofit_required_profit = DecimalParameter(-0.10, 0.00, default=-0.04, decimals=2, space="protection", optimize=True, load=True)
    lowprofit_only_per_pair = BooleanParameter(default=True, space="protection", optimize=True, load=True)

    # MaxDrawdown    
    use_maxdrawdown_protection = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    maxdrawdown_protection_lookback = IntParameter(1, 10, default=6, space="protection", optimize=True, load=True)
    maxdrawdown_trade_limit = IntParameter(1, 20, default=10, space="protection", optimize=True, load=True)
    maxdrawdown_stop_duration = IntParameter(1, 24, default=6, space="protection", optimize=True, load=True)
    maxdrawdown_allowed_drawdown = DecimalParameter(-0.10, 0.00, default=-0.04, decimals=2, space="protection", optimize=True, load=True)

    # Custom Entry
    increment = DecimalParameter(low=1.0005, high=1.002, default=1.001, decimals=4 ,space='buy', optimize=True, load=True)
    entryX = DecimalParameter(low=0.98, high=1.01, default=1.00, decimals=3 ,space='buy', optimize=True, load=True)
    last_entry_price = None

    # Position Management
    # base_trades = IntParameter(4, 10, default=6, space="protection", optimize=True, load=True)
    # stepSize = IntParameter(1, 4, default=2, space="protection", optimize=True, load=True)

    # indicators
    u_window_size = IntParameter(70, 150, default=120, space='buy', optimize=True, load=True)
    l_window_size = IntParameter(20, 50, default=42, space='buy', optimize=True, load=True)
    cp_offset = IntParameter(1, 10, default=1, space='buy', optimize=True, load=True)
    h0_offset = IntParameter(1, 5, default=1, space='buy', optimize=True, load=True)
    h1_offset = IntParameter(1, 5, default=1, space='buy', optimize=True, load=True)
    h2_offset = IntParameter(1, 5, default=1, space='buy', optimize=True, load=True)
    sum_offset = IntParameter(1, 15, default=1, space='buy', optimize=True, load=True)

    suppress_h2 = IntParameter(-20, 20, default=-8, space='buy', optimize=True, load=True)
    suppress_h1 = IntParameter(-20, 20, default=-10, space='buy', optimize=True, load=True)
    suppress_h0 = IntParameter(-20, 20, default=-10, space='buy', optimize=True, load=True)
    suppress_cp = IntParameter(-35, 35, default=-20, space='buy', optimize=True, load=True)
    suppress_sum = IntParameter(-45, 45, default=-20, space='buy', optimize=True, load=True)

    # Entry Exit Logic Selection
    use0 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use1 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use2 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use3 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use4 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use5 = BooleanParameter(default=False, space="buy", optimize=True, load=True)
    # use6 = BooleanParameter(default=False, space="buy", optimize=True, load=True)
    # use7 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use8 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use9 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use10 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use11 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use12 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use13 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use14 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    # use15 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    # use16 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    # use17 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use18 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use19 = BooleanParameter(default=True, space="sell", optimize=True, load=True)

    def custom_entry_price(self, pair: str, trade: Optional['Trade'], current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)

        entry_price = ((dataframe['close'].iat[-1] + dataframe['open'].iat[-1] + proposed_rate + proposed_rate) / 4) * self.entryX.value
        if entry_price >= proposed_rate:
            entry_price = proposed_rate
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
        # balance = self.wallets.get_total_stake_amount()
        # max_balance = 40000
        # if balance > 10000:
        #     max_open_trades_at_max_balance = self.base_trades.value + int((max_balance - 10000) / 10000) * self.stepSize.value
        #     self.config['max_open_trades'] = min(self.base_trades.value + int((balance - 10000) / 10000) * self.stepSize.value, max_open_trades_at_max_balance)
        # else:
        #     self.config['max_open_trades'] = self.base_trades.value

        return entry_price

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

        # in the future toggle these on certain conditions with indicators.
        elif SLT1 is not None and current_profit > SLT1 and self.use18.value == True:
            new_stoploss = (SLT1 - SLT0)
            level = 2
        elif SLT0 is not None and current_profit > SLT0 and self.use19.value == True:
            new_stoploss = (SLT1 - SLT0)
            level = 1

        if new_stoploss is not None:
            if pair not in self.locked_stoploss or new_stoploss > self.locked_stoploss[pair]:
                self.locked_stoploss[pair] = new_stoploss
                if (self.dp.runmode.value in ('live', 'dry_run')):
                    self.dp.send_msg(f'*** {pair} *** Profit {level} {display_profit:.3f}%% - New stoploss: {new_stoploss:.4f} activated')
                    logger.info(f'*** {pair} *** Profit {level} {display_profit:.3f}%% - New stoploss: {new_stoploss:.4f} activated')
            return self.locked_stoploss[pair]

        return self.stoploss

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                       rate: float, time_in_force: str, exit_reason: str,
                       current_time: datetime, **kwargs) -> bool:
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        trade_duration = (current_time - trade.open_date_utc).seconds / 60 
        # heikinashi = qtpylib.heikinashi(dataframe)

        # dataframe['ha_open'] = heikinashi['open']
        # dataframe['ha_close'] = heikinashi['close']
        # hodl = last_candle['frac_HODL'] 
        # TP2 = current_candle['h0_move_mean']

        # # Handle freak events and stoploss engagement
        # if hodl == 1 and trade.calc_profit_ratio(rate) > TP2:
        #     logger.info(f"{trade.pair} HODL!!!")
        #     return False

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

        # Initialize Hurst Cycles for startup errors
        cycle_period = 80
        harmonics = [0, 0, 0]
        harmonics[0] = 40
        harmonics[1] = 27
        harmonics[2] = 20

        if len(dataframe) < self.u_window_size.value:
            raise ValueError(f"Insufficient data points for FFT: {len(dataframe)}. Need at least {self.u_window_size.value} data points.")

        # Perform FFT to identify cycles with a rolling window
        freq, power = perform_fft(dataframe['ha_close'], window_size=self.u_window_size.value)

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

        dataframe['dc_EWM'] = dataframe['ha_close'].ewm(span=int(cycle_period)).mean()
        dataframe['dc_1/2'] = dataframe['ha_close'].ewm(span=int(harmonics[0])).mean()
        dataframe['dc_1/3'] = dataframe['ha_close'].ewm(span=int(harmonics[1])).mean()
        dataframe['dc_1/4'] = dataframe['ha_close'].ewm(span=int(harmonics[2])).mean()
        dataframe['dc-trend'] = dataframe['dc_EWM'].diff() / dataframe['dc_EWM']

        # Apply rolling window operation to the 'OHLC4' column
        rolling_windowc = dataframe['ha_close'].rolling(cycle_period) 
        rolling_windowh0 = dataframe['ha_close'].rolling(int(harmonics[0]))
        rolling_windowh1 = dataframe['ha_close'].rolling(int(harmonics[1])) 
        rolling_windowh2 = dataframe['ha_close'].rolling(int(harmonics[2])) 

        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_valuec = rolling_windowc.apply(lambda x: np.ptp(x))
        ptp_valueh0 = rolling_windowh0.apply(lambda x: np.ptp(x))
        ptp_valueh1 = rolling_windowh1.apply(lambda x: np.ptp(x))
        ptp_valueh2 = rolling_windowh2.apply(lambda x: np.ptp(x))

        # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['cycle_move'] = ptp_valuec / dataframe['ha_close']
        dataframe['h0_move'] = ptp_valueh0 / dataframe['ha_close']
        dataframe['h1_move'] = ptp_valueh1 / dataframe['ha_close']
        dataframe['h2_move'] = ptp_valueh2 / dataframe['ha_close']

        dataframe['cycle_move_mean'] = dataframe['cycle_move'].rolling(self.cp).mean()        
        dataframe['h0_move_mean'] = dataframe['h0_move'].rolling(self.cp).mean()
        dataframe['h1_move_mean'] = dataframe['h1_move'].rolling(self.cp).mean() 
        dataframe['h2_move_mean'] = dataframe['h2_move'].rolling(self.cp).mean()

        dataframe['ha_dir'] = np.where(dataframe['ha_close'] > dataframe['ha_open'].shift(), 1, -1)
        dataframe['ha_trend_cp'] = dataframe['ha_dir'].rolling(self.cp).sum()
        dataframe['ha_trend_lower_cp'] = (dataframe['ha_trend_cp'].rolling(self.cp).min() + self.cp_offset.value)
        dataframe['ha_trend_upper_cp'] = (dataframe['ha_trend_cp'].rolling(self.cp).max() - self.cp_offset.value)
        dataframe['ha_trend_h0'] = dataframe['ha_dir'].rolling(self.h0).sum()
        dataframe['ha_trend_lower_h0'] = (dataframe['ha_trend_h0'].rolling(self.h0).min() + self.h0_offset.value)
        dataframe['ha_trend_upper_h0'] = (dataframe['ha_trend_h0'].rolling(self.h0).max() - self.h0_offset.value)
        dataframe['ha_trend_h1'] = dataframe['ha_dir'].rolling(self.h1).sum()
        dataframe['ha_trend_lower_h1'] = (dataframe['ha_trend_h1'].rolling(self.h1).min() + self.h1_offset.value)
        dataframe['ha_trend_upper_h1'] = (dataframe['ha_trend_h1'].rolling(self.h1).max() - self.h1_offset.value)
        dataframe['ha_trend_h2'] = dataframe['ha_dir'].rolling(self.h2).sum()
        dataframe['ha_trend_lower_h2'] = (dataframe['ha_trend_h2'].rolling(self.h2).min() + self.h2_offset.value)
        dataframe['ha_trend_upper_h2'] = (dataframe['ha_trend_h2'].rolling(self.h2).max() - self.h2_offset.value)

        dataframe['ha_trend_sum'] = dataframe['ha_trend_cp'] + dataframe['ha_trend_h0'] + dataframe['ha_trend_h1'] + dataframe['ha_trend_h2']
        dataframe['ha_trend_lower_sum'] = (dataframe['ha_trend_sum'].rolling(self.cp).min() + self.sum_offset.value)
        dataframe['ha_trend_upper_sum'] = (dataframe['ha_trend_sum'].rolling(self.cp).max() - self.sum_offset.value)
        dataframe['ha_trend_sum_mid'] = ((dataframe['ha_trend_upper_sum'] - abs(dataframe['ha_trend_lower_sum']))/2) 

        h = self.h2
        r = 8.0
        x_0 = self.h2
        smoothColors = False
        lag = 0
        nadaraya_watson(dataframe, h, r, x_0, smoothColors, lag, mult = 2.5)

        dataframe['channellowermid'] = dataframe['yhat1'] * (1 - (dataframe['h2_move_mean']/2))
        dataframe['channeluppermid'] = dataframe['yhat1'] * (1 + (dataframe['h2_move_mean']/2))
        dataframe['channellower'] = dataframe['yhat1'] * (1 - dataframe['h2_move_mean'])
        dataframe['channelupper'] = dataframe['yhat1'] * (1 + dataframe['h2_move_mean'])

        dataframe['zero'] = 0

        return dataframe



    ### ENTRY CONDITIONS ###
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (self.use0.value == True) &
                (df['ha_trend_cp'] < self.suppress_cp.value) &
                (df['ha_trend_cp'] > df['ha_trend_lower_cp']) &
                (df['ha_trend_cp'].shift() < df['ha_trend_lower_cp'].shift()) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'ha trend cp')

#        df.loc[
#            (
#                (self.use1.value == True) &
#                (df['ha_trend_h0'] < self.suppress_h0.value) &
#                (df['ha_trend_h0'] > df['ha_trend_lower_h0']) &
#                (df['ha_trend_h0'].shift() < df['ha_trend_lower_h0'].shift()) &
#                (df['volume'] > 0)   # Make sure Volume is not 0
#            ),
#            ['enter_long', 'enter_tag']] = (1, 'ha trend h0')

        df.loc[
            (
                (self.use2.value == True) &
                (df['ha_trend_h1'] < self.suppress_h1.value) &
                (df['ha_trend_h1'] > df['ha_trend_lower_h1']) &
                (df['nw_up'] > df['ha_close']) &
                (df['ha_trend_h1'].shift() < df['ha_trend_lower_h1'].shift()) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'ha trend h1')

        df.loc[
            (
                (self.use3.value == True) &
                (df['ha_trend_lower_h2'] < 0 ) &
                (df['yhat1'] > df['ha_open'] ) &
                # (df['ha_trend_h2'] > df['ha_trend_h1'] ) &
                (df['ha_trend_h2'] < self.suppress_h2.value) &
                (df['ha_trend_h2'] > df['ha_trend_lower_h2']) &
                (df['ha_trend_h2'].shift() < df['ha_trend_lower_h2'].shift()) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'ha trend h2')


        df.loc[
            (
                (self.use4.value == True) &
                (df['ha_trend_h2'] < self.suppress_sum.value) &
                (df['ha_trend_sum'] > df['ha_trend_lower_sum']) &
                (df['ha_trend_sum'].shift() < df['ha_trend_lower_sum'].shift()) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'ha trend sum')

        return df


    ### EXIT CONDITIONS ###
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (self.use10.value == True) &
                (0 < df['ha_trend_upper_cp']) &
                (df['ha_trend_cp'] < df['ha_trend_upper_cp']) &
                (df['ha_trend_cp'].shift() > df['ha_trend_upper_cp'].shift()) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'ha trend cp')

        df.loc[
            (
                (self.use11.value == True) &
                (df['ha_trend_h0'] > abs(self.suppress_h0.value)) &
                (df['ha_trend_h0'] < df['ha_trend_upper_h0']) &
                (df['ha_trend_h0'].shift() > df['ha_trend_upper_h0'].shift()) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'ha trend h0')

        df.loc[
            (
                (self.use12.value == True) &
                (df['ha_trend_h1'] > abs(self.suppress_h1.value)) &
                (df['ha_trend_h1'] < df['ha_trend_upper_h1']) &
                (df['ha_trend_h1'].shift() > df['ha_trend_upper_h1'].shift()) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'ha trend h1')

        df.loc[
            (
                (self.use13.value == True) &
                (df['ha_trend_h2'] > abs(self.suppress_h2.value)) &
                (df['ha_trend_h2'] < df['ha_trend_upper_h2']) &
                (df['ha_trend_h2'].shift() > df['ha_trend_upper_h2'].shift()) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'ha trend h2')

        df.loc[
            (
                (self.use14.value == True) &
                (df['ha_trend_sum'] < df['ha_trend_upper_sum']) &
                (df['ha_trend_sum'].shift() > df['ha_trend_upper_sum'].shift()) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'ha trend sum')


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
    src = df['ha_close']
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