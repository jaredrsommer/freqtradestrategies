# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,IStrategy, IntParameter, RealParameter, merge_informative_pair)
from typing import Dict, List, Optional, Union, Tuple
from functools import reduce
from pandas import DataFrame
import numpy as np
import pandas as pd
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import logging
from datetime import datetime, timedelta, timezone
from freqtrade.persistence import Trade, Order 
from pykalman import KalmanFilter
logger = logging.getLogger(__name__)


class TwoCandleTheory(IStrategy):
    """
    TwoCandleTheory Strategy - Converted from TradingView
    
    Based on the combination of multiple indicators:
    - RSI
    - Volume
    - VWAP
    - Supertrend
    - VWMA
    - PSAR
    - EMA
    
    Strategy has two main entry signals:
    1. Two Candle Theory - Based on volume and candle patterns with indicator confirmations
    2. Golden Cross - Based on crossovers of Supertrend/VWMA with VWAP
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Minimal ROI designed for the strategy.
    # This dict defines the minimum ROI designed for the strategy
    # ROI table is automatically adjusted based on the risk:reward input
    minimal_roi = {
        "0": 0.1  # ROI is dynamically adjusted at runtime based on risk:reward
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.1  # This will be dynamically set based on previous candle

    # Trailing stoploss (not used in this strategy as we use fixed R:R)
    trailing_stop = False
    
    # Timeframe for the strategy
    timeframe = '5m'

    # Run "populate_indicators" only for new candle (using .iloc[-1])
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30
    
    can_short = False
    
    custom_info = {}
    
    process_noise_slow = DecimalParameter(0.01, 1.0, default=0.0005, space="sell")
    measurement_noise_slow = DecimalParameter(0.01, 1.0, default=1, space="sell") 
    

    # Strategy parameters
    buy_params = {
        'volume_cutoff': 50000,
        'profit_multiplier': 2.0,
        'ema_length': 15,  # 0 means disabled
        'rsi_length': 14,
        'rsi_ma_length': 14,
        'supertrend_length': 10,
        'supertrend_multiplier': 2.0,
        'vwma_length': 20,
        'psar_start': 0.02,
        'psar_increment': 0.02,
        'psar_maximum': 0.2,
        'measurement_noise_slow': 1.0,
        'process_noise_slow': 0.0005,
    }
    
    def apply_kalman(self, df: DataFrame, process_noise: float, measurement_noise: float) -> DataFrame:
        # State transition matrix (includes velocity)
        A = [[1, 1],    # x(t) = x(t-1) + v(t-1)
            [0, 1]]    # v(t) = v(t-1)
        
        # Observation matrix
        H = [[1, 0]]    # We only observe the position (price), not velocity
        
        kf = KalmanFilter(
            transition_matrices=A,
            observation_matrices=H,
            initial_state_mean=[df['close'].iloc[0], 0],  # [position, velocity]
            initial_state_covariance=[[1, 0], [0, 1]],
            observation_covariance=measurement_noise,
            transition_covariance=[[process_noise, 0], [0, process_noise]]
        )
        
        state_means, _ = kf.filter(df['close'].values)
        # Return both position and velocity estimates
        return state_means[:, 0], state_means[:, 1]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.buy_params['rsi_length'])
        dataframe['rsi_ma'] = ta.SMA(dataframe['rsi'], timeperiod=self.buy_params['rsi_ma_length'])
        
        # EMA Filter (if enabled)
        if self.buy_params['ema_length'] > 0:
            dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.buy_params['ema_length'])
        else:
            dataframe['ema'] = 0  # Placeholder

        # VWMA
        dataframe['vwma'] = self.vwma(dataframe, length=self.buy_params['vwma_length'])
        
        # VWAP
        dataframe['vwap'] = self.vwap(dataframe)
        
        # PSAR
        dataframe['psar'] = ta.SAR(dataframe, 
                                  acceleration=self.buy_params['psar_start'], 
                                  maximum=self.buy_params['psar_maximum'])
        
        # Supertrend
        supertrend = self.supertrend(dataframe, 
                                    length=self.buy_params['supertrend_length'], 
                                    multiplier=self.buy_params['supertrend_multiplier'])
        
        dataframe['supertrend_direction'] = supertrend['direction']
        dataframe['supertrend_long_stop'] = supertrend['long_stop']
        dataframe['supertrend_short_stop'] = supertrend['short_stop']
        
        # Volume condition
        dataframe['high_volume'] = (dataframe['volume'] > self.buy_params['volume_cutoff']).astype('int')
        dataframe['volume_filter'] = ((dataframe['high_volume'] > 0) & 
                                     (dataframe['high_volume'].shift(1) > 0)).astype('int')
                                     
        # Candle patterns for volume filter
        dataframe['green_candle'] = (dataframe['close'] > dataframe['open']).astype('int')
        dataframe['red_candle'] = (dataframe['close'] < dataframe['open']).astype('int')
        
        # Volume Long/Short Entry Conditions
        dataframe['volume_long_cond'] = ((dataframe['green_candle'] > 0) & 
                                        (dataframe['green_candle'].shift(1) > 0)).astype('int')
        
        dataframe['volume_short_cond'] = ((dataframe['red_candle'] > 0) & 
                                         (dataframe['red_candle'].shift(1) > 0)).astype('int')
        
        dataframe['volume_long_entry'] = ((dataframe['volume_filter'] > 0) & 
                                         (dataframe['volume_long_cond'] > 0)).astype('int')
        
        dataframe['volume_short_entry'] = ((dataframe['volume_filter'] > 0) & 
                                          (dataframe['volume_short_cond'] > 0)).astype('int')
        
        # Crossover signals for Golden Cross
        dataframe['supertrend_cross_bull'] = qtpylib.crossed_above(
            dataframe['supertrend_long_stop'], dataframe['vwap']
        ).astype('int')
        
        dataframe['supertrend_cross_bear'] = qtpylib.crossed_below(
            dataframe['supertrend_short_stop'], dataframe['vwap']
        ).astype('int')
        
        dataframe['vwma_cross_bull'] = qtpylib.crossed_above(
            dataframe['vwma'], dataframe['vwap']
        ).astype('int')
        
        dataframe['vwma_cross_bear'] = qtpylib.crossed_below(
            dataframe['vwma'], dataframe['vwap']
        ).astype('int')
        
        kalman_slow_pos, kalman_slow_vel = self.apply_kalman(
            dataframe,
            self.process_noise_slow.value,
            self.measurement_noise_slow.value 
        )
        dataframe['kalman_slow'] = kalman_slow_pos
        dataframe['kalman_slow_vel'] = (kalman_slow_vel/kalman_slow_pos) * 100
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        conditions = []
        
        # RSI Filter - Long
        rsi_long_filter = (
            (dataframe['rsi'] > 10) &
            (dataframe['rsi'] <= 58)
        )
        
        # VWAP Filter - Long
        vwap_long_filter = (
            (dataframe['close'] > dataframe['vwap'])
        )
        
        # Supertrend Filter - Long
        supertrend_long_filter = (
            (dataframe['close'] > dataframe['supertrend_long_stop'])
        )
        
        # VWMA Filter - Long
        vwma_long_filter = (
            (dataframe['close'] > dataframe['vwma'])
        )
        
        # PSAR Filter - Long
        psar_long_filter = (
            (dataframe['close'] > dataframe['psar'])
        )
        
        # Kalman Filter - Long
        kalman_long_filter = (
            (dataframe['close'] > dataframe['kalman_slow'])
        )
        
        # EMA Filter - Long (if enabled)
        if self.buy_params['ema_length'] > 0:
            ema_long_filter = (dataframe['close'] > dataframe['ema'])
        else:
            ema_long_filter = True  # Not used if ema_length is 0
        
        # Volume condition with two green candles - Long
        volume_long_condition = (
            (dataframe['volume_long_entry'] > 0)
        )
        
        # Combine all filters for Two Candle Theory - Long
        long_conditions = (
            rsi_long_filter &
            vwap_long_filter &
            supertrend_long_filter &
            vwma_long_filter &
            psar_long_filter &
            kalman_long_filter &
            volume_long_condition 
        )
        
        # Golden Cross Long Entry
        golden_cross_long = (
            (dataframe['supertrend_cross_bull'] > 0) &
            (dataframe['vwma_cross_bull'] > 0) &
            (dataframe['volume_long_entry'] > 0) &
            rsi_long_filter
        )
        
        # Combine both entry signals
        conditions.append(long_conditions | golden_cross_long)
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1
            
        # === SHORT CONDITIONS ===
        short_conditions = []
        
        # RSI Filter - Short
        rsi_short_filter = (
            (dataframe['rsi'] < 80) &
            (dataframe['rsi'] >= 40)
        )
        
        # VWAP Filter - Short
        vwap_short_filter = (
            (dataframe['close'] < dataframe['vwap'])
        )
        
        # Supertrend Filter - Short
        supertrend_short_filter = (
            (dataframe['close'] < dataframe['supertrend_short_stop'])
        )
        
        # VWMA Filter - Short
        vwma_short_filter = (
            (dataframe['close'] < dataframe['vwma'])
        )
        
        # PSAR Filter - Short
        psar_short_filter = (
            (dataframe['close'] < dataframe['psar'])
        )
        
        # Kalman Filter - Short
        kalman_short_filter = (
            (dataframe['close'] < dataframe['kalman_slow'])
        )
        
        # EMA Filter - Short (if enabled)
        if self.buy_params['ema_length'] > 0:
            ema_short_filter = (dataframe['close'] < dataframe['ema'])
        else:
            ema_short_filter = True  # Not used if ema_length is 0
        
        # Volume condition with two red candles - Short
        volume_short_condition = (
            (dataframe['volume_short_entry'] > 0)
        )
        
        # Combine all filters for Two Candle Theory - Short
        short_all_conditions = (
            rsi_short_filter &
            vwap_short_filter &
            supertrend_short_filter &
            vwma_short_filter &
            psar_short_filter &
            kalman_short_filter &
            volume_short_condition 
        )
        
        # Golden Cross Short Entry
        golden_cross_short = (
            (dataframe['supertrend_cross_bear'] > 0) &
            (dataframe['vwma_cross_bear'] > 0) &
            (dataframe['volume_short_entry'] > 0) &
            rsi_short_filter
        )
        
        # Combine both short entry signals
        short_conditions.append(short_all_conditions | golden_cross_short)
        
        if short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, short_conditions),
                'enter_short'] = 1
            
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        # No explicit exit signals - we use stoploss and ROI
        # Exits are handled through stoploss and take profit using the trade.adjust_* methods
        
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, 
                           time_in_force: str, current_time, **kwargs) -> bool:
        """
        Called before placing a buy order.
        Timing for this function is critical, so avoid adding slow calculations here.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Set dynamic ROI and stoploss based on entry price and previous low/high
        if order_type == 'limit' and rate > 0:
            if self.is_pair_locked(pair):
                return False
                
            trade = Trade.get_trades_proxy(is_open=True)
            
            if trade is None:
                # Calculate the dynamic stoploss and take profit based on entry
                # For long positions
                if kwargs.get('side', 'buy') == 'buy':
                    # Set stoploss at the previous candle's low
                    sl_price = dataframe.iloc[-2]['low']
                    sl_percentage = (sl_price - rate) / rate
                    # Risk-to-reward based take profit
                    tp_price = rate + (rate - sl_price) * self.buy_params['profit_multiplier']
                    
                    # Store these in the trade's custom_info field
                    custom_info = {
                        'stoploss_price': sl_price,
                        'take_profit_price': tp_price,
                    }
                    
                    # We'll use these in the custom_stoploss and custom_exit functions
                    self.custom_info[pair] = custom_info
                else:
                    # For short positions
                    sl_price = dataframe.iloc[-2]['high']
                    sl_percentage = (rate - sl_price) / rate
                    # Risk-to-reward based take profit
                    tp_price = rate - (sl_price - rate) * self.buy_params['profit_multiplier']
                    
                    # Store these in the trade's custom_info field
                    custom_info = {
                        'stoploss_price': sl_price,
                        'take_profit_price': tp_price,
                    }
                    
                    # We'll use these in the custom_stoploss and custom_exit functions
                    self.custom_info[pair] = custom_info
                    
                # Remember to update your stoploss for this specific trade
                if trade:
                    trade.adjust_stop_loss(rate, sl_percentage)
        
        return True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic, returning the new stoploss as a percentage.
        """
        # Get the custom_info for this trade
        custom_info = self.custom_info.get(pair, {})
        
        if custom_info and 'stoploss_price' in custom_info:
            # Calculate the stoploss percentage based on entry price
            if trade.is_short:
                return (current_rate - custom_info['stoploss_price']) / current_rate
            else:
                return (custom_info['stoploss_price'] - current_rate) / current_rate
        
        # Default stoploss - should not reach here
        return self.stoploss

    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs) -> bool:
        """
        Custom exit signal logic based on take profit.
        """
        # Get the custom_info for this trade
        custom_info = self.custom_info.get(pair, {})
        
        if custom_info and 'take_profit_price' in custom_info:
            # Check if we've hit the take profit level
            if trade.is_short:
                # For short positions, exit when price <= take profit target
                if current_rate <= custom_info['take_profit_price']:
                    return True
            else:
                # For long positions, exit when price >= take profit target
                if current_rate >= custom_info['take_profit_price']:
                    return True
        
        return False


    # Helper functions
    def vwap(self, dataframe):
        """
        Volume Weighted Average Price
        """
        # Reset on daily timeframe
        df = dataframe.copy()
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df['date_reset'] = df['date'].dt.floor('D')
        
        # Group by date and get cumulative values
        groups = df.groupby('date_reset')
        df['vwap'] = groups.apply(lambda x: (x['volume'] * (x['high'] + x['low'] + x['close']) / 3).cumsum() / x['volume'].cumsum()).reset_index(level=0, drop=True)
        
        return df['vwap']

    def vwma(self, dataframe, length=20):
        """
        Volume Weighted Moving Average
        """
        return ((dataframe['close'] * dataframe['volume']).rolling(length).sum() / 
                dataframe['volume'].rolling(length).sum())

    def supertrend(self, dataframe, length=10, multiplier=2.0):
        """
        SuperTrend Indicator
        """
        df = dataframe.copy()
        
        # Calculate ATR
        df['atr'] = ta.ATR(df, timeperiod=length)
        
        # SuperTrend calculation
        df['basic_upperband'] = ((df['high'] + df['low']) / 2) + (multiplier * df['atr'])
        df['basic_lowerband'] = ((df['high'] + df['low']) / 2) - (multiplier * df['atr'])
        
        # Initialize bands and direction
        df['final_upperband'] = np.nan
        df['final_lowerband'] = np.nan
        df['direction'] = np.nan
        
        # Set initial values
        for i in range(length, len(df)):
            if i == length:
                df.loc[df.index[i], 'direction'] = 1
                df.loc[df.index[i], 'final_upperband'] = df.loc[df.index[i], 'basic_upperband']
                df.loc[df.index[i], 'final_lowerband'] = df.loc[df.index[i], 'basic_lowerband']
            else:
                prev_direction = df.loc[df.index[i-1], 'direction']
                curr_close = df.loc[df.index[i], 'close']
                prev_upperband = df.loc[df.index[i-1], 'final_upperband']
                prev_lowerband = df.loc[df.index[i-1], 'final_lowerband']
                curr_upperband = df.loc[df.index[i], 'basic_upperband']
                curr_lowerband = df.loc[df.index[i], 'basic_lowerband']
                
                # Current direction
                if prev_direction == 1:
                    # Long to short switch
                    if curr_close < prev_lowerband:
                        df.loc[df.index[i], 'direction'] = -1
                        df.loc[df.index[i], 'final_upperband'] = curr_upperband
                        df.loc[df.index[i], 'final_lowerband'] = curr_lowerband
                    else:
                        # Continue long
                        df.loc[df.index[i], 'direction'] = 1
                        df.loc[df.index[i], 'final_upperband'] = max(curr_upperband, prev_upperband)
                        df.loc[df.index[i], 'final_lowerband'] = max(curr_lowerband, prev_lowerband)
                else:
                    # Short to long switch
                    if curr_close > prev_upperband:
                        df.loc[df.index[i], 'direction'] = 1
                        df.loc[df.index[i], 'final_upperband'] = curr_upperband
                        df.loc[df.index[i], 'final_lowerband'] = curr_lowerband
                    else:
                        # Continue short
                        df.loc[df.index[i], 'direction'] = -1
                        df.loc[df.index[i], 'final_upperband'] = min(curr_upperband, prev_upperband)
                        df.loc[df.index[i], 'final_lowerband'] = min(curr_lowerband, prev_lowerband)
        
        # Prepare return values
        result = {
            'direction': df['direction'],
            'long_stop': df['final_lowerband'],
            'short_stop': df['final_upperband']
        }
        
        return result
    '''
    lev_X = IntParameter(1, 5, default=5, space="buy", optimize=True, load=True)
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
            proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
            side: str, **kwargs) -> float:

        return self.lev_X.value
        '''
