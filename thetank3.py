
import logging
import numpy as np
import pandas as pd
from technical import qtpylib, pivots_points
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


class thetank3(IStrategy):


    ### Strategy parameters ###
    exit_profit_only = True ### No selling at a loss
    use_custom_stoploss = True
    trailing_stop = True
    position_adjustment_enable = True
    ignore_roi_if_entry_signal = True
    use_exit_signal = True
    stoploss = -0.09
    startup_candle_count: int = 30
    timeframe = '15m'
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
    max_epa = CategoricalParameter([0, 1, 2, 3], default=0, space="buy", optimize=True)

    # protections
    cooldown_lookback = IntParameter(4, 48, default=16, space="protection", optimize=True)
    stop_duration = IntParameter(5, 96, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # indicators
    wavelength = IntParameter(low=3, high=10, default=8, space='buy', optimize=True)
    crosslength = IntParameter(low=3, high=10, default=3, space='sell', optimize=True)
    filterlength = IntParameter(low=25, high=35, default=25, space='buy', optimize=True)
   

    # trading
    buy_rsi = IntParameter(low=20, high=30, default=25, space='buy', optimize=True, load=True)
    buy_rsi_bear = IntParameter(low=40, high=55, default=45, space='buy', optimize=True, load=True)
    buy_rsi_bull = IntParameter(low=55, high=75, default=65, space='buy', optimize=True, load=True)
    buy_wt_bear = IntParameter(low=20, high=55, default=45, space='buy', optimize=True, load=True)
    buy_wt_bull = IntParameter(low=30, high=75, default=65, space='buy', optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=80, default=55, space='sell', optimize=True, load=True)

    # dca level optimization
    dca1 = DecimalParameter(low=0.03, high=0.08, decimals=2, default=0.05, space='buy', optimize=True, load=True)
    dca2 = DecimalParameter(low=0.08, high=0.15, decimals=2, default=0.10, space='buy', optimize=True, load=True)
    dca3 = DecimalParameter(low=0.15, high=0.25, decimals=2, default=0.15, space='buy', optimize=True, load=True)

    #trailing stop loss optimiziation
    tsl_target5 = DecimalParameter(low=0.2, high=0.4, decimals=1, default=0.3, space='sell', optimize=True, load=True)
    ts5 = DecimalParameter(low=0.04, high=0.06, default=0.05, space='sell', optimize=True, load=True)
    tsl_target4 = DecimalParameter(low=0.18, high=0.3, default=0.2, space='sell', optimize=True, load=True)
    ts4 = DecimalParameter(low=0.03, high=0.05, default=0.045, space='sell', optimize=True, load=True)
    tsl_target3 = DecimalParameter(low=0.10, high=0.15, default=0.15, space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.07, high=0.12, default=0.1, space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.05, high=0.06, default=0.07, space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.02, high=0.05, default=0.03, space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.008, high=0.015, default=0.013, space='sell', optimize=True, load=True)



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

        # Pivot Points
        pivots = pivots_points.pivots_points(dataframe)
        dataframe['pivot'] = pivots['pivot']
        dataframe['s1'] = pivots['s1']
        dataframe['r1'] = pivots['r1']
        dataframe['s2'] = pivots['s2']
        dataframe['r2'] = pivots['r2']
        dataframe['s3'] = pivots['s3']
        dataframe['r3'] = pivots['r3']     
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
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['rsi_ma'] = ta.SMA(dataframe['rsi'], timeperiod=7)

        # WaveTrend using OHLC4 or HA close - 3/21
        ap = (0.25 * (dataframe['high'] + dataframe['low'] + dataframe["close"] + dataframe["open"]))
        
        for wave in self.wavelength.range:

            dataframe[f'esa{wave}'] = ta.EMA(ap, timeperiod = wave)
            dataframe[f'd{wave}'] = ta.EMA(abs(ap - dataframe[f'esa{wave}']), timeperiod = wave)
            dataframe[f'wave_ci{wave}'] = (ap-dataframe[f'esa{wave}']) / (0.015 * dataframe[f'd{wave}'])
            dataframe[f'wave_t1{wave}'] = ta.EMA(dataframe[f'wave_ci{wave}'], timeperiod = 21)
            
            for cross in self.crosslength.range:
                dataframe[f'wave_t2{cross}_{wave}'] = ta.SMA(dataframe[f'wave_t1{wave}'], timeperiod = cross)

        # SMA
        dataframe['200_SMA'] = ta.SMA(dataframe["close"], timeperiod = 200)
        dataframe['30_SMA'] = ta.SMA(dataframe["close"], timeperiod = 30)

        # TTM Squeeze
        ttm_Squeeze = pta.squeeze(high = dataframe['high'], low = dataframe['low'], close = dataframe["close"], lazybear = True)
        dataframe['ttm_Squeeze'] = ttm_Squeeze['SQZ_20_2.0_20_1.5_LB']
        dataframe['ttm_ema'] = ta.EMA(dataframe['ttm_Squeeze'], timeperiod = 4)
        dataframe['squeeze_ON'] = ttm_Squeeze['SQZ_ON']
        dataframe['squeeze_OFF'] = ttm_Squeeze['SQZ_OFF']
        dataframe['NO_squeeze'] = ttm_Squeeze['SQZ_NO']

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
        #     self.max_epa.value = 1
        # elif (dataframe['30_SMA'].iloc[-1] > dataframe['200_SMA'].iloc[-1] 
        #     and dataframe['30_SMA'].iloc[-1] > dataframe['30_SMA'].iloc[-2].all()):
        #     self.max_epa.value = 1
        # elif (dataframe['30_SMA'].iloc[-1] < dataframe['200_SMA'].iloc[-1] 
        #     and dataframe['30_SMA'].iloc[-1] > dataframe['30_SMA'].iloc[-2].all()):
        #     self.max_epa.value = 2
        # else:
        #     self.max_epa.value = 2
        # # print(f"Trading Pair: {dataframe.name}")
        # print(f"self.max_epa.value: {self.max_epa.value}")
        # print(f"self.max_dca_multiplier: {self.max_dca_multiplier}")
        # print(f"Trading pair: {ticker['symbol']}, max_epa: {self.max_epa.value}, dca_multiplier: {self.max_dca_multiplier}")
 
        return dataframe

    ### ENTRY CONDITIONS ###
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:


        df.loc[
            (
                (qtpylib.crossed_above(df[f'zema_{self.filterlength.value}'], df['s3'])) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'S3 - XO')

        df.loc[
            (
                (qtpylib.crossed_above(df[f'zema_{self.filterlength.value}'], df['s2'])) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'S2 - XO')

        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['close'] > df[f'zema_{self.filterlength.value}'])&
                (df['close'] < df['r2']) &
                (df['squeeze_ON'] == 1)&
                (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                (df['200_SMA'] < df['200_SMA'].shift(1)) &
                (df['rsi'] < self.buy_rsi_bear.value) &
                (df[f'wave_t1{self.wavelength.value}'] < self.buy_wt_bear.value)&
                (df[f'wave_t1{self.wavelength.value}'] > df[f'wave_t1{self.wavelength.value}'].shift(1)) &  # Guard: Wave 1 is raising
                (qtpylib.crossed_above(df[f'wave_t1{self.wavelength.value}'], df[f'wave_t2{self.crosslength.value}_{self.wavelength.value}'])) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'ttm_Squeeze/WT - bear')

        df.loc[
            (
                (df['close'] > df[f'zema_{self.filterlength.value}'])&
                (df['close'] < df['r2']) &
                (df['squeeze_ON'] == 1)&
                (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                (df['rsi'] < self.buy_rsi_bear.value) &
                (df['200_SMA'] < df['200_SMA'].shift(1)) &
                (df[f'wave_t1{self.wavelength.value}'] < self.buy_wt_bear.value)&
                (df[f'wave_t1{self.wavelength.value}'] > df[f'wave_t1{self.wavelength.value}'].shift(1)) &  # Guard: Wave 1 is raising
                (df[f'wave_t1{self.wavelength.value}'].shift(2) > df[f'wave_t1{self.wavelength.value}'].shift(1)) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'ttm_Squeeze/WTT - bear')

        df.loc[
            (
                (df['close'] > df[f'zema_{self.filterlength.value}'])&
                (df['close'] < df['r2']) &
                (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                (df['rsi'] < self.buy_rsi_bear.value) &
                (df['200_SMA'] < df['200_SMA'].shift(1)) &
                (df[f'wave_t1{self.wavelength.value}'] < self.buy_wt_bear.value)&
                (df[f'wave_t1{self.wavelength.value}'] > df[f'wave_t1{self.wavelength.value}'].shift(1)) &  # Guard: Wave 1 is raising
                (df[f'wave_t1{self.wavelength.value}'].shift(2) > df[f'wave_t1{self.wavelength.value}'].shift(1)) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'WT transition - bear')

        df.loc[
            (
                (df['close'] > df[f'zema_{self.filterlength.value}'])&
                (df['close'] < df['r2']) &
                (df['200_SMA'] < df['200_SMA'].shift(1)) &
                (df[f'wave_t1{self.wavelength.value}'] > df[f'wave_t1{self.wavelength.value}'].shift(1)) &  # Guard: Wave 1 is raising
                (df[f'wave_t1{self.wavelength.value}'] < self.buy_wt_bear.value) & 
                (df[f'wave_t1{self.wavelength.value}'].shift(2) > df[f'wave_t1{self.wavelength.value}'].shift(1)) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'WT - bear')

        df.loc[
            (
                (df['close'] > df[f'zema_{self.filterlength.value}'])&
                (df['close'] < df['r2']) &
                (df['squeeze_ON'] == 1)&
                (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                (df['rsi'] < self.buy_rsi_bull.value) &
                (df['200_SMA'] > df['200_SMA'].shift(1)) &
                (df[f'wave_t1{self.wavelength.value}'] > df[f'wave_t1{self.wavelength.value}'].shift(1)) &  # Guard: Wave 1 is raising
                (qtpylib.crossed_above(df[f'wave_t1{self.wavelength.value}'], df[f'wave_t2{self.crosslength.value}_{self.wavelength.value}'])) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'ttm_Squeeze/WT - bull')

        df.loc[
            (
                (df['close'] > df[f'zema_{self.filterlength.value}'])&
                (df['close'] < df['r2']) &
                (df['squeeze_ON'] == 1)&
                (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                (df['rsi'] < self.buy_rsi_bull.value) &
                (df['200_SMA'] > df['200_SMA'].shift(1)) &
                (df[f'wave_t1{self.wavelength.value}'] > df[f'wave_t1{self.wavelength.value}'].shift(1)) & 
                (df[f'wave_t1{self.wavelength.value}'] < self.buy_wt_bull.value) & 
                (df[f'wave_t1{self.wavelength.value}'].shift(2) > df[f'wave_t1{self.wavelength.value}'].shift(1)) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'ttm_Squeeze/WTT - bull')

        df.loc[
            (
                (df['close'] > df[f'zema_{self.filterlength.value}'])&
                (df['close'] < df['r2']) &
                (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                (df['rsi'] < self.buy_rsi_bull.value) &
                (df['200_SMA'] > df['200_SMA'].shift(1)) &
                (df[f'wave_t1{self.wavelength.value}'] > df[f'wave_t1{self.wavelength.value}'].shift(1)) &  # Guard: Wave 1 is raising
                (df[f'wave_t1{self.wavelength.value}'] < self.buy_wt_bull.value) & 
                (df[f'wave_t1{self.wavelength.value}'].shift(2) > df[f'wave_t1{self.wavelength.value}'].shift(1)) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'WT transition - bull')

        df.loc[
            (
                (df['close'] > df[f'zema_{self.filterlength.value}'])&
                (df['close'] < df['r2']) &
                (df['200_SMA'] > df['200_SMA'].shift(1)) &
                (df[f'wave_t1{self.wavelength.value}'] > df[f'wave_t1{self.wavelength.value}'].shift(1)) &  # Guard: Wave 1 is raising
                (df[f'wave_t1{self.wavelength.value}'] < self.buy_wt_bull.value) & 
                (df[f'wave_t1{self.wavelength.value}'].shift(2) > df[f'wave_t1{self.wavelength.value}'].shift(1)) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'WT - bull')


        df.loc[
            (
                (df['close'] > df[f'zema_{self.filterlength.value}'])&
                (df['close'] < df['r2']) &
                (df['rsi'] >  self.buy_rsi.value) &
                (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                (df['rsi'] < self.buy_rsi_bear.value) &
                (df['200_SMA'] < df['200_SMA'].shift(1)) &
                (qtpylib.crossed_above(df['rsi'], df['rsi_ma'])) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'RSI-XO bear')

        df.loc[
            (
                (df['close'] > df[f'zema_{self.filterlength.value}'])&
                (df['close'] < df['r2']) &
                (df['rsi'] >  self.buy_rsi.value) &
                (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                (df['rsi'] < self.buy_rsi_bull.value) &
                (df['200_SMA'] > df['200_SMA'].shift(1)) &
                (qtpylib.crossed_above(df['rsi'], df['rsi_ma'])) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'RSI-XO bull')


        return df


    ### EXIT CONDITIONS ###
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:


        # df.loc[
        #     (
        #         # Signal: RSI crosses above 30
        #         (df['squeeze_ON'] == 1)&
        #         (df[f'wave_t1{self.wavelength.value}'] < df[f'wave_t1{self.wavelength.value}'].shift(1)) &  # Guard: Wave 1 is raising
        #         (df[f'wave_t1{self.wavelength.value}'].shift(2) < df[f'wave_t1{self.wavelength.value}'].shift(1)) &
        #         (df['volume'] > 0)   # Make sure Volume is not 0
        #     ),
        #     ['exit_long', 'exit_tag']] = (1, 'ttm_Squeeze/WT transition')

        # df.loc[
        #     (
        #         # Signal: RSI crosses above 30
        #         (df[f'wave_t1{self.wavelength.value}'] < df[f'wave_t1{self.wavelength.value}'].shift(1)) &  # Guard: Wave 1 is raising
        #         (df[f'wave_t1{self.wavelength.value}'].shift(2) < df[f'wave_t1{self.wavelength.value}'].shift(1)) &
        #         (df['volume'] > 0)   # Make sure Volume is not 0
        #     ),
        #     ['exit_long', 'exit_tag']] = (1, 'WT transition bear')

        # df.loc[
        #     (
        #         # Signal: RSI crosses above 30
        #         (df[f'wave_t1{self.wavelength.value}'] < df[f'wave_t1{self.wavelength.value}'].shift(1)) &  # Guard: Wave 1 is raising
        #         (df[f'wave_t1{self.wavelength.value}'].shift(2) < df[f'wave_t1{self.wavelength.value}'].shift(1)) &
        #         (df['volume'] > 0)   # Make sure Volume is not 0
        #     ),



        # df.loc[
        #     (
        #         # Signal: RSI crosses above 30
        #         (df['rsi'] < df['rsi_ma']) &
        #         # (df['tf'] > df['close']) &
        #         (df[f'wave_t1{self.wavelength.value}'] < df[f'wave_t1{self.wavelength.value}'].shift(1)) &  # Guard: Wave 1 is raising
        #         (qtpylib.crossed_above(df[f'wave_t2{self.crosslength.value}_{self.wavelength.value}'], df[f'wave_t1{self.wavelength.value}'])) &
        #         (df['volume'] > 0)   # Make sure Volume is not 0
        #     ),
        #     ['exit_long', 'exit_tag']] = (1, 'WT/RSI-XO')

        # df.loc[
        #     (
        #         (df['rsi'] > self.sell_rsi.value) &
        #         (qtpylib.crossed_above(df['rsi_ma'], df['rsi'])) &
        #         (df['volume'] > 0)   # Make sure Volume is not 0
        #     ),
        #     ['exit_long', 'exit_tag']] = (1, 'RSI-XO')

        # df.loc[
        #     (
        #         # Signal: RSI crosses above 30
        #         (df['rsi'] >  self.sell_rsi.value) &
        #         # (df['rsi'] < df['rsi_ma'])&
        #         (df[f'wave_t1{self.wavelength.value}'] < df[f'wave_t1{self.wavelength.value}'].shift(1)) &  # Guard: Wave 1 is raising
        #         (df[f'wave_t1{self.wavelength.value}'].shift(1) > df[f'wave_t1{self.wavelength.value}'].shift(2)) & 
        #         # (abs(df[f'wave_t2{self.crosslength.value}_{self.wavelength.value}'] - df[f'wave_t1{self.wavelength.value}']) > 2.5) &
        #         (df['volume'] > 0)   # Make sure Volume is not 0
        #     ),
        #     ['exit_long', 'exit_tag']] = (1, 'WT/RSI TURN DOWN')

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

# Best result:

#     40/250:   1027 trades. 699/290/38 Wins/Draws/Losses. Avg profit   0.91%. Median profit   0.25%. Total profit 6051.24032389 USDT (  60.51%). Avg duration 11:33:00 min. Objective: -6051.24032


#     # Buy hyperspace params:
#     buy_params = {
#         "buy_rsi": 21,
#         "buy_rsi_bear": 48,
#         "buy_rsi_bull": 75,
#         "buy_wt_bear": 46,
#         "buy_wt_bull": 70,
#         "dca1": 0.01,
#         "dca2": 0.11,
#         "dca3": 0.23,
#         "max_epa": 2,
#         "wavelength": 10,
#     }

#     # Sell hyperspace params:
#     sell_params = {
#         "sell_rsi": 71,
#         "ts0": 0.012,
#         "ts1": 0.011,
#         "ts2": 0.022,
#         "ts3": 0.03,
#         "ts4": 0.038,
#         "ts5": 0.045,
#         "tsl_target0": 0.029,
#         "tsl_target1": 0.058,
#         "tsl_target2": 0.091,
#         "tsl_target3": 0.143,
#         "tsl_target4": 0.183,
#         "tsl_target5": 0.2,
#     }

#     # Protection hyperspace params:
#     protection_params = {
#         "cooldown_lookback": 38,  # value loaded from strategy
#         "stop_duration": 47,  # value loaded from strategy
#         "use_stop_protection": False,  # value loaded from strategy
#     }

#     # ROI table:
#     minimal_roi = {
#         "0": 0.364,
#         "45": 0.145,
#         "160": 0.048,
#         "520": 0
#     }

#     # Stoploss:
#     stoploss = -0.286

#     # Trailing stop:
#     trailing_stop = True  # value loaded from strategy
#     trailing_stop_positive = 0.269  # value loaded from strategy
#     trailing_stop_positive_offset = 0.317  # value loaded from strategy
#     trailing_only_offset_is_reached = True  # value loaded from strategy
    

#     # Max Open Trades:
#     max_open_trades = 12  # value loaded from strategy
# Best result:

#     53/250:    886 trades. 608/262/16 Wins/Draws/Losses. Avg profit   0.90%. Median profit   0.23%. Total profit 5306.73713390 USDT (  53.07%). Avg duration 12:10:00 min. Objective: -5306.73713


#     # Buy hyperspace params:
#     buy_params = {
#         "buy_rsi": 22,
#         "buy_rsi_bear": 40,
#         "buy_rsi_bull": 59,
#         "buy_wt_bear": 48,
#         "buy_wt_bull": 56,
#         "dca1": 0.01,
#         "dca2": 0.14,
#         "dca3": 0.21,
#         "max_epa": 1,
#         "wavelength": 9,
#     }

#     # Sell hyperspace params:
#     sell_params = {
#         "sell_rsi": 68,
#         "ts0": 0.009,
#         "ts1": 0.013,
#         "ts2": 0.025,
#         "ts3": 0.028,
#         "ts4": 0.04,
#         "ts5": 0.047,
#         "tsl_target0": 0.047,
#         "tsl_target1": 0.053,
#         "tsl_target2": 0.099,
#         "tsl_target3": 0.135,
#         "tsl_target4": 0.25,
#         "tsl_target5": 0.4,
#     }

#     # Protection hyperspace params:
#     protection_params = {
#         "cooldown_lookback": 26,
#         "stop_duration": 83,
#         "use_stop_protection": False,
#     }

#     # ROI table:
#     minimal_roi = {
#         "0": 0.154,
#         "120": 0.09,
#         "300": 0.044,
#         "554": 0
#     }

#     # Stoploss:
#     stoploss = -0.205

#     # Trailing stop:
#     trailing_stop = True
#     trailing_stop_positive = 0.342
#     trailing_stop_positive_offset = 0.437
#     trailing_only_offset_is_reached = True
    

#     # Max Open Trades:
#     max_open_trades = 12  # value loaded from strategy
# (.env) core@core:~/freqtrade$ freqtrade hyperopt --hyperopt-loss OnlyProfitHyperOptLoss --strategy thetank2 --config ho.json -e 250 --timerange 20230101-20230215 --spaces buy sell roi stoploss trailing protection

# Result for strategy thetank2 45 pairs 1hr
# ============================================================= BACKTESTING REPORT =============================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# |   LDO/USDT |        30 |           5.71 |         171.35 |           264.999 |           2.65 |         22:04:00 |    23     4     3  76.7 |
# |  AGIX/USDT |        42 |           3.84 |         161.20 |           224.104 |           2.24 |         14:41:00 |    27     3    12  64.3 |
# |   AKT/USDT |        44 |           3.63 |         159.82 |           210.718 |           2.11 |         13:08:00 |    38     3     3  86.4 |
# |  RNDR/USDT |        37 |           3.91 |         144.50 |           196.314 |           1.96 |         15:52:00 |    32     3     2  86.5 |
# | OCEAN/USDT |        38 |           3.42 |         130.13 |           186.827 |           1.87 |         17:09:00 |    34     3     1  89.5 |
# |   INJ/USDT |        32 |           3.89 |         124.64 |           185.401 |           1.85 |         19:30:00 |    23     6     3  71.9 |
# |  SCRT/USDT |        34 |           2.79 |          94.71 |           135.555 |           1.36 |         20:12:00 |    28     5     1  82.4 |
# |   GMX/USDT |        29 |           2.93 |          85.11 |           125.473 |           1.25 |   1 day, 0:39:00 |    25     4     0   100 |
# |  HBAR/USDT |        30 |           2.70 |          81.12 |           114.205 |           1.14 |         22:12:00 |    25     3     2  83.3 |
# |   IMX/USDT |        33 |           2.50 |          82.46 |           112.595 |           1.13 |         19:25:00 |    25     4     4  75.8 |
# |    OP/USDT |        26 |           2.96 |          76.91 |           104.512 |           1.05 |   1 day, 2:18:00 |    20     4     2  76.9 |
# |  ORAI/USDT |        34 |           2.29 |          77.91 |            94.056 |           0.94 |         17:16:00 |    20     2    12  58.8 |
# |  DYDX/USDT |        28 |           2.54 |          71.20 |            90.262 |           0.90 |         23:51:00 |    21     4     3  75.0 |
# |   GRT/USDT |        27 |           2.24 |          60.47 |            86.385 |           0.86 |   1 day, 0:53:00 |    20     5     2  74.1 |
# |  CSPR/USDT |        29 |           2.06 |          59.67 |            80.551 |           0.81 |   1 day, 0:06:00 |    23     4     2  79.3 |
# |  ROSE/USDT |        27 |           2.02 |          54.56 |            76.839 |           0.77 |   1 day, 1:29:00 |    18     8     1  66.7 |
# |  NEAR/USDT |        27 |           2.24 |          60.58 |            71.576 |           0.72 |   1 day, 0:22:00 |    21     5     1  77.8 |
# |  KLAY/USDT |        24 |           1.95 |          46.79 |            58.648 |           0.59 |   1 day, 6:10:00 |    18     5     1  75.0 |
# |  IOTA/USDT |        30 |           1.87 |          55.96 |            58.406 |           0.58 |   1 day, 0:34:00 |    24     5     1  80.0 |
# |   ENJ/USDT |        26 |           2.21 |          57.37 |            55.862 |           0.56 |   1 day, 2:51:00 |    20     5     1  76.9 |
# |  AVAX/USDT |        22 |           1.75 |          38.55 |            51.655 |           0.52 |   1 day, 7:08:00 |    18     2     2  81.8 |
# |   FIL/USDT |        20 |           1.96 |          39.13 |            47.378 |           0.47 |  1 day, 13:57:00 |    14     5     1  70.0 |
# | MATIC/USDT |        25 |           1.52 |          38.07 |            44.904 |           0.45 |   1 day, 3:53:00 |    20     4     1  80.0 |
# | THETA/USDT |        20 |           1.61 |          32.19 |            40.040 |           0.40 |  1 day, 13:03:00 |    13     6     1  65.0 |
# |   ADA/USDT |        24 |           1.19 |          28.48 |            36.719 |           0.37 |   1 day, 6:28:00 |    18     4     2  75.0 |
# |   VET/USDT |        27 |           1.36 |          36.76 |            36.663 |           0.37 |   1 day, 2:27:00 |    22     4     1  81.5 |
# |  ATOM/USDT |        17 |           1.40 |          23.88 |            34.406 |           0.34 |  1 day, 19:21:00 |    13     2     2  76.5 |
# |   ETC/USDT |        17 |           1.76 |          29.89 |            32.255 |           0.32 |  1 day, 19:14:00 |    13     3     1  76.5 |
# |  LINK/USDT |        25 |           0.86 |          21.60 |            30.926 |           0.31 |   1 day, 6:24:00 |    18     6     1  72.0 |
# |   TRX/USDT |        24 |           1.16 |          27.84 |            28.180 |           0.28 |   1 day, 7:40:00 |    16     7     1  66.7 |
# |  DOGE/USDT |        20 |           1.04 |          20.88 |            25.900 |           0.26 |  1 day, 14:12:00 |    14     5     1  70.0 |
# |  OSMO/USDT |        18 |           1.12 |          20.21 |            22.729 |           0.23 |   1 day, 0:47:00 |    15     1     2  83.3 |
# |  KAVA/USDT |        17 |           1.59 |          27.02 |            21.355 |           0.21 |  1 day, 21:49:00 |    14     2     1  82.4 |
# |   DOT/USDT |        22 |           0.86 |          18.99 |            19.816 |           0.20 |   1 day, 9:25:00 |    16     5     1  72.7 |
# |   QNT/USDT |        15 |           0.99 |          14.82 |            19.477 |           0.19 |  2 days, 7:44:00 |    11     3     1  73.3 |
# |   ETH/USDT |        19 |           1.37 |          26.02 |            19.227 |           0.19 |  1 day, 15:19:00 |    14     4     1  73.7 |
# |   SOL/USDT |        18 |           1.38 |          24.75 |            16.865 |           0.17 |  1 day, 15:47:00 |    11     6     1  61.1 |
# |  ALGO/USDT |        21 |           0.91 |          19.21 |            16.804 |           0.17 |  1 day, 11:49:00 |    15     5     1  71.4 |
# |   BTC/USDT |        23 |           0.81 |          18.65 |            15.550 |           0.16 |   1 day, 7:50:00 |    20     2     1  87.0 |
# |  EGLD/USDT |        24 |           0.70 |          16.70 |            14.247 |           0.14 |   1 day, 8:08:00 |    19     4     1  79.2 |
# |   UNI/USDT |        11 |           0.78 |           8.56 |            11.334 |           0.11 |  3 days, 4:11:00 |     6     4     1  54.5 |
# |   XLM/USDT |        20 |           0.58 |          11.63 |             5.577 |           0.06 |  1 day, 14:27:00 |    13     6     1  65.0 |
# |   XTZ/USDT |        15 |           0.82 |          12.33 |             0.802 |           0.01 |  2 days, 2:32:00 |     8     6     1  53.3 |
# |   XRP/USDT |        17 |           0.49 |           8.25 |             0.122 |           0.00 |  1 day, 21:18:00 |    14     2     1  82.4 |
# |   XDC/USDT |        18 |           0.28 |           5.09 |            -3.780 |          -0.04 |  1 day, 20:30:00 |    13     4     1  72.2 |
# |   ZEC/USDT |         7 |           0.24 |           1.66 |            -3.944 |          -0.04 | 4 days, 23:00:00 |     4     2     1  57.1 |
# |      TOTAL |      1133 |           2.14 |        2427.63 |          3118.496 |          31.18 |   1 day, 4:47:00 |   857   189    87  75.6 |
# =================================================================== ENTER TAG STATS ===================================================================
# |                   TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
# |-----------------------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
# |             WT - bull |       706 |           2.13 |        1500.70 |          1903.689 |          19.04 | 1 day, 4:41:00 |   531   128    47  75.2 |
# |           RSI-XO bull |       376 |           2.23 |         838.42 |          1106.063 |          11.06 | 1 day, 5:30:00 |   289    53    34  76.9 |
# |             WT - bear |        33 |           2.17 |          71.71 |            93.210 |           0.93 | 1 day, 5:00:00 |    25     6     2  75.8 |
# |           RSI-XO bear |         5 |           3.10 |          15.49 |            26.693 |           0.27 |       16:00:00 |     5     0     0   100 |
# | ttm_Squeeze/WT - bull |        13 |           0.10 |           1.31 |           -11.159 |          -0.11 |       18:18:00 |     7     2     4  53.8 |
# |                 TOTAL |      1133 |           2.14 |        2427.63 |          3118.496 |          31.18 | 1 day, 4:47:00 |   857   189    87  75.6 |
# ======================================================= EXIT REASON STATS ========================================================
# |        Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
# |--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
# | WT transition bull |     437 |    437     0     0   100 |           0.96 |         418.88 |           521.784 |           9.31 |
# | trailing_stop_loss |     430 |    381     0    49  88.6 |           5.07 |        2179.53 |          2991.88  |          48.43 |
# |                roi |     228 |     39   189     0   100 |           0.85 |         194.11 |           310.197 |           4.31 |
# |         force_exit |      38 |      0     0    38     0 |          -9.6  |        -364.88 |          -705.368 |          -8.11 |
# =========================================================== LEFT OPEN TRADES REPORT ===========================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |      Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+-------------------+-------------------------|
# |   TRX/USDT |         1 |          -0.19 |          -0.19 |            -0.197 |          -0.00 |           5:00:00 |     0     0     1     0 |
# |   AKT/USDT |         1 |          -0.20 |          -0.20 |            -0.203 |          -0.00 |           0:00:00 |     0     0     1     0 |
# |  SCRT/USDT |         1 |          -0.35 |          -0.35 |            -0.710 |          -0.01 |   1 day, 14:00:00 |     0     0     1     0 |
# |  KLAY/USDT |         1 |          -1.35 |          -1.35 |            -2.679 |          -0.03 |  5 days, 23:00:00 |     0     0     1     0 |
# |   ADA/USDT |         1 |          -1.97 |          -1.97 |            -3.926 |          -0.04 |   6 days, 6:00:00 |     0     0     1     0 |
# |   BTC/USDT |         1 |          -5.44 |          -5.44 |           -10.373 |          -0.10 |  11 days, 8:00:00 |     0     0     1     0 |
# | MATIC/USDT |         1 |          -5.27 |          -5.27 |           -10.443 |          -0.10 |   6 days, 2:00:00 |     0     0     1     0 |
# |   VET/USDT |         1 |          -6.56 |          -6.56 |           -13.027 |          -0.13 |  5 days, 23:00:00 |     0     0     1     0 |
# |   ETH/USDT |         1 |          -6.59 |          -6.59 |           -13.060 |          -0.13 |  6 days, 16:00:00 |     0     0     1     0 |
# |  EGLD/USDT |         1 |          -6.62 |          -6.62 |           -13.166 |          -0.13 |  6 days, 10:00:00 |     0     0     1     0 |
# |   GRT/USDT |         1 |          -7.36 |          -7.36 |           -14.650 |          -0.15 |   6 days, 5:00:00 |     0     0     1     0 |
# |   SOL/USDT |         1 |          -7.71 |          -7.71 |           -15.343 |          -0.15 |   6 days, 9:00:00 |     0     0     1     0 |
# |  IOTA/USDT |         1 |          -8.05 |          -8.05 |           -15.938 |          -0.16 |  6 days, 16:00:00 |     0     0     1     0 |
# |   DOT/USDT |         1 |          -8.41 |          -8.41 |           -16.744 |          -0.17 |   6 days, 7:00:00 |     0     0     1     0 |
# |   XLM/USDT |         1 |          -8.80 |          -8.80 |           -17.025 |          -0.17 |  10 days, 2:00:00 |     0     0     1     0 |
# |  OSMO/USDT |         1 |          -8.82 |          -8.82 |           -17.624 |          -0.18 |   5 days, 8:00:00 |     0     0     1     0 |
# |   XRP/USDT |         1 |         -10.23 |         -10.23 |           -18.174 |          -0.18 | 21 days, 12:00:00 |     0     0     1     0 |
# |   ZEC/USDT |         1 |         -11.10 |         -11.10 |           -19.042 |          -0.19 | 24 days, 20:00:00 |     0     0     1     0 |
# | THETA/USDT |         1 |          -9.77 |          -9.77 |           -19.353 |          -0.19 |  6 days, 19:00:00 |     0     0     1     0 |
# |   UNI/USDT |         1 |         -10.19 |         -10.19 |           -19.748 |          -0.20 |  10 days, 3:00:00 |     0     0     1     0 |
# |    OP/USDT |         1 |         -10.35 |         -10.35 |           -20.543 |          -0.21 |  5 days, 23:00:00 |     0     0     1     0 |
# |  ATOM/USDT |         1 |         -10.34 |         -10.34 |           -20.552 |          -0.21 |   6 days, 8:00:00 |     0     0     1     0 |
# |  NEAR/USDT |         1 |         -10.71 |         -10.71 |           -21.371 |          -0.21 |  5 days, 12:00:00 |     0     0     1     0 |
# | OCEAN/USDT |         1 |         -10.84 |         -10.84 |           -21.493 |          -0.21 |  6 days, 18:00:00 |     0     0     1     0 |
# |   QNT/USDT |         1 |         -12.01 |         -12.01 |           -21.609 |          -0.22 | 19 days, 19:00:00 |     0     0     1     0 |
# |   XDC/USDT |         1 |         -11.28 |         -11.28 |           -21.801 |          -0.22 |  10 days, 3:00:00 |     0     0     1     0 |
# |  DOGE/USDT |         1 |         -11.32 |         -11.32 |           -22.009 |          -0.22 |  9 days, 14:00:00 |     0     0     1     0 |
# |  ORAI/USDT |         1 |         -11.02 |         -11.02 |           -22.151 |          -0.22 |  2 days, 20:00:00 |     0     0     1     0 |
# |   FIL/USDT |         1 |         -12.00 |         -12.00 |           -22.643 |          -0.23 | 12 days, 16:00:00 |     0     0     1     0 |
# |  LINK/USDT |         1 |         -11.37 |         -11.37 |           -22.733 |          -0.23 |   5 days, 5:00:00 |     0     0     1     0 |
# |   ETC/USDT |         1 |         -12.23 |         -12.23 |           -23.738 |          -0.24 |  9 days, 21:00:00 |     0     0     1     0 |
# |  ALGO/USDT |         1 |         -12.70 |         -12.70 |           -25.172 |          -0.25 |   6 days, 2:00:00 |     0     0     1     0 |
# |   XTZ/USDT |         1 |         -12.77 |         -12.77 |           -25.310 |          -0.25 |   6 days, 3:00:00 |     0     0     1     0 |
# |   ENJ/USDT |         1 |         -12.93 |         -12.93 |           -25.846 |          -0.26 |   5 days, 8:00:00 |     0     0     1     0 |
# |  CSPR/USDT |         1 |         -16.05 |         -16.05 |           -31.813 |          -0.32 |   6 days, 4:00:00 |     0     0     1     0 |
# |  AVAX/USDT |         1 |         -18.27 |         -18.27 |           -34.645 |          -0.35 |  12 days, 1:00:00 |     0     0     1     0 |
# |  KAVA/USDT |         1 |         -21.51 |         -21.51 |           -38.360 |          -0.38 | 20 days, 23:00:00 |     0     0     1     0 |
# |  DYDX/USDT |         1 |         -22.20 |         -22.20 |           -42.154 |          -0.42 |  12 days, 3:00:00 |     0     0     1     0 |
# |      TOTAL |        38 |          -9.60 |        -364.88 |          -705.368 |          -7.05 |   8 days, 8:55:00 |     0     0    38     0 |
# ================== SUMMARY METRICS ==================
# | Metric                      | Value               |
# |-----------------------------+---------------------|
# | Backtesting from            | 2023-01-01 00:00:00 |
# | Backtesting to              | 2023-02-14 22:00:00 |
# | Max open trades             | 45                  |
# |                             |                     |
# | Total/Daily Avg Trades      | 1133 / 25.75        |
# | Starting balance            | 10000 USDT          |
# | Final balance               | 13118.496 USDT      |
# | Absolute profit             | 3118.496 USDT       |
# | Total profit %              | 31.18%              |
# | CAGR %                      | 850.39%             |
# | Sortino                     | 111.14              |
# | Sharpe                      | 193.21              |
# | Calmar                      | 265.37              |
# | Profit factor               | 4.85                |
# | Expectancy                  | 0.13                |
# | Trades per day              | 25.75               |
# | Avg. daily profit %         | 0.71%               |
# | Avg. stake amount           | 138.885 USDT        |
# | Total trade volume          | 157356.955 USDT     |
# |                             |                     |
# | Best Pair                   | LDO/USDT 171.35%    |
# | Worst Pair                  | ZEC/USDT 1.66%      |
# | Best trade                  | LDO/USDT 56.01%     |
# | Worst trade                 | AGIX/USDT -31.62%   |
# | Best day                    | 257.53 USDT         |
# | Worst day                   | -655.852 USDT       |
# | Days win/draw/lose          | 37 / 0 / 1          |
# | Avg. Duration Winners       | 12:48:00            |
# | Avg. Duration Loser         | 3 days, 15:56:00    |
# | Rejected Entry signals      | 43                  |
# | Entry/Exit Timeouts         | 0 / 0               |
# |                             |                     |
# | Min balance                 | 10003.176 USDT      |
# | Max balance                 | 13823.864 USDT      |
# | Max % of account underwater | 5.10%               |
# | Absolute Drawdown (Account) | 5.10%               |
# | Absolute Drawdown           | 705.368 USDT        |
# | Drawdown high               | 3823.864 USDT       |
# | Drawdown low                | 3118.496 USDT       |
# | Drawdown Start              | 2023-02-14 16:00:00 |
# | Drawdown End                | 2023-02-14 22:00:00 |
# | Market change               | 102.18%             |
# =====================================================

# Result for strategy thetank2 15m 45 piars
# ============================================================= BACKTESTING REPORT ============================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+-----------------+-------------------------|
# |  AGIX/USDT |        73 |           3.01 |         219.69 |           527.631 |           5.28 |        12:33:00 |    61     4     8  83.6 |
# |  RNDR/USDT |        81 |           2.22 |         179.44 |           436.878 |           4.37 |        10:57:00 |    75     4     2  92.6 |
# | OCEAN/USDT |        85 |           1.93 |         164.20 |           376.178 |           3.76 |        10:37:00 |    81     3     1  95.3 |
# |   LDO/USDT |        55 |           3.48 |         191.38 |           354.554 |           3.55 |        17:25:00 |    48     4     3  87.3 |
# |  ORAI/USDT |        66 |           2.57 |         169.42 |           346.247 |           3.46 |        13:26:00 |    61     3     2  92.4 |
# |   INJ/USDT |        66 |           1.78 |         117.72 |           296.749 |           2.97 |        14:09:00 |    57     8     1  86.4 |
# |   AKT/USDT |        80 |           1.45 |         115.76 |           288.802 |           2.89 |        11:33:00 |    70     9     1  87.5 |
# |   GRT/USDT |        89 |           1.31 |         116.68 |           287.505 |           2.88 |        10:09:00 |    83     5     1  93.3 |
# |  SCRT/USDT |        89 |           1.33 |         117.99 |           287.219 |           2.87 |        10:40:00 |    81     8     0   100 |
# |   IMX/USDT |        62 |           1.64 |         101.63 |           260.913 |           2.61 |        15:41:00 |    55     5     2  88.7 |
# |   GMX/USDT |        59 |           1.68 |          99.24 |           241.928 |           2.42 |        16:14:00 |    52     6     1  88.1 |
# |  CSPR/USDT |        69 |           1.28 |          88.15 |           174.721 |           1.75 |        13:38:00 |    62     6     1  89.9 |
# |  ROSE/USDT |        53 |           1.40 |          74.12 |           171.549 |           1.72 |        18:26:00 |    47     6     0   100 |
# |   FIL/USDT |        59 |           1.36 |          80.20 |           168.312 |           1.68 |        16:00:00 |    50     7     2  84.7 |
# |  HBAR/USDT |        70 |           0.99 |          69.36 |           160.933 |           1.61 |        13:33:00 |    64     4     2  91.4 |
# |  DYDX/USDT |        64 |           1.25 |          79.90 |           159.421 |           1.59 |        14:09:00 |    57     5     2  89.1 |
# |  OSMO/USDT |        40 |           1.98 |          79.24 |           149.002 |           1.49 |        17:36:00 |    34     5     1  85.0 |
# |    OP/USDT |        47 |           1.56 |          73.54 |           145.772 |           1.46 |        20:57:00 |    39     6     2  83.0 |
# | THETA/USDT |        75 |           0.87 |          65.10 |           145.628 |           1.46 |        12:00:00 |    69     6     0   100 |
# |   ENJ/USDT |        65 |           1.18 |          76.58 |           142.510 |           1.43 |        14:16:00 |    59     5     1  90.8 |
# |  NEAR/USDT |        44 |           1.47 |          64.53 |           131.603 |           1.32 |        22:12:00 |    36     6     2  81.8 |
# |  KLAY/USDT |        69 |           0.85 |          58.61 |           119.886 |           1.20 |        14:04:00 |    61     7     1  88.4 |
# |  KAVA/USDT |        56 |           1.10 |          61.48 |           107.707 |           1.08 |        17:29:00 |    47     7     2  83.9 |
# |  AVAX/USDT |        55 |           1.08 |          59.30 |           104.919 |           1.05 |        17:24:00 |    47     7     1  85.5 |
# |   SOL/USDT |        39 |           1.58 |          61.45 |            95.080 |           0.95 |  1 day, 1:12:00 |    35     3     1  89.7 |
# |   ETC/USDT |        32 |           1.44 |          46.10 |            85.405 |           0.85 |  1 day, 6:38:00 |    26     5     1  81.2 |
# | MATIC/USDT |        54 |           0.79 |          42.50 |            83.013 |           0.83 |        18:09:00 |    45     8     1  83.3 |
# |   ADA/USDT |        54 |           0.77 |          41.55 |            82.207 |           0.82 |        17:57:00 |    48     6     0   100 |
# |  EGLD/USDT |        71 |           0.54 |          38.56 |            79.878 |           0.80 |        13:43:00 |    65     5     1  91.5 |
# |   XTZ/USDT |        62 |           0.79 |          49.21 |            77.834 |           0.78 |        15:40:00 |    54     6     2  87.1 |
# |   DOT/USDT |        46 |           0.98 |          44.98 |            71.382 |           0.71 |        21:34:00 |    41     4     1  89.1 |
# |  ALGO/USDT |        59 |           0.66 |          39.09 |            71.233 |           0.71 |        16:23:00 |    52     6     1  88.1 |
# |   VET/USDT |        51 |           0.61 |          31.18 |            60.874 |           0.61 |        19:16:00 |    44     6     1  86.3 |
# |   QNT/USDT |        36 |           0.79 |          28.29 |            58.083 |           0.58 |  1 day, 4:10:00 |    31     4     1  86.1 |
# |  IOTA/USDT |        51 |           0.56 |          28.32 |            52.226 |           0.52 |        19:40:00 |    42     8     1  82.4 |
# |  ATOM/USDT |        47 |           0.71 |          33.14 |            48.707 |           0.49 |        20:46:00 |    38     8     1  80.9 |
# |   BTC/USDT |        43 |           0.65 |          28.06 |            48.449 |           0.48 |        22:52:00 |    36     7     0   100 |
# |  DOGE/USDT |        38 |           0.59 |          22.49 |            38.060 |           0.38 |  1 day, 2:05:00 |    32     5     1  84.2 |
# |  LINK/USDT |        41 |           0.59 |          24.12 |            34.947 |           0.35 |        23:56:00 |    34     6     1  82.9 |
# |   XRP/USDT |        38 |           0.60 |          22.92 |            34.514 |           0.35 |  1 day, 2:12:00 |    33     4     1  86.8 |
# |   TRX/USDT |        24 |           0.82 |          19.65 |            30.057 |           0.30 | 1 day, 18:02:00 |    20     4     0   100 |
# |   UNI/USDT |        43 |           0.54 |          23.36 |            29.039 |           0.29 |        23:07:00 |    37     5     1  86.0 |
# |   ZEC/USDT |        29 |           0.75 |          21.63 |            28.218 |           0.28 | 1 day, 10:39:00 |    26     2     1  89.7 |
# |   XLM/USDT |        36 |           0.48 |          17.14 |            19.628 |           0.20 |  1 day, 4:15:00 |    32     3     1  88.9 |
# |   ETH/USDT |        15 |           0.57 |           8.60 |            19.366 |           0.19 |  1 day, 3:30:00 |    12     3     0   100 |
# |   XDC/USDT |        59 |           0.15 |           8.96 |            -0.376 |          -0.00 |        16:49:00 |    53     5     1  89.8 |
# |      TOTAL |      2539 |           1.26 |        3204.57 |          6734.393 |          67.34 |        17:08:00 |  2232   249    58  87.9 |
# =================================================================== ENTER TAG STATS ===================================================================
# |                   TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
# |-----------------------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
# |             WT - bull |      1642 |           1.25 |        2051.71 |          4198.825 |          41.99 |       16:33:00 |  1447   160    35  88.1 |
# |           RSI-XO bull |       738 |           1.24 |         918.32 |          1933.208 |          19.33 |       19:20:00 |   638    77    23  86.4 |
# |             WT - bear |       134 |           1.32 |         176.68 |           468.563 |           4.69 |       13:49:00 |   123    11     0   100 |
# | ttm_Squeeze/WT - bull |        12 |           3.42 |          40.99 |           101.769 |           1.02 |        6:52:00 |    12     0     0   100 |
# |           RSI-XO bear |        13 |           1.30 |          16.87 |            32.028 |           0.32 |        8:43:00 |    12     1     0   100 |
# |                 TOTAL |      2539 |           1.26 |        3204.57 |          6734.393 |          67.34 |       17:08:00 |  2232   249    58  87.9 |
# ======================================================= EXIT REASON STATS ========================================================
# |        Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
# |--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
# | WT transition bull |    1836 |   1836     0     0   100 |           0.8  |        1461.08 |          3045.38  |          32.47 |
# | trailing_stop_loss |     396 |    369     0    27  93.2 |           4.55 |        1802.3  |          4114.74  |          40.05 |
# |                roi |     273 |     24   249     0   100 |           0.5  |         137.35 |           285.285 |           3.05 |
# |         force_exit |      34 |      3     0    31   8.8 |          -5.77 |        -196.16 |          -711.015 |          -4.36 |
# =========================================================== LEFT OPEN TRADES REPORT ===========================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |      Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+-------------------+-------------------------|
# |   AKT/USDT |         1 |           1.03 |           1.03 |             1.983 |           0.02 |           1:00:00 |     1     0     0   100 |
# |   TRX/USDT |         1 |           0.62 |           0.62 |             1.188 |           0.01 |           4:30:00 |     1     0     0   100 |
# |  SCRT/USDT |         1 |           0.31 |           0.31 |             0.601 |           0.01 |           0:45:00 |     1     0     0   100 |
# | MATIC/USDT |         1 |          -0.96 |          -0.96 |            -3.608 |          -0.04 |   6 days, 9:30:00 |     0     0     1     0 |
# |   LDO/USDT |         1 |          -1.20 |          -1.20 |            -4.527 |          -0.05 |  6 days, 22:00:00 |     0     0     1     0 |
# |   IMX/USDT |         1 |          -1.43 |          -1.43 |            -5.484 |          -0.05 |           0:30:00 |     0     0     1     0 |
# |  RNDR/USDT |         1 |          -1.44 |          -1.44 |            -5.516 |          -0.06 |           1:30:00 |     0     0     1     0 |
# |  EGLD/USDT |         1 |          -1.81 |          -1.81 |            -6.844 |          -0.07 |   7 days, 7:15:00 |     0     0     1     0 |
# |   FIL/USDT |         1 |          -2.14 |          -2.14 |            -8.108 |          -0.08 |   7 days, 7:45:00 |     0     0     1     0 |
# |  HBAR/USDT |         1 |          -3.00 |          -3.00 |           -11.356 |          -0.11 |   3 days, 8:45:00 |     0     0     1     0 |
# |   VET/USDT |         1 |          -3.19 |          -3.19 |           -12.062 |          -0.12 |   7 days, 9:45:00 |     0     0     1     0 |
# |  IOTA/USDT |         1 |          -3.36 |          -3.36 |           -12.691 |          -0.13 |  7 days, 18:45:00 |     0     0     1     0 |
# |   ZEC/USDT |         1 |          -6.30 |          -6.30 |           -19.084 |          -0.19 |  25 days, 0:15:00 |     0     0     1     0 |
# |   UNI/USDT |         1 |          -5.39 |          -5.39 |           -19.253 |          -0.19 |  11 days, 3:15:00 |     0     0     1     0 |
# |   XLM/USDT |         1 |          -6.08 |          -6.08 |           -19.256 |          -0.19 | 22 days, 14:15:00 |     0     0     1     0 |
# |   ETC/USDT |         1 |          -5.99 |          -5.99 |           -21.531 |          -0.22 | 10 days, 19:45:00 |     0     0     1     0 |
# | OCEAN/USDT |         1 |          -5.75 |          -5.75 |           -21.785 |          -0.22 |   6 days, 8:15:00 |     0     0     1     0 |
# |   XRP/USDT |         1 |          -6.97 |          -6.97 |           -21.927 |          -0.22 | 22 days, 23:30:00 |     0     0     1     0 |
# |  LINK/USDT |         1 |          -6.35 |          -6.35 |           -24.048 |          -0.24 |   6 days, 6:00:00 |     0     0     1     0 |
# |  ALGO/USDT |         1 |          -6.39 |          -6.39 |           -24.219 |          -0.24 |   7 days, 6:30:00 |     0     0     1     0 |
# |   QNT/USDT |         1 |          -7.60 |          -7.60 |           -24.366 |          -0.24 | 20 days, 23:15:00 |     0     0     1     0 |
# |   DOT/USDT |         1 |          -7.02 |          -7.02 |           -24.723 |          -0.25 |  12 days, 4:00:00 |     0     0     1     0 |
# |  OSMO/USDT |         1 |          -6.90 |          -6.90 |           -25.965 |          -0.26 |  6 days, 22:45:00 |     0     0     1     0 |
# |  ATOM/USDT |         1 |          -7.08 |          -7.08 |           -26.581 |          -0.27 |  6 days, 21:30:00 |     0     0     1     0 |
# |   XTZ/USDT |         1 |          -7.05 |          -7.05 |           -26.654 |          -0.27 |  7 days, 13:45:00 |     0     0     1     0 |
# |   ENJ/USDT |         1 |          -7.73 |          -7.73 |           -29.174 |          -0.29 |  7 days, 19:30:00 |     0     0     1     0 |
# |    OP/USDT |         1 |          -8.74 |          -8.74 |           -33.011 |          -0.33 |  7 days, 19:45:00 |     0     0     1     0 |
# |   XDC/USDT |         1 |          -9.53 |          -9.53 |           -33.663 |          -0.34 | 11 days, 21:00:00 |     0     0     1     0 |
# |   SOL/USDT |         1 |         -10.18 |         -10.18 |           -33.937 |          -0.34 | 16 days, 23:30:00 |     0     0     1     0 |
# |  DOGE/USDT |         1 |          -9.51 |          -9.51 |           -34.058 |          -0.34 |  11 days, 2:00:00 |     0     0     1     0 |
# |  NEAR/USDT |         1 |         -10.02 |         -10.02 |           -37.974 |          -0.38 |   7 days, 6:15:00 |     0     0     1     0 |
# |  AVAX/USDT |         1 |         -12.67 |         -12.67 |           -43.741 |          -0.44 |  13 days, 6:45:00 |     0     0     1     0 |
# |  CSPR/USDT |         1 |         -12.30 |         -12.30 |           -46.448 |          -0.46 |  7 days, 21:15:00 |     0     0     1     0 |
# |  KAVA/USDT |         1 |         -14.05 |         -14.05 |           -53.195 |          -0.53 |   7 days, 8:30:00 |     0     0     1     0 |
# |      TOTAL |        34 |          -5.77 |        -196.16 |          -711.015 |          -7.11 |  8 days, 23:24:00 |     3     0    31   8.8 |
# ================== SUMMARY METRICS ==================
# | Metric                      | Value               |
# |-----------------------------+---------------------|
# | Backtesting from            | 2023-01-01 00:00:00 |
# | Backtesting to              | 2023-02-16 00:00:00 |
# | Max open trades             | 45                  |
# |                             |                     |
# | Total/Daily Avg Trades      | 2539 / 55.2         |
# | Starting balance            | 10000 USDT          |
# | Final balance               | 16734.393 USDT      |
# | Absolute profit             | 6734.393 USDT       |
# | Total profit %              | 67.34%              |
# | CAGR %                      | 5846.98%            |
# | Sortino                     | 86.72               |
# | Sharpe                      | 344.76              |
# | Calmar                      | 706.66              |
# | Profit factor               | 6.08                |
# | Expectancy                  | 0.02                |
# | Trades per day              | 55.2                |
# | Avg. daily profit %         | 1.46%               |
# | Avg. stake amount           | 221.665 USDT        |
# | Total trade volume          | 562807.669 USDT     |
# |                             |                     |
# | Best Pair                   | AGIX/USDT 219.69%   |
# | Worst Pair                  | ETH/USDT 8.60%      |
# | Best trade                  | LDO/USDT 51.65%     |
# | Worst trade                 | RNDR/USDT -32.61%   |
# | Best day                    | 522.912 USDT        |
# | Worst day                   | -436.965 USDT       |
# | Days win/draw/lose          | 41 / 1 / 4          |
# | Avg. Duration Winners       | 7:07:00             |
# | Avg. Duration Loser         | 5 days, 11:58:00    |
# | Rejected Entry signals      | 672                 |
# | Entry/Exit Timeouts         | 0 / 0               |
# |                             |                     |
# | Min balance                 | 10000.046 USDT      |
# | Max balance                 | 17445.973 USDT      |
# | Max % of account underwater | 3.96%               |
# | Absolute Drawdown (Account) | 3.96%               |
# | Absolute Drawdown           | 689.65 USDT         |
# | Drawdown high               | 7424.042 USDT       |
# | Drawdown low                | 6734.393 USDT       |
# | Drawdown Start              | 2023-02-15 21:00:00 |
# | Drawdown End                | 2023-02-16 00:00:00 |
# | Market change               | 114.44%             |
# =====================================================


# {
#   "strategy_name": "thetank2",
#   "params": {
#     "max_open_trades": {
#       "max_open_trades": 45
#     },
#     "buy": {
#       "buy_rsi": 26,
#       "buy_rsi_bear": 49,
#       "buy_rsi_bull": 75,
#       "buy_wt_bear": 50,
#       "buy_wt_bull": 72,
#       "dca1": 0.01,
#       "dca2": 0.12,
#       "dca3": 0.15,
#       "max_epa": 0,
#       "wavelength": 8
#     },
#     "sell": {
#       "crosslength": 9,
#       "sell_rsi": 58,
#       "ts0": 0.011,
#       "ts1": 0.011,
#       "ts2": 0.017,
#       "ts3": 0.03,
#       "ts4": 0.046,
#       "ts5": 0.045,
#       "tsl_target0": 0.049,
#       "tsl_target1": 0.058,
#       "tsl_target2": 0.092,
#       "tsl_target3": 0.116,
#       "tsl_target4": 0.191,
#       "tsl_target5": 0.2
#     },
#     "protection": {
#       "cooldown_lookback": 25,
#       "stop_duration": 181,
#       "use_stop_protection": false
#     },
#     "roi": {
#       "0": 0.517,
#       "328": 0.121,
#       "947": 0.071,
#       "2365": 0
#     },
#     "stoploss": {
#       "stoploss": -0.334
#     },
#     "trailing": {
#       "trailing_stop": true,
#       "trailing_stop_positive": 0.33,
#       "trailing_stop_positive_offset": 0.398,
#       "trailing_only_offset_is_reached": false
#     }
#   },
#   "ft_stratparam_v": 1,
#   "export_time": "2023-02-17 01:55:25.719816+00:00"
# }


### pivot points changes
# 3-02-21 16:43:47,576 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using timeframe: 15m
# 2023-02-21 16:43:47,576 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using stoploss: -0.319
# 2023-02-21 16:43:47,576 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_stop: True
# 2023-02-21 16:43:47,576 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_stop_positive: 0.104
# 2023-02-21 16:43:47,576 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_stop_positive_offset: 0.179
# 2023-02-21 16:43:47,576 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_only_offset_is_reached: True
# 2023-02-21 16:43:47,576 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using use_custom_stoploss: True
# 2023-02-21 16:43:47,576 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using process_only_new_candles: True
# 2023-02-21 16:43:47,576 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using order_types: {'entry': 'limit', 'exit': 'limit', 'stoploss': 'limit', 'stoploss_on_exchange': False, 'stoploss_on_exchange_interval': 60}
# 2023-02-21 16:43:47,576 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using order_time_in_force: {'entry': 'GTC', 'exit': 'GTC'}
# 2023-02-21 16:43:47,576 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using stake_currency: USDT
# 2023-02-21 16:43:47,577 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using stake_amount: unlimited
# 2023-02-21 16:43:47,577 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using protections: [{'method': 'CooldownPeriod', 'stop_duration_candles': 16}, {'method': 'StoplossGuard', 'lookback_period_candles': 72, 'trade_limit': 1, 'stop_duration_candles': 5, 'only_per_pair': False}]
# 2023-02-21 16:43:47,577 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using startup_candle_count: 30
# 2023-02-21 16:43:47,577 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using unfilledtimeout: {'entry': 30, 'exit': 30, 'unit': 'minutes', 'exit_timeout_count': 0}
# 2023-02-21 16:43:47,577 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using use_exit_signal: True
# 2023-02-21 16:43:47,577 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using exit_profit_only: True
# 2023-02-21 16:43:47,577 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using ignore_roi_if_entry_signal: True
# 2023-02-21 16:43:47,577 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using exit_profit_offset: 0.0
# 2023-02-21 16:43:47,577 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using disable_dataframe_checks: False
# 2023-02-21 16:43:47,577 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using ignore_buying_expired_candle_after: 0
# 2023-02-21 16:43:47,577 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using position_adjustment_enable: True
# 2023-02-21 16:43:47,577 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using max_entry_position_adjustment: 0
# 2023-02-21 16:43:47,577 - freqtrade.configuration.config_validation - INFO - Validating configuration ...
# 2023-02-21 16:43:47,582 - freqtrade.resolvers.iresolver - INFO - Using resolved pairlist StaticPairList from '/home/jared/freq/freqtrade/plugins/pairlist/StaticPairList.py'...
# 2023-02-21 16:43:47,668 - freqtrade.data.history.history_utils - INFO - Using indicator startup period: 30 ...
# 2023-02-21 16:43:48,279 - freqtrade.data.history.idatahandler - INFO - Price jump in AKT/USDT, 15m, spot between two candles of 58.17% detected.
# 2023-02-21 16:43:48,402 - freqtrade.data.history.idatahandler - INFO - Price jump in IOTA/USDT, 15m, spot between two candles of 47.10% detected.
# 2023-02-21 16:43:48,653 - freqtrade.data.history.idatahandler - WARNING - OSMO/USDT, spot, 15m, data starts at 2023-01-13 10:00:00
# 2023-02-21 16:43:53,370 - freqtrade.optimize.backtesting - INFO - Loading data from 2022-12-31 16:30:00 up to 2023-02-20 00:00:00 (50 days).
# 2023-02-21 16:43:53,370 - freqtrade.optimize.backtesting - INFO - Dataload complete. Calculating indicators
# 2023-02-21 16:43:53,378 - freqtrade.optimize.backtesting - INFO - Running backtesting for Strategy thetank3
# 2023-02-21 16:43:53,378 - freqtrade.strategy.hyper - INFO - Strategy Parameter: buy_rsi = 29
# 2023-02-21 16:43:53,378 - freqtrade.strategy.hyper - INFO - Strategy Parameter: buy_rsi_bear = 53
# 2023-02-21 16:43:53,378 - freqtrade.strategy.hyper - INFO - Strategy Parameter: buy_rsi_bull = 67
# 2023-02-21 16:43:53,378 - freqtrade.strategy.hyper - INFO - Strategy Parameter: buy_wt_bear = 23
# 2023-02-21 16:43:53,378 - freqtrade.strategy.hyper - INFO - Strategy Parameter: buy_wt_bull = 65
# 2023-02-21 16:43:53,379 - freqtrade.strategy.hyper - INFO - Strategy Parameter: dca1 = 0.06
# 2023-02-21 16:43:53,379 - freqtrade.strategy.hyper - INFO - Strategy Parameter: dca2 = 0.09
# 2023-02-21 16:43:53,379 - freqtrade.strategy.hyper - INFO - Strategy Parameter: dca3 = 0.17
# 2023-02-21 16:43:53,379 - freqtrade.strategy.hyper - INFO - Strategy Parameter: filterlength = 26
# 2023-02-21 16:43:53,379 - freqtrade.strategy.hyper - INFO - Strategy Parameter: max_epa = 3
# 2023-02-21 16:43:53,379 - freqtrade.strategy.hyper - INFO - Strategy Parameter: wavelength = 10
# 2023-02-21 16:43:53,379 - freqtrade.strategy.hyper - INFO - Strategy Parameter: crosslength = 4
# 2023-02-21 16:43:53,379 - freqtrade.strategy.hyper - INFO - Strategy Parameter: sell_rsi = 54
# 2023-02-21 16:43:53,379 - freqtrade.strategy.hyper - INFO - Strategy Parameter: ts0 = 0.008
# 2023-02-21 16:43:53,379 - freqtrade.strategy.hyper - INFO - Strategy Parameter: ts1 = 0.01
# 2023-02-21 16:43:53,379 - freqtrade.strategy.hyper - INFO - Strategy Parameter: ts2 = 0.016
# 2023-02-21 16:43:53,380 - freqtrade.strategy.hyper - INFO - Strategy Parameter: ts3 = 0.025
# 2023-02-21 16:43:53,380 - freqtrade.strategy.hyper - INFO - Strategy Parameter: ts4 = 0.033
# 2023-02-21 16:43:53,380 - freqtrade.strategy.hyper - INFO - Strategy Parameter: ts5 = 0.048
# 2023-02-21 16:43:53,380 - freqtrade.strategy.hyper - INFO - Strategy Parameter: tsl_target0 = 0.045
# 2023-02-21 16:43:53,380 - freqtrade.strategy.hyper - INFO - Strategy Parameter: tsl_target1 = 0.053
# 2023-02-21 16:43:53,380 - freqtrade.strategy.hyper - INFO - Strategy Parameter: tsl_target2 = 0.093
# 2023-02-21 16:43:53,380 - freqtrade.strategy.hyper - INFO - Strategy Parameter: tsl_target3 = 0.141
# 2023-02-21 16:43:53,380 - freqtrade.strategy.hyper - INFO - Strategy Parameter: tsl_target4 = 0.265
# 2023-02-21 16:43:53,380 - freqtrade.strategy.hyper - INFO - Strategy Parameter: tsl_target5 = 0.2
# 2023-02-21 16:43:53,380 - freqtrade.strategy.hyper - INFO - Strategy Parameter: cooldown_lookback = 11
# 2023-02-21 16:43:53,380 - freqtrade.strategy.hyper - INFO - Strategy Parameter: stop_duration = 48
# 2023-02-21 16:43:53,381 - freqtrade.strategy.hyper - INFO - Strategy Parameter: use_stop_protection = False
# 2023-02-21 16:43:58,822 - freqtrade.optimize.backtesting - INFO - Backtesting with data from 2023-01-01 00:00:00 up to 2023-02-20 00:00:00 (50 days).
# 2023-02-21 16:47:07,574 - freqtrade.misc - INFO - dumping json to "/home/jared/freq/user_data/backtest_results/backtest-result-2023-02-21_16-47-07.meta.json"
# 2023-02-21 16:47:07,575 - freqtrade.misc - INFO - dumping json to "/home/jared/freq/user_data/backtest_results/backtest-result-2023-02-21_16-47-07.json"
# 2023-02-21 16:47:07,615 - freqtrade.misc - INFO - dumping json to "/home/jared/freq/user_data/backtest_results/.last_result.json"
# Result for strategy thetank3
# ============================================================= BACKTESTING REPORT =============================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# |  AGIX/USDT |        40 |           4.48 |         179.28 |            51.240 |           5.12 |         18:52:00 |    37     2     1  92.5 |
# |   LDO/USDT |        35 |           3.78 |         132.44 |            36.126 |           3.61 |   1 day, 0:02:00 |    28     4     3  80.0 |
# |  RNDR/USDT |        46 |           2.37 |         108.97 |            32.203 |           3.22 |         15:14:00 |    41     3     2  89.1 |
# |   AKT/USDT |        34 |           2.67 |          90.82 |            27.604 |           2.76 |   1 day, 0:30:00 |    26     7     1  76.5 |
# |   INJ/USDT |        35 |           2.32 |          81.18 |            24.033 |           2.40 |         22:58:00 |    27     7     1  77.1 |
# |  SCRT/USDT |        34 |           2.15 |          73.15 |            23.488 |           2.35 |   1 day, 0:45:00 |    25     8     1  73.5 |
# |   IMX/USDT |        31 |           2.50 |          77.63 |            23.486 |           2.35 |   1 day, 1:56:00 |    25     5     1  80.6 |
# | OCEAN/USDT |        34 |           2.49 |          84.49 |            22.984 |           2.30 |   1 day, 0:26:00 |    31     2     1  91.2 |
# |   GMX/USDT |        33 |           2.09 |          68.83 |            22.188 |           2.22 |   1 day, 1:00:00 |    26     6     1  78.8 |
# |   GRT/USDT |        31 |           2.13 |          65.99 |            19.237 |           1.92 |   1 day, 3:19:00 |    27     3     1  87.1 |
# |   ENJ/USDT |        33 |           1.90 |          62.73 |            18.204 |           1.82 |   1 day, 1:55:00 |    27     5     1  81.8 |
# |  IOTA/USDT |        33 |           1.86 |          61.44 |            17.572 |           1.76 |   1 day, 2:05:00 |    28     5     0   100 |
# |  HBAR/USDT |        31 |           1.85 |          57.25 |            17.516 |           1.75 |   1 day, 2:56:00 |    25     5     1  80.6 |
# | THETA/USDT |        32 |           1.66 |          53.27 |            16.856 |           1.69 |   1 day, 2:30:00 |    25     6     1  78.1 |
# |  NEAR/USDT |        29 |           2.17 |          62.80 |            16.781 |           1.68 |   1 day, 5:43:00 |    23     5     1  79.3 |
# |  OSMO/USDT |        23 |           2.18 |          50.11 |            16.366 |           1.64 |   1 day, 2:50:00 |    19     3     1  82.6 |
# |   FIL/USDT |        27 |           1.97 |          53.14 |            15.925 |           1.59 |   1 day, 6:12:00 |    17     9     1  63.0 |
# |  KLAY/USDT |        31 |           1.57 |          48.74 |            14.776 |           1.48 |   1 day, 2:14:00 |    23     8     0   100 |
# |    OP/USDT |        26 |           1.99 |          51.79 |            14.002 |           1.40 |  1 day, 10:00:00 |    18     6     2  69.2 |
# |  DYDX/USDT |        31 |           1.63 |          50.50 |            13.468 |           1.35 |   1 day, 2:22:00 |    25     4     2  80.6 |
# |  KAVA/USDT |        26 |           2.00 |          51.92 |            13.445 |           1.34 |  1 day, 11:20:00 |    21     4     1  80.8 |
# |   SOL/USDT |        21 |           2.47 |          51.83 |            13.220 |           1.32 |  1 day, 20:39:00 |    15     6     0   100 |
# |  ROSE/USDT |        30 |           1.45 |          43.61 |            13.126 |           1.31 |   1 day, 3:38:00 |    23     6     1  76.7 |
# |  EGLD/USDT |        34 |           1.15 |          39.16 |            11.870 |           1.19 |   1 day, 2:20:00 |    28     5     1  82.4 |
# |  ORAI/USDT |        30 |           1.80 |          54.04 |            11.217 |           1.12 |   1 day, 2:08:00 |    26     2     2  86.7 |
# |   XTZ/USDT |        26 |           1.53 |          39.85 |            10.767 |           1.08 |  1 day, 10:33:00 |    19     5     2  73.1 |
# |   ETC/USDT |        16 |           2.51 |          40.23 |            10.534 |           1.05 | 2 days, 14:27:00 |    11     4     1  68.8 |
# |  LINK/USDT |        35 |           0.97 |          33.98 |            10.426 |           1.04 |         23:55:00 |    27     7     1  77.1 |
# |  CSPR/USDT |        23 |           1.60 |          36.83 |             9.727 |           0.97 |  1 day, 18:00:00 |    17     5     1  73.9 |
# | MATIC/USDT |        28 |           1.24 |          34.59 |             9.583 |           0.96 |   1 day, 6:55:00 |    19     8     1  67.9 |
# |  AVAX/USDT |        21 |           1.77 |          37.08 |             9.514 |           0.95 |  1 day, 21:00:00 |    16     4     1  76.2 |
# |   VET/USDT |        31 |           1.05 |          32.70 |             9.224 |           0.92 |   1 day, 3:44:00 |    25     5     1  80.6 |
# |  DOGE/USDT |        25 |           1.23 |          30.86 |             8.109 |           0.81 |  1 day, 14:34:00 |    18     6     1  72.0 |
# |  ALGO/USDT |        23 |           1.23 |          28.25 |             7.626 |           0.76 |  1 day, 14:53:00 |    16     6     1  69.6 |
# |  ATOM/USDT |        25 |           1.10 |          27.61 |             7.368 |           0.74 |  1 day, 11:55:00 |    20     4     1  80.0 |
# |   TRX/USDT |        26 |           0.91 |          23.60 |             7.176 |           0.72 |  1 day, 11:13:00 |    18     7     1  69.2 |
# |   ETH/USDT |        28 |           0.84 |          23.56 |             6.473 |           0.65 |   1 day, 8:15:00 |    22     5     1  78.6 |
# |   XLM/USDT |        28 |           0.83 |          23.16 |             6.227 |           0.62 |   1 day, 8:18:00 |    20     7     1  71.4 |
# |   ADA/USDT |        24 |           0.97 |          23.34 |             6.039 |           0.60 |  1 day, 13:11:00 |    16     7     1  66.7 |
# |   BTC/USDT |        28 |           0.56 |          15.78 |             4.239 |           0.42 |   1 day, 6:27:00 |    24     3     1  85.7 |
# |   DOT/USDT |        23 |           0.69 |          15.95 |             4.228 |           0.42 |  1 day, 13:53:00 |    16     6     1  69.6 |
# |   UNI/USDT |        20 |           0.73 |          14.70 |             3.762 |           0.38 |  1 day, 23:53:00 |    12     7     1  60.0 |
# |   ZEC/USDT |        13 |           1.14 |          14.88 |             3.715 |           0.37 |  3 days, 9:15:00 |     9     4     0   100 |
# |   QNT/USDT |        22 |           0.46 |          10.10 |             2.111 |           0.21 |  1 day, 21:52:00 |    18     3     1  81.8 |
# |   XRP/USDT |        16 |           0.25 |           3.95 |             0.377 |           0.04 | 2 days, 17:52:00 |    12     3     1  75.0 |
# |   XDC/USDT |        24 |          -0.00 |          -0.06 |            -0.696 |          -0.07 |  1 day, 16:19:00 |    17     6     1  70.8 |
# |      TOTAL |      1295 |           1.75 |        2266.02 |           643.464 |          64.35 |   1 day, 7:01:00 |  1008   238    49  77.8 |
# =================================================================== ENTER TAG STATS ====================================================================
# |                   TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |
# |-----------------------+-----------+----------------+----------------+-------------------+----------------+-----------------+-------------------------|
# |           RSI-XO bull |       470 |           1.80 |         843.69 |           240.266 |          24.03 |  1 day, 6:35:00 |   360    93    17  76.6 |
# |             WT - bull |       374 |           1.63 |         610.01 |           174.715 |          17.47 | 1 day, 10:05:00 |   286    74    14  76.5 |
# |               S2 - XO |       200 |           1.65 |         329.06 |            86.370 |           8.64 |  1 day, 1:17:00 |   161    31     8  80.5 |
# |               S3 - XO |       106 |           1.99 |         211.30 |            59.029 |           5.90 |        20:36:00 |    92     9     5  86.8 |
# |           RSI-XO bear |        58 |           2.33 |         135.00 |            39.659 |           3.97 |        19:35:00 |    47    10     1  81.0 |
# | ttm_Squeeze/WT - bull |        33 |           2.04 |          67.41 |            22.182 |           2.22 | 2 days, 7:34:00 |    25     7     1  75.8 |
# |             WT - bear |        53 |           1.24 |          65.71 |            19.870 |           1.99 | 2 days, 5:19:00 |    36    14     3  67.9 |
# | ttm_Squeeze/WT - bear |         1 |           3.84 |           3.84 |             1.373 |           0.14 |         9:00:00 |     1     0     0   100 |
# |                 TOTAL |      1295 |           1.75 |        2266.02 |           643.464 |          64.35 |  1 day, 7:01:00 |  1008   238    49  77.8 |
# ======================================================= EXIT REASON STATS ========================================================
# |        Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
# |--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
# | trailing_stop_loss |     393 |    386     0     7  98.2 |           4.67 |        1835.3  |           547.526 |          40.78 |
# |                roi |     271 |     33   238     0   100 |           0.56 |         152.48 |            41.018 |           3.39 |
# |            R3 - XO |     234 |    234     0     0   100 |           1.24 |         289.21 |            77.611 |           6.43 |
# |         R2.25 - XO |     100 |    100     0     0   100 |           0.65 |          65.46 |            19.207 |           1.45 |
# |         R2.75 - XO |      99 |     99     0     0   100 |           0.91 |          90.23 |            25.376 |           2.01 |
# |          R2.5 - XO |      90 |     90     0     0   100 |           0.7  |          63.11 |            18.483 |           1.4  |
# |            R2 - XO |      63 |     63     0     0   100 |           0.58 |          36.52 |            11.01  |           0.81 |
# |         force_exit |      45 |      3     0    42   6.7 |          -5.92 |        -266.29 |           -96.766 |          -5.92 |
# =========================================================== LEFT OPEN TRADES REPORT ===========================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |      Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+-------------------+-------------------------|
# |  IOTA/USDT |         1 |           1.29 |           1.29 |             0.493 |           0.05 |           2:45:00 |     1     0     0   100 |
# |   ZEC/USDT |         1 |           0.96 |           0.96 |             0.369 |           0.04 |           2:45:00 |     1     0     0   100 |
# |   SOL/USDT |         1 |           0.43 |           0.43 |             0.165 |           0.02 |           3:00:00 |     1     0     0   100 |
# |   ENJ/USDT |         1 |          -0.32 |          -0.32 |            -0.124 |          -0.01 |           1:45:00 |     0     0     1     0 |
# |   ETH/USDT |         1 |          -0.63 |          -0.63 |            -0.240 |          -0.02 |           3:00:00 |     0     0     1     0 |
# |   ADA/USDT |         1 |          -0.72 |          -0.72 |            -0.276 |          -0.03 |           2:45:00 |     0     0     1     0 |
# |   TRX/USDT |         1 |          -0.77 |          -0.77 |            -0.295 |          -0.03 |           3:00:00 |     0     0     1     0 |
# |   XLM/USDT |         1 |          -0.78 |          -0.78 |            -0.299 |          -0.03 |           2:15:00 |     0     0     1     0 |
# |   AKT/USDT |         1 |          -0.88 |          -0.88 |            -0.338 |          -0.03 |           0:15:00 |     0     0     1     0 |
# |  HBAR/USDT |         1 |          -0.91 |          -0.91 |            -0.350 |          -0.04 |           1:00:00 |     0     0     1     0 |
# |   UNI/USDT |         1 |          -1.04 |          -1.04 |            -0.399 |          -0.04 |           1:30:00 |     0     0     1     0 |
# |   FIL/USDT |         1 |          -1.12 |          -1.12 |            -0.429 |          -0.04 |           2:30:00 |     0     0     1     0 |
# |   BTC/USDT |         1 |          -1.22 |          -1.22 |            -0.467 |          -0.05 |           3:00:00 |     0     0     1     0 |
# |   VET/USDT |         1 |          -1.53 |          -1.53 |            -0.586 |          -0.06 |           2:45:00 |     0     0     1     0 |
# |  LINK/USDT |         1 |          -1.78 |          -1.78 |            -0.679 |          -0.07 |          12:30:00 |     0     0     1     0 |
# | THETA/USDT |         1 |          -1.97 |          -1.97 |            -0.751 |          -0.08 |   1 day, 10:45:00 |     0     0     1     0 |
# |   DOT/USDT |         1 |          -2.28 |          -2.28 |            -0.875 |          -0.09 |           3:00:00 |     0     0     1     0 |
# |   XTZ/USDT |         1 |          -2.75 |          -2.75 |            -0.998 |          -0.10 | 11 days, 11:45:00 |     0     0     1     0 |
# |  OSMO/USDT |         1 |          -2.79 |          -2.79 |            -1.066 |          -0.11 |           7:30:00 |     0     0     1     0 |
# |  EGLD/USDT |         1 |          -2.86 |          -2.86 |            -1.089 |          -0.11 |    1 day, 5:15:00 |     0     0     1     0 |
# | MATIC/USDT |         1 |          -3.43 |          -3.43 |            -1.301 |          -0.13 |   1 day, 11:45:00 |     0     0     1     0 |
# |  SCRT/USDT |         1 |          -3.76 |          -3.76 |            -1.439 |          -0.14 |           7:30:00 |     0     0     1     0 |
# |  NEAR/USDT |         1 |          -5.01 |          -5.01 |            -1.819 |          -0.18 |  11 days, 7:00:00 |     0     0     1     0 |
# |   GMX/USDT |         1 |          -4.90 |          -4.90 |            -1.867 |          -0.19 |    1 day, 9:30:00 |     0     0     1     0 |
# |   ETC/USDT |         1 |          -5.50 |          -5.50 |            -1.917 |          -0.19 | 14 days, 19:15:00 |     0     0     1     0 |
# |  ALGO/USDT |         1 |          -5.38 |          -5.38 |            -1.955 |          -0.20 |  11 days, 7:00:00 |     0     0     1     0 |
# |  DOGE/USDT |         1 |          -5.60 |          -5.60 |            -2.030 |          -0.20 | 11 days, 12:15:00 |     0     0     1     0 |
# |  ATOM/USDT |         1 |          -6.21 |          -6.21 |            -2.257 |          -0.23 |  11 days, 7:00:00 |     0     0     1     0 |
# |   INJ/USDT |         1 |          -6.70 |          -6.70 |            -2.553 |          -0.26 |          23:00:00 |     0     0     1     0 |
# |  DYDX/USDT |         1 |          -7.60 |          -7.60 |            -2.864 |          -0.29 |  2 days, 10:15:00 |     0     0     1     0 |
# |   XRP/USDT |         1 |          -9.25 |          -9.25 |            -2.873 |          -0.29 | 26 days, 20:00:00 |     0     0     1     0 |
# |   QNT/USDT |         1 |         -10.11 |         -10.11 |            -3.259 |          -0.33 | 22 days, 20:00:00 |     0     0     1     0 |
# |   XDC/USDT |         1 |          -9.66 |          -9.66 |            -3.318 |          -0.33 | 15 days, 21:00:00 |     0     0     1     0 |
# |   GRT/USDT |         1 |          -9.64 |          -9.64 |            -3.492 |          -0.35 | 11 days, 11:45:00 |     0     0     1     0 |
# |   LDO/USDT |         1 |          -9.26 |          -9.26 |            -3.531 |          -0.35 |          21:45:00 |     0     0     1     0 |
# |  AVAX/USDT |         1 |         -10.56 |         -10.56 |            -3.582 |          -0.36 |  17 days, 3:00:00 |     0     0     1     0 |
# | OCEAN/USDT |         1 |         -12.48 |         -12.48 |            -4.473 |          -0.45 |  12 days, 0:15:00 |     0     0     1     0 |
# |  KAVA/USDT |         1 |         -12.55 |         -12.55 |            -4.562 |          -0.46 |  11 days, 8:15:00 |     0     0     1     0 |
# |   IMX/USDT |         1 |         -12.23 |         -12.23 |            -4.577 |          -0.46 |  2 days, 21:30:00 |     0     0     1     0 |
# |  ROSE/USDT |         1 |         -12.50 |         -12.50 |            -4.620 |          -0.46 |  3 days, 20:15:00 |     0     0     1     0 |
# |  CSPR/USDT |         1 |         -13.71 |         -13.71 |            -4.942 |          -0.49 | 11 days, 21:15:00 |     0     0     1     0 |
# |    OP/USDT |         1 |         -14.52 |         -14.52 |            -5.255 |          -0.53 | 11 days, 12:45:00 |     0     0     1     0 |
# |  RNDR/USDT |         1 |         -14.20 |         -14.20 |            -5.309 |          -0.53 |   3 days, 5:00:00 |     0     0     1     0 |
# |  AGIX/USDT |         1 |         -17.17 |         -17.17 |            -6.346 |          -0.63 |  3 days, 17:30:00 |     0     0     1     0 |
# |  ORAI/USDT |         1 |         -22.70 |         -22.70 |            -8.391 |          -0.84 |  3 days, 22:00:00 |     0     0     1     0 |
# |      TOTAL |        45 |          -5.92 |        -266.29 |           -96.766 |          -9.68 |   5 days, 9:25:00 |     3     0    42   6.7 |
# ================== SUMMARY METRICS ==================
# | Metric                      | Value               |
# |-----------------------------+---------------------|
# | Backtesting from            | 2023-01-01 00:00:00 |
# | Backtesting to              | 2023-02-20 00:00:00 |
# | Max open trades             | 45                  |
# |                             |                     |
# | Total/Daily Avg Trades      | 1295 / 25.9         |
# | Starting balance            | 1000 USDT           |
# | Final balance               | 1643.464 USDT       |
# | Absolute profit             | 643.464 USDT        |
# | Total profit %              | 64.35%              |
# | CAGR %                      | 3658.80%            |
# | Profit factor               | 5.88                |
# | Trades per day              | 25.9                |
# | Avg. daily profit %         | 1.29%               |
# | Avg. stake amount           | 29.399 USDT         |
# | Total trade volume          | 38071.743 USDT      |
# |                             |                     |
# | Best Pair                   | AGIX/USDT 179.28%   |
# | Worst Pair                  | XDC/USDT -0.06%     |
# | Best trade                  | AKT/USDT 19.18%     |
# | Worst trade                 | ORAI/USDT -31.80%   |
# | Best day                    | 40.159 USDT         |
# | Worst day                   | -96.766 USDT        |
# | Days win/draw/lose          | 48 / 0 / 3          |
# | Avg. Duration Winners       | 12:32:00            |
# | Avg. Duration Loser         | 5 days, 5:33:00     |
# | Rejected Entry signals      | 237                 |
# | Entry/Exit Timeouts         | 0 / 0               |
# |                             |                     |
# | Min balance                 | 1000.11 USDT        |
# | Max balance                 | 1740.23 USDT        |
# | Max % of account underwater | 5.56%               |
# | Absolute Drawdown (Account) | 5.56%               |
# | Absolute Drawdown           | 96.766 USDT         |
# | Drawdown high               | 740.23 USDT         |
# | Drawdown low                | 643.464 USDT        |
# | Drawdown Start              | 2023-02-19 20:45:00 |
# | Drawdown End                | 2023-02-20 00:00:00 |
# | Market change               | 112.61%             |
# =====================================================

# ### with epa modifiier
# Result for strategy thetank3
# ============================================================= BACKTESTING REPORT =============================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# |  AGIX/USDT |        40 |           4.48 |         179.28 |            51.240 |           5.12 |         18:52:00 |    37     2     1  92.5 |
# |   LDO/USDT |        35 |           3.78 |         132.44 |            36.126 |           3.61 |   1 day, 0:02:00 |    28     4     3  80.0 |
# |  RNDR/USDT |        46 |           2.37 |         108.97 |            32.203 |           3.22 |         15:14:00 |    41     3     2  89.1 |
# |   AKT/USDT |        34 |           2.67 |          90.82 |            27.604 |           2.76 |   1 day, 0:30:00 |    26     7     1  76.5 |
# |   INJ/USDT |        35 |           2.32 |          81.18 |            24.033 |           2.40 |         22:58:00 |    27     7     1  77.1 |
# |  SCRT/USDT |        34 |           2.15 |          73.15 |            23.488 |           2.35 |   1 day, 0:45:00 |    25     8     1  73.5 |
# |   IMX/USDT |        31 |           2.50 |          77.63 |            23.486 |           2.35 |   1 day, 1:56:00 |    25     5     1  80.6 |
# | OCEAN/USDT |        34 |           2.49 |          84.49 |            22.984 |           2.30 |   1 day, 0:26:00 |    31     2     1  91.2 |
# |   GMX/USDT |        33 |           2.09 |          68.83 |            22.188 |           2.22 |   1 day, 1:00:00 |    26     6     1  78.8 |
# |   GRT/USDT |        31 |           2.13 |          65.99 |            19.237 |           1.92 |   1 day, 3:19:00 |    27     3     1  87.1 |
# |   ENJ/USDT |        33 |           1.90 |          62.73 |            18.204 |           1.82 |   1 day, 1:55:00 |    27     5     1  81.8 |
# |  IOTA/USDT |        33 |           1.86 |          61.44 |            17.572 |           1.76 |   1 day, 2:05:00 |    28     5     0   100 |
# |  HBAR/USDT |        31 |           1.85 |          57.25 |            17.516 |           1.75 |   1 day, 2:56:00 |    25     5     1  80.6 |
# | THETA/USDT |        32 |           1.66 |          53.27 |            16.856 |           1.69 |   1 day, 2:30:00 |    25     6     1  78.1 |
# |  NEAR/USDT |        29 |           2.17 |          62.80 |            16.781 |           1.68 |   1 day, 5:43:00 |    23     5     1  79.3 |
# |  OSMO/USDT |        23 |           2.18 |          50.11 |            16.366 |           1.64 |   1 day, 2:50:00 |    19     3     1  82.6 |
# |   FIL/USDT |        27 |           1.97 |          53.14 |            15.925 |           1.59 |   1 day, 6:12:00 |    17     9     1  63.0 |
# |  KLAY/USDT |        31 |           1.57 |          48.74 |            14.776 |           1.48 |   1 day, 2:14:00 |    23     8     0   100 |
# |    OP/USDT |        26 |           1.99 |          51.79 |            14.002 |           1.40 |  1 day, 10:00:00 |    18     6     2  69.2 |
# |  DYDX/USDT |        31 |           1.63 |          50.50 |            13.468 |           1.35 |   1 day, 2:22:00 |    25     4     2  80.6 |
# |  KAVA/USDT |        26 |           2.00 |          51.92 |            13.445 |           1.34 |  1 day, 11:20:00 |    21     4     1  80.8 |
# |   SOL/USDT |        21 |           2.47 |          51.83 |            13.220 |           1.32 |  1 day, 20:39:00 |    15     6     0   100 |
# |  ROSE/USDT |        30 |           1.45 |          43.61 |            13.126 |           1.31 |   1 day, 3:38:00 |    23     6     1  76.7 |
# |  EGLD/USDT |        34 |           1.15 |          39.16 |            11.870 |           1.19 |   1 day, 2:20:00 |    28     5     1  82.4 |
# |  ORAI/USDT |        30 |           1.80 |          54.04 |            11.217 |           1.12 |   1 day, 2:08:00 |    26     2     2  86.7 |
# |   XTZ/USDT |        26 |           1.53 |          39.85 |            10.767 |           1.08 |  1 day, 10:33:00 |    19     5     2  73.1 |
# |   ETC/USDT |        16 |           2.51 |          40.23 |            10.534 |           1.05 | 2 days, 14:27:00 |    11     4     1  68.8 |
# |  LINK/USDT |        35 |           0.97 |          33.98 |            10.426 |           1.04 |         23:55:00 |    27     7     1  77.1 |
# |  CSPR/USDT |        23 |           1.60 |          36.83 |             9.727 |           0.97 |  1 day, 18:00:00 |    17     5     1  73.9 |
# | MATIC/USDT |        28 |           1.24 |          34.59 |             9.583 |           0.96 |   1 day, 6:55:00 |    19     8     1  67.9 |
# |  AVAX/USDT |        21 |           1.77 |          37.08 |             9.514 |           0.95 |  1 day, 21:00:00 |    16     4     1  76.2 |
# |   VET/USDT |        31 |           1.05 |          32.70 |             9.224 |           0.92 |   1 day, 3:44:00 |    25     5     1  80.6 |
# |  DOGE/USDT |        25 |           1.23 |          30.86 |             8.109 |           0.81 |  1 day, 14:34:00 |    18     6     1  72.0 |
# |  ALGO/USDT |        23 |           1.23 |          28.25 |             7.626 |           0.76 |  1 day, 14:53:00 |    16     6     1  69.6 |
# |  ATOM/USDT |        25 |           1.10 |          27.61 |             7.368 |           0.74 |  1 day, 11:55:00 |    20     4     1  80.0 |
# |   TRX/USDT |        26 |           0.91 |          23.60 |             7.176 |           0.72 |  1 day, 11:13:00 |    18     7     1  69.2 |
# |   ETH/USDT |        28 |           0.84 |          23.56 |             6.473 |           0.65 |   1 day, 8:15:00 |    22     5     1  78.6 |
# |   XLM/USDT |        28 |           0.83 |          23.16 |             6.227 |           0.62 |   1 day, 8:18:00 |    20     7     1  71.4 |
# |   ADA/USDT |        24 |           0.97 |          23.34 |             6.039 |           0.60 |  1 day, 13:11:00 |    16     7     1  66.7 |
# |   BTC/USDT |        28 |           0.56 |          15.78 |             4.239 |           0.42 |   1 day, 6:27:00 |    24     3     1  85.7 |
# |   DOT/USDT |        23 |           0.69 |          15.95 |             4.228 |           0.42 |  1 day, 13:53:00 |    16     6     1  69.6 |
# |   UNI/USDT |        20 |           0.73 |          14.70 |             3.762 |           0.38 |  1 day, 23:53:00 |    12     7     1  60.0 |
# |   ZEC/USDT |        13 |           1.14 |          14.88 |             3.715 |           0.37 |  3 days, 9:15:00 |     9     4     0   100 |
# |   QNT/USDT |        22 |           0.46 |          10.10 |             2.111 |           0.21 |  1 day, 21:52:00 |    18     3     1  81.8 |
# |   XRP/USDT |        16 |           0.25 |           3.95 |             0.377 |           0.04 | 2 days, 17:52:00 |    12     3     1  75.0 |
# |   XDC/USDT |        24 |          -0.00 |          -0.06 |            -0.696 |          -0.07 |  1 day, 16:19:00 |    17     6     1  70.8 |
# |      TOTAL |      1295 |           1.75 |        2266.02 |           643.464 |          64.35 |   1 day, 7:01:00 |  1008   238    49  77.8 |
# =================================================================== ENTER TAG STATS ====================================================================
# |                   TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |
# |-----------------------+-----------+----------------+----------------+-------------------+----------------+-----------------+-------------------------|
# |           RSI-XO bull |       470 |           1.80 |         843.69 |           240.266 |          24.03 |  1 day, 6:35:00 |   360    93    17  76.6 |
# |             WT - bull |       374 |           1.63 |         610.01 |           174.715 |          17.47 | 1 day, 10:05:00 |   286    74    14  76.5 |
# |               S2 - XO |       200 |           1.65 |         329.06 |            86.370 |           8.64 |  1 day, 1:17:00 |   161    31     8  80.5 |
# |               S3 - XO |       106 |           1.99 |         211.30 |            59.029 |           5.90 |        20:36:00 |    92     9     5  86.8 |
# |           RSI-XO bear |        58 |           2.33 |         135.00 |            39.659 |           3.97 |        19:35:00 |    47    10     1  81.0 |
# | ttm_Squeeze/WT - bull |        33 |           2.04 |          67.41 |            22.182 |           2.22 | 2 days, 7:34:00 |    25     7     1  75.8 |
# |             WT - bear |        53 |           1.24 |          65.71 |            19.870 |           1.99 | 2 days, 5:19:00 |    36    14     3  67.9 |
# | ttm_Squeeze/WT - bear |         1 |           3.84 |           3.84 |             1.373 |           0.14 |         9:00:00 |     1     0     0   100 |
# |                 TOTAL |      1295 |           1.75 |        2266.02 |           643.464 |          64.35 |  1 day, 7:01:00 |  1008   238    49  77.8 |
# ======================================================= EXIT REASON STATS ========================================================
# |        Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
# |--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
# | trailing_stop_loss |     393 |    386     0     7  98.2 |           4.67 |        1835.3  |           547.526 |          40.78 |
# |                roi |     271 |     33   238     0   100 |           0.56 |         152.48 |            41.018 |           3.39 |
# |            R3 - XO |     234 |    234     0     0   100 |           1.24 |         289.21 |            77.611 |           6.43 |
# |         R2.25 - XO |     100 |    100     0     0   100 |           0.65 |          65.46 |            19.207 |           1.45 |
# |         R2.75 - XO |      99 |     99     0     0   100 |           0.91 |          90.23 |            25.376 |           2.01 |
# |          R2.5 - XO |      90 |     90     0     0   100 |           0.7  |          63.11 |            18.483 |           1.4  |
# |            R2 - XO |      63 |     63     0     0   100 |           0.58 |          36.52 |            11.01  |           0.81 |
# |         force_exit |      45 |      3     0    42   6.7 |          -5.92 |        -266.29 |           -96.766 |          -5.92 |
# =========================================================== LEFT OPEN TRADES REPORT ===========================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |      Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+-------------------+-------------------------|
# |  IOTA/USDT |         1 |           1.29 |           1.29 |             0.493 |           0.05 |           2:45:00 |     1     0     0   100 |
# |   ZEC/USDT |         1 |           0.96 |           0.96 |             0.369 |           0.04 |           2:45:00 |     1     0     0   100 |
# |   SOL/USDT |         1 |           0.43 |           0.43 |             0.165 |           0.02 |           3:00:00 |     1     0     0   100 |
# |   ENJ/USDT |         1 |          -0.32 |          -0.32 |            -0.124 |          -0.01 |           1:45:00 |     0     0     1     0 |
# |   ETH/USDT |         1 |          -0.63 |          -0.63 |            -0.240 |          -0.02 |           3:00:00 |     0     0     1     0 |
# |   ADA/USDT |         1 |          -0.72 |          -0.72 |            -0.276 |          -0.03 |           2:45:00 |     0     0     1     0 |
# |   TRX/USDT |         1 |          -0.77 |          -0.77 |            -0.295 |          -0.03 |           3:00:00 |     0     0     1     0 |
# |   XLM/USDT |         1 |          -0.78 |          -0.78 |            -0.299 |          -0.03 |           2:15:00 |     0     0     1     0 |
# |   AKT/USDT |         1 |          -0.88 |          -0.88 |            -0.338 |          -0.03 |           0:15:00 |     0     0     1     0 |
# |  HBAR/USDT |         1 |          -0.91 |          -0.91 |            -0.350 |          -0.04 |           1:00:00 |     0     0     1     0 |
# |   UNI/USDT |         1 |          -1.04 |          -1.04 |            -0.399 |          -0.04 |           1:30:00 |     0     0     1     0 |
# |   FIL/USDT |         1 |          -1.12 |          -1.12 |            -0.429 |          -0.04 |           2:30:00 |     0     0     1     0 |
# |   BTC/USDT |         1 |          -1.22 |          -1.22 |            -0.467 |          -0.05 |           3:00:00 |     0     0     1     0 |
# |   VET/USDT |         1 |          -1.53 |          -1.53 |            -0.586 |          -0.06 |           2:45:00 |     0     0     1     0 |
# |  LINK/USDT |         1 |          -1.78 |          -1.78 |            -0.679 |          -0.07 |          12:30:00 |     0     0     1     0 |
# | THETA/USDT |         1 |          -1.97 |          -1.97 |            -0.751 |          -0.08 |   1 day, 10:45:00 |     0     0     1     0 |
# |   DOT/USDT |         1 |          -2.28 |          -2.28 |            -0.875 |          -0.09 |           3:00:00 |     0     0     1     0 |
# |   XTZ/USDT |         1 |          -2.75 |          -2.75 |            -0.998 |          -0.10 | 11 days, 11:45:00 |     0     0     1     0 |
# |  OSMO/USDT |         1 |          -2.79 |          -2.79 |            -1.066 |          -0.11 |           7:30:00 |     0     0     1     0 |
# |  EGLD/USDT |         1 |          -2.86 |          -2.86 |            -1.089 |          -0.11 |    1 day, 5:15:00 |     0     0     1     0 |
# | MATIC/USDT |         1 |          -3.43 |          -3.43 |            -1.301 |          -0.13 |   1 day, 11:45:00 |     0     0     1     0 |
# |  SCRT/USDT |         1 |          -3.76 |          -3.76 |            -1.439 |          -0.14 |           7:30:00 |     0     0     1     0 |
# |  NEAR/USDT |         1 |          -5.01 |          -5.01 |            -1.819 |          -0.18 |  11 days, 7:00:00 |     0     0     1     0 |
# |   GMX/USDT |         1 |          -4.90 |          -4.90 |            -1.867 |          -0.19 |    1 day, 9:30:00 |     0     0     1     0 |
# |   ETC/USDT |         1 |          -5.50 |          -5.50 |            -1.917 |          -0.19 | 14 days, 19:15:00 |     0     0     1     0 |
# |  ALGO/USDT |         1 |          -5.38 |          -5.38 |            -1.955 |          -0.20 |  11 days, 7:00:00 |     0     0     1     0 |
# |  DOGE/USDT |         1 |          -5.60 |          -5.60 |            -2.030 |          -0.20 | 11 days, 12:15:00 |     0     0     1     0 |
# |  ATOM/USDT |         1 |          -6.21 |          -6.21 |            -2.257 |          -0.23 |  11 days, 7:00:00 |     0     0     1     0 |
# |   INJ/USDT |         1 |          -6.70 |          -6.70 |            -2.553 |          -0.26 |          23:00:00 |     0     0     1     0 |
# |  DYDX/USDT |         1 |          -7.60 |          -7.60 |            -2.864 |          -0.29 |  2 days, 10:15:00 |     0     0     1     0 |
# |   XRP/USDT |         1 |          -9.25 |          -9.25 |            -2.873 |          -0.29 | 26 days, 20:00:00 |     0     0     1     0 |
# |   QNT/USDT |         1 |         -10.11 |         -10.11 |            -3.259 |          -0.33 | 22 days, 20:00:00 |     0     0     1     0 |
# |   XDC/USDT |         1 |          -9.66 |          -9.66 |            -3.318 |          -0.33 | 15 days, 21:00:00 |     0     0     1     0 |
# |   GRT/USDT |         1 |          -9.64 |          -9.64 |            -3.492 |          -0.35 | 11 days, 11:45:00 |     0     0     1     0 |
# |   LDO/USDT |         1 |          -9.26 |          -9.26 |            -3.531 |          -0.35 |          21:45:00 |     0     0     1     0 |
# |  AVAX/USDT |         1 |         -10.56 |         -10.56 |            -3.582 |          -0.36 |  17 days, 3:00:00 |     0     0     1     0 |
# | OCEAN/USDT |         1 |         -12.48 |         -12.48 |            -4.473 |          -0.45 |  12 days, 0:15:00 |     0     0     1     0 |
# |  KAVA/USDT |         1 |         -12.55 |         -12.55 |            -4.562 |          -0.46 |  11 days, 8:15:00 |     0     0     1     0 |
# |   IMX/USDT |         1 |         -12.23 |         -12.23 |            -4.577 |          -0.46 |  2 days, 21:30:00 |     0     0     1     0 |
# |  ROSE/USDT |         1 |         -12.50 |         -12.50 |            -4.620 |          -0.46 |  3 days, 20:15:00 |     0     0     1     0 |
# |  CSPR/USDT |         1 |         -13.71 |         -13.71 |            -4.942 |          -0.49 | 11 days, 21:15:00 |     0     0     1     0 |
# |    OP/USDT |         1 |         -14.52 |         -14.52 |            -5.255 |          -0.53 | 11 days, 12:45:00 |     0     0     1     0 |
# |  RNDR/USDT |         1 |         -14.20 |         -14.20 |            -5.309 |          -0.53 |   3 days, 5:00:00 |     0     0     1     0 |
# |  AGIX/USDT |         1 |         -17.17 |         -17.17 |            -6.346 |          -0.63 |  3 days, 17:30:00 |     0     0     1     0 |
# |  ORAI/USDT |         1 |         -22.70 |         -22.70 |            -8.391 |          -0.84 |  3 days, 22:00:00 |     0     0     1     0 |
# |      TOTAL |        45 |          -5.92 |        -266.29 |           -96.766 |          -9.68 |   5 days, 9:25:00 |     3     0    42   6.7 |
# ================== SUMMARY METRICS ==================
# | Metric                      | Value               |
# |-----------------------------+---------------------|
# | Backtesting from            | 2023-01-01 00:00:00 |
# | Backtesting to              | 2023-02-20 00:00:00 |
# | Max open trades             | 45                  |
# |                             |                     |
# | Total/Daily Avg Trades      | 1295 / 25.9         |
# | Starting balance            | 1000 USDT           |
# | Final balance               | 1643.464 USDT       |
# | Absolute profit             | 643.464 USDT        |
# | Total profit %              | 64.35%              |
# | CAGR %                      | 3658.80%            |
# | Profit factor               | 5.88                |
# | Trades per day              | 25.9                |
# | Avg. daily profit %         | 1.29%               |
# | Avg. stake amount           | 29.399 USDT         |
# | Total trade volume          | 38071.743 USDT      |
# |                             |                     |
# | Best Pair                   | AGIX/USDT 179.28%   |
# | Worst Pair                  | XDC/USDT -0.06%     |
# | Best trade                  | AKT/USDT 19.18%     |
# | Worst trade                 | ORAI/USDT -31.80%   |
# | Best day                    | 40.159 USDT         |
# | Worst day                   | -96.766 USDT        |
# | Days win/draw/lose          | 48 / 0 / 3          |
# | Avg. Duration Winners       | 12:32:00            |
# | Avg. Duration Loser         | 5 days, 5:33:00     |
# | Rejected Entry signals      | 237                 |
# | Entry/Exit Timeouts         | 0 / 0               |
# |                             |                     |
# | Min balance                 | 1000.11 USDT        |
# | Max balance                 | 1740.23 USDT        |
# | Max % of account underwater | 5.56%               |
# | Absolute Drawdown (Account) | 5.56%               |
# | Absolute Drawdown           | 96.766 USDT         |
# | Drawdown high               | 740.23 USDT         |
# | Drawdown low                | 643.464 USDT        |
# | Drawdown Start              | 2023-02-19 20:45:00 |
# | Drawdown End                | 2023-02-20 00:00:00 |
# | Market change               | 112.61%             |
# =====================================================
