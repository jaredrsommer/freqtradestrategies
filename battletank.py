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
                                IStrategy, IntParameter, RealParameter, merge_informative_pair, informative)
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class thetank2(IStrategy):


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
    max_entry_position_adjustment = 2
    max_dca_multiplier = 2
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
    max_epa = CategoricalParameter([0, 1, 2], default=2, space="buy", optimize=True)

    # protections
    cooldown_lookback = IntParameter(24, 48, default=46, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # indicators
    wavelength = IntParameter(low=7, high=10, default=8, space='buy', optimize=True)

    # Filter Wave
    esa_length = IntParameter(3, 20, default=10, space="buy", optimize=True)
    d_length = IntParameter(3, 20, default=10, space="buy", optimize=True)
    ci_mult = DecimalParameter(0.001, 0.050 , default=0.015, space="buy", optimize=True)
    wt1_length = IntParameter(6, 50, default=21, space="buy", optimize=True)
    wt2_length = IntParameter(3, 15, default=4, space="sell", optimize=True)

    # trading
    buy_rsi = IntParameter(low=20, high=30, default=25, space='buy', optimize=True, load=True)
    buy_rsi_bear = IntParameter(low=40, high=55, default=45, space='buy', optimize=True, load=True)
    buy_rsi_bull = IntParameter(low=55, high=75, default=65, space='buy', optimize=True, load=True)
    buy_wt_bear = IntParameter(low=40, high=55, default=45, space='buy', optimize=True, load=True)
    buy_wt_bull = IntParameter(low=55, high=75, default=65, space='buy', optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=80, default=55, space='sell', optimize=True, load=True)

    # dca level optimization
    dca1 = DecimalParameter(low=0.01, high=0.08, decimals=2, default=0.05, space='buy', optimize=True, load=True)
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
    tsl_target1 = DecimalParameter(low=0.04, high=0.06, default=0.06, space='sell', optimize=True, load=True)
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
    #     # if self.max_epa.value == 0:
    #     #     self.max_dca_multiplier = 1
    #     # elif self.max_epa.value == 1:
    #     #     self.max_dca_multiplier = 2
    #     # elif self.max_epa.value == 2:
    #     #     self.max_dca_multiplier = 3
    #     # else:
    #     #     self.max_dca_multiplier = 4

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

    # add 2h Support and resistance
    def informative_pairs(self):

        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1h') for pair in pairs]
        
        return informative_pairs

    @informative('1h')
    ### INDICATORS ###
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe

        inf_tf = '1h'
        # Get the informative pair
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        # Get the 14 day rsi
        informative['rsi'] = ta.RSI(informative, timeperiod=14)

        informative['200_SMA'] = ta.SMA(informative["close"], timeperiod = 200)

        # Parabolic SAR
        informative['sar'] = ta.SAR(informative)

        # WaveTrend using OHLC4 or HA close - 3/21
        ap = (0.25 * (informative['high'] + informative['low'] + informative["close"] + informative["open"]))
        
        informative['esa'] = ta.EMA(ap, timeperiod = 10)
        informative['d'] = ta.EMA(abs(ap - informative['esa']), timeperiod = 10)
        informative['wave_ci'] = (ap-informative['esa']) / (0.015 * informative['d'])
        informative['wave_t1'] = ta.EMA(informative['wave_ci'], timeperiod = 21)  
        informative['wave_t2'] = ta.SMA(informative['wave_t1'], timeperiod = 4)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        ### NORMAL TIMEFRAME INDICATORS ###
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['rsi_ma'] = ta.SMA(dataframe['rsi'], timeperiod=10)

        # SMA
        for buy_ma in self.buy_ma_length.range:
            dataframe[f'buy_ma{buy_ma}'] = ta.SMA(dataframe["close"], timeperiod = buy_ma)
        for sell_ma in self.sell_ma_length.range:
            dataframe[f'sell_ma{sell_ma}'] = ta.SMA(dataframe["close"], timeperiod = sell_ma)

        # Parabolic SAR
        dataframe['sar'] = ta.SAR(dataframe)

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
        #     self.max_epa.value = 0
        # elif (dataframe['30_SMA'].iloc[-1] < dataframe['200_SMA'].iloc[-1] 
        #     and dataframe['30_SMA'].iloc[-1] > dataframe['30_SMA'].iloc[-2].all()):
        #     self.max_epa.value = 0
        # else:
        #     self.max_epa.value = 1
        # # print(f"Trading Pair: {dataframe.name}")
        # print(f"self.max_epa.value: {self.max_epa.value}")
        # print(f"self.max_dca_multiplier: {self.max_dca_multiplier}")
        # print(f"Trading pair: {ticker['symbol']}, max_epa: {self.max_epa.value}, dca_multiplier: {self.max_dca_multiplier}")
 
        return dataframe

    ### ENTRY CONDITIONS ###
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        # df.loc[
        #     (
        #         # Signal: RSI crosses above 30
        #         (df['rsi'] >  self.buy_rsi.value) &
        #         (df['rsi'] < 60) &
        #         (df['rsi'] > df['rsi_ma'])&
        #         (df['wave_t1'] > df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
        #         (qtpylib.crossed_above(df['wave_t1'], df['wave_t2'])) &
        #         (df['volume'] > 0)   # Make sure Volume is not 0
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'WT/RSI')

        # df.loc[
        #     (
        #         # Signal: RSI crosses above 30
        #         (df['rsi'] >  self.buy_rsi.value) &
        #         (df['rsi'] > df['rsi_ma'])&
        #         (df['tf'] < df['close']) &
        #         (df['tf'] < df['high'].shift(1)) &
        #         (df['wave_t1'] > df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
        #         (df['volume'] > 0)   # Make sure Volume is not 0
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'WT/RSI/ATR-RSL')

        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['squeeze_ON'] == 1)&
                # (df['close'] > df[f'buy_ma{self.buy_ma_length.value}']) &
                # (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                (df['200_SMA_1h'] < df['200_SMA_1h'].shift(1)) &
                # (df['rsi'] < self.buy_rsi_bear.value) &
                (df['wave_t1_1h'] < self.buy_wt_bear.value)&
                (df['wave_t1_1h'] > df['wave_t1_1h'].shift(1)) &  # Guard: Wave 1 is raising
                (qtpylib.crossed_above(df['wave_t1_1h'], df['wave_t2_1h'])) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'ttm_Squeeze/WT - bear')

        df.loc[
            (
                (df['squeeze_ON'] == 1)&
                # (df['close'] > df[f'buy_ma{self.buy_ma_length.value}']) &
                # (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                # (df['rsi'] < self.buy_rsi_bear.value) &
                (df['200_SMA_1h'] < df['200_SMA_1h'].shift(1)) &
                (df['wave_t1_1h'] < self.buy_wt_bear.value)&
                (df['wave_t1_1h'] > df['wave_t1_1h'].shift(1)) &  # Guard: Wave 1 is raising
                (df['wave_t1_1h'].shift(2) > df['wave_t1_1h'].shift(1)) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'ttm_Squeeze/WTT - bear')

        df.loc[
            (
                # (df['close'] > df[f'buy_ma{self.buy_ma_length.value}']) &
                # (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                # (df['rsi'] < self.buy_rsi_bear.value) &
                (df['200_SMA_1h'] < df['200_SMA_1h'].shift(1)) &
                (df['wave_t1_1h'] < self.buy_wt_bear.value)&
                (df['wave_t1_1h'] > df['wave_t1_1h'].shift(1)) &  # Guard: Wave 1 is raising
                (df['wave_t1_1h'].shift(2) > df['wave_t1_1h'].shift(1)) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'WT transition - bear')

        df.loc[
            (
                (df['squeeze_ON'] == 1)&
                # (df['close'] > df[f'buy_ma{self.buy_ma_length.value}']) &
                # (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                # (df['rsi'] < self.buy_rsi_bull.value) &
                (df['200_SMA_1h'] > df['200_SMA_1h'].shift(1)) &
                (df['wave_t1_1h'] > df['wave_t1_1h'].shift(1)) &  # Guard: Wave 1 is raising
                (qtpylib.crossed_above(df['wave_t1_1h'], df['wave_t2_1h'])) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'ttm_Squeeze/WT - bull')

        df.loc[
            (
                (df['squeeze_ON'] == 1)&
                # (df['close'] > df[f'buy_ma{self.buy_ma_length.value}']) &
                # (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                # (df['rsi'] < self.buy_rsi_bull.value) &
                (df['200_SMA_1h'] > df['200_SMA_1h'].shift(1)) &
                (df['wave_t1_1h'] > df['wave_t1_1h'].shift(1)) & 
                (df['wave_t1_1h'] < self.buy_wt_bull.value) & 
                (df['wave_t1_1h'].shift(2) > df['wave_t1_1h'].shift(1)) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'ttm_Squeeze/WTT - bull')

        df.loc[
            (
                # (df['close'] > df[f'buy_ma{self.buy_ma_length.value}']) &
                # (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                # (df['rsi'] < self.buy_rsi_bull.value) &
                (df['200_SMA_1h'] > df['200_SMA_1h'].shift(1)) &
                (df['wave_t1_1h'] > df['wave_t1_1h'].shift(1)) &  # Guard: Wave 1 is raising
                (df['wave_t1_1h'] < self.buy_wt_bull.value) & 
                (df['wave_t1_1h'].shift(2) > df['wave_t1_1h'].shift(1)) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'WT transition - bull')


        df.loc[
            (
                # (df['close'] > df[f'buy_ma{self.buy_ma_length.value}']) &
                # (df['rsi'] >  self.buy_rsi.value) &
                # (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                # (df['rsi'] < self.buy_rsi_bear.value) &
                (df['200_SMA_1h'] < df['200_SMA_1h'].shift(1)) &
                (qtpylib.crossed_above(df['rsi'], df['rsi_ma'])) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'RSI-XO bear')

        df.loc[
            (
                # (df['close'] > df[f'buy_ma{self.buy_ma_length.value}']) &
                # # Signal: RSI crosses above 30
                # (df['rsi'] >  self.buy_rsi.value) &
                # (df['rsi_ma'] > df['rsi_ma'].shift(1)) &
                # (df['rsi'] < self.buy_rsi_bull.value) &
                (df['200_SMA_1h'] > df['200_SMA_1h'].shift(1)) &
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
        #         (df['rsi'] > self.sell_rsi.value) &
        #         (df['tf'] > df['close']) &
        #         (df['wave_t1'] < df['wave_t1'].shift(1)) &  # Guard: Wave 1 is falling
        #         (qtpylib.crossed_above(df['wave_t2'], df['wave_t1'])) &
        #         (df['volume'] > 0)   # Make sure Volume is not 0
        #     ),
        #     ['exit_long', 'exit_tag']] = (1, 'WT/RSI')

        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['squeeze_ON'] == 1)&
                # (df['close'] < df[f'sell_ma{self.sell_ma_length.value}']) &
                (df['wave_t1_1h'] < df['wave_t1_1h'].shift(1)) &  # Guard: Wave 1 is raising
                (df['wave_t1_1h'].shift(2) < df['wave_t1_1h'].shift(1)) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'ttm_Squeeze/WT transition')

        df.loc[
            (
                # Signal: RSI crosses above 30
                # (df['close'] < df[f'sell_ma{self.sell_ma_length.value}']) &
                # (df['rsi'] < df['rsi_ma']) &
                # (df['tf'] > df['close']) &
                (df['wave_t1_1h'] < df['wave_t1_1h'].shift(1)) &  # Guard: Wave 1 is raising
                (qtpylib.crossed_above(df['wave_t2_1h'], df['wave_t1_1h'])) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'WT/RSI-XO')

        df.loc[
            (
                # (df['close'] < df[f'sell_ma{self.sell_ma_length.value}']) &
                (df['rsi'] > self.sell_rsi.value) &
                # (df['tf'] > df['close']) &
                (qtpylib.crossed_above(df['rsi_ma'], df['rsi'])) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'RSI-XO')

        # df.loc[
        #     (
        #         # Signal: RSI crosses above 30
        #         (df['rsi'] >  self.sell_rsi.value) &
        #         (df['rsi'] < df['rsi_ma'])&
        #         (df['wave_t1'] < df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
        #         (df['wave_t1'].shift(1) > df['wave_t1'].shift(2)) & 
        #         # (abs(df['wave_t2'] - df['wave_t1']) > 2.5) &
        #         (df['volume'] > 0)   # Make sure Volume is not 0
        #     ),
        #     ['exit_long', 'exit_tag']] = (1, 'WT/RSI TURN DOWN')

        return df

        # 2023-02-10 10:20:38,359 - freqtrade.strategy.hyper - INFO - Loading parameters from file /home/core/freqtrade/user_data/strategies/thetank2.json
        # 2023-02-10 10:20:38,360 - freqtrade.resolvers.strategy_resolver - INFO - Override strategy 'process_only_new_candles' with value in config file: True.
        # 2023-02-10 10:20:38,360 - freqtrade.resolvers.strategy_resolver - INFO - Override strategy 'stake_currency' with value in config file: USDT.
        # 2023-02-10 10:20:38,360 - freqtrade.resolvers.strategy_resolver - INFO - Override strategy 'stake_amount' with value in config file: unlimited.
        # 2023-02-10 10:20:38,360 - freqtrade.resolvers.strategy_resolver - INFO - Override strategy 'unfilledtimeout' with value in config file: {'unit': 'minutes', 'entry': 30, 'exit': 30, 'exit_timeout_count': 0}.
        # 2023-02-10 10:20:38,360 - freqtrade.resolvers.strategy_resolver - INFO - Override strategy 'max_open_trades' with value in config file: 12.
        # 2023-02-10 10:20:38,360 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using minimal_roi: {'0': 0.58, '410': 0.165, '972': 0.034, '1475': 0}
        # 2023-02-10 10:20:38,360 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using timeframe: 1h
        # 2023-02-10 10:20:38,360 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using stoploss: -0.17
        # 2023-02-10 10:20:38,360 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_stop: True
        # 2023-02-10 10:20:38,360 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_stop_positive: 0.257
        # 2023-02-10 10:20:38,360 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_stop_positive_offset: 0.333
        # 2023-02-10 10:20:38,360 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_only_offset_is_reached: False
        # 2023-02-10 10:20:38,360 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using use_custom_stoploss: True
        # 2023-02-10 10:20:38,360 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using process_only_new_candles: True
        # 2023-02-10 10:20:38,361 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using order_types: {'entry': 'limit', 'exit': 'limit', 'stoploss': 'limit', 'stoploss_on_exchange': False, 'stoploss_on_exchange_interval': 60}
        # 2023-02-10 10:20:38,361 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using order_time_in_force: {'entry': 'GTC', 'exit': 'GTC'}
        # 2023-02-10 10:20:38,361 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using stake_currency: USDT
        # 2023-02-10 10:20:38,361 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using stake_amount: unlimited
        # 2023-02-10 10:20:38,361 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using protections: [{'method': 'CooldownPeriod', 'stop_duration_candles': 46}, {'method': 'StoplossGuard', 'lookback_period_candles': 72, 'trade_limit': 1, 'stop_duration_candles': 5, 'only_per_pair': False}]
        # 2023-02-10 10:20:38,361 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using startup_candle_count: 30
        # 2023-02-10 10:20:38,361 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using unfilledtimeout: {'unit': 'minutes', 'entry': 30, 'exit': 30, 'exit_timeout_count': 0}
        # 2023-02-10 10:20:38,361 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using use_exit_signal: True
        # 2023-02-10 10:20:38,361 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using exit_profit_only: True
        # 2023-02-10 10:20:38,361 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using ignore_roi_if_entry_signal: True
        # 2023-02-10 10:20:38,361 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using exit_profit_offset: 0.0
        # 2023-02-10 10:20:38,361 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using disable_dataframe_checks: False
        # 2023-02-10 10:20:38,361 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using ignore_buying_expired_candle_after: 0
        # 2023-02-10 10:20:38,361 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using position_adjustment_enable: False
        # 2023-02-10 10:20:38,361 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using max_entry_position_adjustment: 0
        # 2023-02-10 10:20:38,361 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using max_open_trades: 12
        # 2023-02-10 10:20:38,361 - freqtrade.configuration.config_validation - INFO - Validating configuration ...
        # 2023-02-10 10:20:38,367 - freqtrade.resolvers.iresolver - INFO - Using resolved pairlist StaticPairList from '/home/core/freqtrade/freqtrade/plugins/pairlist/StaticPairList.py'...
        # 2023-02-10 10:20:38,410 - freqtrade.data.history.history_utils - INFO - Using indicator startup period: 30 ...
        # 2023-02-10 10:20:38,616 - freqtrade.data.history.idatahandler - WARNING - OSMO/USDT, spot, 1h, data starts at 2023-01-13 10:00:00
        # 2023-02-10 10:20:39,260 - freqtrade.optimize.backtesting - INFO - Loading data from 2022-12-30 18:00:00 up to 2023-02-05 00:00:00 (36 days).
        # 2023-02-10 10:20:39,260 - freqtrade.optimize.backtesting - INFO - Dataload complete. Calculating indicators
        # 2023-02-10 10:20:39,264 - freqtrade.optimize.backtesting - INFO - Running backtesting for Strategy thetank2
        # 2023-02-10 10:20:39,264 - freqtrade.strategy.hyper - INFO - Strategy Parameter: buy_rsi = 27
        # 2023-02-10 10:20:39,264 - freqtrade.strategy.hyper - INFO - Strategy Parameter: buy_rsi_bear = 45
        # 2023-02-10 10:20:39,264 - freqtrade.strategy.hyper - INFO - Strategy Parameter: buy_rsi_bull = 61
        # 2023-02-10 10:20:39,264 - freqtrade.strategy.hyper - INFO - Strategy Parameter: max_epa = 1
        # 2023-02-10 10:20:39,264 - freqtrade.strategy.hyper - INFO - Strategy Parameter: wavelength = 9
        # 2023-02-10 10:20:39,264 - freqtrade.strategy.hyper - INFO - Strategy Parameter: sell_rsi = 72
        # 2023-02-10 10:20:39,265 - freqtrade.strategy.hyper - INFO - Strategy Parameter: ts0 = 0.014
        # 2023-02-10 10:20:39,265 - freqtrade.strategy.hyper - INFO - Strategy Parameter: ts1 = 0.01
        # 2023-02-10 10:20:39,265 - freqtrade.strategy.hyper - INFO - Strategy Parameter: ts2 = 0.023
        # 2023-02-10 10:20:39,265 - freqtrade.strategy.hyper - INFO - Strategy Parameter: ts3 = 0.028
        # 2023-02-10 10:20:39,265 - freqtrade.strategy.hyper - INFO - Strategy Parameter: ts4 = 0.032
        # 2023-02-10 10:20:39,265 - freqtrade.strategy.hyper - INFO - Strategy Parameter: ts5 = 0.06
        # 2023-02-10 10:20:39,265 - freqtrade.strategy.hyper - INFO - Strategy Parameter: tsl_target0 = 0.049
        # 2023-02-10 10:20:39,265 - freqtrade.strategy.hyper - INFO - Strategy Parameter: tsl_target1 = 0.049
        # 2023-02-10 10:20:39,265 - freqtrade.strategy.hyper - INFO - Strategy Parameter: tsl_target2 = 0.097
        # 2023-02-10 10:20:39,265 - freqtrade.strategy.hyper - INFO - Strategy Parameter: tsl_target3 = 0.149
        # 2023-02-10 10:20:39,265 - freqtrade.strategy.hyper - INFO - Strategy Parameter: tsl_target4 = 0.213
        # 2023-02-10 10:20:39,265 - freqtrade.strategy.hyper - INFO - Strategy Parameter: tsl_target5 = 0.3
        # 2023-02-10 10:20:39,265 - freqtrade.strategy.hyper - INFO - Strategy Parameter: cooldown_lookback = 48
        # 2023-02-10 10:20:39,265 - freqtrade.strategy.hyper - INFO - Strategy Parameter: stop_duration = 92
        # 2023-02-10 10:20:39,265 - freqtrade.strategy.hyper - INFO - Strategy Parameter: use_stop_protection = True
        # 2023-02-10 10:20:40,166 - freqtrade.optimize.backtesting - INFO - Backtesting with data from 2023-01-01 00:00:00 up to 2023-02-05 00:00:00 (35 days).
        # 2023-02-10 10:20:46,263 - freqtrade.misc - INFO - dumping json to "/home/core/freqtrade/user_data/backtest_results/backtest-result-2023-02-10_10-20-46.meta.json"
        # 2023-02-10 10:20:46,264 - freqtrade.misc - INFO - dumping json to "/home/core/freqtrade/user_data/backtest_results/backtest-result-2023-02-10_10-20-46.json"
        # 2023-02-10 10:20:46,272 - freqtrade.misc - INFO - dumping json to "/home/core/freqtrade/user_data/backtest_results/.last_result.json"
        # Result for strategy thetank2
        # ============================================================= BACKTESTING REPORT =============================================================
        # |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
        # |------------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
        # |   AKT/USDT |        23 |           2.63 |          60.52 |           585.596 |           5.86 |         12:05:00 |    18     3     2  78.3 |
        # |  RNDR/USDT |        17 |           2.82 |          47.92 |           524.616 |           5.25 |         15:32:00 |    12     5     0   100 |
        # |  SCRT/USDT |        19 |           2.63 |          49.89 |           492.403 |           4.92 |         19:35:00 |    13     6     0   100 |
        # |  IOTA/USDT |        16 |           2.61 |          41.74 |           430.524 |           4.31 |         19:22:00 |    13     3     0   100 |
        # |   INJ/USDT |        10 |           3.70 |          36.96 |           416.199 |           4.16 |         13:30:00 |     8     1     1  80.0 |
        # |  OSMO/USDT |        12 |           1.88 |          22.59 |           263.427 |           2.63 |         20:25:00 |     9     1     2  75.0 |
        # |  ATOM/USDT |        15 |           1.59 |          23.92 |           248.307 |           2.48 |   1 day, 9:00:00 |     7     7     1  46.7 |
        # |  AGIX/USDT |        17 |           1.71 |          29.07 |           245.429 |           2.45 |          8:35:00 |    10     1     6  58.8 |
        # |  LINK/USDT |        11 |           1.97 |          21.64 |           229.213 |           2.29 |  1 day, 13:16:00 |     8     3     0   100 |
        # |  CSPR/USDT |        17 |           1.25 |          21.28 |           219.967 |           2.20 |         20:25:00 |    10     6     1  58.8 |
        # |   ADA/USDT |        14 |           1.40 |          19.57 |           191.747 |           1.92 |   1 day, 9:26:00 |     7     6     1  50.0 |
        # |  KAVA/USDT |        13 |           1.57 |          20.47 |           182.562 |           1.83 |   1 day, 9:28:00 |     8     4     1  61.5 |
        # |  AVAX/USDT |        10 |           1.67 |          16.70 |           169.661 |           1.70 |   1 day, 1:12:00 |     6     3     1  60.0 |
        # |   GRT/USDT |         9 |           1.53 |          13.80 |           165.922 |           1.66 |   1 day, 1:47:00 |     5     4     0   100 |
        # | MATIC/USDT |         6 |           2.61 |          15.66 |           157.543 |           1.58 |         22:30:00 |     4     2     0   100 |
        # |   ETH/USDT |        10 |           1.40 |          13.99 |           139.876 |           1.40 |         22:12:00 |     5     5     0   100 |
        # |   ETC/USDT |         6 |           2.40 |          14.39 |           139.319 |           1.39 |  1 day, 18:40:00 |     3     3     0   100 |
        # |   XRP/USDT |         9 |           0.99 |           8.87 |            86.141 |           0.86 | 2 days, 15:53:00 |     4     4     1  44.4 |
        # |   QNT/USDT |        13 |           0.63 |           8.18 |            74.738 |           0.75 |  1 day, 17:46:00 |     7     5     1  53.8 |
        # |   DOT/USDT |         8 |           0.62 |           4.98 |            57.442 |           0.57 |  1 day, 19:52:00 |     3     4     1  37.5 |
        # |   XDC/USDT |        16 |           0.37 |           6.00 |            50.751 |           0.51 |   1 day, 4:00:00 |    11     4     1  68.8 |
        # |   XLM/USDT |         8 |           0.67 |           5.35 |            46.710 |           0.47 |  2 days, 6:30:00 |     4     3     1  50.0 |
        # |   BTC/USDT |        10 |           0.39 |           3.89 |            34.865 |           0.35 |         22:12:00 |     7     2     1  70.0 |
        # |  HBAR/USDT |        10 |           0.44 |           4.38 |            22.433 |           0.22 |         23:06:00 |     7     2     1  70.0 |
        # |  NEAR/USDT |         5 |           0.08 |           0.40 |             4.848 |           0.05 |   1 day, 8:12:00 |     1     3     1  20.0 |
        # |  ALGO/USDT |         8 |           0.13 |           1.02 |             2.492 |           0.02 |   1 day, 2:45:00 |     3     4     1  37.5 |
        # |  ORAI/USDT |        10 |          -0.59 |          -5.86 |           -27.010 |          -0.27 |         20:48:00 |     5     3     2  50.0 |
        # |      TOTAL |       322 |           1.58 |         507.33 |          5155.722 |          51.56 |   1 day, 2:04:00 |   198    97    27  61.5 |
        # ================================================================ ENTER TAG STATS ================================================================
        # |            TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |
        # |----------------+-----------+----------------+----------------+-------------------+----------------+-----------------+-------------------------|
        # |  WT transition |       211 |           1.55 |         327.28 |          3290.095 |          32.90 |  1 day, 2:40:00 |   122    77    12  57.8 |
        # |    RSI-XO bull |        88 |           1.84 |         161.58 |          1683.302 |          16.83 |        18:52:00 |    63    14    11  71.6 |
        # | ttm_Squeeze/WT |        19 |           0.59 |          11.12 |            95.446 |           0.95 | 2 days, 7:00:00 |     9     6     4  47.4 |
        # |    RSI-XO bear |         4 |           1.84 |           7.34 |            86.879 |           0.87 |        14:30:00 |     4     0     0   100 |
        # |          TOTAL |       322 |           1.58 |         507.33 |          5155.722 |          51.56 |  1 day, 2:04:00 |   198    97    27  61.5 |
        # =========================================================== EXIT REASON STATS ===========================================================
        # |               Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
        # |---------------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
        # |                       roi |     138 |     41    97     0   100 |           0.82 |         112.59 |          1117.04  |           9.38 |
        # |        trailing_stop_loss |      93 |     79     0    14  84.9 |           4.32 |         402.06 |          4208.7   |          33.5  |
        # |                 WT/RSI-XO |      59 |     59     0     0   100 |           0.81 |          47.67 |           472.152 |           3.97 |
        # | ttm_Squeeze/WT transition |      17 |     17     0     0   100 |           0.46 |           7.81 |            82.249 |           0.65 |
        # |                force_exit |      11 |      0     0    11     0 |          -2.84 |         -31.26 |          -371.783 |          -2.6  |
        # |                    RSI-XO |       2 |      2     0     0   100 |           1.38 |           2.76 |            24.487 |           0.23 |
        # |                 stop_loss |       2 |      0     0     2     0 |         -17.15 |         -34.3  |          -377.123 |          -2.86 |
        # ========================================================== LEFT OPEN TRADES REPORT ===========================================================
        # |      Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |      Avg Duration |   Win  Draw  Loss  Win% |
        # |-----------+-----------+----------------+----------------+-------------------+----------------+-------------------+-------------------------|
        # |  BTC/USDT |         1 |          -1.37 |          -1.37 |           -17.057 |          -0.17 |    1 day, 8:00:00 |     0     0     1     0 |
        # |  DOT/USDT |         1 |          -1.41 |          -1.41 |           -17.740 |          -0.18 |           6:00:00 |     0     0     1     0 |
        # | ALGO/USDT |         1 |          -1.53 |          -1.53 |           -19.258 |          -0.19 |          12:00:00 |     0     0     1     0 |
        # | AVAX/USDT |         1 |          -1.60 |          -1.60 |           -20.385 |          -0.20 |           3:00:00 |     0     0     1     0 |
        # |  ADA/USDT |         1 |          -1.78 |          -1.78 |           -21.591 |          -0.22 |   2 days, 6:00:00 |     0     0     1     0 |
        # | ATOM/USDT |         1 |          -1.74 |          -1.74 |           -22.211 |          -0.22 |           4:00:00 |     0     0     1     0 |
        # |  XLM/USDT |         1 |          -2.44 |          -2.44 |           -27.188 |          -0.27 | 11 days, 20:00:00 |     0     0     1     0 |
        # |  XDC/USDT |         1 |          -2.18 |          -2.18 |           -27.454 |          -0.27 |           7:00:00 |     0     0     1     0 |
        # |  XRP/USDT |         1 |          -3.35 |          -3.35 |           -37.576 |          -0.38 |  11 days, 4:00:00 |     0     0     1     0 |
        # | OSMO/USDT |         1 |          -4.13 |          -4.13 |           -51.562 |          -0.52 |          15:00:00 |     0     0     1     0 |
        # |  QNT/USDT |         1 |          -9.73 |          -9.73 |          -109.762 |          -1.10 |  9 days, 23:00:00 |     0     0     1     0 |
        # |     TOTAL |        11 |          -2.84 |         -31.26 |          -371.783 |          -3.72 |  3 days, 12:00:00 |     0     0    11     0 |
        # ================== SUMMARY METRICS ==================
        # | Metric                      | Value               |
        # |-----------------------------+---------------------|
        # | Backtesting from            | 2023-01-01 00:00:00 |
        # | Backtesting to              | 2023-02-05 00:00:00 |
        # | Max open trades             | 12                  |
        # |                             |                     |
        # | Total/Daily Avg Trades      | 322 / 9.2           |
        # | Starting balance            | 10000 USDT          |
        # | Final balance               | 15155.722 USDT      |
        # | Absolute profit             | 5155.722 USDT       |
        # | Total profit %              | 51.56%              |
        # | CAGR %                      | 7541.11%            |
        # | Sortino                     | 40.74               |
        # | Sharpe                      | 61.72               |
        # | Calmar                      | 1175.38             |
        # | Profit factor               | 4.13                |
        # | Expectancy                  | -0.04               |
        # | Trades per day              | 9.2                 |
        # | Avg. daily profit %         | 1.47%               |
        # | Avg. stake amount           | 1027.451 USDT       |
        # | Total trade volume          | 330839.375 USDT     |
        # |                             |                     |
        # | Best Pair                   | AKT/USDT 60.52%     |
        # | Worst Pair                  | ORAI/USDT -5.86%    |
        # | Best trade                  | AKT/USDT 32.05%     |
        # | Worst trade                 | AKT/USDT -17.16%    |
        # | Best day                    | 689.52 USDT         |
        # | Worst day                   | -371.783 USDT       |
        # | Days win/draw/lose          | 30 / 0 / 6          |
        # | Avg. Duration Winners       | 11:39:00            |
        # | Avg. Duration Loser         | 1 day, 22:07:00     |
        # | Rejected Entry signals      | 5295                |
        # | Entry/Exit Timeouts         | 0 / 0               |
        # |                             |                     |
        # | Min balance                 | 10044.087 USDT      |
        # | Max balance                 | 15527.505 USDT      |
        # | Max % of account underwater | 2.39%               |
        # | Absolute Drawdown (Account) | 2.39%               |
        # | Absolute Drawdown           | 371.783 USDT        |
        # | Drawdown high               | 5527.505 USDT       |
        # | Drawdown low                | 5155.722 USDT       |
        # | Drawdown Start              | 2023-02-04 23:00:00 |
        # | Drawdown End                | 2023-02-05 00:00:00 |
        # | Market change               | 97.41%              |
        # =====================================================
