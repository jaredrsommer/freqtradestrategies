import logging
from functools import reduce
import datetime
import talib.abstract as ta
import pandas_ta as pta
import logging
import numpy as np
import pandas as pd
#from murrey_math import calculate_murrey_math_levels
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

class TankAi(IStrategy):
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
    startup_candle_count: int = 200
    stoploss = -0.99
    timeframe = '30m'

    minimal_roi = {
        "0": 0.50,
        "60": 0.45,
        "120":0.40
    }

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

    # protections
    cooldown_lookback = IntParameter(24, 48, default=12, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    #trailing stop loss optimiziation
    tsl_target5 = DecimalParameter(low=0.3, high=0.4, default=0.3, decimals=2, space='sell', optimize=True, load=True)
    ts5 = DecimalParameter(low=0.04, high=0.06, default=0.05,decimals=2, space='sell', optimize=True, load=True)
    tsl_target4 = DecimalParameter(low=0.18, high=0.3, default=0.2, decimals=2, space='sell', optimize=True, load=True)
    ts4 = DecimalParameter(low=0.03, high=0.05, default=0.045, decimals=3, space='sell', optimize=True, load=True)
    tsl_target3 = DecimalParameter(low=0.12, high=0.18, default=0.15, decimals=2,space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, decimals=3,space='sell', optimize=True, load=True)
    # tsl_target2 = DecimalParameter(low=0.07, high=0.12, default=0.1, decimals=2,space='sell', optimize=True, load=True)
    # ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, decimals=2,space='sell', optimize=True, load=True)
    # tsl_target1 = DecimalParameter(low=0.04, high=0.07, default=0.06, decimals=3,space='sell', optimize=True, load=True)
    # ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, decimals=3,space='sell', optimize=True, load=True)
    # tsl_target0 = DecimalParameter(low=0.02, high=0.05, default=0.040, decimals=3 ,space='sell', optimize=True, load=True)
    # ts0 = DecimalParameter(low=0.008, high=0.015, default=0.012, decimals=3,space='sell', optimize=True, load=True)

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

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        trade_duration = (current_time - trade.open_date_utc).seconds / 60

        # Take Profit
        if current_profit > 0.15 and trade.nr_of_successful_exits == 0:
            # Take half of the profit at +15%
            return -(trade.stake_amount / 4)
        if current_profit > 0.20 and trade.nr_of_successful_exits == 1:
            # Take half of the profit at +15%
            return -(trade.stake_amount / 3)

        # Profit Based DCA    
        if current_profit > -0.0155 and trade.nr_of_successful_entries == 1:
            return None

        if current_profit > -0.045 and trade.nr_of_successful_entries == 2:
            return None

        if current_profit > -0.105 and trade.nr_of_successful_entries == 3:
            return None

        if current_profit > -0.15 and trade.nr_of_successful_entries == 4:
            return None

        if current_profit > -0.20 and trade.nr_of_successful_entries == 5:
            return None

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
            elif count_of_entries == 4:
                stake_amount = stake_amount * 1
            elif count_of_entries == 5:
                stake_amount = stake_amount * 1
            else:
                stake_amount = stake_amount

            return stake_amount
        except Exception as exception:
            return None

        return None


    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        trade_duration = (current_time - trade.open_date_utc).seconds / 60

        if current_candle['rsi'] < 80:

            for stop5 in self.tsl_target5.range:
                if (current_profit > stop5):
                    for stop5a in self.ts5.range:
                        self.dp.send_msg(f'*** {pair} *** Profit: {current_profit} - lvl5 {stop5}/{stop5a} activated')
                        return stop5a 
            for stop4 in self.tsl_target4.range:
                if (current_profit > stop4):
                    for stop4a in self.ts4.range:
                        self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl4 {stop4}/{stop4a} activated')
                        return stop4a 
            for stop3 in self.tsl_target3.range:
                if (current_profit > stop3):
                    for stop3a in self.ts3.range:
                        self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl3 {stop3}/{stop3a} activated')
                        return stop3a 
            # for stop2 in self.tsl_target2.range:
            #     if (current_profit > stop2):
            #         for stop2a in self.ts2.range:
            #             self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl2 {stop2}/{stop2a} activated')
            #             return stop2a 
            # if trade_duration < 600:
            #     print(f"{pair}: 2nd trailing stop engaged {trade_duration} elapsed")
            #     for stop1 in self.tsl_target1.range:
            #         if (current_profit > stop1):
            #             for stop1a in self.ts1.range:
            #                 self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl1 {stop1}/{stop1a} activated')
            #                 return stop1a 
            # if trade_duration < 360:
            #     print(f"{pair} 1st trailing stop engaged {trade_duration} elapsed")
            #     for stop0 in self.tsl_target0.range:
            #         if (current_profit > stop0):
            #             for stop0a in self.ts0.range:
            #                 self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl0 {stop0}/{stop0a} activated')
            #                 return stop0a 
        else:
            for stop0 in self.tsl_target0.range:
                if (current_profit > stop0):
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} SWINGING FOR THE MOON!!!')
                    return 0.99

        return self.stoploss


    def feature_engineering_expand_all(self, dataframe, period, **kwargs):
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-rocr-period"] = ta.ROCR(dataframe, timeperiod=period)
        dataframe["%-cmf-period"] = chaikin_mf(dataframe, periods=period)
        dataframe["%-chop-period"] = qtpylib.chopiness(dataframe, period)
        dataframe["%-linear-period"] = ta.LINEARREG_ANGLE(
            dataframe['close'], timeperiod=period)
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        dataframe["%-atr-periodp"] = dataframe["%-atr-period"] / \
            dataframe['close'] * 1000
        return dataframe


    def feature_engineering_expand_basic(self, dataframe, metadata, **kwargs):
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-obv"] = ta.OBV(dataframe)
        dataframe["dpo"] = pta.dpo(dataframe['close'], length=40, centered=False)
        dataframe["%-dpo"] = dataframe["dpo"]
        # Williams R%
        dataframe['%-willr14'] = pta.willr(dataframe['high'], dataframe['low'], dataframe['close'])

        # VWAP
        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe['%-vwap_upperband'] = vwap_high
        dataframe['%-vwap_middleband'] = vwap
        dataframe['%-vwap_lowerband'] = vwap_low
        dataframe['%-vwap_width'] = ((dataframe['%-vwap_upperband'] -
                                     dataframe['%-vwap_lowerband']) / dataframe['%-vwap_middleband']) * 100
        dataframe = dataframe.copy()
        dataframe['%-dist_to_vwap_upperband'] = get_distance(dataframe['close'], dataframe['%-vwap_upperband'])
        dataframe['%-dist_to_vwap_middleband'] = get_distance(dataframe['close'], dataframe['%-vwap_middleband'])
        dataframe['%-dist_to_vwap_lowerband'] = get_distance(dataframe['close'], dataframe['%-vwap_lowerband'])
        dataframe['%-tail'] = (dataframe['close'] - dataframe['low']).abs()
        dataframe['%-wick'] = (dataframe['high'] - dataframe['close']).abs()
        dataframe['%-rawclose'] = dataframe['close']
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_low"] = dataframe["low"]
        dataframe["%-raw_high"] = dataframe["high"]

        heikinashi = qtpylib.heikinashi(dataframe)

        dataframe['%-ha_open'] = heikinashi['open']
        dataframe['%-ha_close'] = heikinashi['close']
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['%-ha_high'] = heikinashi['high']
        dataframe['%-ha_low'] = heikinashi['low']
        dataframe['%-ha_closedelta'] = (heikinashi['close'] - heikinashi['close'].shift())
        dataframe['%-ha_tail'] = (heikinashi['close'] - heikinashi['low'])
        dataframe['%-ha_wick'] = (heikinashi['high'] - heikinashi['close'])

        dataframe['%-HLC3'] = (heikinashi['high'] + heikinashi['low'] + heikinashi['close'])/3

        murrey_math_levels = calculate_murrey_math_levels(dataframe)
        for level, value in murrey_math_levels.items():
            dataframe[level] = value
    
        dataframe["%-+3/8"] = dataframe["[+3/8]P"]
        dataframe["%-+2/8"] = dataframe["[+2/8]P"]
        dataframe["%-+1/8"] = dataframe["[+1/8]P"]
        dataframe["%-8/8"] = dataframe["[8/8]P"]
        dataframe["%-7/8"] = dataframe["[7/8]P"]
        dataframe["%-6/8"] = dataframe["[6/8]P"]
        dataframe["%-5/8"] = dataframe["[5/8]P"]
        dataframe["%-4/8"] = dataframe["[4/8]P"]
        dataframe["%-3/8"] = dataframe["[3/8]P"]
        dataframe["%-2/8"] = dataframe["[2/8]P"]
        dataframe["%-1/8"] = dataframe["[1/8]P"]
        dataframe["%-0/8"] = dataframe["[0/8]P"]
        dataframe["%--1/8"] = dataframe["[-1/8]P"]
        dataframe["%--2/8"] = dataframe["[-2/8]P"]
        dataframe["%--3/8"] = dataframe["[-3/8]P"]


        dataframe['ema_2'] = ta.EMA(dataframe, timeperiod=2)
        dataframe['%-distema2'] = get_distance(dataframe['ema_2'], dataframe['[+3/8]P'])
        dataframe['%-distema2'] = get_distance(dataframe['ema_2'], dataframe['[+2/8]P'])
        dataframe['%-distema2'] = get_distance(dataframe['ema_2'], dataframe['[+1/8]P'])
        dataframe['%-distema2'] = get_distance(dataframe['ema_2'], dataframe['[8/8]P'])
        dataframe['%-distema2'] = get_distance(dataframe['ema_2'], dataframe['[4/8]P'])
        dataframe['%-distema2'] = get_distance(dataframe['ema_2'], dataframe['[0/8]P'])
        dataframe['%-distema2'] = get_distance(dataframe['ema_2'], dataframe['[-1/8]P'])
        dataframe['%-distema2'] = get_distance(dataframe['ema_2'], dataframe['[-2/8]P'])
        dataframe['%-distema2'] = get_distance(dataframe['ema_2'], dataframe['[-3/8]P'])

        dataframe['%-entrythreshold4'] = (dataframe['%-tail'] - dataframe['[0/8]P'])
        dataframe["%-entrythreshold5"] = (dataframe['%-tail'] - dataframe['[-1/8]P'])
        dataframe["%-entrythreshold6"] = (dataframe['%-tail'] - dataframe['[-2/8]P'])
        dataframe["%-entrythreshold7"] = (dataframe['%-tail'] - dataframe['[-3/8]P'])

        dataframe["%-exitthreshold4"] = (dataframe['%-wick'] - dataframe['[8/8]P'])
        dataframe["%-exitthreshold5"] = (dataframe['%-wick'] - dataframe['[+1/8]P'])
        dataframe["%-exitthreshold6"] = (dataframe['%-wick'] - dataframe['[+2/8]P'])
        dataframe["%-exitthreshold7"] = (dataframe['%-wick'] - dataframe['[+3/8]P'])

        dataframe['mmlextreme_oscillator'] = 100 * ((dataframe['close'] - dataframe["[-3/8]P"]) / (dataframe["[+3/8]P"] - dataframe["[-3/8]P"]))
        dataframe['%-mmlextreme_oscillator'] = dataframe['mmlextreme_oscillator']
    

        # Calculate the percentage change between the high and open prices for each 5-minute candle
        dataframe['%-perc_change'] = (dataframe['high'] / dataframe['open'] - 1) * 100

        # Create a custom indicator that checks if any of the past 100 5-minute candles' high price is 3% or more above the open price
        dataframe['%-candle_1perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x >= 1, 1, 0).sum()).shift()
        dataframe['%-candle_2perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x >= 2, 1, 0).sum()).shift()
        dataframe['%-candle_3perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x >= 3, 1, 0).sum()).shift()
        dataframe['%-candle_5perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x >= 5, 1, 0).sum()).shift()

        dataframe['%-candle_-1perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x <= -1, -1, 0).sum()).shift()
        dataframe['%-candle_-2perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x <= -2, -1, 0).sum()).shift()
        dataframe['%-candle_-3perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x <= -3, -1, 0).sum()).shift()
        dataframe['%-candle_-5perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x <= -5, -1, 0).sum()).shift()

        # Calculate the percentage of the current candle's range where the close price is
        dataframe['%-close_percentage'] = (dataframe['close'] - dataframe['low']) / (dataframe['high'] - dataframe['low'])

        dataframe['%-body_size'] = abs(dataframe['open'] - dataframe['close'])
        dataframe['%-range_size'] = dataframe['high'] - dataframe['low']
        dataframe['%-body_range_ratio'] = dataframe['%-body_size'] / dataframe['%-range_size']

        dataframe['%-upper_wick_size'] = dataframe['high'] - dataframe[['open', 'close']].max(axis=1)
        dataframe['%-upper_wick_range_ratio'] = dataframe['%-upper_wick_size'] / dataframe['%-range_size']
        
        lookback_period = 10
        dataframe['%-max_high'] = dataframe['high'].rolling(50).max()
        dataframe['%-min_low'] = dataframe['low'].rolling(50).min()
        dataframe['%-close_position'] = (dataframe['close'] - dataframe['%-min_low']) / (dataframe['%-max_high'] - dataframe['%-min_low'])
        dataframe['%-current_candle_perc_change'] = (dataframe['high'] / dataframe['open'] - 1) * 100

        # Lazy Bear Impulse Macd
        dataframe['%-hi'] = ta.SMA(dataframe['high'], timeperiod = 28)
        dataframe['%-lo'] = ta.SMA(dataframe['low'], timeperiod = 28)
        dataframe['%-ema1'] = ta.EMA(dataframe['%-HLC3'], timeperiod = 28)
        dataframe['%-ema2'] = ta.EMA(dataframe['%-ema1'], timeperiod = 28)
        dataframe['%-d'] = dataframe['%-ema1'] - dataframe['%-ema2']
        dataframe['%-mi'] = dataframe['%-ema1'] + dataframe['%-d']
        dataframe['%-md'] = np.where(dataframe['%-mi'] > dataframe['%-hi'], 
            dataframe['%-mi'] - dataframe['%-hi'], 
            np.where(dataframe['%-mi'] < dataframe['%-lo'], 
            dataframe['%-mi'] - dataframe['%-lo'], 0))
        dataframe['%-sb'] = ta.SMA(dataframe['%-md'], timeperiod = 8)
        dataframe['%-sh'] = dataframe['%-md'] - dataframe['%-sb']
        
        # WaveTrend using OHLC4 or HA close - 3/21
        ap = (0.333 * (heikinashi['high'] + heikinashi['low'] + heikinashi["close"]))
        
        dataframe['esa'] = ta.EMA(ap, timeperiod = 9)
        dataframe['d'] = ta.EMA(abs(ap - dataframe['esa']), timeperiod = 9)
        dataframe['%-wave_ci'] = (ap-dataframe['esa']) / (0.015 * dataframe['d'])
        dataframe['%-wave_t1'] = ta.EMA(dataframe['%-wave_ci'], timeperiod = 12)  
        dataframe['%-wave_t2'] = ta.SMA(dataframe['%-wave_t1'], timeperiod = 4)

        # 200 SMA and distance
        dataframe['%-200sma'] = ta.SMA(dataframe, timeperiod = 200)
        dataframe['%-200sma_dist'] = get_distance(heikinashi["close"], dataframe['%-200sma'])

        return dataframe

    def feature_engineering_standard(self, dataframe, **kwargs):
        dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        dataframe["%-hour_of_day"] = (dataframe["date"].dt.hour + 1) / 25
        return dataframe


    def set_freqai_targets(self, dataframe, **kwargs):
        dataframe["&s-extrema"] = 0
        min_peaks = argrelextrema(
            dataframe["open"].values, np.less,
            order=self.freqai_info["feature_parameters"]["label_period_candles"]
        )
        max_peaks = argrelextrema(
            dataframe["close"].values, np.greater,
            order=self.freqai_info["feature_parameters"]["label_period_candles"]
        )
        for mp in min_peaks[0]:
            dataframe.at[mp, "&s-extrema"] = -1
        for mp in max_peaks[0]:
            dataframe.at[mp, "&s-extrema"] = 1
        dataframe["minima"] = np.where(dataframe["&s-extrema"] == -1, 1, 0)
        dataframe["maxima"] = np.where(dataframe["&s-extrema"] == 1, 1, 0)
        dataframe['&s-extrema'] = dataframe['&s-extrema'].rolling(
            window=5, win_type='gaussian', center=True).mean(std=0.5)
        dataframe['&s-extrema10'] = dataframe['&s-extrema'].rolling(
            window=10, win_type='gaussian', center=True).mean(std=0.5)
        dataframe['&s-extrema15'] = dataframe['&s-extrema'].rolling(
            window=15, win_type='gaussian', center=True).mean(std=0.5)
        return dataframe


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe)

        dataframe = self.freqai.start(dataframe, metadata, self)

        murrey_math_levels = calculate_murrey_math_levels(dataframe)
        for level, value in murrey_math_levels.items():
            dataframe[level] = value

        dataframe['mmlextreme_oscillator'] = 100 * ((dataframe['close'] - dataframe["[4/8]P"]) / (dataframe["[+3/8]P"] - dataframe["[-3/8]P"]))
        dataframe["DI_catch"] = np.where(
            dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1,
        )
        
        dataframe["minima_sort_threshold"] = dataframe["&s-minima_sort_threshold"]
        dataframe["maxima_sort_threshold"] = dataframe["&s-maxima_sort_threshold"]

        dataframe['min_threshold_mean'] = dataframe["minima_sort_threshold"].expanding().mean()
        dataframe['max_threshold_mean'] = dataframe["maxima_sort_threshold"].expanding().mean()

        dataframe['maxima_check'] = dataframe['maxima'].rolling(12).apply(lambda x: int((x != 1).all()), raw=True).fillna(0)
        dataframe['minima_check'] = dataframe['minima'].rolling(12).apply(lambda x: int((x != 1).all()), raw=True).fillna(0)

        pair = metadata['pair']
        if dataframe['maxima'].iloc[-3] == 1 and dataframe['maxima_check'].iloc[-1] == 0:
            self.dp.send_msg(f'*** {pair} *** Maxima Detected - Potential Short!!!'
                )

        if dataframe['minima'].iloc[-3] == 1 and dataframe['minima_check'].iloc[-1] == 0:
            self.dp.send_msg(f'*** {pair} *** Minima Detected - Potential Long!!!'
                )

        return dataframe


    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        # df.loc[
        #     (
        #         (df["do_predict"] == 1) &  
        #         (df["DI_catch"] == 1) & 
        #         (df["maxima_check"] != 1) & 
        #         (df["&s-extrema"] > df["&s-extrema"].shift(1)) & 
        #         (df["&s-extrema"] < df["minima_sort_threshold"]) & 
        #         (df["&s-extrema"].shift(1) < df["minima_sort_threshold"].shift(1)) &
        #         (df["&s-extrema"].shift(2) < df["minima_sort_threshold"].shift(2)) &
        #         (df["close"].shift(1) < df["open"].shift(1)) &
        #         (df["close"].shift(2) < df["open"].shift(2)) &
        #         (df["maxima"] == 1) &
        #         (df['volume'] > 0)   # Make sure Volume is not 0

        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'Extrema < Minima Threshold 3rd Descending')

        # df.loc[
        #     (
        #         (df["do_predict"] == 1) &  
        #         (df["DI_catch"] == 1) & 
        #         (df["maxima_check"] == 1) & 
        #         (df["&s-extrema"] > df["&s-extrema"].shift(1)) & 
        #         (df["&s-extrema"] < df["minima_sort_threshold"]) & 
        #         (df["&s-extrema"].shift(1) < df["minima_sort_threshold"].shift(1)) &
        #         (df["&s-extrema"].shift(2) < df["minima_sort_threshold"].shift(2)) &
        #         (df["close"] > df["open"]) &
        #         (df["close"].shift(1) < df["open"].shift(1)) &
        #         (df["maxima"] != 1) &
        #         (df['volume'] > 0)   # Make sure Volume is not 0

        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'Extrema < Minima Threshold Quick Reversal')

        df.loc[
            (
                (df["do_predict"] == 1) &  # Guard: tema is raising
                (df["DI_catch"] == 1) & 
                (df["maxima_check"] == 1) & 
                (df["&s-extrema"] < 0) & 
                (df["minima"].shift(1) == 1) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['enter_long', 'enter_tag']] = (1, 'Minima')

        df.loc[
            (
                # (df["do_predict"] == 1) &  # Guard: tema is raising
                # (df["DI_catch"] == 1) & 
                # (df["maxima_check"] == 1) & 
                # (df["&s-extrema"] < 0) & 
                (df["minima_check"] == 0) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['enter_long', 'enter_tag']] = (1, 'Minima Full Send')

        df.loc[
            (
                (df["do_predict"] == 1) &  # Guard: tema is raising
                (df["DI_catch"] == 1) & 
                (df["minima_check"] == 0) & 
                (df["minima_check"].shift(5) == 1) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['enter_long', 'enter_tag']] = (1, 'Minima Check')


        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:


        df.loc[
            (
                (df["do_predict"] == 1) &  # Guard: tema is raising
                (df["maxima_check"] == 0) & 

                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['exit_long', 'exit_tag']] = (1, 'Maxima Check')

        df.loc[
            (
                (df["do_predict"] == 1) &  # Guard: tema is raising
                (df["DI_catch"] == 1) & 
                (df["&s-extrema"] > 0) & 
                (df["maxima"].shift(1) == 1) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['exit_long', 'exit_tag']] = (1, 'Maxima')

        df.loc[
            (
                # (df["do_predict"] == 1) &  # Guard: tema is raising
                # (df["DI_catch"] == 1) & 
                # (df["&s-extrema"] > 0) & 
                (df["maxima_check"] == 0) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['exit_long', 'exit_tag']] = (1, 'Maxima Full Send')

        # df.loc[
        #     (
        #         (df["do_predict"] == 1) &  # Guard: tema is raising
        #         (df["DI_catch"] == 0) & 
        #         (df['volume'] > 0)   # Make sure Volume is not 0

        #     ),
        #     ['exit_long', 'exit_tag']] = (1, 'Outlier')

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

def calculate_murrey_math_levels(df, window_size=64):

    #df = df.iloc[-window_size:]
    
    # Calculate rolling 64-bar maximum and minimum values
    rolling_max_H = df['high'].rolling(window=window_size).max()
    rolling_min_L = df['low'].rolling(window=window_size).min()

    max_H = rolling_max_H
    min_L = rolling_min_L
    range_HL = max_H - min_L

    def calculate_fractal(v2):
        fractal = 0
        if 25000 < v2 <= 250000:
            fractal = 100000
        elif 2500 < v2 <= 25000:
            fractal = 10000
        elif 250 < v2 <= 2500:
            fractal = 1000
        elif 25 < v2 <= 250:
            fractal = 100
        elif 12.5 < v2 <= 25:
            fractal = 12.5
        elif 6.25 < v2 <= 12.5:
            fractal = 12.5
        elif 3.125 < v2 <= 6.25:
            fractal = 6.25
        elif 1.5625 < v2 <= 3.125:
            fractal = 3.125
        elif 0.390625 < v2 <= 1.5625:
            fractal = 1.5625
        elif 0 < v2 <= 0.390625:
            fractal = 0.1953125
        return fractal

    def calculate_octave(v1, v2, mn, mx):
        range_ = v2 - v1
        sum_ = np.floor(np.log(calculate_fractal(v1) / range_) / np.log(2))
        octave = calculate_fractal(v1) * (0.5 ** sum_)
        mn = np.floor(v1 / octave) * octave
        if mn + octave > v2:
            mx = mn + octave
        else:
            mx = mn + (2 * octave)
        return mx

    def calculate_x_values(v1, v2, mn, mx):
        dmml = (v2 - v1) / 8
        x_values = []

        # Calculate the midpoints of each segment
        midpoints = [mn + i * dmml for i in range(8)]

        for i in range(7):
            x_i = (midpoints[i] + midpoints[i + 1]) / 2
            x_values.append(x_i)

        finalH = max(x_values)  # Maximum of the x_values is the finalH

        return x_values, finalH

    def calculate_y_values(x_values, mn):
        y_values = []

        for x in x_values:
            if x > 0:
                y = mn
            else:
                y = 0
            y_values.append(y)

        return y_values

    def calculate_mml(mn, finalH, mx):
        dmml = ((finalH - finalL) / 8) * 1.0699
        mml = (float([mx][0]) * 0.99875) + (dmml * 3) 
        # mml = (float([mx]) * 0.99875) + (dmml * 3) 

        ml = []
        for i in range(0, 16):
            calc = mml - (dmml * (i))
            ml.append(calc)

        murrey_math_levels = {
            "[-3/8]P": ml[14],
            "[-2/8]P": ml[13],
            "[-1/8]P": ml[12],
            "[0/8]P": ml[11],
            "[1/8]P": ml[10],
            "[2/8]P": ml[9],
            "[3/8]P": ml[8],
            "[4/8]P": ml[7],
            "[5/8]P": ml[6],
            "[6/8]P": ml[5],
            "[7/8]P": ml[4],
            "[8/8]P": ml[3],
            "[+1/8]P": ml[2],
            "[+2/8]P": ml[1],
            "[+3/8]P": ml[0]
        }


        return mml, murrey_math_levels

    for i in range(len(df)):
        mn = np.min(min_L.iloc[:i + 1])
        mx = np.max(max_H.iloc[:i + 1])
        x_values, finalH = calculate_x_values(mn, mx, mn, mx)
        y_values = calculate_y_values(x_values, mn)
        finalL = np.min(y_values)
        mml, murrey_math_levels = calculate_mml(finalL, finalH, mx)

        # Add Murrey Math levels to the DataFrame at each time step
        for level, value in murrey_math_levels.items():
            df.at[df.index[i], level] = value

    return df


def PC(dataframe, in1, in2):
    df = dataframe.copy()
    pc = ((in2-in1)/in1) * 100
    return pc
