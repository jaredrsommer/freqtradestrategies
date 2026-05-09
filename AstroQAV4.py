import logging
from functools import reduce
import datetime
import ephem
import talib.abstract as ta
import pandas_ta as pta
import numpy as np
import pandas as pd
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical import qtpylib
from datetime import timedelta, datetime, timezone
from pandas import DataFrame, Series
from technical.pivots_points import pivots_points
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_minutes
from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
from scipy.signal import argrelextrema
from typing import Optional

logger = logging.getLogger(__name__)

class AstroQAV4(IStrategy):
    exit_profit_only = True
    use_custom_stoploss = True
    trailing_stop = True
    position_adjustment_enable = True
    ignore_roi_if_entry_signal = True
    max_entry_position_adjustment = 2
    max_dca_multiplier = 2.5
    process_only_new_candles = True
    can_short = False
    use_exit_signal = True
    startup_candle_count: int = 50
    stoploss = -0.99
    timeframe = '30m'

    minimal_roi = {
        "48000": 0.01,
        "24000": 0.025,
        "12000": 0.05,
        "4800": 0.10,
        "300": 0.15,
        "180": 0.30,
        "120": 0.40,
        "60": 0.45,
        "0": 0.50
    }

    # Protections
    cooldown_lookback = IntParameter(24, 48, default=12, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # Trailing stop loss optimization
    tsl_enable = IntParameter(low=70, high=90, default=80, space='sell', optimize=True, load=True)
    tsl_target5 = DecimalParameter(low=0.3, high=0.4, default=0.3, decimals=2, space='sell', optimize=True, load=True)
    ts5 = DecimalParameter(low=0.04, high=0.06, default=0.05, decimals=2, space='sell', optimize=True, load=True)
    tsl_target4 = DecimalParameter(low=0.18, high=0.3, default=0.2, decimals=2, space='sell', optimize=True, load=True)
    ts4 = DecimalParameter(low=0.03, high=0.05, default=0.045, decimals=3, space='sell', optimize=True, load=True)
    tsl_target3 = DecimalParameter(low=0.12, high=0.18, default=0.15, decimals=2, space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, decimals=3, space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.07, high=0.12, default=0.1, decimals=2, space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, decimals=2, space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.04, high=0.08, default=0.075, decimals=3, space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.16, default=0.013, decimals=3, space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.02, high=0.05, default=0.05, decimals=3, space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.008, high=0.015, default=0.01, decimals=3, space='sell', optimize=True, load=True)

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

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        return proposed_stake / self.max_dca_multiplier

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float, **kwargs) -> Optional[float]:
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        trade_duration = (current_time - trade.open_date_utc).seconds / 60

        if current_profit > 0.10 and trade.nr_of_successful_exits == 0:
            return -(trade.stake_amount / 2)

        if current_profit > -0.015 and trade.nr_of_successful_entries == 1:
            return None
        if current_profit > -0.065 and trade.nr_of_successful_entries == 2:
            return None
        if current_profit > -0.105 and trade.nr_of_successful_entries == 3:
            return None
        if current_profit > -0.13 and trade.nr_of_successful_entries == 4:
            return None
        if current_profit > -0.18 and trade.nr_of_successful_entries == 5:
            return None

        try:
            stake_amount = filled_entries[0].cost
            if count_of_entries == 1:
                stake_amount = stake_amount * 1
            elif count_of_entries == 2:
                stake_amount = stake_amount * 1
            elif count_of_entries == 3:
                stake_amount = stake_amount * 1
            elif count_of_entries == 4:
                stake_amount = stake_amount * 1.5
            elif count_of_entries == 5:
                stake_amount = stake_amount * 1.625
            else:
                stake_amount = stake_amount
            return stake_amount
        except Exception:
            return None

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        trade_duration = (current_time - trade.open_date_utc).seconds / 60

        # Tighten stoploss during challenging aspects (Square, Opposition, Quincunx)
        challenging_aspects = [col for col in dataframe.columns if any(
            aspect in col for aspect in ['square', 'opposition', 'quincunx']
        )]
        if any(current_candle[col] == 1 for col in challenging_aspects) and current_profit > 0:
            self.dp.send_msg(f'*** {pair} *** Challenging aspect detected - Tightening stoploss')
            return 0.02  # Tighten to 2%

        if current_candle['rsi'] < self.tsl_enable.value:
            for stop5 in self.tsl_target5.range:
                if current_profit > stop5:
                    for stop5a in self.ts5.range:
                        self.dp.send_msg(f'*** {pair} *** Profit: {current_profit} - lvl5 {stop5}/{stop5a} activated')
                        return stop5a
            for stop4 in self.tsl_target4.range:
                if current_profit > stop4:
                    for stop4a in self.ts4.range:
                        self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl4 {stop4}/{stop4a} activated')
                        return stop4a
            for stop3 in self.tsl_target3.range:
                if current_profit > stop3:
                    for stop3a in self.ts3.range:
                        self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl3 {stop3}/{stop3a} activated')
                        return stop3a
            for stop2 in self.tsl_target2.range:
                if current_profit > stop2:
                    for stop2a in self.ts2.range:
                        self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl2 {stop2}/{stop2a} activated')
                        return stop2a
            if trade_duration < 360:
                for stop1 in self.tsl_target1.range:
                    if current_profit > stop1:
                        for stop1a in self.ts1.range:
                            self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl1 {stop1}/{stop1a} activated')
                            return stop1a
                for stop0 in self.tsl_target0.range:
                    if current_profit > stop0:
                        for stop0a in self.ts0.range:
                            self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl0 {stop0}/{stop0a} activated')
                            return stop0a
        else:
            for stop0 in self.tsl_target0.range:
                if current_profit > stop0:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} SWINGING FOR THE MOON!!!')
                    return 0.99
        return self.stoploss

    def compute_planet_position(self, planet_name, date):
        observer = ephem.Observer()
        observer.lat = '33.4484'  # Phoenix, Arizona
        observer.lon = '-112.0740'
        planet = getattr(ephem, planet_name)(observer)
        planet.compute(date)
        return planet.ra, planet.dec

    def calculate_angular_separation(self, ra1, ra2):
        """Calculate angular separation between two right ascensions."""
        separation = abs(ra1 - ra2)
        if separation > np.pi:
            separation = 2 * np.pi - separation
        return separation * 180 / np.pi  # Convert to degrees

    def feature_engineering_expand_all(self, dataframe, period, **kwargs):
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, window=period)
        dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=period)
        dataframe["%-er-period"] = pta.er(dataframe['close'], length=period)
        dataframe["%-rocr-period"] = ta.ROCR(dataframe, timeperiod=period)
        dataframe["%-cmf-period"] = chaikin_mf(dataframe, periods=period)
        dataframe["%-tcp-period"] = top_percent_change(dataframe, period)
        dataframe["%-cti-period"] = pta.cti(dataframe['close'], length=period)
        dataframe["%-chop-period"] = qtpylib.chopiness(dataframe, period)
        dataframe["%-linear-period"] = ta.LINEARREG_ANGLE(dataframe['close'], timeperiod=period)
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        dataframe["%-atr-periodp"] = dataframe["%-atr-period"] / dataframe['close'] * 1000
        return dataframe

    def feature_engineering_expand_basic(self, dataframe, metadata, **kwargs):
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-ibs"] = ((dataframe['close'] - dataframe['low']) / (dataframe['high'] - dataframe['low']))
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-obv"] = ta.OBV(dataframe)
        dataframe['%-willr14'] = pta.willr(dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['%-willr14PC'] = PC(dataframe, dataframe['%-willr14'], dataframe['%-willr14'].shift(1))
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_34'] = ta.EMA(dataframe, timeperiod=34)
        dataframe['%-ewo'] = EWO(dataframe, dataframe['ema_8'], dataframe['ema_34'])
        dataframe['%-distema8'] = get_distance(dataframe['close'], dataframe['ema_8'])
        dataframe['%-distema34'] = get_distance(dataframe['close'], dataframe['ema_34'])

        crsi_closechange = dataframe['close'] / dataframe['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        dataframe['%-crsi'] = (ta.RSI(dataframe['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(dataframe['close'], 100)) / 3

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['%-ha_open'] = heikinashi['open']
        dataframe['%-ha_close'] = heikinashi['close']
        dataframe['%-ha_high'] = heikinashi['high']
        dataframe['%-ha_low'] = heikinashi['low']
        dataframe['%-ha_closedelta'] = (heikinashi['close'] - heikinashi['close'].shift())
        dataframe['%-ha_tail'] = (heikinashi['close'] - heikinashi['low'])
        dataframe['%-ha_wick'] = (heikinashi['high'] - heikinashi['close'])
        dataframe['%-HLC3'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3

        #murrey_math_levels = calculate_murrey_math_levels(dataframe)
        #for level, value in murrey_math_levels.items():
        #    dataframe[level] = value

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

        # TTM Squeeze
        ttm_Squeeze = pta.squeeze(high=dataframe['high'], low=dataframe['low'], close=dataframe["close"], lazybear=True)
        dataframe['%-ttm_Squeeze'] = ttm_Squeeze['SQZ_20_2.0_20_1.5_LB']
        dataframe['%-ttm_ema'] = ta.EMA(dataframe['%-ttm_Squeeze'], timeperiod=4)
        dataframe['%-squeeze_ON'] = ttm_Squeeze['SQZ_ON']
        dataframe['%-squeeze_OFF'] = ttm_Squeeze['SQZ_OFF']
        dataframe['%-NO_squeeze'] = ttm_Squeeze['SQZ_NO']

        # Astro Features
        if metadata["tf"] == "1d":
            dataframe['e_date'] = dataframe['date'].apply(lambda x: ephem.Date(x.strftime("%Y/%m/%d %H:%M:%S")))
            moon_phase = dataframe['e_date'].apply(lambda x: ephem.Moon(x).phase)
            dataframe['%-Moon_Phase'] = moon_phase

            planets = [
                'Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 
                'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto'
            ]

            # Compute planet positions
            for planet_name in planets:
                dataframe[f'{planet_name}_ra'], dataframe[f'{planet_name}_dec'] = zip(*dataframe['e_date'].apply(
                    lambda x: self.compute_planet_position(planet_name, x)
                ))
                dataframe[planet_name] = dataframe['e_date'].apply(
                    lambda x: ephem.constellation(getattr(ephem, planet_name)(x))[1]
                )
                dataframe[f'%-{planet_name}_rax'] = dataframe[f'{planet_name}_ra'] * 180 / np.pi
                dataframe[f'%-{planet_name}_dec'] = (dataframe[f'{planet_name}_dec'] + 0.5) * 100

            # Define aspects with angles and tolerances
            aspects = {
                'conjunction': (0, 8),
                'semi_sextile': (30, 4),
                'semi_square': (45, 4),
                'sextile': (60, 6),
                'square': (90, 8),
                'trine': (120, 8),
                'sesquiquadrate': (135, 4),
                'quincunx': (150, 5),
                'opposition': (180, 8)
            }

            # Calculate aspects for all planet pairs
            for i, planet1 in enumerate(planets):
                for planet2 in planets[i+1:]:  # Avoid duplicate pairs
                    angle_col = f'{planet1.lower()}_{planet2.lower()}_angle'
                    dataframe[angle_col] = dataframe.apply(
                        lambda row: self.calculate_angular_separation(
                            row[f'%-{planet1}_rax'], row[f'%-{planet2}_rax']
                        ), axis=1
                    )
                    for aspect, (angle, orb) in aspects.items():
                        dataframe[f'{planet1.lower()}_{planet2.lower()}_{aspect}'] = np.where(
                            (dataframe[angle_col] >= angle - orb) & 
                            (dataframe[angle_col] <= angle + orb), 1, 0
                        )

            # Balsamic Moon
            dataframe['balsamic_moon'] = np.where(
                (dataframe['%-Moon_Phase'] >= 315) & (dataframe['%-Moon_Phase'] < 360), 1, 0
            )

        return dataframe

    def feature_engineering_standard(self, dataframe, **kwargs):
        dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        dataframe["%-hour_of_day"] = (dataframe["date"].dt.hour + 1) / 25
        return dataframe

    def set_freqai_targets(self, dataframe, **kwargs):
        dataframe["&s-extrema"] = 0
        min_peaks = argrelextrema(
            dataframe["low"].values, np.less,
            order=self.freqai_info["feature_parameters"]["label_period_candles"]
        )
        max_peaks = argrelextrema(
            dataframe["high"].values, np.greater,
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
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe = self.freqai.start(dataframe, metadata, self)
        dataframe["DI_catch"] = np.where(dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1)
        dataframe["minima_sort_threshold"] = dataframe["&s-minima_sort_threshold"]
        dataframe["maxima_sort_threshold"] = dataframe["&s-maxima_sort_threshold"]
        dataframe['maxima_check'] = dataframe['maxima'].rolling(12).apply(lambda x: int((x != 1).all()), raw=True).fillna(0)
        dataframe['minima_check'] = dataframe['minima'].rolling(12).apply(lambda x: int((x != 1).all()), raw=True).fillna(0)
        dataframe['min_threshold_mean'] = dataframe["minima_sort_threshold"].expanding().mean()
        dataframe['max_threshold_mean'] = dataframe["maxima_sort_threshold"].expanding().mean()

        maxima_indices = dataframe[dataframe['maxima'] == 1].index
        minima_indices = dataframe[dataframe['minima'] == 1].index

        maxima_to_minima_diff = []
        for max_idx in maxima_indices:
            for min_idx in minima_indices:
                if max_idx < min_idx:
                    maxima_to_minima_diff.append(dataframe.loc[min_idx, 'close'] - dataframe.loc[max_idx, 'close'])
        average_maxima_to_minima_diff = sum(maxima_to_minima_diff) / (len(maxima_to_minima_diff) + 0.000000000001)

        minima_to_maxima_diff = []
        for min_idx in minima_indices:
            for max_idx in maxima_indices:
                if min_idx < max_idx:
                    minima_to_maxima_diff.append(dataframe.loc[max_idx, 'close'] - dataframe.loc[min_idx, 'close'])
        average_minima_to_maxima_diff = sum(minima_to_maxima_diff) / (len(minima_to_maxima_diff) + 0.000000000001)

        dataframe['avg_max_to_min_diff'] = average_maxima_to_minima_diff
        dataframe['avg_min_to_max_diff'] = average_minima_to_maxima_diff

        pair = metadata['pair']
        if dataframe['maxima_check'].iloc[-1] == 0 and dataframe['maxima_check'].iloc[-2] == 0:
            self.dp.send_msg(f'*** {pair} *** Maxima Detected - Potential Short!!!')
        if dataframe['minima_check'].iloc[-1] == 0 and dataframe['minima_check'].iloc[-2] == 0:
            self.dp.send_msg(f'*** {pair} *** Minima Detected - Potential Long!!!')
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Harmonious aspects (Trine, Sextile, Conjunction)
        harmonious_aspects = [col for col in df.columns if any(
            aspect in col for aspect in ['trine', 'sextile', 'conjunction']
        )]

        # Entry with harmonious aspects
        df.loc[
            (
                (df["do_predict"] == 1) &
                (df["DI_catch"] == 1) &
                (df["maxima_check"] == 1) &
                (df["&s-extrema"] < 0) &
                (df["minima"].shift(1) == 1) &
                (df[harmonious_aspects].sum(axis=1) > 0) &  # At least one harmonious aspect
                (df['balsamic_moon'] == 0) &  # Avoid consolidation
                (df['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'Minima with Harmonious Aspect')

        # Entry with minima check, no challenging aspects
        challenging_aspects = [col for col in df.columns if any(
            aspect in col for aspect in ['square', 'opposition', 'quincunx']
        )]
        df.loc[
            (
                (df["do_predict"] == 1) &
                (df["DI_catch"] == 1) &
                (df["minima_check"] == 0) &
                (df["minima_check"].shift(5) == 1) &
                (df[challenging_aspects].sum(axis=1) == 0) &  # No challenging aspects
                (df['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'Minima Check Clear')
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Exit during challenging aspects
        challenging_aspects = [col for col in df.columns if any(
            aspect in col for aspect in ['square', 'opposition', 'quincunx', 'semi_square', 'sesquiquadrate']
        )]
        df.loc[
            (
                (df["do_predict"] == 1) &
                (df["DI_catch"] == 1) &
                (df[challenging_aspects].sum(axis=1) > 0) &  # At least one challenging aspect
                (df['volume'] > 0)
            ),
            ['exit_long', 'exit_tag']] = (1, 'Challenging Aspect')

        # Standard exit with maxima
        df.loc[
            (
                (df["do_predict"] == 1) &
                (df["DI_catch"] == 1) &
                (df["&s-extrema"] > 0) &
                (df["maxima"].shift(1) == 1) &
                (df['volume'] > 0)
            ),
            ['exit_long', 'exit_tag']] = (1, 'Maxima')
        return df

# Helper Functions
def top_percent_change(dataframe: DataFrame, length: int) -> float:
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

def EWO(dataframe, sma1, sma2):
    df = dataframe.copy()
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif

def get_distance(p1, p2):
    return abs((p1) - (p2))

def celestial_house(planet_name):
    house = 0
    if planet_name == "Aries":
        house = 1
    elif planet_name == "Taurus":
        house = 2
    elif planet_name == "Gemini":
        house = 3
    elif planet_name == "Cancer":
        house = 4
    elif planet_name == "Leo":
        house = 5
    elif planet_name == "Virgo":
        house = 6
    elif planet_name == "Libra":
        house = 7
    elif planet_name == "Scorpio":
        house = 8
    elif planet_name == "Sagittarius":
        house = 9
    elif planet_name == "Capricorn":
        house = 10
    elif planet_name == "Aquarius":
        house = 11
    elif planet_name == "Pisces":
        house = 12
    return house

def calculate_murrey_math_levels(df, window_size=64):
    df = df.iloc[-window_size:]
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
        midpoints = [mn + i * dmml for i in range(8)]
        for i in range(7):
            x_i = (midpoints[i] + midpoints[i + 1]) / 2
            x_values.append(x_i)
        finalH = max(x_values)
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
        return murrey_math_levels

    mn = np.min(min_L)
    mx = np.max(max_H)
    x_values, finalH = calculate_x_values(mn, mx, mn, mx)
    y_values = calculate_y_values(x_values, mn)
    finalL = np.min(y_values)
    mml = calculate_mml(finalL, finalH, mx)
    return mml

def PC(dataframe, in1, in2):
    df = dataframe.copy()
    pc = ((in2 - in1) / in1) * 100
    return pc