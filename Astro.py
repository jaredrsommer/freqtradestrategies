import logging
from functools import reduce
import datetime
import ephem
import talib.abstract as ta
import pandas_ta as pta
import numpy as np
import pandas as pd
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import timedelta, datetime, timezone
from pandas import DataFrame, Series
from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter)
from typing import Optional

logger = logging.getLogger(__name__)

class Astro(IStrategy):
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
    timeframe = '1d'

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
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, decimals=3, space='sell', optimize=True, load=True)
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

        # Tighten stoploss during challenging aspects
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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Core Technical Indicators
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_34'] = ta.EMA(dataframe, timeperiod=34)

        # VWAP
        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe['%-vwap_middleband'] = vwap
        dataframe['%-dist_to_vwap'] = get_distance(dataframe['close'], dataframe['%-vwap_middleband'])

        # TTM Squeeze
        ttm_Squeeze = pta.squeeze(high=dataframe['high'], low=dataframe['low'], close=dataframe["close"], lazybear=True)
        dataframe['%-ttm_Squeeze'] = ttm_Squeeze['SQZ_20_2.0_20_1.5_LB']
        dataframe['%-squeeze_ON'] = ttm_Squeeze['SQZ_ON']

        # Astrological Features
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
            for planet2 in planets[i+1:]:
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

        dataframe['aspects'] = [col for col in dataframe.columns if any(
            aspect in col for aspect in ['trine', 'sextile', 'conjunction']
        )]


        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Harmonious aspects
        harmonious_aspects = [col for col in df.columns if any(
            aspect in col for aspect in ['trine', 'sextile', 'conjunction']
        )]

        # Entry: RSI oversold, close below VWAP, harmonious aspect
        df.loc[
            (
                (df['rsi'] < 40) &
                (df['close'] < df['%-vwap_middleband']) &
                (df['ema_8'] > df['ema_34']) &
                (df[harmonious_aspects].sum(axis=1) > 0) &
                (df['balsamic_moon'] == 0) &
                (df['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'Oversold with Harmonious Aspect')

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Challenging aspects
        challenging_aspects = [col for col in df.columns if any(
            aspect in col for aspect in ['square', 'opposition', 'quincunx', 'semi_square', 'sesquiquadrate']
        )]

        # Exit: RSI overbought or challenging aspect
        df.loc[
            (
                ((df['rsi'] > 70) | (df['%-squeeze_ON'] == 1)) &
                (df[challenging_aspects].sum(axis=1) > 0) &
                (df['volume'] > 0)
            ),
            ['exit_long', 'exit_tag']] = (1, 'Overbought or Challenging Aspect')

        return df

# Helper Functions
def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df, window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']

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