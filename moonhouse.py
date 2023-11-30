import ephem
import pandas as pd
from freqtrade.strategy import IStrategy
import math
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
import pandas_ta as pta
from typing import Optional
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import BooleanParameter, stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, \
    CategoricalParameter
import technical.indicators as ftt
from freqtrade.exchange import timeframe_to_minutes



class moonhouse(IStrategy):

    # Optimal timeframe for the strategy
    timeframe = '1d'
    # Stoploss:
    stoploss = -0.318
    """
    Simple Astrology-based Trading Strategy using Freqtrade.
    Tracks moon phases and planet transits through astrological signs.
    """
    #hyperopt the astrology


    def compute_planet_position(self, planet_name, date):
        observer = ephem.Observer()
        observer.lat = '33.4484'  # Latitude of Phoenix, Arizona
        observer.lon = '-112.0740'  # Longitude of Phoenix, Arizona

        planet = getattr(ephem, planet_name)(observer)
        planet.compute(date)

        return planet.ra, planet.dec

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Convert timestamp to ephem date format (UTC time)
        df['e_date'] = df['date'].apply(lambda x: ephem.Date(x.strftime("%Y/%m/%d %H:%M:%S")))

        # Calculate moon phase
        moon_phase = df['e_date'].apply(lambda x: ephem.Moon(x).phase)

        # Calculate planet transits through astrological signs
        planets = {
            'Sun': ephem.Sun,
            'Mercury': ephem.Mercury,
            'Venus': ephem.Venus,
            'Mars': ephem.Mars,
            'Jupiter': ephem.Jupiter,
            'Saturn': ephem.Saturn,
            'Uranus': ephem.Uranus,
            'Neptune': ephem.Neptune,
            'Pluto': ephem.Pluto,
            'Moon': ephem.Moon,
        }

        for planet_name, planet in planets.items():
            df[planet_name + '_ra'], df[planet_name + '_dec'] = zip(*df['e_date'].apply(
                lambda x: self.compute_planet_position(planet_name, x)
            ))
            df[planet_name] = df['e_date'].apply(lambda x: ephem.constellation(planet(x))[1])

        # Add moon phase column to DataFrame
        df['Moon_Phase'] = moon_phase
        df['Sun_rax'] = df['Sun_ra'] * 15.923566879
        df['Mercury_rax'] = df['Mercury_ra'] * 15.923566879
        df['Venus_rax'] = df['Venus_ra'] * 15.923566879
        df['Mars_rax'] = df['Mars_ra'] * 15.923566879
        df['Jupiter_rax'] = df['Jupiter_ra'] * 15.923566879
        df['Saturn_rax'] = df['Saturn_ra'] * 15.923566879
        df['Uranus_rax'] = df['Uranus_ra'] * 15.923566879
        df['Neptune_rax'] = df['Neptune_ra'] * 15.923566879
        df['Pluto_rax'] = df['Pluto_ra'] * 15.923566879
        df['Moon_Dec_Phase'] = (df['Moon_dec']+0.5) * 100
        #Traditional Indicators section
        df['200 SMA'] = ta.SMA(df, timeperiod=200)
        df['50 SMA'] = ta.SMA(df, timeperiod=50)
        
        return df

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (
                (df['Moon_Phase'] > df['Moon_Phase'].shift(1) ) & 
                (df['Moon_Phase'] > 95) & 
                (df['Mercury_dec'] < df['Sun_dec']) &
                (df['Jupiter_rax'] < df['Sun_rax']) &
                (df['Mercury_dec'] > df['Mercury_dec'].shift(1)) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'FullMoon - Mercury below the Sun & Mercury Rising')

        df.loc[
            (
                (df['Moon_Phase'] > df['Moon_Phase'].shift(1) ) & 
                (df['Moon_Phase'] < 5) & 
                (df['Jupiter_rax'] < df['Sun_rax']) &
                (df['Mercury_dec'] < df['Sun_dec']) &
                (df['Mercury_dec'] > df['Mercury_dec'].shift(1)) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'NewMoon - Mercury below the Sun & Mercury Rising')

        
        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df['Moon_Phase'] > df['Moon_Phase'].shift(1) ) & 
                (df['Sun_rax'] > 50) &
                (df['Mercury_dec'] < df['Mercury_dec'].shift(1)) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'Mercury decreasing and Past Full Moon')

        df.loc[
            (
                (df['Moon_Phase'] > df['Moon_Phase'].shift(1) ) & 
                (df['Mercury_dec'] > df['Sun_dec']) &
                (df['Moon_Phase'] > 95) &
                (df['Sun_rax'] > 50) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'Mercury above the Sun')

        # Example: Sell when the moon is in the Full Moon phase and Jupiter is in Sagittarius
        return df
