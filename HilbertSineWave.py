from freqtrade.strategy import IStrategy
import numpy as np
import pandas as pd
from math import sin, cos, atan, floor, pi

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

class HilbertSineWave(IStrategy):
    # Strategy parameters
    timeframe = '1h'
    minimal_roi = {"0": 0.1}
    stoploss = -0.05

    # Custom parameters
    alpha = 0.09  # Smoothing factor
    amplitude_factor = 2.0  # Controls the amplitude of support/resistance levels

    def custom_percentile_rank(self, series, window, percent=50):
        """Custom implementation of percentile rank similar to Pine Script's percentile_nearest_rank"""
        rolling = series.rolling(window=window, min_periods=1)
        return rolling.apply(lambda x: np.percentile(x, percent) if not np.all(np.isnan(x)) else np.nan, raw=True)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Calculate Smooth price (handle initial NaN values)
        dataframe['smooth'] = (
            dataframe['close'] + 
            2 * dataframe['close'].shift(1).fillna(dataframe['close']) + 
            2 * dataframe['close'].shift(2).fillna(dataframe['close']) + 
            dataframe['close'].shift(3).fillna(dataframe['close'])
        ) / 6

        # Initialize cycle
        dataframe['cycle'] = 0.0
        for i in range(2, len(dataframe)):
            dataframe.loc[dataframe.index[i], 'cycle'] = (
                (1 - 0.5 * self.alpha) * (1 - 0.5 * self.alpha) * 
                (dataframe['smooth'].iloc[i] - 2 * dataframe['smooth'].iloc[i-1] + 
                 dataframe['smooth'].iloc[i-2]) +
                2 * (1 - self.alpha) * dataframe['cycle'].iloc[i-1] -
                (1 - self.alpha) * (1 - self.alpha) * dataframe['cycle'].iloc[i-2]
            )

        # Initialize inst_period with default value
        dataframe['inst_period'] = 15.0  # Initial value before calculations

        # Calculate Q1 and I1
        dataframe['q1'] = (
            0.0962 * dataframe['cycle'] + 
            0.5769 * dataframe['cycle'].shift(2).fillna(0) -
            0.5769 * dataframe['cycle'].shift(4).fillna(0) -
            0.0962 * dataframe['cycle'].shift(6).fillna(0)
        ) * (0.5 + 0.08 * dataframe['inst_period'].shift(1))

        dataframe['i1'] = dataframe['cycle'].shift(3).fillna(0)

        # Calculate DeltaPhase with NaN handling
        denominator = (1 + dataframe['i1']*dataframe['i1'].shift(1)/(dataframe['q1']*dataframe['q1'].shift(1)))
        dataframe['delta_phase'] = np.where(
            (dataframe['q1'] != 0) & (dataframe['q1'].shift(1) != 0) & (denominator != 0),
            (dataframe['i1']/dataframe['q1'] - dataframe['i1'].shift(1)/dataframe['q1'].shift(1))/denominator,
            np.nan
        )
        dataframe['delta_phase'] = dataframe['delta_phase'].clip(lower=0.1, upper=1.1)

        # Calculate median delta using custom percentile rank
        dataframe['median_delta'] = self.custom_percentile_rank(dataframe['delta_phase'], window=5, percent=50)
        dataframe['dc'] = np.where(dataframe['median_delta'] == 0, 
                                 15, 
                                 6.28318/dataframe['median_delta'].replace(0, np.nan) + 0.5)
        
        # Update inst_period after dc is calculated
        for i in range(1, len(dataframe)):
            dataframe.loc[dataframe.index[i], 'inst_period'] = (
                0.33 * dataframe['dc'].iloc[i] + 
                0.67 * dataframe['inst_period'].iloc[i-1]
            )
        
        # Handle NaN in dc_period calculation
        dataframe['dc_period'] = dataframe['inst_period'].apply(lambda x: floor(x) if not pd.isna(x) else 15)

        # Calculate Sine and LeadSine
        def calculate_sines(row):
            dc_period = int(row['dc_period'])
            real_part = 0
            imag_part = 0
            for count in range(dc_period):
                cycle_shift = dataframe['cycle'].shift(count).iloc[row.name]
                if pd.isna(cycle_shift):
                    cycle_shift = 0
                real_part += sin(6.28318 * count / dc_period) * cycle_shift
                imag_part += cos(6.28318 * count / dc_period) * cycle_shift
            
            dc_phase = atan(real_part/imag_part) if abs(imag_part) > 0.001 else (1.572963 * np.sign(real_part))
            dc_phase += 1.572963
            if imag_part < 0:
                dc_phase += 3.1415926
            if dc_phase > 5.49778705:
                dc_phase -= 6.28318
                
            return pd.Series({
                'sine': sin(dc_phase),
                'lead_sine': sin(dc_phase + 0.78539815)
            })

        sines = dataframe.apply(calculate_sines, axis=1)
        dataframe['sine'] = sines['sine'].fillna(0)
        dataframe['lead_sine'] = sines['lead_sine'].fillna(0)

        # Calculate support/resistance levels with separate columns
        dataframe['support_green'] = np.nan
        dataframe['resistance_red'] = np.nan
        
        dc_offset = lambda x: int(floor(x['dc_period']/8))
        prev_support = True
        curr_value = 0

        for i in range(len(dataframe)):
            offset = dc_offset(dataframe.iloc[i])
            if i < offset:
                continue
                
            lead_sine_shift = dataframe['lead_sine'].iloc[i - offset]
            sine_shift = dataframe['sine'].iloc[i - offset]

            if pd.isna(lead_sine_shift) or pd.isna(sine_shift):
                continue

            if lead_sine_shift <= sine_shift and prev_support:
                curr_value = dataframe['high'].iloc[i] * (1 + 0.01 * self.amplitude_factor)
                dataframe.loc[dataframe.index[i], 'resistance_red'] = curr_value
                prev_support = False
            elif lead_sine_shift > sine_shift and not prev_support:
                curr_value = dataframe['low'].iloc[i] * (1 - 0.01 * self.amplitude_factor)
                dataframe.loc[dataframe.index[i], 'support_green'] = curr_value
                prev_support = True
            else:
                if prev_support:
                    dataframe.loc[dataframe.index[i], 'support_green'] = curr_value
                else:
                    dataframe.loc[dataframe.index[i], 'resistance_red'] = curr_value

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (dataframe['close'] <= dataframe['support_green']) & 
            (dataframe['support_green'].notna()),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (dataframe['close'] >= dataframe['resistance_red']) & 
            (dataframe['resistance_red'].notna()),
            'exit_long'] = 1
        return dataframe