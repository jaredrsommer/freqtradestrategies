from freqtrade.strategy import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter
from pandas import DataFrame
import numpy as np
import pandas_ta as ta

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

class OmaGann(IStrategy):
    # Hyperopt parameters
    len_param = IntParameter(5, 20, default=10, space="buy")
    const_param = DecimalParameter(1.5, 5, default=2.5, space="buy")
    clsper_param = IntParameter(1, 5, default=1, space="buy")

    timeframe = '30m'
    minimal_roi = {"0": 0.1}
    stoploss = -0.1

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate OMA with adaptive logic
        def oma_series(src_series, length, const, adaptive=True):
            series = src_series.copy()
            e1 = series.ewm(alpha=1/length, adjust=adaptive).mean()
            e2 = e1.ewm(alpha=1/length, adjust=adaptive).mean()
            v1 = 1.5*e1 - 0.5*e2
            
            e3 = v1.ewm(alpha=1/length, adjust=adaptive).mean()
            e4 = e3.ewm(alpha=1/length, adjust=adaptive).mean()
            v2 = 1.5*e3 - 0.5*e4
            
            e5 = v2.ewm(alpha=1/length, adjust=adaptive).mean()
            e6 = e5.ewm(alpha=1/length, adjust=adaptive).mean()
            return 1.5*e5 - 0.5*e6

        # Get hyperopt values
        len_val = int(self.len_param.value)
        const_val = float(self.const_param.value)
        clsper_val = int(self.clsper_param.value)

        # Calculate OMA values
        dataframe['high_ma'] = oma_series(dataframe['high'], len_val, const_val)
        dataframe['low_ma'] = oma_series(dataframe['low'], len_val, const_val)
        dataframe['close_ma'] = oma_series(dataframe['close'], clsper_val, const_val)

        # Calculate Gann HiLo Activator
        jfghla = []
        for i in range(len(dataframe)):
            if i == 0:
                jfghla.append(np.nan)
                continue
                
            close_ma = dataframe['close_ma'].iloc[i]
            prev_high_ma = dataframe['high_ma'].iloc[i-1]
            prev_low_ma = dataframe['low_ma'].iloc[i-1]
            
            if close_ma > prev_high_ma:
                jfghla.append(dataframe['low_ma'].iloc[i])
            elif close_ma < prev_low_ma:
                jfghla.append(dataframe['high_ma'].iloc[i])
            else:
                jfghla.append(jfghla[-1] if len(jfghla) > 0 else np.nan)
        
        dataframe['jfghla'] = jfghla
        dataframe['trend'] = np.where(dataframe['close_ma'] > dataframe['jfghla'], 1, 
                                    np.where(dataframe['close_ma'] < dataframe['jfghla'], -1, 0))
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long entry when trend turns positive
        dataframe.loc[
            (dataframe['trend'] == 1) &
            (dataframe['trend'].shift(1) <= 0),
            'enter_long'] = 1

        # Short entry when trend turns negative
        dataframe.loc[
            (dataframe['trend'] == -1) &
            (dataframe['trend'].shift(1) >= 0),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long when trend turns negative
        dataframe.loc[
            (dataframe['trend'] == -1) &
            (dataframe['trend'].shift(1) >= 0),
            'exit_long'] = 1

        # Exit short when trend turns positive
        dataframe.loc[
            (dataframe['trend'] == 1) &
            (dataframe['trend'].shift(1) <= 0),
            'exit_short'] = 1

        return dataframe