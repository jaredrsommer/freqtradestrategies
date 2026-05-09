from freqtrade.strategy import IStrategy
import pandas as pd
import numpy as np
import talib
from scipy.fft import fft
from typing import Dict, List

class HurstCycle3(IStrategy):
    timeframe = '15m'
    minimal_roi = {"0": 0.05, "60": 0.03, "120": 0.01}
    stoploss = -0.04
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.02

    # Hyperparameters to optimize
    base_cycle_period = 20
    envelope_factor_short = 1.2
    envelope_factor_long = 2.0
    filter_weights = [1, 2, 3, 2, 1]  # Symmetric weights for smoothing (p. 177)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Spectral Analysis
        close = dataframe['close'].values
        n = min(len(close), 500)
        yf = fft(close[-n:])
        freq = np.fft.fftfreq(n)
        power = np.abs(yf) ** 2
        dominant_idx = np.argmax(power[1:n//2]) + 1
        dominant_period = int(1 / freq[dominant_idx]) if freq[dominant_idx] != 0 else self.base_cycle_period
        self.short_cycle = max(10, min(30, dominant_period))
        self.long_cycle = self.short_cycle * 2

        # Numerical Filter
        weights = np.array(self.filter_weights) / sum(self.filter_weights)
        filtered = np.convolve(close, weights, mode='valid')
        dataframe['filtered_close'] = pd.Series(filtered, index=dataframe.index[-len(filtered):])

        # FLD with shift and extrapolation
        half_short = int(self.short_cycle / 2)
        half_long = int(self.long_cycle / 2)
        fld_short_base = dataframe['filtered_close'].rolling(window=self.short_cycle, center=True).mean()
        fld_long_base = dataframe['filtered_close'].rolling(window=self.long_cycle, center=True).mean()
        dataframe['fld_short'] = fld_short_base.shift(-half_short)
        dataframe['fld_long'] = fld_long_base.shift(-half_short)
        for i in range(1, half_short + 1):
            if len(dataframe) > half_short + 1:  # Ensure enough data
                dataframe['fld_short'].iloc[-i] = dataframe['fld_short'].iloc[-half_short-1] + \
                                                  (dataframe['fld_short'].iloc[-half_short-1] - dataframe['fld_short'].iloc[-half_short-2])
                dataframe['fld_long'].iloc[-i] = dataframe['fld_long'].iloc[-half_short-1] + \
                                                 (dataframe['fld_long'].iloc[-half_short-1] - dataframe['fld_long'].iloc[-half_short-2])

        # [Rest of indicators: envelopes, VTLs, trend...]
        dataframe['sma_short'] = talib.SMA(dataframe['filtered_close'], timeperiod=self.short_cycle)
        dataframe['sma_long'] = talib.SMA(dataframe['filtered_close'], timeperiod=self.long_cycle)
        dataframe['atr_short'] = talib.ATR(dataframe['high'], dataframe['low'], dataframe['filtered_close'], timeperiod=self.short_cycle)
        dataframe['atr_long'] = talib.ATR(dataframe['high'], dataframe['low'], dataframe['filtered_close'], timeperiod=self.long_cycle)
        dataframe['upper_env_short'] = dataframe['sma_short'] + self.envelope_factor_short * dataframe['atr_short']
        dataframe['lower_env_short'] = dataframe['sma_short'] - self.envelope_factor_short * dataframe['atr_short']
        dataframe['upper_env_long'] = dataframe['sma_long'] + self.envelope_factor_long * dataframe['atr_long']
        dataframe['lower_env_long'] = dataframe['sma_long'] - self.envelope_factor_long * dataframe['atr_long']
        dataframe['trough'] = dataframe['filtered_close'].rolling(self.short_cycle).min()
        dataframe['crest'] = dataframe['filtered_close'].rolling(self.short_cycle).max()
        dataframe['is_trough'] = np.where(dataframe['filtered_close'] == dataframe['trough'], 1, 0)
        dataframe['is_crest'] = np.where(dataframe['filtered_close'] == dataframe['crest'], 1, 0)
        troughs = dataframe[dataframe['is_trough'] == 1].index
        crests = dataframe[dataframe['is_crest'] == 1].index
        if len(troughs) >= 2:
            t1, t2 = troughs[-2], troughs[-1]
            dataframe['vtl_up_slope'] = (dataframe['filtered_close'][t2] - dataframe['filtered_close'][t1]) / (t2 - t1)
        if len(crests) >= 2:
            c1, c2 = crests[-2], crests[-1]
            dataframe['vtl_down_slope'] = (dataframe['filtered_close'][c2] - dataframe['filtered_close'][c1]) / (c2 - c1)
        dataframe['trend'] = np.where(dataframe['filtered_close'] > dataframe['sma_long'], 1, -1)

        return dataframe
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Long Entry: Trough, below FLD, short envelope breach, VTL breakout
        dataframe.loc[
            # (dataframe['is_trough'] == 1) &
            (dataframe['filtered_close'] < dataframe['fld_short']) &
            (dataframe['filtered_close'].shift() > dataframe['fld_short'].shift()),
            # (dataframe['trend'] == 1),
            'enter_long'] = 1

        # # Short Entry: Crest, above FLD, short envelope breach, VTL breakout
        # dataframe.loc[
        #     (dataframe['is_crest'] == 1) &
        #     (dataframe['filtered_close'] > dataframe['fld_short']) &
        #     (dataframe['filtered_close'] > dataframe['upper_env_short']) &
        #     (dataframe['trend'] == -1) &
        #     (dataframe['vtl_down_slope'].notna() & (dataframe['vtl_down_slope'] < 0)),
        #     'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Exit Long: Hit FLD or upper long envelope
        dataframe.loc[
            (dataframe['filtered_close'] > dataframe['fld_short']) &
            (dataframe['filtered_close'].shift() < dataframe['fld_short'].shift()),
            'exit_long'] = 1

        # # Exit Short: Hit FLD or lower long envelope
        # dataframe.loc[
        #     (dataframe['enter_short'].shift(1) == 1) &
        #     ((dataframe['filtered_close'] <= dataframe['fld_long']) | 
        #      (dataframe['filtered_close'] <= dataframe['lower_env_long'])),
        #     'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time, current_rate: float, proposed_leverage: float, 
                 max_leverage: float, side: str, **kwargs) -> float:
        return 3.0
