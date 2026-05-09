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
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade


logger = logging.getLogger(__name__)

class TGMA(IStrategy):
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

    ### Strategy parameters ###
    timeframe = '1h'
    exit_profit_only = False ### No selling at a loss
    ignore_roi_if_entry_signal = True
    process_only_new_candles = True
    can_short = False
    use_exit_signal = True
    startup_candle_count = 20
    use_custom_stoploss = True
    trailing_stop = False

    locked_stoploss = {}
    minimal_roi = {}
    

    # Stoploss:
    stoploss = -0.12  # Fail Safe Default do not hyper opt Stoploss Space

    # Trailing stop:
    trailing_stop = True  # value loaded from strategy
    trailing_stop_positive = 0.035 # value loaded from strategy
    trailing_stop_positive_offset = 0.013 # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy

    # indicators
    u_window_size = IntParameter(60, 240, default=240, space='buy', optimize=True)
    l_window_size = IntParameter(20, 40, default=30, space='buy', optimize=True)

    #limits
    buylimit = IntParameter(0, 15, default=10, space='buy', optimize=True)
    selllimit = IntParameter(15, 23, default=20, space='sell', optimize=True)

    # CooldownPeriod 
    cooldown_lookback = IntParameter(0, 12, default=5, space="protection", optimize=True, load=True)
    
    # StoplossGuard    
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True, load=True)
    stop_duration = IntParameter(6, 40, default=39, space="protection", optimize=True, load=True)
    stop_protection_only_per_pair = BooleanParameter(default=True, space="protection", optimize=True, load=True)
    stop_protection_only_per_side = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    stop_protection_trade_limit = IntParameter(1, 10, default=4, space="protection", optimize=True, load=True)
    stop_protection_required_profit = DecimalParameter(-0.10, 0.01, default=-0.04, decimals=2, space="protection", optimize=True, load=True)

    # LowProfitPairs    
    use_lowprofit_protection = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    lowprofit_protection_lookback = IntParameter(1, 24, default=10, space="protection", optimize=True, load=True)
    lowprofit_trade_limit = IntParameter(1, 10, default=6, space="protection", optimize=True, load=True)
    lowprofit_stop_duration = IntParameter(1, 70, default=65, space="protection", optimize=True, load=True)
    lowprofit_required_profit = DecimalParameter(-0.10, 0.00, default=-0.04, decimals=2, space="protection", optimize=True, load=True)
    lowprofit_only_per_pair = BooleanParameter(default=True, space="protection", optimize=True, load=True)

    # MaxDrawdown    
    use_maxdrawdown_protection = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    maxdrawdown_protection_lookback = IntParameter(1, 10, default=6, space="protection", optimize=True, load=True)
    maxdrawdown_trade_limit = IntParameter(1, 20, default=10, space="protection", optimize=True, load=True)
    maxdrawdown_stop_duration = IntParameter(1, 40, default=6, space="protection", optimize=True, load=True)
    maxdrawdown_allowed_drawdown = DecimalParameter(-0.10, 0.00, default=-0.04, decimals=2, space="protection", optimize=True, load=True)

    # Custom Entry
    increment = DecimalParameter(low=1.0005, high=1.002, default=1.001, decimals=4 ,space='buy', optimize=True, load=True)
    entryX = DecimalParameter(low=0.995, high=1.01, default=1.00, decimals=3 ,space='buy', optimize=True, load=True)
    last_entry_price = None

    # Position Management
    base_trades = IntParameter(4, 10, default=6, space="protection", optimize=True)
    stepSize = IntParameter(1, 4, default=2, space="protection", optimize=True)

    def custom_entry_price(self, pair: str, trade: Optional['Trade'], current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)

        entry_price = ((dataframe['close'].iat[-1] + dataframe['open'].iat[-1] + proposed_rate + proposed_rate) / 4) * self.entryX.value
        if entry_price >= proposed_rate:
            entry_price = proposed_rate
        if (self.dp.runmode.value in ('live', 'dry_run')):
            logger.info(f"{pair} Using Entry Price: {entry_price} | close: {dataframe['close'].iat[-1]} open: {dataframe['open'].iat[-1]} proposed_rate: {proposed_rate}")
            self.dp.send_msg(f"{pair} Using Entry Price: {entry_price} | close: {dataframe['close'].iat[-1]} open: {dataframe['open'].iat[-1]} proposed_rate: {proposed_rate}")

        # Check if there is a stored last entry price and if it matches the proposed entry price
        if self.last_entry_price is not None and abs(entry_price - self.last_entry_price) < 0.0001:  # Tolerance for floating-point comparison
            entry_price *= self.increment.value # Increment by 0.2%%
            if (self.dp.runmode.value in ('live', 'dry_run')):
                logger.info(f"{pair} Incremented entry price: {entry_price} based on previous entry price : {self.last_entry_price}.")

        # Update the last entry price
        self.last_entry_price = entry_price

        # Define balance thresholds and stake multipliers
        # EDIT These to match your funding level <--- !!!!!
        # balance = self.wallets.get_total_stake_amount()
        # max_balance = 40000
        # if balance > 10000:
        #     max_open_trades_at_max_balance = self.base_trades.value + int((max_balance - 10000) / 10000) * self.stepSize.value
        #     self.config['max_open_trades'] = min(self.base_trades.value + int((balance - 10000) / 10000) * self.stepSize.value, max_open_trades_at_max_balance)
        # else:
        #     self.config['max_open_trades'] = self.base_trades.value

        return entry_price

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
                "trade_limit": self.stop_protection_trade_limit.value,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": self.stop_protection_only_per_pair.value,
                "required_profit": self.stop_protection_required_profit.value,
            "only_per_side": self.stop_protection_only_per_side.value
        })

        if self.use_lowprofit_protection.value:
            prot.append({
                    "method": "LowProfitPairs",
                    "lookback_period_candles": self.lowprofit_protection_lookback.value,
                    "trade_limit": self.lowprofit_trade_limit.value,
                    "stop_duration_candles": self.lowprofit_stop_duration.value,
                    "required_profit": self.lowprofit_required_profit.value,
                    "only_per_pair": self.lowprofit_only_per_pair.value
        })

        if self.use_maxdrawdown_protection.value:
            prot.append({
                    "method": "MaxDrawdown",
                    "lookback_period_candles": self.maxdrawdown_protection_lookback.value,
                    "trade_limit": self.maxdrawdown_trade_limit.value,
                    "stop_duration_candles": self.maxdrawdown_stop_duration.value,
                    "max_allowed_drawdown": self.maxdrawdown_allowed_drawdown.value
        })

        return prot    

    # Plot configuration for the UI
    plot_config = {
        "main_plot": {
            "ma": {"color": "blue"},
            "gradient": {"color": "orange", "type": "scatter"},
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        heikinashi = qtpylib.heikinashi(dataframe)

        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # Initialize Hurst Cycles for startup errors
        cycle_period = 80
        harmonics = [0, 0, 0]
        harmonics[0] = 40
        harmonics[1] = 27
        harmonics[2] = 20

        if len(dataframe) < self.u_window_size.value:
            raise ValueError(f"Insufficient data points for FFT: {len(dataframe)}. Need at least {self.u_window_size.value} data points.")

        # Perform FFT to identify cycles with a rolling window
        freq, power = perform_fft(dataframe['ha_close'], window_size=self.u_window_size.value)

        if len(freq) == 0 or len(power) == 0:
            raise ValueError("FFT resulted in zero or invalid frequencies. Check the data or the FFT implementation.")

        # Filter out the zero-frequency component and limit the frequency to below 500
        positive_mask = (1 / freq > self.l_window_size.value) & (1 / freq < self.u_window_size.value)
        positive_freqs = freq[positive_mask]
        positive_power = power[positive_mask]

        # Check if there are valid frequencies after filtering
        if len(positive_power) == 0:
            raise ValueError("No positive frequencies meet the filtering criteria. Adjust window sizes or check the data.")

        # Convert frequencies to periods
        cycle_periods = 1 / positive_freqs
        print(cycle_periods)
        # Set a threshold to filter out insignificant cycles based on power
        power_threshold = 0 if len(positive_power) == 0 else 0.01 * np.max(positive_power)
        significant_indices = positive_power > power_threshold
        significant_periods = cycle_periods[significant_indices]
        significant_power = positive_power[significant_indices]


        # Identify the dominant cycle
        dominant_freq_index = np.argmax(significant_power)
        dominant_freq = positive_freqs[dominant_freq_index]
        # logger.info(f'{pair} Hurst Exponent: {dominant_freq}')
        cycle_period = int(np.abs(1 / dominant_freq)) if dominant_freq != 0 else 100

        if cycle_period == np.inf:
            raise ValueError("No dominant frequency found. Check the data or the method used.")

        # Calculate harmonics for the dominant cycle
        harmonics = [cycle_period / (i + 1) for i in range(1, 4)]
        # print(cycle_period, harmonics)
        self.cp = int(cycle_period)
        self.h0 = int(harmonics[0])
        self.h1 = int(harmonics[1])
        self.h2 = int(harmonics[2])

        dataframe['Cycle'] = dataframe['ha_close'].ewm(span=int(cycle_period)).mean()
        dataframe['H0'] = dataframe['ha_close'].ewm(span=int(harmonics[0])).mean()
        dataframe['H1'] = dataframe['ha_close'].ewm(span=int(harmonics[1])).mean()
        dataframe['H2'] = dataframe['ha_close'].ewm(span=int(harmonics[2])).mean()

        # Apply rolling window operation to the 'OHLC4' column
        rolling_windowc = dataframe['ha_close'].rolling(cycle_period) 
        rolling_windowh0 = dataframe['ha_close'].rolling(int(harmonics[0]))
        rolling_windowh1 = dataframe['ha_close'].rolling(int(harmonics[1])) 
        rolling_windowh2 = dataframe['ha_close'].rolling(int(harmonics[2])) 

        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_valuec = rolling_windowc.apply(lambda x: np.ptp(x))
        ptp_valueh0 = rolling_windowh0.apply(lambda x: np.ptp(x))
        ptp_valueh1 = rolling_windowh1.apply(lambda x: np.ptp(x))
        ptp_valueh2 = rolling_windowh2.apply(lambda x: np.ptp(x))

        # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['cycle_move'] = ptp_valuec / dataframe['ha_close']
        dataframe['h0_move'] = ptp_valueh0 / dataframe['ha_close']
        dataframe['h1_move'] = ptp_valueh1 / dataframe['ha_close']
        dataframe['h2_move'] = ptp_valueh2 / dataframe['ha_close']

        dataframe['cycle_move_mean'] = dataframe['cycle_move'].rolling(self.cp).mean()        
        dataframe['h0_move_mean'] = dataframe['h0_move'].rolling(self.cp).mean()
        dataframe['h1_move_mean'] = dataframe['h1_move'].rolling(self.cp).mean() 
        dataframe['h2_move_mean'] = dataframe['h2_move'].rolling(self.cp).mean()

        # Add EMA3 for each cycle period
        dataframe['ema3Cycle'] = ta.EMA(dataframe['Cycle'], timeperiod=3)
        dataframe['ema3H0'] = ta.EMA(dataframe['H0'], timeperiod=3)
        dataframe['ema3H1'] = ta.EMA(dataframe['H1'], timeperiod=3)
        dataframe['ema3H2'] = ta.EMA(dataframe['H2'], timeperiod=3)


        dataframe['selllimit'] = self.selllimit.value
        # Call TGMA as a class method
        dataframe = self.calculate_tgma(dataframe, "H2", self.h2)
        dataframe = self.calculate_tgma(dataframe, "H1", self.h1)
        dataframe = self.calculate_tgma(dataframe, "H0", self.h0)
        dataframe = self.calculate_tgma(dataframe, "Cycle", self.cp)

        dataframe['Composite'] = (dataframe['gradientH2'] + dataframe['gradientH1'] + dataframe['gradientH0'] + dataframe['gradientCycle'])/4
        dataframe['Comp_signal'] = ta.SMA(dataframe['Composite'], 4)

        dataframe['minh2'], dataframe['maxh2'] = calculate_minima_maxima(dataframe, self.h2)
        dataframe['minh1'], dataframe['maxh1'] = calculate_minima_maxima(dataframe, self.h1)
        dataframe['minh0'], dataframe['maxh0'] = calculate_minima_maxima(dataframe, self.h0)
        dataframe['mincp'], dataframe['maxcp'] = calculate_minima_maxima(dataframe, self.cp)

        return dataframe

    def calculate_tgma(self, dataframe: DataFrame, avg_str: str, length: int) -> DataFrame:
        # Parameters
        ma_length = int(length/3)
        steps = length  # Max gradient steps

        # Initialize Gradient Strength
        qty_adv_dec = np.zeros(len(dataframe))

        # Calculate Gradient
        for i in range(1, len(dataframe)):
            if np.isnan(dataframe[f"{avg_str}"].iloc[i]) or np.isnan(dataframe[f"ema3{avg_str}"].iloc[i]):
                qty_adv_dec[i] = qty_adv_dec[i - 1]
                continue

            chg = dataframe[f"{avg_str}"].iloc[i] - dataframe[f"{avg_str}"].iloc[i - 1]
            is_bull = dataframe[f"{avg_str}"].iloc[i] > dataframe[f"ema3{avg_str}"].iloc[i]
            is_bear = dataframe[f"{avg_str}"].iloc[i] < dataframe[f"ema3{avg_str}"].iloc[i]

            if is_bull:
                qty_adv_dec[i] = qty_adv_dec[i - 1] + 1 if chg > 0 else qty_adv_dec[i - 1] - 1
            elif is_bear:
                qty_adv_dec[i] = qty_adv_dec[i - 1] - 1 if chg < 0 else qty_adv_dec[i - 1] + 1

            qty_adv_dec[i] = max(1, min(steps, qty_adv_dec[i]))

        # Normalize the gradient values
        gradient_normalized = qty_adv_dec
        dataframe[f"gradient{avg_str}"] = gradient_normalized
        dataframe[f'buylimit{avg_str}'] = dataframe[f"gradient{avg_str}"].shift(2)
        
        # Calculate gradient impulse
        dataframe['gradientImpulse'] = (abs(dataframe[f'buylimit{avg_str}'].shift() - 
                                        dataframe[f"gradient{avg_str}"].shift()) - 
                                      abs(dataframe[f'buylimit{avg_str}'].shift(2) - 
                                        dataframe[f"gradient{avg_str}"].shift(2)))
        
        # Assign color mapping
        dataframe[f"color{avg_str}"] = np.where(
            dataframe[f"{avg_str}"] > dataframe[f"ema3{avg_str}"],
            gradient_normalized,
            -gradient_normalized,
        )


        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define the buy conditions.
        """

        # steps = self.h2  # Max Gradient Steps

        dataframe.loc[
            # (dataframe["ma"] > dataframe["close"]) & 
            # (dataframe["gradient"].shift() > dataframe["ema3"]) &  # Bullish MA crossing above EMA
            (dataframe["gradientH2"] < dataframe['selllimit']) &
            (qtpylib.crossed_above(dataframe["gradientH2"], dataframe['buylimitH2'])), #&
            # (dataframe["gradient"] > (steps/2)),  # Strong gradient
            "buy"
        ] = 1

        # df.loc[
        #     (
        #         (df['ha_trend_cp'] < self.suppress_cp.value) &
        #         (df['ha_trend_cp'] > df['ha_trend_lower_cp']) &
        #         (df['ha_trend_cp'].shift() < df['ha_trend_lower_cp'].shift()) &
        #         (df['volume'] > 0)   # Make sure Volume is not 0
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'ha trend cp')

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define the sell conditions.
        """

        steps = self.h2  # Max Gradient Steps

        dataframe.loc[
            # (dataframe["gradientImpulse"] > 1) &  # Bearish MA crossing below EMA
            (qtpylib.crossed_below(dataframe["gradientH2"], dataframe['buylimitH2'])),# &
            # (dataframe["gradient"] < (steps/2)),  # Weak gradient
            "sell"
        ] = 1

        return dataframe

def perform_fft(price_data, window_size=None):
    if window_size is not None:
        # Apply rolling window to smooth the data
        price_data = price_data.rolling(window=window_size, center=True).mean().dropna()

    normalized_data = (price_data - np.mean(price_data)) / np.std(price_data)
    n = len(normalized_data)
    fft_data = np.fft.fft(normalized_data)
    freq = np.fft.fftfreq(n)
    power = np.abs(fft_data) ** 2
    power[np.isinf(power)] = 0
    return freq, power

def calculate_minima_maxima(df, window):
    if df is None or df.empty:
        return np.zeros(0), np.zeros(0)  # Return empty arrays instead of None

    minima = np.zeros(len(df))
    maxima = np.zeros(len(df))

    for i in range(window, len(df)):  # Ensure index does not go out of bounds
        window_data = df['ha_close'].iloc[i - window:i + 1]

        if df['ha_close'].iloc[i] == window_data.min() and (window_data == df['ha_close'].iloc[i]).sum() == 1:
            minima[i] = -window
        if df['ha_close'].iloc[i] == window_data.max() and (window_data == df['ha_close'].iloc[i]).sum() == 1:
            maxima[i] = window

    return minima, maxima
