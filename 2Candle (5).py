from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, informative
from pandas import DataFrame
import numpy as np
import talib.abstract as ta
from technical import qtpylib
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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

class Candle2(IStrategy):
    # Strategy parameters
    timeframe = "1h"  
    minimal_roi = {}  
    locked_stoploss = {}
    stoploss = -0.04  

    # Trailing stop:
    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = 0.025 # value loaded from strategy
    trailing_stop_positive_offset = 0.10 # value loaded from strategy
    trailing_only_offset_is_reached = False  # value loaded from strategy
    use_custom_stoploss = True
    position_adjustment_enable = True

    buy_threshold = DecimalParameter(3.0, 5.0, default=4.0, decimals=1, space="buy")
    rsi_threshold = DecimalParameter(40.0, 70.0, default=65.0, decimals=1, space="buy")
    bull_rsi_limit = DecimalParameter(65.0, 85.0, default=70.0, decimals=1, space="buy")
    bull_4h_threshold = DecimalParameter(40.0, 70.0, default=60.0, decimals=1, space="buy")
    bear_rsi_threshold = DecimalParameter(40.0, 70.0, default=40.0, decimals=1, space="sell")
    sell_threshold = DecimalParameter(3.0, 4.0, default=4.0, decimals=1, space="sell")
    u_window_size = IntParameter(60, 90, default=90, space='buy', optimize=True, load=True)
    l_window_size = IntParameter(20, 40, default=30, space='buy', optimize=True, load=True)
    sr_length = IntParameter(12, 50, default=24, space="buy")
    sr_shift = IntParameter(12, 50, default=24, space="buy")
    fast_rsi = IntParameter(4, 20, default=12, space="buy")
    block_exit = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use_tsl1 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use_tsl2 = BooleanParameter(default=True, space="sell", optimize=True, load=True)

    use_1 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use_2 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use_3 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use_4 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use_5 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use_6 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use_7 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use_8 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use_9 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use_10 = BooleanParameter(default=True, space="buy", optimize=True, load=True)

    use_11 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use_12 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use_13 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    # use_14 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use_15 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use_16 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use_17 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use_18 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use_19 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use_20 = BooleanParameter(default=True, space="buy", optimize=True, load=True)

    plot_config = {
        "main_plot": {
            "resistance": {
              "color": "#c05967",
              "type": "line"
            },
            "support": {
              "color": "#7ebd8d",
              "type": "line"
            },
            "CO2_4h": {
              "color": "#f9c37d"
            },
            "inflection": {
              "color": "#f7af68",
              "type": "line"
            },
            "enter_tag": {
              "color": "#200da0",
              "type": "line"
            },
            "exit_tag": {
              "color": "#f45b5d"
            }
        },
        "subplots": {
            "pattern": {
              "pattern": {
                "color": "#5adb55"
              },
              "pattern_4h": {
                "color": "#4b5b8d",
                "type": "line"
              },
              "pattern_avg": {
                "color": "#fdc703",
                "type": "line"
              },
              "pattern_CO2": {
                "color": "#2c6a7c"
              }
            },
            "rsi4": {
              "rsi_4h": {
                "color": "#555eae"
              },
              "rsi": {
                "color": "#7f9463"
              }
            },
            "bb": {
              "bull_bear": {
                "color": "#77f66d"
              }
            },
            "cc": {
              "close_comparison_4h": {
                "color": "#ed90e7",
                "type": "line",
                "fill_to": "close_comparison_4h"
              },
              "close_c": {
                "color": "#51a1f7",
                "type": "bar"
              }
            }
          }
    }

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if self.can_short == True:
            lev = self.lev_X.value
            lev = 3
        else:
            lev = 1

        SLT0 = current_candle['h2_move_mean'] * lev
        SLT1 = current_candle['h1_move_mean'] * lev
        SLT2 = current_candle['h0_move_mean'] * lev
        SLT3 = current_candle['cycle_move_mean'] * lev
        
        display_profit = current_profit * 100

        if current_profit < -0.01:
            if pair in self.locked_stoploss:
                del self.locked_stoploss[pair]
                if (self.dp.runmode.value in ('live', 'dry_run')):
                    self.dp.send_msg(f'*** {pair} *** Stoploss reset.')
                    logger.info(f'*** {pair} *** Stoploss reset.')
            return self.stoploss

        new_stoploss = None
        if SLT3 is not None and current_profit > SLT3:
            new_stoploss = (SLT2 - SLT1)
            level = 4
        elif SLT2 is not None and current_profit > SLT2:
            new_stoploss = (SLT2 - SLT1)
            level = 3
        elif SLT1 is not None and current_profit > SLT1 and self.use_tsl2.value == True:
            new_stoploss = (SLT1 - SLT0)
            level = 2
        elif SLT0 is not None and current_profit > SLT0 and self.use_tsl1.value == True:
            new_stoploss = (SLT1 - SLT0)
            level = 1

        if new_stoploss is not None:
            if pair not in self.locked_stoploss or new_stoploss > self.locked_stoploss[pair]:
                self.locked_stoploss[pair] = new_stoploss
                if (self.dp.runmode.value in ('live', 'dry_run')):
                    self.dp.send_msg(f'*** {pair} *** Profit {level} {display_profit:.3f}%% - New stoploss: {new_stoploss:.4f} activated')
                    logger.info(f'*** {pair} *** Profit {level} {display_profit:.3f}%% - New stoploss: {new_stoploss:.4f} activated')
            return self.locked_stoploss[pair]

        return self.stoploss

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        # filled_entries = trade.select_filled_orders(trade.entry_side)
        # count_of_entries = trade.nr_of_successful_entries
        # trade_duration = (current_time - trade.open_date_utc).seconds / 60
        # last_fill = (current_time - trade.date_last_filled_utc).seconds / 60 

        current_candle = dataframe.iloc[-1].squeeze()

        STOP = current_candle['move_mean_min']
        TP0 = current_candle['h2_move_mean'] 
        TP1 = current_candle['h1_move_mean'] 
        TP2 = current_candle['h0_move_mean'] 
        TP3 = current_candle['cycle_move_mean']
        display_profit = current_profit * 100
        Stop = STOP * 100
        tp0 = TP0 * 100
        tp1 = TP1 * 100
        tp2 = TP2 * 100
        tp3 = TP3 * 100

        if current_profit is not None:
            if (self.dp.runmode.value in ('live', 'dry_run')):
                logger.info(f"{trade.pair} - 💰 Current Profit: {display_profit:.3}% BE: {Stop:.3}% | TP0: {tp0:.3}% | TP1: {tp1:.3}% | TP2: {tp2:.3}% | TP3: {tp3:.3}%")

        # Take Profit if m00n
        if current_profit > TP1 and trade.nr_of_successful_exits == 0:
            # Take quarter of the profit at next fib%%
            return -(trade.stake_amount / 4)
        if current_profit > TP2 and trade.nr_of_successful_exits == 1:
            # Take half of the profit at last fib%%
            return -(trade.stake_amount / 2)
        if current_profit > TP3 and trade.nr_of_successful_exits == 2:
            # Take half of the profit at last fib%%
            return -(trade.stake_amount / 2)
        if current_profit > (TP3 * 2.0) and trade.nr_of_successful_exits == 3:
            # Take profit at last fib%%
            return -(trade.stake_amount)
            
        # DCA
        # if trade.nr_of_successful_entries == self.max_epa.value + 1:
        #     return None 
        # # Block concurrent buys and Hold otherwise if the dip is not large enough
        # if current_profit > -TP1:
        #     return None

        # try:
        #     # This returns first order stake size 
        #     # Modify the following parameters to enable more levels or different buy size:
        #     # max_entry_position_adjustment = 3 
        #     # max_dca_multiplier = 3.5 

        #     stake_amount = filled_entries[0].cost
        #     # This then calculates current safety order size when a secondary buy signal is generated.
        #     if (last_fill > self.filldelay.value):
        #         if (signal == 1 and current_profit < -TP0):
        #             if count_of_entries >= 1: 
        #                 stake_amount = stake_amount * 2
        #             else:
        #                 stake_amount = stake_amount

        #             return stake_amount

        #     # This then calculates current safety order size when below -Take Profit 1.
        #     if (last_fill > self.filldelay.value):
        #         if current_profit < -TP0:
        #             if count_of_entries >= 1: 
        #                 stake_amount = stake_amount * 1.5
        #             else:
        #                 stake_amount = stake_amount

        #             return stake_amount

        #     # This accommadates a one shot at buying the dip on a big wick with one 
        #     # large buy if the funds are available...
        #     if (last_fill > self.filldelay.value):
        #         if (current_profit < -TP3):
        #             if count_of_entries == 1: 
        #                 stake_amount = stake_amount * 4
        #             else:
        #                 stake_amount = stake_amount

        #             return stake_amount

        # except Exception as exception:
        #     return None

        return None



    @informative('4h')
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate candle range (high - low)
        dataframe['range'] = dataframe['high'] - dataframe['low']
        dataframe['range_third'] = dataframe['range'] / 3

        # Close position: Determine if close is in upper, mid, or lower third
        dataframe['close_position'] = 0  # Default: mid
        dataframe.loc[dataframe['close'] > (dataframe['high'] - dataframe['range_third']), 'close_position'] = 1  # High close
        dataframe.loc[dataframe['close'] < (dataframe['low'] + dataframe['range_third']), 'close_position'] = -1  # Low close

        # Close comparison: Compare current close to previous candle's range
        dataframe['prev_high'] = dataframe['high'].shift(1)
        dataframe['prev_low'] = dataframe['low'].shift(1)
        dataframe['close_comparison'] = 0  # Default: range
        dataframe.loc[dataframe['close'] > dataframe['prev_high'], 'close_comparison'] = 1  # Bull candle
        dataframe.loc[dataframe['close'] < dataframe['prev_low'], 'close_comparison'] = -1  # Bear candle
        dataframe['CO2'] = ((dataframe['close'] - dataframe['open']) / 2) + dataframe['open']


        # Combine close position and close comparison into 9 patterns
        dataframe['pattern'] = dataframe['close_position'] * 3 + dataframe['close_comparison'] + 4  # Maps to 0-8 (9 patterns)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=6)
        return dataframe

    # @informative('2h')
    # def populate_indicators_2h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    #     # Calculate candle range (high - low)
    #     dataframe['range'] = dataframe['high'] - dataframe['low']
    #     dataframe['range_third'] = dataframe['range'] / 3

    #     # Close position: Determine if close is in upper, mid, or lower third
    #     dataframe['close_position'] = 0  # Default: mid
    #     dataframe.loc[dataframe['close'] > (dataframe['high'] - dataframe['range_third']), 'close_position'] = 1  # High close
    #     dataframe.loc[dataframe['close'] < (dataframe['low'] + dataframe['range_third']), 'close_position'] = -1  # Low close

    #     # Close comparison: Compare current close to previous candle's range
    #     dataframe['prev_high'] = dataframe['high'].shift(1)
    #     dataframe['prev_low'] = dataframe['low'].shift(1)
    #     dataframe['close_comparison'] = 0  # Default: range
    #     dataframe.loc[dataframe['close'] > dataframe['prev_high'], 'close_comparison'] = 1  # Bull candle
    #     dataframe.loc[dataframe['close'] < dataframe['prev_low'], 'close_comparison'] = -1  # Bear candle
    #     dataframe['CO2'] = ((dataframe['close'] - dataframe['open']) / 2) + dataframe['open']


    #     # Combine close position and close comparison into 9 patterns
    #     dataframe['pattern'] = dataframe['close_position'] * 3 + dataframe['close_comparison'] + 4  # Maps to 0-8 (9 patterns)
    #     dataframe['rsi'] = ta.RSI(dataframe, timeperiod=6)
    #     return dataframe

    # Define custom variables
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate candle range (high - low)
        dataframe['range'] = dataframe['high'] - dataframe['low']
        dataframe['range_third'] = dataframe['range'] / 3

        # Close position: Determine if close is in upper, mid, or lower third
        dataframe['close_position'] = 0  # Default: mid
        dataframe.loc[dataframe['close'] > (dataframe['high'] - dataframe['range_third']), 'close_position'] = 1  # High close
        dataframe.loc[dataframe['close'] < (dataframe['low'] + dataframe['range_third']), 'close_position'] = -1  # Low close

        # Close comparison: Compare current close to previous candle's range
        dataframe['prev_high'] = dataframe['high'].shift(1)
        dataframe['prev_low'] = dataframe['low'].shift(1)
        dataframe['close_comparison'] = 0  # Default: range
        dataframe.loc[dataframe['close'] > dataframe['prev_high'], 'close_comparison'] = 1  # Bull candle
        dataframe.loc[dataframe['close'] < dataframe['prev_low'], 'close_comparison'] = -1  # Bear candle

        # Combine close position and close comparison into 9 patterns
        dataframe['pattern'] = dataframe['close_position'] * 3 + dataframe['close_comparison'] + 4  # Maps to 0-8 (9 patterns)
        dataframe['pattern_avg'] = ((dataframe['pattern'] + dataframe['pattern_4h']) / 2).rolling(2).mean()
        dataframe['pattern_avg_std'] = (dataframe['close'] - dataframe['open'].rolling(4).mean()) / dataframe['open'].rolling(4).std()
        dataframe['range_4h_total'] = dataframe['high'].rolling(4).max() - dataframe['low'].rolling(4).min()
        # Simple support/resistance levels using rolling min/max
        dataframe['support'] = dataframe['low_4h'].rolling(window=self.sr_length.value).min().shift(self.sr_shift.value)
        dataframe['resistance'] = dataframe['high_4h'].rolling(window=self.sr_length.value).max().shift(self.sr_shift.value)

        # 4h S/R 
        dataframe['range_4h'] = dataframe['resistance'] - dataframe['support']
        dataframe['inflection'] = (dataframe['range_4h']/2) + dataframe['support']
        dataframe['range_third_4h'] = dataframe['range_4h'] / 3

        # Close position: Determine if close is in upper, mid, or lower third
        dataframe['CO2_position'] = 0  # Default: mid
        dataframe.loc[dataframe['CO2_4h'] > (dataframe['resistance'] - dataframe['range_third_4h']), 'CO2_position'] = 1  # High close
        dataframe.loc[dataframe['CO2_4h'] < (dataframe['support'] + dataframe['range_third_4h']), 'CO2_position'] = -1  # Low close

        # Close comparison: Compare current close to previous candle's range
        dataframe['prev_high_4h'] = dataframe['resistance'].shift(1)
        dataframe['prev_low_4h'] = dataframe['support'].shift(1)
        dataframe['close_c'] = 0
        dataframe.loc[dataframe['close_4h'] > dataframe['close_4h'].shift(4), 'close_c'] = 1  # Bull candle
        dataframe.loc[dataframe['close_4h'] < dataframe['close_4h'].shift(4), 'close_c'] = -1  # Bear candle
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.fast_rsi.value)

        # # Combine close position and close comparison into 9 patterns
        dataframe['pattern_CO2'] = dataframe['CO2_position'] * 3 + dataframe['close_c'] + 4  # Maps to 0-8 (9 patterns)
        dataframe['bull_bear'] = 0 
        dataframe.loc[dataframe['CO2_4h'] < dataframe['close'], 'bull_bear'] = 1  # High close
        dataframe.loc[dataframe['CO2_4h'] > dataframe['close'], 'bull_bear'] = -1  # Low close

        pair = metadata['pair']
        # Heikin-Ashi Candles
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # Initialize Hurst Cycles
        cycle_period = 80
        harmonics = [0, 0, 0]
        harmonics[0] = 40
        harmonics[1] = 27
        harmonics[2] = 20

        if len(dataframe) < self.u_window_size.value:
            raise ValueError(f"Insufficient data points for FFT: {len(dataframe)}. Need at least {self.u_window_size.value} data points.")

        # Perform FFT
        freq, power = perform_fft(dataframe['ha_close'], window_size=self.u_window_size.value)
        if len(freq) == 0 or len(power) == 0:
            raise ValueError("FFT resulted in zero or invalid frequencies.")

        positive_mask = (1 / freq > self.l_window_size.value) & (1 / freq < self.u_window_size.value)
        positive_freqs = freq[positive_mask]
        positive_power = power[positive_mask]
        if len(positive_power) == 0:
            raise ValueError("No positive frequencies meet the filtering criteria.")

        cycle_periods = 1 / positive_freqs
        power_threshold = 0 if len(positive_power) == 0 else 0.01 * np.max(positive_power)
        significant_indices = positive_power > power_threshold
        significant_periods = cycle_periods[significant_indices]
        significant_power = positive_power[significant_indices]

        dominant_freq_index = np.argmax(significant_power)
        dominant_freq = positive_freqs[dominant_freq_index]
        cycle_period = int(np.abs(1 / dominant_freq)) if dominant_freq != 0 else 100
        if cycle_period == np.inf:
            raise ValueError("No dominant frequency found.")

        harmonics = [cycle_period / (i + 1) for i in range(1, 4)]
        self.cp = int(cycle_period)
        self.h0 = int(harmonics[0])
        self.h1 = int(harmonics[1])
        self.h2 = int(harmonics[2])

        # EWMA for cycles
        dataframe['cp'] = dataframe['ha_close'].ewm(span=self.cp).mean()
        dataframe['h0'] = dataframe['ha_close'].ewm(span=self.h0).mean()
        dataframe['h1'] = dataframe['ha_close'].ewm(span=self.h1).mean()
        dataframe['h2'] = dataframe['ha_close'].ewm(span=self.h2).mean()

        # Peak-to-Peak Movement
        rolling_windowc = dataframe['ha_close'].rolling(self.cp)
        rolling_windowh0 = dataframe['ha_close'].rolling(self.h0)
        rolling_windowh1 = dataframe['ha_close'].rolling(self.h1)
        rolling_windowh2 = dataframe['ha_close'].rolling(self.h2)
        ptp_valuec = rolling_windowc.apply(lambda x: np.ptp(x))
        ptp_valueh0 = rolling_windowh0.apply(lambda x: np.ptp(x))
        ptp_valueh1 = rolling_windowh1.apply(lambda x: np.ptp(x))
        ptp_valueh2 = rolling_windowh2.apply(lambda x: np.ptp(x))
        dataframe['cycle_move'] = ptp_valuec / dataframe['ha_close']
        dataframe['h0_move'] = ptp_valueh0 / dataframe['ha_close']
        dataframe['h1_move'] = ptp_valueh1 / dataframe['ha_close']
        dataframe['h2_move'] = ptp_valueh2 / dataframe['ha_close']
        dataframe['move'] = (dataframe['cycle_move'] + dataframe['h0_move'] + dataframe['h1_move'] + dataframe['h2_move']) / 4 
        dataframe['cycle_move_mean'] = dataframe['cycle_move'].rolling(self.cp).mean()
        dataframe['h0_move_mean'] = dataframe['h0_move'].rolling(self.cp).mean()
        dataframe['h1_move_mean'] = dataframe['h1_move'].rolling(self.cp).mean()
        dataframe['h2_move_mean'] = dataframe['h2_move'].rolling(self.cp).mean()
        dataframe['move_mean'] = (dataframe['cycle_move_mean'] + dataframe['h0_move_mean'] + dataframe['h1_move_mean'] + dataframe['h2_move_mean']) / 4
        dataframe['move_mean_min'] = dataframe['h2_move_mean'].min() / 2
        dataframe['entry_limit'] = dataframe['ha_close'].rolling(self.cp).max() * (1 - dataframe['move_mean_min'])
        dataframe['entry_limit_lower'] = dataframe['ha_close'].rolling(self.cp).min() * (1 + dataframe['move_mean_min'])

        # timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
        # pair = metadata['pair'].replace('/', '_')  # Replace '/' with '_' for valid filename
        # filename = f"{pair}_{timestamp}.csv"
        # dataframe.to_csv(filename, index=True)
        # print(f"Exported DataFrame for {pair} to {filename}")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Buy conditions based on Two Candle Theory
        dataframe.loc[
            (
                (dataframe['pattern_avg'] >= self.buy_threshold.value) &
                (dataframe['pattern_avg'].shift() < 8) &
                (self.use_1.value == True) &
                (dataframe['rsi_4h'] < self.rsi_threshold.value) &
                (dataframe['rsi'] > dataframe['rsi_4h'])
            ),
            ['enter_long', 'enter_tag']] = (1, 'Pattern and RSI')

        dataframe.loc[
            (
                (dataframe['pattern_avg'] >= self.buy_threshold.value) &
                (dataframe['pattern_avg'].shift() < 8) &
                (self.use_2.value == True) &
                (dataframe['CO2_4h'] > dataframe['resistance']) &
                (dataframe['CO2_4h'].shift() < dataframe['resistance'].shift()) & 
                (dataframe['rsi_4h'] < self.rsi_threshold.value) 
            ),
            ['enter_long', 'enter_tag']] = (1, 'Resistance Cross')

        dataframe.loc[
            (
                (dataframe['pattern_avg'] >= self.buy_threshold.value) &
                (self.use_3.value == True) &
                (dataframe['CO2_4h'] > dataframe['support']) &
                (dataframe['CO2_4h'].shift() < dataframe['support'].shift()) 
            ),
            ['enter_long', 'enter_tag']] = (1, 'Support Cross')

        dataframe.loc[
            (
                # High close bull candle (pattern 4: most bullish)
                (dataframe['pattern_avg'] >= self.buy_threshold.value) &
                (self.use_4.value == True) &
                (dataframe['CO2_4h'] < dataframe['support']) &
                (dataframe['CO2_4h'].shift(4) < dataframe['CO2_4h']) &
                (dataframe['rsi'] > dataframe['rsi_4h'])

            ),
            ['enter_long', 'enter_tag']] = (1, 'Below Support 4h pull up')


        dataframe.loc[
            (
                (dataframe['CO2_4h'] < dataframe['support']) &
                (self.use_5.value == True) &
                (dataframe['rsi_4h'] < 10) &
                (dataframe['rsi'] > dataframe['rsi_4h'])
            ),
            ['enter_long', 'enter_tag']] = (1, '4h Extreme RSI')

        dataframe.loc[
            (
                (dataframe['rsi_4h'] > 50) &  # Proxy for uptrend
                (dataframe['pattern_4h'] >= 5) &  # Strong bull pattern on 4h
                (self.use_6.value == True) &
                (dataframe['close'] > dataframe['inflection']) &
                (dataframe['close'].shift() < dataframe['inflection']) &  # Cross above midpoint
                (dataframe['rsi'] > dataframe['rsi_4h'])  # Local RSI strength
            ),
            ['enter_long', 'enter_tag']] = (1, 'Uptrend Inflection Cross')

        dataframe.loc[
            (
                (dataframe['bull_bear'] == 1) &
                (dataframe['pattern_avg'] > self.buy_threshold.value - 1) &
                (self.use_7.value == True) &
                (dataframe['rsi_4h'] > self.bull_4h_threshold.value) &
                (dataframe['rsi'] < self.bull_rsi_limit.value) &
                (dataframe['close'] > dataframe['close'].shift())
            ),
            ['enter_long', 'enter_tag']] = (1, 'Trending Bull Entry')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['rsi_4h'] < 50) &  # Proxy for downtrend
                (dataframe['pattern_4h'] <= 3) &  # Strong bear pattern on 4h
                (self.use_11.value == True) &
                (dataframe['close'] < dataframe['inflection']) &
                (dataframe['close'].shift() > dataframe['inflection']) &  # Cross below midpoint
                (dataframe['rsi'] < dataframe['rsi_4h'])  # Local RSI weakness
            ),
            ['exit_long', 'exit_tag']] = (1, 'Downtrend Inflection Cross')

        dataframe.loc[
            (
                (dataframe['CO2_4h'] > dataframe['resistance']) &
                (dataframe['close'] > dataframe['resistance']) &
                (self.use_12.value == True) &
                (dataframe['close'] < dataframe['CO2_4h']) &
                (dataframe['rsi_4h'] > 85) 
            ),
            ['exit_long', 'exit_tag']] = (1, 'Above Resistance 4h')

        dataframe.loc[
            (
                (dataframe['bull_bear'] == -1) &
                (dataframe['pattern_avg'] < self.sell_threshold.value) &
                (self.use_13.value == True) &
                (dataframe['rsi_4h'] < self.bear_rsi_threshold.value) &
                (dataframe['close'] < dataframe['close'].shift())
            ),
            ['exit_long', 'exit_tag']] = (1, 'Trending Bear Exit')


        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                               rate: float, time_in_force: str, exit_reason: str,
                               current_time: datetime, **kwargs) -> bool:
            # Block exits in strong uptrend if profitable
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) == 0:
                return True
            last_candle = dataframe.iloc[-1].squeeze()
            if exit_reason in ['partial_exit', 'exit_signal'] and trade.calc_profit_ratio(rate) > 0 and self.block_exit.value:
                if last_candle['rsi_4h'] > self.rsi_threshold.value and last_candle['pattern_4h'] >= 5:
                    return False
            return True

def perform_fft(price_data, window_size=None):
    if window_size is not None:
        price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
    normalized_data = (price_data - np.mean(price_data)) / np.std(price_data)
    n = len(normalized_data)
    fft_data = np.fft.fft(normalized_data)
    freq = np.fft.fftfreq(n)
    power = np.abs(fft_data) ** 2
    power[np.isinf(power)] = 0
    return freq, power
