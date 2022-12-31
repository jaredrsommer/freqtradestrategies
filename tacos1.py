from pandas import DataFrame
from functools import reduce
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter)

#Start Strategy
class tacos1(IStrategy):

    minimal_roi = {
        "0":  0.10
    }
    stoploss = -0.03
    timeframe = '6h'
    

    ### hyper-opt parameters ###
    # entry optizimation
    max_epa = CategoricalParameter([-1, 0, 1, 3, 5, 10], default=1, space="buy", optimize=True)

    # protections
    cooldown_lookback = IntParameter(2, 48, default=5, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # indicators
    buy_ema_long = IntParameter(5, 15, default=5)
    sell_ema_long = IntParameter(5, 30, default=5)


    ### entry opt.
    @property
    def max_entry_position_adjustment(self):
        return self.max_epa.value


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
                "trade_limit": 4,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False
            })

        return prot


    ### indicators ###
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate all indicators used by the strategy"""

        # Heikin Ashi Strategy
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']



        # Calculate all ema_long values
        for val in self.buy_ema_long.range:
            dataframe[f'sma_ha_close{val}'] = ta.SMA(dataframe['ha_close'], timeperiod=val)
        for val in self.sell_ema_long.range:
            dataframe[f'sma_ha_open{val}'] = ta.SMA(dataframe['ha_open'], timeperiod=val)

        return dataframe


    ### buy logic ###
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(qtpylib.crossed_above(
                dataframe[f'ema_bshort_{self.buy_ema_short.value}'], dataframe[f'ema_blong_{self.buy_ema_long.value}']
            ))

        # Check that volume is not 0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1
        return dataframe


    ### sell logic ###
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(qtpylib.crossed_above(
                dataframe[f'ema_slong_{self.sell_ema_long.value}'], dataframe[f'ema_sshort_{self.sell_ema_short.value}']
            ))

        # Check that volume is not 0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1
        return dataframe


#### 2022-11-05 22:31:07,841 - freqtrade.optimize.hyperopt - INFO - Hyperopting with data from 2022-08-01 00:00:00 up to 2022-10-31 00:00:00 (91 days)..

# Best result:

#    756/3000:    111 trades. 43/4/64 Wins/Draws/Losses. Avg profit   2.29%. Median profit  -2.17%. Total profit 262.30014003 USDT (  26.23%). Avg duration 5 days, 9:24:00 min. Objective: -262.30014


#     # Buy hyperspace params:
#     buy_params = {
#         "buy_ema_long": 16,
#         "buy_ema_short": 10,
#         "max_epa": 0,
#     }

#     # Sell hyperspace params:
#     sell_params = {
#         "sell_ema_long": 25,
#         "sell_ema_short": 15,
#     }

#     # Protection hyperspace params:
#     protection_params = {
#         "cooldown_lookback": 5,  # value loaded from strategy
#         "stop_duration": 5,  # value loaded from strategy
#         "use_stop_protection": True,  # value loaded from strategy
#     }

#     # ROI table:
#     minimal_roi = {
#         "0": 0.663,
#         "2874": 0.288,
#         "7052": 0.068,
#         "13423": 0
#     }

#     # Stoploss:
#     stoploss = -0.322

#     # Trailing stop:
#     trailing_stop = False  # value loaded from strategy
#     trailing_stop_positive = None  # value loaded from strategy
#     trailing_stop_positive_offset = 0.0  # value loaded from strategy
#     trailing_only_offset_is_reached = False  # value loaded from strategy
