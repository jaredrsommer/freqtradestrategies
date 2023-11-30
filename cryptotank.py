from pandas import DataFrame
from datetime import datetime, timezone
from typing import Optional
from functools import reduce
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter)
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade

class cryptotank(IStrategy):

    use_custom_stoploss = True
    trailing_stop = True
    ignore_roi_if_entry_signal = True
    use_exit_signal = True
    minimal_roi = {
        "0":  0.10
    }
    # DCA settings
    exit_profit_only = True
    position_adjustment_enable = True
    max_entry_position_adjustment = 0
    max_dca_multiplier = 1
    stoploss = -0.25
    timeframe = '1h'

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30 

    ### HYPER-OPT PARAMETERS ###

    # protections
    cooldown_lookback = IntParameter(2, 48, default=5, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # entry optimization
    max_epa = CategoricalParameter([0, 1], default=1, space="buy", optimize=True)

    # indicators
    reference_ma_length = IntParameter(180, 400, default=200, space="buy" ,optimize=True)
    smoothing_length = IntParameter(5, 50, default=30, space="buy", optimize=True)
    max_length = CategoricalParameter([24, 48, 72, 96, 144, 192, 240], default=48, space="buy", optimize=False)

    #trailing stop loss optimiziation
    tsl_target5 = DecimalParameter(low=0.2, high=0.4, decimals=1, default=0.3, space='sell', optimize=True, load=True)
    ts5 = DecimalParameter(low=0.04, high=0.06, default=0.05, space='sell', optimize=True, load=True)
    tsl_target4 = DecimalParameter(low=0.18, high=0.3, default=0.2, space='sell', optimize=True, load=True)
    ts4 = DecimalParameter(low=0.03, high=0.05, default=0.045, space='sell', optimize=True, load=True)
    tsl_target3 = DecimalParameter(low=0.10, high=0.15, default=0.15, space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.07, high=0.12, default=0.1, space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.05, high=0.06, default=0.07, space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.02, high=0.05, default=0.03, space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.008, high=0.015, default=0.013, space='sell', optimize=True, load=True)


    buy_ma_slope = DecimalParameter(-0.10, 0.10, default=-0.35, decimals=2, space="buy", optimize=True)

    sell_ma_slope = DecimalParameter(-0.10, 0.10, default=0.35, decimals=2, space="sell", optimize=True)
    avg = IntParameter(30, 100, default=50, space="buy", optimize=True)
    shift = IntParameter(3, 15, default=5, space="buy", optimize=True)

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
                "trade_limit": 1,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": True
            })

        return prot
        

    @property
    def max_entry_position_adjustment(self):
        return self.max_epa.value


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

        if current_profit > 0.1 and current_profit < 0.15 and trade.nr_of_successful_exits == 0:
            # Take 50% of the profit at +5%
            return -(trade.stake_amount / 2)



        if current_profit > -0.075 and trade.nr_of_successful_entries == 1:
            return None

        if current_profit > -0.05 and trade.nr_of_successful_entries == 2:
            return None

        if current_profit > -0.6 and trade.nr_of_successful_entries == 3:
            return None

        # # Obtain pair dataframe (just to show how to access it)
        # dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        # # Only buy when not actively falling price.
        # last_candle = dataframe.iloc[-1].squeeze()
        # previous_candle = dataframe.iloc[-2].squeeze()
        # if last_candle['close'] < previous_candle['close']:
        #     return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        # Allow up to 3 additional increasingly larger buys (4 in total)
        # Initial buy is 1x
        # If that falls to -5% profit, we buy 1.25x more, average profit should increase to roughly -2.2%
        # If that falls down to -5% again, we buy 1.5x more
        # If that falls once again down to -5%, we buy 1.75x more
        # Total stake for this trade would be 1 + 1.25 + 1.5 + 1.75 = 5.5x of the initial allowed stake.
        # Total stake for this trade would be 1 + 1.5 + 2 + 2.5 = 5.5x of the initial allowed stake.
        # That is why max_dca_multiplier is 5.5
        # Hope you have a deep wallet!
        try:
            # This returns first order stake size
            stake_amount = filled_entries[0].cost
            # This then calculates current safety order size
            stake_amount = stake_amount * (1 + (count_of_entries * 0.5))
            return stake_amount
        except Exception as exception:
            return None

        return None

    ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:


        for stop5 in self.tsl_target5.range:
            if (current_profit > stop5):
                for stop5a in self.ts5.range:
                    return stop5a 
        for stop4 in self.tsl_target4.range:
            if (current_profit > stop4):
                for stop4a in self.ts4.range:
                    return stop4a 
        for stop3 in self.tsl_target3.range:
            if (current_profit > stop3):
                for stop3a in self.ts3.range:
                    return stop3a 
        for stop2 in self.tsl_target2.range:
            if (current_profit > stop2):
                for stop2a in self.ts2.range:
                    return stop2a 
        for stop1 in self.tsl_target1.range:
            if (current_profit > stop1):
                for stop1a in self.ts1.range:
                    return stop1a 
        for stop0 in self.tsl_target0.range:
            if (current_profit > stop0):
                for stop0a in self.ts0.range:
                    return stop0a 

        return self.stoploss


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate all indicators used by the strategy"""

        # RSI
        # Calculate all rsi_buy values
        # for valb in self.buy_rsi_length.range:
        #     dataframe[f'buy_rsi_{valb}'] = ta.RSI(dataframe, timeperiod=valb)

        # # Calculate all rsi_buy ma values
        # for valbm in self.buy_rsi_ma_length.range:
        #     dataframe[f'buy_rsi_ma_{valbm}'] = ta.SMA(dataframe[f'buy_rsi_{valb}'], timeperiod=valbm)

        # dataframe['buy_rsi_slope'] = pta.momentum.slope(dataframe[f'buy_rsi_{valb}'])
        # dataframe['buy_rsi_ma_slope'] = pta.momentum.slope(dataframe[f'buy_rsi_ma_{valbm}'])
        # dataframe['sell_rsi_slope'] = pta.momentum.slope(dataframe[f'buy_rsi_{valb}'])
        # dataframe['sell_rsi_ma_slope'] = pta.momentum.slope(dataframe[f'buy_rsi_ma_{valbm}'])

        # % from reference MA
        for valma in self.reference_ma_length.range:
            dataframe[f'reference_ma_{valma}'] = ta.SMA(dataframe['close'], timeperiod=valma)
        # distance from reference ma to current close
        dataframe['change'] = ((dataframe['close'] - dataframe[f'reference_ma_{valma}']) / dataframe['close']) * 100
        
        for valsma in self.smoothing_length.range:
            dataframe[f'smooth_change_{valsma}'] = ta.SMA(dataframe['change'], timeperiod=valsma)

        dataframe['smooth_ma_slope'] = pta.momentum.slope(dataframe[f'smooth_change_{valsma}'])
        # 300 Candle Rolling Min-Max
        for l in self.max_length.range:
            dataframe['min'] = dataframe['smooth_ma_slope'].rolling(l).min()
            dataframe['max'] = dataframe['smooth_ma_slope'].rolling(l).max()

        dataframe['avg_slope_min'] = dataframe['smooth_ma_slope'].shift(self.shift.value).rolling(self.avg.value).min()
        dataframe['avg_slope_max'] = dataframe['smooth_ma_slope'].shift(self.shift.value).rolling(self.avg.value).max()
        dataframe['avg_slope_mean'] = dataframe['avg_slope_max'] + (dataframe['avg_slope_min'])

        return dataframe


    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # conditions = []

        # ### RSI ###
        # # conditions.append(dataframe[f'buy_rsi_{self.buy_rsi_length.value}'] > dataframe[f'buy_rsi_ma_{self.buy_rsi_ma_length.value}'])

        # # conditions.append((dataframe['buy_rsi_slope'] > self.buy_rsi_slope.value))
        # # conditions.append((dataframe['buy_rsi_ma_slope'] > self.buy_rsi_ma_slope.value))
        # # conditions.append((dataframe[f'buy_rsi_{self.buy_rsi_length.value}'] <= self.buy_rsi_upper.value))
        # # conditions.append((dataframe[f'buy_rsi_{self.buy_rsi_length.value}'] >= self.buy_rsi_lower.value))
       
        # # Reference MA
        # conditions.append(qtpylib.crossed_above(dataframe['change'], dataframe[f'smooth_change_{self.smoothing_length.value}']))
        # # conditions.append(qtpylib.crossed_above(dataframe['change'], dataframe['smooth_change_8']))
        # # conditions.append(dataframe[f'smooth_change_{self.smoothing_length.value}'] < self.buy_ma_upper.value)
        # # conditions.append(dataframe[f'smooth_change_{self.smoothing_length.value}'] > self.buy_ma_lower.value)
        # conditions.append(dataframe['smooth_ma_slope'] > self.buy_ma_slope.value)


        # # Check that volume is not 0
        # conditions.append(dataframe['volume'] > 0)

        df.loc[
            (
                # (qtpylib.crossed_above(df['change'], df[f'smooth_change_{self.smoothing_length.value}'])) &
                (df['min'] < self.buy_ma_slope.value) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'SLOPE')


        return df


    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # conditions = []

        # ### RSI ###
        # # conditions.append(qtpylib.crossed_above(
        # #         dataframe[f'buy_rsi_ma_{self.buy_rsi_ma_length.value}'], dataframe[f'buy_rsi_{self.buy_rsi_length.value}']
        # #     ))

        # # conditions.append((dataframe['buy_rsi_ma_slope'] <= self.buy_rsi_ma_slope.value))
        # # conditions.append((dataframe[f'buy_rsi_{self.buy_rsi_length.value}'] <= self.sell_rsi_upper.value))
        # # conditions.append((dataframe[f'buy_rsi_{self.buy_rsi_length.value}'] >= self.sell_rsi_lower.value))

        # # Reference MA
        # conditions.append(dataframe['change'] < dataframe[f'smooth_change_{self.smoothing_length.value}'])
        # # conditions.append(dataframe[f'smooth_change_{self.smoothing_length.value}'] > self.sell_ma_upper.value)
        # # conditions.append(dataframe[f'smooth_change_{self.smoothing_length.value}'] < self.sell_ma_lower.value)
        # # conditions.append(dataframe['smooth_ma_slope'] < self.buy_ma_slope.value)


        # # Check that volume is not 0
        # conditions.append(dataframe['volume'] > 0)

        # if conditions:
        #     dataframe.loc[
        #         reduce(lambda x, y: x & y, conditions),
        #         'exit_long'] = 1

        df.loc[
            (
                (df['max'] > self.sell_ma_slope.value) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'SLOPE SELL')
        return df


# 2022-11-18 20:07:53,863 - freqtrade.optimize.hyperopt - INFO - Hyperopting with data from 2020-06-07 12:00:00 up to 2022-11-17 06:00:00 (892 days)..
# 2022-11-18 20:07:53,899 - freqtrade.optimize.hyperopt - INFO - Found 4 CPU cores. Let's make them scream!
# 2022-11-18 20:07:53,899 - freqtrade.optimize.hyperopt - INFO - Number of parallel jobs set as: -1
# 2022-11-18 20:07:53,899 - freqtrade.optimize.hyperopt - INFO - Using estimator ET.
# 2022-11-18 20:07:53,913 - freqtrade.optimize.hyperopt - INFO - Effective number of parallel workers used: 4
# +--------+-----------+----------+------------------+--------------+-------------------------------+-----------------+-------------+-------------------------------+
# |   Best |     Epoch |   Trades |    Win Draw Loss |   Avg profit |                        Profit |    Avg duration |   Objective |           Max Drawdown (Acct) |
# |--------+-----------+----------+------------------+--------------+-------------------------------+-----------------+-------------+-------------------------------|
# | * Best |    1/2500 |      488 |    199   10  279 |       -0.05% |       -42.993 USDT   (-4.30%) | 2 days 01:25:00 |     42.9931 |       420.687 USDT   (30.71%) |
# | * Best |   11/2500 |       10 |      2    0    8 |       -2.12% |       -21.071 USDT   (-2.11%) | 1 days 05:24:00 |     21.0714 |        20.961 USDT    (2.10%) |                             
# | * Best |   13/2500 |      283 |    128   76   79 |        0.78% |       186.769 USDT   (18.68%) | 6 days 03:57:00 |    -186.769 |       690.132 USDT   (36.77%) |
# |   Best |   40/2500 |      448 |    232  141   75 |        1.15% |       252.707 USDT   (25.27%) | 11 days 02:03:00 |    -252.707 |      2510.625 USDT   (66.71%) |                            
# |   Best |   50/2500 |      287 |    141  105   41 |        1.21% |       338.977 USDT   (33.90%) | 8 days 09:13:00 |    -338.977 |       650.687 USDT   (34.61%) |                             
# |   Best |   70/2500 |      408 |    251  106   51 |        1.66% |       763.326 USDT   (76.33%) | 8 days 00:26:00 |    -763.326 |      1506.866 USDT   (46.08%) |                             
# |   Best |  133/2500 |      462 |    243  156   63 |        1.61% |       766.485 USDT   (76.65%) | 7 days 23:32:00 |    -766.485 |      1787.819 USDT   (50.30%) |                             
# |   Best |  167/2500 |      331 |    180   86   65 |        2.67% |      1121.223 USDT  (112.12%) | 8 days 19:30:00 | -1,121.22292 |      1073.657 USDT   (33.61%) |                            
# |   Best | 1759/2500 |      422 |    246  133   43 |        1.95% |      1125.524 USDT  (112.55%) | 6 days 22:17:00 | -1,125.52434 |      1021.432 USDT   (37.11%) |                            
#  [Epoch 2500 of 2500 (100%)] ||                                                                                                                       | [Time: 11:05:55, Elapsed Time: 11:05:55]
# 2022-11-19 07:14:16,216 - freqtrade.optimize.hyperopt - INFO - 2500 epochs saved to '/home/jared/freqtrade/user_data/hyperopt_results/strategy_cryptotank_2022-11-18_20-07-53.fthypt'.
# 2022-11-19 07:14:16,225 - freqtrade.resolvers.iresolver - WARNING - Could not import /home/jared/freqtrade/user_data/strategies/NASOSv5.py due to 'name 'TrailingBuySellStrat' is not defined'
# 2022-11-19 07:14:16,227 - freqtrade.optimize.hyperopt_tools - INFO - Dumping parameters to /home/jared/freqtrade/user_data/strategies/cryptotank.json

# Best result:

#   1759/2500:    422 trades. 246/133/43 Wins/Draws/Losses. Avg profit   1.95%. Median profit   4.94%. Total profit 1125.52433814 USDT ( 112.55%). Avg duration 6 days, 22:17:00 min. Objective: -1125.52434


#     # Buy hyperspace params:
#     buy_params = {
#         "buy_ma_slope": -2,
#         "buy_rsi_length": 8,
#         "buy_rsi_ma_length": 30,
#         "max_epa": 10,
#         "reference_ma_length": 194,
#         "smoothing_length": 16,
#     }

#     # Sell hyperspace params:
#     sell_params = {
#         "sell_ma_slope": -5,
#     }

#     # Protection hyperspace params:
#     protection_params = {
#         "cooldown_lookback": 21,
#         "stop_duration": 165,
#         "use_stop_protection": False,
#     }

#     # ROI table:
#     minimal_roi = {
#         "0": 0.401,
#         "2140": 0.279,
#         "5877": 0.128,
#         "6949": 0
#     }

#     # Stoploss:
#     stoploss = -0.35

#     # Trailing stop:
#     trailing_stop = True
#     trailing_stop_positive = 0.015
#     trailing_stop_positive_offset = 0.086
#     trailing_only_offset_is_reached = True

# 1hr
# 2022-11-19 19:05:08,509 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using minimal_roi: {'0': 0.401, '2140': 0.279, '5877': 0.128, '6949': 0}
# 2022-11-19 19:05:08,509 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using timeframe: 1h
# 2022-11-19 19:05:08,509 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using stoploss: -0.35
# 2022-11-19 19:05:08,509 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_stop: True
# 2022-11-19 19:05:08,509 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_stop_positive: 0.015
# 2022-11-19 19:05:08,509 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_stop_positive_offset: 0.086
# 2022-11-19 19:05:08,509 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_only_offset_is_reached: True
# 2022-11-19 19:05:08,509 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using use_custom_stoploss: False
# 2022-11-19 19:05:08,509 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using process_only_new_candles: True
# 2022-11-19 19:05:08,509 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using order_types: {'entry': 'limit', 'exit': 'limit', 'stoploss': 'limit', 'stoploss_on_exchange': False, 'stoploss_on_exchange_interval': 60}
# 2022-11-19 19:05:08,509 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using order_time_in_force: {'entry': 'GTC', 'exit': 'GTC'}
# 2022-11-19 19:05:08,509 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using stake_currency: USDT
# 2022-11-19 19:05:08,509 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using stake_amount: unlimited
# 2022-11-19 19:05:08,509 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using protections: [{'method': 'CooldownPeriod', 'stop_duration_candles': 5}, {'method': 'StoplossGuard', 'lookback_period_candles': 72, 'trade_limit': 1, 'stop_duration_candles': 5, 'only_per_pair': True}]
# 2022-11-19 19:05:08,509 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using startup_candle_count: 30
# 2022-11-19 19:05:08,509 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using unfilledtimeout: {'entry': 10, 'exit': 10, 'exit_timeout_count': 0, 'unit': 'minutes'}
# 2022-11-19 19:05:08,510 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using use_exit_signal: True
# 2022-11-19 19:05:08,510 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using exit_profit_only: False
# 2022-11-19 19:05:08,510 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using ignore_roi_if_entry_signal: False
# 2022-11-19 19:05:08,510 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using exit_profit_offset: 0.0
# 2022-11-19 19:05:08,510 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using disable_dataframe_checks: False
# 2022-11-19 19:05:08,510 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using ignore_buying_expired_candle_after: 0
# 2022-11-19 19:05:08,510 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using position_adjustment_enable: False
# 2022-11-19 19:05:08,510 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using max_entry_position_adjustment: 1
# 2022-11-19 19:05:08,510 - freqtrade.configuration.config_validation - INFO - Validating configuration ...
# 2022-11-19 19:05:08,518 - freqtrade.resolvers.iresolver - INFO - Using resolved pairlist StaticPairList from '/home/jared/freqtrade/freqtrade/plugins/pairlist/StaticPairList.py'...
# 2022-11-19 19:05:08,549 - freqtrade.strategy.hyper - INFO - Strategy Parameter: buy_ma_slope = -2
# 2022-11-19 19:05:08,549 - freqtrade.strategy.hyper - INFO - Strategy Parameter: buy_rsi_length = 8
# 2022-11-19 19:05:08,549 - freqtrade.strategy.hyper - INFO - Strategy Parameter: buy_rsi_ma_length = 30
# 2022-11-19 19:05:08,549 - freqtrade.strategy.hyper - INFO - Strategy Parameter: max_epa = 10
# 2022-11-19 19:05:08,549 - freqtrade.strategy.hyper - INFO - Strategy Parameter: reference_ma_length = 194
# 2022-11-19 19:05:08,549 - freqtrade.strategy.hyper - INFO - Strategy Parameter: smoothing_length = 16
# 2022-11-19 19:05:08,550 - freqtrade.strategy.hyper - INFO - Strategy Parameter: sell_ma_slope = -5
# 2022-11-19 19:05:08,550 - freqtrade.strategy.hyper - INFO - Strategy Parameter: cooldown_lookback = 21
# 2022-11-19 19:05:08,550 - freqtrade.strategy.hyper - INFO - Strategy Parameter: stop_duration = 165
# 2022-11-19 19:05:08,550 - freqtrade.strategy.hyper - INFO - Strategy Parameter: use_stop_protection = False
# 2022-11-19 19:05:08,551 - freqtrade.resolvers.iresolver - INFO - Using resolved hyperoptloss OnlyProfitHyperOptLoss from '/home/jared/freqtrade/freqtrade/optimize/hyperopt_loss/hyperopt_loss_onlyprofit.py'...
# 2022-11-19 19:05:08,552 - freqtrade.optimize.hyperopt - INFO - Removing `/home/jared/freqtrade/user_data/hyperopt_results/hyperopt_tickerdata.pkl`.
# 2022-11-19 19:05:08,554 - freqtrade.optimize.hyperopt - INFO - Using optimizer random state: 14557
# 2022-11-19 19:05:08,563 - freqtrade.optimize.hyperopt_interface - INFO - Min roi table: {0: 0.069, 120: 0.046, 240: 0.023, 360: 0}
# 2022-11-19 19:05:08,563 - freqtrade.optimize.hyperopt_interface - INFO - Max roi table: {0: 0.711, 480: 0.252, 1200: 0.092, 2640: 0}
# 2022-11-19 19:05:08,576 - freqtrade.data.history.history_utils - INFO - Using indicator startup period: 30 ...
# 2022-11-19 19:05:08,724 - freqtrade.data.history.idatahandler - INFO - Price jump in XRP/USDT, 1h, spot between two candles of 19.29% detected.
# 2022-11-19 19:05:08,835 - freqtrade.data.history.idatahandler - INFO - Price jump in AKT/USDT, 1h, spot between two candles of 44.13% detected.
# 2022-11-19 19:05:09,017 - freqtrade.data.history.idatahandler - INFO - Price jump in IOTA/USDT, 1h, spot between two candles of 47.10% detected.
# 2022-11-19 19:05:09,341 - freqtrade.optimize.backtesting - INFO - Loading data from 2020-06-02 00:00:00 up to 2022-11-19 23:00:00 (900 days).
# 2022-11-19 19:05:09,341 - freqtrade.configuration.timerange - WARNING - Moving start-date by 30 candles to account for startup time.
# 2022-11-19 19:05:09,341 - freqtrade.optimize.hyperopt - INFO - Dataload complete. Calculating indicators
# 2022-11-19 19:05:09,709 - freqtrade.optimize.hyperopt - INFO - Hyperopting with data from 2020-06-03 06:00:00 up to 2022-11-19 23:00:00 (899 days)..
# 2022-11-19 19:05:09,808 - freqtrade.optimize.hyperopt - INFO - Found 4 CPU cores. Let's make them scream!
# 2022-11-19 19:05:09,809 - freqtrade.optimize.hyperopt - INFO - Number of parallel jobs set as: -1
# 2022-11-19 19:05:09,809 - freqtrade.optimize.hyperopt - INFO - Using estimator ET.
# 2022-11-19 19:05:09,821 - freqtrade.optimize.hyperopt - INFO - Effective number of parallel workers used: 4
# +--------+-----------+----------+------------------+--------------+-------------------------------+-----------------+-------------+-------------------------------+                             
# |   Best |     Epoch |   Trades |    Win Draw Loss |   Avg profit |                        Profit |    Avg duration |   Objective |           Max Drawdown (Acct) |
# |--------+-----------+----------+------------------+--------------+-------------------------------+-----------------+-------------+-------------------------------|
# | * Best |   17/2500 |     1468 |    634  752   82 |        0.18% |        -6.554 USDT   (-0.66%) | 3 days 05:43:00 |     6.55404 |      1369.568 USDT   (57.96%) |
# | * Best |   28/2500 |     1397 |    574  764   59 |        0.14% |        34.575 USDT    (3.46%) | 2 days 14:56:00 |     -34.575 |       925.861 USDT   (47.23%) |                             
# |   Best |   37/2500 |     1365 |    655  614   96 |        0.28% |       106.656 USDT   (10.67%) | 3 days 20:01:00 |    -106.656 |      1738.455 USDT   (61.10%) |                             
# |   Best |   46/2500 |     1279 |    589  594   96 |        0.34% |       127.483 USDT   (12.75%) | 4 days 00:33:00 |    -127.483 |      2328.349 USDT   (67.37%) |                             
# |   Best |   49/2500 |     1567 |    726  736  105 |        0.34% |       282.992 USDT   (28.30%) | 3 days 04:11:00 |    -282.992 |      2016.114 USDT   (62.97%) |                             
# |   Best |   56/2500 |     1368 |    640  637   91 |        0.43% |       368.814 USDT   (36.88%) | 3 days 16:25:00 |    -368.814 |      2176.995 USDT   (63.34%) |                             
# |   Best |   77/2500 |     1689 |    795  796   98 |        0.43% |       624.177 USDT   (62.42%) | 3 days 05:13:00 |    -624.177 |      1904.064 USDT   (57.60%) |                             
# |   Best |   93/2500 |     1671 |    749  811  111 |        0.49% |       697.160 USDT   (69.72%) | 3 days 03:41:00 |     -697.16 |      2144.834 USDT   (61.69%) |                             
# |   Best |  150/2500 |     1468 |    734  627  107 |        0.72% |      1012.366 USDT  (101.24%) | 3 days 21:30:00 | -1,012.36640 |      3244.120 USDT   (63.93%) |                            
#  [Epoch 1403 of 2500 ( 56%)] |███████████████████████████████████████████████████████████████-                                                  | [ETA:  19:34:31, Elapsed Time: 1 day, 1:02:09]^C
# User interrupted..
# 2022-11-20 20:16:28,729 - freqtrade.optimize.hyperopt - INFO - 1404 epochs saved to '/home/jared/freqtrade/user_data/hyperopt_results/strategy_cryptotank_2022-11-19_19-05-08.fthypt'.
# 2022-11-20 20:16:28,793 - freqtrade.resolvers.iresolver - WARNING - Could not import /home/jared/freqtrade/user_data/strategies/NASOSv5.py due to 'name 'TrailingBuySellStrat' is not defined'
# 2022-11-20 20:16:28,804 - freqtrade.optimize.hyperopt_tools - INFO - Dumping parameters to /home/jared/freqtrade/user_data/strategies/cryptotank.json

# Best result:

#    150/2500:   1468 trades. 734/627/107 Wins/Draws/Losses. Avg profit   0.72%. Median profit   0.01%. Total profit 1012.36640305 USDT ( 101.24%). Avg duration 3 days, 21:30:00 min. Objective: -1012.36640


#     # Buy hyperspace params:
#     buy_params = {
#         "buy_ma_slope": -6,
#         "buy_rsi_length": 12,
#         "buy_rsi_ma_length": 22,
#         "max_epa": -1,
#         "reference_ma_length": 195,
#         "smoothing_length": 15,
#     }

#     # Sell hyperspace params:
#     sell_params = {
#         "sell_ma_slope": 10,
#     }

#     # Protection hyperspace params:
#     protection_params = {
#         "cooldown_lookback": 4,
#         "stop_duration": 114,
#         "use_stop_protection": False,
#     }

#     # ROI table:
#     minimal_roi = {
#         "0": 0.62,
#         "265": 0.162,
#         "795": 0.08,
#         "1860": 0
#     }

#     # Stoploss:
#     stoploss = -0.322

#     # Trailing stop:
#     trailing_stop = True
#     trailing_stop_positive = 0.048
#     trailing_stop_positive_offset = 0.147
#     trailing_only_offset_is_reached = True
    

# Result for strategy cryptotank 1h
# ============================================================= BACKTESTING REPORT ============================================================
# |      Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |-----------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# |  ETH/USDT |       242 |           0.75 |         182.22 |           190.880 |          19.09 |  3 days, 9:56:00 |   126   103    13  52.1 |
# | ATOM/USDT |       255 |           0.59 |         149.88 |           170.598 |          17.06 |  3 days, 9:06:00 |   123   109    23  48.2 |
# |  NGM/USDT |        85 |          -0.33 |         -28.24 |           144.391 |          14.44 |  5 days, 9:44:00 |    39    36    10  45.9 |
# |  XRP/USDT |       192 |           0.81 |         155.89 |           135.732 |          13.57 | 4 days, 11:54:00 |   100    72    20  52.1 |
# |  ADA/USDT |       231 |           0.63 |         145.39 |            65.337 |           6.53 | 3 days, 17:38:00 |   110   103    18  47.6 |
# | CSPR/USDT |        24 |           0.08 |           2.03 |             5.567 |           0.56 | 4 days, 17:40:00 |    11    11     2  45.8 |
# |  AKT/USDT |        50 |           0.80 |          40.01 |             3.095 |           0.31 | 3 days, 11:42:00 |    24    22     4  48.0 |
# | SCRT/USDT |        26 |          -0.78 |         -20.30 |           -41.853 |          -4.19 |  5 days, 5:18:00 |    13    10     3  50.0 |
# |  QNT/USDT |       185 |           0.27 |          50.38 |           -56.080 |          -5.61 |  3 days, 7:37:00 |    89    76    20  48.1 |
# |  BTC/USDT |       147 |           0.10 |          14.90 |          -156.347 |         -15.63 | 5 days, 22:08:00 |    68    71     8  46.3 |
# | IOTA/USDT |        79 |          -0.83 |         -65.91 |          -224.184 |         -22.42 | 3 days, 19:59:00 |    37    34     8  46.8 |
# |     TOTAL |      1516 |           0.41 |         626.25 |           237.136 |          23.71 |  4 days, 0:19:00 |   740   647   129  48.8 |
# =========================================================== ENTER TAG STATS ============================================================
# |   TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |
# |-------+-----------+----------------+----------------+-------------------+----------------+-----------------+-------------------------|
# | TOTAL |      1516 |           0.41 |         626.25 |           237.136 |          23.71 | 4 days, 0:19:00 |   740   647   129  48.8 |
# ======================================================= EXIT REASON STATS ========================================================
# |        Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
# |--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
# |                roi |    1338 |    691   647     0   100 |           3.02 |        4041.3  |          8864.29  |         404.13 |
# |          stop_loss |     119 |      0     0   119     0 |         -32.33 |       -3847.57 |         -9814.92  |        -384.76 |
# | trailing_stop_loss |      49 |     49     0     0   100 |          11.66 |         571.39 |          1387.65  |          57.14 |
# |         force_exit |      10 |      0     0    10     0 |         -13.89 |        -138.86 |          -199.888 |         -13.89 |
# ========================================================== LEFT OPEN TRADES REPORT ==========================================================
# |      Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |-----------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# |  XRP/USDT |         1 |          -0.39 |          -0.39 |            -0.561 |          -0.06 |          1:00:00 |     0     0     1     0 |
# |  QNT/USDT |         1 |          -2.60 |          -2.60 |            -3.697 |          -0.37 |  2 days, 4:00:00 |     0     0     1     0 |
# |  ETH/USDT |         1 |          -2.85 |          -2.85 |            -4.046 |          -0.40 | 3 days, 18:00:00 |     0     0     1     0 |
# |  BTC/USDT |         1 |          -2.89 |          -2.89 |            -4.393 |          -0.44 |  9 days, 5:00:00 |     0     0     1     0 |
# | CSPR/USDT |         1 |         -15.50 |         -15.50 |            -8.309 |          -0.83 |  7 days, 0:00:00 |     0     0     1     0 |
# | ATOM/USDT |         1 |         -17.03 |         -17.03 |           -25.254 |          -2.53 |  9 days, 9:00:00 |     0     0     1     0 |
# |  NGM/USDT |         1 |         -26.54 |         -26.54 |           -33.079 |          -3.31 | 7 days, 23:00:00 |     0     0     1     0 |
# | IOTA/USDT |         1 |         -18.90 |         -18.90 |           -34.939 |          -3.49 | 14 days, 6:00:00 |     0     0     1     0 |
# |  AKT/USDT |         1 |         -28.40 |         -28.40 |           -41.443 |          -4.14 | 9 days, 19:00:00 |     0     0     1     0 |
# |  ADA/USDT |         1 |         -23.76 |         -23.76 |           -44.167 |          -4.42 | 14 days, 5:00:00 |     0     0     1     0 |
# |     TOTAL |        10 |         -13.89 |        -138.86 |          -199.888 |         -19.99 | 7 days, 18:36:00 |     0     0    10     0 |
# ================== SUMMARY METRICS ==================
# | Metric                      | Value               |
# |-----------------------------+---------------------|
# | Backtesting from            | 2020-06-03 06:00:00 |
# | Backtesting to              | 2022-11-19 23:00:00 |
# | Max open trades             | 10                  |
# |                             |                     |
# | Total/Daily Avg Trades      | 1516 / 1.69         |
# | Starting balance            | 1000 USDT           |
# | Final balance               | 1237.136 USDT       |
# | Absolute profit             | 237.136 USDT        |
# | Total profit %              | 23.71%              |
# | CAGR %                      | 9.02%               |
# | Profit factor               | 1.02                |
# | Trades per day              | 1.69                |
# | Avg. daily profit %         | 0.03%               |
# | Avg. stake amount           | 223.956 USDT        |
# | Total trade volume          | 339516.594 USDT     |
# |                             |                     |
# | Best Pair                   | ETH/USDT 182.22%    |
# | Worst Pair                  | IOTA/USDT -65.91%   |
# | Best trade                  | NGM/USDT 21.40%     |
# | Worst trade                 | QNT/USDT -32.34%    |
# | Best day                    | 148.838 USDT        |
# | Worst day                   | -606.623 USDT       |
# | Days win/draw/lose          | 388 / 432 / 71      |
# | Avg. Duration Winners       | 23:03:00            |
# | Avg. Duration Loser         | 14 days, 3:30:00    |
# | Rejected Entry signals      | 3136                |
# | Entry/Exit Timeouts         | 0 / 0               |
# |                             |                     |
# | Min balance                 | 1000 USDT           |
# | Max balance                 | 4132.33 USDT        |
# | Max % of account underwater | 70.06%              |
# | Absolute Drawdown (Account) | 70.06%              |
# | Absolute Drawdown           | 2895.194 USDT       |
# | Drawdown high               | 3132.33 USDT        |
# | Drawdown low                | 237.136 USDT        |
# | Drawdown Start              | 2021-09-19 15:00:00 |
# | Drawdown End                | 2022-11-19 23:00:00 |
# | Market change               | 87.77%              |
# =====================================================

# Result for strategy cryptotank 6hr hyper opt
# ============================================================= BACKTESTING REPORT ============================================================
# |      Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |-----------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# |  QNT/USDT |       111 |           3.08 |         341.82 |          1077.560 |         107.76 | 4 days, 18:32:00 |    71    26    14  64.0 |
# |  NGM/USDT |        61 |           1.40 |          85.18 |           542.060 |          54.21 | 6 days, 18:41:00 |    34    17    10  55.7 |
# | CSPR/USDT |        14 |           5.22 |          73.10 |           329.822 |          32.98 |  2 days, 6:26:00 |    10     3     1  71.4 |
# | ATOM/USDT |       141 |           1.71 |         241.40 |           327.495 |          32.75 |  5 days, 9:24:00 |    89    34    18  63.1 |
# |  AKT/USDT |        22 |           4.20 |          92.43 |           296.019 |          29.60 |  5 days, 7:22:00 |    13     7     2  59.1 |
# | SCRT/USDT |        13 |           9.37 |         121.78 |           262.748 |          26.27 |  8 days, 0:28:00 |     9     2     2  69.2 |
# |  ETH/USDT |       119 |           1.72 |         204.63 |           181.520 |          18.15 |  6 days, 7:13:00 |    70    38    11  58.8 |
# |  XRP/USDT |       120 |           1.92 |         230.45 |           118.476 |          11.85 |  6 days, 6:12:00 |    76    30    14  63.3 |
# |  BTC/USDT |        77 |           1.32 |         101.48 |            -4.060 |          -0.41 | 10 days, 1:05:00 |    40    32     5  51.9 |
# | IOTA/USDT |        29 |          -0.71 |         -20.72 |          -151.159 |         -15.12 |  8 days, 8:17:00 |    13    11     5  44.8 |
# |  ADA/USDT |       115 |           0.71 |          81.50 |          -476.224 |         -47.62 | 6 days, 14:24:00 |    67    32    16  58.3 |
# |     TOTAL |       822 |           1.89 |        1553.06 |          2504.258 |         250.43 |  6 days, 8:45:00 |   492   232    98  59.9 |
# =========================================================== ENTER TAG STATS ============================================================
# |   TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |
# |-------+-----------+----------------+----------------+-------------------+----------------+-----------------+-------------------------|
# | TOTAL |       822 |           1.89 |        1553.06 |          2504.258 |         250.43 | 6 days, 8:45:00 |   492   232    98  59.9 |
# ======================================================= EXIT REASON STATS ========================================================
# |        Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
# |--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
# | trailing_stop_loss |     425 |    425     0     0   100 |           9.86 |        4192.48 |         13481.3   |         419.25 |
# |                roi |     299 |     67   232     0   100 |           1.38 |         411.24 |          1376.22  |          41.12 |
# |          stop_loss |      70 |      0     0    70     0 |         -35.13 |       -2458.94 |        -10063.5   |        -245.89 |
# |        exit_signal |      19 |      0     0    19     0 |         -25.23 |        -479.29 |         -1847.9   |         -47.93 |
# |         force_exit |       9 |      0     0     9     0 |         -12.49 |        -112.43 |          -441.865 |         -11.24 |
# ========================================================== LEFT OPEN TRADES REPORT ===========================================================
# |      Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |      Avg Duration |   Win  Draw  Loss  Win% |
# |-----------+-----------+----------------+----------------+-------------------+----------------+-------------------+-------------------------|
# |  BTC/USDT |         1 |          -0.66 |          -0.66 |            -2.495 |          -0.25 |   2 days, 0:00:00 |     0     0     1     0 |
# |  QNT/USDT |         1 |          -5.04 |          -5.04 |           -19.138 |          -1.91 |   2 days, 0:00:00 |     0     0     1     0 |
# |  NGM/USDT |         1 |          -5.82 |          -5.82 |           -22.088 |          -2.21 |   1 day, 12:00:00 |     0     0     1     0 |
# |  ETH/USDT |         1 |          -6.22 |          -6.22 |           -24.753 |          -2.48 |   5 days, 6:00:00 |     0     0     1     0 |
# | IOTA/USDT |         1 |          -8.04 |          -8.04 |           -32.457 |          -3.25 |   6 days, 6:00:00 |     0     0     1     0 |
# | SCRT/USDT |         1 |         -23.05 |         -23.05 |           -54.834 |          -5.48 |          18:00:00 |     0     0     1     0 |
# | ATOM/USDT |         1 |         -15.74 |         -15.74 |           -63.552 |          -6.36 |  5 days, 18:00:00 |     0     0     1     0 |
# | CSPR/USDT |         1 |         -24.15 |         -24.15 |           -97.543 |          -9.75 |  5 days, 18:00:00 |     0     0     1     0 |
# |  ADA/USDT |         1 |         -23.71 |         -23.71 |          -125.005 |         -12.50 | 11 days, 12:00:00 |     0     0     1     0 |
# |     TOTAL |         9 |         -12.49 |        -112.43 |          -441.865 |         -44.19 |  4 days, 12:40:00 |     0     0     9     0 |
# ================== SUMMARY METRICS ==================
# | Metric                      | Value               |
# |-----------------------------+---------------------|
# | Backtesting from            | 2020-06-07 12:00:00 |
# | Backtesting to              | 2022-11-17 06:00:00 |
# | Max open trades             | 10                  |
# |                             |                     |
# | Total/Daily Avg Trades      | 822 / 0.92          |
# | Starting balance            | 1000 USDT           |
# | Final balance               | 3504.258 USDT       |
# | Absolute profit             | 2504.258 USDT       |
# | Total profit %              | 250.43%             |
# | CAGR %                      | 67.05%              |
# | Profit factor               | 1.20                |
# | Trades per day              | 0.92                |
# | Avg. daily profit %         | 0.28%               |
# | Avg. stake amount           | 339.454 USDT        |
# | Total trade volume          | 279030.824 USDT     |
# |                             |                     |
# | Best Pair                   | QNT/USDT 341.82%    |
# | Worst Pair                  | IOTA/USDT -20.72%   |
# | Best trade                  | SCRT/USDT 40.06%    |
# | Worst trade                 | ATOM/USDT -35.13%   |
# | Best day                    | 258.701 USDT        |
# | Worst day                   | -962.339 USDT       |
# | Days win/draw/lose          | 293 / 501 / 55      |
# | Avg. Duration Winners       | 1 day, 19:02:00     |
# | Avg. Duration Loser         | 17 days, 4:10:00    |
# | Rejected Entry signals      | 366                 |
# | Entry/Exit Timeouts         | 0 / 0               |
# |                             |                     |
# | Min balance                 | 1008.734 USDT       |
# | Max balance                 | 6201.622 USDT       |
# | Max % of account underwater | 59.95%              |
# | Absolute Drawdown (Account) | 59.95%              |
# | Absolute Drawdown           | 3717.959 USDT       |
# | Drawdown high               | 5201.622 USDT       |
# | Drawdown low                | 1483.663 USDT       |
# | Drawdown Start              | 2021-09-19 18:00:00 |
# | Drawdown End                | 2022-06-15 06:00:00 |
# | Market change               | 92.48%              |
# =====================================================
# 1hr 1-1 - 4-9
# Best result:

#     73/100:    117 trades. 111/0/6 Wins/Draws/Losses. Avg profit   2.85%. Median profit   2.67%. Total profit 1384.06462591 USDT ( 138.41%). Avg duration 1 day, 23:30:00 min. Objective: -1384.06463


#     # Buy hyperspace params:
#     buy_params = {
#         "buy_ma_slope": -9,
#         "buy_rsi_length": 9,
#         "buy_rsi_ma_length": 9,
#         "max_epa": 3,
#         "reference_ma_length": 185,
#         "smoothing_length": 7,
#     }

#     # Sell hyperspace params:
#     sell_params = {
#         "sell_ma_slope": -9,
#         "ts0": 0.009,
#         "ts1": 0.01,
#         "ts2": 0.016,
#         "ts3": 0.028,
#         "ts4": 0.04,
#         "ts5": 0.054,
#         "tsl_target0": 0.049,
#         "tsl_target1": 0.053,
#         "tsl_target2": 0.072,
#         "tsl_target3": 0.138,
#         "tsl_target4": 0.287,
#         "tsl_target5": 0.4,
#     }

#     # Protection hyperspace params:
#     protection_params = {
#         "cooldown_lookback": 5,  # value loaded from strategy
#         "stop_duration": 5,  # value loaded from strategy
#         "use_stop_protection": True,  # value loaded from strategy
#     }

#     # ROI table:  # value loaded from strategy
#     minimal_roi = {
#         "0": 0.1
#     }

#     # Stoploss:
#     stoploss = -0.5  # value loaded from strategy

#     # Trailing stop:
#     trailing_stop = False  # value loaded from strategy
#     trailing_stop_positive = None  # value loaded from strategy
#     trailing_stop_positive_offset = 0.0  # value loaded from strategy
#     trailing_only_offset_is_reached = False  # value loaded from strategy
