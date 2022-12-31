from pandas import DataFrame
from functools import reduce
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter)

#Start Strategy
class WTHO(IStrategy):

    minimal_roi = {
        "0":  0.10
    }
    stoploss = -0.025
    timeframe = '2h'
    

    ### hyper-opt parameters ###
    # entry optizimation
    max_epa = CategoricalParameter([-1, 0, 1, 3, 5, 10], default=1, space="buy", optimize=True)

    # protections
    cooldown_lookback = IntParameter(2, 48, default=5, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # indicators 
    buy_esa_length = IntParameter(3, 20, default=10, optimize=True)
    buy_d_length = IntParameter(3, 20, default=10, optimize=True)
    buy_wt1_length = IntParameter(15, 50, default=21, optimize=True)
    buy_wt2_length = IntParameter(3, 14, default=4, optimize=True)
    mfi_length = IntParameter(6, 60, default=40, optimize=True)
    mfi_sma_length = IntParameter(6, 60, default=15, optimize=True)
    sma_length = IntParameter(10, 50, default=25, optimize=True)

    # values
    # buy
    wt_os = IntParameter(-55, 1 , default=-50, space="buy", optimize=True)
    wt_os_sma = IntParameter(-30, 50 , default=0, space="buy", optimize=True)
    buy_wt_pc = IntParameter(-30, 10 , default=0, space="buy", optimize=True)
    buy_mfi = IntParameter(0, 60 , default=50, space="buy", optimize=True)
    buy_mfi_slope = IntParameter(-20, 20 , default=0, space="buy", optimize=True)
    buy_sma_pc = IntParameter(-10, 10 , default=0, space="buy", optimize=True)

    # sell
    wt_ob = IntParameter(20, 55 , default=50, space="sell", optimize=True)
    wt_ob_sma = IntParameter(-20, 55 , default=0, space="sell", optimize=True)
    sell_wt_pc = IntParameter(-10, 30 , default=0, space="sell", optimize=True)
    sell_mfi = IntParameter(40, 100 , default=70, space="sell", optimize=True)
    sell_mfi_slope = IntParameter(-20, 20 , default=0, space="sell", optimize=True)
    sell_sma_pc = IntParameter(-10, 10 , default=0, space="sell", optimize=True)


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

        ### WaveTrend using OHLC4 or HA close
        ap = (0.25 * (dataframe['high'] + dataframe['low'] + dataframe["close"] + dataframe["open"]))
        
        # find ema length for esa
        for vale in self.buy_esa_length.range:
            dataframe[f'esa_{vale}'] = ta.EMA(ap, timeperiod = vale)
       
        # find ema length for d       
        for vald in self.buy_d_length.range:
            dataframe[f'd_{vald}'] = ta.EMA(abs(ap - dataframe[f'esa_{vale}']), timeperiod = vald)


        dataframe['wave_ci'] = (ap-dataframe[f'esa_{vale}']) / (0.015 * dataframe[f'd_{vald}'])

        # find t1 optimimum length
        for valt1 in self.buy_wt1_length.range:
            dataframe[f'wave_t1_{valt1}'] = ta.EMA(dataframe['wave_ci'], timeperiod = valt1)

        # find t2 optimimum length
        for valt2 in self.buy_wt2_length.range:    
            dataframe[f'wave_t2_{valt2}'] = ta.SMA(dataframe[f'wave_t1_{valt1}'], timeperiod = valt2)
        
        dataframe["wave_t1_pc"] = round(
            (dataframe[f'wave_t1_{valt1}'] - dataframe[f'wave_t1_{valt1}'].shift()) / abs(dataframe[f'wave_t1_{valt1}']) * 100, 2)

        ### Money Flow Index
        for valmfi in self.mfi_length.range:
            dataframe[f'mfi_{valmfi}'] = ta.MFI(dataframe, timeperiod = valmfi)
        for valmfisma in self.mfi_sma_length.range:
            dataframe[f'mfi_sma{valmfisma}'] = ta.SMA(dataframe[f'mfi_{valmfi}'], timeperiod = valmfisma)

        dataframe['mfi_pc'] = round(
            (dataframe[f'mfi_sma{valmfisma}'] - dataframe[f'mfi_sma{valmfisma}'].shift()) / abs(dataframe[f'mfi_sma{valmfisma}']) * 100, 2)
 
        ### SMA Decision Monitor OS - OB Levels Decision Monitor

        # find optimum SMA length
        for valsma in self.sma_length.range:
                dataframe[f'sma{valsma}'] = ta.SMA(dataframe, timeperiod = valsma)

        dataframe[f"sma_{valsma}pc"] = round(
            (dataframe[f'sma{valsma}'] - dataframe[f'sma{valsma}'].shift()) / abs(dataframe[f'sma{valsma}']) * 100, 2)

        # if dataframe[f'sma_{self.sma_length.value}pc'] > self.buy_sma_pc.value:
        #     dataframe['buy_os'] = self.wt_os_sma.value
        # else:
        #     dataframe['buy_os']= self.wt_os.value 

        return dataframe


    ### buy logic ###
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # # Wave Trend
        # dataframe.loc[
        #     (
        #         (dataframe[f'sma_{self.sma_length.value}pc'] >= self.buy_sma_pc.value) &
        #         (dataframe[f'wave_t1_{self.buy_wt1_length.value}'] <= self.wt_os.value ) &
        #         (dataframe[f'wave_t1_{self.buy_wt1_length.value}'] > dataframe[f'wave_t2_{self.buy_wt2_length.value}']) &
        #         (dataframe['wave_t1_pc'] >= self.buy_wt_pc.value) &
        #         (dataframe[f'mfi_sma{self.mfi_sma_length.value}'] <= self.buy_mfi.value ) &
        #         (dataframe['mfi_pc'] >= self.buy_mfi_slope.value ) &
        #         (dataframe['volume'] > 0)

        #     ),
        #     ['buy', 'buy_tag']] = (1, 'Bear')

        # dataframe.loc[
        #     (
        #         (dataframe[f'sma_{self.sma_length.value}pc'] > self.buy_sma_pc.value) &
        #         (dataframe[f'wave_t1_{self.buy_wt1_length.value}'] <= self.wt_os_sma.value ) &
        #         (dataframe[f'wave_t1_{self.buy_wt1_length.value}'] > self.wt_os.value ) &
        #         (dataframe[f'wave_t1_{self.buy_wt1_length.value}'] > dataframe[f'wave_t2_{self.buy_wt2_length.value}']) &
        #         (dataframe['wave_t1_pc'] >= self.buy_wt_pc.value) &
        #         (dataframe[f'mfi_sma{self.mfi_sma_length.value}'] <= self.buy_mfi.value ) &
        #         (dataframe['mfi_pc'] >= self.buy_mfi_slope.value ) &
        #         (dataframe['volume'] > 0)

        #     ),
        #     ['buy', 'buy_tag']] = (1, 'Bull')



        # conditions.append(dataframe[f'wave_t1_{self.buy_wt1_length.value}'] >= self.wt_os.value )
        conditions.append(dataframe[f'wave_t1_{self.buy_wt1_length.value}'] > dataframe[f'wave_t2_{self.buy_wt2_length.value}'])
        conditions.append(dataframe['wave_t1_pc'] >= self.buy_wt_pc.value)

        # Money Flow Index
        conditions.append(dataframe[f'mfi_sma{self.mfi_sma_length.value}'] <= self.buy_mfi.value )
        conditions.append(dataframe['mfi_pc'] >= self.buy_mfi_slope.value )

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

        # # Wave Trend
        # dataframe.loc[
        #     (
        #         (dataframe[f'sma_{self.sma_length.value}pc'] <= self.sell_sma_pc.value) &
        #         (dataframe[f'wave_t1_{self.buy_wt1_length.value}'] >= self.wt_ob.value ) &
        #         (dataframe[f'wave_t1_{self.buy_wt1_length.value}'] < dataframe[f'wave_t2_{self.buy_wt2_length.value}']) &
        #         (dataframe['wave_t1_pc'] <= self.sell_wt_pc.value) &
        #         (dataframe[f'mfi_sma{self.mfi_sma_length.value}'] >= self.sell_mfi.value ) &
        #         (dataframe['mfi_pc'] <= self.sell_mfi_slope.value ) &
        #         (dataframe['volume'] > 0)

        #     ),
        #     ['buy', 'buy_tag']] = (1, 'Bull')

        # dataframe.loc[
        #     (
        #         (dataframe[f'sma_{self.sma_length.value}pc'] < self.sell_sma_pc.value) &
        #         (dataframe[f'wave_t1_{self.buy_wt1_length.value}'] <= self.wt_ob.value ) &
        #         (dataframe[f'wave_t1_{self.buy_wt1_length.value}'] >= self.wt_ob_sma.value ) &
        #         (dataframe[f'wave_t1_{self.buy_wt1_length.value}'] < dataframe[f'wave_t2_{self.buy_wt2_length.value}']) &
        #         (dataframe['wave_t1_pc'] <= self.sell_wt_pc.value) &
        #         (dataframe[f'mfi_sma{self.mfi_sma_length.value}'] >= self.sell_mfi.value ) &
        #         (dataframe['mfi_pc'] <= self.sell_mfi_slope.value ) &
        #         (dataframe['volume'] > 0)

        #     ),
        #     ['buy', 'buy_tag']] = (1, 'Bear')

        # #Wave Trend
        # if (dataframe['sma_pc'] < self.sell_sma_pc.value) and (dataframe[f'wave_t1_{self.sell_wt1_length.value}'] < self.wt_ob.value ):
        #     conditions.append(dataframe[f'wave_t1_{self.self_wt1_length.value}'] >= self.wt_ob_sma.value )
        # else:
        # conditions.append(dataframe[f'wave_t1_{self.buy_wt1_length.value}'] <= self.wt_ob.value)

        conditions.append(dataframe[f'wave_t2_{self.buy_wt2_length.value}'] <= dataframe[f'wave_t1_{self.buy_wt1_length.value}'])
        conditions.append(dataframe['wave_t1_pc'] <= self.sell_wt_pc.value)

        # Money Flow Index
        conditions.append(dataframe[f'mfi_sma{self.mfi_sma_length.value}'] >= self.sell_mfi.value )
        conditions.append(dataframe['mfi_pc'] <= self.sell_mfi_slope.value )

        # Check that volume is not 0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1
        return dataframe



### REV1
# +--------+-------------+----------+------------------+--------------+-------------------------------+------------------+-------------+-------------------------------+                                            
# |   Best |       Epoch |   Trades |    Win Draw Loss |   Avg profit |                        Profit |     Avg duration |   Objective |           Max Drawdown (Acct) |
# |--------+-------------+----------+------------------+--------------+-------------------------------+------------------+-------------+-------------------------------|
# | * Best |     5/25000 |       23 |     14    2    7 |       -1.45% |       -37.280 USDT   (-3.73%) | 26 days 19:34:00 |     37.2803 |        61.865 USDT    (6.04%) |
# | * Best |     9/25000 |       20 |     17    1    2 |        5.88% |       120.874 USDT   (12.09%) | 15 days 15:00:00 |    -120.874 |        28.842 USDT    (2.67%) |                                            
# |   Best |    41/25000 |       23 |     17    4    2 |        5.43% |       130.297 USDT   (13.03%) | 24 days 11:13:00 |    -130.297 |        22.262 USDT    (1.93%) |                                            
# |   Best |    83/25000 |       26 |     20    3    3 |        5.74% |       156.108 USDT   (15.61%) | 17 days 05:05:00 |    -156.108 |        30.312 USDT    (2.67%) |                                            
# |   Best |   114/25000 |       23 |     19    2    2 |        7.32% |       179.738 USDT   (17.97%) | 17 days 02:52:00 |    -179.738 |        28.499 USDT    (2.50%) |                                            
# |   Best |   135/25000 |       25 |     21    2    2 |        7.41% |       196.791 USDT   (19.68%) | 16 days 09:36:00 |    -196.791 |        28.446 USDT    (2.50%) |                                            
# |   Best |   159/25000 |       23 |     20    2    1 |        8.36% |       204.810 USDT   (20.48%) | 14 days 03:39:00 |     -204.81 |        28.842 USDT    (2.51%) |                                            
# |   Best |   263/25000 |       24 |     21    2    1 |        8.97% |       230.471 USDT   (23.05%) | 13 days 17:30:00 |    -230.471 |        28.842 USDT    (2.51%) |                                            
# |   Best |   312/25000 |       24 |     22    1    1 |        9.90% |       257.428 USDT   (25.74%) | 12 days 22:45:00 |    -257.428 |        11.006 USDT    (0.87%) |                                            
# |   Best |   947/25000 |       22 |     21    1    0 |       11.60% |       275.747 USDT   (27.57%) | 10 days 21:16:00 |    -275.747 |                            -- |                                            
# |   Best |  2184/25000 |       23 |     22    1    0 |       11.32% |       282.151 USDT   (28.22%) | 11 days 00:31:00 |    -282.151 |                            -- |                                            
# |   Best |  9234/25000 |       26 |     25    0    1 |       10.49% |       297.134 USDT   (29.71%) | 8 days 13:37:00  |    -297.134 |         0.260 USDT    (0.02%) |         

# Best result:

#   9234/25000:     26 trades. 25/0/1 Wins/Draws/Losses. Avg profit  10.49%. Median profit  12.28%. Total profit 297.13362921 USDT (  29.71%). Avg duration 8 days, 13:37:00 min. Objective: -297.13363


#     # Buy hyperspace params:
#     buy_params = {
#         "buy_d_length": 3,
#         "buy_esa_length": 10,
#         "buy_mfi": 53,
#         "buy_mfi_slope": 0,
#         "buy_wt1_length": 26,
#         "buy_wt2_length": 14,
#         "buy_wt_pc": -6,
#         "wt_os": -47,
#     }

#     # Sell hyperspace params:
#     sell_params = {
#         "sell_mfi": 45,
#         "sell_mfi_slope": 0,
#         "sell_wt_pc": 30,
#         "wt_ob": 46,
#     }

#     # ROI table:  # value loaded from strategy
#     minimal_roi = {
#         "0": 0.258,
#         "6213": 0.123,
#         "20752": 0.052,
#         "51551": 0
#     }

#     # Stoploss:
#     stoploss = -0.281  # value loaded from strategy

#     # Trailing stop:
#     trailing_stop = True  # value loaded from strategy
#     trailing_stop_positive = 0.327  # value loaded from strategy
#     trailing_stop_positive_offset = 0.348  # value loaded from strategy
#     trailing_only_offset_is_reached = True  # value loaded from strategy

# no zones - 6h 900 days
# 2022-11-18 20:44:37,399 - freqtrade.data.history.idatahandler - WARNING - QNT/USDT, spot, 6h, data starts at 2021-02-07 12:00:00
# 2022-11-18 20:44:37,419 - freqtrade.optimize.backtesting - INFO - Loading data from 2020-05-31 00:00:00 up to 2022-10-31 00:00:00 (883 days).
# 2022-11-18 20:44:37,419 - freqtrade.optimize.hyperopt - INFO - Dataload complete. Calculating indicators
# 2022-11-18 20:44:38,590 - freqtrade.optimize.hyperopt - INFO - Hyperopting with data from 2020-05-31 00:00:00 up to 2022-10-31 00:00:00 (883 days)..
# 2022-11-18 20:44:38,672 - freqtrade.optimize.hyperopt - INFO - Found 4 CPU cores. Let's make them scream!
# 2022-11-18 20:44:38,672 - freqtrade.optimize.hyperopt - INFO - Number of parallel jobs set as: -1
# 2022-11-18 20:44:38,672 - freqtrade.optimize.hyperopt - INFO - Using estimator ET.
# 2022-11-18 20:44:38,685 - freqtrade.optimize.hyperopt - INFO - Effective number of parallel workers used: 4
# +--------+-----------+----------+------------------+--------------+-------------------------------+-----------------+-------------+-------------------------------+                                               
# |   Best |     Epoch |   Trades |    Win Draw Loss |   Avg profit |                        Profit |    Avg duration |   Objective |           Max Drawdown (Acct) |
# |--------+-----------+----------+------------------+--------------+-------------------------------+-----------------+-------------+-------------------------------|
# | * Best |    2/2500 |        1 |      0    1    0 |        0.00% |                            -- | 5 days 12:00:00 |          -0 |                            -- |
# | * Best |    7/2500 |        1 |      1    0    0 |       21.36% |         8.474 USDT    (0.85%) | 1 days 18:00:00 |     -8.4744 |                            -- |                                               
# | * Best |   10/2500 |      258 |     59    3  196 |        0.51% |        49.312 USDT    (4.93%) | 1 days 19:08:00 |    -49.3118 |       124.140 USDT   (11.03%) |                                               
# | * Best |   13/2500 |      204 |    133   38   33 |        1.35% |       105.040 USDT   (10.50%) | 7 days 17:46:00 |     -105.04 |       100.048 USDT    (8.38%) |                                               
# |   Best |  358/2500 |      511 |    295  133   83 |        1.31% |       224.726 USDT   (22.47%) | 8 days 07:41:00 |    -224.726 |       712.145 USDT   (39.88%) |                                               
#  [Epoch 2500 of 2500 (100%)] ||                                                                                                                                          | [Time:  7:56:09, Elapsed Time: 7:56:09]
# 2022-11-19 04:40:54,230 - freqtrade.optimize.hyperopt - INFO - 2500 epochs saved to '/home/core/freqtrade/freqtrade/user_data/hyperopt_results/strategy_WTHO_2022-11-18_20-44-36.fthypt'.
# 2022-11-19 04:40:54,306 - freqtrade.resolvers.iresolver - WARNING - Could not import /home/core/freqtrade/freqtrade/user_data/strategies/BB_RPB_TSL_SMA_Tranz_1.py due to 'No module named 'finta''
# 2022-11-19 04:40:54,307 - freqtrade.optimize.hyperopt_tools - INFO - Dumping parameters to /home/core/freqtrade/freqtrade/user_data/strategies/WTHO.json

# Best result:

#    358/2500:    511 trades. 295/133/83 Wins/Draws/Losses. Avg profit   1.31%. Median profit   7.58%. Total profit 224.72565930 USDT (  22.47%). Avg duration 8 days, 7:41:00 min. Objective: -224.72566


#     # Buy hyperspace params:
#     buy_params = {
#         "buy_d_length": 12,
#         "buy_esa_length": 6,
#         "buy_mfi": 52,
#         "buy_mfi_slope": -1,
#         "buy_sma_pc": -3,
#         "buy_wt1_length": 24,
#         "buy_wt2_length": 4,
#         "buy_wt_pc": -21,
#         "max_epa": 5,
#         "wt_os": -22,
#         "wt_os_sma": -25,
#     }

#     # Sell hyperspace params:
#     sell_params = {
#         "sell_mfi": 75,
#         "sell_mfi_slope": -18,
#         "sell_sma_pc": 1,
#         "sell_wt_pc": 12,
#         "wt_ob": 39,
#         "wt_ob_sma": 25,
#     }

#     # Protection hyperspace params:
#     protection_params = {
#         "cooldown_lookback": 25,
#         "stop_duration": 142,
#         "use_stop_protection": True,
#     }

#     # ROI table:
#     minimal_roi = {
#         "0": 0.327,
#         "1717": 0.263,
#         "4302": 0.076,
#         "11290": 0
#     }

#     # Stoploss:
#     stoploss = -0.31

#     # Trailing stop:
#     trailing_stop = True
#     trailing_stop_positive = 0.261
#     trailing_stop_positive_offset = 0.355
#     trailing_only_offset_is_reached = True
    
