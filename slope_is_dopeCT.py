# ==============================================================================================
# The Slope is Dope strategy
#
# Made by:
# ______         _         _      _____                      _         ______            _
# |  _  \       | |       | |    /  __ \                    | |        |  _  \          | |
# | | | | _   _ | |_  ___ | |__  | /  \/ _ __  _   _  _ __  | |_  ___  | | | | __ _   __| |
# | | | || | | || __|/ __|| '_ \ | |    | '__|| | | || '_ \ | __|/ _ \ | | | |/ _` | / _` |
# | |/ / | |_| || |_| (__ | | | || \__/\| |   | |_| || |_) || |_| (_) || |/ /| (_| || (_| |
# |___/   \__,_| \__|\___||_| |_| \____/|_|    \__, || .__/  \__|\___/ |___/  \__,_| \__,_|
#                                               __/ || |
#                                              |___/ |_|
# Version : 1.0
# Date    : 2022-10031
# Remarks :
#    As published, explained and tested in my Youtube video:
#    - https://youtu.be/UvS3ixWG2zs
#    -
# ==============================================================================================

# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
# --------------------------------

# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter)
from scipy.spatial.distance import cosine
import numpy as np

class slope_is_dopeCT(IStrategy):
    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.6
    }

    stoploss = -0.9

    timeframe = '4h'

    # protections
    cooldown_lookback = IntParameter(2, 48, default=5, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # hyperopt params
    slope_length = IntParameter(5, 20, default=11, optimize=True)
    stoploss_length = IntParameter(5, 15, default=10, optimize=True)
    rsi_buy = IntParameter(30, 60, default=55, space="buy", optimize=True)
    fslope_buy = IntParameter(-5, 5, default=0, space="buy", optimize=True)
    sslope_buy = IntParameter(-5, 5, default=0, space="buy", optimize=True)
    fslope_sell = IntParameter(-5, 5, default=0, space="sell", optimize=True)

    # Trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.28

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
        

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['marketMA'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['fastMA'] = ta.SMA(dataframe, timeperiod=21)
        dataframe['slowMA'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['entryMA'] = ta.SMA(dataframe, timeperiod=3)
        # Calculate slope of slowMA
        # See: https://www.wikihow.com/Find-the-Slope-of-a-Line
        dataframe['sy1'] = dataframe['slowMA'].shift(+(self.slope_length.value))
        dataframe['sy2'] = dataframe['slowMA'].shift(+1)
        sx1 = 1
        sx2 = (self.slope_length.value)
        dataframe['sy'] = dataframe['sy2'] - dataframe['sy1']
        dataframe['sx'] = sx2 - sx1
        dataframe['slow_slope'] = dataframe['sy']/dataframe['sx']
        dataframe['fy1'] = dataframe['fastMA'].shift(+(self.slope_length.value))
        dataframe['fy2'] = dataframe['fastMA'].shift(+1)
        fx1 = 1
        fx2 = (self.slope_length.value)
        dataframe['fy'] = dataframe['fy2'] - dataframe['fy1']
        dataframe['fx'] = fx2 - fx1
        dataframe['fast_slope'] = dataframe['fy']/dataframe['fx']
        # print(dataframe[['date','close', 'slow_slope','fast_slope']].tail(50))

        # ==== Trailing custom stoploss indicator ====
        dataframe['last_lowest'] = dataframe['low'].rolling((self.stoploss_length.value)).min().shift(1)

        return dataframe

    plot_config = {
        "main_plot": {
            # Configuration for main plot indicators.
            "fastMA": {"color": "red"},
            "slowMA": {"color": "blue"},
        },
        "subplots": {
            # Additional subplots
            "rsi": {"rsi": {"color": "blue"}},
            "fast_slope": {"fast_slope": {"color": "red"}, "slow_slope": {"color": "blue"}},
        },
    }


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Only enter when market is bullish (this is a choice)
                (
                # (dataframe['close'] > dataframe['marketMA']) &
                # Only trade when the fast slope is above 0
                (dataframe['fast_slope'] > self.fslope_buy.value) &
                # Only trade when the slow slope is above 0
                (dataframe['slow_slope'] > self.sslope_buy.value) &
                # Only buy when the close price is higher than the 3day average of ten periods ago
                # (dataframe['close'] > dataframe['entryMA'].shift(+(slope_length))) &
                # Or only buy when the close price is higher than the close price of 3 days ago (this is a choice)
                (dataframe['close'] > dataframe['close'].shift(+(self.slope_length.value))) &
                # Only enter trades when the RSI is higher than 55
                (dataframe['rsi'] > self.rsi_buy.value) 
                # Only trade when the fast MA is above the slow MA
                # (dataframe['fastMA'] > dataframe['slowMA'])
                # Or trade when the fase MA crosses above the slow MA (This is a choice...)
                # (qtpylib.crossed_above(dataframe['fastMA'], dataframe['slowMA']))
                )
            ),
            'buy'] = 1


        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (

                # Close or do not trade when fastMA is below slowMA # modified from strategy
                (dataframe['fast_slope'] < self.fslope_sell.value)
                # Or close position when the close price gets below the last lowest candle price configured
                # (AKA candle based (Trailing) stoploss) 
                | (dataframe['close'] < dataframe['last_lowest'])
                # | (dataframe['close'] < dataframe['fastMA'])
            ),
            'sell'] = 1
        return dataframe

# 4hr pre hyper opt
# ============================================================= BACKTESTING REPORT ============================================================
# |      Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |-----------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# |  QNT/USDT |        64 |           3.62 |         231.57 |           490.426 |          49.04 | 2 days, 14:56:00 |    23     0    41  35.9 |
# |  XRP/USDT |        83 |           3.56 |         295.10 |           434.128 |          43.41 | 2 days, 22:42:00 |    34     0    49  41.0 |
# |  NGM/USDT |        40 |           2.58 |         103.10 |           253.281 |          25.33 | 2 days, 18:30:00 |    15     0    25  37.5 |
# | ATOM/USDT |        82 |           1.77 |         144.93 |           162.935 |          16.29 |  3 days, 8:23:00 |    34     0    48  41.5 |
# |  AKT/USDT |        19 |           3.31 |          62.97 |           161.767 |          16.18 |  2 days, 9:54:00 |     9     0    10  47.4 |
# |  ETH/USDT |        95 |           1.20 |         113.78 |           132.553 |          13.26 |  3 days, 7:02:00 |    32     0    63  33.7 |
# | CSPR/USDT |        11 |           3.90 |          42.93 |           111.517 |          11.15 |  3 days, 9:27:00 |     5     0     6  45.5 |
# |  BTC/USDT |        74 |           1.29 |          95.21 |            78.714 |           7.87 | 3 days, 19:08:00 |    25     0    49  33.8 |
# |  ADA/USDT |        84 |           0.84 |          70.42 |            59.661 |           5.97 |  3 days, 0:09:00 |    23     0    61  27.4 |
# | SCRT/USDT |        13 |           1.37 |          17.87 |            35.509 |           3.55 |  2 days, 5:14:00 |     3     0    10  23.1 |
# | IOTA/USDT |        27 |          -2.50 |         -67.48 |          -177.933 |         -17.79 | 2 days, 19:42:00 |     6     0    21  22.2 |
# |     TOTAL |       592 |           1.88 |        1110.38 |          1742.558 |         174.26 |  3 days, 2:17:00 |   209     0   383  35.3 |
# =========================================================== ENTER TAG STATS ============================================================
# |   TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |
# |-------+-----------+----------------+----------------+-------------------+----------------+-----------------+-------------------------|
# | TOTAL |       592 |           1.88 |        1110.38 |          1742.558 |         174.26 | 3 days, 2:17:00 |   209     0   383  35.3 |
# ======================================================= EXIT REASON STATS ========================================================
# |        Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
# |--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
# |        exit_signal |     507 |    125     0   382  24.7 |          -2.64 |       -1336.95 |         -2830.51  |        -133.7  |
# | trailing_stop_loss |      84 |     84     0     0   100 |          29.17 |        2450.34 |          4581.26  |         245.03 |
# |         force_exit |       1 |      0     0     1     0 |          -3.01 |          -3.01 |            -8.194 |          -0.3  |
# ======================================================== LEFT OPEN TRADES REPORT =========================================================
# |     Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
# |----------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
# | XRP/USDT |         1 |          -3.01 |          -3.01 |            -8.194 |          -0.82 |       12:00:00 |     0     0     1     0 |
# |    TOTAL |         1 |          -3.01 |          -3.01 |            -8.194 |          -0.82 |       12:00:00 |     0     0     1     0 |
# ================== SUMMARY METRICS ==================
# | Metric                      | Value               |
# |-----------------------------+---------------------|
# | Backtesting from            | 2020-06-03 00:00:00 |
# | Backtesting to              | 2022-11-20 20:00:00 |
# | Max open trades             | 10                  |
# |                             |                     |
# | Total/Daily Avg Trades      | 592 / 0.66          |
# | Starting balance            | 1000 USDT           |
# | Final balance               | 2742.558 USDT       |
# | Absolute profit             | 1742.558 USDT       |
# | Total profit %              | 174.26%             |
# | CAGR %                      | 50.56%              |
# | Profit factor               | 1.40                |
# | Trades per day              | 0.66                |
# | Avg. daily profit %         | 0.19%               |
# | Avg. stake amount           | 209.64 USDT         |
# | Total trade volume          | 124106.936 USDT     |
# |                             |                     |
# | Best Pair                   | XRP/USDT 295.10%    |
# | Worst Pair                  | IOTA/USDT -67.48%   |
# | Best trade                  | ATOM/USDT 52.85%    |
# | Worst trade                 | XRP/USDT -37.06%    |
# | Best day                    | 147.924 USDT        |
# | Worst day                   | -111.132 USDT       |
# | Days win/draw/lose          | 134 / 554 / 192     |
# | Avg. Duration Winners       | 4 days, 11:01:00    |
# | Avg. Duration Loser         | 2 days, 8:25:00     |
# | Rejected Entry signals      | 26                  |
# | Entry/Exit Timeouts         | 0 / 0               |
# |                             |                     |
# | Min balance                 | 985.826 USDT        |
# | Max balance                 | 2864.896 USDT       |
# | Max % of account underwater | 16.82%              |
# | Absolute Drawdown (Account) | 16.82%              |
# | Absolute Drawdown           | 470.482 USDT        |
# | Drawdown high               | 1797.644 USDT       |
# | Drawdown low                | 1327.163 USDT       |
# | Drawdown Start              | 2021-09-15 08:00:00 |
# | Drawdown End                | 2022-06-27 00:00:00 |
# | Market change               | 85.26%              |
# =====================================================
# 2022-11-20 20:57:54,321 - freqtrade.optimize.hyperopt - INFO - Effective number of parallel workers used: 4
# +--------+-----------+----------+------------------+--------------+-------------------------------+-----------------+-------------+-------------------------------+
# |   Best |     Epoch |   Trades |    Win Draw Loss |   Avg profit |                        Profit |    Avg duration |   Objective |           Max Drawdown (Acct) |
# |--------+-----------+----------+------------------+--------------+-------------------------------+-----------------+-------------+-------------------------------|
# | * Best |    1/2500 |       91 |     34    9   48 |       -0.21% |       -20.714 USDT   (-2.07%) | 2 days 00:26:00 |     20.7142 |        55.997 USDT    (5.41%) |
# | * Best |    4/2500 |       93 |     39    0   54 |        0.12% |         8.947 USDT    (0.89%) | 2 days 11:03:00 |    -8.94744 |        57.359 USDT    (5.44%) |                             
# | * Best |    7/2500 |      654 |    279   17  358 |        0.07% |        13.194 USDT    (1.32%) | 1 days 16:21:00 |    -13.1936 |       196.774 USDT   (16.74%) |                             
# | * Best |    8/2500 |       94 |     33    0   61 |        0.18% |        15.052 USDT    (1.51%) | 2 days 08:18:00 |    -15.0517 |        86.098 USDT    (7.82%) |                             
# | * Best |   14/2500 |      132 |     52    9   71 |        0.36% |        45.553 USDT    (4.56%) | 2 days 06:11:00 |    -45.5528 |        88.078 USDT    (7.77%) |                             
# |   Best |   61/2500 |      184 |     74   10  100 |        0.42% |        75.918 USDT    (7.59%) | 2 days 08:27:00 |    -75.9177 |        87.909 USDT    (7.61%) |                             
# |   Best |   87/2500 |      535 |    189   11  335 |        0.37% |       169.941 USDT   (16.99%) | 2 days 21:09:00 |    -169.941 |       357.790 USDT   (25.61%) |                             
# |   Best |   98/2500 |      726 |    351    9  366 |        0.27% |       178.256 USDT   (17.83%) | 1 days 16:24:00 |    -178.256 |       340.118 USDT   (24.14%) |                             
# |   Best |  105/2500 |      699 |    307    6  386 |        0.69% |       547.126 USDT   (54.71%) | 2 days 02:17:00 |    -547.126 |       488.887 USDT   (27.98%) |                             
# |   Best |  161/2500 |     1406 |    569   59  778 |        0.48% |       774.584 USDT   (77.46%) | 1 days 19:42:00 |    -774.584 |       880.456 USDT   (40.07%) |                             
# |   Best |  202/2500 |     1174 |    483   48  643 |        0.69% |      1009.295 USDT  (100.93%) | 2 days 10:07:00 | -1,009.29523 |      1096.212 USDT   (42.98%) |                            
# |   Best |  214/2500 |      792 |    398    8  386 |        1.00% |      1084.917 USDT  (108.49%) | 1 days 22:08:00 | -1,084.91742 |       234.977 USDT   (13.60%) |                            
# |   Best |  222/2500 |      777 |    350    3  424 |        1.07% |      1155.454 USDT  (115.55%) | 1 days 23:27:00 | -1,155.45395 |       312.651 USDT   (15.08%) |                            
# |   Best |  341/2500 |     1600 |    838   23  739 |        0.59% |      1272.501 USDT  (127.25%) | 1 days 18:53:00 | -1,272.50148 |       718.750 USDT   (31.01%) |                            
# |   Best |  393/2500 |     1738 |   1010   47  681 |        0.57% |      1398.651 USDT  (139.87%) | 1 days 11:29:00 | -1,398.65073 |       662.969 USDT   (28.30%) |                            
# |   Best |  443/2500 |     1343 |    597    8  738 |        0.87% |      1816.027 USDT  (181.60%) | 2 days 00:33:00 | -1,816.02663 |       770.553 USDT   (32.20%) |                            
# |   Best | 1178/2500 |     1889 |   1058   71  760 |        0.64% |      1968.797 USDT  (196.88%) | 1 days 11:43:00 | -1,968.79678 |       859.072 USDT   (31.93%) |                            
# |   Best | 1343/2500 |     1071 |    543   12  516 |        1.19% |      2294.956 USDT  (229.50%) | 1 days 21:06:00 | -2,294.95604 |       356.680 USDT   (14.14%) |                            
#  [Epoch 2500 of 2500 (100%)] ||                                                                                                                       | [Time: 13:57:39, Elapsed Time: 13:57:39]
# 2022-11-21 10:55:40,738 - freqtrade.optimize.hyperopt - INFO - 2500 epochs saved to '/home/jared/freqtrade/user_data/hyperopt_results/strategy_slope_is_dopeCT_2022-11-20_20-57-53.fthypt'.
# 2022-11-21 10:55:40,763 - freqtrade.resolvers.iresolver - WARNING - Could not import /home/jared/freqtrade/user_data/strategies/NASOSv5.py due to 'name 'TrailingBuySellStrat' is not defined'
# 2022-11-21 10:55:41,050 - NFIX - INFO - pandas_ta successfully imported
# 2022-11-21 10:55:41,055 - freqtrade.optimize.hyperopt_tools - INFO - Dumping parameters to /home/jared/freqtrade/user_data/strategies/slope_is_dopeCT.json

# Best result:

#   1343/2500:   1071 trades. 543/12/516 Wins/Draws/Losses. Avg profit   1.19%. Median profit   0.61%. Total profit 2294.95603739 USDT ( 229.50%). Avg duration 1 day, 21:06:00 min. Objective: -2294.95604


#     # Buy hyperspace params:
#     buy_params = {
#         "fslope_buy": -4,
#         "rsi_buy": 41,
#         "sslope_buy": 0,
#     }

#     # Sell hyperspace params:
#     sell_params = {
#         "fslope_sell": -2,
#     }

#     # Protection hyperspace params:
#     protection_params = {
#         "cooldown_lookback": 2,
#         "stop_duration": 37,
#         "use_stop_protection": False,
#     }

#     # ROI table:
#     minimal_roi = {
#         "0": 0.393,
#         "1628": 0.229,
#         "3817": 0.116,
#         "8490": 0
#     }

#     # Stoploss:
#     stoploss = -0.306

#     # Trailing stop:
#     trailing_stop = True
#     trailing_stop_positive = 0.012
#     trailing_stop_positive_offset = 0.075
#     trailing_only_offset_is_reached = True
# Result for strategy slope_is_dopeCT
# ============================================================= BACKTESTING REPORT ============================================================
# |      Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |-----------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# |  QNT/USDT |       149 |           2.16 |         321.29 |          1036.302 |         103.63 |  1 day, 14:20:00 |    92     0    57  61.7 |
# | ATOM/USDT |       221 |           1.91 |         422.83 |           958.614 |          95.86 |  1 day, 15:21:00 |   137     0    84  62.0 |
# | CSPR/USDT |        39 |           3.43 |         133.94 |           658.118 |          65.81 |   1 day, 7:23:00 |    26     0    13  66.7 |
# |  NGM/USDT |        77 |           2.37 |         182.13 |           646.545 |          64.65 |  1 day, 19:35:00 |    43     1    33  55.8 |
# |  AKT/USDT |        33 |           3.37 |         111.25 |           491.363 |          49.14 |  1 day, 23:45:00 |    19     0    14  57.6 |
# | SCRT/USDT |        35 |           2.58 |          90.31 |           397.798 |          39.78 |   1 day, 6:03:00 |    22     1    12  62.9 |
# |  ETH/USDT |       163 |           1.04 |         170.28 |           291.359 |          29.14 |  2 days, 3:22:00 |    76     3    84  46.6 |
# |  ADA/USDT |       167 |           1.15 |         191.40 |           283.234 |          28.32 |  1 day, 22:05:00 |    91     3    73  54.5 |
# |  XRP/USDT |       168 |           1.22 |         205.03 |           233.085 |          23.31 |  1 day, 23:06:00 |    84     4    80  50.0 |
# |  BTC/USDT |       122 |           0.66 |          80.35 |           -17.478 |          -1.75 | 2 days, 10:22:00 |    48     1    73  39.3 |
# | IOTA/USDT |        43 |          -0.72 |         -31.16 |          -155.348 |         -15.53 |  2 days, 4:33:00 |    16     0    27  37.2 |
# |     TOTAL |      1217 |           1.54 |        1877.65 |          4823.593 |         482.36 |  1 day, 21:10:00 |   654    13   550  53.7 |
# =========================================================== ENTER TAG STATS ============================================================
# |   TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |
# |-------+-----------+----------------+----------------+-------------------+----------------+-----------------+-------------------------|
# | TOTAL |      1217 |           1.54 |        1877.65 |          4823.593 |         482.36 | 1 day, 21:10:00 |   654    13   550  53.7 |
# ======================================================= EXIT REASON STATS ========================================================
# |        Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
# |--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
# | trailing_stop_loss |     616 |    616     0     0   100 |           8.81 |        5429.82 |         15698.4   |         542.98 |
# |        exit_signal |     555 |     17     0   538   3.1 |          -6.15 |       -3414.8  |        -10702.1   |        -341.48 |
# |                roi |      34 |     21    13     0   100 |           5.95 |         202.22 |           516.412 |          20.22 |
# |          stop_loss |      11 |      0     0    11     0 |         -30.74 |        -338.12 |          -680.596 |         -33.81 |
# |         force_exit |       1 |      0     0     1     0 |          -1.47 |          -1.47 |            -8.572 |          -0.15 |
# ========================================================= LEFT OPEN TRADES REPORT =========================================================
# |     Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |
# |----------+-----------+----------------+----------------+-------------------+----------------+-----------------+-------------------------|
# | XRP/USDT |         1 |          -1.47 |          -1.47 |            -8.572 |          -0.86 | 2 days, 0:00:00 |     0     0     1     0 |
# |    TOTAL |         1 |          -1.47 |          -1.47 |            -8.572 |          -0.86 | 2 days, 0:00:00 |     0     0     1     0 |
# ================== SUMMARY METRICS ==================
# | Metric                      | Value               |
# |-----------------------------+---------------------|
# | Backtesting from            | 2020-06-03 00:00:00 |
# | Backtesting to              | 2022-11-20 20:00:00 |
# | Max open trades             | 10                  |
# |                             |                     |
# | Total/Daily Avg Trades      | 1217 / 1.35         |
# | Starting balance            | 1000 USDT           |
# | Final balance               | 5823.593 USDT       |
# | Absolute profit             | 4823.593 USDT       |
# | Total profit %              | 482.36%             |
# | CAGR %                      | 104.33%             |
# | Profit factor               | 1.42                |
# | Trades per day              | 1.35                |
# | Avg. daily profit %         | 0.54%               |
# | Avg. stake amount           | 305.054 USDT        |
# | Total trade volume          | 371251.137 USDT     |
# |                             |                     |
# | Best Pair                   | ATOM/USDT 422.83%   |
# | Worst Pair                  | IOTA/USDT -31.16%   |
# | Best trade                  | QNT/USDT 39.26%     |
# | Worst trade                 | ADA/USDT -30.74%    |
# | Best day                    | 294.801 USDT        |
# | Worst day                   | -322.585 USDT       |
# | Days win/draw/lose          | 299 / 359 / 222     |
# | Avg. Duration Winners       | 1 day, 9:57:00      |
# | Avg. Duration Loser         | 2 days, 7:48:00     |
# | Rejected Entry signals      | 37                  |
# | Entry/Exit Timeouts         | 0 / 0               |
# |                             |                     |
# | Min balance                 | 986.25 USDT         |
# | Max balance                 | 5962.985 USDT       |
# | Max % of account underwater | 15.44%              |
# | Absolute Drawdown (Account) | 14.56%              |
# | Absolute Drawdown           | 576.012 USDT        |
# | Drawdown high               | 2956.538 USDT       |
# | Drawdown low                | 2380.526 USDT       |
# | Drawdown Start              | 2021-10-09 16:00:00 |
# | Drawdown End                | 2021-12-13 16:00:00 |
# | Market change               | 85.26%              |
# =====================================================
# Result for strategy slope_is_dopeCT
# ============================================================ BACKTESTING REPORT ============================================================
# |      Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |
# |-----------+-----------+----------------+----------------+-------------------+----------------+-----------------+-------------------------|
# |  QNT/USDT |       149 |           2.16 |         321.29 |          2782.873 |         278.29 | 1 day, 14:20:00 |    92     0    57  61.7 |
# | ATOM/USDT |       289 |           1.52 |         440.51 |          2634.060 |         263.41 | 1 day, 17:29:00 |   165     0   124  57.1 |
# | CSPR/USDT |        39 |           3.43 |         133.94 |          1767.280 |         176.73 |  1 day, 7:23:00 |    26     0    13  66.7 |
# |  NGM/USDT |        77 |           2.37 |         182.13 |          1736.203 |         173.62 | 1 day, 19:35:00 |    43     1    33  55.8 |
# |  ADA/USDT |       249 |           2.07 |         515.05 |          1434.922 |         143.49 | 1 day, 20:27:00 |   137     4   108  55.0 |
# |  AKT/USDT |        33 |           3.37 |         111.25 |          1319.485 |         131.95 | 1 day, 23:45:00 |    19     0    14  57.6 |
# |  ETH/USDT |       358 |           1.43 |         511.93 |          1240.145 |         124.01 | 2 days, 1:26:00 |   183     4   171  51.1 |
# | SCRT/USDT |        35 |           2.58 |          90.31 |          1068.230 |         106.82 |  1 day, 6:03:00 |    22     1    12  62.9 |
# |  XRP/USDT |       262 |           1.13 |         295.72 |           707.375 |          70.74 | 2 days, 0:45:00 |   126     8   128  48.1 |
# |  BTC/USDT |       266 |           1.26 |         334.25 |           296.002 |          29.60 | 2 days, 9:12:00 |   117     1   148  44.0 |
# | IOTA/USDT |        43 |          -0.72 |         -31.16 |          -417.164 |         -41.72 | 2 days, 4:33:00 |    16     0    27  37.2 |
# |     TOTAL |      1800 |           1.61 |        2905.21 |         14569.411 |        1456.94 | 1 day, 22:37:00 |   946    19   835  52.6 |
# =========================================================== ENTER TAG STATS ============================================================
# |   TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |
# |-------+-----------+----------------+----------------+-------------------+----------------+-----------------+-------------------------|
# | TOTAL |      1800 |           1.61 |        2905.21 |         14569.411 |        1456.94 | 1 day, 22:37:00 |   946    19   835  52.6 |
# ======================================================= EXIT REASON STATS ========================================================
# |        Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
# |--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
# | trailing_stop_loss |     871 |    871     0     0   100 |           8.67 |        7551.9  |          45956.5  |         755.19 |
# |        exit_signal |     849 |     26     0   823   3.1 |          -5.69 |       -4834.75 |         -31469    |        -483.47 |
# |                roi |      68 |     49    19     0   100 |           8.19 |         556.92 |           1985.47 |          55.69 |
# |          stop_loss |      12 |      0     0    12     0 |         -30.74 |        -368.85 |          -1903.54 |         -36.89 |
# ======================================================= LEFT OPEN TRADES REPORT ========================================================
# |   Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
# |--------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
# |  TOTAL |         0 |           0.00 |           0.00 |             0.000 |           0.00 |           0:00 |     0     0     0     0 |
# ================== SUMMARY METRICS ==================
# | Metric                      | Value               |
# |-----------------------------+---------------------|
# | Backtesting from            | 2017-10-18 16:00:00 |
# | Backtesting to              | 2022-11-21 20:00:00 |
# | Max open trades             | 10                  |
# |                             |                     |
# | Total/Daily Avg Trades      | 1800 / 0.97         |
# | Starting balance            | 1000 USDT           |
# | Final balance               | 15569.411 USDT      |
# | Absolute profit             | 14569.411 USDT      |
# | Total profit %              | 1456.94%            |
# | CAGR %                      | 71.38%              |
# | Profit factor               | 1.43                |
# | Trades per day              | 0.97                |
# | Avg. daily profit %         | 0.78%               |
# | Avg. stake amount           | 613.589 USDT        |
# | Total trade volume          | 1104460.99 USDT     |
# |                             |                     |
# | Best Pair                   | ADA/USDT 515.05%    |
# | Worst Pair                  | IOTA/USDT -31.16%   |
# | Best trade                  | ADA/USDT 39.26%     |
# | Worst trade                 | ADA/USDT -30.74%    |
# | Best day                    | 791.645 USDT        |
# | Worst day                   | -866.254 USDT       |
# | Days win/draw/lose          | 467 / 1002 / 380    |
# | Avg. Duration Winners       | 1 day, 13:02:00     |
# | Avg. Duration Loser         | 2 days, 6:59:00     |
# | Rejected Entry signals      | 37                  |
# | Entry/Exit Timeouts         | 0 / 0               |
# |                             |                     |
# | Min balance                 | 1008.338 USDT       |
# | Max balance                 | 16012.748 USDT      |
# | Max % of account underwater | 15.44%              |
# | Absolute Drawdown (Account) | 14.56%              |
# | Absolute Drawdown           | 1546.81 USDT        |
# | Drawdown high               | 9624.716 USDT       |
# | Drawdown low                | 8077.906 USDT       |
# | Drawdown Start              | 2021-10-09 16:00:00 |
# | Drawdown End                | 2021-12-13 16:00:00 |
# | Market change               | 74.19%              |
# =====================================================
