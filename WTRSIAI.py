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


class WTRSIAI(IStrategy):
    """
    Example of a hybrid FreqAI strat, designed to illustrate how a user may employ
    FreqAI to bolster a typical Freqtrade strategy.

    Launching this strategy would be:

    freqtrade trade --strategy WTAI --freqaimodel CatboostClassifier --config ho.json

    To HyperOpt this strategy:

    freqtrade hyperopt --hyperopt-loss SharpeHyperOptLoss --strategy WTAI --freqaimodel CatboostClassifier --config ho.json --timerange 20220601-20221201 -e 1000


    or the user simply adds this to their config:

    "freqai": {
        "enabled": true,
        "fit_live_predictions_candles": 300,
        "purge_old_models": true,
        "train_period_days": 15,
        "identifier": "unique-id",
        "live_retrain_hours": 1.0
        "feature_parameters": {
            "include_timeframes": [
                "15m",
                "1h",
                "4h"
            ],
            "include_corr_pairlist": [
                "BTC/USDT",
                "ATOM/USDT",
                "ETH/USDT",
                "XRP/USDT"
            ],
            "label_period_candles": 20,
            "include_shifted_candles": 2,
            "DI_threshold": 0.9,
            "weight_factor": 0.9,
            "principal_component_analysis": false,
            "use_SVM_to_remove_outliers": true,
            "indicator_periods_candles": [3, 10, 20]
        },
        "data_split_parameters": {
            "test_size": 0,
            "random_state": 1
        },
        "model_training_parameters": {
            "n_estimators": 800
        }
    },

    paste this into your config file
    """

    ### PLOT INDICATORS
    plot_config = {
        'main_plot': {
            'tema': {},
        },
        'subplots': {
            "RSI": {
                'rsi': {'color': 'red'},
            },
            "WAVE":{
                'wave_t1': {'color': 'white'},
                'wave_t2': {'color': 'blue'}
            },
            "Up_or_down": {
                '&s-up_or_down': {'color': 'green'},
            }
        }
    }


    ### Strategy parameters ###
    exit_profit_only = True ### No selling at a loss
    use_custom_stoploss = True
    trailing_stop = True
    position_adjustment_enable = True
    process_only_new_candles = True
    ignore_roi_if_entry_signal = True
    use_exit_signal = True
    stoploss = -0.40
    startup_candle_count: int = 30
    timeframe = '1h'
    # DCA Parameters
    position_adjustment_enable = True
    max_entry_position_adjustment = 3
    max_dca_multiplier = 7
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }


    ### Hyperoptable parameters ###
    # entry optizimation
    max_epa = CategoricalParameter([-1, 0, 1, 3, 5, 10], default=3, space="buy", optimize=True)

    # protections
    cooldown_lookback = IntParameter(2, 48, default=5, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # indicators
    mfi_length = IntParameter(3,60, default=53, space='buy', optimize=True)

    # trading
    buy_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)
    mfi_buy_slope = IntParameter(-30, 30 , default=0, space='buy', optimize=True)
    sell_rsi = IntParameter(low=50, high=100, default=55, space='sell', optimize=True, load=True)
    mfi_sell_slope = IntParameter(-30, 30 , default=0, space='sell', optimize=True)


    ### entry opt. ###
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


    ### Dollar Cost Averaging ###
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
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be
        increased or decreased.
        This means extra buy or sell orders with additional fees.
        Only called when `position_adjustment_enable` is set to True.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns None

        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Current buy rate.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param min_stake: Minimal stake size allowed by exchange (for both entries and exits)
        :param max_stake: Maximum stake allowed (either through balance, or by exchange limits).
        :param current_entry_rate: Current rate using entry pricing.
        :param current_exit_rate: Current rate using exit pricing.
        :param current_entry_profit: Current profit using entry pricing.
        :param current_exit_profit: Current profit using exit pricing.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: Stake amount to adjust your trade,
                       Positive values to increase position, Negative values to decrease position.
                       Return None for no action.
        """

        if current_profit > 0.10 and trade.nr_of_successful_exits == 0:
            # Take half of the profit at +5%
            return -(trade.stake_amount / 2)

        if current_profit > -0.015 and trade.nr_of_successful_entries == 1:
            return None

        if current_profit > -0.05 and trade.nr_of_successful_entries == 2:
            return None

        if current_profit > -0.10 and trade.nr_of_successful_entries == 3:
            return None


        # Obtain pair dataframe (just to show how to access it)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries

        # Allow up to 3 additional increasingly larger buys (4 in total)
        # Initial buy is 1x
        # If that falls to -5% profit, we buy more, 
        # If that falls down to -5% again, we buy 1.5x more
        # If that falls once again down to -5%, we buy  more
        # Total stake for this trade would be 1 + 1.5 + 2 + 2.5 = 7x of the initial allowed stake.
        # That is why max_dca_multiplier is 7
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

        if (current_profit > 0.3):
            return 0.05
        elif (current_profit > 0.1):
            return 0.025
        elif (current_profit > 0.06):
            return 0.01
        elif (current_profit > 0.04):
            return 0.01
        elif (current_profit > 0.025):
            return 0.005

        return self.stoploss

    ### FREQ AI ###
    # FreqAI required function, user can add or remove indicators, but general structure
    # must stay the same.
    def populate_any_indicators(
        self, pair, df, tf, informative=None, set_generalized_indicators=False
    ):
        """
        User feeds these indicators to FreqAI to train a classifier to decide
        if the market will go up or down.

        :param pair: pair to be used as informative
        :param df: strategy dataframe which will receive merges from informatives
        :param tf: timeframe of the dataframe which will modify the feature names
        :param informative: the dataframe associated with the informative pair
        """

        if informative is None:
            informative = self.dp.get_pair_dataframe(pair, tf)

        # first loop is automatically duplicating indicators for time periods
        for t in self.freqai_info["feature_parameters"]["indicator_periods_candles"]:

            t = int(t)
            # WaveTrend using OHLC4 or HA close
            informative[f"-{pair}wt_ap_{t}"] = (0.25 * (informative['high'] + informative['low'] + informative['close'] + informative['open']))
            informative[f"-{pair}wt_esa_{t}"] = ta.EMA(informative[f"-{pair}wt_ap_{t}"], timeperiod=t)
            informative[f"-{pair}wt_d_{t}"] = ta.EMA(abs(informative[f"-{pair}wt_ap_{t}"] - informative[f"-{pair}wt_esa_{t}"]), timeperiod=t)
            informative[f"-{pair}wt_ci_{t}"] = (
                (informative[f"-{pair}wt_ap_{t}"]-informative[f"-{pair}wt_esa_{t}"]) / (0.015 * informative[f"-{pair}wt_d_{t}"])
            )
            informative[f"%-{pair}wt1_{t}"] = ta.EMA(informative[f"-{pair}wt_ci_{t}"], timeperiod=t) 
            informative[f"%-{pair}wt2_{t}"] = ta.SMA(informative[f"%-{pair}wt1_{t}"], timeperiod=t)           


            informative[f"%-{pair}rsi-period_{t}"] = ta.RSI(informative, timeperiod=t)
            informative[f"%-{pair}mfi-period_{t}"] = ta.MFI(informative, timeperiod=t)
            informative[f"%-{pair}sma-period_{t}"] = ta.SMA(informative, timeperiod=t)
            informative[f"%-{pair}roc-period_{t}"] = ta.ROC(informative, timeperiod=t)
            informative[f"%-{pair}relative_volume-period_{t}"] = (
                informative["volume"] / informative["volume"].rolling(t).mean()
            )

        # FreqAI needs the following lines in order to detect features and automatically
        # expand upon them.
        indicators = [col for col in informative if col.startswith("%")]
        # This loop duplicates and shifts all indicators to add a sense of recency to data
        for n in range(self.freqai_info["feature_parameters"]["include_shifted_candles"] + 1):
            if n == 0:
                continue
            informative_shift = informative[indicators].shift(n)
            informative_shift = informative_shift.add_suffix("_shift-" + str(n))
            informative = pd.concat((informative, informative_shift), axis=1)

        df = merge_informative_pair(df, informative, self.config["timeframe"], tf, ffill=True)
        skip_columns = [
            (s + "_" + tf) for s in ["date", "open", "high", "low", "close", "volume"]
        ]
        df = df.drop(columns=skip_columns)

        # User can set the "target" here (in present case it is the
        # "up" or "down")
        if set_generalized_indicators:
            # User "looks into the future" here to figure out if the future
            # will be "up" or "down". This same column name is available to
            # the user
            df['&s-up_or_down'] = np.where(df["close"].shift(-1) >
                                           df["close"] , 'up', 'down')


        return df


    ### NORMAL INDICATORS ###
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # User creates their own custom strat here. Present example is a supertrend
        # based strategy.

        dataframe = self.freqai.start(dataframe, metadata, self)

        # TA indicators to combine with the Freqai targets
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['rsi_ma'] = ta.SMA(dataframe['rsi'], timeperiod=7)

        # TEMA - Triple Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        # WaveTrend using OHLC4 or HA close - 3/21
        ap = (0.25 * (dataframe['high'] + dataframe['low'] + dataframe["close"] + dataframe["open"]))
        
        dataframe['esa'] = ta.EMA(ap, timeperiod = 3)
        dataframe['d'] = ta.EMA(abs(ap - dataframe['esa']), timeperiod = 3)
        dataframe['wave_ci'] = (ap-dataframe['esa']) / (0.015 * dataframe['d'])
        dataframe['wave_t1'] = ta.EMA(dataframe['wave_ci'], timeperiod = 21)  
        dataframe['wave_t2'] = ta.SMA(dataframe['wave_t1'], timeperiod = 4)

        # Money Flow Index

        # find mfi optimum length
        for valmfi in self.mfi_length.range:
            dataframe[f'mfi_{valmfi}'] = ta.MFI(dataframe, timeperiod = valmfi)

        dataframe['mfi_slope'] = pta.momentum.slope(dataframe[f'mfi_{valmfi}'])

        return dataframe

    ### ENTRY CONDITIONS ###
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] >  self.buy_rsi.value) &
                (df['rsi'] < 65) &
                (df['tema'] > df['tema'].shift(1)) &  # Guard: tema is raising
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['mfi_slope'] >= self.mfi_buy_slope.value) & # Money flow index
                (df['do_predict'] == 1) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['&s-up_or_down'] == 'up')
            ),
            ['enter_long', 'enter_tag']] = (1, 'TEMA/RSI/MF-AI')

        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] >  self.buy_rsi.value) &
                (df['rsi'] < 65) &
                (df['tema'] > df['tema'].shift(1)) &  # Guard: teinma is raisg
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['do_predict'] == 1) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['&s-up_or_down'] == 'up')
            ),
            ['enter_long', 'enter_tag']] = (1, 'TEMA/RSI-AI')

        df.loc[
            (
                (df['tema'] > df['tema'].shift(1)) &  # Guard: tema is raising
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['mfi_slope'] >= self.mfi_buy_slope.value) & # Money flow index
                (df['do_predict'] == 1) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['&s-up_or_down'] == 'up')
            ),
            ['enter_long', 'enter_tag']] = (1, 'TEMA/MF-AI')

        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] >  self.buy_rsi.value) &
                (df['rsi'] < 65) &
                (df['wave_t1'] > df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
                (df['wave_t1'] > df['wave_t2']) &
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['mfi_slope'] >= self.mfi_buy_slope.value) & # Money flow index 
                # (df['do_predict'] == 1) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['&s-up_or_down'] == 'up')
            ),
            ['enter_long', 'enter_tag']] = (1, 'WT/RSI/MF-AI')

        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] >  self.buy_rsi.value) &
                (df['rsi'] < 65) &
                (df['wave_t1'] > df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
                (df['wave_t1'] > df['wave_t2']) &
                (df['volume'] > 0) &  # Make sure Volume is not 0
                # (df['do_predict'] == 1) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['&s-up_or_down'] == 'up')
            ),
            ['enter_long', 'enter_tag']] = (1, 'WT/RSI-AI')

        # df.loc[
        #     (
        #         (df['wave_t1'] > df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
        #         (df['wave_t1'] > df['wave_t2']) &
        #         (df['volume'] > 0) &  # Make sure Volume is not 0
        #         # (df['do_predict'] == 1) &  # Make sure Freqai is confident in the prediction
        #         # Only enter trade if Freqai thinks the trend is in this direction
        #         (df['&s-up_or_down'] == 'up')
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'WT-AI')

        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] >  self.buy_rsi.value) &
                (df['rsi'] < 65) &
                (qtpylib.crossed_above(df['rsi'], df['rsi_ma'])) &
                (df['volume'] > 0) &  # Make sure Volume is not 0
                # (df['do_predict'] == 1) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['&s-up_or_down'] == 'up')
            ),
            ['enter_long', 'enter_tag']] = (1, 'RSI-XO-AI')

        ### Outlier Conditions

        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] >  self.buy_rsi.value) &
                (df['rsi'] < 65) &
                (df['tema'] > df['tema'].shift(1)) &  # Guard: tema is raising
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['mfi_slope'] >= self.mfi_buy_slope.value) & # Money flow index
                (df['do_predict'] == -2) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['&s-up_or_down'] == 'up')
            ),
            ['enter_long', 'enter_tag']] = (1, 'TEMA/RSI/MF-AI OL')

        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] >  self.buy_rsi.value) &
                (df['rsi'] < 65) &
                (df['tema'] > df['tema'].shift(1)) &  # Guard: tema is raising
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['do_predict'] == -2) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['&s-up_or_down'] == 'up')
            ),
            ['enter_long', 'enter_tag']] = (1, 'TEMA/RSI-AI-OL')

        df.loc[
            (
                (df['tema'] > df['tema'].shift(1)) &  # Guard: tema is raising
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['mfi_slope'] >= self.mfi_buy_slope.value) & # Money flow index
                (df['do_predict'] == -2) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['&s-up_or_down'] == 'up')
            ),
            ['enter_long', 'enter_tag']] = (1, 'TEMA/MF-AI-OL')

        df.loc[
            (
                (df['tema'] > df['tema'].shift(1)) &  # Guard: tema is raising
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['do_predict'] == -2) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['&s-up_or_down'] == 'up')
            ),
            ['enter_long', 'enter_tag']] = (1, 'TEMA-AI-OL')

        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] >  self.buy_rsi.value) &
                (df['rsi'] < 65) &
                (df['wave_t1'] > df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
                (df['wave_t1'] > df['wave_t2']) &
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['mfi_slope'] >= self.mfi_buy_slope.value) & # Money flow index 
                (df['do_predict'] == -2) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['&s-up_or_down'] == 'up')
            ),
            ['enter_long', 'enter_tag']] = (1, 'WT/RSI/MF-AI-OL')

        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] >  self.buy_rsi.value) &
                (df['rsi'] < 65) &
                (df['wave_t1'] > df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
                (df['wave_t1'] > df['wave_t2']) &
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['do_predict'] == -2) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['&s-up_or_down'] == 'up')
            ),
            ['enter_long', 'enter_tag']] = (1, 'WT/RSI-AI-OL')

        df.loc[
            (
                (df['wave_t1'] > df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
                (df['wave_t1'] > df['wave_t2']) &
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['do_predict'] == -2) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['&s-up_or_down'] == 'up')
            ),
            ['enter_long', 'enter_tag']] = (1, 'WT-AI-OL')

        ### NON A.I.
        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] >  self.buy_rsi.value) &
                (df['rsi'] < 55) &
                (qtpylib.crossed_above(df['rsi'], df['rsi_ma'])) &
                (df['wave_t1'] > df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
                (df['wave_t1'] > df['wave_t2']) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'WT/RSI')

        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] >  self.buy_rsi.value) &
                (df['rsi'] < 50) &
                (qtpylib.crossed_above(df['rsi'], df['rsi_ma'])) &
                (df['volume'] > 0) &  # Make sure Volume is not 0
                # (df['do_predict'] == 1) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['&s-up_or_down'] == 'up')
            ),
            ['enter_long', 'enter_tag']] = (1, 'RSI-XO')

        return df


    ### EXIT CONDITIONS ###
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df['rsi'] > self.sell_rsi.value) &
                (df['tema'] < df['tema'].shift(1)) &  # Guard: tema is falling
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['do_predict'] == 1) &  # Make sure Freqai is confident in the prediction
                # Only exit trade if Freqai thinks the trend is in this direction
                (df['&s-up_or_down'] == 'down')
            ),

            ['exit_long', 'exit_tag']] = (1, 'TEMA/RSI-AI')


        # df.loc[
        #     (
        #         (df['tema'] < df['tema'].shift(1)) &  # Guard: tema is falling
        #         (df['volume'] > 0)  # Make sure Volume is not 0
        #     ),

        #     ['exit_long', 'exit_tag']] = (1, 'TEMA')


        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] > self.sell_rsi.value) &
                (df['wave_t1'] < df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
                (df['wave_t1'] < df['wave_t2']) &
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['do_predict'] == 1) &  # Make sure Freqai is confident in the prediction
                # Only exitr trade if Freqai thinks the trend is in this direction
                (df['&s-up_or_down'] == 'down')
            ),
            ['exit_long', 'exit_tag']] = (1, 'WT/RSI-AI')

        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] > self.sell_rsi.value) &
                (df['wave_t1'] < df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
                (df['wave_t1'] < df['wave_t2']) &
                (df['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'WT/RSI')

        # df.loc[
        #     (
        #         (df['wave_t1'] < df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
        #         (df['wave_t1'] < df['wave_t2']) &
        #         (df['volume'] > 0) &  # Make sure Volume is not 0
        #         (df['do_predict'] == 1) &  # Make sure Freqai is confident in the prediction
        #         # Only exitr trade if Freqai thinks the trend is in this direction
        #         (df['&s-up_or_down'] == 'down')
        #     ),
        #     ['exit_long', 'exit_tag']] = (1, 'WT-AI')

        df.loc[
            (
                # Signal: RSI crosses above 30
                (df['rsi'] > self.sell_rsi.value) &
                (qtpylib.crossed_above(df['rsi_ma'], df['rsi'])) &
                (df['volume'] > 0) &  # Make sure Volume is not 0
                # (df['do_predict'] == 1) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                (df['&s-up_or_down'] == 'down')
            ),
            ['exit_long', 'exit_tag']] = (1, 'RSI-XO-AI')



        # df.loc[
        #     (
        #         (df['wave_t1'] < df['wave_t1'].shift(1)) &  # Guard: Wave 1 is raising
        #         (df['wave_t1'] < df['wave_t2']) &
        #         (df['volume'] > 0)   # Make sure Volume is not 0
        #     ),
        #     ['exit_long', 'exit_tag']] = (1, 'WT')

        return df
