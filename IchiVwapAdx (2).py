from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import DecimalParameter, IntParameter, BooleanParameter
from pandas import DataFrame
import talib.abstract as ta
import pandas as pd
import numpy as np
from datetime import datetime
from freqtrade.persistence import Trade
from typing import Optional, Dict, Any

pd.options.mode.chained_assignment = None


class IchiVwapAdx(IStrategy):
    """
    IchiVwapAdx Strategy - Multi-Confluence Trend Following with Heiken Ashi Enhancement and Dynamic Stack Size Management

    Strategy combining Ichimoku Cloud, ADX momentum, VWAP positioning, and Heiken Ashi smoothing
    for high-probability trend continuation trades with intelligent position sizing.

    Key Features:
    - Complete Ichimoku Cloud system (9-26-52 setup) with Chikou Span confirmation
    - Optional Heiken Ashi candles for smoother trend identification
    - ADX + Directional Movement for momentum confirmation
    - VWAP with standard deviation bands for positioning
    - Confluence scoring system (3-5 points required) with Boolean transition logic
    - Boolean transition logic prevents buying tops and selling bottoms
    - Optional Chikou span validation for signal quality
    - Multi-tier take-profit system
    - Dynamic stack size management with smart controls
    - Emergency exits for trend collapse
    - Spot-only trading support (set can_short = False)
    - Exchange-compatible integer leverage rounding

    Boolean Transition Logic:
    - Prevents entry when score has been maxed out for multiple bars
    - Only triggers entry when score reaches threshold AND wasn't there recently
    - Eliminates late entries during trend exhaustion phases
    - Significantly improves entry timing and reduces buying/selling tops

    Chikou Span Integration:
    - Uses Chikou span for final trend confirmation
    - Validates price is in "clear space" relative to historical action
    - Provides early exit signals when Chikou enters congested areas
    - Can be enabled/disabled via chikou_confirmation parameter

    Dynamic Stack Size Management:
    - Signal confidence scaling: Higher confluence scores → larger positions
    - Market regime alignment: Boost positions when aligned with dominant trend
    - Volatility-based scaling: Reduce size during high volatility periods
    - Momentum scaling: Strong ADX momentum allows larger positions
    - Portfolio heat protection: Automatic size reduction during drawdowns
    - Real-time risk overlays: Time-based, volatility circuit breakers
    - Configurable limits: Min/max stack sizes with safety bounds

    Spot Trading Compatibility:
    - Set can_short = False for spot-only trading platforms
    - Automatically disables all short signal generation and display
    - Maintains all long signal functionality
    """

    INTERFACE_VERSION = 3

    # Strategy settings
    timeframe = '15m'
    startup_candle_count: int = 200
    can_short: bool = True
    stoploss = -0.20  # Base stop loss (5%)
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # ROI table - Conservative
    minimal_roi = {
        "0": 0.20,   # 20% at any time
        "60": 0.15,  # 15% after 1 hour
        "120": 0.10, # 10% after 2 hours
        "240": 0.05, # 5% after 4 hours
        "480": 0.02  # 2% after 8 hours
    }

    # === Ichimoku Parameters ===
    ichimoku_tenkan_period = IntParameter(6, 15, default=9, space="buy", optimize=True)
    ichimoku_kijun_period = IntParameter(20, 35, default=26, space="buy", optimize=True)
    ichimoku_senkou_period = IntParameter(40, 65, default=52, space="buy", optimize=True)

    # === ADX Parameters ===
    adx_period = IntParameter(10, 20, default=14, space="buy", optimize=True)
    adx_threshold = DecimalParameter(15.0, 35.0, default=20.0, space="buy", optimize=True)
    adx_strong_threshold = DecimalParameter(25.0, 45.0, default=30.0, space="buy", optimize=False)

    # === VWAP Parameters ===
    vwap_std_multiplier = DecimalParameter(0.2, 1.0, default=0.5, space="buy", optimize=True)
    vwap_period = IntParameter(10, 30, default=20, space="buy", optimize=True)

    # === Entry Scoring Parameters ===
    ichimoku_weight = IntParameter(2, 3, default=2, space="buy", optimize=False)
    adx_weight = IntParameter(1, 2, default=1, space="buy", optimize=False)
    vwap_weight = IntParameter(1, 2, default=1, space="buy", optimize=False)
    chikou_weight = IntParameter(1, 2, default=1, space="buy", optimize=False)
    min_entry_score = IntParameter(3, 5, default=3, space="buy", optimize=False)

    # === Chikou Span Parameters ===
    chikou_confirmation = BooleanParameter(default=True, space="buy", optimize=False)
    chikou_clear_space_periods = IntParameter(4, 12, default=5, space="buy", optimize=True)

    # === Risk Management Parameters ===
    atr_period = IntParameter(10, 20, default=14, space="sell", optimize=True)
    atr_multiplier = DecimalParameter(1.0, 2.5, default=1.5, space="sell", optimize=True)
    cloud_quality_lookback = IntParameter(15, 30, default=20, space="buy", optimize=False)
    cloud_quality_percentile = DecimalParameter(15.0, 30.0, default=20.0, space="buy", optimize=False)

    # === Dynamic Stack Size Parameters ===
    base_position_size = DecimalParameter(1.0, 2.0, default=1.0, space="buy", optimize=False)

    # Smart Stack Size Controls
    confidence_scaling = BooleanParameter(default=True, space="buy", optimize=False)
    market_regime_scaling = BooleanParameter(default=True, space="buy", optimize=False)
    volatility_scaling = BooleanParameter(default=True, space="buy", optimize=False)
    momentum_scaling = BooleanParameter(default=True, space="buy", optimize=False)

    # Stack Size Limits (Futures-Compatible: Must be >= 1.0)
    min_stack_size = DecimalParameter(1.0, 1.5, default=1.0, space="buy", optimize=False)
    max_stack_size = DecimalParameter(2.0, 5.0, default=3.0, space="buy", optimize=False)

    # Risk Management for Stack Sizing
    max_portfolio_risk = DecimalParameter(0.05, 0.20, default=0.10, space="buy", optimize=False)
    stack_heat_threshold = DecimalParameter(0.15, 0.35, default=0.25, space="buy", optimize=False)

    # Dynamic Scaling Parameters
    confidence_boost_multiplier = DecimalParameter(1.2, 2.0, default=1.5, space="buy", optimize=False)
    regime_alignment_boost = DecimalParameter(1.1, 1.8, default=1.3, space="buy", optimize=False)
    volatility_penalty_factor = DecimalParameter(0.3, 0.8, default=0.6, space="buy", optimize=False)
    momentum_boost_threshold = DecimalParameter(30.0, 50.0, default=35.0, space="buy", optimize=False)

    # Time-of-Day Risk Management Parameters (UTC Hours)
    enable_time_risk_management = BooleanParameter(default=True, space="buy", optimize=False)
    high_risk_hours_start_1 = IntParameter(6, 10, default=8, space="buy", optimize=False)
    high_risk_hours_end_1 = IntParameter(9, 11, default=10, space="buy", optimize=False)
    high_risk_hours_start_2 = IntParameter(14, 16, default=15, space="buy", optimize=False)
    high_risk_hours_end_2 = IntParameter(16, 18, default=17, space="buy", optimize=False)
    time_risk_multiplier = DecimalParameter(0.5, 0.9, default=0.8, space="buy", optimize=False)

    # Volatility Circuit Breaker Parameters
    extreme_volatility_threshold = DecimalParameter(1.8, 2.5, default=2.0, space="buy", optimize=False)
    high_volatility_threshold = DecimalParameter(1.5, 2.0, default=1.8, space="buy", optimize=False)
    extreme_volatility_multiplier = DecimalParameter(0.3, 0.6, default=0.5, space="buy", optimize=False)
    high_volatility_multiplier = DecimalParameter(0.6, 0.8, default=0.7, space="buy", optimize=False)

    # Dynamic Volatility Scaling Parameters (for position sizing)
    dynamic_high_volatility_threshold = DecimalParameter(1.2, 2.0, default=1.5, space="buy", optimize=False)
    dynamic_low_volatility_threshold = DecimalParameter(0.5, 1.0, default=0.7, space="buy", optimize=False)
    low_volatility_boost_multiplier = DecimalParameter(1.1, 1.5, default=1.2, space="buy", optimize=False)

    # === Exit Parameters ===
    emergency_adx_threshold = DecimalParameter(10.0, 20.0, default=15.0, space="sell", optimize=False)
    emergency_adx_slope_threshold = DecimalParameter(-10.0, -2.0, default=-5.0, space="sell", optimize=False)
    cloud_exit_periods = IntParameter(2, 4, default=3, space="sell", optimize=True)
    take_profit_1 = DecimalParameter(1.5, 5.0, default=2.0, space="sell", optimize=True)
    take_profit_2 = DecimalParameter(4.5, 10, default=5, space="sell", optimize=True)
    min_profit_exit_threshold = DecimalParameter(0.005, 0.02, default=0.01, space="sell", optimize=True)
    vwap_exit_multiplier = DecimalParameter(1.5, 3.0, default=2.0, space="sell", optimize=True)

    # Multi-tier take profit parameters
    high_volatility_tp_reduction = DecimalParameter(0.7, 0.9, default=0.8, space="sell", optimize=False)
    strong_momentum_tp_boost = DecimalParameter(1.05, 1.2, default=1.1, space="sell", optimize=False)
    weak_momentum_threshold = DecimalParameter(20.0, 30.0, default=25.0, space="sell", optimize=False)
    momentum_decline_threshold = DecimalParameter(-5.0, -1.0, default=-2.0, space="sell", optimize=False)

    # === Filter Parameters ===
    volatility_filter = BooleanParameter(default=True, space="buy", optimize=False)
    min_volatility_ratio = DecimalParameter(0.3, 0.7, default=0.5, space="buy", optimize=False)

    # === Heiken Ashi Parameters ===
    use_heiken_ashi = BooleanParameter(default=True, space="buy", optimize=False)
    ha_smoothing_factor = DecimalParameter(0.1, 0.5, default=0.2, space="buy", optimize=True)

    # === Transition Logic Parameters ===
    transition_lookback = IntParameter(5, 25, default=9, space="buy", optimize=True)

    # === Heiken Ashi Magic Number Parameters ===
    ha_body_atr_ratio = DecimalParameter(0.05, 0.3, default=0.1, space="buy", optimize=True)

    # === Custom Stoploss Parameters ===
    trailing_profit_multiplier = DecimalParameter(1.0, 3.0, default=1.5, space="sell", optimize=True)
    volatility_trail_threshold = DecimalParameter(1.4, 2.0, default=1.6, space="sell", optimize=True)
    max_trail_tightening = DecimalParameter(0.3, 0.8, default=0.5, space="sell", optimize=True)

    # === Signal Confidence Parameters ===
    high_confidence_threshold = DecimalParameter(0.7, 0.9, default=0.8, space="buy", optimize=False)
    confidence_scaling_factor = DecimalParameter(0.3, 0.7, default=0.5, space="buy", optimize=False)

    # === Market Regime Scaling Parameters ===
    bear_market_long_reduction = DecimalParameter(0.5, 0.8, default=0.7, space="buy", optimize=False)
    sideways_market_reduction = DecimalParameter(0.6, 0.9, default=0.8, space="buy", optimize=False)
    bull_market_short_reduction = DecimalParameter(0.5, 0.8, default=0.7, space="sell", optimize=False)

    # === Momentum Multiplier Cap Parameters ===
    max_momentum_multiplier = DecimalParameter(1.3, 2.0, default=1.5, space="buy", optimize=False)
    momentum_scaling_divisor = DecimalParameter(50.0, 150.0, default=100.0, space="buy", optimize=False)

    # === Exit Signal Conviction Parameters ===
    exit_adx_conviction_threshold = DecimalParameter(20.0, 30.0, default=25.0, space="sell", optimize=False)

    # === Dynamic ROI Parameters ===
    # Core ROI scaling factors - use mathematical formulas instead of lookup tables
    roi_base_scaling = DecimalParameter(0.8, 1.2, default=1.0, space="sell", optimize=True)
    roi_volatility_sensitivity = DecimalParameter(0.3, 0.8, default=0.5, space="sell", optimize=True)
    roi_regime_sensitivity = DecimalParameter(0.2, 0.6, default=0.4, space="sell", optimize=False)
    roi_momentum_sensitivity = DecimalParameter(0.1, 0.4, default=0.2, space="sell", optimize=False)
    roi_confidence_sensitivity = DecimalParameter(0.1, 0.3, default=0.2, space="sell", optimize=False)

    # ROI Safety Bounds
    min_dynamic_roi = DecimalParameter(0.005, 0.020, default=0.01, space="sell", optimize=False)
    max_dynamic_roi = DecimalParameter(0.40, 0.60, default=0.50, space="sell", optimize=False)

    # === Portfolio Heat Management Parameters ===
    portfolio_heat_reduction_threshold = DecimalParameter(0.3, 0.7, default=0.5, space="buy", optimize=False)
    performance_multiplier_placeholder = DecimalParameter(0.8, 1.2, default=1.0, space="buy", optimize=False)

    # === Market Regime Strength Parameters ===
    regime_price_weight = DecimalParameter(0.5, 0.7, default=0.6, space="buy", optimize=False)
    regime_sma_weight = DecimalParameter(0.3, 0.5, default=0.4, space="buy", optimize=False)

    # === Market Regime Detection Parameters ===
    regime_sma_fast = IntParameter(20, 50, default=30, space="buy", optimize=False)
    regime_sma_slow = IntParameter(100, 200, default=150, space="buy", optimize=False)
    regime_lookback = IntParameter(10, 30, default=20, space="buy", optimize=False)
    regime_threshold = DecimalParameter(0.6, 0.9, default=0.75, space="buy", optimize=False)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate all technical indicators"""

        # === ATR for Volatility (Calculate first for HA dependency) ===
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)
        dataframe['atr_50'] = ta.ATR(dataframe, timeperiod=50)
        dataframe['volatility_ratio'] = dataframe['atr'] / dataframe['atr_50']

        # === Heiken Ashi Calculation ===
        if self.use_heiken_ashi.value:
            dataframe = self.calculate_heiken_ashi(dataframe)

        # === Ichimoku Cloud ===
        ichimoku = self.calculate_ichimoku(dataframe)
        dataframe['tenkan'] = ichimoku['tenkan']
        dataframe['kijun'] = ichimoku['kijun']
        dataframe['senkou_a'] = ichimoku['senkou_a']
        dataframe['senkou_b'] = ichimoku['senkou_b']
        dataframe['chikou'] = ichimoku['chikou']
        dataframe['cloud_top'] = np.maximum(dataframe['senkou_a'], dataframe['senkou_b'])
        dataframe['cloud_bottom'] = np.minimum(dataframe['senkou_a'], dataframe['senkou_b'])
        dataframe['cloud_thickness'] = dataframe['cloud_top'] - dataframe['cloud_bottom']

        # === Heiken Ashi Enhanced Ichimoku (if enabled) ===
        if self.use_heiken_ashi.value:
            ha_ichimoku = self.calculate_ichimoku_ha(dataframe)
            dataframe['ha_tenkan'] = ha_ichimoku['tenkan']
            dataframe['ha_kijun'] = ha_ichimoku['kijun']
            dataframe['ha_senkou_a'] = ha_ichimoku['senkou_a']
            dataframe['ha_senkou_b'] = ha_ichimoku['senkou_b']
            dataframe['ha_chikou'] = ha_ichimoku['chikou']
            dataframe['ha_cloud_top'] = np.maximum(dataframe['ha_senkou_a'], dataframe['ha_senkou_b'])
            dataframe['ha_cloud_bottom'] = np.minimum(dataframe['ha_senkou_a'], dataframe['ha_senkou_b'])
            dataframe['ha_cloud_thickness'] = dataframe['ha_cloud_top'] - dataframe['ha_cloud_bottom']

        # === ADX and Directional Movement ===
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_period.value)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=self.adx_period.value)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=self.adx_period.value)
        dataframe['adx_slope'] = dataframe['adx'].diff(3)

        # === VWAP with Standard Deviation Bands ===
        vwap_data = self.calculate_vwap_bands(dataframe)
        dataframe['vwap'] = vwap_data['vwap']
        dataframe['vwap_std'] = vwap_data['vwap_std']
        dataframe['vwap_upper'] = dataframe['vwap'] + (self.vwap_std_multiplier.value * dataframe['vwap_std'])
        dataframe['vwap_lower'] = dataframe['vwap'] - (self.vwap_std_multiplier.value * dataframe['vwap_std'])

        # === Cloud Quality Metrics ===
        dataframe['cloud_thickness_median'] = dataframe['cloud_thickness'].rolling(
            window=self.cloud_quality_lookback.value
        ).quantile(self.cloud_quality_percentile.value / 100.0)

        # Cloud strength calculation for ROI
        dataframe['cloud_strength'] = np.where(
            dataframe['cloud_thickness_median'] > 0,
            np.clip(dataframe['cloud_thickness'] / dataframe['cloud_thickness_median'], 0.0, 2.0) / 2.0,
            0.5  # Default neutral strength
        )

        # === Market Regime Detection ===
        dataframe = self.calculate_market_regime(dataframe)

        # === Signal Components ===
        dataframe = self.calculate_signal_components(dataframe)

        # === Dynamic Stack Size Calculations ===
        dataframe = self.calculate_dynamic_stack_size(dataframe)

        return dataframe

    def calculate_ichimoku(self, dataframe: DataFrame) -> dict:
        """Calculate Ichimoku Cloud components"""
        high = dataframe['high']
        low = dataframe['low']
        close = dataframe['close']

        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=self.ichimoku_tenkan_period.value).max()
        tenkan_low = low.rolling(window=self.ichimoku_tenkan_period.value).min()
        tenkan = (tenkan_high + tenkan_low) / 2

        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=self.ichimoku_kijun_period.value).max()
        kijun_low = low.rolling(window=self.ichimoku_kijun_period.value).min()
        kijun = (kijun_high + kijun_low) / 2

        # Senkou Span A (Leading Span A)
        senkou_a = ((tenkan + kijun) / 2).shift(self.ichimoku_kijun_period.value)

        # Senkou Span B (Leading Span B)
        senkou_high = high.rolling(window=self.ichimoku_senkou_period.value).max()
        senkou_low = low.rolling(window=self.ichimoku_senkou_period.value).min()
        senkou_b = ((senkou_high + senkou_low) / 2).shift(self.ichimoku_kijun_period.value)

        # Chikou Span (Lagging Span)
        chikou = close.shift(-self.ichimoku_kijun_period.value)

        return {
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'chikou': chikou
        }

    def calculate_vwap_bands(self, dataframe: DataFrame) -> dict:
        """Calculate VWAP with standard deviation bands"""
        high = dataframe['high']
        low = dataframe['low']
        close = dataframe['close']
        volume = dataframe['volume']

        # Typical Price
        typical_price = (high + low + close) / 3

        # VWAP calculation
        vwap_num = (typical_price * volume).rolling(window=self.vwap_period.value).sum()
        vwap_den = volume.rolling(window=self.vwap_period.value).sum()
        vwap = vwap_num / vwap_den

        # Standard deviation calculation
        price_diff_sq = ((typical_price - vwap) ** 2 * volume).rolling(window=self.vwap_period.value).sum()
        vwap_var = price_diff_sq / vwap_den
        vwap_std = np.sqrt(vwap_var)

        return {
            'vwap': vwap,
            'vwap_std': vwap_std
        }

    def calculate_heiken_ashi(self, dataframe: DataFrame) -> DataFrame:
        """
        Calculate Heiken Ashi candles for smoother trend analysis

        Heiken Ashi candles smooth out price action and reduce noise,
        making trend identification and reversal detection more reliable.
        """
        # Initialize with first candle
        ha_close = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        ha_open = pd.Series(index=dataframe.index, dtype='float64')
        ha_open.iloc[0] = (dataframe['open'].iloc[0] + dataframe['close'].iloc[0]) / 2

        # Calculate HA open using iterative approach for accuracy
        for i in range(1, len(dataframe)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2

        # Apply smoothing factor for additional noise reduction
        if self.ha_smoothing_factor.value > 0:
            ha_open = ha_open.ewm(alpha=self.ha_smoothing_factor.value).mean()
            ha_close = ha_close.ewm(alpha=self.ha_smoothing_factor.value).mean()

        # Calculate HA high and low
        ha_high = np.maximum.reduce([dataframe['high'], ha_open, ha_close])
        ha_low = np.minimum.reduce([dataframe['low'], ha_open, ha_close])

        # Store HA values
        dataframe['ha_open'] = ha_open
        dataframe['ha_high'] = ha_high
        dataframe['ha_low'] = ha_low
        dataframe['ha_close'] = ha_close

        # Calculate HA trend signals
        dataframe['ha_green'] = ha_close > ha_open
        dataframe['ha_red'] = ha_close < ha_open
        dataframe['ha_doji'] = np.abs(ha_close - ha_open) < (dataframe['atr'] * self.ha_body_atr_ratio.value)

        # HA trend strength indicators
        dataframe['ha_body_size'] = np.abs(ha_close - ha_open)
        dataframe['ha_upper_wick'] = ha_high - np.maximum(ha_open, ha_close)
        dataframe['ha_lower_wick'] = np.minimum(ha_open, ha_close) - ha_low

        # HA trend consistency (consecutive same color candles)
        dataframe['ha_trend_consecutive'] = (
            dataframe['ha_green'].astype(int).groupby(
                (dataframe['ha_green'] != dataframe['ha_green'].shift()).cumsum()
            ).cumsum()
        )

        return dataframe

    def calculate_ichimoku_ha(self, dataframe: DataFrame) -> dict:
        """
        Calculate Ichimoku Cloud components using Heiken Ashi candles
        for smoother trend analysis and reduced false signals
        """
        ha_high = dataframe['ha_high']
        ha_low = dataframe['ha_low']
        ha_close = dataframe['ha_close']

        # Tenkan-sen (Conversion Line) - HA version
        ha_tenkan_high = ha_high.rolling(window=self.ichimoku_tenkan_period.value).max()
        ha_tenkan_low = ha_low.rolling(window=self.ichimoku_tenkan_period.value).min()
        ha_tenkan = (ha_tenkan_high + ha_tenkan_low) / 2

        # Kijun-sen (Base Line) - HA version
        ha_kijun_high = ha_high.rolling(window=self.ichimoku_kijun_period.value).max()
        ha_kijun_low = ha_low.rolling(window=self.ichimoku_kijun_period.value).min()
        ha_kijun = (ha_kijun_high + ha_kijun_low) / 2

        # Senkou Span A (Leading Span A) - HA version
        ha_senkou_a = ((ha_tenkan + ha_kijun) / 2).shift(self.ichimoku_kijun_period.value)

        # Senkou Span B (Leading Span B) - HA version
        ha_senkou_high = ha_high.rolling(window=self.ichimoku_senkou_period.value).max()
        ha_senkou_low = ha_low.rolling(window=self.ichimoku_senkou_period.value).min()
        ha_senkou_b = ((ha_senkou_high + ha_senkou_low) / 2).shift(self.ichimoku_kijun_period.value)

        # Chikou Span (Lagging Span) - HA version
        ha_chikou = ha_close.shift(-self.ichimoku_kijun_period.value)

        return {
            'tenkan': ha_tenkan,
            'kijun': ha_kijun,
            'senkou_a': ha_senkou_a,
            'senkou_b': ha_senkou_b,
            'chikou': ha_chikou
        }

    def calculate_market_regime(self, dataframe: DataFrame) -> DataFrame:
        """
        Detect market regime (bull/bear/sideways) using multiple indicators

        Bull Market: Price consistently above fast SMA, fast SMA > slow SMA
        Bear Market: Price consistently below fast SMA, fast SMA < slow SMA
        Sideways: Mixed signals or choppy conditions
        """
        # Calculate regime SMAs
        dataframe['regime_sma_fast'] = ta.SMA(dataframe, timeperiod=self.regime_sma_fast.value)
        dataframe['regime_sma_slow'] = ta.SMA(dataframe, timeperiod=self.regime_sma_slow.value)

        # Basic trend conditions
        dataframe['price_above_fast_sma'] = dataframe['close'] > dataframe['regime_sma_fast']
        dataframe['fast_above_slow_sma'] = dataframe['regime_sma_fast'] > dataframe['regime_sma_slow']

        # Calculate percentage of recent periods where conditions are met
        lookback = self.regime_lookback.value

        dataframe['bull_strength'] = (
            dataframe['price_above_fast_sma'].rolling(window=lookback).mean() * self.regime_price_weight.value +
            dataframe['fast_above_slow_sma'].rolling(window=lookback).mean() * self.regime_sma_weight.value
        )

        dataframe['bear_strength'] = (
            (~dataframe['price_above_fast_sma']).rolling(window=lookback).mean() * self.regime_price_weight.value +
            (~dataframe['fast_above_slow_sma']).rolling(window=lookback).mean() * self.regime_sma_weight.value
        )

        # Determine regime based on threshold
        threshold = self.regime_threshold.value

        dataframe['is_bull_market'] = dataframe['bull_strength'] > threshold
        dataframe['is_bear_market'] = dataframe['bear_strength'] > threshold
        dataframe['is_sideways_market'] = (
            (dataframe['bull_strength'] <= threshold) &
            (dataframe['bear_strength'] <= threshold)
        )

        # Market regime flag for shorting
        dataframe['shorts_allowed'] = ~dataframe['is_bull_market']  # Allow shorts in bear/sideways markets

        return dataframe

    def calculate_signal_components(self, dataframe: DataFrame) -> DataFrame:
        """Calculate individual signal components and confluence score"""

        # === Chikou Span Signals ===
        # Fill NaN values in chikou for robust calculation
        dataframe['chikou_filled'] = dataframe['chikou'].bfill().ffill()

        # Chikou above/below price from kijun periods ago indicates trend strength
        price_kijun_ago = dataframe['close'].shift(self.ichimoku_kijun_period.value)
        chikou_vs_price_bull = dataframe['chikou_filled'] > price_kijun_ago
        chikou_vs_price_bear = dataframe['chikou_filled'] < price_kijun_ago

        dataframe['chikou_bullish'] = chikou_vs_price_bull.fillna(False)
        dataframe['chikou_bearish'] = chikou_vs_price_bear.fillna(False)

        # Chikou clear space: not overlapping with recent price action
        if self.chikou_confirmation.value:
            # Get high/low from kijun periods ago
            high_kijun_ago = dataframe['high'].shift(self.ichimoku_kijun_period.value)
            low_kijun_ago = dataframe['low'].shift(self.ichimoku_kijun_period.value)

            # Check if chikou is in clear space (above recent highs or below recent lows)
            recent_high = high_kijun_ago.rolling(window=self.chikou_clear_space_periods.value).max()
            recent_low = low_kijun_ago.rolling(window=self.chikou_clear_space_periods.value).min()

            chikou_above_high = dataframe['chikou_filled'] > recent_high
            chikou_below_low = dataframe['chikou_filled'] < recent_low
            chikou_clear_condition = chikou_above_high | chikou_below_low
            dataframe['chikou_clear_space'] = chikou_clear_condition.fillna(False)
        else:
            dataframe['chikou_clear_space'] = True

        # === Enhanced Ichimoku Signals with Heiken Ashi and Chikou ===
        # Choose which data to use for signal generation
        if self.use_heiken_ashi.value:
            # Use Heiken Ashi for smoother signal generation
            signal_close = dataframe['ha_close']
            signal_cloud_top = dataframe['ha_cloud_top']
            signal_cloud_bottom = dataframe['ha_cloud_bottom']
            signal_cloud_thickness = dataframe['ha_cloud_thickness']
            signal_cloud_median = dataframe['ha_cloud_thickness'].rolling(
                window=self.cloud_quality_lookback.value
            ).quantile(self.cloud_quality_percentile.value / 100.0)

            # Additional HA trend confirmation
            ha_bullish_trend = dataframe['ha_green'] & (dataframe['ha_body_size'] > dataframe['atr'] * self.ha_body_atr_ratio.value)
            ha_bearish_trend = dataframe['ha_red'] & (dataframe['ha_body_size'] > dataframe['atr'] * self.ha_body_atr_ratio.value)
        else:
            # Use regular candles
            signal_close = dataframe['close']
            signal_cloud_top = dataframe['cloud_top']
            signal_cloud_bottom = dataframe['cloud_bottom']
            signal_cloud_thickness = dataframe['cloud_thickness']
            signal_cloud_median = dataframe['cloud_thickness_median']
            ha_bullish_trend = True
            ha_bearish_trend = True

        base_ichimoku_bullish = (
            (signal_close > signal_cloud_top) &
            (signal_cloud_thickness > signal_cloud_median) &
            ha_bullish_trend
        )
        base_ichimoku_bearish = (
            (signal_close < signal_cloud_bottom) &
            (signal_cloud_thickness > signal_cloud_median) &
            ha_bearish_trend
        )

        if self.chikou_confirmation.value:
            dataframe['ichimoku_bullish'] = (
                base_ichimoku_bullish &
                dataframe['chikou_bullish'] &
                dataframe['chikou_clear_space']
            )
            dataframe['ichimoku_bearish'] = (
                base_ichimoku_bearish &
                dataframe['chikou_bearish'] &
                dataframe['chikou_clear_space']
            )
        else:
            dataframe['ichimoku_bullish'] = base_ichimoku_bullish
            dataframe['ichimoku_bearish'] = base_ichimoku_bearish

        # === ADX Signals ===
        dataframe['adx_strong'] = (
            (dataframe['adx'] > self.adx_threshold.value) &
            (dataframe['adx_slope'] > 0)
        )
        dataframe['adx_bullish'] = (
            dataframe['adx_strong'] &
            (dataframe['plus_di'] > dataframe['minus_di'])
        )
        dataframe['adx_bearish'] = (
            dataframe['adx_strong'] &
            (dataframe['minus_di'] > dataframe['plus_di'])
        )

        # === VWAP Signals ===
        dataframe['vwap_bullish'] = dataframe['close'] > dataframe['vwap_upper']
        dataframe['vwap_bearish'] = dataframe['close'] < dataframe['vwap_lower']

        # === Volatility Filter ===
        dataframe['volatility_ok'] = (
            ~self.volatility_filter.value |
            (dataframe['volatility_ratio'] > self.min_volatility_ratio.value)
        )

        # === Enhanced Confluence Scoring with Chikou ===
        chikou_long_points = 0
        chikou_short_points = 0

        if self.chikou_confirmation.value:
            chikou_long_points = (dataframe['chikou_bullish'].astype(int) * self.chikou_weight.value)
            chikou_short_points = (dataframe['chikou_bearish'].astype(int) * self.chikou_weight.value)

        dataframe['long_score'] = (
            (dataframe['ichimoku_bullish'].astype(int) * self.ichimoku_weight.value) +
            (dataframe['adx_bullish'].astype(int) * self.adx_weight.value) +
            (dataframe['vwap_bullish'].astype(int) * self.vwap_weight.value) +
            chikou_long_points
        )

        dataframe['short_score'] = (
            (dataframe['ichimoku_bearish'].astype(int) * self.ichimoku_weight.value) +
            (dataframe['adx_bearish'].astype(int) * self.adx_weight.value) +
            (dataframe['vwap_bearish'].astype(int) * self.vwap_weight.value) +
            chikou_short_points
        )

        # === Boolean Transition Logic ===
        # Detect score transitions to prevent buying/selling tops/bottoms

        # Check if score just reached threshold (wasn't there in previous bars)
        lookback = self.transition_lookback.value

        # For long signals: score >= threshold AND was below threshold recently
        long_score_current = dataframe['long_score'] >= self.min_entry_score.value
        long_score_was_low = dataframe['long_score'].shift(1).rolling(window=lookback).max() < self.min_entry_score.value
        dataframe['long_score_transition'] = long_score_current & long_score_was_low

        # For short signals: score >= threshold AND was below threshold recently
        short_score_current = dataframe['short_score'] >= self.min_entry_score.value
        short_score_was_low = dataframe['short_score'].shift(1).rolling(window=lookback).max() < self.min_entry_score.value
        dataframe['short_score_transition'] = short_score_current & short_score_was_low

        return dataframe

    def calculate_dynamic_stack_size(self, dataframe: DataFrame) -> DataFrame:
        """
        Calculate dynamic stack size based on multiple market conditions and confidence factors

        This method implements smart position sizing that adapts to:
        1. Signal confidence (confluence score quality)
        2. Market regime alignment (bull/bear/sideways)
        3. Volatility conditions (high volatility = smaller positions)
        4. Momentum strength (ADX-based scaling)
        5. Risk management overlays (portfolio heat, drawdown protection)
        """

        # === Base Stack Size Calculation ===
        dataframe['base_stack_size'] = self.base_position_size.value

        # === 1. Confidence-Based Scaling ===
        if self.confidence_scaling.value:
            # Higher confluence scores get larger position sizes
            max_possible_score = (
                self.ichimoku_weight.value +
                self.adx_weight.value +
                self.vwap_weight.value +
                (self.chikou_weight.value if self.chikou_confirmation.value else 0)
            )

            # Normalize scores to 0-1 range
            long_confidence = dataframe['long_score'] / max_possible_score
            short_confidence = dataframe['short_score'] / max_possible_score

            # Apply confidence boost for high-quality signals
            long_confidence_multiplier = np.where(
                long_confidence > self.high_confidence_threshold.value,  # High confidence threshold
                self.confidence_boost_multiplier.value,
                1.0 + (long_confidence * self.confidence_scaling_factor.value)  # Gradual scaling
            )

            short_confidence_multiplier = np.where(
                short_confidence > self.high_confidence_threshold.value,  # High confidence threshold
                self.confidence_boost_multiplier.value,
                1.0 + (short_confidence * self.confidence_scaling_factor.value)  # Gradual scaling
            )

            dataframe['long_confidence_mult'] = long_confidence_multiplier
            dataframe['short_confidence_mult'] = short_confidence_multiplier
        else:
            dataframe['long_confidence_mult'] = 1.0
            dataframe['short_confidence_mult'] = 1.0

        # === 2. Market Regime Scaling ===
        if self.market_regime_scaling.value:
            # Separate regime multipliers for long and short positions

            # Long regime multiplier: boost longs in bull market, reduce in bear market
            long_regime_multiplier = np.where(
                dataframe['is_bull_market'],  # Bull market
                self.regime_alignment_boost.value,  # Boost longs in bull market
                np.where(
                    dataframe['is_bear_market'],  # Bear market
                    self.bear_market_long_reduction.value,  # Reduce longs in bear market
                    self.sideways_market_reduction.value   # Reduce longs in sideways markets
                )
            )

            # Short regime multiplier: boost shorts in bear market, reduce in bull market
            short_regime_multiplier = np.where(
                dataframe['is_bear_market'],  # Bear market
                self.regime_alignment_boost.value,  # Boost shorts in bear market
                np.where(
                    dataframe['is_bull_market'],  # Bull market
                    self.bull_market_short_reduction.value,  # Reduce shorts in bull market
                    self.sideways_market_reduction.value   # Reduce shorts in sideways markets
                )
            )

            dataframe['long_regime_mult'] = long_regime_multiplier
            dataframe['short_regime_mult'] = short_regime_multiplier
        else:
            dataframe['long_regime_mult'] = 1.0
            dataframe['short_regime_mult'] = 1.0

        # === 3. Volatility-Based Scaling ===
        if self.volatility_scaling.value:
            # Higher volatility = smaller positions (risk management)
            volatility_multiplier = np.where(
                dataframe['volatility_ratio'] > self.dynamic_high_volatility_threshold.value,
                self.volatility_penalty_factor.value,
                np.where(
                    dataframe['volatility_ratio'] < self.dynamic_low_volatility_threshold.value,
                    self.low_volatility_boost_multiplier.value,
                    1.0   # Normal volatility
                )
            )
            dataframe['volatility_mult'] = volatility_multiplier
        else:
            dataframe['volatility_mult'] = 1.0

        # === 4. Momentum-Based Scaling ===
        if self.momentum_scaling.value:
            # Strong momentum (high ADX) allows larger positions
            momentum_multiplier = np.where(
                dataframe['adx'] > self.momentum_boost_threshold.value,
                1.0 + ((dataframe['adx'] - self.momentum_boost_threshold.value) / self.momentum_scaling_divisor.value),
                1.0
            )
            # Cap the momentum boost
            momentum_multiplier = np.minimum(momentum_multiplier, self.max_momentum_multiplier.value)
            dataframe['momentum_mult'] = momentum_multiplier
        else:
            dataframe['momentum_mult'] = 1.0

        # === 5. Calculate Combined Stack Size ===
        # For long positions
        dataframe['dynamic_long_stack'] = (
            dataframe['base_stack_size'] *
            dataframe['long_confidence_mult'] *
            dataframe['long_regime_mult'] *
            dataframe['volatility_mult'] *
            dataframe['momentum_mult']
        )

        # For short positions
        dataframe['dynamic_short_stack'] = (
            dataframe['base_stack_size'] *
            dataframe['short_confidence_mult'] *
            dataframe['short_regime_mult'] *
            dataframe['volatility_mult'] *
            dataframe['momentum_mult']
        )

        # === 6. Apply Stack Size Limits ===
        dataframe['dynamic_long_stack'] = np.clip(
            dataframe['dynamic_long_stack'],
            self.min_stack_size.value,
            self.max_stack_size.value
        )

        dataframe['dynamic_short_stack'] = np.clip(
            dataframe['dynamic_short_stack'],
            self.min_stack_size.value,
            self.max_stack_size.value
        )

        # === 7. Portfolio Heat Protection ===
        # Calculate recent drawdown impact (simplified)
        # In a real implementation, you'd get this from portfolio status
        dataframe['portfolio_heat_reduction'] = 1.0  # Placeholder for portfolio heat calculation

        # Apply heat reduction if needed
        dataframe['final_long_stack'] = (
            dataframe['dynamic_long_stack'] * dataframe['portfolio_heat_reduction']
        )
        dataframe['final_short_stack'] = (
            dataframe['dynamic_short_stack'] * dataframe['portfolio_heat_reduction']
        )

        # === 8. Signal Quality Gate ===
        # Only apply dynamic sizing when signals are actually present
        dataframe['final_long_stack'] = np.where(
            dataframe['long_score_transition'],
            dataframe['final_long_stack'],
            0.0  # No position if no signal
        )

        dataframe['final_short_stack'] = np.where(
            dataframe['short_score_transition'],
            dataframe['final_short_stack'],
            0.0  # No position if no signal
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Determine entry signals based on confluence scoring with transition logic"""

        # Long entry conditions - FIXED: Use transition logic to prevent buying tops
        dataframe.loc[
            (
                (dataframe['long_score_transition']) &  # Boolean transition instead of rolling high scores
                (dataframe['ichimoku_bullish']) &  # Ichimoku is mandatory
                (dataframe['volatility_ok']) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1

        # Short entry conditions - only if shorting is enabled and market allows
        if self.can_short:
            dataframe.loc[
                (
                    (dataframe['short_score_transition']) &  # Boolean transition instead of rolling high scores
                    (dataframe['ichimoku_bearish']) &  # Ichimoku is mandatory
                    (dataframe['volatility_ok']) &
                    (dataframe['volume'] > 0) &
                    (dataframe['shorts_allowed'])  # Only allow shorts in bear/sideways markets
                ),
                'enter_short'
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Determine exit signals with Chikou confirmation"""

        # Chikou-based exit signals (only when chikou confirmation is enabled)
        chikou_exit_long = False
        chikou_exit_short = False

        if self.chikou_confirmation.value:
            # Chikou entering congested price area signals potential reversal
            chikou_exit_long = (
                (~dataframe['chikou_clear_space']) |  # Chikou entering congested area
                (dataframe['chikou_bearish'])         # Chikou turning bearish
            )
            chikou_exit_short = (
                (~dataframe['chikou_clear_space']) |  # Chikou entering congested area
                (dataframe['chikou_bullish'])         # Chikou turning bullish
            )

        # Primary exit: Strong trend reversal with confluence (enhanced with Chikou)
        strong_bearish_reversal = (
            (dataframe['ichimoku_bearish']) &
            (dataframe['adx'] > self.exit_adx_conviction_threshold.value) &  # Higher threshold for more conviction
            (dataframe['minus_di'] > dataframe['plus_di']) &  # DI confirmation
            (dataframe['close'] < dataframe['vwap'])
        )

        strong_bullish_reversal = (
            (dataframe['ichimoku_bullish']) &
            (dataframe['adx'] > self.exit_adx_conviction_threshold.value) &  # Higher threshold for more conviction
            (dataframe['plus_di'] > dataframe['minus_di']) &  # DI confirmation
            (dataframe['close'] > dataframe['vwap'])
        )

        # Secondary exit: Only extreme VWAP overextension (keep this as it's profitable)
        extreme_vwap_overextension = (
            (dataframe['close'] > dataframe['vwap'] + (self.vwap_exit_multiplier.value * dataframe['vwap_std'])) |
            (dataframe['close'] < dataframe['vwap'] - (self.vwap_exit_multiplier.value * dataframe['vwap_std']))
        )

        # Emergency exit: Only on severe momentum collapse
        emergency_exit = (
            (dataframe['adx'] < self.emergency_adx_threshold.value) &
            (dataframe['adx_slope'] < self.emergency_adx_slope_threshold.value)
        )

        # Enhanced exit logic with optional Chikou confirmation
        long_exit_conditions = [strong_bearish_reversal, emergency_exit, extreme_vwap_overextension]
        short_exit_conditions = [strong_bullish_reversal, emergency_exit, extreme_vwap_overextension]

        if self.chikou_confirmation.value:
            long_exit_conditions.append(chikou_exit_long)
            short_exit_conditions.append(chikou_exit_short)

        # Combine exit conditions
        combined_long_exit = long_exit_conditions[0]
        for condition in long_exit_conditions[1:]:
            combined_long_exit = combined_long_exit | condition

        combined_short_exit = short_exit_conditions[0]
        for condition in short_exit_conditions[1:]:
            combined_short_exit = combined_short_exit | condition

        dataframe.loc[combined_long_exit, 'exit_long'] = 1
        dataframe.loc[combined_short_exit, 'exit_short'] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Enhanced adaptive custom stoploss with intelligent trailing logic

        Features:
        - Volatility-adaptive trail type selection (ATR vs Kijun)
        - Dynamic profit-based trail tightening
        - Volatility-adjusted trail distances
        - Chikou span early exit integration
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is None or len(dataframe) == 0:
            return self.stoploss

        current_candle = dataframe.iloc[-1]

        # === Base Dynamic Stop Calculation ===
        if trade.is_short:
            cloud_stop = current_candle['cloud_top']
            atr_stop = current_rate + (self.atr_multiplier.value * current_candle['atr'])
            base_stop = min(cloud_stop, atr_stop)
            base_stop_pct = (base_stop - trade.open_rate) / trade.open_rate
        else:
            cloud_stop = current_candle['cloud_bottom']
            atr_stop = current_rate - (self.atr_multiplier.value * current_candle['atr'])
            base_stop = max(cloud_stop, atr_stop)
            base_stop_pct = (trade.open_rate - base_stop) / trade.open_rate

        # === Early Exit on Chikou Congestion (if enabled) ===
        if self.chikou_confirmation.value and not current_candle.get('chikou_clear_space', True):
            # Tighten stop when Chikou enters congested area
            chikou_penalty = 0.3  # Reduce trail distance by 30%
            if trade.is_short:
                chikou_stop = current_rate + (chikou_penalty * self.atr_multiplier.value * current_candle['atr'])
                chikou_stop_pct = (chikou_stop - trade.open_rate) / trade.open_rate
                base_stop_pct = max(base_stop_pct, chikou_stop_pct)
            else:
                chikou_stop = current_rate - (chikou_penalty * self.atr_multiplier.value * current_candle['atr'])
                chikou_stop_pct = (trade.open_rate - chikou_stop) / trade.open_rate
                base_stop_pct = max(base_stop_pct, chikou_stop_pct)

        # === Trailing Stop Logic ===
        trailing_threshold = self.trailing_profit_multiplier.value * abs(self.stoploss)

        if current_profit > trailing_threshold:
            # === Adaptive Trail Type Selection ===
            # Use ATR trail in high volatility, Kijun trail in normal/low volatility
            use_atr_trail = current_candle['volatility_ratio'] > self.volatility_trail_threshold.value

            # === Dynamic Trail Tightening Based on Profit ===
            # Calculate profit multiplier (how many R's we're up)
            profit_multiple = current_profit / abs(self.stoploss)

            # Progressive trail tightening: more profit = tighter trail
            # Scale from 1.0 (no tightening) to max_trail_tightening (maximum tightening)
            tightening_factor = min(
                1.0 - (profit_multiple - self.trailing_profit_multiplier.value) *
                self.max_trail_tightening.value / 3.0,  # Normalize to 3R range
                1.0 - self.max_trail_tightening.value
            )
            tightening_factor = max(tightening_factor, 1.0 - self.max_trail_tightening.value)  # Floor

            # === Volatility-Adjusted Trail Distance ===
            # In high volatility: wider trails, in low volatility: tighter trails
            volatility_adjustment = np.clip(current_candle['volatility_ratio'], 0.5, 2.0)

            if use_atr_trail:
                # ATR-based trailing for volatile markets
                atr_trail_distance = (
                    self.atr_multiplier.value *
                    current_candle['atr'] *
                    tightening_factor *
                    volatility_adjustment
                )

                if trade.is_short:
                    trailing_stop = current_rate + atr_trail_distance
                    trailing_pct = (trailing_stop - trade.open_rate) / trade.open_rate
                else:
                    trailing_stop = current_rate - atr_trail_distance
                    trailing_pct = (trade.open_rate - trailing_stop) / trade.open_rate
            else:
                # Kijun-based trailing for trending markets
                if trade.is_short:
                    kijun_stop = current_candle['kijun']
                    # Apply tightening by moving closer to current price
                    adjustment = (current_rate - kijun_stop) * (1.0 - tightening_factor)
                    trailing_stop = kijun_stop + adjustment
                    trailing_pct = (trailing_stop - trade.open_rate) / trade.open_rate
                else:
                    kijun_stop = current_candle['kijun']
                    # Apply tightening by moving closer to current price
                    adjustment = (kijun_stop - current_rate) * (1.0 - tightening_factor)
                    trailing_stop = kijun_stop - adjustment
                    trailing_pct = (trade.open_rate - trailing_stop) / trade.open_rate

            # === Return Most Conservative Stop ===
            return max(base_stop_pct, trailing_pct, self.stoploss)

        return max(base_stop_pct, self.stoploss)

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Dynamic Stack Size Management with Smart Risk Controls

        This method integrates with the calculate_dynamic_stack_size method to provide
        intelligent position sizing based on:
        - Signal confidence and quality
        - Market regime alignment
        - Volatility conditions
        - Momentum strength
        - Portfolio heat management
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is None or len(dataframe) == 0:
            return self.min_stack_size.value

        current_candle = dataframe.iloc[-1]

        # === Use Pre-calculated Dynamic Stack Size ===
        if side == "long":
            dynamic_stack = current_candle.get('final_long_stack', 0.0)
        else:
            dynamic_stack = current_candle.get('final_short_stack', 0.0)

        # If no dynamic stack calculated (no signal), return minimum leverage
        # This prevents opening trades when there are no valid signals
        if dynamic_stack <= 0:
            # For futures compatibility, we must return >= 1.0, but the trade
            # should not be opened anyway since no entry signal is present
            return self.min_stack_size.value

        # === Additional Real-time Risk Overlays ===

        # 1. Portfolio Heat Check (simplified implementation)
        # In production, you would get actual portfolio status here
        portfolio_heat_multiplier = 1.0

        # Example: Reduce size if we're approaching max portfolio risk
        # This is a placeholder - in reality you'd check actual open positions
        estimated_portfolio_risk = 0.05  # Placeholder: 5% of portfolio at risk
        if estimated_portfolio_risk > self.stack_heat_threshold.value:
            portfolio_heat_multiplier = max(self.portfolio_heat_reduction_threshold.value, 1.0 - estimated_portfolio_risk)

        # 2. Recent Performance Adjustment
        # Reduce size after consecutive losses (simplified)
        # In production, you'd track actual win/loss streaks
        performance_multiplier = self.performance_multiplier_placeholder.value  # Placeholder for performance-based adjustments

        # 3. Time-of-day Risk Management (configurable)
        # Reduce size during typically volatile hours
        time_multiplier = 1.0
        if self.enable_time_risk_management.value and hasattr(current_time, 'hour'):
            # Check if current hour falls within configured high-risk periods
            hour = current_time.hour
            in_high_risk_period_1 = (self.high_risk_hours_start_1.value <= hour <= self.high_risk_hours_end_1.value)
            in_high_risk_period_2 = (self.high_risk_hours_start_2.value <= hour <= self.high_risk_hours_end_2.value)

            if in_high_risk_period_1 or in_high_risk_period_2:
                time_multiplier = self.time_risk_multiplier.value

        # 4. Volatility Circuit Breaker (configurable thresholds)
        # Emergency size reduction for extreme volatility
        volatility_circuit_breaker = 1.0
        if current_candle['volatility_ratio'] > self.extreme_volatility_threshold.value:
            volatility_circuit_breaker = self.extreme_volatility_multiplier.value
        elif current_candle['volatility_ratio'] > self.high_volatility_threshold.value:
            volatility_circuit_breaker = self.high_volatility_multiplier.value

        # === Apply All Risk Overlays ===
        final_stack_size = (
            dynamic_stack *
            portfolio_heat_multiplier *
            performance_multiplier *
            time_multiplier *
            volatility_circuit_breaker
        )

        # === Final Safety Checks ===
        # Ensure we stay within configured limits
        final_stack_size = max(final_stack_size, self.min_stack_size.value)
        final_stack_size = min(final_stack_size, self.max_stack_size.value)
        final_stack_size = min(final_stack_size, max_leverage)

        # Ensure we don't go below absolute minimum for valid signals
        if final_stack_size < self.min_stack_size.value and dynamic_stack > 0:
            final_stack_size = self.min_stack_size.value

        # === Exchange Compatibility: Round to Integer ===
        # Most exchanges require integer leverage values (1x, 2x, 3x, etc.)
        final_stack_size = round(final_stack_size)

        # Ensure minimum 1x leverage for exchange compatibility
        final_stack_size = max(1, final_stack_size)

        return final_stack_size

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                           rate: float, time_in_force: str, current_time: datetime,
                           entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """
        Final trade confirmation with additional filters and entry state storage
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is None or len(dataframe) == 0:
            return False

        current_candle = dataframe.iloc[-1]

        # Ensure all mandatory conditions are still met
        if side == "long":
            score_ok = current_candle['long_score'] >= self.min_entry_score.value
            ichimoku_ok = current_candle['ichimoku_bullish']
            entry_score = current_candle['long_score']
        else:
            score_ok = current_candle['short_score'] >= self.min_entry_score.value
            ichimoku_ok = current_candle['ichimoku_bearish']
            entry_score = current_candle['short_score']

        volatility_ok = current_candle['volatility_ok']

        # If all conditions are met, prepare entry state for ROI calculations
        if score_ok and ichimoku_ok and volatility_ok:
            # Store entry signal state that will be used for consistent ROI calculations
            # This captures the market conditions at entry time for the trade thesis
            entry_state = {
                'entry_score': float(entry_score),
                'entry_is_bull_market': bool(current_candle.get('is_bull_market', False)),
                'entry_is_bear_market': bool(current_candle.get('is_bear_market', False)),
                'entry_bull_strength': float(current_candle.get('bull_strength', 0.5)),
                'entry_bear_strength': float(current_candle.get('bear_strength', 0.5)),
                'entry_volatility_ratio': float(current_candle.get('volatility_ratio', 1.0)),
                'entry_adx': float(current_candle.get('adx', 20.0)),
                'entry_cloud_strength': float(current_candle.get('cloud_strength', 0.5)),
                'entry_confidence': float(entry_score / ((
                    self.ichimoku_weight.value +
                    self.adx_weight.value +
                    self.vwap_weight.value +
                    (self.chikou_weight.value if self.chikou_confirmation.value else 0)
                ) if (
                    self.ichimoku_weight.value +
                    self.adx_weight.value +
                    self.vwap_weight.value +
                    (self.chikou_weight.value if self.chikou_confirmation.value else 0)
                ) > 0 else 1)),
                'entry_time': current_time.isoformat(),
                'side': side
            }

            # Store in kwargs for FreqTrade to save in trade.custom_info
            # FreqTrade will automatically save custom_info when the trade is created
            if 'custom_info' not in kwargs:
                kwargs['custom_info'] = {}
            kwargs['custom_info']['entry_state'] = entry_state

        return score_ok and ichimoku_ok and volatility_ok

    def custom_roi(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Unified Dynamic ROI - Single adaptive system that automatically detects
        pair behavior and adjusts ROI calculations in real-time.

        Real-time Adaptive Features:
        - Volatility pattern detection (stable vs volatile behavior)
        - Trending strength analysis (trending vs ranging behavior)
        - Market regime responsiveness (bull/bear reaction strength)
        - Automatic sensitivity adjustment based on detected patterns
        """
        try:
            # Get current market analysis
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

            if dataframe is None or len(dataframe) == 0:
                return 0.20  # Safe fallback

            current_candle = dataframe.iloc[-1]

            # === Base ROI Calculation ===
            minutes_open = int((current_time - trade.open_date_utc).total_seconds() / 60)
            base_roi = self._get_base_roi_for_time(minutes_open) * self.roi_base_scaling.value

            # === Real-Time Behavior Detection ===
            # Analyze recent 30 candles to understand pair's behavior patterns
            lookback_period = min(30, len(dataframe))
            recent_data = dataframe.iloc[-lookback_period:]

            # 1. Volatility Pattern Detection
            recent_vol_ratios = recent_data['volatility_ratio'].fillna(1.0)
            avg_volatility = recent_vol_ratios.mean()
            vol_stability = 1.0 / (1.0 + recent_vol_ratios.std())  # Higher = more stable

            # 2. Trending Strength Detection
            recent_adx = recent_data['adx'].fillna(20.0)
            avg_momentum = recent_adx.mean()
            momentum_consistency = 1.0 / (1.0 + recent_adx.std() / max(recent_adx.mean(), 1.0))

            # 3. Market Regime Responsiveness
            recent_regime_changes = 0
            if 'bull_strength' in recent_data.columns and 'bear_strength' in recent_data.columns:
                bull_changes = recent_data['bull_strength'].diff().abs().sum()
                bear_changes = recent_data['bear_strength'].diff().abs().sum()
                recent_regime_changes = (bull_changes + bear_changes) / lookback_period

            regime_responsiveness = min(recent_regime_changes * 2.0, 1.0)  # Cap at 1.0

            # === Dynamic Sensitivity Calculation ===
            # Automatically adjust sensitivities based on detected behavior

            # Volatility Sensitivity: Higher for volatile pairs, lower for stable pairs
            dynamic_vol_sensitivity = self.roi_volatility_sensitivity.value * (
                0.7 + (avg_volatility - 1.0) * 0.6  # Range: 0.4 to 1.3 of base sensitivity
            )
            dynamic_vol_sensitivity = max(0.2, min(1.0, dynamic_vol_sensitivity))

            # Momentum Sensitivity: Higher for trending pairs, lower for ranging pairs
            momentum_factor = (avg_momentum - 20.0) / 30.0  # Normalize around typical ADX range
            dynamic_momentum_sensitivity = self.roi_momentum_sensitivity.value * (
                0.8 + momentum_factor * 0.4 + momentum_consistency * 0.3
            )
            dynamic_momentum_sensitivity = max(0.1, min(0.6, dynamic_momentum_sensitivity))

            # Regime Sensitivity: Higher for regime-responsive pairs
            dynamic_regime_sensitivity = self.roi_regime_sensitivity.value * (
                0.6 + regime_responsiveness * 0.8
            )
            dynamic_regime_sensitivity = max(0.1, min(0.8, dynamic_regime_sensitivity))

            # Confidence Sensitivity: Adjust based on signal consistency
            signal_consistency = vol_stability * momentum_consistency
            dynamic_confidence_sensitivity = self.roi_confidence_sensitivity.value * (
                0.7 + signal_consistency * 0.6
            )
            dynamic_confidence_sensitivity = max(0.1, min(0.4, dynamic_confidence_sensitivity))

            # === Get Entry State ===
            entry_state = {}
            if hasattr(trade, 'custom_info') and trade.custom_info:
                if isinstance(trade.custom_info, dict):
                    entry_state = trade.custom_info.get('entry_state', {})
                elif isinstance(trade.custom_info, str):
                    try:
                        import json
                        parsed_info = json.loads(trade.custom_info)
                        entry_state = parsed_info.get('entry_state', {})
                    except:
                        entry_state = {}

            # === Market Regime Formula with Dynamic Sensitivity ===
            regime_factor = 0.0

            if entry_state:
                entry_is_bull = entry_state.get('entry_is_bull_market', False)
                entry_is_bear = entry_state.get('entry_is_bear_market', False)
                entry_bull_strength = entry_state.get('entry_bull_strength', 0.5)
                entry_bear_strength = entry_state.get('entry_bear_strength', 0.5)

                if trade.is_short:
                    if entry_is_bear:
                        regime_factor = (entry_bear_strength - 0.5) * 2.0
                    elif entry_is_bull:
                        regime_factor = -(entry_bull_strength - 0.5) * 2.0
                else:
                    if entry_is_bull:
                        regime_factor = (entry_bull_strength - 0.5) * 2.0
                    elif entry_is_bear:
                        regime_factor = -(entry_bear_strength - 0.5) * 2.0
            else:
                # Use current regime with adaptive weighting
                if trade.is_short:
                    if current_candle.get('is_bear_market', False):
                        regime_strength = current_candle.get('bear_strength', 0.5)
                        regime_factor = (regime_strength - 0.5) * 2.0
                    elif current_candle.get('is_bull_market', False):
                        bull_strength = current_candle.get('bull_strength', 0.5)
                        regime_factor = -(bull_strength - 0.5) * 2.0
                else:
                    if current_candle.get('is_bull_market', False):
                        regime_strength = current_candle.get('bull_strength', 0.5)
                        regime_factor = (regime_strength - 0.5) * 2.0
                    elif current_candle.get('is_bear_market', False):
                        bear_strength = current_candle.get('bear_strength', 0.5)
                        regime_factor = -(bear_strength - 0.5) * 2.0

            regime_multiplier = 1.0 + (regime_factor * dynamic_regime_sensitivity)

            # === Volatility Formula ===
            # Use current volatility (market conditions can change rapidly)
            volatility_ratio = current_candle.get('volatility_ratio', 1.0)

            # Formula: High volatility -> lower ROI threshold (exit faster)
            #          Low volatility -> higher ROI threshold (hold longer)
            volatility_factor = 1.0 - volatility_ratio  # Inverse relationship
            volatility_multiplier = 1.0 + (volatility_factor * dynamic_vol_sensitivity)

            # === Momentum Formula ===
            # Blend entry momentum with current momentum for trend persistence
            if entry_state and 'entry_adx' in entry_state:
                entry_adx = entry_state.get('entry_adx', 20.0)
                current_adx = current_candle.get('adx', 20.0)
                # Weight: 70% entry momentum, 30% current momentum
                blended_adx = (entry_adx * 0.7) + (current_adx * 0.3)
            else:
                blended_adx = current_candle.get('adx', 20.0)

            # Normalize ADX to [-1, 1] range (assuming typical ADX range of 10-50)
            normalized_adx = (blended_adx - 30.0) / 20.0  # Center around 30, scale by 20
            normalized_adx = max(-1.0, min(1.0, normalized_adx))  # Clamp to [-1, 1]

            # Strong momentum -> higher ROI threshold (hold longer in trends)
            momentum_multiplier = 1.0 + (normalized_adx * dynamic_momentum_sensitivity)

            # === Signal Confidence Formula ===
            # Use entry confidence if available for consistency with original thesis
            if entry_state and 'entry_confidence' in entry_state:
                signal_confidence = entry_state.get('entry_confidence', 0.5)
            else:
                # Fallback to current signal strength
                if trade.is_short:
                    current_score = current_candle.get('short_score', 0)
                else:
                    current_score = current_candle.get('long_score', 0)

                max_possible_score = (
                    self.ichimoku_weight.value +
                    self.adx_weight.value +
                    self.vwap_weight.value +
                    (self.chikou_weight.value if self.chikou_confirmation.value else 0)
                )

                signal_confidence = current_score / max_possible_score if max_possible_score > 0 else 0.5

            # Transform confidence to [-1, 1] range (high confidence -> hold longer)
            confidence_factor = (signal_confidence - 0.5) * 2.0
            confidence_multiplier = 1.0 + (confidence_factor * dynamic_confidence_sensitivity)

            # === Trend Strength Formula ===
            # Use current cloud strength (trend strength can evolve)
            cloud_strength = current_candle.get('cloud_strength', 0.5)

            # Transform cloud strength to [-1, 1] range
            trend_factor = (cloud_strength - 0.5) * 2.0
            trend_multiplier = 1.0 + (trend_factor * self.roi_confidence_sensitivity.value * 0.5)  # Reduced impact

            # === Profit Protection Formula ===
            # Progressive reduction of ROI threshold as profit increases
            profit_protection_multiplier = 1.0

            if current_profit > 0.40:  # 40%+ profit
                profit_protection_multiplier = 0.7  # Much more aggressive exits
            elif current_profit > 0.25:  # 25%+ profit
                profit_protection_multiplier = 0.8  # More aggressive exits
            elif current_profit > 0.15:  # 15%+ profit
                profit_protection_multiplier = 0.9  # Slightly more aggressive exits

            # === Calculate Final Dynamic ROI ===
            dynamic_roi = (
                base_roi *
                regime_multiplier *
                volatility_multiplier *
                momentum_multiplier *
                confidence_multiplier *
                trend_multiplier *
                profit_protection_multiplier
            )

            # === Apply Safety Bounds ===
            dynamic_roi = max(self.min_dynamic_roi.value, min(dynamic_roi, self.max_dynamic_roi.value))

            # Enhanced logging to show entry vs current state usage
            entry_source = "entry_state" if entry_state else "current_state"

            return dynamic_roi

        except Exception as e:
            # Safe fallback in case of any errors
            self.log(f"Error in custom_roi for {pair}: {e}")
            return 0.20

    def _get_base_roi_for_time(self, minutes_open: int) -> float:
        """
        Get base ROI from the static minimal_roi table based on time.

        Args:
            minutes_open: Minutes since trade opened

        Returns:
            float: Base ROI threshold for the given time
        """
        # Convert minutes to seconds for comparison with minimal_roi keys
        seconds_open = minutes_open * 60

        # Find the appropriate ROI value from the table
        applicable_roi = None

        # Sort ROI table by time (seconds) in descending order
        sorted_roi = sorted(self.minimal_roi.items(), key=lambda x: int(x[0]), reverse=True)

        # Find the first (highest time threshold) that we've exceeded
        for time_str, roi_value in sorted_roi:
            time_seconds = int(time_str)
            if seconds_open >= time_seconds:
                applicable_roi = roi_value
                break

        # If no time thresholds are met, use the initial (0-time) ROI
        if applicable_roi is None:
            applicable_roi = self.minimal_roi["0"]

        return applicable_roi

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs) -> Optional[str]:
        """
        Simple multi-tier take-profit using existing market analysis and parameterized thresholds
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None

        current_candle = dataframe.iloc[-1]
        profit_pct = current_profit * 100

        # Use existing market analysis
        is_bull_market = current_candle.get('is_bull_market', False)
        volatility_ratio = current_candle.get('volatility_ratio', 1.0)
        adx_strength = current_candle.get('adx', 20.0)
        adx_slope = current_candle.get('adx_slope', 0.0)

        # Calculate dynamic targets using parameters
        tp1_base = self.take_profit_1.value
        tp2_base = self.take_profit_2.value

        # Adjust targets based on volatility
        if volatility_ratio > 1.5:  # High volatility - take profits faster
            tp1_target = tp1_base * self.high_volatility_tp_reduction.value
            tp2_target = tp2_base * self.high_volatility_tp_reduction.value
        elif adx_strength > 35 and is_bull_market == (not trade.is_short):  # Strong aligned momentum
            tp1_target = tp1_base * self.strong_momentum_tp_boost.value
            tp2_target = tp2_base * self.strong_momentum_tp_boost.value
        else:  # Normal conditions
            tp1_target = tp1_base
            tp2_target = tp2_base

        # Take profits at calculated levels
        if profit_pct >= tp2_target:
            return "take_profit_2"
        elif profit_pct >= tp1_target:
            # Only exit at TP1 if momentum is weakening
            if adx_strength < self.weak_momentum_threshold.value or adx_slope < self.momentum_decline_threshold.value:
                return "take_profit_1"

        return None
