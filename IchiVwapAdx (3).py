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
    - Optional FFT cycle detection with Hurst harmonic analysis (experimental)
    - Enhanced custom entry pricing with dynamic fishing factor based on candle analysis
    - Advanced trade exit protection with configurable minimum profit thresholds

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
    transition_lookback = IntParameter(2, 15, default=9, space="buy", optimize=False)

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

    # === FFT Cycle Detection Parameters ===
    enable_fft_cycle_detection = BooleanParameter(default=False, space="buy", optimize=False)
    u_window_size = IntParameter(100, 250, default=150, space='buy', optimize=False)
    l_window_size = IntParameter(20, 50, default=50, space='buy', optimize=False)

    # === Enhanced Custom Entry Price Parameters ===
    enable_custom_entry_price = BooleanParameter(default=True, space="buy", optimize=False)
    entry_price_fishing_factor = DecimalParameter(0.96, 1.0, default=0.98, space="buy", optimize=False)
    entry_price_increment = DecimalParameter(0.998, 1.005, default=1.001, space="buy", optimize=False)

    # === Trade Exit Protection Parameters ===
    enable_exit_protection = BooleanParameter(default=True, space="sell", optimize=False)
    min_roi_exit_threshold = DecimalParameter(0.003, 0.010, default=0.005, space="sell", optimize=False)
    min_partial_exit_threshold = DecimalParameter(0.005, 0.015, default=0.010, space="sell", optimize=False)
    min_trailing_stop_threshold = DecimalParameter(0.005, 0.015, default=0.010, space="sell", optimize=False)

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
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value).ffill().fillna(0.01)
        dataframe['atr_50'] = ta.ATR(dataframe, timeperiod=50).ffill().fillna(0.01)
        # Protect against division by zero in volatility ratio with NaN handling
        dataframe['volatility_ratio'] = (dataframe['atr'] / dataframe['atr_50'].replace(0, np.nan)).fillna(1.0)

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

        # === ADX and Directional Movement with NaN handling ===
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_period.value).ffill().fillna(20.0)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=self.adx_period.value).ffill().fillna(20.0)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=self.adx_period.value).ffill().fillna(20.0)
        dataframe['adx_slope'] = dataframe['adx'].diff(3).fillna(0.0)

        # === VWAP with Standard Deviation Bands with NaN handling ===
        vwap_data = self.calculate_vwap_bands(dataframe)
        dataframe['vwap'] = vwap_data['vwap'].ffill().fillna(dataframe['close'])
        dataframe['vwap_std'] = vwap_data['vwap_std'].ffill().fillna(0.01)
        dataframe['vwap_upper'] = dataframe['vwap'] + (self.vwap_std_multiplier.value * dataframe['vwap_std'])
        dataframe['vwap_lower'] = dataframe['vwap'] - (self.vwap_std_multiplier.value * dataframe['vwap_std'])

        # === Cloud Quality Metrics with NaN handling ===
        dataframe['cloud_thickness_median'] = dataframe['cloud_thickness'].rolling(
            window=self.cloud_quality_lookback.value
        ).quantile(self.cloud_quality_percentile.value / 100.0).ffill().fillna(0.01)

        # Cloud strength calculation for ROI with NaN handling
        dataframe['cloud_strength'] = np.where(
            dataframe['cloud_thickness_median'] > 0,
            np.clip(dataframe['cloud_thickness'] / dataframe['cloud_thickness_median'].replace(0, np.nan), 0.0, 2.0) / 2.0,
            0.5  # Default neutral strength
        )
        dataframe['cloud_strength'] = pd.Series(dataframe['cloud_strength']).fillna(0.5)

        # === Market Regime Detection ===
        dataframe = self.calculate_market_regime(dataframe)

        # === Signal Components ===
        dataframe = self.calculate_signal_components(dataframe)

        # === Dynamic Stack Size Calculations ===
        dataframe = self.calculate_dynamic_stack_size(dataframe)

        # === Enhanced Custom Entry Price Indicators (Optional) ===
        if self.enable_custom_entry_price.value:
            # Calculate candle size relative to ATR for dynamic entry pricing
            dataframe['candle_size'] = np.abs(dataframe['close'] - dataframe['open']) / dataframe['atr']
            dataframe['candle_size_target'] = np.clip(dataframe['candle_size'], 0.0, 0.3)  # Cap at 30% ATR

        # === FFT Cycle Detection (Optional) ===
        if self.enable_fft_cycle_detection.value:
            dataframe = self.calculate_fft_cycles(dataframe)

        return dataframe

    def perform_fft(self, price_data, window_size=None):
        """
        Perform Fast Fourier Transform on price data to identify dominant cycles

        Args:
            price_data: Price series (typically close or HA close)
            window_size: Optional rolling window size for smoothing

        Returns:
            tuple: (frequencies, power) arrays
        """
        if window_size is not None:
            # Apply rolling window to smooth the data
            price_data = price_data.rolling(window=window_size, center=True).mean().dropna()

        # Normalize data
        normalized_data = (price_data - np.mean(price_data)) / np.std(price_data)
        n = len(normalized_data)

        # Perform FFT
        fft_data = np.fft.fft(normalized_data)
        freq = np.fft.fftfreq(n)
        power = np.abs(fft_data) ** 2

        # Handle infinite values
        power[np.isinf(power)] = 0

        return freq, power

    def calculate_fft_cycles(self, dataframe: DataFrame) -> DataFrame:
        """
        Calculate FFT-based cycle detection and harmonic indicators
        Optimized version with caching and reduced computational complexity
        """
        # Cache key based on data length and settings
        cache_key = f"fft_{len(dataframe)}_{self.u_window_size.value}_{self.l_window_size.value}"

        # Check cache first
        if hasattr(self, '_fft_cache') and cache_key in self._fft_cache:
            cached_result = self._fft_cache[cache_key]
            # Apply cached cycle parameters to new data
            cycle_period, h0, h1, h2 = cached_result
        else:
            # Initialize cache if needed
            if not hasattr(self, '_fft_cache'):
                self._fft_cache = {}

            # Use Heiken Ashi close if available, otherwise regular close
            if self.use_heiken_ashi.value and 'ha_close' in dataframe.columns:
                price_series = dataframe['ha_close']
            else:
                price_series = dataframe['close']

            # Check if we have enough data
            if len(dataframe) < self.u_window_size.value:
                cycle_period, h0, h1, h2 = 80, 40, 27, 20  # Default values
            else:
                try:
                    # Simplified FFT: Only use last u_window_size.value points for efficiency
                    analysis_data = price_series.tail(self.u_window_size.value)

                    # Perform FFT to identify cycles
                    freq, power = self.perform_fft(analysis_data, window_size=None)  # Skip double smoothing

                    if len(freq) == 0 or len(power) == 0:
                        cycle_period, h0, h1, h2 = 80, 40, 27, 20  # Default values
                    else:
                        # Filter frequencies within specified range and avoid division by very small numbers
                        freq_safe = freq[np.abs(freq) > 1e-8]  # Avoid near-zero frequencies
                        power_safe = power[np.abs(freq) > 1e-8]

                        if len(freq_safe) == 0:
                            cycle_period, h0, h1, h2 = 80, 40, 27, 20  # Default values
                        else:
                            # Vectorized filtering - much faster than individual operations
                            periods = 1 / np.abs(freq_safe)
                            valid_mask = (periods >= self.l_window_size.value) & (periods <= self.u_window_size.value)

                            if not valid_mask.any():
                                cycle_period, h0, h1, h2 = 80, 40, 27, 20  # Default values
                            else:
                                valid_periods = periods[valid_mask]
                                valid_power = power_safe[valid_mask]

                                # Find dominant cycle more efficiently
                                if len(valid_power) > 0:
                                    dominant_idx = np.argmax(valid_power)
                                    cycle_period = int(np.clip(valid_periods[dominant_idx], 20, 200))
                                else:
                                    cycle_period = 80

                                # Calculate harmonics efficiently
                                h0 = max(10, cycle_period // 2)
                                h1 = max(5, cycle_period // 3)
                                h2 = max(3, cycle_period // 4)

                except Exception:
                    cycle_period, h0, h1, h2 = 80, 40, 27, 20  # Default values

            # Cache the result for future use
            self._fft_cache[cache_key] = (cycle_period, h0, h1, h2)

            # Limit cache size to prevent memory issues
            if len(self._fft_cache) > 10:
                oldest_key = next(iter(self._fft_cache))
                del self._fft_cache[oldest_key]

        # Use cached or computed cycle parameters
        price_series = dataframe['ha_close'] if (self.use_heiken_ashi.value and 'ha_close' in dataframe.columns) else dataframe['close']

        try:
            # Vectorized EWM calculations - much faster than individual operations
            dataframe['dc_EWM'] = price_series.ewm(span=cycle_period, adjust=False).mean()
            dataframe['dc_1/2'] = price_series.ewm(span=h0, adjust=False).mean()
            dataframe['dc_1/3'] = price_series.ewm(span=h1, adjust=False).mean()
            dataframe['dc_1/4'] = price_series.ewm(span=h2, adjust=False).mean()

            # Optimized trend calculation with better zero protection
            dc_ewm_safe = dataframe['dc_EWM'].replace(0, np.nan)
            dataframe['dc-trend'] = dataframe['dc_EWM'].diff() / dc_ewm_safe

            # Optimized rolling peak-to-peak calculations using built-in min/max
            price_series_safe = price_series.replace(0, np.nan)

            # Use vectorized operations instead of apply() for much better performance
            for period, col_prefix in [(cycle_period, 'cycle'), (h0, 'h0'), (h1, 'h1'), (h2, 'h2')]:
                rolling_max = price_series.rolling(period, min_periods=1).max()
                rolling_min = price_series.rolling(period, min_periods=1).min()
                ptp_value = rolling_max - rolling_min

                # Store move and mean calculations
                move_col = f'{col_prefix}_move'
                mean_col = f'{col_prefix}_move_mean'

                dataframe[move_col] = ptp_value / price_series_safe
                dataframe[mean_col] = dataframe[move_col].rolling(period, min_periods=1).mean()

            # Fill NaN values with sensible defaults
            dataframe['cycle_move_mean'] = dataframe['cycle_move_mean'].fillna(0.02)
            dataframe['h0_move_mean'] = dataframe['h0_move_mean'].fillna(0.015)
            dataframe['h1_move_mean'] = dataframe['h1_move_mean'].fillna(0.01)
            dataframe['h2_move_mean'] = dataframe['h2_move_mean'].fillna(0.005)

        except Exception:
            # Error in calculation - use safe default values
            dataframe['cycle_move_mean'] = 0.02
            dataframe['h0_move_mean'] = 0.015
            dataframe['h1_move_mean'] = 0.01
            dataframe['h2_move_mean'] = 0.005

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
        try:
            # Input validation
            if dataframe is None or len(dataframe) == 0:
                return {'vwap': pd.Series(dtype='float64'), 'vwap_std': pd.Series(dtype='float64')}

            required_columns = ['high', 'low', 'close', 'volume']
            if not all(col in dataframe.columns for col in required_columns):
                raise ValueError(f"Missing required columns for VWAP calculation: {required_columns}")

            high = dataframe['high']
            low = dataframe['low']
            close = dataframe['close']
            volume = dataframe['volume']

            # Validate volume is positive
            if (volume <= 0).any():
                volume = volume.replace(0, np.nan).ffill().fillna(1.0)

            # Typical Price
            typical_price = (high + low + close) / 3

            # VWAP calculation with zero-division protection
            vwap_num = (typical_price * volume).rolling(window=self.vwap_period.value).sum()
            vwap_den = volume.rolling(window=self.vwap_period.value).sum()

            # Protect against division by zero
            vwap_den_safe = vwap_den.replace(0, np.nan)
            vwap = vwap_num / vwap_den_safe

            # Standard deviation calculation with protection
            price_diff_sq = ((typical_price - vwap) ** 2 * volume).rolling(window=self.vwap_period.value).sum()
            vwap_var = price_diff_sq / vwap_den_safe
            vwap_std = np.sqrt(np.maximum(vwap_var, 0))  # Ensure non-negative for sqrt

            return {
                'vwap': vwap,
                'vwap_std': vwap_std
            }
        except Exception as e:
            # Fallback to safe default values
            return {
                'vwap': dataframe['close'] if 'close' in dataframe.columns else pd.Series(dtype='float64'),
                'vwap_std': pd.Series([0.01] * len(dataframe), index=dataframe.index) if len(dataframe) > 0 else pd.Series(dtype='float64')
            }

    def calculate_heiken_ashi(self, dataframe: DataFrame) -> DataFrame:
        """
        Calculate Heiken Ashi candles for smoother trend analysis

        Heiken Ashi candles smooth out price action and reduce noise,
        making trend identification and reversal detection more reliable.
        """
        try:
            # Input validation
            if dataframe is None or len(dataframe) == 0:
                return dataframe

            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in dataframe.columns for col in required_columns):
                raise ValueError(f"Missing required columns for Heiken Ashi calculation: {required_columns}")

            # Initialize with first candle
            ha_close = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
            ha_open = pd.Series(index=dataframe.index, dtype='float64')

            if len(dataframe) > 0:
                ha_open.iloc[0] = (dataframe['open'].iloc[0] + dataframe['close'].iloc[0]) / 2

                # Optimized vectorized HA open calculation using expanding mean approach
                # This replaces the slow iterative loop with fast pandas operations
                if len(dataframe) > 1:
                    # Create shifted series for vectorized calculation
                    ha_close_shifted = ha_close.shift(1)
                    ha_open_shifted = ha_open.shift(1)

                    # Use fillna with expanding calculation for first few values
                    ha_open_calc = (ha_open_shifted + ha_close_shifted) / 2
                    ha_open_calc.iloc[0] = ha_open.iloc[0]  # Keep first value

                    # Forward fill using expanding calculation for better performance
                    for i in range(1, min(len(dataframe), 10)):  # Only iterate for first 10 values
                        if pd.isna(ha_open_calc.iloc[i]):
                            ha_open_calc.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2

                    # Use expanding approach for remaining values (much faster)
                    ha_open = ha_open_calc.ffill() if len(dataframe) > 10 else ha_open_calc

                    # Ensure no NaN values remain
                    ha_open = ha_open.fillna(dataframe['open'])

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

        except Exception as e:
            # Fallback: return original dataframe with basic HA columns
            dataframe['ha_open'] = dataframe['open']
            dataframe['ha_high'] = dataframe['high']
            dataframe['ha_low'] = dataframe['low']
            dataframe['ha_close'] = dataframe['close']
            dataframe['ha_green'] = dataframe['close'] > dataframe['open']
            dataframe['ha_red'] = dataframe['close'] < dataframe['open']
            dataframe['ha_doji'] = False
            dataframe['ha_body_size'] = np.abs(dataframe['close'] - dataframe['open'])
            dataframe['ha_upper_wick'] = dataframe['high'] - np.maximum(dataframe['open'], dataframe['close'])
            dataframe['ha_lower_wick'] = np.minimum(dataframe['open'], dataframe['close']) - dataframe['low']
            dataframe['ha_trend_consecutive'] = 1
            return dataframe

    def calculate_ichimoku_ha(self, dataframe: DataFrame) -> dict:
        """
        Calculate Ichimoku Cloud components using Heiken Ashi candles
        for smoother trend analysis and reduced false signals
        """
        try:
            # Input validation
            if dataframe is None or len(dataframe) == 0:
                return {
                    'tenkan': pd.Series(dtype='float64'),
                    'kijun': pd.Series(dtype='float64'),
                    'senkou_a': pd.Series(dtype='float64'),
                    'senkou_b': pd.Series(dtype='float64'),
                    'chikou': pd.Series(dtype='float64')
                }

            required_ha_columns = ['ha_high', 'ha_low', 'ha_close']
            if not all(col in dataframe.columns for col in required_ha_columns):
                raise ValueError(f"Missing required Heiken Ashi columns: {required_ha_columns}")

            # Validate period parameters
            if self.ichimoku_tenkan_period.value <= 0 or self.ichimoku_kijun_period.value <= 0 or self.ichimoku_senkou_period.value <= 0:
                raise ValueError("Ichimoku periods must be positive integers")

            ha_high = dataframe['ha_high'].ffill().fillna(dataframe.get('high', 0))
            ha_low = dataframe['ha_low'].ffill().fillna(dataframe.get('low', 0))
            ha_close = dataframe['ha_close'].ffill().fillna(dataframe.get('close', 0))

            # Tenkan-sen (Conversion Line) - HA version with error handling
            ha_tenkan_high = ha_high.rolling(window=self.ichimoku_tenkan_period.value, min_periods=1).max()
            ha_tenkan_low = ha_low.rolling(window=self.ichimoku_tenkan_period.value, min_periods=1).min()
            ha_tenkan = ((ha_tenkan_high + ha_tenkan_low) / 2).fillna(ha_close)

            # Kijun-sen (Base Line) - HA version with error handling
            ha_kijun_high = ha_high.rolling(window=self.ichimoku_kijun_period.value, min_periods=1).max()
            ha_kijun_low = ha_low.rolling(window=self.ichimoku_kijun_period.value, min_periods=1).min()
            ha_kijun = ((ha_kijun_high + ha_kijun_low) / 2).fillna(ha_close)

            # Senkou Span A (Leading Span A) - HA version with error handling
            ha_senkou_a = ((ha_tenkan + ha_kijun) / 2).shift(self.ichimoku_kijun_period.value).fillna(ha_close)

            # Senkou Span B (Leading Span B) - HA version with error handling
            ha_senkou_high = ha_high.rolling(window=self.ichimoku_senkou_period.value, min_periods=1).max()
            ha_senkou_low = ha_low.rolling(window=self.ichimoku_senkou_period.value, min_periods=1).min()
            ha_senkou_b = ((ha_senkou_high + ha_senkou_low) / 2).shift(self.ichimoku_kijun_period.value).fillna(ha_close)

            # Chikou Span (Lagging Span) - HA version with error handling
            ha_chikou = ha_close.shift(-self.ichimoku_kijun_period.value).fillna(ha_close)

            return {
                'tenkan': ha_tenkan,
                'kijun': ha_kijun,
                'senkou_a': ha_senkou_a,
                'senkou_b': ha_senkou_b,
                'chikou': ha_chikou
            }

        except Exception as e:
            # Fallback: return safe default Ichimoku values using regular close if HA data fails
            fallback_close = dataframe.get('close', pd.Series([1.0] * len(dataframe), index=dataframe.index))
            return {
                'tenkan': fallback_close,
                'kijun': fallback_close,
                'senkou_a': fallback_close,
                'senkou_b': fallback_close,
                'chikou': fallback_close
            }

    def calculate_market_regime(self, dataframe: DataFrame) -> DataFrame:
        """
        Detect market regime (bull/bear/sideways) using multiple indicators

        Bull Market: Price consistently above fast SMA, fast SMA > slow SMA
        Bear Market: Price consistently below fast SMA, fast SMA < slow SMA
        Sideways: Mixed signals or choppy conditions
        """
        try:
            # Input validation
            if dataframe is None or len(dataframe) == 0:
                return dataframe

            required_columns = ['close']
            if not all(col in dataframe.columns for col in required_columns):
                raise ValueError(f"Missing required columns for market regime calculation: {required_columns}")

            # Validate parameters
            if self.regime_sma_fast.value <= 0 or self.regime_sma_slow.value <= 0 or self.regime_lookback.value <= 0:
                raise ValueError("Regime detection parameters must be positive integers")

            if self.regime_sma_fast.value >= self.regime_sma_slow.value:
                raise ValueError("Fast SMA period must be less than slow SMA period")

            # Calculate regime SMAs with NaN handling
            dataframe['regime_sma_fast'] = ta.SMA(dataframe, timeperiod=self.regime_sma_fast.value).bfill().fillna(dataframe['close'])
            dataframe['regime_sma_slow'] = ta.SMA(dataframe, timeperiod=self.regime_sma_slow.value).bfill().fillna(dataframe['close'])

            # Basic trend conditions with NaN handling
            dataframe['price_above_fast_sma'] = (dataframe['close'] > dataframe['regime_sma_fast'])
            dataframe['fast_above_slow_sma'] = (dataframe['regime_sma_fast'] > dataframe['regime_sma_slow'])

            # Calculate percentage of recent periods where conditions are met
            lookback = min(self.regime_lookback.value, len(dataframe))  # Ensure lookback doesn't exceed data length

            dataframe['bull_strength'] = (
                dataframe['price_above_fast_sma'].rolling(window=lookback, min_periods=1).mean().fillna(0.0) * self.regime_price_weight.value +
                dataframe['fast_above_slow_sma'].rolling(window=lookback, min_periods=1).mean().fillna(0.0) * self.regime_sma_weight.value
            )

            dataframe['bear_strength'] = (
                (~dataframe['price_above_fast_sma']).rolling(window=lookback, min_periods=1).mean().fillna(0.0) * self.regime_price_weight.value +
                (~dataframe['fast_above_slow_sma']).rolling(window=lookback, min_periods=1).mean().fillna(0.0) * self.regime_sma_weight.value
            )

            # Determine regime based on threshold with NaN handling
            threshold = self.regime_threshold.value

            dataframe['is_bull_market'] = (dataframe['bull_strength'] > threshold)
            dataframe['is_bear_market'] = (dataframe['bear_strength'] > threshold)
            dataframe['is_sideways_market'] = (
                (dataframe['bull_strength'] <= threshold) &
                (dataframe['bear_strength'] <= threshold)
            )

            # Market regime flag for shorting with NaN handling
            dataframe['shorts_allowed'] = (~dataframe['is_bull_market'])  # Allow shorts in bear/sideways markets

            # Fill any remaining NaN values with safe defaults
            dataframe['is_bull_market'] = dataframe['is_bull_market'].fillna(False)
            dataframe['is_bear_market'] = dataframe['is_bear_market'].fillna(False)
            dataframe['is_sideways_market'] = dataframe['is_sideways_market'].fillna(True)  # Default to sideways when uncertain
            dataframe['shorts_allowed'] = dataframe['shorts_allowed'].fillna(True)  # Allow shorts by default

            return dataframe

        except Exception as e:
            # Fallback: return dataframe with safe default regime values
            dataframe['regime_sma_fast'] = dataframe.get('close', pd.Series([1.0] * len(dataframe), index=dataframe.index))
            dataframe['regime_sma_slow'] = dataframe.get('close', pd.Series([1.0] * len(dataframe), index=dataframe.index))
            dataframe['price_above_fast_sma'] = False
            dataframe['fast_above_slow_sma'] = False
            dataframe['bull_strength'] = 0.0
            dataframe['bear_strength'] = 0.0
            dataframe['is_bull_market'] = False
            dataframe['is_bear_market'] = False
            dataframe['is_sideways_market'] = True  # Default to sideways market
            dataframe['shorts_allowed'] = True  # Allow shorts by default
            return dataframe

    def calculate_signal_components(self, dataframe: DataFrame) -> DataFrame:
        """Calculate individual signal components and confluence score"""
        try:
            # Input validation
            if dataframe is None or len(dataframe) == 0:
                return dataframe

            required_columns = ['close', 'chikou', 'high', 'low', 'adx', 'adx_slope', 'plus_di', 'minus_di', 'vwap_upper', 'vwap_lower', 'volatility_ratio']
            if not all(col in dataframe.columns for col in required_columns):
                # Create missing columns with safe defaults
                for col in required_columns:
                    if col not in dataframe.columns:
                        if col in ['close', 'high', 'low']:
                            dataframe[col] = 1.0
                        elif col == 'chikou':
                            dataframe[col] = dataframe.get('close', 1.0)
                        elif col in ['adx', 'plus_di', 'minus_di']:
                            dataframe[col] = 20.0
                        elif col == 'adx_slope':
                            dataframe[col] = 0.0
                        elif col in ['vwap_upper', 'vwap_lower']:
                            dataframe[col] = dataframe.get('close', 1.0)
                        elif col == 'volatility_ratio':
                            dataframe[col] = 1.0

            # === Chikou Span Signals ===
            # Fill NaN values in chikou for robust calculation
            dataframe['chikou_filled'] = dataframe['chikou'].bfill().ffill()
            if dataframe['chikou_filled'].isna().all():
                dataframe['chikou_filled'] = dataframe['close']

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

                # Check if chikou is in clear space (above recent highs or below recent lows) with NaN handling
                recent_high = high_kijun_ago.rolling(window=self.chikou_clear_space_periods.value, min_periods=1).max().bfill().fillna(dataframe['high'])
                recent_low = low_kijun_ago.rolling(window=self.chikou_clear_space_periods.value, min_periods=1).min().bfill().fillna(dataframe['low'])

                chikou_above_high = (dataframe['chikou_filled'] > recent_high)
                chikou_below_low = (dataframe['chikou_filled'] < recent_low)
                chikou_clear_condition = chikou_above_high | chikou_below_low
                dataframe['chikou_clear_space'] = chikou_clear_condition.fillna(False)
            else:
                dataframe['chikou_clear_space'] = True

            # === Enhanced Ichimoku Signals with Heiken Ashi and Chikou ===
            # Choose which data to use for signal generation
            if self.use_heiken_ashi.value and 'ha_close' in dataframe.columns:
                # Use Heiken Ashi for smoother signal generation
                signal_close = dataframe['ha_close'].fillna(dataframe['close'])
                signal_cloud_top = dataframe.get('ha_cloud_top', dataframe.get('cloud_top', dataframe['close']))
                signal_cloud_bottom = dataframe.get('ha_cloud_bottom', dataframe.get('cloud_bottom', dataframe['close']))
                signal_cloud_thickness = dataframe.get('ha_cloud_thickness', dataframe.get('cloud_thickness', 0.01))
                signal_cloud_median = signal_cloud_thickness.rolling(
                    window=self.cloud_quality_lookback.value, min_periods=1
                ).quantile(self.cloud_quality_percentile.value / 100.0).fillna(0.01)

                # Additional HA trend confirmation
                ha_green = dataframe.get('ha_green', dataframe['close'] > dataframe['close'].shift(1))
                ha_red = dataframe.get('ha_red', dataframe['close'] < dataframe['close'].shift(1))
                ha_body_size = dataframe.get('ha_body_size', abs(dataframe['close'] - dataframe['close'].shift(1)))
                atr = dataframe.get('atr', 0.01)

                ha_bullish_trend = ha_green & (ha_body_size > atr * self.ha_body_atr_ratio.value)
                ha_bearish_trend = ha_red & (ha_body_size > atr * self.ha_body_atr_ratio.value)
            else:
                # Use regular candles
                signal_close = dataframe['close']
                signal_cloud_top = dataframe.get('cloud_top', dataframe['close'])
                signal_cloud_bottom = dataframe.get('cloud_bottom', dataframe['close'])
                signal_cloud_thickness = dataframe.get('cloud_thickness', 0.01)
                signal_cloud_median = dataframe.get('cloud_thickness_median', 0.01)
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

            # === Enhanced Confluence Scoring with Chikou (Optimized) ===
            # Use vectorized operations instead of multiple .astype(int) conversions
            # Pre-convert boolean columns to numeric for efficiency
            ichimoku_bull_numeric = dataframe['ichimoku_bullish'].astype(np.int8)
            ichimoku_bear_numeric = dataframe['ichimoku_bearish'].astype(np.int8)
            adx_bull_numeric = dataframe['adx_bullish'].astype(np.int8)
            adx_bear_numeric = dataframe['adx_bearish'].astype(np.int8)
            vwap_bull_numeric = dataframe['vwap_bullish'].astype(np.int8)
            vwap_bear_numeric = dataframe['vwap_bearish'].astype(np.int8)

            # Vectorized scoring calculation
            dataframe['long_score'] = (
                (ichimoku_bull_numeric * self.ichimoku_weight.value) +
                (adx_bull_numeric * self.adx_weight.value) +
                (vwap_bull_numeric * self.vwap_weight.value)
            )

            dataframe['short_score'] = (
                (ichimoku_bear_numeric * self.ichimoku_weight.value) +
                (adx_bear_numeric * self.adx_weight.value) +
                (vwap_bear_numeric * self.vwap_weight.value)
            )

            # Add Chikou points if enabled (more efficient conditional)
            if self.chikou_confirmation.value:
                chikou_bull_numeric = dataframe['chikou_bullish'].astype(np.int8)
                chikou_bear_numeric = dataframe['chikou_bearish'].astype(np.int8)
                dataframe['long_score'] += (chikou_bull_numeric * self.chikou_weight.value)
                dataframe['short_score'] += (chikou_bear_numeric * self.chikou_weight.value)

            # Fill NaN values efficiently
            dataframe['long_score'] = dataframe['long_score'].fillna(0.0)
            dataframe['short_score'] = dataframe['short_score'].fillna(0.0)

            # === Boolean Transition Logic with NaN handling ===
            # Detect score transitions to prevent buying/selling tops/bottoms

            # Check if score just reached threshold (wasn't there in previous bars)
            lookback = min(self.transition_lookback.value, len(dataframe))

            # For long signals: score >= threshold AND was below threshold recently with NaN handling
            long_score_current = (dataframe['long_score'] >= self.min_entry_score.value)
            long_score_was_low = (dataframe['long_score'].shift(1).rolling(window=lookback, min_periods=1).max().fillna(0.0) < self.min_entry_score.value)
            dataframe['long_score_transition'] = (long_score_current & long_score_was_low).fillna(False)

            # For short signals: score >= threshold AND was below threshold recently with NaN handling
            short_score_current = (dataframe['short_score'] >= self.min_entry_score.value)
            short_score_was_low = (dataframe['short_score'].shift(1).rolling(window=lookback, min_periods=1).max().fillna(0.0) < self.min_entry_score.value)
            dataframe['short_score_transition'] = (short_score_current & short_score_was_low).fillna(False)

            return dataframe

        except Exception as e:
            # Fallback: return dataframe with safe default signal values
            dataframe['chikou_filled'] = dataframe.get('close', pd.Series([1.0] * len(dataframe), index=dataframe.index))
            dataframe['chikou_bullish'] = False
            dataframe['chikou_bearish'] = False
            dataframe['chikou_clear_space'] = True
            dataframe['ichimoku_bullish'] = False
            dataframe['ichimoku_bearish'] = False
            dataframe['adx_strong'] = False
            dataframe['adx_bullish'] = False
            dataframe['adx_bearish'] = False
            dataframe['vwap_bullish'] = False
            dataframe['vwap_bearish'] = False
            dataframe['volatility_ok'] = True
            dataframe['long_score'] = 0.0
            dataframe['short_score'] = 0.0
            dataframe['long_score_transition'] = False
            dataframe['short_score_transition'] = False
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
        try:
            # Input validation
            if dataframe is None or len(dataframe) == 0:
                return dataframe

            required_columns = ['long_score', 'short_score', 'is_bull_market', 'is_bear_market', 'volatility_ratio', 'adx', 'long_score_transition', 'short_score_transition']
            missing_columns = [col for col in required_columns if col not in dataframe.columns]

            # Create missing columns with safe defaults
            for col in missing_columns:
                if 'score' in col:
                    dataframe[col] = 0.0
                elif 'bull_market' in col or 'bear_market' in col:
                    dataframe[col] = False
                elif 'transition' in col:
                    dataframe[col] = False
                elif col == 'volatility_ratio':
                    dataframe[col] = 1.0
                elif col == 'adx':
                    dataframe[col] = 20.0

            # === Base Stack Size Calculation ===
            dataframe['base_stack_size'] = self.base_position_size.value

            # === 1. Confidence-Based Scaling ===
            if self.confidence_scaling.value:
                # Higher confluence scores get larger position sizes
                max_possible_score = max(1, (  # Ensure non-zero
                    self.ichimoku_weight.value +
                    self.adx_weight.value +
                    self.vwap_weight.value +
                    (self.chikou_weight.value if self.chikou_confirmation.value else 0)
                ))

                # Normalize scores to 0-1 range with NaN handling
                long_confidence = (dataframe['long_score'] / max_possible_score).fillna(0.0)
                short_confidence = (dataframe['short_score'] / max_possible_score).fillna(0.0)

                # Clip confidence values to reasonable range
                long_confidence = np.clip(long_confidence, 0.0, 2.0)
                short_confidence = np.clip(short_confidence, 0.0, 2.0)

                # Apply confidence boost for high-quality signals
                long_confidence_multiplier = np.where(
                    long_confidence > self.high_confidence_threshold.value,
                    self.confidence_boost_multiplier.value,
                    1.0 + (long_confidence * self.confidence_scaling_factor.value)
                )

                short_confidence_multiplier = np.where(
                    short_confidence > self.high_confidence_threshold.value,
                    self.confidence_boost_multiplier.value,
                    1.0 + (short_confidence * self.confidence_scaling_factor.value)
                )

                dataframe['long_confidence_mult'] = np.clip(long_confidence_multiplier, 0.1, 5.0)
                dataframe['short_confidence_mult'] = np.clip(short_confidence_multiplier, 0.1, 5.0)
            else:
                dataframe['long_confidence_mult'] = 1.0
                dataframe['short_confidence_mult'] = 1.0

            # === 2. Market Regime Scaling ===
            if self.market_regime_scaling.value:
                # Separate regime multipliers for long and short positions

                # Long regime multiplier: boost longs in bull market, reduce in bear market
                long_regime_multiplier = np.where(
                    dataframe['is_bull_market'],
                    self.regime_alignment_boost.value,
                    np.where(
                        dataframe['is_bear_market'],
                        self.bear_market_long_reduction.value,
                        self.sideways_market_reduction.value
                    )
                )

                # Short regime multiplier: boost shorts in bear market, reduce in bull market
                short_regime_multiplier = np.where(
                    dataframe['is_bear_market'],
                    self.regime_alignment_boost.value,
                    np.where(
                        dataframe['is_bull_market'],
                        self.bull_market_short_reduction.value,
                        self.sideways_market_reduction.value
                    )
                )

                dataframe['long_regime_mult'] = np.clip(long_regime_multiplier, 0.1, 3.0)
                dataframe['short_regime_mult'] = np.clip(short_regime_multiplier, 0.1, 3.0)
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
                        1.0
                    )
                )
                dataframe['volatility_mult'] = np.clip(volatility_multiplier, 0.1, 3.0)
            else:
                dataframe['volatility_mult'] = 1.0

            # === 4. Momentum-Based Scaling ===
            if self.momentum_scaling.value:
                # Strong momentum (high ADX) allows larger positions
                adx_safe = np.clip(dataframe['adx'].fillna(20.0), 0.0, 100.0)
                momentum_multiplier = np.where(
                    adx_safe > self.momentum_boost_threshold.value,
                    1.0 + ((adx_safe - self.momentum_boost_threshold.value) / self.momentum_scaling_divisor.value),
                    1.0
                )
                # Cap the momentum boost
                momentum_multiplier = np.minimum(momentum_multiplier, self.max_momentum_multiplier.value)
                dataframe['momentum_mult'] = np.clip(momentum_multiplier, 0.1, 3.0)
            else:
                dataframe['momentum_mult'] = 1.0

            # === 5. Calculate Combined Stack Size ===
            # For long positions with safe multiplication
            dataframe['dynamic_long_stack'] = (
                dataframe['base_stack_size'] *
                dataframe['long_confidence_mult'] *
                dataframe['long_regime_mult'] *
                dataframe['volatility_mult'] *
                dataframe['momentum_mult']
            )

            # For short positions with safe multiplication
            dataframe['dynamic_short_stack'] = (
                dataframe['base_stack_size'] *
                dataframe['short_confidence_mult'] *
                dataframe['short_regime_mult'] *
                dataframe['volatility_mult'] *
                dataframe['momentum_mult']
            )

            # === 6. Apply Stack Size Limits ===
            dataframe['dynamic_long_stack'] = np.clip(
                dataframe['dynamic_long_stack'].fillna(self.min_stack_size.value),
                self.min_stack_size.value,
                self.max_stack_size.value
            )

            dataframe['dynamic_short_stack'] = np.clip(
                dataframe['dynamic_short_stack'].fillna(self.min_stack_size.value),
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

        except Exception as e:
            # Fallback: return dataframe with safe default stack size values
            dataframe['base_stack_size'] = self.base_position_size.value
            dataframe['long_confidence_mult'] = 1.0
            dataframe['short_confidence_mult'] = 1.0
            dataframe['long_regime_mult'] = 1.0
            dataframe['short_regime_mult'] = 1.0
            dataframe['volatility_mult'] = 1.0
            dataframe['momentum_mult'] = 1.0
            dataframe['dynamic_long_stack'] = self.base_position_size.value
            dataframe['dynamic_short_stack'] = self.base_position_size.value
            dataframe['portfolio_heat_reduction'] = 1.0
            dataframe['final_long_stack'] = 0.0  # No position if calculation fails
            dataframe['final_short_stack'] = 0.0  # No position if calculation fails
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

    def custom_entry_price(self, pair: str, trade: Optional[Trade], current_time: datetime,
                          proposed_rate: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        Enhanced Custom Entry Price with dynamic pricing based on candle analysis

        Features:
        - Dynamic fishing factor based on candle size relative to ATR
        - Intelligent price averaging using OHLC + proposed rate
        - Entry price increment logic to avoid duplicate entries
        - Integration with existing Heiken Ashi and trend analysis
        """
        # Return proposed rate if custom entry pricing is disabled
        if not self.enable_custom_entry_price.value:
            return proposed_rate

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is None or len(dataframe) == 0:
            return proposed_rate

        current_candle = dataframe.iloc[-1]

        # === Dynamic Fishing Factor Calculation ===
        # Base fishing factor from parameters
        base_fishing = self.entry_price_fishing_factor.value

        # Get candle size analysis (calculated in populate_indicators if enabled)
        candle_size_factor = current_candle.get('candle_size_target', 0.1)

        # Calculate dynamic fishing: larger candles = more aggressive entry pricing
        # Formula: fish deeper (lower price) on larger candles for better entry
        dynamic_fishing = base_fishing - (candle_size_factor * 0.05)  # Max 1.5% deeper on large candles
        dynamic_fishing = max(0.94, min(dynamic_fishing, 1.0))  # Bound between 94% and 100%

        # === Smart Price Averaging ===
        # Use OHLC + proposed rate for more balanced entry pricing
        if self.use_heiken_ashi.value and 'ha_open' in current_candle:
            # Use Heiken Ashi values for smoother entry pricing
            entry_price = (
                current_candle['ha_close'] +
                current_candle['ha_open'] +
                current_candle['ha_low'] +
                proposed_rate
            ) / 4
        else:
            # Use regular OHLC
            entry_price = (
                current_candle['close'] +
                current_candle['open'] +
                current_candle['low'] +
                proposed_rate
            ) / 4

        # Apply dynamic fishing factor
        entry_price = entry_price * dynamic_fishing

        # === Entry Price Increment Logic ===
        # Prevent duplicate entry prices by checking against stored last entry price
        # This helps avoid multiple entries at the same price level
        if hasattr(self, 'last_entry_price') and self.last_entry_price is not None:
            price_diff = abs(entry_price - self.last_entry_price) / self.last_entry_price
            if price_diff < 0.001:  # Less than 0.1% difference
                # Increment entry price slightly to avoid duplicate
                entry_price *= self.entry_price_increment.value

        # Store current entry price for next comparison
        self.last_entry_price = entry_price

        # === Logging for Live/Dry Run ===
        if self.dp.runmode.value in ('live', 'dry_run'):
            self.dp.send_msg(
                f"{pair} Custom Entry - Price: {entry_price:.8f} "
                f"(Proposed: {proposed_rate:.8f}, Fishing: {dynamic_fishing:.3f}, "
                f"Candle Size: {candle_size_factor:.3f})"
            )

        return entry_price

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """
        Enhanced Trade Exit Protection - Prevents exits below minimum profit thresholds

        Features:
        - Configurable minimum profit thresholds for different exit types
        - Special handling for ROI, partial exits, and trailing stops
        - FFT-based profit targets if available (experimental feature)
        - Protection against losing trades during volatile conditions

        Args:
            pair: Trading pair
            trade: Trade object
            order_type: Type of order being placed
            amount: Amount being sold
            rate: Exit rate
            time_in_force: Order time in force
            exit_reason: Reason for exit
            current_time: Current time
            **kwargs: Additional arguments

        Returns:
            bool: True to allow exit, False to block exit
        """
        # Return True (allow exit) if exit protection is disabled
        if not self.enable_exit_protection.value:
            return True

        # Get current market data
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is None or len(dataframe) == 0:
            return True  # Allow exit if no data available

        current_candle = dataframe.iloc[-1]

        # Calculate current profit ratio for this exit
        current_profit = trade.calc_profit_ratio(rate)

        # === Enhanced Exit Protection Logic ===

        # 1. ROI Exit Protection
        if exit_reason == 'roi' and current_profit < self.min_roi_exit_threshold.value:
            if self.dp.runmode.value in ('live', 'dry_run'):
                self.dp.send_msg(f"{pair} ROI exit blocked - Profit {current_profit:.1%} below threshold {self.min_roi_exit_threshold.value:.1%}")
            return False

        # 2. Partial Exit Protection
        if exit_reason == 'partial_exit' and current_profit < self.min_partial_exit_threshold.value:
            if self.dp.runmode.value in ('live', 'dry_run'):
                self.dp.send_msg(f"{pair} Partial exit blocked - Profit {current_profit:.1%} below threshold {self.min_partial_exit_threshold.value:.1%}")
            return False

        # 3. Trailing Stop Protection
        if exit_reason == 'trailing_stop_loss' and current_profit < self.min_trailing_stop_threshold.value:
            if self.dp.runmode.value in ('live', 'dry_run'):
                self.dp.send_msg(f"{pair} Trailing stop blocked - Profit {current_profit:.1%} below threshold {self.min_trailing_stop_threshold.value:.1%}")
            return False

        # 4. FFT-Based Exit Protection (Experimental)
        # If FFT cycle detection is enabled and we have the FFT-based profit targets,
        # use them for more sophisticated exit protection
        if (self.enable_fft_cycle_detection.value and
            all(col in current_candle for col in ['h2_move_mean', 'h1_move_mean', 'h0_move_mean', 'cycle_move_mean'])):

            # Get FFT-based profit targets
            tp0 = current_candle['h2_move_mean']  # Smallest target
            tp1 = current_candle['h1_move_mean']  # Small target
            tp2 = current_candle['h0_move_mean']  # Medium target
            tp3 = current_candle['cycle_move_mean']  # Large target

            # Block certain exits if we haven't reached the first FFT profit target
            # This prevents early exits during strong trending moves
            if (exit_reason in ['roi', 'custom_exit'] and
                current_profit < tp0 and
                current_profit > 0):  # Only for profitable trades

                if self.dp.runmode.value in ('live', 'dry_run'):
                    self.dp.send_msg(
                        f"{pair} FFT exit protection - Profit {current_profit:.1%} below TP0 {tp0:.1%}"
                    )
                return False

        # 5. Volatility-Based Protection
        # In high volatility, be more conservative with exits to avoid noise
        volatility_ratio = current_candle.get('volatility_ratio', 1.0)
        if (volatility_ratio > 1.8 and  # High volatility
            exit_reason in ['roi', 'exit_signal'] and
            current_profit < self.min_roi_exit_threshold.value * 1.5):  # 1.5x normal threshold

            if self.dp.runmode.value in ('live', 'dry_run'):
                self.dp.send_msg(
                    f"{pair} High volatility exit protection - Profit {current_profit:.1%} "
                    f"below enhanced threshold {self.min_roi_exit_threshold.value * 1.5:.1%}"
                )
            return False

        # 6. Never block stop loss exits (safety first)
        # Stop losses should always be honored for risk management
        if exit_reason in ['stop_loss', 'stoploss']:
            return True

        # 7. Never block forced exits or emergency exits
        if exit_reason in ['force_exit', 'emergency_exit', 'force_sell']:
            return True

        # === Additional Logging for Allowed Exits ===
        if self.dp.runmode.value in ('live', 'dry_run'):
            self.dp.send_msg(
                f"{pair} Exit allowed - Reason: {exit_reason}, "
                f"Profit: {current_profit:.1%}, Rate: {rate:.8f}"
            )

        # Allow exit if none of the protection conditions were triggered
        return True
