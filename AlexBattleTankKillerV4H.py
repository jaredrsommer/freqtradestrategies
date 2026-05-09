import logging
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import talib.abstract as ta
import pandas_ta as pta  # pandas_ta is imported but not explicitly used in the provided code.
# If it's for future use or part of an older version, that's okay.
# Otherwise, it can be removed if not needed.
from scipy.signal import argrelextrema

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, BooleanParameter
from freqtrade.persistence import Trade

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)

# Define Murrey Math level names for consistency
MML_LEVEL_NAMES = [
    "[-3/8]P", "[-2/8]P", "[-1/8]P", "[0/8]P", "[1/8]P",
    "[2/8]P", "[3/8]P", "[4/8]P", "[5/8]P", "[6/8]P",
    "[7/8]P", "[8/8]P", "[+1/8]P", "[+2/8]P", "[+3/8]P"
]
def calculate_minima_maxima(df, window):
    if df is None or df.empty:
        return np.zeros(0), np.zeros(0)

    minima = np.zeros(len(df))
    maxima = np.zeros(len(df))

    for i in range(window, len(df)):
        window_data = df['ha_close'].iloc[i - window:i + 1]
        if df['ha_close'].iloc[i] == window_data.min() and (window_data == df['ha_close'].iloc[i]).sum() == 1:
            minima[i] = -window
        if df['ha_close'].iloc[i] == window_data.max() and (window_data == df['ha_close'].iloc[i]).sum() == 1:
            maxima[i] = window

    return minima, maxima


class AlexBattleTankKillerV40H(IStrategy):
    """
    Enhanced strategy on the 15-minute timeframe with Market Correlation Filters.

    Key improvements:
      - Added Hurst..
      - Added Option to use Stoploss on Exchange for Trades
      - Fixed Stop Loss and Added Long and Short Exits
      - Dynamic stoploss based on ATR.
      - Dynamic leverage calculation.
      - Murrey Math level calculation (rolling window for performance).
      - Enhanced DCA (Average Price) logic.
      - Translated to English and code structured for clarity.
      - Parameterization of internal constants for optimization.
      - Changed Exit Signals for Opposite.
      - Change SL to -0.07
      - Changed stake amout for renentry
      - FIXED: Prevents opening opposite position when trade is active
      - FIXED: Trailing stop properly disabled
      - NEW: Market correlation filters for better entry timing
    """

    # General strategy parameters
    timeframe = "1h"
    startup_candle_count: int = 200
    stoploss = -0.07  # Fixed 8% stop loss
    use_custom_stoploss = True # Disable custom logic
    stoploss_on_exchange = False  # Let exchange handle it
    stoploss_on_exchange_interval = 60  # Check every 60 seconds
    #trailing_stop = False
    #trailing_stop_positive = 0.015  # Start trailing at 1.5%
    #trailing_stop_positive_offset = 0.03   # Begin after 3% profit
    #trailing_only_offset_is_reached = True
    position_adjustment_enable = True
    can_short = False
    use_exit_signal = True
    ignore_roi_if_entry_signal = False
    max_stake_per_trade = 5000.0
    max_portfolio_percentage_per_trade = 0.05
    max_entry_position_adjustment = 1
    process_only_new_candles = True
    max_dca_orders = 1
    max_total_stake_per_pair = 10
    max_single_dca_amount = 4
    use_custom_exits_advanced = True
    use_emergency_exits = True
    
    enable_position_flip = True
    disable_volatility_filter = BooleanParameter(default=True, space="buy", optimize=False)
    # 🎯 HURST PARAMETERS
    hurst_mean_rev = DecimalParameter(0.2, 0.5, default=0.4, decimals=2, space="buy")
    hurst_trending = DecimalParameter(0.6, 0.9, default=0.7, decimals=2, space="buy")
    # 🎯 REENTRY PARAMETERS
    rebuy_pullback_pct = DecimalParameter(0.01, 0.05, default=0.03, decimals=3, space="buy")
    reentry_cooldown_candles = IntParameter(5, 20, default=10, space="buy")
    enable_reentry = BooleanParameter(default=False, space='buy')
    reentry_cooldown_minutes = IntParameter(15, 120, default=45, space='buy')     # 15min-2h (1-8 candles)
    max_reentries_per_direction = IntParameter(1, 3, default=2, space='buy')      # Reduced from 7→2
    enable_quick_reentry_on_roi = BooleanParameter(default=True, space='buy')
    quick_reentry_cooldown = IntParameter(15, 45, default=30, space='buy')        # 15-45min (1-3 candles)
    reentry_signal_strength_threshold = DecimalParameter(0.3, 0.8, default=0.4, space='buy')

    signal_quality_enabled = BooleanParameter(default=True, space='buy', optimize=True)
    min_signal_strength = DecimalParameter(0.3, 0.8, default=0.5, space='buy', optimize=True)

    # 🎯 NEW SIGNAL FLIP PARAMETERS
    signal_flip_enabled = BooleanParameter(default=True, space="sell", optimize=True, load=True)  # NEW
    signal_flip_min_profit = DecimalParameter(-0.05, 0.02, default=-0.02, decimals=3, space="sell", optimize=True, load=True)  # NEW
    signal_flip_max_loss = DecimalParameter(-0.10, -0.03, default=-0.06, decimals=2, space="sell", optimize=True, load=True)  # NEW
    
    # Volume confirmation (referenced but not defined!)
    volume_confirmation_enabled = BooleanParameter(default=True, space='buy', optimize=True)
    require_volume_confirmation = BooleanParameter(default=False, space='buy', optimize=True)
    # Signal confirmation parameters
    signal_confirmation_candles = IntParameter(1, 5, default=2, space="buy", optimize=True, load=True)
    signal_cooldown_period = IntParameter(3, 15, default=5, space="buy", optimize=True, load=True)
    min_signal_strength = DecimalParameter(0.7, 0.9, default=0.8, decimals=2, space="buy", optimize=True, load=True)

    # Trend stability parameters  
    trend_stability_period = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)
    min_trend_strength = DecimalParameter(0.01, 0.05, default=0.02, decimals=3, space="buy", optimize=True, load=True)

    # Signal filtering parameters
    signal_quality_enabled = BooleanParameter(default=True, space='buy')
    min_signal_strength = DecimalParameter(0.3, 0.8, default=0.5, space='buy')
    
    flip_profit_threshold = DecimalParameter(low=0.01, high=0.10, default=0.03, decimals=3, space="buy", optimize=True, load=True)
    # 🚀 FOLLOW THE PRICE PARAMETERS (NEU)
    follow_price_enabled = BooleanParameter(default=False, space="sell", optimize=True, load=True)
    follow_price_activation_profit = DecimalParameter(0.01, 0.05, default=0.02, decimals=3, space="sell", optimize=True, load=True)  # CHANGED: Was 0.01, now 0.02
    follow_price_pullback_pct = DecimalParameter(0.005, 0.03, default=0.015, decimals=3, space="sell", optimize=True, load=True)  # CHANGED: Range tightened
    follow_price_min_profit = DecimalParameter(0.005, 0.04, default=0.01, decimals=3, space="sell", optimize=True, load=True)
    
    # 🎯 MML-BASIERTE FOLLOW PRICE (NEU)
    follow_price_use_mml = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    follow_price_mml_buffer = DecimalParameter(0.001, 0.01, default=0.003, decimals=4, space="sell", optimize=True, load=True)
    
    # 📊 MARKET CONDITION ADJUSTMENTS (NEU)
    follow_price_market_adjustment = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    follow_price_volatile_multiplier = DecimalParameter(0.5, 1.0, default=0.7, decimals=2, space="sell", optimize=True, load=True)
    follow_price_bearish_multiplier = DecimalParameter(0.6, 1.0, default=0.8, decimals=2, space="sell", optimize=True, load=True)
    
    # 🔧 ATR STOPLOSS PARAMETERS (Anpassbar machen)
    atr_stoploss_multiplier = DecimalParameter(2.5, 3.5, default=2.8, space="sell")
    atr_stoploss_minimum = DecimalParameter(-0.30, -0.15, default=-0.25, space="sell")
    atr_stoploss_maximum = DecimalParameter(-0.12, -0.06, default=-0.08, space="sell")
    early_trade_multiplier = DecimalParameter(1.2, 1.6, default=1.4, space="sell")
    early_trade_hours = DecimalParameter(1.0, 3.0, default=2.0, space="sell")
    # DCA parameters
    initial_safety_order_trigger = DecimalParameter(
        low=-0.02, high=-0.01, default=-0.018, decimals=3, space="buy", optimize=True, load=True
    )
    max_safety_orders = IntParameter(1, 3, default=2, space="buy", optimize=True, load=True)
    safety_order_step_scale = DecimalParameter(
        low=1.05, high=1.5, default=1.25, decimals=2, space="buy", optimize=True, load=True
    )
    safety_order_volume_scale = DecimalParameter(
        low=1.1, high=2.0, default=1.4, decimals=1, space="buy", optimize=True, load=True
    )
    h2 = IntParameter(20, 60, default=40, space="buy", optimize=True, load=True)
    h1 = IntParameter(10, 40, default=20, space="buy", optimize=True, load=True)
    h0 = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)
    cp = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)

    # Entry parameters
    increment_for_unique_price = DecimalParameter(
        low=1.0005, high=1.002, default=1.001, decimals=4, space="buy", optimize=True, load=True
    )
    last_entry_price: Optional[float] = None

    # Protection parameters
    cooldown_lookback = IntParameter(2, 48, default=1, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=4, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # Murrey Math level parameters
    mml_const1 = DecimalParameter(1.0, 1.1, default=1.0699, decimals=4, space="buy", optimize=True, load=True)
    mml_const2 = DecimalParameter(0.99, 1.0, default=0.99875, decimals=5, space="buy", optimize=True, load=True)
    indicator_mml_window = IntParameter(32, 128, default=64, space="buy", optimize=True, load=True)

    # Dynamic Stoploss parameters
    use_dynamic_stoploss = BooleanParameter(default=True, space='sell', optimize=True)
    atr_multiplier = DecimalParameter(1.5, 3.0, default=2.0, space='sell', optimize=True)
    max_stoploss = DecimalParameter(-0.25, -0.10, default=-0.15, space='sell', optimize=True)
    min_stoploss = DecimalParameter(-0.10, -0.05, default=-0.08, space='sell', optimize=True)
    stoploss_atr_multiplier = DecimalParameter(1.0, 3.0, default=1.5, decimals=1, space="sell", optimize=True,
                                               load=True)
    stoploss_max_reasonable = DecimalParameter(-0.30, -0.10, default=-0.20, decimals=2, space="sell", optimize=True,
                                               load=True)
    # === Hyperopt Parameters ===
    dominance_threshold = IntParameter(1, 10, default=3, space="buy", optimize=True)
    tightness_factor = DecimalParameter(0.5, 2.0, default=1.0, space="buy", optimize=True)
    long_rsi_threshold = IntParameter(50, 65, default=55, space="buy", optimize=True)
    short_rsi_threshold = IntParameter(30, 45, default=40, space="sell", optimize=True)

    # Dynamic Leverage parameters
    leverage_window_size = IntParameter(20, 100, default=50, space="buy", optimize=True, load=True)
    leverage_base = DecimalParameter(5.0, 20.0, default=7.0, decimals=1, space="buy", optimize=True, load=True)
    leverage_rsi_low = DecimalParameter(20.0, 40.0, default=30.0, decimals=1, space="buy", optimize=True, load=True)
    leverage_rsi_high = DecimalParameter(60.0, 80.0, default=70.0, decimals=1, space="buy", optimize=True, load=True)
    leverage_long_increase_factor = DecimalParameter(1.1, 2.0, default=1.5, decimals=1, space="buy", optimize=True,
                                                     load=True)
    leverage_long_decrease_factor = DecimalParameter(0.3, 0.9, default=0.5, decimals=1, space="buy", optimize=True,
                                                     load=True)
    leverage_volatility_decrease_factor = DecimalParameter(0.5, 0.95, default=0.8, decimals=2, space="buy",
                                                           optimize=True, load=True)
    leverage_atr_threshold_pct = DecimalParameter(0.01, 0.05, default=0.03, decimals=3, space="buy", optimize=True,
                                                  load=True)

    # Indicator parameters
    indicator_extrema_order = IntParameter(3, 10, default=5, space="buy", optimize=True, load=True)
    indicator_rolling_window_threshold = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)
    indicator_rolling_check_window = IntParameter(2, 10, default=4, space="buy", optimize=True, load=True)

    # === Market Correlation Parameters ===
    # Bitcoin correlation parameters
    enable_btc_correlation = BooleanParameter(default=True, space="buy", optimize=False)
    btc_correlation_enabled = BooleanParameter(default=False, space="buy", optimize=False)
    btc_correlation_threshold = DecimalParameter(0.3, 0.8, default=0.5, decimals=2, space="buy", optimize=True)
    btc_trend_filter_enabled = BooleanParameter(default=False, space="buy", optimize=True)
    
    # Market Breadth
    # Add these to your strategy class:
    market_breadth_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    market_breadth_threshold = DecimalParameter(0.3, 0.6, default=0.45, space="buy", optimize=True)
    market_breadth_bull_threshold = DecimalParameter(0.35, 0.65, default=0.50, space="buy", optimize=True, load=True)
    market_breadth_bear_threshold = DecimalParameter(0.35, 0.65, default=0.50, space="buy", optimize=True, load=True)
    market_breadth_sensitivity = DecimalParameter(0.05, 0.20, default=0.10, space="buy", optimize=True, load=True)
    breadth_weighting_enabled = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    volume_confirmation_enabled = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    
    # Total market cap parameters
    total_mcap_filter_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    total_mcap_ma_period = IntParameter(20, 100, default=50, space="buy", optimize=True)
    
    # Market regime parameters
    regime_filter_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    regime_lookback_period = IntParameter(24, 168, default=72, space="buy", optimize=True)  # hours
    
    # Fear & Greed parameters
    fear_greed_enabled = BooleanParameter(default=False, space="buy", optimize=True)  # Optional
    fear_greed_extreme_threshold = IntParameter(20, 30, default=25, space="buy", optimize=True)
    fear_greed_greed_threshold = IntParameter(70, 80, default=75, space="buy", optimize=True)
    # Momentum
    avoid_strong_trends = BooleanParameter(default=True, space="buy", optimize=True)
    trend_strength_threshold = DecimalParameter(0.01, 0.05, default=0.02, space="buy", optimize=True)
    momentum_confirmation_candles = IntParameter(1, 5, default=2, space="buy", optimize=True)
    # ROI table (minutes to decimal)
    minimal_roi = {
        "0": 0.17,      
        "10": 0.15,     
        "20": 0.10,     
        "30": 0.08,    
        "45": 0.06,     # Slightly more aggressive here
        "75": 0.04,     # Add a step
        "120": 0.03,    
        "180": 0.015,   # Small profit after 3h
        "240": 0        
    }
    plot_config = {
        "main_plot": {
            # Price action and key levels
            "actual_stoploss": {"color": "white", "type": "line"},
            "follow_price_level": {"color": "green", "type": "line"},
            
            # Key MML levels on main chart for better visibility
            "[2/8]P": {"color": "#ff6b6b", "type": "line"},  # 25% - Support/Resistance
            "[4/8]P": {"color": "#4ecdc4", "type": "line"},  # 50% - Key level
            "[6/8]P": {"color": "#45b7d1", "type": "line"},  # 75% - Support/Resistance
            
            # Entry signals on main chart
            "enter_long": {"color": "lime", "type": "scatter"},
            "enter_short": {"color": "red", "type": "scatter"},
        },
        "subplots": {
            # 📊 EXIT SIGNALS ANALYSIS (NEW - Most Important)
            "exit_signals": {
                "exit_long": {"color": "#ff4757", "type": "scatter"},          # Red exit longs
                "exit_short": {"color": "#2ed573", "type": "scatter"},         # Green exit shorts
                "Signal_Flip_Short": {"color": "#ff9ff3", "type": "scatter"}, # Pink signal flips
                "Signal_Flip_Long": {"color": "#54a0ff", "type": "scatter"},   # Blue signal flips
                "Momentum_Loss": {"color": "#ff6b35", "type": "scatter"},      # Orange momentum loss
                "Support_Break": {"color": "#ff3838", "type": "scatter"},      # Dark red breaks
                "Resistance_Break": {"color": "#2ecc71", "type": "scatter"},   # Green breaks
            },
            
            # 🎯 TRADE PERFORMANCE INDICATORS (NEW)
            "trade_metrics": {
                "price_change_5": {"color": "#74b9ff", "type": "line"},       # 5-period price change
                "rsi_change": {"color": "#fd79a8", "type": "line"},           # RSI momentum
                "volume_surge": {"color": "#fdcb6e", "type": "line"},         # Volume spikes
            },
            
            # 📈 RSI WITH EXIT ZONES (ENHANCED)
            "rsi_analysis": {
                "rsi": {"color": "#6c5ce7", "type": "line"},
                "rsi_overbought_70": {"color": "#fd79a8", "type": "line"},    # 70 level
                "rsi_oversold_30": {"color": "#00b894", "type": "line"},      # 30 level
            },
            
            # 🏔️ EXTREMA ANALYSIS (KEEP - Important for entries)
            "extrema_analysis": {
                "s_extrema": {"color": "#f53580", "type": "line"},
                "minima_sort_threshold": {"color": "#4ae747", "type": "line"},
                "maxima_sort_threshold": {"color": "#5b5e4b", "type": "line"},
            },
            
            # 📍 MIN/MAX VISUALIZATION (KEEP - Entry confirmation)
            "min_max_viz": {
                "maxima": {"color": "#a29db9", "type": "scatter"},            # Changed to scatter
                "minima": {"color": "#aac7fc", "type": "scatter"},            # Changed to scatter
                "maxima_check": {"color": "#e17055", "type": "line"},
                "minima_check": {"color": "#74b9ff", "type": "line"},
            },
            
            # 🎯 ADDITIONAL MML LEVELS (ENHANCED)
            "murrey_math_levels": {
                "[1/8]P": {"color": "#d63031", "type": "line"},               # 12.5% - Extreme oversold
                "[3/8]P": {"color": "#fd79a8", "type": "line"},               # 37.5% - Support
                "[4/8]P": {"color": "#0984e3", "type": "line"},               # 50% - Key level (duplicate for subplot)
                "[5/8]P": {"color": "#00b894", "type": "line"},               # 62.5% - Resistance  
                "[7/8]P": {"color": "#e84393", "type": "line"},               # 87.5% - Extreme overbought
                "[8/8]P": {"color": "#d63031", "type": "line"},               # 100% - Top
                "[0/8]P": {"color": "#d63031", "type": "line"},               # 0% - Bottom
            },
            
            # 🌐 MARKET CORRELATION (KEEP - Market context)
            "market_correlation": {
                "btc_correlation": {"color": "#0984e3", "type": "line"},
                "market_breadth": {"color": "#00b894", "type": "line"},
                "market_score": {"color": "#6c5ce7", "type": "line"},
                "market_direction": {"color": "#fd79a8", "type": "line"},     # NEW
            },
            
            # 📊 MARKET REGIME (ENHANCED)
            "market_regime": {
                "market_volatility": {"color": "#e17055", "type": "line"},
                "mcap_trend": {"color": "#fdcb6e", "type": "line"},
                "market_adx": {"color": "#74b9ff", "type": "line"},           # NEW - Trend strength
            },
            
            # 📉 VOLUME ANALYSIS (NEW - Important for exit confirmation)
            "volume_analysis": {
                "volume": {"color": "#636e72", "type": "bar"},
                "volume_ratio": {"color": "#00cec9", "type": "line"},         # Volume vs average
                "avg_volume": {"color": "#fd79a8", "type": "line"},           # 20-period average
            },
            
            # 🎯 TREND ANALYSIS (NEW - For exit timing)
            "trend_analysis": {
                "trend_consistency": {"color": "#a29bfe", "type": "line"},    # Multi-timeframe trend
                "trend_strength": {"color": "#fd79a8", "type": "line"},       # Trend momentum
                "strong_uptrend": {"color": "#00b894", "type": "line"},       # Boolean uptrend
                "strong_downtrend": {"color": "#d63031", "type": "line"},     # Boolean downtrend
            },
            
            # 🔥 SIGNAL QUALITY (NEW - Signal strength analysis)
            "signal_quality": {
                "combined_signal_strength": {"color": "#fd79a8", "type": "line"},  # Overall signal strength
                "mml_signal_strength": {"color": "#74b9ff", "type": "line"},       # MML-based strength
                "rsi_signal_strength": {"color": "#00b894", "type": "line"},       # RSI-based strength
                "volume_strength": {"color": "#fdcb6e", "type": "line"},           # Volume confirmation
            }
        },
    }
    def get_or_init_reentry_data(self, pair: str):
        """
        📝 Initialize re-entry data for pair if not exists
        (Use this approach if __init__ causes issues)
        """
        
        # Initialize recent_exits if not exists
        if not hasattr(self, 'recent_exits'):
            self.recent_exits = {}
        
        if pair not in self.recent_exits:
            self.recent_exits[pair] = []
        
        # Initialize reentry_count if not exists
        if not hasattr(self, 'reentry_count'):
            self.reentry_count = {}
            
        if pair not in self.reentry_count:
            self.reentry_count[pair] = {'long': 0, 'short': 0}
        
        # Initialize signal_strength_cache if not exists
        if not hasattr(self, 'signal_strength_cache'):
            self.signal_strength_cache = {}
    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to download
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        
        # Add BTC data for correlation analysis
        informative_pairs.extend([
            ('BTC/USDT', '1h'),
            ('BTC/USDT', '4h'),
        ])
        
        # Add informative timeframes for current pairs
        for tf in ['1h', '4h']:
            informative_pairs.extend([(pair, tf) for pair in pairs])
        
        return informative_pairs

    def calculate_signal_strength(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        🔧 FIXED: Enhanced signal strength calculation that actually works
        """
        
        # Initialize all strength columns
        dataframe['rsi_signal_strength'] = 0.0
        dataframe['mml_signal_strength'] = 0.0
        dataframe['mml_bounce_strength'] = 0.0
        dataframe['volume_strength'] = 0.0
        dataframe['extrema_strength'] = 0.0
        
        # RSI signal strength
        rsi = dataframe['rsi'].fillna(50)
        
        # Long RSI strength
        dataframe.loc[rsi < 25, 'rsi_signal_strength'] = 1.0  # Very oversold
        dataframe.loc[rsi.between(25, 35), 'rsi_signal_strength'] = 0.8  # Oversold
        dataframe.loc[rsi.between(35, 40), 'rsi_signal_strength'] = 0.6  # Mildly oversold
        
        # Short RSI strength  
        dataframe.loc[rsi > 75, 'rsi_signal_strength'] = 1.0  # Very overbought
        dataframe.loc[rsi.between(65, 75), 'rsi_signal_strength'] = 0.8  # Overbought
        dataframe.loc[rsi.between(60, 65), 'rsi_signal_strength'] = 0.6  # Mildly overbought
        
        # MML signal strength (safe column access)
        try:
            if '[1/8]P' in dataframe.columns and '[7/8]P' in dataframe.columns:
                # Extreme MML signals
                dataframe.loc[dataframe['close'] < dataframe['[1/8]P'], 'mml_signal_strength'] = 1.0
                dataframe.loc[dataframe['close'] > dataframe['[7/8]P'], 'mml_signal_strength'] = 1.0
            
            if '[2/8]P' in dataframe.columns and '[6/8]P' in dataframe.columns:
                # Medium MML signals
                dataframe.loc[dataframe['close'] < dataframe['[2/8]P'], 'mml_signal_strength'] = 0.8
                dataframe.loc[dataframe['close'] > dataframe['[6/8]P'], 'mml_signal_strength'] = 0.8
                
                # MML bounces
                bounce_conditions = (
                    ((dataframe['low'] <= dataframe['[2/8]P']) & (dataframe['close'] > dataframe['[2/8]P'])) |
                    ((dataframe['high'] >= dataframe['[6/8]P']) & (dataframe['close'] < dataframe['[6/8]P']))
                )
                dataframe.loc[bounce_conditions, 'mml_bounce_strength'] = 0.9
                
        except Exception as e:
            logger.warning(f"MML signal strength calculation error: {e}")
        
        # Volume strength (safe calculation)
        try:
            avg_volume = dataframe['volume'].rolling(20).mean()
            
            dataframe.loc[dataframe['volume'] > avg_volume * 1.5, 'volume_strength'] = 1.0
            dataframe.loc[dataframe['volume'] > avg_volume * 1.2, 'volume_strength'] = 0.8
            dataframe.loc[dataframe['volume'] > avg_volume, 'volume_strength'] = 0.6
            dataframe.loc[dataframe['volume'] <= avg_volume, 'volume_strength'] = 0.3
            
        except Exception as e:
            logger.warning(f"Volume strength calculation error: {e}")
            dataframe['volume_strength'] = 0.5
        
        # Extrema strength (safe access)
        try:
            if 'minima' in dataframe.columns:
                dataframe.loc[dataframe['minima'] == 1, 'extrema_strength'] = 0.8
            if 'maxima' in dataframe.columns:
                dataframe.loc[dataframe['maxima'] == 1, 'extrema_strength'] = 0.8
        except Exception as e:
            logger.warning(f"Extrema strength calculation error: {e}")
        
        # Combined signal strength (weighted average)
        dataframe['combined_signal_strength'] = (
            dataframe['rsi_signal_strength'] * 0.3 +
            dataframe['mml_signal_strength'] * 0.25 +
            dataframe['mml_bounce_strength'] * 0.2 +
            dataframe['volume_strength'] * 0.15 +
            dataframe['extrema_strength'] * 0.1
        ).clip(0, 1)
        
        return dataframe
    
    def check_signal_confirmation(self, dataframe: pd.DataFrame, signal_type: str) -> pd.Series:
        """
        Check if signal persists for required confirmation period
        """
        confirmation_period = self.signal_confirmation_candles.value
        
        if signal_type == 'long':
            # Check if bullish conditions persist
            bullish_conditions = (
                (dataframe['rsi'] < 45) |  # Oversold or neutral
                (dataframe['close'] > dataframe['[4/8]P']) |  # Above 50% MML
                (dataframe['minima'] == 1) |  # Local bottom
                (dataframe['close'] > dataframe['close'].shift(1))  # Price rising
            )
            
            # Signal confirmed if conditions met for X candles
            confirmed = bullish_conditions.rolling(confirmation_period).sum() >= (confirmation_period * 0.7)
            
        elif signal_type == 'short':
            # Check if bearish conditions persist
            bearish_conditions = (
                (dataframe['rsi'] > 55) |  # Overbought or neutral
                (dataframe['close'] < dataframe['[4/8]P']) |  # Below 50% MML
                (dataframe['maxima'] == 1) |  # Local top
                (dataframe['close'] < dataframe['close'].shift(1))  # Price falling
            )
            
            # Signal confirmed if conditions met for X candles
            confirmed = bearish_conditions.rolling(confirmation_period).sum() >= (confirmation_period * 0.7)
        else:
            confirmed = pd.Series([False] * len(dataframe))
        
        return confirmed
    
    def check_signal_cooldown(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Prevent signals too close together
        """
        cooldown_period = self.signal_cooldown_period.value
        
        # Track recent signals
        dataframe['recent_long_signal'] = dataframe['enter_long'].rolling(cooldown_period).sum()
        dataframe['recent_short_signal'] = dataframe['enter_short'].rolling(cooldown_period).sum()
        
        # Only allow new signals if no recent signals
        can_signal = (
            (dataframe['recent_long_signal'] == 0) & 
            (dataframe['recent_short_signal'] == 0)
        )
        
        return can_signal

    def detect_market_state(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            """
            Detect current market state to avoid bad timing
            """
            
            # Price action analysis
            dataframe['price_volatility'] = dataframe['atr'] / dataframe['close']
            dataframe['volatility_percentile'] = dataframe['price_volatility'].rolling(50).rank(pct=True)
            
            # Volume analysis
            dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume'].rolling(20).mean()
            dataframe['volume_increasing'] = dataframe['volume'] > dataframe['volume'].shift(1)
            
            # Momentum analysis
            dataframe['momentum_3'] = dataframe['close'].pct_change(3)
            dataframe['momentum_5'] = dataframe['close'].pct_change(5)
            dataframe['momentum_10'] = dataframe['close'].pct_change(10)
            
            # Trend consistency
            dataframe['consistent_uptrend'] = (
                (dataframe['momentum_3'] > 0) &
                (dataframe['momentum_5'] > 0) &
                (dataframe['momentum_10'] > 0)
            )
            
            dataframe['consistent_downtrend'] = (
                (dataframe['momentum_3'] < 0) &
                (dataframe['momentum_5'] < 0) &
                (dataframe['momentum_10'] < 0)
            )
            
            # Market state classification
            dataframe['market_state'] = 'neutral'
            
            # Strong trending states
            dataframe.loc[
                dataframe['consistent_uptrend'] & 
                (dataframe['adx'] > 30) & 
                (dataframe['volume_ratio'] > 1.2),
                'market_state'
            ] = 'strong_uptrend'
            
            dataframe.loc[
                dataframe['consistent_downtrend'] & 
                (dataframe['adx'] > 30) & 
                (dataframe['volume_ratio'] > 1.2),
                'market_state'
            ] = 'strong_downtrend'
            
            # Choppy/ranging states
            dataframe.loc[
                (dataframe['adx'] < 20) & 
                (dataframe['volatility_percentile'] < 0.5),
                'market_state'
            ] = 'ranging'
            
            # High volatility states
            dataframe.loc[
                dataframe['volatility_percentile'] > 0.8,
                'market_state'
            ] = 'high_volatility'
            
            # Reversal detection
            dataframe['potential_reversal'] = (
                # Price making new extremes but momentum diverging
                (
                    (dataframe['high'] >= dataframe['high'].rolling(10).max()) &
                    (dataframe['momentum_3'] < dataframe['momentum_3'].shift(3))
                ) |
                (
                    (dataframe['low'] <= dataframe['low'].rolling(10).min()) &
                    (dataframe['momentum_3'] > dataframe['momentum_3'].shift(3))
                )
            )
            
            # Breakout detection
            dataframe['potential_breakout'] = (
                (dataframe['close'] > dataframe['high'].rolling(20).max().shift(1)) |
                (dataframe['close'] < dataframe['low'].rolling(20).min().shift(1))
            ) & (dataframe['volume_ratio'] > 1.5)
            
            return dataframe
    
    def get_timing_filters(self, dataframe: pd.DataFrame, direction: str) -> pd.Series:
        """
        Get timing-based filters to avoid bad entry timing
        """
        
        if direction == 'long':
            good_timing = (
                # Avoid entering during strong downtrends
                (dataframe['market_state'] != 'strong_downtrend') &
                
                # Prefer breakouts or oversold conditions
                (
                    dataframe['potential_breakout'] |
                    (dataframe['rsi'] < 40) |
                    (dataframe['market_state'] == 'strong_uptrend')
                ) &
                
                # Avoid high volatility unless extremely oversold
                (
                    (dataframe['market_state'] != 'high_volatility') |
                    (dataframe['rsi'] < 25)
                ) &
                
                # Volume confirmation if required
                (
                    (~self.require_volume_confirmation.value) |
                    (dataframe['volume_ratio'] > 1.1)
                )
            )
            
        else:  # short
            good_timing = (
                # Avoid entering during strong uptrends
                (dataframe['market_state'] != 'strong_uptrend') &
                
                # Prefer breakdowns or overbought conditions
                (
                    dataframe['potential_breakout'] |
                    (dataframe['rsi'] > 60) |
                    (dataframe['market_state'] == 'strong_downtrend')
                ) &
                
                # Avoid high volatility unless extremely overbought
                (
                    (dataframe['market_state'] != 'high_volatility') |
                    (dataframe['rsi'] > 75)
                ) &
                
                # Volume confirmation if required
                (
                    (~self.require_volume_confirmation.value) |
                    (dataframe['volume_ratio'] > 1.1)
                )
            )
        
        return good_timing
    # Helper method to check if we have an active position in the opposite direction
    def has_active_trade(self, pair: str, side: str) -> bool:
        """
        Check if there's an active trade in the specified direction
        """
        try:
            trades = Trade.get_open_trades()
            for trade in trades:
                if trade.pair == pair:
                    if side == "long" and not trade.is_short:
                        return True
                    elif side == "short" and trade.is_short:
                        return True
        except Exception as e:
            logger.warning(f"Error checking active trades for {pair}: {e}")
        return False

    @staticmethod
    def _calculate_mml_core(mn: float, finalH: float, mx: float, finalL: float,
                            mml_c1: float, mml_c2: float) -> Dict[str, float]:
        dmml_calc = ((finalH - finalL) / 8.0) * mml_c1
        if dmml_calc == 0 or np.isinf(dmml_calc) or np.isnan(dmml_calc) or finalH == finalL:
            return {key: finalL for key in MML_LEVEL_NAMES}
        mml_val = (mx * mml_c2) + (dmml_calc * 3)
        if np.isinf(mml_val) or np.isnan(mml_val):
            return {key: finalL for key in MML_LEVEL_NAMES}
        ml = [mml_val - (dmml_calc * i) for i in range(16)]
        return {
            "[-3/8]P": ml[14], "[-2/8]P": ml[13], "[-1/8]P": ml[12],
            "[0/8]P": ml[11], "[1/8]P": ml[10], "[2/8]P": ml[9],
            "[3/8]P": ml[8], "[4/8]P": ml[7], "[5/8]P": ml[6],
            "[6/8]P": ml[5], "[7/8]P": ml[4], "[8/8]P": ml[3],
            "[+1/8]P": ml[2], "[+2/8]P": ml[1], "[+3/8]P": ml[0],
        }

    def calculate_rolling_murrey_math_levels_optimized(self, df: pd.DataFrame, window_size: int) -> Dict[str, pd.Series]:
        """
        OPTIMIZED: Calculate Murrey Math Levels using a rolling window with caching and interpolation.
        Only recalculates every 5 candles for performance, interpolating intermediate values.
        Compatible with Pandas 2.0+ and handles edge cases robustly.
        """
        # Initialize cache if not present
        if not hasattr(self, 'mml_cache'):
            self.mml_cache = {}
        pair = getattr(self, 'current_pair', 'default_pair')  # Fallback for pair context
        if pair not in self.mml_cache or len(self.mml_cache[pair]) != len(df):
            murrey_levels_data = {key: [np.nan] * len(df) for key in MML_LEVEL_NAMES}
            
            # Use rolling max/min with minimum periods to avoid NaN issues
            rolling_high = df["high"].rolling(window=window_size, min_periods=1).max()
            rolling_low = df["low"].rolling(window=window_size, min_periods=1).min()
            mml_c1 = self.mml_const1.value
            mml_c2 = self.mml_const2.value
            
            # Optimize by calculating only every 5th candle
            calculation_step = 5
            
            for i in range(0, len(df), calculation_step):
                if i < window_size - 1:
                    continue
                    
                mn_period = rolling_low.iloc[i]
                mx_period = rolling_high.iloc[i]
                current_close = df["close"].iloc[i]
                
                # Handle edge cases where data might be invalid
                if pd.isna(mn_period) or pd.isna(mx_period) or mn_period == mx_period or mn_period == 0:
                    for key in MML_LEVEL_NAMES:
                        murrey_levels_data[key][i] = current_close
                    continue
                    
                dmml_calc = ((mx_period - mn_period) / 8.0) * mml_c1
                if dmml_calc == 0 or np.isinf(dmml_calc) or np.isnan(dmml_calc):
                    for key in MML_LEVEL_NAMES:
                        murrey_levels_data[key][i] = current_close
                    continue
                    
                mml_val = (mx_period * mml_c2) + (dmml_calc * 3)
                if np.isinf(mml_val) or np.isnan(mml_val):
                    for key in MML_LEVEL_NAMES:
                        murrey_levels_data[key][i] = current_close
                    continue
                    
                ml = [mml_val - (dmml_calc * i) for i in range(16)]
                levels = {
                    "[-3/8]P": ml[14], "[-2/8]P": ml[13], "[-1/8]P": ml[12],
                    "[0/8]P": ml[11], "[1/8]P": ml[10], "[2/8]P": ml[9],
                    "[3/8]P": ml[8], "[4/8]P": ml[7], "[5/8]P": ml[6],
                    "[6/8]P": ml[5], "[7/8]P": ml[4], "[8/8]P": ml[3],
                    "[+1/8]P": ml[2], "[+2/8]P": ml[1], "[+3/8]P": ml[0],
                }
                
                for key in MML_LEVEL_NAMES:
                    murrey_levels_data[key][i] = levels.get(key, current_close)
            
            # Interpolate and fill missing values efficiently
            for key in MML_LEVEL_NAMES:
                series = pd.Series(murrey_levels_data[key], index=df.index)
                series = series.interpolate(method='linear', limit_direction='both').ffill().bfill()
                murrey_levels_data[key] = series.tolist()
            
            self.mml_cache[pair] = {key: pd.Series(data, index=df.index) for key, data in murrey_levels_data.items()}
        
        return self.mml_cache[pair]

    def calculate_market_correlation_simple(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Simplified correlation calculation that avoids index issues
        """
        pair = metadata['pair']
        base_currency = pair.split('/')[0]
        
        # Skip if this IS BTC
        if base_currency == 'BTC':
            dataframe['btc_correlation'] = 1.0
            dataframe['btc_trend'] = 1
            dataframe['btc_close'] = dataframe['close']
            dataframe['btc_sma20'] = dataframe['close'].rolling(20).mean()
            dataframe['btc_sma50'] = dataframe['close'].rolling(50).mean()
            dataframe['returns'] = dataframe['close'].pct_change()
            return dataframe
        
        # Try to get BTC data
        btc_pairs = ["BTC/USDT", "BTC/USDT", "BTC/USD"]
        btc_dataframe = None
        
        for btc_pair in btc_pairs:
            try:
                btc_dataframe, _ = self.dp.get_analyzed_dataframe(btc_pair, self.timeframe)
                if btc_dataframe is not None and not btc_dataframe.empty and len(btc_dataframe) >= 50:
                    logger.info(f"Using {btc_pair} for simplified BTC correlation with {pair}")
                    break
            except:
                continue
        
        # If no BTC data, use neutral values
        if btc_dataframe is None or btc_dataframe.empty:
            dataframe['btc_correlation'] = 0.5
            dataframe['btc_trend'] = 0
            dataframe['btc_close'] = dataframe['close']
            dataframe['btc_sma20'] = dataframe['close']
            dataframe['btc_sma50'] = dataframe['close']
            dataframe['returns'] = dataframe['close'].pct_change()
            logger.warning(f"No BTC data for {pair}, using neutral correlation")
            return dataframe
        
        # Simple correlation using only latest values
        try:
            # Use shorter period to avoid length issues
            correlation_period = min(20, len(dataframe) // 5, len(btc_dataframe) // 5)
            correlation_period = max(5, correlation_period)
            
            # Calculate returns
            pair_returns = dataframe['close'].pct_change().dropna()
            btc_returns = btc_dataframe['close'].pct_change().dropna()
            
            # Use only last N periods for correlation
            if len(pair_returns) >= correlation_period and len(btc_returns) >= correlation_period:
                pair_recent = pair_returns.tail(correlation_period)
                btc_recent = btc_returns.tail(correlation_period)
                
                # Align lengths
                min_len = min(len(pair_recent), len(btc_recent))
                correlation = pair_recent.tail(min_len).corr(btc_recent.tail(min_len))
                
                if pd.isna(correlation):
                    correlation = 0.5
            else:
                correlation = 0.5
            
            # BTC trend (simple)
            btc_close = btc_dataframe['close'].iloc[-1]
            btc_sma20 = btc_dataframe['close'].rolling(20).mean().iloc[-1]
            btc_sma50 = btc_dataframe['close'].rolling(50).mean().iloc[-1]
            
            if pd.isna(btc_sma20) or pd.isna(btc_sma50):
                btc_trend = 0
            elif btc_close > btc_sma20 > btc_sma50:
                btc_trend = 1
            elif btc_close < btc_sma20 < btc_sma50:
                btc_trend = -1
            else:
                btc_trend = 0
            
            # Set constant values for entire dataframe
            dataframe['btc_correlation'] = correlation
            dataframe['btc_trend'] = btc_trend
            dataframe['btc_close'] = btc_close
            dataframe['btc_sma20'] = btc_sma20
            dataframe['btc_sma50'] = btc_sma50
            
            logger.debug(f"{pair} Simple correlation: {correlation:.3f}, BTC trend: {btc_trend}")
            
        except Exception as e:
            logger.warning(f"Simple correlation failed for {pair}: {e}")
            dataframe['btc_correlation'] = 0.5
            dataframe['btc_trend'] = 0
            dataframe['btc_close'] = dataframe['close']
            dataframe['btc_sma20'] = dataframe['close']
            dataframe['btc_sma50'] = dataframe['close']
        
        # Always calculate returns
        dataframe['returns'] = dataframe['close'].pct_change()
        
        return dataframe
    def calculate_market_breadth_weighted(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        🔧 FIXED: Proper logging levels for market breadth calculation
        """
        current_pair = metadata['pair']
        is_futures = ':' in current_pair
        
        if is_futures:
            settlement = current_pair.split(':')[1]
            primary_pairs = [
                (f"BTC/USDT:{settlement}", 0.40),
                (f"ETH/USDT:{settlement}", 0.25),
                (f"BNB/USDT:{settlement}", 0.10),
                (f"SOL/USDT:{settlement}", 0.08),
                (f"XRP/USDT:{settlement}", 0.07),
                (f"ADA/USDT:{settlement}", 0.05),
                (f"AVAX/USDT:{settlement}", 0.02),
            ]
            
            fallback_pairs = [
                ("BTC/USDT", 0.40),
                ("ETH/USDT", 0.25),
                ("BNB/USDT", 0.10),
                ("SOL/USDT", 0.08),
                ("XRP/USDT", 0.07),
                ("ADA/USDT", 0.05),
                ("AVAX/USDT", 0.02),
            ]
        else:
            primary_pairs = [
                ("BTC/USDT", 0.40),
                ("ETH/USDT", 0.25), 
                ("BNB/USDT", 0.10),
                ("SOL/USDT", 0.08),
                ("XRP/USDT", 0.07),
                ("ADA/USDT", 0.05),
                ("AVAX/USDT", 0.02),
            ]
            fallback_pairs = []
        
        bullish_weight = 0.0
        bearish_weight = 0.0
        neutral_weight = 0.0
        total_weight = 0.0
        pairs_checked = 0
        
        logger.debug(f"🔍 DEBUG BREADTH for {metadata['pair']}:")  # Changed from ERROR to DEBUG
        
        # Try primary pairs with proper logging levels
        for check_pair, weight in primary_pairs:
            try:
                pair_data = self.dp.get_pair_dataframe(check_pair, self.timeframe)
                
                if pair_data is None or pair_data.empty:
                    logger.debug(f"   ❌ {check_pair}: No data")  # Changed from ERROR to DEBUG
                    continue
                    
                if len(pair_data) < 20:
                    logger.debug(f"   ❌ {check_pair}: Only {len(pair_data)} candles")  # Changed from ERROR to DEBUG
                    continue
                    
                current_close = pair_data['close'].iloc[-1]
                sma20 = pair_data['close'].rolling(20).mean().iloc[-1]
                
                if pd.isna(current_close) or pd.isna(sma20) or sma20 <= 0:
                    logger.debug(f"   ❌ {check_pair}: Invalid data (close: {current_close}, sma: {sma20})")  # Changed from ERROR to DEBUG
                    continue
                
                # Enhanced threshold logic
                threshold = 1.002  # 0.2% buffer
                price_vs_sma = current_close / sma20
                
                logger.debug(f"   📊 {check_pair}: Close=${current_close:.3f}, SMA20=${sma20:.3f}, Ratio={price_vs_sma:.6f}")  # Changed from ERROR to DEBUG
                
                if current_close > sma20 * threshold:
                    bullish_weight += weight
                    trend_status = "BULLISH"
                elif current_close < sma20 / threshold:
                    bearish_weight += weight
                    trend_status = "BEARISH"
                else:
                    neutral_weight += weight
                    trend_status = "NEUTRAL"
                
                total_weight += weight
                pairs_checked += 1
                
                logger.debug(f"   ✅ {check_pair}: {trend_status} (weight: {weight:.2f})")  # Changed from ERROR to DEBUG
                
            except Exception as e:
                logger.warning(f"   💥 {check_pair} ERROR: {str(e)}")  # Changed from ERROR to WARNING
                continue
        
        # Enhanced calculation logic with proper logging
        logger.debug(f"📊 BREADTH CALCULATION:")  # Changed from ERROR to DEBUG
        logger.debug(f"   Bullish weight: {bullish_weight:.3f}")  # Changed from ERROR to DEBUG
        logger.debug(f"   Bearish weight: {bearish_weight:.3f}")  # Changed from ERROR to DEBUG
        logger.debug(f"   Neutral weight: {neutral_weight:.3f}")  # Changed from ERROR to DEBUG
        logger.debug(f"   Total weight: {total_weight:.3f}")  # Changed from ERROR to DEBUG
        logger.debug(f"   Pairs checked: {pairs_checked}")  # Changed from ERROR to DEBUG
        
        if pairs_checked > 0 and total_weight > 0:
            # Include neutral in calculation
            directional_weight = bullish_weight + bearish_weight
            
            if directional_weight > 0:
                market_breadth = bullish_weight / directional_weight
            else:
                # All neutral = 50%
                market_breadth = 0.5
                logger.debug(f"   🔧 All pairs neutral, setting breadth to 50%")  # Changed from ERROR to DEBUG
            
            # Market direction
            if bullish_weight > bearish_weight:
                market_direction = 1
            elif bearish_weight > bullish_weight:
                market_direction = -1
            else:
                market_direction = 0
                
            logger.info(f"📊 {metadata['pair']} Market Breadth: {market_breadth:.1%} ({pairs_checked} pairs)")  # Changed from ERROR to INFO
            
        else:
            logger.info(f"🔄 {metadata['pair']} No breadth data - using fallback calculation")  # Changed from ERROR to INFO
            
            # Fallback: Use current pair trend
            try:
                current_close = dataframe['close'].iloc[-1]
                sma20 = dataframe['close'].rolling(20).mean().iloc[-1]
                
                if not pd.isna(sma20) and sma20 > 0:
                    pair_ratio = current_close / sma20
                    logger.debug(f"   📈 Current pair trend: {pair_ratio:.6f}")  # Changed from ERROR to DEBUG
                    
                    if pair_ratio > 1.01:  # 1% above SMA
                        market_breadth = 0.65
                        market_direction = 1
                        logger.debug(f"   📈 Pair bullish -> 65% breadth")  # Changed from ERROR to DEBUG
                    elif pair_ratio < 0.99:  # 1% below SMA  
                        market_breadth = 0.35
                        market_direction = -1
                        logger.debug(f"   📉 Pair bearish -> 35% breadth")  # Changed from ERROR to DEBUG
                    else:
                        market_breadth = 0.5
                        market_direction = 0
                        logger.debug(f"   📊 Pair neutral -> 50% breadth")  # Changed from ERROR to DEBUG
                else:
                    market_breadth = 0.5
                    market_direction = 0
                    logger.debug(f"   🔧 Invalid pair data -> 50% breadth")  # Changed from ERROR to DEBUG
                    
                pairs_checked = 1
                
            except Exception as e:
                logger.warning(f"   💥 Fallback failed: {e}")  # Changed from ERROR to WARNING
                market_breadth = 0.5
                market_direction = 0
                pairs_checked = 0
        
        # Prevent 0% breadth with proper logging
        if market_breadth == 0.0:
            logger.info(f"🔧 {metadata['pair']} Adjusting 0% breadth to 35% (extreme bearish)")  # Changed from ERROR to INFO
            market_breadth = 0.35  # Assume bearish but not extreme
            market_direction = -1
        
        logger.info(f"🎯 {metadata['pair']} Final Breadth: {market_breadth:.1%} (Direction: {market_direction})")  # Changed from ERROR to INFO, improved formatting
        
        # Set dataframe values
        dataframe['market_breadth'] = market_breadth
        dataframe['market_direction'] = market_direction
        dataframe['bullish_weight'] = bullish_weight
        dataframe['bearish_weight'] = bearish_weight
        dataframe['total_weight'] = total_weight
        dataframe['breadth_pairs_checked'] = pairs_checked
        
        return dataframe

    def calculate_market_regime_aligned(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        🔍 Determine the current market regime based on volatility, trend strength, and ADX.
        Uses BTC data as a proxy and aligns with the strategy's timeframe for consistency.
        Handles NumPy array issues by ensuring Pandas Series compatibility.
        """
        lookback = self.regime_lookback_period.value
        current_pair = metadata['pair']
        
        # Detect mode and set BTC pairs to try
        if ':' in current_pair:
            settlement = current_pair.split(':')[1]
            btc_pairs_to_try = [f'BTC/USDT:{settlement}', 'BTC/USDT']
        else:
            btc_pairs_to_try = ['BTC/USDT', 'BTC/USD']
        
        btc_data = None
        btc_pair_used = None
        
        # Attempt to fetch BTC data with appropriate logging
        for btc_pair in btc_pairs_to_try:
            try:
                logger.debug(f"🔍 {current_pair} Trying BTC pair: {btc_pair}")
                btc_data = self.dp.get_pair_dataframe(btc_pair, self.timeframe)
                if btc_data is not None and not btc_data.empty and len(btc_data) >= lookback:
                    btc_pair_used = btc_pair
                    logger.debug(f"✅ {current_pair} Successfully fetched {len(btc_data)} candles from {btc_pair}")
                    break
                else:
                    available_candles = len(btc_data) if btc_data is not None and not btc_data.empty else 0
                    logger.debug(f"⚠️ {current_pair} {btc_pair} insufficient: {available_candles}/{lookback} candles")
            except Exception as e:
                logger.debug(f"💥 {current_pair} Error accessing {btc_pair}: {str(e)}")
                continue
        
        # Fallback to neutral regime if no BTC data
        if btc_data is None or btc_data.empty or len(btc_data) < lookback:
            logger.info(f"🔄 {current_pair} No BTC data available, using neutral regime")
            dataframe['market_regime'] = 'neutral'
            dataframe['market_volatility'] = 0.02
            dataframe['market_adx'] = 25.0
            dataframe['btc_trend_strength'] = 0.0
            return dataframe
        
        try:
            # Calculate market indicators with Pandas Series compatibility
            logger.debug(f"📊 {current_pair} Calculating regime using {btc_pair_used} ({len(btc_data)} candles)")
            
            # Daily volatility based on timeframe (assuming 30m)
            timeframe_minutes = 30
            btc_returns = btc_data['close'].pct_change().to_frame('returns')
            market_volatility = btc_returns['returns'].rolling(window=lookback).std() * np.sqrt(1440 / timeframe_minutes)
            
            # Trend strength using ADX and price movement
            adx_values = ta.ADX(btc_data['high'], btc_data['low'], btc_data['close'], timeperiod=14)
            market_adx = pd.Series(adx_values, index=btc_data.index) if isinstance(adx_values, np.ndarray) else adx_values
            btc_mean = btc_data['close'].rolling(window=lookback).mean()
            btc_std = btc_data['close'].rolling(window=lookback).std()
            btc_trend_strength = ((btc_data['close'].iloc[-1] - btc_mean.iloc[-1]) / btc_std.iloc[-1] 
                                if btc_std.iloc[-1] != 0 else 0.0)
            
            # Ensure we have the latest values
            last_volatility = market_volatility.iloc[-1] if not market_volatility.empty else 0.02
            last_adx = market_adx.iloc[-1] if not market_adx.empty else 25.0
            last_trend_strength = btc_trend_strength if not pd.isna(btc_trend_strength) else 0.0
            
            if last_volatility > 0.05 and last_trend_strength > 1.5 and last_adx > 35:
                regime = 'bull_run'
            elif last_volatility > 0.05 and last_trend_strength < -1.5 and last_adx > 35:
                regime = 'bear_market'
            elif last_volatility < 0.008 and last_adx < 15:
                regime = 'ranging'
            elif last_volatility > 0.08:  # Only extreme volatility
                regime = 'high_volatility'
            else:
                regime = 'neutral'  # Default to neutral instead of high_volatility
            
            # Apply to dataframe
            dataframe['market_regime'] = regime
            dataframe['market_volatility'] = last_volatility
            dataframe['market_adx'] = last_adx
            dataframe['btc_trend_strength'] = last_trend_strength
            
            logger.info(f"🏛️ {current_pair} Market regime: {regime}, Volatility: {last_volatility:.4f}, "
                       f"ADX: {last_adx:.1f}, Trend Strength: {last_trend_strength:.2f}")
            return dataframe
            
        except Exception as e:
            logger.warning(f"💥 {current_pair} Regime calculation failed: {str(e)}, using neutral regime")
            dataframe['market_regime'] = 'neutral'
            dataframe['market_volatility'] = 0.02
            dataframe['market_adx'] = 25.0
            dataframe['btc_trend_strength'] = 0.0
            return dataframe
    def apply_market_regime_filters(self, df: pd.DataFrame, any_long_signal: pd.Series, 
                                   any_short_signal: pd.Series, metadata: dict) -> tuple:
        """
        🧠 INTELLIGENT: Adjust signal sensitivity based on market conditions
        
        This method modifies your entry signals based on the current market regime
        to improve timing and reduce stop losses.
        
        Returns: (enhanced_long_signal, enhanced_short_signal)
        """
        
        try:
            # Get current market regime (use last available value)
            current_regime = df['market_regime'].iloc[-1] if 'market_regime' in df.columns else 'neutral'
            btc_strength = df['btc_strength'].iloc[-1] if 'btc_strength' in df.columns else 0
            fear_greed = df['market_fear_greed'].iloc[-1] if 'market_fear_greed' in df.columns else 50
            
            # ===========================================
            # REGIME-SPECIFIC SIGNAL ADJUSTMENTS
            # ===========================================
            
            if current_regime == 'bull_run':
                # 🚀 BULL RUN MODE: Favor longs, restrict shorts
                long_boost = df['rsi'] < 60  # More aggressive long entries
                short_restrict = df['rsi'] > 80  # Only extreme short entries
                
                enhanced_long_signal = any_long_signal & long_boost
                enhanced_short_signal = any_short_signal & short_restrict
                
                logger.info(f"{metadata['pair']} 🚀 BULL RUN MODE: Boosting longs, restricting shorts")
                
            elif current_regime == 'bear_market':
                # 🐻 BEAR MARKET MODE: Favor shorts, restrict longs
                short_boost = df['rsi'] > 40  # More aggressive short entries  
                long_restrict = df['rsi'] < 20  # Only extreme long entries
                
                enhanced_long_signal = any_long_signal & long_restrict
                enhanced_short_signal = any_short_signal & short_boost
                
                logger.info(f"{metadata['pair']} 🐻 BEAR MARKET MODE: Boosting shorts, restricting longs")
                
            elif current_regime == 'high_volatility':
                # ⚡ HIGH VOLATILITY MODE: Only trade extremes
                extreme_oversold = df['rsi'] < 25
                extreme_overbought = df['rsi'] > 75
                
                enhanced_long_signal = any_long_signal & extreme_oversold
                enhanced_short_signal = any_short_signal & extreme_overbought
                
                logger.info(f"{metadata['pair']} ⚡ HIGH VOLATILITY MODE: Only extreme entries")
                
            elif current_regime == 'sideways':
                # 📊 SIDEWAYS MODE: Range trading, both directions OK
                range_long = (df['rsi'] < 35) & (df['close'] <= df.get('[3/8]P', df['close']))
                range_short = (df['rsi'] > 65) & (df['close'] >= df.get('[5/8]P', df['close']))
                
                enhanced_long_signal = any_long_signal & range_long
                enhanced_short_signal = any_short_signal & range_short
                
                logger.info(f"{metadata['pair']} 📊 SIDEWAYS MODE: Range trading activated")
                
            elif current_regime == 'transitional':
                # 🔄 TRANSITIONAL MODE: Be more selective
                quality_long = (df['rsi'] < 40) & (df['volume'] > df['volume'].rolling(20).mean() * 1.2)
                quality_short = (df['rsi'] > 60) & (df['volume'] > df['volume'].rolling(20).mean() * 1.2)
                
                enhanced_long_signal = any_long_signal & quality_long
                enhanced_short_signal = any_short_signal & quality_short
                
                logger.info(f"{metadata['pair']} 🔄 TRANSITIONAL MODE: Quality-focused entries")
                
            else:  # neutral
                # 😐 NEUTRAL MODE: Standard filtering
                enhanced_long_signal = any_long_signal
                enhanced_short_signal = any_short_signal
                
            # ===========================================
            # FEAR/GREED ADJUSTMENTS
            # ===========================================
            
            if fear_greed > 75:  # Extreme greed
                # Be more careful with longs, favor shorts
                enhanced_long_signal = enhanced_long_signal & (df['rsi'] < 30)
                logger.info(f"{metadata['pair']} 🤑 EXTREME GREED: Restricting longs")
                
            elif fear_greed < 25:  # Extreme fear
                # Be more careful with shorts, favor longs
                enhanced_short_signal = enhanced_short_signal & (df['rsi'] > 70)
                logger.info(f"{metadata['pair']} 😨 EXTREME FEAR: Restricting shorts")
            
            return enhanced_long_signal, enhanced_short_signal
            
        except Exception as e:
            logger.error(f"{metadata['pair']} ❌ Market regime filter error: {e}")
            # Return original signals on error
            return any_long_signal, any_short_signal
    def calculate_total_market_cap_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Calculate total crypto market cap trend
        FIXED: Handles futures format
        """
        # Detect futures mode
        is_futures = ':' in metadata['pair']
        
        if is_futures:
            settlement = metadata['pair'].split(':')[1]
            # Futures weighted pairs
            major_coins = {
                f"BTC/USDT:{settlement}": 0.4,
                f"ETH/USDT:{settlement}": 0.2,
                f"BNB/USDT:{settlement}": 0.1,
                f"SOL/USDT:{settlement}": 0.05,
                f"ADA/USDT:{settlement}": 0.05,
            }
        else:
            # Spot pairs
            major_coins = {
                "BTC/USDT": 0.4,
                "ETH/USDT": 0.2,
                "BNB/USDT": 0.1,
                "SOL/USDT": 0.05,
                "ADA/USDT": 0.05,
            }
        
        weighted_trend = 0
        total_weight = 0
        
        for coin_pair, weight in major_coins.items():
            try:
                coin_data, _ = self.dp.get_analyzed_dataframe(coin_pair, self.timeframe)
                if coin_data.empty or len(coin_data) < self.total_mcap_ma_period.value:
                    # Try spot equivalent if futures not found
                    if is_futures and ':' in coin_pair:
                        spot_pair = coin_pair.split(':')[0]
                        coin_data, _ = self.dp.get_analyzed_dataframe(spot_pair, self.timeframe)
                        if coin_data.empty:
                            continue
                    else:
                        continue
                
                # Calculate trend using MA
                ma_period = self.total_mcap_ma_period.value
                current_price = coin_data['close'].iloc[-1]
                ma_value = coin_data['close'].rolling(ma_period).mean().iloc[-1]
                
                # Trend strength
                trend_strength = (current_price - ma_value) / ma_value
                weighted_trend += trend_strength * weight
                total_weight += weight
                
            except Exception as e:
                logger.debug(f"Could not process {coin_pair} for mcap trend: {e}")
                continue
        
        if total_weight > 0:
            mcap_trend = weighted_trend / total_weight
        else:
            mcap_trend = 0
        
        # Classify trend
        if mcap_trend > 0.05:
            mcap_status = 'bullish'
        elif mcap_trend < -0.05:
            mcap_status = 'bearish'
        else:
            mcap_status = 'neutral'
        
        dataframe['mcap_trend'] = mcap_trend
        dataframe['mcap_status'] = mcap_status
        
        return dataframe
    
    def apply_correlation_filters(self, dataframe: pd.DataFrame, direction: str = 'long') -> pd.Series:
        """
        Apply market correlation filters using ENHANCED correlation data
        Returns a boolean Series indicating whether market conditions are favorable
        """
        conditions = pd.Series(True, index=dataframe.index)
        
        # ===========================================
        # ENHANCED BTC CORRELATION FILTER
        # ===========================================
        if self.enable_btc_correlation.value and 'btc_correlation_ok' in dataframe.columns:
            # Simple approach: Only trade when BTC conditions are favorable
            conditions &= dataframe['btc_correlation_ok']
            
            # Optional: More restrictive based on BTC trend score
            if 'btc_trend_score' in dataframe.columns:
                if direction == 'long':
                    # For longs: Allow when BTC trend score >= 2 (at least neutral)
                    # OR when extremely oversold (RSI < 25)
                    btc_long_ok = (
                        (dataframe['btc_trend_score'] >= 2) |
                        (dataframe.get('rsi', 50) < 25)  # Emergency oversold exception
                    )
                    conditions &= btc_long_ok
                    
                else:  # short
                    # For shorts: Allow when BTC trend score <= 3 (not too bullish)
                    # OR when extremely overbought (RSI > 75)
                    btc_short_ok = (
                        (dataframe['btc_trend_score'] <= 3) |
                        (dataframe.get('rsi', 50) > 75)  # Emergency overbought exception
                    )
                    conditions &= btc_short_ok
        
        # ===========================================
        # MARKET BREADTH FILTER (Keep as is)
        # ===========================================
        if self.market_breadth_enabled.value and 'market_breadth' in dataframe.columns:
            if direction == 'long':
                # Long only when majority of market is bullish
                conditions &= (dataframe['market_breadth'] > self.market_breadth_threshold.value)
            else:  # short
                # Short only when majority of market is bearish
                conditions &= (dataframe['market_breadth'] < (1 - self.market_breadth_threshold.value))
        
        # ===========================================
        # MARKET CAP TREND FILTER (Keep as is)
        # ===========================================
        if self.total_mcap_filter_enabled.value and 'mcap_status' in dataframe.columns:
            if direction == 'long':
                conditions &= (dataframe['mcap_status'] != 'bearish')
            else:  # short
                conditions &= (dataframe['mcap_status'] != 'bullish')
        
        # ===========================================
        # MARKET REGIME FILTER (Keep as is)
        # ===========================================
        if self.regime_filter_enabled.value and 'market_regime' in dataframe.columns:
            # Avoid trading in high volatility regimes
            conditions &= (dataframe['market_regime'] != 'high_volatility')
        
        return conditions
    def calculate_enhanced_trend_filters(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            """
            Enhanced trend detection to avoid choppy market entries
            """
            
            # Multi-timeframe trend alignment
            dataframe['ema_5'] = ta.EMA(dataframe['close'], timeperiod=5)
            dataframe['ema_13'] = ta.EMA(dataframe['close'], timeperiod=13)
            dataframe['ema_21'] = ta.EMA(dataframe['close'], timeperiod=21)
            dataframe['ema_50'] = ta.EMA(dataframe['close'], timeperiod=50)
            
            # Trend alignment score
            dataframe['trend_alignment_bull'] = (
                (dataframe['close'] > dataframe['ema_5']) &
                (dataframe['ema_5'] > dataframe['ema_13']) &
                (dataframe['ema_13'] > dataframe['ema_21']) &
                (dataframe['ema_21'] > dataframe['ema_50'])
            ).astype(int)
            
            dataframe['trend_alignment_bear'] = (
                (dataframe['close'] < dataframe['ema_5']) &
                (dataframe['ema_5'] < dataframe['ema_13']) &
                (dataframe['ema_13'] < dataframe['ema_21']) &
                (dataframe['ema_21'] < dataframe['ema_50'])
            ).astype(int)
            
            # ADX for trend strength
            dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
            dataframe['strong_trend'] = dataframe['adx'] > 25
            dataframe['very_strong_trend'] = dataframe['adx'] > 40
            
            # Momentum indicators
            dataframe['macd'], dataframe['macd_signal'], dataframe['macd_hist'] = ta.MACD(
                dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9
            )
            
            # MACD momentum alignment
            dataframe['macd_bull'] = (
                (dataframe['macd'] > dataframe['macd_signal']) &
                (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1))
            )
            
            dataframe['macd_bear'] = (
                (dataframe['macd'] < dataframe['macd_signal']) &
                (dataframe['macd_hist'] < dataframe['macd_hist'].shift(1))
            )
            
            # Price action patterns
            dataframe['higher_highs'] = (
                (dataframe['high'] > dataframe['high'].shift(1)) &
                (dataframe['high'].shift(1) > dataframe['high'].shift(2))
            )
            
            dataframe['lower_lows'] = (
                (dataframe['low'] < dataframe['low'].shift(1)) &
                (dataframe['low'].shift(1) < dataframe['low'].shift(2))
            )
            
            # Higher lows (bullish structure)
            dataframe['higher_lows'] = (
                (dataframe['low'] > dataframe['low'].shift(1)) &
                (dataframe['low'].shift(1) > dataframe['low'].shift(2))
            )
            
            # Lower highs (bearish structure)
            dataframe['lower_highs'] = (
                (dataframe['high'] < dataframe['high'].shift(1)) &
                (dataframe['high'].shift(1) < dataframe['high'].shift(2))
            )
            
            # Volatility filter using Bollinger Bands
            dataframe['bb_upper'], dataframe['bb_middle'], dataframe['bb_lower'] = ta.BBANDS(
                dataframe['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0
            )
            
            # BB squeeze detection (low volatility)
            dataframe['bb_squeeze'] = (
                (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle'] < 0.1
            )
            
            # BB expansion (high volatility breakout)
            dataframe['bb_expansion'] = (
                (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle'] > 0.2
            )
            
            # Market regime classification
            dataframe['trending_up'] = (
                dataframe['trend_alignment_bull'] &
                dataframe['strong_trend'] &
                dataframe['macd_bull'] &
                (~dataframe['bb_squeeze'])
            )
            
            dataframe['trending_down'] = (
                dataframe['trend_alignment_bear'] &
                dataframe['strong_trend'] &
                dataframe['macd_bear'] &
                (~dataframe['bb_squeeze'])
            )
            
            dataframe['choppy_market'] = (
                (dataframe['adx'] < 20) |
                dataframe['bb_squeeze'] |
                (
                    (~dataframe['trend_alignment_bull']) & 
                    (~dataframe['trend_alignment_bear'])
                )
            )
            
            # Market structure breaks
            dataframe['structure_break_bull'] = (
                (dataframe['close'] > dataframe['high'].rolling(20).max().shift(1)) &
                dataframe['trending_up'] &
                (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.2)
            )
            
            dataframe['structure_break_bear'] = (
                (dataframe['close'] < dataframe['low'].rolling(20).min().shift(1)) &
                dataframe['trending_down'] &
                (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.2)
            )
            
            # Trend momentum strength
            dataframe['trend_momentum'] = dataframe['close'].pct_change(10)
            dataframe['trend_momentum_strength'] = abs(dataframe['trend_momentum'])
            
            # Consistent trend detection (multiple timeframes)
            dataframe['short_term_trend'] = np.where(
                dataframe['close'] > dataframe['ema_5'], 1,
                np.where(dataframe['close'] < dataframe['ema_5'], -1, 0)
            )
            
            dataframe['medium_term_trend'] = np.where(
                dataframe['close'] > dataframe['ema_21'], 1,
                np.where(dataframe['close'] < dataframe['ema_21'], -1, 0)
            )
            
            dataframe['long_term_trend'] = np.where(
                dataframe['close'] > dataframe['ema_50'], 1,
                np.where(dataframe['close'] < dataframe['ema_50'], -1, 0)
            )
            
            # Trend consistency score
            dataframe['trend_consistency'] = (
                dataframe['short_term_trend'] + 
                dataframe['medium_term_trend'] + 
                dataframe['long_term_trend']
            ) / 3
            
            # Strong consistent trends
            dataframe['strong_consistent_uptrend'] = dataframe['trend_consistency'] > 0.6
            dataframe['strong_consistent_downtrend'] = dataframe['trend_consistency'] < -0.6
            
            # Trend change detection
            dataframe['trend_change_up'] = (
                (dataframe['trend_consistency'] > 0) &
                (dataframe['trend_consistency'].shift(1) <= 0)
            )
            
            dataframe['trend_change_down'] = (
                (dataframe['trend_consistency'] < 0) &
                (dataframe['trend_consistency'].shift(1) >= 0)
            )
            
            # Support and resistance levels based on EMAs
            dataframe['dynamic_resistance'] = np.maximum.reduce([
                dataframe['ema_5'], dataframe['ema_13'], 
                dataframe['ema_21'], dataframe['ema_50']
            ])
            
            dataframe['dynamic_support'] = np.minimum.reduce([
                dataframe['ema_5'], dataframe['ema_13'], 
                dataframe['ema_21'], dataframe['ema_50']
            ])
            
            # Price position relative to dynamic levels
            dataframe['above_all_emas'] = (
                (dataframe['close'] > dataframe['ema_5']) &
                (dataframe['close'] > dataframe['ema_13']) &
                (dataframe['close'] > dataframe['ema_21']) &
                (dataframe['close'] > dataframe['ema_50'])
            )
            
            dataframe['below_all_emas'] = (
                (dataframe['close'] < dataframe['ema_5']) &
                (dataframe['close'] < dataframe['ema_13']) &
                (dataframe['close'] < dataframe['ema_21']) &
                (dataframe['close'] < dataframe['ema_50'])
            )
            
            # Momentum divergence detection
            # Price vs RSI divergence
            price_peaks = dataframe['high'].rolling(10).max()
            price_troughs = dataframe['low'].rolling(10).min()
            
            dataframe['bullish_divergence'] = (
                (dataframe['low'] == price_troughs) &
                (dataframe['low'] < dataframe['low'].shift(10)) &
                (dataframe['rsi'] > dataframe['rsi'].shift(10))
            )
            
            dataframe['bearish_divergence'] = (
                (dataframe['high'] == price_peaks) &
                (dataframe['high'] > dataframe['high'].shift(10)) &
                (dataframe['rsi'] < dataframe['rsi'].shift(10))
            )
            
            return dataframe
    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend strength to avoid entering against strong trends
        """
        # Linear regression slope
        def calc_slope(series, period=10):
            """Calculate linear regression slope"""
            if len(series) < period:
                return 0
            x = np.arange(period)
            y = series.iloc[-period:].values
            if np.isnan(y).any():
                return 0
            slope = np.polyfit(x, y, 1)[0]
            return slope
        
        # Calculate trend strength using multiple timeframes
        df['slope_5'] = df['close'].rolling(5).apply(lambda x: calc_slope(x, 5), raw=False)
        df['slope_10'] = df['close'].rolling(10).apply(lambda x: calc_slope(x, 10), raw=False)
        df['slope_20'] = df['close'].rolling(20).apply(lambda x: calc_slope(x, 20), raw=False)
        
        # Normalize slopes by price
        df['trend_strength_5'] = df['slope_5'] / df['close'] * 100
        df['trend_strength_10'] = df['slope_10'] / df['close'] * 100
        df['trend_strength_20'] = df['slope_20'] / df['close'] * 100
        
        # Combined trend strength
        df['trend_strength'] = (df['trend_strength_5'] + df['trend_strength_10'] + df['trend_strength_20']) / 3
        
        # Trend classification
        strong_up_threshold = self.trend_strength_threshold.value
        strong_down_threshold = -self.trend_strength_threshold.value
        
        df['strong_uptrend'] = df['trend_strength'] > strong_up_threshold
        df['strong_downtrend'] = df['trend_strength'] < strong_down_threshold
        df['ranging'] = (df['trend_strength'].abs() < strong_up_threshold * 0.5)
        
        return df
    @property
    def protections(self):
        prot = [{"method": "CooldownPeriod", "stop_duration_candles": self.cooldown_lookback.value}]
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 72,
                "trade_limit": 2,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False,
            })
        return prot


    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        
        # ===========================================
        # STAKE LIMITS DEFINIEREN
        # ===========================================
        
        # Maximaler Stake pro Trade (in USDT) - kannst du anpassen
        MAX_STAKE_PER_TRADE = self.max_stake_per_trade
        
        # Maximaler Stake basierend auf Portfolio
        try:
            total_portfolio = self.wallets.get_total_stake_amount()
            MAX_STAKE_PERCENTAGE = self.max_portfolio_percentage_per_trade
            max_stake_from_portfolio = total_portfolio * MAX_STAKE_PERCENTAGE
        except:
            # Fallback wenn wallets nicht verfügbar
            max_stake_from_portfolio = MAX_STAKE_PER_TRADE
            total_portfolio = 1000.0  # Dummy value
        
        # Market condition check für volatility-based stake reduction (DEIN CODE)
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if not dataframe.empty:
            last_candle = dataframe.iloc[-1]
            current_volatility = last_candle.get("volatility", 0.02)
            
            # Reduziere Stake in hochvolatilen Märkten
            if current_volatility > 0.05:  # 5% ATR/Price ratio
                volatility_reduction = min(0.5, current_volatility * 10)  # Max 50% reduction
                proposed_stake *= (1 - volatility_reduction)
                logger.info(f"{pair} Stake reduced by {volatility_reduction:.1%} due to high volatility ({current_volatility:.2%})")
        
        # DCA Multiplier Berechnung (DEIN CODE)
        calculated_max_dca_multiplier = 1.0
        if self.position_adjustment_enable:
            num_safety_orders = int(self.max_safety_orders.value)
            volume_scale = self.safety_order_volume_scale.value
            if num_safety_orders > 0 and volume_scale > 0:
                current_order_relative_size = 1.0
                for _ in range(num_safety_orders):
                    current_order_relative_size *= volume_scale
                    calculated_max_dca_multiplier += current_order_relative_size
            else:
                logger.warning(f"{pair}: Could not calculate max_dca_multiplier due to "
                               f"invalid max_safety_orders ({num_safety_orders}) or "
                               f"safety_order_volume_scale ({volume_scale}). Defaulting to 1.0.")
        else:
            logger.debug(f"{pair}: Position adjustment not enabled. max_dca_multiplier is 1.0.")

        if calculated_max_dca_multiplier > 0:
            stake_amount = proposed_stake / calculated_max_dca_multiplier
            
            # ===========================================
            # NEUE STAKE LIMITS ANWENDEN
            # ===========================================
            
            # Verschiedene Limits prüfen
            final_stake = min(
                stake_amount,
                MAX_STAKE_PER_TRADE,
                max_stake_from_portfolio,
                max_stake  # Freqtrade's max_stake
            )
            
            # Bestimme welches Limit gegriffen hat
            limit_reason = "calculated"
            if final_stake == MAX_STAKE_PER_TRADE:
                limit_reason = "max_per_trade"
            elif final_stake == max_stake_from_portfolio:
                limit_reason = "portfolio_percentage"
            elif final_stake == max_stake:
                limit_reason = "freqtrade_max"
            
            logger.info(f"{pair} Initial stake calculated: {final_stake:.8f} (Proposed: {proposed_stake:.8f}, "
                        f"Calculated Max DCA Multiplier: {calculated_max_dca_multiplier:.2f}, "
                        f"Limited by: {limit_reason}, Portfolio %: {(final_stake/total_portfolio)*100:.1f}%)")
            
            # Min stake prüfen (DEIN CODE)
            if min_stake is not None and final_stake < min_stake:
                logger.info(f"{pair} Initial stake {final_stake:.8f} was below min_stake {min_stake:.8f}. "
                            f"Adjusting to min_stake. Consider tuning your DCA parameters or proposed stake.")
                final_stake = min_stake
            
            return final_stake
        else:
            # Fallback (DEIN CODE)
            logger.warning(
                f"{pair} Calculated max_dca_multiplier is {calculated_max_dca_multiplier:.2f}, which is invalid. "
                f"Using proposed_stake: {proposed_stake:.8f}")
            return proposed_stake

    def custom_entry_price(self, pair: str, trade: Optional[Trade], current_time: datetime,
                           proposed_rate: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        🎯 IMPROVED: Wait for better entry prices to reduce immediate stop loss risk
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe.empty:
            return proposed_rate
        
        last_candle = dataframe.iloc[-1]
        
        # Calculate volatility-adjusted entry
        atr = last_candle.get('atr', 0)
        if atr == 0:
            atr = (last_candle['high'] - last_candle['low'])
        
        # Get MML levels for better entry timing
        mml_25 = last_candle.get('[2/8]P', last_candle['close'])
        mml_50 = last_candle.get('[4/8]P', last_candle['close'])
        mml_75 = last_candle.get('[6/8]P', last_candle['close'])
        
        if side == "long":
            # For longs: Try to enter closer to support or on pullbacks
            if entry_tag in ["MML_50_Reclaim", "MML_Support_Bounce"]:
                # Enter slightly above support with buffer
                support_buffer = atr * 0.2  # 20% of ATR buffer
                optimal_entry = max(mml_25, mml_50) + support_buffer
                entry_price = min(proposed_rate, optimal_entry)
            elif entry_tag == "MML_Bullish_Breakout":
                # For breakouts: Enter on slight pullback, not at peak
                pullback_entry = proposed_rate - (atr * 0.1)  # 10% ATR pullback
                entry_price = max(pullback_entry, proposed_rate * 0.999)  # Max 0.1% below
            else:
                # Conservative entry for other signals
                entry_price = proposed_rate - (atr * 0.05)  # Small discount
                
        elif side == "short":
            # For shorts: Try to enter closer to resistance
            if entry_tag in ["MML_50_Breakdown", "MML_Resistance_Reject"]:
                # Enter slightly below resistance with buffer
                resistance_buffer = atr * 0.2
                optimal_entry = min(mml_75, mml_50) - resistance_buffer
                entry_price = max(proposed_rate, optimal_entry)
            elif entry_tag == "MML_Bearish_Breakdown":
                # For breakdowns: Enter on slight bounce, not at bottom
                bounce_entry = proposed_rate + (atr * 0.1)
                entry_price = min(bounce_entry, proposed_rate * 1.001)
            else:
                entry_price = proposed_rate + (atr * 0.05)
        else:
            entry_price = proposed_rate
        
        # Ensure entry price is reasonable
        max_deviation = proposed_rate * 0.005  # Max 0.5% from proposed
        if side == "long":
            entry_price = max(entry_price, proposed_rate - max_deviation)
        else:
            entry_price = min(entry_price, proposed_rate + max_deviation)
        
        logger.info(f"{pair} Optimized entry: {entry_price:.6f} (was {proposed_rate:.6f}, tag: {entry_tag})")
        return entry_price

    # ═══════════════════════════════════════════════════════════════
    # 🛑 CRITICAL FIX: TIME-BASED STOP LOSS LOOSENING
    # ═══════════════════════════════════════════════════════════════

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        🔧 REVOLUTIONARY: Time-based stop loss loosening + MML support
        
        Based on your backtest results:
        - ROI hits at ~1.5 hours with 4% profit
        - Stop loss hits at ~1.25 hours with -6.5% loss
        - Need to give trades more time to develop before stopping out
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe.empty or 'atr' not in dataframe.columns or len(dataframe) < 10:
            logger.warning(f"{pair} Insufficient data for dynamic stop loss. Using conservative default: -0.03")
            return -0.03  # Conservative default
        
        last_candle = dataframe.iloc[-1]
        last_atr = last_candle.get("atr", 0)
        
        # Handle missing or invalid ATR
        if pd.isna(last_atr) or last_atr == 0:
            valid_atr = dataframe["atr"].dropna()
            if not valid_atr.empty:
                last_atr = valid_atr.rolling(window=5).mean().iloc[-1]
                logger.info(f"{pair} Using smoothed ATR: {last_atr:.8f}")
            else:
                logger.warning(f"{pair} No valid ATR found. Using fallback stop loss -0.03")
                return -0.03

        if last_atr == 0 or current_rate == 0:
            logger.warning(f"{pair} ATR or rate is 0. Using fallback stop loss -0.03")
            return -0.03
        
        # ⏰ TIME IN TRADE CALCULATION
        time_in_trade = (current_time - trade.open_date_utc).total_seconds() / 3600  # Hours
        
        # Get MML levels for dynamic support/resistance
        if trade.is_short:
            # For shorts: Use MML resistance as dynamic stop
            resistance_level = last_candle.get('[6/8]P', current_rate * 1.02)
            if pd.isna(resistance_level):
                resistance_level = current_rate * 1.02
            mml_stop_distance = abs((resistance_level - current_rate) / current_rate)
        else:
            # For longs: Use MML support as dynamic stop  
            support_level = last_candle.get('[2/8]P', current_rate * 0.98)
            if pd.isna(support_level):
                support_level = current_rate * 0.98
            mml_stop_distance = abs((current_rate - support_level) / current_rate)
        
        # Base ATR stop loss calculation
        base_atr_stop = (last_atr / current_rate) * 2.0  # Start with 2x ATR
        
        # 🕐 CRITICAL: TIME-BASED STOP LOSS LOOSENING
        # Based on your data: Need more room in first 1.5 hours
        if time_in_trade < 0.25:  # First 15 minutes - tightest (but not too tight)
            time_multiplier = 1.2  # Give some room immediately
        elif time_in_trade < 0.5:  # 15-30 minutes - slight loosening
            time_multiplier = 1.5
        elif time_in_trade < 1.0:  # 30-60 minutes - more room
            time_multiplier = 2.0
        elif time_in_trade < 1.5:  # 1-1.5 hours - CRITICAL PERIOD
            time_multiplier = 2.5  # Much more room where stop losses currently hit
        elif time_in_trade < 2.0:  # 1.5-2 hours - ROI development time
            time_multiplier = 3.0  # Maximum room for ROI to develop
        elif time_in_trade < 4.0:  # 2-4 hours - long-term holds
            time_multiplier = 2.8  # Slight tightening but still wide
        else:  # 4+ hours - prevent runaway losses
            time_multiplier = 2.5
        
        # 📈 PROFIT-BASED ADJUSTMENTS (override time loosening when profitable)
        if current_profit > 0.08:  # 8%+ profit - lock in gains aggressively
            profit_stop = max(-0.015, -(base_atr_stop * 0.3))  # Very tight
        elif current_profit > 0.06:  # 6%+ profit - good protection
            profit_stop = max(-0.02, -(base_atr_stop * 0.5))
        elif current_profit > 0.04:  # 4%+ profit - moderate protection
            profit_stop = max(-0.025, -(base_atr_stop * 0.7))
        elif current_profit > 0.02:  # 2%+ profit - light protection
            profit_stop = max(-0.03, -(base_atr_stop * 1.0))
        elif current_profit > 0:  # Any profit - don't tighten
            profit_stop = -(base_atr_stop * time_multiplier)
        else:
            # Apply time-based loosening for losing trades
            profit_stop = -(base_atr_stop * time_multiplier)
        
        # 🎯 MML-BASED STOP LOSS ADJUSTMENT
        # Use MML levels as natural stop areas, but don't make stops too wide
        mml_adjusted_stop = min(
            abs(profit_stop),  # Don't make tighter than profit-based
            max(mml_stop_distance * 1.3, 0.02),  # At least 2%, max 30% past MML
            0.10  # Never wider than 10%
        )
        
        # 🛡️ SAFETY BOUNDARIES
        # Ensure stop loss isn't too tight or too wide
        if current_profit <= 0:
            # For losing trades: Ensure minimum room based on time
            if time_in_trade < 1.5:  # Critical period - be generous
                min_stop = -0.08  # Minimum 8% room
            else:
                min_stop = -0.06  # Standard minimum
            
            final_stop = -max(mml_adjusted_stop, abs(min_stop))
        else:
            # For profitable trades: Use calculated stop
            final_stop = -mml_adjusted_stop
        
        # Final safety check: Never tighter than -1.5% unless big profit
        if final_stop > -0.015 and current_profit < 0.06:
            final_stop = -0.015
        
        # Never wider than -12%
        final_stop = max(final_stop, -0.12)
        
        # 📊 DETAILED LOGGING
        logger.info(f"{pair} 🛑 DYNAMIC STOP LOSS:")
        logger.info(f"   ⏰ Time in trade: {time_in_trade:.2f}h")
        logger.info(f"   📈 Current profit: {current_profit:.2%}")
        logger.info(f"   🔢 Time multiplier: {time_multiplier}x")
        logger.info(f"   💰 ATR base: {base_atr_stop:.4f}")
        logger.info(f"   🎯 MML distance: {mml_stop_distance:.4f}")
        logger.info(f"   🛡️ Final stop: {final_stop:.4f}")
        
        return final_stop
        
    def minimal_roi_market_adjusted(self, pair: str, trade: Trade, current_time: datetime, 
                                   current_rate: float, current_profit: float) -> Dict[int, float]:
        """
        Dynamically adjust ROI based on market conditions
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.minimal_roi
        
        last_candle = dataframe.iloc[-1]
        market_score = last_candle.get('market_score', 0.5)
        market_regime = last_candle.get('market_regime', 'normal')
        
        # Copy original ROI
        adjusted_roi = self.minimal_roi.copy()
        
        # In high volatility or bear market, take profits earlier
        if market_regime == 'high_volatility' or market_score < 0.3:
            # Reduce all ROI targets by 20%
            adjusted_roi = {k: v * 0.8 for k, v in adjusted_roi.items()}
            logger.info(f"{pair} ROI adjusted down due to market conditions")
        
        # In strong bull market, let winners run
        elif market_score > 0.7 and market_regime == 'strong_trend':
            # Increase ROI targets by 20%
            adjusted_roi = {k: v * 1.2 for k, v in adjusted_roi.items()}
            logger.info(f"{pair} ROI adjusted up due to bullish market")
        
        return adjusted_roi
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                              current_profit: float, min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float, current_entry_profit: float,
                              current_exit_profit: float, **kwargs) -> Optional[float]:
        
        # Get market conditions
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if not dataframe.empty:
            last_candle = dataframe.iloc[-1]
            btc_trend = last_candle.get('btc_trend', 0)
            market_score = last_candle.get('market_score', 0.5)
            market_breadth = last_candle.get('market_breadth', 0.5)
            rsi = last_candle.get('rsi', 50)
        else:
            market_score = 0.5
            market_breadth = 0.5
            rsi = 50
        
        count_of_entries = trade.nr_of_successful_entries
        count_of_exits = trade.nr_of_successful_exits
        
        # === ENHANCED PROFIT TAKING BASED ON MARKET CONDITIONS ===
        
        # More aggressive profit taking in overbought market
        if market_score > 0.75 and current_profit > 0.15 and count_of_exits == 0:
            logger.info(f"{trade.pair} Taking profit early due to market greed: {market_score:.2f}")
            amount_to_sell = (trade.amount * current_rate) * 0.33  # Sell 33%
            return -amount_to_sell
        
        # Original profit taking logic (keep as is)
        if current_profit > 0.25 and count_of_exits == 0:
            logger.info(f"{trade.pair} Taking partial profit (25%) at {current_profit:.2%}")
            amount_to_sell = (trade.amount * current_rate) * 0.25
            return -amount_to_sell

        # === BEAR MARKET PROFIT TAKING ===
        # Nimm Gewinne schneller mit im Bärenmarkt
        if not trade.is_short and btc_trend < 0:  # Long im Downtrend
            if current_profit > 0.10 and count_of_exits == 0:  # Statt 0.25
                logger.info(f"{trade.pair} Bear market quick profit taking at {current_profit:.2%}")
                amount_to_sell = (trade.amount * current_rate) * 0.5  # 50% verkaufen
                return -amount_to_sell

        if current_profit > 0.40 and count_of_exits == 1:
            logger.info(f"{trade.pair} Taking additional profit (33%) at {current_profit:.2%}")
            amount_to_sell = (trade.amount * current_rate) * (1 / 3)
            return -amount_to_sell
        
        # === 🔧 ENHANCED DCA LOGIC WITH STRICT CONTROLS ===
        if not self.position_adjustment_enable:
            return None
        
        # 🛑 USE STRATEGY VARIABLES FOR DCA LIMITS
        max_dca_for_pair = self.max_dca_orders
        max_total_stake = self.max_total_stake_per_pair
        max_single_dca = self.max_single_dca_amount
        
        # 🛑 CHECK: Already too many DCA orders?
        if count_of_entries > max_dca_for_pair:
            logger.info(f"{trade.pair} 🛑 MAX DCA REACHED: {count_of_entries}/{max_dca_for_pair}")
            return None
        
        # 🛑 CHECK: Total stake amount already too high?
        if trade.stake_amount >= max_total_stake:
            logger.info(f"{trade.pair} 🛑 MAX STAKE REACHED: {trade.stake_amount:.2f}/{max_total_stake} USDT")
            return None
        
        # Block DCA in crashing market
        if market_breadth < 0.25 and trade.is_short == False:
            logger.info(f"{trade.pair} Blocking DCA due to bearish market breadth: {market_breadth:.2%}")
            return None
        
        # More conservative DCA triggers in volatile markets
        if dataframe.iloc[-1].get('market_regime') == 'high_volatility':
            dca_multiplier = 1.5  # Require 50% deeper drawdown
        else:
            dca_multiplier = 1.0
        
        # Apply multiplier to original DCA logic
        trigger = self.initial_safety_order_trigger.value * dca_multiplier
        
        if (current_profit > trigger / 2.0 and count_of_entries == 1) or \
           (current_profit > trigger and count_of_entries == 2) or \
           (current_profit > trigger * 1.5 and count_of_entries == 3):
            logger.info(f"{trade.pair} DCA condition not met. Current profit {current_profit:.2%} above threshold")
            return None
        
        # 🛑 CHECK: Original max safety orders (falls du das noch nutzt)
        if hasattr(self, 'max_safety_orders') and count_of_entries >= self.max_safety_orders.value + 1:
            logger.info(f"{trade.pair} 🛑 Original max_safety_orders reached: {count_of_entries}")
            return None

        try:
            filled_entry_orders = trade.select_filled_orders(trade.entry_side)
            if not filled_entry_orders:
                logger.error(f"{trade.pair} No filled entry orders found for DCA calculation")
                return None
            
            last_order_cost = filled_entry_orders[-1].cost
            
            # 🔧 USE STRATEGY VARIABLE FOR DCA SIZING
            base_dca_amount = max_single_dca  # Use strategy variable
            
            # Progressive DCA sizing (each DCA gets smaller!)
            dca_multipliers = [1.0, 0.8, 0.6]  # 1st: 5 USDT, 2nd: 4 USDT, 3rd: 3 USDT
            
            if count_of_entries <= len(dca_multipliers):
                current_multiplier = dca_multipliers[count_of_entries - 1]
            else:
                current_multiplier = 0.5  # Fallback für unerwartete Orders
            
            # Calculate DCA amount
            dca_stake_amount = base_dca_amount * current_multiplier
            
            # 🛑 HARD CAP: Never exceed remaining budget
            remaining_budget = max_total_stake - trade.stake_amount
            if dca_stake_amount > remaining_budget:
                if remaining_budget > 1:  # Only proceed if at least 1 USDT remaining
                    dca_stake_amount = remaining_budget
                    logger.info(f"{trade.pair} 🔧 DCA capped to remaining budget: {dca_stake_amount:.2f} USDT")
                else:
                    logger.info(f"{trade.pair} 🛑 Insufficient remaining budget: {remaining_budget:.2f} USDT")
                    return None
            
            # Standard min/max stake checks
            if min_stake is not None and dca_stake_amount < min_stake:
                logger.warning(f"{trade.pair} DCA below min_stake. Adjusting to {min_stake:.2f} USDT")
                dca_stake_amount = min_stake
            
            if max_stake is not None and (trade.stake_amount + dca_stake_amount) > max_stake:
                available_for_dca = max_stake - trade.stake_amount
                if available_for_dca > (min_stake or 0):
                    dca_stake_amount = available_for_dca
                    logger.warning(f"{trade.pair} DCA reduced due to max_stake: {dca_stake_amount:.2f} USDT")
                else:
                    logger.warning(f"{trade.pair} Cannot DCA due to max_stake limit")
                    return None
            
            # 🔧 FINAL SAFETY CHECK
            new_total_stake = trade.stake_amount + dca_stake_amount
            if new_total_stake > max_total_stake:
                logger.error(f"{trade.pair} 🚨 SAFETY VIOLATION: Would exceed max total stake!")
                return None
            
            logger.info(f"{trade.pair} ✅ DCA #{count_of_entries}: +{dca_stake_amount:.2f} USDT "
                       f"(Total: {new_total_stake:.2f}/{max_total_stake} USDT)")
            
            return dca_stake_amount
            
        except IndexError:
            logger.error(f"Error calculating DCA stake for {trade.pair}: IndexError accessing last_order")
            return None
        except Exception as e:
            logger.error(f"Error calculating DCA stake for {trade.pair}: {e}")
            return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float,
                 max_leverage: float, side: str, **kwargs) -> float:
        window_size = self.leverage_window_size.value
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if len(dataframe) < window_size:
            logger.warning(
                f"{pair} Not enough data ({len(dataframe)} candles) to calculate dynamic leverage (requires {window_size}). Using proposed: {proposed_leverage}")
            return proposed_leverage
        close_prices_series = dataframe["close"].tail(window_size)
        high_prices_series = dataframe["high"].tail(window_size)
        low_prices_series = dataframe["low"].tail(window_size)
        base_leverage = self.leverage_base.value
        rsi_array = ta.RSI(close_prices_series, timeperiod=14)
        atr_array = ta.ATR(high_prices_series, low_prices_series, close_prices_series, timeperiod=14)
        sma_array = ta.SMA(close_prices_series, timeperiod=20)
        macd_output = ta.MACD(close_prices_series, fastperiod=12, slowperiod=26, signalperiod=9)

        current_rsi = rsi_array[-1] if rsi_array.size > 0 and not np.isnan(rsi_array[-1]) else 50.0
        current_atr = atr_array[-1] if atr_array.size > 0 and not np.isnan(atr_array[-1]) else 0.0
        current_sma = sma_array[-1] if sma_array.size > 0 and not np.isnan(sma_array[-1]) else current_rate
        current_macd_hist = 0.0

        if isinstance(macd_output, pd.DataFrame):
            if not macd_output.empty and 'macdhist' in macd_output.columns:
                valid_macdhist_series = macd_output['macdhist'].dropna()
                if not valid_macdhist_series.empty:
                    current_macd_hist = valid_macdhist_series.iloc[-1]

        # Apply rules based on indicators
        if side == "long":
            if current_rsi < self.leverage_rsi_low.value:
                base_leverage *= self.leverage_long_increase_factor.value
            elif current_rsi > self.leverage_rsi_high.value:
                base_leverage *= self.leverage_long_decrease_factor.value

            if current_atr > 0 and current_rate > 0:
                if (current_atr / current_rate) > self.leverage_atr_threshold_pct.value:
                    base_leverage *= self.leverage_volatility_decrease_factor.value

            if current_macd_hist > 0:
                base_leverage *= self.leverage_long_increase_factor.value

            if current_sma > 0 and current_rate < current_sma:
                base_leverage *= self.leverage_long_decrease_factor.value

        adjusted_leverage = round(max(1.0, min(base_leverage, max_leverage)), 2)
        logger.info(
            f"{pair} Dynamic Leverage: {adjusted_leverage:.2f} (Base: {base_leverage:.2f}, RSI: {current_rsi:.2f}, "
            f"ATR: {current_atr:.4f}, MACD Hist: {current_macd_hist:.4f}, SMA: {current_sma:.4f})")
        return adjusted_leverage

    def calculate_hurst_exponent(self, series: pd.Series, lookback: int = 100) -> pd.Series:
        if len(series) < lookback:
            logger.warning(f"Insufficient data for Hurst calculation: {len(series)} candles, need {lookback}")
            return pd.Series(np.nan, index=series.index)
        try:
            from hurst import compute_Hc
            hurst = []
            for i in range(len(series)):
                if i < lookback:
                    hurst.append(np.nan)
                else:
                    H, _, _ = compute_Hc(series[i - lookback:i], kind='price', simplified=True)
                    hurst.append(H)
            hurst_series = pd.Series(hurst, index=series.index)
            hurst_smooth = hurst_series.ewm(span=10, adjust=False).mean()
            logger.info(f"Hurst Range: min={hurst_smooth.min():.3f}, max={hurst_smooth.max():.3f}, mean={hurst_smooth.mean():.3f}")
            return hurst_smooth
        except Exception as e:
            logger.warning(f"Error calculating Hurst Exponent: {e}")
            return pd.Series(np.nan, index=series.index)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        🔧 Enhanced indicator calculations with Hurst Exponent-based signals for longs and shorts
        """
        dataframe['zero'] = 0
        # Initialize columns
        dataframe['actual_stoploss'] = np.nan
        dataframe['follow_price_level'] = np.nan

        # ===========================================
        # CORE TECHNICAL INDICATORS
        # ===========================================
        # ATR
        dataframe["atr"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=7)
        if dataframe["atr"].isna().all():
            logger.warning(f"No valid ATR calculated for {metadata['pair']}. Filling with 0 temporarily.")
            dataframe["atr"] = 0.0
        else:
            dataframe["atr"] = dataframe["atr"].fillna(dataframe["atr"].rolling(window=5, min_periods=1).mean())
        logger.info(f"ATR values for {metadata['pair']}: {', '.join(dataframe['atr'].tail(5).map('{:.6f}'.format))}")

        # EMAs
        dataframe["ema20"] = ta.EMA(dataframe["close"], timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe["close"], timeperiod=50)
        dataframe['baseline'] = ta.SMA(dataframe['close'], timeperiod=50)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe["close"], timeperiod=7)
        dataframe["rsi_overbought"] = (dataframe["rsi"] > 70).astype(int)
        dataframe["rsi_oversold"] = (dataframe["rsi"] < 30).astype(int)

        # Directional Indicators
        dataframe["plus_di"] = ta.PLUS_DI(dataframe)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe)
        dataframe["DI_values"] = dataframe["plus_di"] - dataframe["minus_di"]
        dataframe["DI_cutoff"] = 0

        # Extrema Detection
        extrema_order = self.indicator_extrema_order.value
        dataframe["maxima"] = (
            dataframe["close"] == dataframe["close"].shift(1).rolling(window=extrema_order).max()
        ).astype(int)
        dataframe["minima"] = (
            dataframe["close"] == dataframe["close"].shift(1).rolling(window=extrema_order).min()
        ).astype(int)
        dataframe["s_extrema"] = 0
        dataframe.loc[dataframe["minima"] == 1, "s_extrema"] = -1
        dataframe.loc[dataframe["maxima"] == 1, "s_extrema"] = 1

        # Heikin-Ashi
        logger.debug(f"{metadata['pair']} 🛠️ Calculating Heikin-Ashi and extrema")
        dataframe["ha_close"] = (dataframe["open"] + dataframe["high"] + dataframe["low"] + dataframe["close"]) / 4
        if dataframe["ha_close"].isna().all():
            logger.error(f"{metadata['pair']} 🚨 All ha_close values are NaN!")

        # Rolling Extrema
        dataframe["minh2"], dataframe["maxh2"] = calculate_minima_maxima(dataframe, self.h2.value)
        dataframe["minh1"], dataframe["maxh1"] = calculate_minima_maxima(dataframe, self.h1.value)
        dataframe["minh0"], dataframe["maxh0"] = calculate_minima_maxima(dataframe, self.h0.value)
        dataframe["mincp"], dataframe["maxcp"] = calculate_minima_maxima(dataframe, self.cp.value)
        logger.debug(f"{metadata['pair']} ✅ Extrema calculated: minh2={dataframe['minh2'].sum()}, maxh2={dataframe['maxh2'].sum()}")

        # Murrey Math Levels
        mml_window = self.indicator_mml_window.value
        murrey_levels = self.calculate_rolling_murrey_math_levels_optimized(dataframe, window_size=mml_window)
        for level_name in MML_LEVEL_NAMES:
            if level_name in murrey_levels:
                dataframe[level_name] = murrey_levels[level_name]
            else:
                dataframe[level_name] = dataframe["close"]

        # MML Oscillator
        mml_4_8 = dataframe.get("[4/8]P")
        mml_plus_3_8 = dataframe.get("[+3/8]P")
        mml_minus_3_8 = dataframe.get("[-3/8]P")
        if mml_4_8 is not None and mml_plus_3_8 is not None and mml_minus_3_8 is not None:
            osc_denominator = (mml_plus_3_8 - mml_minus_3_8).replace(0, np.nan)
            dataframe["mmlextreme_oscillator"] = 100 * ((dataframe["close"] - mml_4_8) / osc_denominator)
        else:
            dataframe["mmlextreme_oscillator"] = np.nan

        # DI Catch and Checks
        dataframe["DI_catch"] = np.where(dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1)
        rolling_window_threshold = self.indicator_rolling_window_threshold.value
        dataframe["minima_sort_threshold"] = dataframe["close"].rolling(
            window=rolling_window_threshold, min_periods=1
        ).min()
        dataframe["maxima_sort_threshold"] = dataframe["close"].rolling(
            window=rolling_window_threshold, min_periods=1
        ).max()
        rolling_check_window = self.indicator_rolling_check_window.value
        dataframe["minima_check"] = (
            dataframe["minima"].rolling(window=rolling_check_window, min_periods=1).sum() == 0
        ).astype(int)
        dataframe["maxima_check"] = (
            dataframe["maxima"].rolling(window=rolling_check_window, min_periods=1).sum() == 0
        ).astype(int)

        # Volatility Calculations
        dataframe["volatility_range"] = dataframe["high"] - dataframe["low"]
        dataframe["avg_volatility"] = dataframe["volatility_range"].rolling(window=10).mean()
        dataframe["avg_volume"] = dataframe["volume"].rolling(window=20).mean()

        # Hurst Exponent
        dataframe['hurst_smooth_cp'] = self.calculate_hurst_exponent(dataframe['close'], lookback=100)
        if dataframe['hurst_smooth_cp'].isna().all():
            logger.warning(f"{metadata['pair']} Hurst calculation failed, using fallback value 0.5")
            dataframe['hurst_smooth_cp'] = 0.5

        # Initialize signal columns
        dataframe['mean_reversion_cp_bull'] = 0
        dataframe['hurst_trend_cp_bull'] = 0
        dataframe['mean_reversion_bear'] = 0
        dataframe['hurst_trend_bear'] = 0

        # Hurst-based signals
        dataframe.loc[
            (
                (dataframe["hurst_smooth_cp"] < self.hurst_mean_rev.value) &
                (dataframe['close'] < dataframe['baseline'])
            ),
            "mean_reversion_cp_bull"
        ] = 1

        dataframe.loc[
            (
                (dataframe["hurst_smooth_cp"] > self.hurst_trending.value) &
                (dataframe['close'] > dataframe['baseline'])
            ),
            "hurst_trend_cp_bull"
        ] = 1

        dataframe.loc[
            (
                (dataframe["hurst_smooth_cp"] < self.hurst_mean_rev.value + 0.1) &
                (dataframe['close'] > dataframe['baseline'])
            ),
            "mean_reversion_bear"
        ] = 1

        dataframe.loc[
            (
                (dataframe["hurst_smooth_cp"] > self.hurst_trending.value) &
                (dataframe['close'] < dataframe['baseline'])
            ),
            "hurst_trend_bear"
        ] = 1

        # Diagnostic logging
        logger.info(f"{metadata['pair']} Hurst Stats: min={dataframe['hurst_smooth_cp'].min():.3f}, max={dataframe['hurst_smooth_cp'].max():.3f}, mean={dataframe['hurst_smooth_cp'].mean():.3f}")
        logger.info(f"{metadata['pair']} Hurst Signals: Mean Reversion Bull={dataframe['mean_reversion_cp_bull'].sum()}, Trend Bull={dataframe['hurst_trend_cp_bull'].sum()}, Mean Reversion Bear={dataframe['mean_reversion_bear'].sum()}, Trend Bear={dataframe['hurst_trend_bear'].sum()}")

        # ===========================================
        # MARKET CORRELATION
        # ===========================================
        if self.enable_btc_correlation.value:
            logger.info(f"🔍 {metadata['pair']} Calculating enhanced BTC correlation...")
            dataframe = self.enhanced_btc_correlation_filter(dataframe, metadata)

        if self.btc_correlation_enabled.value or self.btc_trend_filter_enabled.value:
            logger.info(f"📈 {metadata['pair']} Calculating legacy BTC correlation...")
            dataframe = self.calculate_market_correlation_simple(dataframe, metadata)

        # ===========================================
        # MARKET BREADTH
        # ===========================================
        if self.market_breadth_enabled.value:
            logger.info(f"📊 {metadata['pair']} Calculating weighted market breadth...")
            try:
                dataframe = self.calculate_market_breadth_weighted(dataframe, metadata)
                if len(dataframe) > 0:
                    breadth = dataframe['market_breadth'].iloc[-1]
                    pairs_checked = dataframe.get('breadth_pairs_checked', [0]).iloc[-1]
                    logger.info(f"🎯 {metadata['pair']} Breadth result: {breadth:.2%} ({pairs_checked} pairs)")
            except Exception as e:
                logger.warning(f"⚠️ {metadata['pair']} Market breadth calculation failed: {e}")
                dataframe['market_breadth'] = 0.5
                dataframe['market_direction'] = 0

        # ===========================================
        # MARKET REGIME
        # ===========================================
        if self.regime_filter_enabled.value:
            logger.info(f"🏛️ {metadata['pair']} Calculating market regime...")
            try:
                lookback = self.regime_lookback_period.value if hasattr(self, 'regime_lookback_period') else 48
                logger.info(f"📊 {metadata['pair']} Using lookback period: {lookback}")
                dataframe = self.calculate_market_regime_aligned(dataframe, metadata)
                if len(dataframe) > 0:
                    regime = dataframe['market_regime'].iloc[-1]
                    volatility = dataframe.get('market_volatility', [0.02]).iloc[-1]
                    adx = dataframe.get('market_adx', [25]).iloc[-1]
                    logger.info(f"🎯 {metadata['pair']} Regime result: {regime} (vol: {volatility:.3f}, adx: {adx:.1f})")
            except Exception as e:
                logger.warning(f"⚠️ {metadata['pair']} Market regime calculation failed: {e}")
                dataframe['market_regime'] = 'neutral'
                dataframe['market_volatility'] = 0.02
                dataframe['market_adx'] = 25

        # ===========================================
        # MARKET CAP TREND
        # ===========================================
        if self.total_mcap_filter_enabled.value:
            logger.info(f"💰 {metadata['pair']} Calculating market cap trend...")
            try:
                dataframe = self.calculate_total_market_cap_trend(dataframe, metadata)
            except Exception as e:
                logger.warning(f"⚠️ {metadata['pair']} Market cap trend calculation failed: {e}")
                dataframe['mcap_trend'] = 0

        # ===========================================
        # ENHANCED SIGNAL FILTERING
        # ===========================================
        try:
            dataframe = self.calculate_enhanced_trend_filters(dataframe)
            logger.debug(f"✅ {metadata['pair']} Enhanced trend filters calculated")
        except Exception as e:
            logger.warning(f"⚠️ {metadata['pair']} Enhanced trend filters failed: {e}")
            dataframe['trend_alignment_bull'] = 0.5
            dataframe['trend_alignment_bear'] = 0.5

        try:
            dataframe = self.detect_market_state(dataframe)
            logger.debug(f"✅ {metadata['pair']} Market state detected")
        except Exception as e:
            logger.warning(f"⚠️ {metadata['pair']} Market state detection failed: {e}")
            dataframe['market_state'] = 'normal'

        try:
            dataframe = self.calculate_signal_strength(dataframe)
            logger.debug(f"✅ {metadata['pair']} Signal strength calculated")
        except Exception as e:
            logger.warning(f"⚠️ {metadata['pair']} Signal strength calculation failed: {e}")
            dataframe['signal_strength'] = 0.5

        # ===========================================
        # MARKET SCORE CALCULATION
        # ===========================================
        # dataframe['market_score'] = 0.5
        # if 'btc_correlation' in dataframe.columns:
        #     dataframe['market_score'] += dataframe['btc_correlation'] * 0.15
        # if 'btc_correlation_ok' in dataframe.columns:
        #     dataframe['market_score'] += dataframe['btc_correlation_ok'].astype(float) * 0.20
        # if 'market_breadth' in dataframe.columns:
        #     breadth_contribution = (dataframe['market_breadth'] - 0.5) * 0.25
        #     dataframe['market_score'] += breadth_contribution
        # if 'mcap_trend' in dataframe.columns:
        #     dataframe['market_score'] += dataframe['mcap_trend'] * 0.10
        # if 'market_regime' in dataframe.columns:
        #     regime_scores = {
        #         'bull_run': 0.85,
        #         'bear_market': 0.15,
        #         'sideways': 0.50,
        #         'high_volatility': 0.30,
        #         'transitional': 0.50,
        #         'neutral': 0.50
        #     }
        #     dataframe['regime_score'] = dataframe['market_regime'].map(lambda x: regime_scores.get(x, 0.5))
        #     dataframe['market_score'] += (dataframe['regime_score'] - 0.5) * 0.15
        # if 'trend_alignment_bull' in dataframe.columns:
        #     dataframe['market_score'] += dataframe['trend_alignment_bull'] * 0.10
        #     dataframe['market_score'] -= dataframe['trend_alignment_bear'] * 0.10
        # if 'macd_bull' in dataframe.columns:
        #     dataframe['market_score'] += dataframe['macd_bull'].astype(int) * 0.05
        #     dataframe['market_score'] -= dataframe['macd_bear'].astype(int) * 0.05
        # if 'volume_confirmation' in dataframe.columns:
        #     volume_contribution = (dataframe['volume_confirmation'] - 0.5) * 0.10
        #     dataframe['market_score'] += volume_contribution
        # dataframe['market_score'] = dataframe['market_score'].clip(0, 1)

        # ===========================================
        # MARKET SCORE CALCULATION
        # ===========================================
        dataframe['market_score'] = 0.5  # Neutral base

        # BTC correlation (-1 to 1) normalized to (-0.075 to 0.075)
        if 'btc_correlation' in dataframe.columns:
            btc_corr_normalized = (dataframe['btc_correlation'] + 1) / 2  # Maps [-1, 1] to [0, 1]
            dataframe['market_score'] += (btc_corr_normalized - 0.5) * 0.15  # Contribution: [-0.075, 0.075]

        # BTC correlation OK (0 or 1) normalized to (-0.1 or 0.1)
        if 'btc_correlation_ok' in dataframe.columns:
            btc_ok_normalized = dataframe['btc_correlation_ok'].astype(float)  # 0 or 1
            dataframe['market_score'] += (btc_ok_normalized - 0.5) * 0.20  # Contribution: [-0.1, 0.1]

        # Market breadth (0 to 1) normalized to (-0.125 to 0.125)
        if 'market_breadth' in dataframe.columns:
            breadth_contribution = (dataframe['market_breadth'] - 0.5) * 0.25
            dataframe['market_score'] += breadth_contribution  # Contribution: [-0.125, 0.125]

        # Market cap trend (-1 to 1) normalized to (-0.05 to 0.05)
        if 'mcap_trend' in dataframe.columns:
            mcap_trend_normalized = (dataframe['mcap_trend'] + 1) / 2  # Maps [-1, 1] to [0, 1]
            dataframe['market_score'] += (mcap_trend_normalized - 0.5) * 0.10  # Contribution: [-0.05, 0.05]

        # Market regime (0.15 to 0.85) normalized to (-0.0525 to 0.0525)
        if 'market_regime' in dataframe.columns:
            regime_scores = {
                'bull_run': 0.85,
                'bear_market': 0.15,
                'sideways': 0.50,
                'high_volatility': 0.30,
                'transitional': 0.50,
                'neutral': 0.50
            }
            dataframe['regime_score'] = dataframe['market_regime'].map(lambda x: regime_scores.get(x, 0.5))
            regime_normalized = (dataframe['regime_score'] - 0.5) * 0.15
            dataframe['market_score'] += regime_normalized  # Contribution: [-0.0525, 0.0525]

        # Trend alignment (bull - bear) normalized to (-0.05 to 0.05)
        if 'trend_alignment_bull' in dataframe.columns:
            trend_net = dataframe['trend_alignment_bull'] - dataframe.get('trend_alignment_bear', 0)
            trend_normalized = trend_net / 2  # Maps [-1, 1] to [-0.5, 0.5]
            dataframe['market_score'] += trend_normalized * 0.10  # Contribution: [-0.05, 0.05]

        # MACD signals (bull - bear) normalized to (-0.025 to 0.025)
        if 'macd_bull' in dataframe.columns:
            macd_net = dataframe['macd_bull'].astype(int) - dataframe.get('macd_bear', 0).astype(int)
            macd_normalized = macd_net / 2  # Maps [-1, 1] to [-0.5, 0.5]
            dataframe['market_score'] += macd_normalized * 0.05  # Contribution: [-0.025, 0.025]

        # Volume confirmation (0 to 1) normalized to (-0.05 to 0.05)
        if 'volume_confirmation' in dataframe.columns:
            volume_normalized = (dataframe['volume_confirmation'] - 0.5) * 0.10
            dataframe['market_score'] += volume_normalized  # Contribution: [-0.05, 0.05]

        # Normalize to [0, 1] range using dataframe min and max
        min_score = dataframe['market_score'].min()
        max_score = dataframe['market_score'].max()
        if max_score > min_score:  # Avoid division by zero
            dataframe['market_score'] = (dataframe['market_score'] - min_score) / (max_score - min_score)
        else:
            dataframe['market_score'] = 0.5  # Neutral if all scores are identical
        dataframe['market_score'] = dataframe['market_score'].clip(0, 1)  # Ensure no edge case overflow

        # ===========================================
        # SIGNAL QUALITY INDICATORS
        # ===========================================
        dataframe['signal_persistence'] = 0
        for window in [3, 5, 10]:
            dataframe[f'bullish_persistence_{window}'] = (
                (dataframe['rsi'] < 50).rolling(window).sum() / window
            )
            dataframe[f'bearish_persistence_{window}'] = (
                (dataframe['rsi'] > 50).rolling(window).sum() / window
            )

        dataframe['volume_momentum'] = (
            dataframe['volume'].rolling(3).mean() / 
            dataframe['volume'].rolling(20).mean()
        ).fillna(1.0)

        dataframe['price_momentum_strength'] = abs(
            dataframe['close'].pct_change(5)
        ).rolling(10).mean()

        # ===========================================
        # BREADTH-SPECIFIC INDICATORS
        # ===========================================
        if 'market_breadth' in dataframe.columns:
            dataframe['breadth_momentum'] = dataframe['market_breadth'].diff()
            dataframe['breadth_trend'] = dataframe['market_breadth'].rolling(5).mean()
            price_direction = (dataframe['close'] > dataframe['close'].shift(5)).astype(int)
            breadth_direction = (dataframe['market_breadth'] > 0.5).astype(int)
            dataframe['breadth_price_divergence'] = abs(price_direction - breadth_direction)

        if 'market_adx' in dataframe.columns:
            dataframe['regime_strength'] = dataframe['market_adx'] / 50
            dataframe['strong_regime'] = dataframe['market_adx'] > 30

        # ===========================================
        # FINAL LOGGING & VALIDATION
        # ===========================================
        try:
            if len(dataframe) > 0:
                last_row = dataframe.iloc[-1]
                market_breadth = last_row.get('market_breadth', 0.5)
                market_regime = last_row.get('market_regime', 'unknown')
                market_score = last_row.get('market_score', 0.5)
                btc_ok = last_row.get('btc_correlation_ok', True)
                logger.info(f"📊 FINAL MARKET CONDITIONS for {metadata['pair']}:")
                logger.info(f"   🌐 Breadth: {market_breadth:.2%}")
                logger.info(f"   🏛️ Regime: {market_regime}")
                logger.info(f"   📈 Score: {market_score:.2%}")
                logger.info(f"   ₿ BTC OK: {btc_ok}")
                if market_regime == 'unknown':
                    logger.warning(f"⚠️ {metadata['pair']} Market regime is unknown - check calculation method")
            else:
                logger.error(f"🚨 {metadata['pair']} Empty dataframe after indicators!")
        except Exception as e:
            logger.error(f"💥 {metadata['pair']} Final logging error: {e}")

        return dataframe

    def enhanced_btc_correlation_filter(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        🛡️ ROBUST: BTC correlation with comprehensive error handling and logging
        """
        
        pair = metadata['pair']
        
        # Skip BTC correlation for BTC pairs
        if 'BTC' in pair:
            logger.info(f"{pair} 🟡 Skipping BTC correlation for BTC pair")
            dataframe['btc_correlation_ok'] = True
            return dataframe
        
        try:
            # ===========================================
            # BTC DATA RETRIEVAL WITH VALIDATION
            # ===========================================
            
            current_pair = metadata['pair']
            if ':' in current_pair:
                # Futures mode - extract settlement currency
                settlement = current_pair.split(':')[1]
                btc_pair = f'BTC/USDT:{settlement}'
            else:
                # Spot mode
                btc_pair = 'BTC/USDT'

            logger.info(f"{current_pair} Using BTC pair: {btc_pair}")

            # Now use the detected format
            btc_1h = self.dp.get_pair_dataframe(btc_pair, '1h') 
            btc_4h = self.dp.get_pair_dataframe(btc_pair, '4h')
            
            # Comprehensive validation
            btc_data_available = False
            btc_timeframes_ok = []
            
            for tf_name, btc_df in [('1h', btc_1h), ('4h', btc_4h)]:
                if not btc_df.empty and len(btc_df) > 30:
                    btc_timeframes_ok.append(tf_name)
                    btc_data_available = True
                    logger.info(f"{pair} ✅ BTC {tf_name} data: {len(btc_df)} candles")
                else:
                    logger.warning(f"{pair} ❌ BTC {tf_name} data insufficient: {len(btc_df) if not btc_df.empty else 0} candles")
            
            if not btc_data_available:
                logger.error(f"{pair} 🚨 NO BTC DATA AVAILABLE - DISABLING BTC CORRELATION")
                logger.error(f"{pair} 📊 This means backtest results may not match live trading!")
                
                # Fallback: Allow all trades (backtest won't match live)
                dataframe['btc_correlation_ok'] = True
                dataframe['btc_status'] = 'NO_DATA'
                return dataframe
            
            # Use best available timeframe (prefer 1h, fallback to others)
            if '1h' in btc_timeframes_ok:
                btc_main = btc_1h
                main_tf = '1h'
            elif '4h' in btc_timeframes_ok:
                btc_main = btc_4h  
                main_tf = '4h'
            else:
                raise Exception("No usable BTC timeframe data")
            
            logger.info(f"{pair} 🎯 Using BTC {main_tf} data for correlation ({len(btc_main)} candles)")
            
            # ===========================================
            # BTC TREND ANALYSIS
            # ===========================================
            
            # Calculate BTC indicators
            btc_main['sma_24'] = btc_main['close'].rolling(24).mean()
            btc_main['sma_50'] = btc_main['close'].rolling(50).mean()
            btc_main['rsi'] = ta.RSI(btc_main['close'], timeperiod=14)
            
            # BTC trend determination
            btc_last = btc_main.iloc[-1]
            btc_prev = btc_main.iloc[-2] if len(btc_main) > 1 else btc_last
            
            # Multiple trend indicators
            btc_above_sma24 = btc_last['close'] > btc_last['sma_24']
            btc_above_sma50 = btc_last['close'] > btc_last['sma_50']
            btc_sma_bullish = btc_last['sma_24'] > btc_last['sma_50']
            btc_price_momentum = (btc_last['close'] - btc_prev['close']) / btc_prev['close']
            
            # BTC trend classification
            btc_bullish_signals = sum([
                btc_above_sma24,
                btc_above_sma50, 
                btc_sma_bullish,
                btc_price_momentum > 0.001,  # 0.1% positive momentum
                30 < btc_last['rsi'] < 70    # Healthy RSI range
            ])
            
            btc_trend_strong = btc_bullish_signals >= 4
            btc_trend_weak = btc_bullish_signals <= 1
            btc_trend_neutral = not btc_trend_strong and not btc_trend_weak
            
            # ===========================================
            # CORRELATION FILTER LOGIC
            # ===========================================
            
            # Conservative approach: Allow trades when BTC is not strongly bearish
            btc_correlation_ok = not btc_trend_weak
            
            # Enhanced logging
            logger.info(f"{pair} 🔍 BTC CORRELATION ANALYSIS:")
            logger.info(f"   📊 BTC Price: ${btc_last['close']:.0f}")
            logger.info(f"   📈 Above SMA24: {btc_above_sma24}")
            logger.info(f"   📈 Above SMA50: {btc_above_sma50}")
            logger.info(f"   🎯 Bullish Signals: {btc_bullish_signals}/5")
            logger.info(f"   🚦 BTC Trend: {'STRONG' if btc_trend_strong else 'WEAK' if btc_trend_weak else 'NEUTRAL'}")
            logger.info(f"   ✅ Correlation OK: {btc_correlation_ok}")
            
            # Apply to dataframe
            dataframe['btc_correlation_ok'] = btc_correlation_ok
            dataframe['btc_trend_score'] = btc_bullish_signals
            dataframe['btc_status'] = f"OK_{main_tf}"
            
            return dataframe
            
        except Exception as e:
            logger.error(f"{pair} 💥 BTC CORRELATION ERROR: {str(e)}")
            logger.error(f"{pair} 🚨 FALLING BACK TO NO BTC FILTER")
            
            # Emergency fallback
            dataframe['btc_correlation_ok'] = True
            dataframe['btc_status'] = 'ERROR'
            return dataframe
    def safe_time_diff(self, time1, time2):
        """Safe time difference calculation"""
        try:
            # Handle various timestamp formats
            if isinstance(time1, (int, float)):
                time1 = pd.Timestamp.fromtimestamp(time1)
            if isinstance(time2, (int, float)):
                time2 = pd.Timestamp.fromtimestamp(time2)
                
            return (pd.Timestamp(time1) - pd.Timestamp(time2)).total_seconds() / 60
        except Exception as e:
            logger.warning(f"Time diff error: {e}")
            return 0

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        🔄 Optimized Implementation: Simplified long and short signals with adaptive filters
        🔧 Incorporates Hurst Exponent-based signals alongside existing conditions
        """
        pair = metadata['pair']
        
        # Safe initialization for re-entry system
        self.get_or_init_reentry_data(pair)
        
        # ===========================================
        # INITIALIZE COLUMNS
        # ===========================================
        df["enter_long"] = 0
        df["enter_short"] = 0  
        df["enter_tag"] = ""
        df["exit_long"] = 0    
        df["exit_short"] = 0
        
        # ===========================================
        # MARKET CONDITION CALCULATIONS
        # ===========================================
        df['green_candle'] = (df['close'] > df['open']).astype(int)
        df['red_candle'] = (df['close'] < df['open']).astype(int)
        df['consecutive_green'] = df['green_candle'].rolling(3).sum()
        df['consecutive_red'] = df['red_candle'].rolling(3).sum()
        
        df['trend_consistency'] = (df['close'] / df['close'].shift(10) - 1).fillna(0)
        df['market_score'] = df.get('market_score', 0.5)
        
        # Conditions for long signals
        df['below_midpoint'] = df['close'] < df['[4/8]P']
        lookback_candles = getattr(self, 'lookback_candles', 15)
        min_drop_pct = getattr(self, 'min_drop_pct', 0.015)
        df['highest_close'] = df['close'].rolling(lookback_candles).max()
        df['pct_drop_from_high'] = (df['highest_close'] - df['close']) / df['highest_close']
        df['significant_pullback'] = df['pct_drop_from_high'] > min_drop_pct
        
        momentum_lookback = getattr(self, 'momentum_lookback', 3)
        max_drop_pct = getattr(self, 'max_drop_pct', 0.03)
        df['recent_drop'] = (df['close'].shift(1) - df['close'].rolling(momentum_lookback).min()) / df['close'].shift(1)
        df['momentum_safe'] = df['recent_drop'] <= max_drop_pct
        
        # Conditions for short signals
        df['above_midpoint'] = df['close'] > df['[4/8]P']
        min_rise_pct = getattr(self, 'min_rise_pct', 0.015)
        df['lowest_close'] = df['close'].rolling(lookback_candles).min()
        df['pct_rise_from_low'] = (df['close'] - df['lowest_close']) / df['lowest_close']
        df['significant_rise'] = df['pct_rise_from_low'] > min_rise_pct
        
        max_rise_pct = getattr(self, 'max_rise_pct', 0.03)
        df['recent_rise'] = (df['close'].rolling(momentum_lookback).max() - df['close'].shift(1)) / df['close'].shift(1)
        df['momentum_safe_short'] = df['recent_rise'] <= max_rise_pct
        
        # MML analysis
        bullish_mml = (
            (df["close"] > df["[6/8]P"]) |
            ((df["close"] > df["[4/8]P"]) & (df["close"].shift(5) < df["[4/8]P"].shift(5)))
        )
        bearish_mml = (
            (df["close"] < df["[2/8]P"]) |
            ((df["close"] < df["[4/8]P"]) & (df["close"].shift(5) > df["[4/8]P"].shift(5)))
        )
        range_bound = (
            (df["close"] >= df["[2/8]P"]) & 
            (df["close"] <= df["[6/8]P"]) &
            (~bullish_mml) & (~bearish_mml)
        )
        mml_support_bounce = (
            ((df["low"] <= df["[2/8]P"]) & (df["close"] > df["[2/8]P"])) |
            ((df["low"] <= df["[4/8]P"]) & (df["close"] > df["[4/8]P"]))
        )
        mml_resistance_reject = (
            ((df["high"] >= df["[6/8]P"]) & (df["close"] < df["[6/8]P"])) |
            ((df["high"] >= df["[8/8]P"]) & (df["close"] < df["[8/8]P"]))
        )
        
        # ===========================================
        # LONG SIGNALS
        # ===========================================
        long_signal_mml_breakout = (
            (df["close"] > df["[6/8]P"]) &
            (df["close"].shift(1) <= df["[6/8]P"].shift(1)) &
            (df["volume"] > df["volume"].rolling(20).mean() * 1.1) &
            (df["rsi"] > 50) &
            (df["close"] > df["ema20"]) &
            (df["below_midpoint"] | (df["rsi"].shift(1) < df["rsi"]))
        )
        
        long_signal_support_bounce = (
            mml_support_bounce &
            (df["rsi"] < 45) &
            (df["close"] > df["close"].shift(1)) &
            (df["volume"] > df["volume"].rolling(10).mean() * 1.1) &
            (df["minima"] == 1) &
            (df["significant_pullback"]) &
            (df["momentum_safe"])
        )
        
        long_signal_50_reclaim = (
            (df["close"] > df["[4/8]P"]) &
            (df["close"].shift(1) <= df["[4/8]P"].shift(1)) &
            (df["rsi"] > 50) &
            (df["volume"] > df["volume"].rolling(15).mean() * 1.1) &
            (df["close"] > df["ema20"]) &
            (df["market_score"] > 0.2)
        )
        
        long_signal_range_long = (
            range_bound &
            (df["low"] <= df["[2/8]P"]) &
            (df["close"] > df["[2/8]P"]) &
            (df["rsi"] < 50) &
            (df["close"] > df["close"].shift(1)) &
            (df["minima"] == 1) &
            (df["volume"] > df["volume"].rolling(10).mean() * 1.1) &
            (df["significant_pullback"])
        )
        
        long_signal_trend_continuation = (
            bullish_mml &
            (df["close"] > df["[5/8]P"]) &
            (df["rsi"].between(40, 65)) &
            (df["close"] > df["ema20"]) &
            (df["volume"] > df["volume"].rolling(15).mean() * 1.1)
        )
        
        long_signal_extreme_oversold = (
            (df["low"] <= df["[1/8]P"]) &
            (df["close"] > df["[1/8]P"]) &
            (df["rsi"] < 35) &
            (df["volume"] > df["volume"].rolling(20).mean() * 1.2) &
            (df["close"] > df["close"].shift(1)) &
            (df["minima"] == 1) &
            (df["significant_pullback"]) &
            (df["momentum_safe"])
        )
        
        # New Hurst-based long signals
        long_signal_hurst_mean_reversion = (
            (df["mean_reversion_cp_bull"] == 1) &  # Fixed from mean_reversion_bull
            (df["rsi"] < 50) &
            (df["close"] > df["close"].shift(1)) &
            (df["volume"] > df["volume"].rolling(10).mean() * 1.0) &
            (df["significant_pullback"]) &
            (df["market_score"] > 0.1)
        )
        
        long_signal_hurst_trend = (
            (df["hurst_trend_cp_bull"] == 1) &  # Fixed from hurst_trend_bull
            (df["rsi"].between(35, 70)) &
            (df["close"] > df["ema20"]) &
            (df["volume"] > df["volume"].rolling(15).mean() * 1.0) &
            (df["market_score"] > 0.1)
        )

        long_signal_taco = (
            (df["market_score"] > df["market_score"].shift()) &
            # (df["volume"] > df["volume"].rolling(15).mean() * 1.0) &
            (df["market_score"].shift() == df["market_score"].min())
        )
        
        any_long_signal = (
            # long_signal_mml_breakout |
            # long_signal_support_bounce |
            # long_signal_50_reclaim |
            # long_signal_range_long |
            # long_signal_trend_continuation |
            # long_signal_extreme_oversold |
            # long_signal_hurst_mean_reversion |
            # long_signal_hurst_trend |
            long_signal_taco
        )
        
        # ===========================================
        # SHORT SIGNALS
        # ===========================================
        if self.can_short:
            short_signal_bearish_breakdown = (
                (df["close"] < df["[2/8]P"]) &
                (df["close"].shift(1) >= df["[2/8]P"].shift(1)) &
                (df["volume"] > df["volume"].rolling(20).mean() * 1.1) &
                (df["rsi"] < 50) &
                (df["close"] < df["ema20"]) &
                (df["above_midpoint"] | (df["rsi"].shift(1) > df["rsi"]))
            )
            
            short_signal_resistance_reject = (
                mml_resistance_reject &
                (df["rsi"] > 50) &
                (df["close"] < df["close"].shift(1)) &
                (df["volume"] > df["volume"].rolling(10).mean() * 1.1) &
                (df["maxima"] == 1) &
                (df["significant_rise"]) &
                (df["momentum_safe_short"])
            )
            
            short_signal_50_breakdown = (
                (df["close"] < df["[4/8]P"]) &
                (df["close"].shift(1) >= df["[4/8]P"].shift(1)) &
                (df["rsi"] < 50) &
                (df["volume"] > df["volume"].rolling(15).mean() * 1.1) &
                (df["close"] < df["ema20"]) &
                (df["market_score"] < 0.8)
            )
            
            short_signal_range_short = (
                range_bound &
                (df["high"] >= df["[6/8]P"]) &
                (df["close"] < df["[6/8]P"]) &
                (df["rsi"] > 50) &
                (df["close"] < df["close"].shift(1)) &
                (df["maxima"] == 1) &
                (df["volume"] > df["volume"].rolling(10).mean() * 1.1) &
                (df["significant_rise"])
            )
            
            short_signal_trend_continuation = (
                bearish_mml &
                (df["close"] < df["[3/8]P"]) &
                (df["rsi"].between(35, 60)) &
                (df["close"] < df["ema20"]) &
                (df["volume"] > df["volume"].rolling(15).mean() * 1.1)
            )
            
            short_signal_extreme_overbought = (
                (df["high"] >= df["[7/8]P"]) &
                (df["close"] < df["[7/8]P"]) &
                (df["rsi"] > 65) &
                (df["volume"] > df["volume"].rolling(20).mean() * 1.2) &
                (df["close"] < df["close"].shift(1)) &
                (df["maxima"] == 1) &
                (df["significant_rise"]) &
                (df["momentum_safe_short"])
            )
            
            # New Hurst-based short signals
            short_signal_hurst_mean_reversion = (
                (df["mean_reversion_bear"] == 1) &
                (df["rsi"] > 50) &
                (df["close"] < df["close"].shift(1)) &
                (df["volume"] > df["volume"].rolling(10).mean() * 1.0) &
                (df["significant_rise"]) &
                (df["market_score"] < 0.9)
            )
            
            short_signal_hurst_trend = (
                (df["hurst_trend_bear"] == 1) &
                (df["rsi"].between(30, 65)) &
                (df["close"] < df["ema20"]) &
                (df["volume"] > df["volume"].rolling(15).mean() * 1.0) &
                (df["market_score"] < 0.9)
            )
            
            any_short_signal = (
                short_signal_bearish_breakdown |
                short_signal_resistance_reject |
                short_signal_50_breakdown |
                short_signal_range_short |
                short_signal_trend_continuation |
                short_signal_extreme_overbought |
                short_signal_hurst_mean_reversion |
                short_signal_hurst_trend
            )
        else:
            any_short_signal = pd.Series([False] * len(df), index=df.index)
        
        # Apply market regime filter
        if 'market_regime' in df.columns:
            regime = df['market_regime'].iloc[-1]
            if regime == 'bear_market':
                any_long_signal = any_long_signal & (df['rsi'] < 35)
                any_short_signal = any_short_signal & (df['rsi'] > 45)
            elif regime == 'bull_run':
                any_long_signal = any_long_signal & (df['rsi'] > 40)
                any_short_signal = any_short_signal & (df['rsi'] > 65)
            logger.info(f"{pair} 🏛️ Market regime: {regime}")
        
        # ===========================================
        # RE-ENTRY AND MARKET FILTERS
        # ===========================================
        # Add parameters


        # In populate_entry_trend, replace reentry section:
        if self.enable_reentry.value:
            try:
                current_time = df.index[-1] if not df.empty else datetime.now()
                recent_exits = self.get_recent_exits(pair, current_time)
                
                logger.info(f"{pair} 🔄 Re-entry check: {len(recent_exits)} recent exits")
                
                reentry_long_signals = pd.Series([False] * len(df), index=df.index)
                reentry_short_signals = pd.Series([False] * len(df), index=df.index)
                
                market_score = df['market_score'].fillna(0.5)
                market_breadth = df.get('market_breadth', pd.Series([0.5] * len(df), index=df.index)).fillna(0.5)
                btc_ok = df.get('btc_correlation_ok', pd.Series([True] * len(df), index=df.index)).fillna(True)
                
                # Track last reentry candle to enforce cooldown
                if not hasattr(self, 'last_reentry_candle'):
                    self.last_reentry_candle = {}
                
                for exit_time, direction, reason, profit in recent_exits:
                    try:
                        time_since_exit = self.safe_time_diff(current_time, exit_time)
                        if pair in self.reentry_count:
                            current_count = self.reentry_count[pair].get(direction, 0)
                            max_reentries = int(self.max_reentries_per_direction.value)
                            if current_count >= max_reentries:
                                logger.info(f"{pair} 🛑 Max {direction} re-entries: {current_count}")
                                continue
                        # Check candle-based cooldown
                        if pair in self.last_reentry_candle:
                            last_reentry_time = self.last_reentry_candle.get(pair, {}).get(direction)
                            if last_reentry_time and (current_time - last_reentry_time).total_seconds() / 60 < self.reentry_cooldown_candles.value * 15:
                                continue
                    except Exception as e:
                        logger.warning(f"{pair} Re-entry processing error: {e}, skipping this exit")
                        continue
                    
                    if direction == 'long':
                        if (reason == 'roi' and 
                            profit > max(0.015, df['atr'].iloc[-1] * 0.5) and 
                            self.quick_reentry_cooldown.value <= time_since_exit <= 30):
                            roi_long_conditions = (
                                (df["rsi"] < 60) &  # Relaxed
                                (df["close"] > df["[2/8]P"]) &
                                (df["close"] < df["baseline"] * (1 - self.rebuy_pullback_pct.value)) &  # Pullback
                                (df["volume"] > df["volume"].rolling(10).mean()) &
                                (market_score > 0.3) &  # Relaxed
                                (market_breadth > 0.25) &  # Relaxed
                                btc_ok
                            )
                            reentry_long_signals.iloc[-1] = roi_long_conditions.iloc[-1]
                            if roi_long_conditions.iloc[-1]:
                                logger.info(f"{pair} 🎯 LONG ROI re-entry: {profit:.2%} profit")
                                self.last_reentry_candle.setdefault(pair, {})[direction] = current_time
                        
                        elif (self.reentry_cooldown_minutes.value <= time_since_exit <= 120):
                            original_long_active = (
                                long_signal_mml_breakout |
                                long_signal_support_bounce |
                                long_signal_50_reclaim |
                                long_signal_range_long |
                                long_signal_trend_continuation |
                                long_signal_extreme_oversold |
                                long_signal_hurst_mean_reversion |
                                long_signal_hurst_trend
                            )
                            std_long_conditions = (
                                original_long_active &
                                (df["rsi"] < 50) &  # Relaxed
                                (df["close"] > df["[1/8]P"]) &
                                (df["close"] < df["baseline"] * (1 - self.rebuy_pullback_pct.value)) &  # Pullback
                                (df["volume"] > df["volume"].rolling(10).mean()) &
                                (market_breadth > 0.3) &  # Relaxed
                                btc_ok
                            )
                            reentry_long_signals.iloc[-1] = std_long_conditions.iloc[-1]
                            if std_long_conditions.iloc[-1]:
                                logger.info(f"{pair} 🔄 LONG standard re-entry ({time_since_exit:.0f}m)")
                                self.last_reentry_candle.setdefault(pair, {})[direction] = current_time
                    
                    elif direction == 'short' and self.can_short:
                        if (reason == 'roi' and 
                            profit > max(0.015, df['atr'].iloc[-1] * 0.5) and 
                            self.quick_reentry_cooldown.value <= time_since_exit <= 30):
                            roi_short_conditions = (
                                (df["rsi"] > 50) &  # Relaxed
                                (df["close"] < df["[6/8]P"]) &
                                (df["close"] > df["baseline"] * (1 + self.rebuy_pullback_pct.value)) &  # Rise
                                (df["volume"] > df["volume"].rolling(10).mean()) &
                                (market_score < 0.7) &  # Relaxed
                                (market_breadth < 0.75) &  # Relaxed
                                btc_ok
                            )
                            reentry_short_signals.iloc[-1] = roi_short_conditions.iloc[-1]
                            if roi_short_conditions.iloc[-1]:
                                logger.info(f"{pair} 🎯 SHORT ROI re-entry: {profit:.2%} profit")
                                self.last_reentry_candle.setdefault(pair, {})[direction] = current_time
                        
                        elif (self.reentry_cooldown_minutes.value <= time_since_exit <= 120):
                            original_short_active = (
                                short_signal_bearish_breakdown |
                                short_signal_resistance_reject |
                                short_signal_50_breakdown |
                                short_signal_range_short |
                                short_signal_trend_continuation |
                                short_signal_extreme_overbought |
                                short_signal_hurst_mean_reversion |
                                short_signal_hurst_trend
                            )
                            std_short_conditions = (
                                original_short_active &
                                (df["rsi"] > 45) &  # Relaxed
                                (df["close"] < df["[7/8]P"]) &
                                (df["close"] > df["baseline"] * (1 + self.rebuy_pullback_pct.value)) &  # Rise
                                (df["volume"] > df["volume"].rolling(10).mean()) &
                                (market_breadth < 0.7) &  # Relaxed
                                btc_ok
                            )
                            reentry_short_signals.iloc[-1] = std_short_conditions.iloc[-1]
                            if std_short_conditions.iloc[-1]:
                                logger.info(f"{pair} 🔄 SHORT standard re-entry ({time_since_exit:.0f}m)")
                                self.last_reentry_candle.setdefault(pair, {})[direction] = current_time
                
                any_long_signal = any_long_signal | reentry_long_signals
                any_short_signal = any_short_signal | reentry_short_signals
                
                if pair not in self.reentry_count:
                    self.reentry_count[pair] = {'long': 0, 'short': 0}
                
                reentry_long_count = reentry_long_signals.sum()
                reentry_short_count = reentry_short_signals.sum()
                
                if reentry_long_count > 0:
                    self.reentry_count[pair]['long'] += 1
                    logger.info(f"{pair} 📈 LONG re-entry #{self.reentry_count[pair]['long']} ({reentry_long_count} signals)")
                
                if reentry_short_count > 0:
                    self.reentry_count[pair]['short'] += 1
                    logger.info(f"{pair} 📉 SHORT re-entry #{self.reentry_count[pair]['short']} ({reentry_short_count} signals)")
            
            except Exception as e:
                logger.error(f"{pair} 💥 Re-entry error: {e}")
        
        # Apply market breadth filter
        if 'market_breadth' in df.columns:
            breadth = df['market_breadth'].fillna(0.5)
            breadth_long_ok = breadth > 0.25
            breadth_short_ok = breadth < 0.75
            any_long_signal = any_long_signal & breadth_long_ok
            any_short_signal = any_short_signal & breadth_short_ok
            logger.info(f"{pair} 📊 Market breadth: {breadth.iloc[-1]:.1%}")
        
        # Apply BTC correlation filter
        if 'btc_correlation_ok' in df.columns:
            btc_ok = df['btc_correlation_ok'].fillna(True).astype(bool)
            # any_long_signal = any_long_signal & btc_ok  # Commented out to disable BTC correlation filter
            # any_short_signal = any_short_signal & btc_ok  # Commented out to disable BTC correlation filter
            logger.info(f"{pair} ₿ BTC correlation: {'✅' if btc_ok.iloc[-1] else '❌'}")
        
        # Apply market score filter
        if 'market_score' in df.columns:
            score = df['market_score'].fillna(0.5)
            score_long_ok = score > 0.20
            score_short_ok = score < 0.80
            any_long_signal = any_long_signal & score_long_ok
            any_short_signal = any_short_signal & score_short_ok
            logger.info(f"{pair} 📈 Market score: {score.iloc[-1]:.1%}")
        
        # ===========================================
        # FINAL SIGNAL ASSIGNMENT
        # ===========================================
        df.loc[any_long_signal, "enter_long"] = 1
        if self.can_short:
            df.loc[any_short_signal, "enter_short"] = 1
        
        # Long signal tags
        df.loc[any_long_signal & long_signal_mml_breakout, "enter_tag"] = "MML_Breakout_75"
        df.loc[any_long_signal & long_signal_support_bounce & (df["enter_tag"] == ""), "enter_tag"] = "Support_Bounce"
        df.loc[any_long_signal & long_signal_50_reclaim & (df["enter_tag"] == ""), "enter_tag"] = "MML_50_Reclaim"
        df.loc[any_long_signal & long_signal_range_long & (df["enter_tag"] == ""), "enter_tag"] = "Range_Long"
        df.loc[any_long_signal & long_signal_trend_continuation & (df["enter_tag"] == ""), "enter_tag"] = "Trend_Cont"
        df.loc[any_long_signal & long_signal_extreme_oversold & (df["enter_tag"] == ""), "enter_tag"] = "Extreme_Oversold"
        df.loc[any_long_signal & long_signal_hurst_mean_reversion & (df["enter_tag"] == ""), "enter_tag"] = "Hurst_Mean_Reversion"
        df.loc[any_long_signal & long_signal_hurst_trend & (df["enter_tag"] == ""), "enter_tag"] = "Hurst_Trend"
        
        # Short signal tags
        if self.can_short:
            df.loc[any_short_signal & short_signal_bearish_breakdown, "enter_tag"] = "Bear_Breakdown_25"
            df.loc[any_short_signal & short_signal_resistance_reject & (df["enter_tag"] == ""), "enter_tag"] = "Resistance_Reject"
            df.loc[any_short_signal & short_signal_50_breakdown & (df["enter_tag"] == ""), "enter_tag"] = "MML_50_Breakdown"
            df.loc[any_short_signal & short_signal_range_short & (df["enter_tag"] == ""), "enter_tag"] = "Range_Short"
            df.loc[any_short_signal & short_signal_trend_continuation & (df["enter_tag"] == ""), "enter_tag"] = "Bear_Trend_Cont"
            df.loc[any_short_signal & short_signal_extreme_overbought & (df["enter_tag"] == ""), "enter_tag"] = "Extreme_Overbought"
            df.loc[any_short_signal & short_signal_hurst_mean_reversion & (df["enter_tag"] == ""), "enter_tag"] = "Hurst_Mean_Reversion_Short"
            df.loc[any_short_signal & short_signal_hurst_trend & (df["enter_tag"] == ""), "enter_tag"] = "Hurst_Trend_Short"
        
        # Add re-entry suffix
        if self.enable_reentry.value and hasattr(self, 'reentry_count') and pair in self.reentry_count:
            total_reentries = self.reentry_count[pair]['long'] + self.reentry_count[pair]['short']
            if total_reentries > 0:
                reentry_suffix = f"_RE{total_reentries}"
                mask = df["enter_tag"] != ""
                df.loc[mask, "enter_tag"] = df.loc[mask, "enter_tag"] + reentry_suffix
        
        # Debug logging
        logger.info(f"{pair} 🎯 FINAL SIGNALS: Long={any_long_signal.sum()}, Short={any_short_signal.sum()}")
        
        # ===========================================
        # FINAL VALIDATION & STATISTICS
        # ===========================================
        final_long_count = any_long_signal.sum()
        final_short_count = any_short_signal.sum()

        if final_long_count > 0:
            signal_breakdown = {
                'support_bounce': long_signal_support_bounce.sum(),
                '50_reclaim': long_signal_50_reclaim.sum(),
                'range_long': long_signal_range_long.sum(),
                'trend_continuation': long_signal_trend_continuation.sum(),
                'extreme_oversold': long_signal_extreme_oversold.sum(),
                'hurst_mean_reversion': long_signal_hurst_mean_reversion.sum(),
                'hurst_trend': long_signal_hurst_trend.sum()
            }
            
            logger.info(f"{pair} 📋 LONG SIGNAL BREAKDOWN:")
            for signal_type, count in signal_breakdown.items():
                if count > 0:
                    logger.info(f"   {signal_type}: {count}")

        if final_short_count > 0 and self.can_short:
            short_breakdown = {
                'bearish_breakdown': short_signal_bearish_breakdown.sum(),
                'resistance_reject': short_signal_resistance_reject.sum(),
                '50_breakdown': short_signal_50_breakdown.sum(),
                'range_short': short_signal_range_short.sum(),
                'trend_continuation': short_signal_trend_continuation.sum(),
                'extreme_overbought': short_signal_extreme_overbought.sum(),
                'hurst_mean_reversion': short_signal_hurst_mean_reversion.sum(),
                'hurst_trend': short_signal_hurst_trend.sum()
            }
            
            logger.info(f"{pair} 📋 SHORT SIGNAL BREAKDOWN:")
            for signal_type, count in short_breakdown.items():
                if count > 0:
                    logger.info(f"   {signal_type}: {count}")

        # Final safety check
        df["enter_long"] = df["enter_long"].fillna(0).astype(int)
        df["enter_short"] = df["enter_short"].fillna(0).astype(int)
        df["enter_tag"] = df["enter_tag"].fillna("")
        
        return df

    def calculate_mml_signal_strengths(self, df: pd.DataFrame, bullish_mml, bearish_mml,
                                     mml_support_bounce, mml_resistance_reject, range_bound) -> pd.DataFrame:
        """
        📊 Calculate signal strength for your MML signals
        """
        
        df['long_signal_strength'] = 0.0
        df['short_signal_strength'] = 0.0
        
        # ===========================================
        # LONG SIGNAL STRENGTH COMPONENTS
        # ===========================================
        
        long_strength_components = []
        
        # MML structure strength (40% weight)
        mml_long_strength = (
            bullish_mml.astype(float) * 0.4 +  # Strong bullish structure
            mml_support_bounce.astype(float) * 0.3 +  # Support bounce
            ((df["close"] > df["[4/8]P"]).astype(float) * 0.2) +  # Above midpoint
            ((df["close"] > df["[6/8]P"]).astype(float) * 0.1)    # Above 75%
        )
        long_strength_components.append(mml_long_strength.clip(0, 0.4))
        
        # RSI strength (20% weight)
        rsi_long_strength = (
            ((df["rsi"] < 30).astype(float) * 0.2) +  # Oversold
            ((df["rsi"] < 40).astype(float) * 0.15) +  # Getting oversold
            ((df["rsi"] > df["rsi"].shift(1)).astype(float) * 0.05)  # RSI improving
        ).clip(0, 0.2)
        long_strength_components.append(rsi_long_strength)
        
        # Volume strength (20% weight)
        volume_long_strength = (
            ((df["volume"] > df["volume"].rolling(20).mean() * 1.5).astype(float) * 0.2) +
            ((df["volume"] > df["volume"].rolling(10).mean()).astype(float) * 0.1)
        ).clip(0, 0.2)
        long_strength_components.append(volume_long_strength)
        
        # Momentum strength (20% weight)
        momentum_long_strength = (
            ((df["close"] > df["close"].shift(1)).astype(float) * 0.1) +
            ((df["close"] > df["open"]).astype(float) * 0.1)  # Green candle
        ).clip(0, 0.2)
        long_strength_components.append(momentum_long_strength)
        
        # ===========================================
        # SHORT SIGNAL STRENGTH COMPONENTS
        # ===========================================
        
        short_strength_components = []
        
        # MML structure strength (40% weight)
        mml_short_strength = (
            bearish_mml.astype(float) * 0.4 +  # Strong bearish structure
            mml_resistance_reject.astype(float) * 0.3 +  # Resistance rejection
            ((df["close"] < df["[4/8]P"]).astype(float) * 0.2) +  # Below midpoint
            ((df["close"] < df["[2/8]P"]).astype(float) * 0.1)    # Below 25%
        )
        short_strength_components.append(mml_short_strength.clip(0, 0.4))
        
        # RSI strength (20% weight)
        rsi_short_strength = (
            ((df["rsi"] > 70).astype(float) * 0.2) +  # Overbought
            ((df["rsi"] > 60).astype(float) * 0.15) +  # Getting overbought
            ((df["rsi"] < df["rsi"].shift(1)).astype(float) * 0.05)  # RSI declining
        ).clip(0, 0.2)
        short_strength_components.append(rsi_short_strength)
        
        # Volume strength (20% weight) - same as long
        short_strength_components.append(volume_long_strength)
        
        # Momentum strength (20% weight)
        momentum_short_strength = (
            ((df["close"] < df["close"].shift(1)).astype(float) * 0.1) +
            ((df["close"] < df["open"]).astype(float) * 0.1)  # Red candle
        ).clip(0, 0.2)
        short_strength_components.append(momentum_short_strength)
        
        # ===========================================
        # COMBINE STRENGTH COMPONENTS
        # ===========================================
        
        df['long_signal_strength'] = sum(long_strength_components).clip(0, 1)
        df['short_signal_strength'] = sum(short_strength_components).clip(0, 1)
        
        return df
    
    def apply_mml_reentry_logic(self, df: pd.DataFrame, pair: str, 
                               any_long_signal: pd.Series, any_short_signal: pd.Series) -> tuple:
        current_time = pd.Timestamp.now() if df.empty else pd.Timestamp(df.index[-1])
        if pair not in self.reentry_count:
            self.reentry_count[pair] = {'long': 0, 'short': 0}
        
        recent_exits = self.get_recent_exits(pair, current_time)
        reentry_long_mask = pd.Series([False] * len(df), index=df.index)
        reentry_short_mask = pd.Series([False] * len(df), index=df.index)

        for i in range(len(df)):
            current_candle_time = df.index[i]
            for direction in ['long', 'short']:
                if (self.should_attempt_reentry(pair, direction, current_candle_time, recent_exits) and 
                    self.reentry_count[pair][direction] < self.max_reentries_per_direction.value):
                    favorable = (df.iloc[i]['close'] > df.iloc[i]['[2/8]P'] if direction == 'long' 
                               else df.iloc[i]['close'] < df.iloc[i]['[6/8]P'])
                    strength_threshold = df.iloc[i][f'{direction}_signal_strength'] >= self.reentry_signal_strength_threshold.value
                    
                    if favorable and strength_threshold:
                        conditions = (
                            (df.iloc[i]['rsi'] < 45 if direction == 'long' else df.iloc[i]['rsi'] > 55) &
                            (df.iloc[i]['volume'] > df.iloc[i]['volume_avg_10']) &
                            (df.iloc[i]['close'] > df.iloc[i]['close'].shift(1) if direction == 'long' else 
                             df.iloc[i]['close'] < df.iloc[i]['close'].shift(1) if i > 0 else True)
                        )
                        if conditions:
                            if direction == 'long':
                                reentry_long_mask.iloc[i] = True
                                self.reentry_count[pair]['long'] += 1
                                logger.info(f"{pair} 🔄 STANDARD {direction.upper()} RE-ENTRY #{self.reentry_count[pair]['long']}")
                            else:
                                reentry_short_mask.iloc[i] = True
                                self.reentry_count[pair]['short'] += 1
                                logger.info(f"{pair} 🔄 STANDARD {direction.upper()} RE-ENTRY #{self.reentry_count[pair]['short']}")

        return any_long_signal | reentry_long_mask, any_short_signal | reentry_short_mask
    
    def set_enhanced_entry_tags(self, df: pd.DataFrame, final_long_signal, final_short_signal,
                               long_signal_mml_breakout, long_signal_support_bounce,
                               long_signal_50_reclaim, long_signal_sextrema,
                               long_signal_rolling_MinH2, short_signal_bearish_breakdown,
                               short_signal_confirmed_max_entry, short_signal_rolling_MaxH2_Entry,
                               pair: str) -> pd.DataFrame:
        """
        🏷️ Set enhanced entry tags that include re-entry information
        """
        
        # Your existing tag logic with re-entry enhancement
        df.loc[final_long_signal & long_signal_mml_breakout, "enter_tag"] = "MML_Bullish_Breakout"
        df.loc[final_long_signal & long_signal_support_bounce & (df["enter_tag"] == ""), "enter_tag"] = "MML_Support_Bounce"
        df.loc[final_long_signal & long_signal_50_reclaim & (df["enter_tag"] == ""), "enter_tag"] = "MML_50_Reclaim"
        df.loc[final_long_signal & long_signal_rolling_MinH2 & (df["enter_tag"] == ""), "enter_tag"] = "Rolling_MinH2"
        df.loc[final_long_signal & long_signal_sextrema & (df["enter_tag"] == ""), "enter_tag"] = "Sextrema"
        
        if self.can_short:
            df.loc[final_short_signal & short_signal_bearish_breakdown, "enter_tag"] = "MML_Bearish_Breakdown"
            df.loc[final_short_signal & short_signal_rolling_MaxH2_Entry & (df["enter_tag"] == ""), "enter_tag"] = "Rolling_MaxH2_Short"
            df.loc[final_short_signal & short_signal_confirmed_max_entry & (df["enter_tag"] == ""), "enter_tag"] = "Confirmed_Max_Entry_Short"
        
        # Add re-entry indicators to tags
        if pair in self.reentry_count:
            reentry_suffix = ""
            if self.reentry_count[pair]['long'] > 0 or self.reentry_count[pair]['short'] > 0:
                total_reentries = self.reentry_count[pair]['long'] + self.reentry_count[pair]['short']
                reentry_suffix = f"_RE{total_reentries}"
            
            # Append re-entry info to existing tags
            df.loc[df["enter_tag"] != "", "enter_tag"] = df.loc[df["enter_tag"] != "", "enter_tag"] + reentry_suffix
        
        return df
    
    # ===========================================
    # 🛠️ HELPER FUNCTIONS (same as before)
    # ===========================================
    
    def should_attempt_reentry(self, pair: str, direction: str, current_time: datetime, recent_exits: list) -> bool:
        """Check if we should attempt re-entry"""
        
        try:
            if self.reentry_count[pair][direction] >= self.max_reentries_per_direction.value:
                return False
            
            recent_exit_in_direction = None
            for exit_time, exit_direction, exit_reason, profit in recent_exits:
                if exit_direction == direction:
                    recent_exit_in_direction = (exit_time, exit_reason)
                    break
            
            if not recent_exit_in_direction:
                return False
            
            exit_time, exit_reason = recent_exit_in_direction
            
            if exit_reason == 'roi' and self.enable_quick_reentry_on_roi.value:
                cooldown = float(self.quick_reentry_cooldown.value)  # Ensure float
            else:
                cooldown = float(self.reentry_cooldown_minutes.value)  # Ensure float
            
            # Use your safe_time_diff function instead
            time_since_exit = self.safe_time_diff(current_time, exit_time)
            
            return time_since_exit >= cooldown
            
        except Exception as e:
            logger.warning(f"Re-entry check error for {pair} {direction}: {e}")
            return False  # Don't attempt re-entry if there's an error
    
    def was_recent_roi_exit(self, pair: str, direction: str, current_time: datetime, recent_exits: list) -> bool:
        """Check if there was a recent ROI exit"""
        
        try:
            for exit_time, exit_direction, exit_reason, profit in recent_exits:
                if (exit_direction == direction and 
                    exit_reason == 'roi'):
                    
                    # Use safe_time_diff instead of manual calculation
                    time_since_exit = self.safe_time_diff(current_time, exit_time)
                    
                    if time_since_exit <= float(self.quick_reentry_cooldown.value):
                        return True
            return False
            
        except Exception as e:
            logger.warning(f"Recent ROI check error for {pair} {direction}: {e}")
            return False
    
    def get_recent_exits(self, pair: str, current_time: datetime) -> list:
        """Get recent exits for this pair"""
        
        try:
            if not hasattr(self, 'recent_exits') or pair not in self.recent_exits:
                return []
            
            # Safe cutoff time calculation
            try:
                cutoff_time = pd.Timestamp(current_time) - pd.Timedelta(hours=1)
            except:
                cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=1)
            
            recent_exits = []
            for exit_time, direction, reason, profit in self.recent_exits[pair]:
                try:
                    # Safe time comparison using safe_time_diff
                    time_since_exit = self.safe_time_diff(current_time, exit_time)
                    
                    # If less than 60 minutes ago, include it
                    if time_since_exit <= 60:  # 60 minutes
                        recent_exits.append((exit_time, direction, reason, profit))
                        
                except Exception as e:
                    logger.warning(f"Exit time comparison error: {e}")
                    continue
            
            return recent_exits
            
        except Exception as e:
            logger.warning(f"Get recent exits error for {pair}: {e}")
            return []
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        UNIFIED EXIT SYSTEM - Choose between Custom MML Exits or Simple Opposite Signal Exits
        """
        # ===========================================
        # INITIALIZE EXIT COLUMNS
        # ===========================================
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""
        
        # ===========================================
        # CHOOSE EXIT SYSTEM
        # ===========================================
        if self.use_custom_exits_advanced:
            # Use Alex's Advanced MML-based Exit System
            return self._populate_custom_exits_advanced(dataframe, metadata)
        else:
            # Use Simple Opposite Signal Exit System
            return self._populate_simple_exits(dataframe, metadata)
    
    def _populate_custom_exits_advanced(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        ALEX'S ADVANCED MML-BASED EXIT SYSTEM
        Profit-protecting exit strategy with Hurst Exponent-based signals
        """
        pair = metadata['pair']
        
        # ===========================================
        # MML MARKET STRUCTURE FOR EXITS
        # ===========================================
        bullish_mml = (
            (df["close"] > df["[6/8]P"]) |
            ((df["close"] > df["[4/8]P"]) & (df["close"].shift(5) < df["[4/8]P"].shift(5)))
        )
        
        bearish_mml = (
            (df["close"] < df["[2/8]P"]) |
            ((df["close"] < df["[4/8]P"]) & (df["close"].shift(5) > df["[4/8]P"].shift(5)))
        )
        
        at_resistance = (
            (df["high"] >= df["[6/8]P"]) |
            (df["high"] >= df["[7/8]P"]) |
            (df["high"] >= df["[8/8]P"])
        )
        
        at_support = (
            (df["low"] <= df["[2/8]P"]) |
            (df["low"] <= df["[1/8]P"]) |
            (df["low"] <= df["[0/8]P"])
        )
        
        # ===========================================
        # LONG EXIT SIGNALS
        # ===========================================
        logger.info(f"{pair} Data types - close: {df['close'].dtype}, high: {df['high'].dtype}, atr: {df['atr'].dtype}")
        logger.info(f"{pair} Sample values - close: {df['close'].iloc[-1]:.6f}, high: {df['high'].iloc[-1]:.6f}, atr: {df['atr'].iloc[-1]:.6f}")
        
        # 1. Profit-Taking Exits
        atr_adjusted = df["atr"].fillna(0).astype(float) * 2
        trailing_stop_level = df["high"] - atr_adjusted
        long_exit_resistance_profit = (
            at_resistance &
            (df["close"] < df["high"]) &
            (df["rsi"] > 60) &
            (df["maxima"] == 1) &
            (df["volume"] > df["volume"].rolling(10).mean()) &
            (df["close"] < trailing_stop_level) &
            (df["market_score"] > 0.3) &
            (df["consecutive_red"] >= 1)
        )
        
        long_exit_extreme_overbought = (
            (df["close"] > df["[7/8]P"]) &
            (df["rsi"] > 75) &
            (df["close"] < df["close"].shift(1)) &
            (df["maxima"] == 1)
        )
        
        long_exit_volume_exhaustion = (
            at_resistance &
            (df["volume"] < df["volume"].rolling(20).mean() * 0.8) &
            (df["rsi"] > 70) &
            (df["close"] < df["close"].shift(1))
        )
        
        # 2. Structure Breakdown
        long_exit_structure_breakdown = (
            (df["close"] < df["[4/8]P"]) &
            (df["close"].shift(1) >= df["[4/8]P"].shift(1)) &
            bullish_mml.shift(1) &
            (df["close"] < df["[4/8]P"] * 0.995) &
            (df["close"] < df["close"].shift(1)) &
            (df["close"] < df["close"].shift(2)) &
            (df["rsi"] < 50) &
            (df["volume"] > df["volume"].rolling(15).mean() * 1.2) &
            (df["close"] < df["open"]) &
            (df["low"] < df["low"].shift(1)) &
            (df["close"] < df["close"].rolling(3).mean()) &
            (df["atr"] > df["atr"].rolling(10).mean()) &
            (df["market_regime"] != 'bull_run')
        )
        
        # 3. Momentum Divergence
        long_exit_momentum_divergence = (
            at_resistance &
            (df["rsi"] < df["rsi"].shift(1)) &
            (df["rsi"].shift(1) < df["rsi"].shift(2)) &
            (df["rsi"] < df["rsi"].shift(3)) &
            (df["close"] >= df["close"].shift(1)) &
            (df["maxima"] == 1) &
            (df["rsi"] > 60) &
            (df["consecutive_red"].shift(1) >= 1)
        )
        
        # 4. Range Exit
        long_exit_range = (
            (df["close"] >= df["[2/8]P"]) &
            (df["close"] <= df["[6/8]P"]) &
            (df["high"] >= df["[6/8]P"]) &
            (df["close"] < df["[6/8]P"] * 0.995) &
            (df["rsi"] > 65) &
            (df["maxima"] == 1) &
            (df["volume"] > df["volume"].rolling(10).mean() * 1.2)
        )
        
        # 5. Emergency Exit
        long_exit_emergency = (
            (df["close"] < df["[0/8]P"]) &
            (df["rsi"] < 25) &
            (df["volume"] > df["volume"].rolling(20).mean() * 2) &
            (df["close"] < df["close"].shift(1)) &
            (df["close"] < df["close"].shift(2)) &
            (df["close"] < df["open"])
        ) if self.use_emergency_exits else pd.Series([False] * len(df), index=df.index)
        
        # 6. Hurst-Based Long Exits
        long_exit_hurst_mean_reversion = (
            (df["mean_reversion_bear"] == 1) &
            (df["rsi"] > 55) &
            (df["close"] < df["close"].shift(1)) &
            (df["volume"] > df["volume"].rolling(10).mean() * 1.1) &
            (df["market_score"] < 0.8) &
            (df["consecutive_red"] >= 1)
        )
        
        long_exit_hurst_trend = (
            (df["hurst_trend_bear"] == 1) &
            (df["rsi"] < 50) &
            (df["close"] < df["ema20"]) &
            (df["volume"] > df["volume"].rolling(15).mean() * 1.1) &
            (df["market_score"] < 0.8) &
            (df["market_regime"] != 'bull_run')
        )
        
        # Combine all Long Exit signals
        any_long_exit = (
            long_exit_resistance_profit |
            long_exit_extreme_overbought |
            long_exit_volume_exhaustion |
            long_exit_structure_breakdown |
            long_exit_momentum_divergence |
            long_exit_range |
            long_exit_emergency |
            long_exit_hurst_mean_reversion |
            long_exit_hurst_trend
        )
        
        # ===========================================
        # SHORT EXIT SIGNALS
        # ===========================================
        if self.can_short:
            # 1. Profit-Taking Exits
            atr_adjusted_short = df["atr"].fillna(0).astype(float) * 2
            trailing_stop_level_short = df["low"] + atr_adjusted_short
            short_exit_support_profit = (
                at_support &
                (df["close"] > df["low"]) &
                (df["rsi"] < 40) &
                (df["minima"] == 1) &
                (df["volume"] > df["volume"].rolling(10).mean()) &
                (df["close"] > (df["low"] + df["atr"].fillna(0).astype(float) * 1.5)) &
                (df["market_score"] < 0.7) &
                (~(df["consecutive_red"] >= 1)) &
                (df["volume"] < df["volume"].rolling(20).mean() * 0.8)
            )
            
            short_exit_extreme_oversold = (
                (df["close"] < df["[1/8]P"]) &
                (df["rsi"] < 25) &
                (df["close"] > df["close"].shift(1)) &
                (df["minima"] == 1)
            )
            
            short_exit_volume_exhaustion = (
                at_support &
                (df["volume"] < df["volume"].rolling(20).mean() * 0.8) &
                (df["rsi"] < 30) &
                (df["close"] > df["close"].shift(1))
            )
            
            # 2. Structure Breakout
            short_exit_structure_breakout = (
                (df["close"] > df["[4/8]P"]) &
                (df["close"].shift(1) <= df["[4/8]P"].shift(1)) &
                bearish_mml.shift(1) &
                (df["close"] > df["[4/8]P"] * 1.005) &
                (df["close"] > df["close"].shift(1)) &
                (df["close"] > df["close"].shift(2)) &
                (df["rsi"] > 50) &
                (df["volume"] > df["volume"].rolling(15).mean() * 1.2) &
                (df["close"] > df["open"]) &
                (df["high"] > df["high"].shift(1)) &
                (df["atr"] > df["atr"].rolling(10).mean()) &
                (df["market_regime"] != 'bear_market')
            )
            
            # 3. Momentum Divergence
            short_exit_momentum_divergence = (
                at_support &
                (df["rsi"] > df["rsi"].shift(1)) &
                (df["rsi"].shift(1) > df["rsi"].shift(2)) &
                (df["rsi"] > df["rsi"].shift(3)) &
                (df["close"] <= df["close"].shift(1)) &
                (df["minima"] == 1) &
                (df["rsi"] < 40)
            )
            
            # 4. Range Exit
            short_exit_range = (
                (df["close"] >= df["[2/8]P"]) &
                (df["close"] <= df["[6/8]P"]) &
                (df["low"] <= df["[2/8]P"]) &
                (df["close"] > df["[2/8]P"] * 1.005) &
                (df["rsi"] < 35) &
                (df["minima"] == 1) &
                (df["volume"] > df["volume"].rolling(10).mean() * 1.2)
            )
            
            # 5. Emergency Exit
            short_exit_emergency = (
                (df["close"] > df["[8/8]P"]) &
                (df["rsi"] > 85) &
                (df["volume"] > df["volume"].rolling(20).mean() * 2) &
                (df["close"] > df["close"].shift(1)) &
                (df["close"] > df["close"].shift(2)) &
                (df["close"] > df["open"])
            ) if self.use_emergency_exits else pd.Series([False] * len(df), index=df.index)
            
            # 6. Hurst-Based Short Exits
            short_exit_hurst_mean_reversion = (
                (df["mean_reversion_cp_bull"] == 1) &
                (df["rsi"] < 45) &
                (df["close"] > df["close"].shift(1)) &
                (df["volume"] > df["volume"].rolling(10).mean() * 1.1) &
                (df["market_score"] > 0.2) &
                (~(df["consecutive_red"] >= 1))
            )
            
            short_exit_hurst_trend = (
                (df["hurst_trend_cp_bull"] == 1) &
                (df["rsi"] > 50) &
                (df["close"] > df["ema20"]) &
                (df["volume"] > df["volume"].rolling(15).mean() * 1.1) &
                (df["market_score"] > 0.2) &
                (df["market_regime"] != 'bear_market')
            )
            
            # Combine all Short Exit signals
            any_short_exit = (
                short_exit_support_profit |
                short_exit_extreme_oversold |
                short_exit_volume_exhaustion |
                short_exit_structure_breakout |
                short_exit_momentum_divergence |
                short_exit_range |
                short_exit_emergency |
                short_exit_hurst_mean_reversion |
                short_exit_hurst_trend
            )
        else:
            any_short_exit = pd.Series([False] * len(df), index=df.index)
        
        # ===========================================
        # COORDINATION WITH ENTRY SIGNALS
        # ===========================================
        if 'enter_long' in df.columns and 'enter_short' in df.columns:
            df.loc[df['enter_short'] == 1, 'exit_long'] = 1
            # df.loc[df['enter_long'] == 1, 'exit_short'] = 1
            df.loc[df['exit_long'] == 1, 'exit_tag'] = 'Reversal_Short_Entry'
            if self.can_short:
                df.loc[df['exit_short'] == 1, 'exit_tag'] = 'Reversal_Long_Entry'
        else:
            logger.warning(f"{pair} Missing 'enter_long' or 'enter_short' columns; skipping reversal-based exits")
        
        # ===========================================
        # SET FINAL EXIT SIGNALS AND TAGS
        # ===========================================
        df.loc[any_long_exit, "exit_long"] = 1
        if self.can_short:
            df.loc[any_short_exit, "exit_short"] = 1
        
        # Exit tags
        df.loc[any_long_exit & long_exit_emergency, "exit_tag"] = "MML_Emergency_Long_Exit"
        df.loc[any_long_exit & long_exit_structure_breakdown & (df["exit_tag"] == ""), "exit_tag"] = "MML_Structure_Breakdown_Confirmed"
        df.loc[any_long_exit & long_exit_resistance_profit & (df["exit_tag"] == ""), "exit_tag"] = "MML_Resistance_Profit"
        df.loc[any_long_exit & long_exit_extreme_overbought & (df["exit_tag"] == ""), "exit_tag"] = "MML_Extreme_Overbought"
        df.loc[any_long_exit & long_exit_volume_exhaustion & (df["exit_tag"] == ""), "exit_tag"] = "MML_Volume_Exhaustion_Long"
        df.loc[any_long_exit & long_exit_momentum_divergence & (df["exit_tag"] == ""), "exit_tag"] = "MML_Momentum_Divergence_Long"
        df.loc[any_long_exit & long_exit_range & (df["exit_tag"] == ""), "exit_tag"] = "MML_Range_Exit_Long"
        df.loc[any_long_exit & long_exit_hurst_mean_reversion & (df["exit_tag"] == ""), "exit_tag"] = "Hurst_Mean_Reversion_Long_Exit"
        df.loc[any_long_exit & long_exit_hurst_trend & (df["exit_tag"] == ""), "exit_tag"] = "Hurst_Trend_Long_Exit"
        
        if self.can_short:
            df.loc[any_short_exit & short_exit_emergency, "exit_tag"] = "MML_Emergency_Short_Exit"
            df.loc[any_short_exit & short_exit_structure_breakout & (df["exit_tag"] == ""), "exit_tag"] = "MML_Structure_Breakout_Confirmed"
            df.loc[any_short_exit & short_exit_support_profit & (df["exit_tag"] == ""), "exit_tag"] = "MML_Support_Profit"
            df.loc[any_short_exit & short_exit_extreme_oversold & (df["exit_tag"] == ""), "exit_tag"] = "MML_Extreme_Oversold"
            df.loc[any_short_exit & short_exit_volume_exhaustion & (df["exit_tag"] == ""), "exit_tag"] = "MML_Volume_Exhaustion_Short"
            df.loc[any_short_exit & short_exit_momentum_divergence & (df["exit_tag"] == ""), "exit_tag"] = "MML_Momentum_Divergence_Short"
            df.loc[any_short_exit & short_exit_range & (df["exit_tag"] == ""), "exit_tag"] = "MML_Range_Exit_Short"
            df.loc[any_short_exit & short_exit_hurst_mean_reversion & (df["exit_tag"] == ""), "exit_tag"] = "Hurst_Mean_Reversion_Short_Exit"
            df.loc[any_short_exit & short_exit_hurst_trend & (df["exit_tag"] == ""), "exit_tag"] = "Hurst_Trend_Short_Exit"
        
        # ===========================================
        # FINAL VALIDATION & LOGGING
        # ===========================================
        final_long_exit_count = any_long_exit.sum()
        final_short_exit_count = any_short_exit.sum()
        
        if final_long_exit_count > 0:
            long_exit_breakdown = {
                'resistance_profit': long_exit_resistance_profit.sum(),
                'extreme_overbought': long_exit_extreme_overbought.sum(),
                'volume_exhaustion': long_exit_volume_exhaustion.sum(),
                'structure_breakdown': long_exit_structure_breakdown.sum(),
                'momentum_divergence': long_exit_momentum_divergence.sum(),
                'range_exit': long_exit_range.sum(),
                'emergency_exit': long_exit_emergency.sum(),
                'hurst_mean_reversion': long_exit_hurst_mean_reversion.sum(),
                'hurst_trend': long_exit_hurst_trend.sum()
            }
            logger.info(f"{pair} 📋 LONG EXIT BREAKDOWN:")
            for exit_type, count in long_exit_breakdown.items():
                if count > 0:
                    logger.info(f"   {exit_type}: {count}")
        
        if final_short_exit_count > 0 and self.can_short:
            short_exit_breakdown = {
                'support_profit': short_exit_support_profit.sum(),
                'extreme_oversold': short_exit_extreme_oversold.sum(),
                'volume_exhaustion': short_exit_volume_exhaustion.sum(),
                'structure_breakout': short_exit_structure_breakout.sum(),
                'momentum_divergence': short_exit_momentum_divergence.sum(),
                'range_exit': short_exit_range.sum(),
                'emergency_exit': short_exit_emergency.sum(),
                'hurst_mean_reversion': short_exit_hurst_mean_reversion.sum(),
                'hurst_trend': short_exit_hurst_trend.sum()
            }
            logger.info(f"{pair} 📋 SHORT EXIT BREAKDOWN:")
            for exit_type, count in short_exit_breakdown.items():
                if count > 0:
                    logger.info(f"   {exit_type}: {count}")
        
        logger.info(f"{pair} 🎯 FINAL EXITS: Long={final_long_exit_count}, Short={final_short_exit_count}")
        
        # Final safety check
        df["exit_long"] = df["exit_long"].fillna(0).astype(int)
        df["exit_short"] = df["exit_short"].fillna(0).astype(int)
        df["exit_tag"] = df["exit_tag"].fillna("")
        
        return df

    
    def _populate_simple_exits(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        SIMPLE OPPOSITE SIGNAL EXIT SYSTEM - SYNTAX FIXED
        """
        
        # Exit LONG when any SHORT signal appears
        long_exit_on_short = (dataframe["enter_short"] == 1)
        
        # Exit SHORT when any LONG signal appears  
        short_exit_on_long = (dataframe["enter_long"] == 1)
        
        # Emergency exits (if enabled)
        if self.use_emergency_exits:
            emergency_long_exit = (
                (dataframe['rsi'] > 85) &
                (dataframe['volume'] > dataframe['avg_volume'] * 3) &
                (dataframe['close'] < dataframe['open']) &
                (dataframe['close'] < dataframe['low'].shift(1))
            ) | (
                (dataframe.get('structure_break_down', 0) == 1) &
                (dataframe['volume'] > dataframe['avg_volume'] * 2.5) &
                (dataframe['atr'] > dataframe['atr'].rolling(20).mean() * 2)
            )
            
            emergency_short_exit = (
                (dataframe['rsi'] < 15) &
                (dataframe['volume'] > dataframe['avg_volume'] * 3) &
                (dataframe['close'] > dataframe['open']) &
                (dataframe['close'] > dataframe['high'].shift(1))
            ) | (
                (dataframe.get('structure_break_up', 0) == 1) &
                (dataframe['volume'] > dataframe['avg_volume'] * 2.5) &
                (dataframe['atr'] > dataframe['atr'].rolling(20).mean() * 2)
            )
        else:
            emergency_long_exit = pd.Series([False] * len(dataframe), index=dataframe.index)
            emergency_short_exit = pd.Series([False] * len(dataframe), index=dataframe.index)
        
    
        
        # DEBUGGING (FIXED THE ERROR HERE)
        if metadata['pair'] in ['BTC/USDT', 'ETH/USDT']:
            recent_exits = dataframe['exit_long'].tail(5).sum() + dataframe['exit_short'].tail(5).sum()
            if recent_exits > 0:
                exit_tag = dataframe['exit_tag'].iloc[-1]
                logger.info(f"{metadata['pair']} EXIT SIGNAL - Tag: {exit_tag}")
                # ✅ FIXED: Use the correct attribute name
                logger.info(f"  Exit System: {'Custom MML' if self.use_custom_exits_advanced else 'Simple Opposite'}")
                logger.info(f"  RSI: {dataframe['rsi'].iloc[-1]:.1f}")
        
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, 
                           rate: float, time_in_force: str, exit_reason: str, 
                           current_time: datetime, **kwargs) -> bool:
        
        current_profit_ratio = trade.calc_profit_ratio(rate)
        time_in_trade = (current_time - trade.open_date_utc).total_seconds() / 3600
        
        logger.warning(f"🚪 {pair} EXIT REQUEST: {exit_reason}")
        logger.warning(f"   💰 Profit: {current_profit_ratio:.4f} ({current_profit_ratio*100:.2f}%)")
        logger.warning(f"   ⏰ Time in trade: {time_in_trade:.1f} hours")
        
        # 🛑 BLOCK FORCE EXITS
        if exit_reason in ["force_exit", "force_sell", "emergency_exit"]:
            logger.warning(f"{pair} 🛑 BLOCKED FORCE EXIT: {exit_reason} (Profit: {current_profit_ratio:.2%})")
            return False
        
        # 🛑 BLOCK PROTECTION EXITS
        if exit_reason in ["timeout", "protection", "cooldown"]:
            logger.warning(f"{pair} 🛑 BLOCKED PROTECTION EXIT: {exit_reason}")
            return False
        
        # 🛑 BLOCK TRAILING STOPS (NEW!)
        if exit_reason in ["trailing_stop_loss", "trailing_stop"]:
            logger.warning(f"{pair} 🛑 BLOCKED TRAILING STOP: {exit_reason} (Profit: {current_profit_ratio:.2%})")
            return False
        
        # ✅ ALLOW ROI EXITS
        if exit_reason == "roi":
            logger.info(f"{pair} ✅ ROI EXIT: {current_profit_ratio:.2%} after {time_in_trade:.1f}h")
            return True
        
        # ✅ ALLOW OTHER LEGITIMATE EXITS
        if exit_reason in ["stop_loss", "exit_signal", "sell_signal", "custom_exit"]:
            logger.info(f"{pair} ✅ EXIT: {exit_reason} ({current_profit_ratio:.2%})")
            return True
        
        # ✅ DEFAULT ALLOW (for any other exit reasons)
        logger.info(f"{pair} ✅ FINAL CHECK PASSED: Profit {current_profit_ratio:.2%}")
        return True
    # 🔍 ADD THIS METHOD HERE (anywhere in the class)
    
    def should_exit_early_warning(self, pair: str, trade: Trade, current_time: datetime,
                                  current_rate: float, current_profit: float) -> bool:
        """
        ⚠️ EARLY WARNING: Exit before stop loss if conditions deteriorate
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return False
            
            last_candle = dataframe.iloc[-1]
            time_in_trade = (current_time - trade.open_date_utc).total_seconds() / 3600
            
            # Don't exit too early
            if time_in_trade < 0.25:  # Give trade at least 15 minutes
                return False
            
            # Critical deterioration signals
            rsi = last_candle.get('rsi', 50)
            
            if trade.is_short:
                # Short position warning signs
                warning_signals = (
                    (rsi < 30 and current_profit < -0.02) or  # RSI turning bullish while losing
                    (last_candle.get('close', 0) > last_candle.get('[6/8]P', 0)) or  # Breaking above 75%
                    (current_profit < -0.04 and time_in_trade > 2)  # Deep loss after 2h
                )
            else:
                # Long position warning signs  
                warning_signals = (
                    (rsi > 70 and current_profit < -0.02) or  # RSI turning bearish while losing
                    (last_candle.get('close', 0) < last_candle.get('[2/8]P', 0)) or  # Breaking below 25%
                    (current_profit < -0.04 and time_in_trade > 2)  # Deep loss after 2h
                )
            
            if warning_signals:
                logger.warning(f"⚠️ {pair} EARLY WARNING: Deteriorating conditions detected")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Early warning error for {pair}: {e}")
            return False
            


    def validate_backtest_conditions(self, dataframe: pd.DataFrame, metadata: dict) -> None:
        """
        📊 VALIDATION: Check if backtest conditions match live trading
        """
        
        pair = metadata['pair']
        
        # Check BTC correlation status
        if 'btc_status' in dataframe.columns:
            btc_status_counts = dataframe['btc_status'].value_counts()
            total_candles = len(dataframe)
            
            logger.warning(f"{pair} 📊 BACKTEST VALIDATION REPORT:")
            logger.warning(f"   Total candles: {total_candles}")
            
            for status, count in btc_status_counts.items():
                percentage = (count / total_candles) * 100
                logger.warning(f"   BTC Status '{status}': {count} candles ({percentage:.1f}%)")
            
            # Alert if significant portion has no BTC data
            no_data_percentage = btc_status_counts.get('NO_DATA', 0) / total_candles * 100
            if no_data_percentage > 10:
                logger.error(f"{pair} 🚨 WARNING: {no_data_percentage:.1f}% of backtest has no BTC data!")
                logger.error(f"{pair} 🚨 BACKTEST RESULTS MAY NOT MATCH LIVE TRADING!")
            
            # Count signal reduction due to BTC filter
            if 'btc_correlation_ok' in df.columns:
                btc_ok = df['btc_correlation_ok'].fillna(True).astype(bool)
                # any_long_signal = any_long_signal & btc_ok  # Commented out to disable BTC correlation filter
                # any_short_signal = any_short_signal & btc_ok  # Commented out to disable BTC correlation filter
                logger.info(f"{pair} ₿ BTC correlation: {'✅' if btc_ok.iloc[-1] else '❌'}")
        
        else:
            logger.error(f"{pair} 🚨 NO BTC CORRELATION DATA IN BACKTEST!")
            logger.error(f"{pair} 🚨 BACKTEST IS DEFINITELY NOT MATCHING LIVE CONDITIONS!")

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, 
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """
        🚪 Smart exit logic to prevent force exits
        """
        
        # Calculate how long trade has been running
        trade_duration_hours = (current_time - trade.open_date_utc).total_seconds() / 3600
        
        # Get entry tag for smart handling
        entry_tag = getattr(trade, 'enter_tag', '') or ''
        
        # 🎯 PREVENT LONG-RUNNING TRADES THAT BECOME FORCE EXITS
        
        # MML_Breakout_75 - these are getting force exited at -33%
        if 'MML_Breakout_75' in entry_tag:
            # Exit after 7 days if still negative to prevent worse force exits
            if trade_duration_hours > (7 * 24) and current_profit < -0.05:  # -5%
                logger.info(f"{pair} 🚪 Smart exit MML_Breakout_75: {current_profit:.2%} after {trade_duration_hours/24:.1f} days")
                return "smart_exit_mml75"
        
        # Bear_Breakdown_25 - these are better but still getting force exited
        elif 'Bear_Breakdown_25' in entry_tag:
            # Exit after 14 days if still negative
            if trade_duration_hours > (14 * 24) and current_profit < -0.02:  # -2%
                logger.info(f"{pair} 🚪 Smart exit Bear_Breakdown_25: {current_profit:.2%} after {trade_duration_hours/24:.1f} days")
                return "smart_exit_bear25"
        
        # General rule for any trade running over 21 days
        if trade_duration_hours > (21 * 24):
            logger.info(f"{pair} 🚪 Smart exit long trade: {current_profit:.2%} after {trade_duration_hours/24:.1f} days")
            return "smart_exit_timeout"
        
        return None

    def bot_loop_start(self, **kwargs) -> None:
        """
        🔍 Log any trades that might be candidates for force exit
        """
        try:
            trades = Trade.get_open_trades()
            current_time = datetime.now()
            
            for trade in trades:
                time_in_trade = (current_time - trade.open_date_utc).total_seconds() / 3600
                current_profit = trade.calc_profit_ratio(trade.close_rate) if trade.close_rate else 0
                
                # Warn about very old trades (might become force exits)
                if time_in_trade > 48:  # 2+ days
                    logger.warning(f"⚠️ OLD TRADE {trade.pair}: {time_in_trade:.1f}h old, "
                                 f"Profit: {current_profit:.2%}")
                    
                # Warn about large losses (might become force exits)  
                if current_profit < -0.10:  # >10% loss
                    logger.warning(f"⚠️ LARGE LOSS {trade.pair}: {current_profit:.2%}, "
                                 f"Time: {time_in_trade:.1f}h")
                    
        except Exception as e:
            # Don't let debugging break the bot
            logger.debug(f"bot_loop_start debug error: {e}")