import logging
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
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


class AlexBattleTankKiller(IStrategy):
    """
    Enhanced strategy on the 15-minute timeframe with Market Correlation Filters.

    Key improvements:
      - Dynamic stoploss based on ATR.
      - Dynamic leverage calculation.
      - Murrey Math level calculation (rolling window for performance).
      - Enhanced DCA (Average Price) logic.
      - Translated to English and code structured for clarity.
      - Parameterization of internal constants for optimization.
      - Changed Exit Signals for Opposite.
      - Change SL to -0.15
      - Changed stake amout for renentry
      - FIXED: Prevents opening opposite position when trade is active
      - FIXED: Trailing stop properly disabled
      - NEW: Market correlation filters for better entry timing
    """

    # General strategy parameters
    timeframe = "15m"
    startup_candle_count: int = 100
    stoploss = -0.30
    trailing_stop = False  # Explicitly disabled
    trailing_stop_positive = None  # Ensure no trailing stop
    trailing_stop_positive_offset = None  # Ensure no trailing stop
    trailing_only_offset_is_reached = None  # Ensure no trailing stop
    position_adjustment_enable = True
    can_short = False
    use_exit_signal = True
    ignore_roi_if_entry_signal = True

    max_entry_position_adjustment = 1
    process_only_new_candles = True

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
    leverage_base = DecimalParameter(5.0, 20.0, default=10.0, decimals=1, space="buy", optimize=True, load=True)
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
    btc_correlation_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    btc_correlation_threshold = DecimalParameter(0.3, 0.8, default=0.5, decimals=2, space="buy", optimize=True)
    btc_trend_filter_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    
    # Market breadth parameters
    market_breadth_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    market_breadth_threshold = DecimalParameter(0.3, 0.6, default=0.45, space="buy", optimize=True)
    
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

    # ROI table (minutes to decimal)
    minimal_roi = {
        "0": 0.25,      # 25% sofort (ROI war sehr erfolgreich)
        "15": 0.20,     # 20% nach 15min
        "30": 0.15,     # 15% nach 30min  
        "60": 0.12,     # 12% nach 1h
        "120": 0.10,    # 10% nach 2h
        "240": 0.08,    # 8% nach 4h
        "480": 0.06,    # 6% nach 8h
        "720": 0.05,    # 5% nach 12h
        "1440": 0.04,   # 4% nach 1 Tag
        "2880": 0.03,   # 3% nach 2 Tagen
        "4320": 0.02,   # 2% nach 3 Tagen
        "5760": 0.01,   # 1% nach 4 Tagen
    }
    # Plot configuration for backtesting UI
    plot_config = {
        "main_plot": {
            "btc_sma20": {"color": "orange"},
            "btc_sma50": {"color": "red"}
        },
        "subplots": {
            "extrema_analysis": {
                "s_extrema": {"color": "#f53580", "type": "line"},
                "minima_sort_threshold": {"color": "#4ae747", "type": "line"},
                "maxima_sort_threshold": {"color": "#5b5e4b", "type": "line"},
            },
            "min_max_viz": {
                "maxima": {"color": "#a29db9", "type": "line"},
                "minima": {"color": "#aac7fc", "type": "line"},
                "maxima_check": {"color": "#a29db9", "type": "line"},
                "minima_check": {"color": "#aac7fc", "type": "line"},
            },
            "murrey_math_levels": {
                "[4/8]P": {"color": "blue", "type": "line"},
                "[8/8]P": {"color": "red", "type": "line"},
                "[0/8]P": {"color": "red", "type": "line"},
            },
            "market_correlation": {
                "btc_correlation": {"color": "blue", "type": "line"},
                "market_breadth": {"color": "green", "type": "line"},
                "market_score": {"color": "purple", "type": "line"}
            },
            "market_regime": {
                "market_volatility": {"color": "red", "type": "line"},
                "mcap_trend": {"color": "yellow", "type": "line"}
            }
        },
    }

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
        OPTIMIERTE Version - nur alle 5 Candles berechnen und interpolieren
        FIXED: Pandas 2.0+ kompatibel
        """
        murrey_levels_data: Dict[str, list] = {key: [np.nan] * len(df) for key in MML_LEVEL_NAMES}
        rolling_high = df["high"].rolling(window=window_size, min_periods=window_size).max()
        rolling_low = df["low"].rolling(window=window_size, min_periods=window_size).min()
        mml_c1 = self.mml_const1.value
        mml_c2 = self.mml_const2.value
        
        # Nur alle 5 Candles berechnen für Performance
        calculation_step = 5
        
        for i in range(0, len(df), calculation_step):
            if i < window_size - 1:
                continue
                
            mn_period = rolling_low.iloc[i]
            mx_period = rolling_high.iloc[i]
            current_close = df["close"].iloc[i]
            
            if pd.isna(mn_period) or pd.isna(mx_period) or mn_period == mx_period:
                for key in MML_LEVEL_NAMES:
                    murrey_levels_data[key][i] = current_close
                continue
                
            levels = AlexBattleTankKiller._calculate_mml_core(mn_period, mx_period, mx_period, mn_period, mml_c1, mml_c2)
            
            for key in MML_LEVEL_NAMES:
                murrey_levels_data[key][i] = levels.get(key, current_close)
        
        # FIXED: Moderne pandas syntax
        for key in MML_LEVEL_NAMES:
            series = pd.Series(murrey_levels_data[key], index=df.index)
            series = series.interpolate(method='linear').bfill().ffill()  # FIXED
            murrey_levels_data[key] = series.tolist()
        
        return {key: pd.Series(data, index=df.index) for key, data in murrey_levels_data.items()}

    def calculate_market_correlation(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Calculate correlation with Bitcoin and overall market direction
        FIXED: Handles futures pairs (BTC/USDT:USDT format)
        """
        pair = metadata['pair']
        
        # Extract base currency from both spot and futures formats
        base_currency = pair.split('/')[0]
        if ':' in pair:
            # Futures format: BTC/USDT:USDT -> extract quote and settlement
            quote_part = pair.split('/')[1]
            quote_currency = quote_part.split(':')[0]
            settlement_currency = quote_part.split(':')[1]
            is_futures = True
        else:
            # Spot format: BTC/USDT
            quote_currency = pair.split('/')[1]
            settlement_currency = quote_currency
            is_futures = False
        
        # Skip if this IS BTC (no self-correlation)
        if base_currency == 'BTC':
            dataframe['btc_correlation'] = 1.0
            dataframe['btc_trend'] = 1
            dataframe['btc_close'] = dataframe['close']
            dataframe['btc_sma20'] = dataframe['close'].rolling(20).mean()
            dataframe['btc_sma50'] = dataframe['close'].rolling(50).mean()
            return dataframe
        
        # Try multiple BTC pair variations for futures
        btc_pairs_to_try = []
        
        if is_futures:
            # For futures, try futures format first
            btc_pairs_to_try.extend([
                f"BTC/{quote_currency}:{settlement_currency}",  # Same format as input
                f"BTC/USDT:USDT",  # Most common futures
                f"BTC/BUSD:BUSD",  # Binance futures alternative
            ])
        
        # Also try spot pairs as fallback (correlation should be similar)
        btc_pairs_to_try.extend([
            f"BTC/{quote_currency}",  # Spot equivalent
            "BTC/USDT",               # Most common spot
            "BTC/USD",                # Alternative
            "BTC/BUSD",               # Binance
        ])
        
        btc_dataframe = None
        btc_pair_found = None
        
        # Try each BTC pair until we find one with data
        for btc_pair in btc_pairs_to_try:
            try:
                btc_dataframe, _ = self.dp.get_analyzed_dataframe(btc_pair, self.timeframe)
                if btc_dataframe is not None and not btc_dataframe.empty and len(btc_dataframe) >= 100:
                    btc_pair_found = btc_pair
                    logger.info(f"Using {btc_pair} for BTC correlation data with {pair}")
                    break
            except Exception as e:
                logger.debug(f"Could not get data for {btc_pair}: {e}")
                continue
        
        # If still no BTC data, disable correlation features
        if btc_dataframe is None or btc_dataframe.empty:
            logger.warning(f"No BTC data available for correlation. Disabling correlation features for {pair}")
            logger.warning(f"Tried pairs: {btc_pairs_to_try}")
            # Set neutral values that won't block trades
            dataframe['btc_correlation'] = 0.5  # Neutral
            dataframe['btc_trend'] = 0          # Neutral
            dataframe['btc_close'] = dataframe['close']
            dataframe['btc_sma20'] = dataframe['close']
            dataframe['btc_sma50'] = dataframe['close']
            dataframe['returns'] = dataframe['close'].pct_change()
            return dataframe
        
        # Calculate rolling correlation
        correlation_period = min(24, len(dataframe) // 4)  # Adaptive period
        
        # Ensure we have matching timestamps
        dataframe['returns'] = dataframe['close'].pct_change()
        btc_dataframe['btc_returns'] = btc_dataframe['close'].pct_change()
        
        # Method 1: Try merge on date
        try:
            merged = pd.merge(
                dataframe[['date', 'returns']], 
                btc_dataframe[['date', 'btc_returns']], 
                on='date', 
                how='left'
            )
            
            # Calculate rolling correlation
            dataframe['btc_correlation'] = merged['returns'].rolling(
                window=correlation_period, min_periods=correlation_period // 2
            ).corr(merged['btc_returns'])
            
        except Exception as e:
            logger.warning(f"Date merge failed, using index alignment: {e}")
            # Method 2: Align by index position
            min_len = min(len(dataframe), len(btc_dataframe))
            dataframe.loc[dataframe.index[:min_len], 'btc_correlation'] = (
                dataframe['returns'][:min_len].rolling(
                    window=correlation_period, min_periods=correlation_period // 2
                ).corr(btc_dataframe['btc_returns'][:min_len])
            )
        
        # Calculate BTC trend
        btc_dataframe['btc_sma20'] = ta.SMA(btc_dataframe['close'], timeperiod=20)
        btc_dataframe['btc_sma50'] = ta.SMA(btc_dataframe['close'], timeperiod=50)
        
        # Get latest BTC trend data
        if len(btc_dataframe) > 0:
            latest_btc_close = btc_dataframe['close'].iloc[-1]
            latest_btc_sma20 = btc_dataframe['btc_sma20'].iloc[-1]
            latest_btc_sma50 = btc_dataframe['btc_sma50'].iloc[-1]
            
            # Simple trend determination
            if latest_btc_close > latest_btc_sma20 > latest_btc_sma50:
                btc_trend_value = 1  # Uptrend
            elif latest_btc_close < latest_btc_sma20 < latest_btc_sma50:
                btc_trend_value = -1  # Downtrend
            else:
                btc_trend_value = 0  # Neutral
            
            dataframe['btc_trend'] = btc_trend_value
            dataframe['btc_close'] = latest_btc_close
            dataframe['btc_sma20'] = latest_btc_sma20
            dataframe['btc_sma50'] = latest_btc_sma50
        else:
            dataframe['btc_trend'] = 0
            dataframe['btc_close'] = dataframe['close']
            dataframe['btc_sma20'] = dataframe['close']
            dataframe['btc_sma50'] = dataframe['close']
        
        # Fill NaN values
        dataframe['btc_correlation'] = dataframe['btc_correlation'].fillna(0.5)
        dataframe['btc_trend'] = dataframe['btc_trend'].fillna(0)
        
        return dataframe

    def calculate_market_breadth(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Calculate market breadth using top cryptocurrencies
        FIXED: Better thresholds and debugging
        """
        # Detect if we're in futures mode
        current_pair = metadata['pair']
        is_futures = ':' in current_pair
        
        if is_futures:
            # Extract settlement currency
            settlement = current_pair.split(':')[1]
            # Define top futures pairs
            top_pairs = [
                f"BTC/USDT:{settlement}", f"ETH/USDT:{settlement}", 
                f"BNB/USDT:{settlement}", f"SOL/USDT:{settlement}", 
                f"ADA/USDT:{settlement}", f"AVAX/USDT:{settlement}", 
                f"DOT/USDT:{settlement}", f"MATIC/USDT:{settlement}"
            ]
        else:
            # Original spot pairs
            top_pairs = [
                "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", 
                "ADA/USDT", "AVAX/USDT", "XRP/USDT", "SUI/USDT"
            ]
        
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0  # NEU: Zähle auch neutrale
        total_checked = 0
        
        for check_pair in top_pairs:
            try:
                pair_data, _ = self.dp.get_analyzed_dataframe(check_pair, self.timeframe)
                if pair_data.empty or len(pair_data) < 50:
                    # If futures pair not found, try spot equivalent
                    if is_futures and ':' in check_pair:
                        spot_pair = check_pair.split(':')[0]
                        pair_data, _ = self.dp.get_analyzed_dataframe(spot_pair, self.timeframe)
                        if pair_data.empty or len(pair_data) < 50:
                            continue
                    else:
                        continue
                
                # Simple trend check: price above/below SMA20
                current_close = pair_data['close'].iloc[-1]
                sma20 = pair_data['close'].rolling(20).mean().iloc[-1]
                
                # GEÄNDERT: Kleinere Thresholds (0.5% statt 1%)
                if current_close > sma20 * 1.005:  # 0.5% above SMA
                    bullish_count += 1
                elif current_close < sma20 * 0.995:  # 0.5% below SMA
                    bearish_count += 1
                else:
                    neutral_count += 1  # NEU: Zähle neutrale
                
                total_checked += 1
                
            except Exception as e:
                logger.debug(f"Could not check {check_pair}: {e}")
                continue
        
        # Calculate market breadth
        if total_checked > 0:
            # OPTION 1: Neutrale als leicht bullish zählen (0.5 Gewichtung)
            effective_bullish = bullish_count + (neutral_count * 0.5)
            market_breadth = effective_bullish / total_checked
            
            # OPTION 2: Nur bullish vs bearish (original)
            # market_breadth = bullish_count / total_checked
            
            market_direction = 1 if bullish_count > bearish_count else -1 if bearish_count > bullish_count else 0
        else:
            market_breadth = 0.5
            market_direction = 0
        
        dataframe['market_breadth'] = market_breadth
        dataframe['market_direction'] = market_direction
        dataframe['market_coins_checked'] = total_checked
        
        # ERWEITERTE LOGS
        logger.info(
            f"Market breadth for {metadata['pair']}: {market_breadth:.2%} "
            f"({bullish_count} bullish, {bearish_count} bearish, {neutral_count} neutral / {total_checked} total)"
        )
        
        return dataframe

    def calculate_market_regime(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Determine overall market regime (trending, ranging, volatile)
        FIXED: Handles futures format
        """
        lookback = self.regime_lookback_period.value
        
        # Detect futures mode
        is_futures = ':' in metadata['pair']
        
        # Use BTC as market proxy
        if is_futures:
            settlement = metadata['pair'].split(':')[1]
            btc_pairs = [f"BTC/USDT:{settlement}", "BTC/USDT"]  # Try futures first, then spot
        else:
            btc_pairs = ["BTC/USDT", "BTC/USD"]
        
        btc_data = None
        for btc_pair in btc_pairs:
            try:
                btc_data, _ = self.dp.get_analyzed_dataframe(btc_pair, self.timeframe)
                if btc_data is not None and not btc_data.empty and len(btc_data) >= lookback:
                    logger.debug(f"Using {btc_pair} for market regime calculation")
                    break
            except:
                continue
        
        if btc_data is None or btc_data.empty or len(btc_data) < lookback:
            dataframe['market_regime'] = 'unknown'
            dataframe['market_volatility'] = 0.02  # Default 2%
            dataframe['market_adx'] = 25
            return dataframe
        
        # Calculate market volatility
        btc_returns = btc_data['close'].pct_change()
        market_volatility = btc_returns.rolling(lookback).std()
        
        # Calculate trend strength using ADX
        btc_adx = ta.ADX(btc_data, timeperiod=14)
        
        # Determine regime
        current_volatility = market_volatility.iloc[-1] if not market_volatility.empty else 0.02
        current_adx = btc_adx.iloc[-1] if len(btc_adx) > 0 and not pd.isna(btc_adx.iloc[-1]) else 25
        
        # Define regimes
        if current_adx > 40 and current_volatility < 0.03:
            regime = 'strong_trend'
        elif current_adx > 25 and current_volatility < 0.04:
            regime = 'trending'
        elif current_adx < 20 and current_volatility < 0.02:
            regime = 'ranging'
        elif current_volatility > 0.05:
            regime = 'high_volatility'
        else:
            regime = 'normal'
        
        dataframe['market_regime'] = regime
        dataframe['market_volatility'] = current_volatility
        dataframe['market_adx'] = current_adx
        
        return dataframe

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
        Apply market correlation filters to generate additional conditions
        Returns a boolean Series indicating whether market conditions are favorable
        """
        conditions = pd.Series(True, index=dataframe.index)
        
        # BTC Correlation Filter
        if self.btc_correlation_enabled.value and 'btc_correlation' in dataframe.columns:
            if direction == 'long':
                # For longs: prefer positive correlation when BTC is bullish
                conditions &= (
                    (dataframe['btc_correlation'] > self.btc_correlation_threshold.value) |
                    (dataframe['btc_correlation'] < -self.btc_correlation_threshold.value)  # Or strong inverse
                )
            else:  # short
                # For shorts: prefer negative correlation or when BTC is bearish
                conditions &= (
                    (dataframe['btc_correlation'] < -self.btc_correlation_threshold.value) |
                    ((dataframe['btc_correlation'] > self.btc_correlation_threshold.value) & 
                     (dataframe['btc_trend'] == -1))
                )
        
        # BTC Trend Filter - KORRIGIERT
        if self.btc_trend_filter_enabled.value and 'btc_trend' in dataframe.columns:
            if direction == 'long':
                # Erlaube Longs auch im Downtrend, aber nur bei extremen Oversold-Bedingungen
                if 'rsi' in dataframe.columns:
                    # Bei RSI < 30 ignoriere BTC-Trend (Bounce-Trades)
                    conditions &= ((dataframe['btc_trend'] >= 0) | (dataframe['rsi'] < 30))
                else:
                    conditions &= (dataframe['btc_trend'] >= -1)  # Weniger restriktiv
            else:  # short
                # Only short when BTC is not in strong uptrend
                conditions &= (dataframe['btc_trend'] <= 0)
        
        # Market Breadth Filter
        if self.market_breadth_enabled.value and 'market_breadth' in dataframe.columns:
            if direction == 'long':
                # Long when majority of market is bullish
                conditions &= (dataframe['market_breadth'] > self.market_breadth_threshold.value)
            else:  # short
                # Short when majority of market is bearish
                conditions &= (dataframe['market_breadth'] < (1 - self.market_breadth_threshold.value))
        
        # Market Cap Trend Filter
        if self.total_mcap_filter_enabled.value and 'mcap_status' in dataframe.columns:
            if direction == 'long':
                conditions &= (dataframe['mcap_status'] != 'bearish')
            else:  # short
                conditions &= (dataframe['mcap_status'] != 'bullish')
        
        # Market Regime Filter
        if self.regime_filter_enabled.value and 'market_regime' in dataframe.columns:
            # Avoid trading in high volatility regimes
            conditions &= (dataframe['market_regime'] != 'high_volatility')
            
            # For mean reversion strategies, prefer ranging markets
            # For trend strategies, prefer trending markets
            # This can be customized based on entry type
        
        return conditions

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
            logger.info(f"{pair} Initial stake calculated: {stake_amount:.8f} (Proposed: {proposed_stake:.8f}, "
                        f"Calculated Max DCA Multiplier: {calculated_max_dca_multiplier:.2f})")
            if side == "long" and entry_tag and "Bear" in entry_tag:
                stake_amount *= 0.5  # Halbiere die Position Size für Bear Market Bounces
                logger.info(f"{pair} Reducing long position size in bear market to {stake_amount:.8f}")
            if min_stake is not None and stake_amount < min_stake:
                logger.info(f"{pair} Initial stake {stake_amount:.8f} was below min_stake {min_stake:.8f}. "
                            f"Adjusting to min_stake. Consider tuning your DCA parameters or proposed stake.")
                stake_amount = min_stake
            return stake_amount
        else:
            logger.warning(
                f"{pair} Calculated max_dca_multiplier is {calculated_max_dca_multiplier:.2f}, which is invalid. "
                f"Using proposed_stake: {proposed_stake:.8f}")
            return proposed_stake

    def custom_entry_price(self, pair: str, trade: Optional[Trade], current_time: datetime,
                           proposed_rate: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe.empty:
            logger.warning(f"{pair} Empty DataFrame in custom_entry_price. Returning proposed_rate.")
            return proposed_rate
        last_candle = dataframe.iloc[-1]
        entry_price = (last_candle["close"] + last_candle["open"] + proposed_rate) / 3.0
        if side == "long":
            if proposed_rate < entry_price:
                entry_price = proposed_rate
        elif side == "short":
            if proposed_rate > entry_price:
                entry_price = proposed_rate
        logger.info(
            f"{pair} Calculated Entry Price: {entry_price:.8f} | Last Close: {last_candle['close']:.8f}, "
            f"Last Open: {last_candle['open']:.8f}, Proposed Rate: {proposed_rate:.8f}")
        if self.last_entry_price is not None and abs(entry_price - self.last_entry_price) < 0.000005:
            increment_factor = self.increment_for_unique_price.value if side == "long" else (
                1.0 / self.increment_for_unique_price.value)
            entry_price *= increment_factor
            logger.info(
                f"{pair} Entry price incremented to {entry_price:.8f} (previous: {self.last_entry_price:.8f}) due to proximity.")
        self.last_entry_price = entry_price
        return entry_price

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                                   current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Weniger aggressiver Stoploss um force exits zu vermeiden
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe.empty or 'atr' not in dataframe.columns:
            return -0.12  # Weniger aggressiv als -0.15
        
        last_atr = dataframe["atr"].iat[-1]
        if pd.isna(last_atr) or last_atr == 0:
            return -0.12
        
        # Weniger aggressive Stoploss-Berechnung
        atr_multiplier = 1.0  # Reduziert von 1.5
        dynamic_sl_ratio = atr_multiplier * last_atr / current_rate
        calculated_stoploss = -abs(dynamic_sl_ratio)
        
        # Nie unter -20% gehen
        final_stoploss = max(calculated_stoploss, -0.20)
        return final_stoploss

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
        
        # === ENHANCED DCA LOGIC WITH MARKET AWARENESS ===
        if not self.position_adjustment_enable:
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
        
        # Rest of original DCA logic continues...
        if count_of_entries >= self.max_safety_orders.value + 1:
            return None

        try:
            filled_entry_orders = trade.select_filled_orders(trade.entry_side)
            if not filled_entry_orders:
                logger.error(
                    f"{trade.pair} No filled entry orders found for DCA calculation, although entry count is {count_of_entries}.")
                return None
            last_order_cost = filled_entry_orders[-1].cost
            MAX_DCA_STAKE = 5  # Cap each DCA to max $20 (or your quote currency)
            scaled_dca = abs(last_order_cost * self.safety_order_volume_scale.value)
            dca_stake_amount = min(scaled_dca, MAX_DCA_STAKE)

            if min_stake is not None and dca_stake_amount < min_stake:
                logger.warning(
                    f"{trade.pair} DCA stake {dca_stake_amount:.8f} below min_stake {min_stake:.8f}. Adjusting to min_stake.")
                dca_stake_amount = min_stake
            if max_stake is not None and (trade.stake_amount + dca_stake_amount) > max_stake:
                available_for_dca = max_stake - trade.stake_amount
                if dca_stake_amount > available_for_dca and available_for_dca > (min_stake or 0):
                    logger.warning(
                        f"{trade.pair} DCA stake {dca_stake_amount:.8f} reduced to {available_for_dca:.8f} due to max_stake limit.")
                    dca_stake_amount = available_for_dca
                elif available_for_dca <= (min_stake or 0):
                    logger.warning(
                        f"{trade.pair} Cannot DCA. Adding {available_for_dca:.8f} would exceed max_stake or is below min_stake.")
                    return None
            logger.info(f"{trade.pair} Adjusting position with DCA. Adding {dca_stake_amount:.8f}. "
                        f"Entry count: {count_of_entries}, Max safety: {self.max_safety_orders.value}")
            return dca_stake_amount
        except IndexError:
            logger.error(
                f"Error calculating DCA stake for {trade.pair}: IndexError accessing last_order. Filled orders: {filled_entry_orders}")
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

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        OPTIMIERTE Indicator-Berechnung für schnellere Backtests mit Market Correlation
        """
        # Standard indicators (diese sind schnell)
        dataframe["ema50"] = ta.EMA(dataframe["close"], timeperiod=50)
        dataframe["rsi"] = ta.RSI(dataframe["close"])
        dataframe["atr"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe)
        dataframe["DI_values"] = dataframe["plus_di"] - dataframe["minus_di"]
        dataframe["DI_cutoff"] = 0

        # Vereinfachte Extrema detection (weniger rechenintensiv)
        extrema_order = 3  # Fest gesetzt statt Parameter
        dataframe["maxima"] = (
            dataframe["close"] == dataframe["close"].rolling(window=extrema_order).max()
        ).astype(int)
        dataframe["minima"] = (
            dataframe["close"] == dataframe["close"].rolling(window=extrema_order).min()
        ).astype(int)

        dataframe["s_extrema"] = 0
        dataframe.loc[dataframe["minima"] == 1, "s_extrema"] = -1
        dataframe.loc[dataframe["maxima"] == 1, "s_extrema"] = 1

        # Heikin-Ashi close
        dataframe["ha_close"] = (dataframe["open"] + dataframe["high"] + dataframe["low"] + dataframe["close"]) / 4

        # Vereinfachte Rolling extrema (nur h2 für Performance)
        dataframe["minh2"], dataframe["maxh2"] = calculate_minima_maxima(dataframe, 40)  # Fest gesetzt
        
        # Setze andere als 0 für Kompatibilität
        dataframe["minh1"] = 0
        dataframe["maxh1"] = 0
        dataframe["minh0"] = 0
        dataframe["maxh0"] = 0
        dataframe["mincp"] = 0
        dataframe["maxcp"] = 0

        # OPTIMIERTE Murrey Math levels (nur wichtigste)
        mml_window = 64  # Fest gesetzt
        murrey_levels = self.calculate_rolling_murrey_math_levels_optimized(dataframe, window_size=mml_window)
        
        # Nur die wichtigsten Levels für Performance
        important_levels = ["[0/8]P", "[2/8]P", "[4/8]P", "[6/8]P", "[7/8]P", "[8/8]P"]
        for level_name in important_levels:
            if level_name in murrey_levels:
                dataframe[level_name] = murrey_levels[level_name]
        
        # Setze nicht verwendete Levels auf Mittellinie für Kompatibilität
        all_levels = ["[-3/8]P", "[-2/8]P", "[-1/8]P", "[1/8]P", "[3/8]P", "[5/8]P", "[+1/8]P", "[+2/8]P", "[+3/8]P"]
        for level in all_levels:
            if level not in dataframe.columns:
                dataframe[level] = dataframe["[4/8]P"] if "[4/8]P" in dataframe.columns else dataframe["close"]

        # DI Catch
        dataframe["DI_catch"] = np.where(dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1)

        # Vereinfachte Rolling thresholds
        dataframe["minima_sort_threshold"] = dataframe["close"].rolling(window=10).min()
        dataframe["maxima_sort_threshold"] = dataframe["close"].rolling(window=10).max()

        # Vereinfachte Extrema checks
        dataframe["minima_check"] = (dataframe["minima"].rolling(window=4).sum() == 0).astype(int)
        dataframe["maxima_check"] = (dataframe["maxima"].rolling(window=4).sum() == 0).astype(int)

        # OPTIMIERTE VVRP Integration (reduzierte Komplexität)
        dataframe = calculate_rolling_vvrp_optimized(dataframe, lookback_period=60, num_bins=5, max_bars=100)  # Reduziert
        
        # Vereinfachte VVRP High Volume Detection
        dataframe['VVRP_High_Volume_Long'] = 0
        dataframe['VVRP_High_Volume_Short'] = 0
        
        # Nur 3 wichtigste Bins prüfen statt 9
        for i in range(len(dataframe)):
            for j in [2, 3, 4]:  # Nur mittlere Bins
                if f'VVRP_Buy_Bars_{j}' in dataframe.columns and f'VVRP_Sell_Bars_{j}' in dataframe.columns:
                    buy_bars = dataframe[f'VVRP_Buy_Bars_{j}'].iloc[i]
                    sell_bars = dataframe[f'VVRP_Sell_Bars_{j}'].iloc[i]
                    
                    if buy_bars - sell_bars > 5:  # Fest gesetzt statt Parameter
                        dataframe.at[i, 'VVRP_High_Volume_Long'] = 1
                    if sell_bars - buy_bars > 5:
                        dataframe.at[i, 'VVRP_High_Volume_Short'] = 1

        # Einfache Volatility indicators
        dataframe["volatility_range"] = dataframe["high"] - dataframe["low"]
        dataframe["avg_volatility"] = dataframe["volatility_range"].rolling(window=20).mean()
        dataframe["avg_volume"] = dataframe["volume"].rolling(window=20).mean()

        # === ADD MARKET CORRELATION INDICATORS ===
        if self.btc_correlation_enabled.value or self.btc_trend_filter_enabled.value:
            dataframe = self.calculate_market_correlation(dataframe, metadata)
        
        if self.market_breadth_enabled.value:
            dataframe = self.calculate_market_breadth(dataframe, metadata)
        
        if self.regime_filter_enabled.value:
            dataframe = self.calculate_market_regime(dataframe, metadata)
        
        if self.total_mcap_filter_enabled.value:
            dataframe = self.calculate_total_market_cap_trend(dataframe, metadata)
        
        # Add composite market score
        dataframe['market_score'] = 0
        
        # BTC correlation score
        if 'btc_correlation' in dataframe.columns:
            dataframe['market_score'] += dataframe['btc_correlation'] * 0.3
        
        # Market breadth score
        if 'market_breadth' in dataframe.columns:
            dataframe['market_score'] += dataframe['market_breadth'] * 0.3
        
        # Market trend score
        if 'mcap_trend' in dataframe.columns:
            dataframe['market_score'] += (dataframe['mcap_trend'] + 1) * 0.2
        
        # Regime score
        if 'market_regime' in dataframe.columns:
            regime_scores = {
                'strong_trend': 1.0,
                'trending': 0.7,
                'normal': 0.5,
                'ranging': 0.3,
                'high_volatility': 0.0
            }
            dataframe['regime_score'] = dataframe['market_regime'].map(
                lambda x: regime_scores.get(x, 0.5)
            )
            dataframe['market_score'] += dataframe['regime_score'] * 0.2

        return dataframe

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Entry logic with Market Correlation Filters
        """
        
        # Get market conditions for filtering
        long_market_conditions = self.apply_correlation_filters(df, direction='long')
        short_market_conditions = self.apply_correlation_filters(df, direction='short') if self.can_short else pd.Series(False, index=df.index)
        # === BEAR MARKET BOUNCE ENTRIES ===
        # Extreme Oversold Bounce (funktioniert auch im Bärenmarkt)
        df.loc[
            (df["rsi"] < 25) &                          # Extrem oversold
            (df["close"] > df["close"].shift(1)) &      # Momentum dreht
            (df["volume"] > df["avg_volume"] * 1.5) &   # Volumen-Spike
            (df["close"] < df["[1/8]P"]) &              # Unter Support
            (df['DI_values'] > df['DI_values'].shift(1)),  # DI verbessert sich
            ["enter_long", "enter_tag"]
        ] = (1, "Bear_Market_Bounce")

        # Quick Scalp Long (für schnelle Trades)
        df.loc[
            (df["rsi"] < 30) &
            (df["rsi"] > df["rsi"].shift(1)) &          # RSI dreht nach oben
            (df["close"] > df["open"]) &                # Grüne Kerze
            (df["minima"] == 1) &                       # Lokales Tief
            ~long_market_conditions &                    # NUR wenn normale Bedingungen NICHT erfüllt
            (df["volume"] > 0),
            ["enter_long", "enter_tag"]
        ] = (1, "Bear_Scalp_Long")
        # === LONG ENTRIES (Bounce from Lows/Support) ===
        
        # Confirmed long entry - Bounce von bestätigtem Minimum
        df.loc[
            (df["DI_catch"] == 1) &
            (df["minima_check"] == 1) &           
            (df["s_extrema"] < 0) &               
            (df["minima"].shift(1) == 1) &        
            (df["volume"] > 0) &
            (df["rsi"] < 35) &                    
            (df["close"] > df["close"].shift(1)) &
            long_market_conditions,               # Market filter added
            ["enter_long", "enter_tag"]
        ] = (1, "Confirmed_Min_Entry_MC")
        
        # Aggressive long entry - Früher Einstieg bei Oversold
        df.loc[
            (df["minima_check"] == 0) &           
            (df["volume"] > 0) &
            (df["rsi"] < 30) &                    
            (df["close"] > df["close"].shift(1)) &
            long_market_conditions,               # Market filter added
            ["enter_long", "enter_tag"]
        ] = (1, "Aggressive_Min_Entry_MC")
        
        # Transitional long entry
        df.loc[
            (df["DI_catch"] == 1) &
            (df["minima_check"] == 0) &
            (df["minima_check"].shift(5) == 1) &
            (df["volume"] > 0) &
            (df["rsi"] < 32) &
            (df["close"] > df["close"].shift(1)) &
            long_market_conditions,               # Market filter added
            ["enter_long", "enter_tag"]
        ] = (1, "Transitional_Min_Entry_MC")
        
        # Rolling long entry - H2 Minima Bounce
        df.loc[
            (df["minh2"] < 0) &                   
            (df["rsi"] < 36) &
            (df["volume"] > 0) &
            (df["close"] > df["close"].shift(1)) &
            long_market_conditions,               # Market filter added
            ["enter_long", "enter_tag"]
        ] = (1, "Rolling_MinH2_Entry_MC")

        # === TREND LONG ENTRIES ===
        df.loc[
            (df["close"] > df["ema50"]) &         
            (df["rsi"] > 40) & (df["rsi"] < 60) & 
            (df["DI_values"] > 0) &               
            (df["close"] > df["[4/8]P"]) &        
            (df["volume"] > df["volume"].rolling(10).mean()) &
            long_market_conditions,               # Market filter added
            ["enter_long", "enter_tag"]
        ] = (1, "Trend_Long_MC")

        # === MEAN REVERSION LONG ===
        df.loc[
            (df["rsi"] < 25) &                    
            (df["rsi"] > df["rsi"].shift(1)) &    
            (df["close"] < df["[2/8]P"]) &        
            (df["minima"] == 1) &                 
            (df["volume"] > df["volume"].rolling(5).mean() * 1.2) &
            long_market_conditions,               # Market filter added
            ["enter_long", "enter_tag"]
        ] = (1, "Reversal_Long_MC")

        # === SHORT ENTRIES (Rejection from Highs/Resistance) ===
        if self.can_short:
            
            # Confirmed short entry - Rejection von bestätigtem Maximum
            df.loc[
                (df["DI_catch"] == 1) &
                (df["maxima_check"] == 1) &           
                (df["s_extrema"] > 0) &               
                (df["maxima"].shift(1) == 1) &        
                (df["volume"] > 0) &
                (df["rsi"] > 65) &                    
                (df["rsi"] < df["rsi"].shift(1)) &    
                (df["close"] < df["ema50"]) &         
                (df["close"] < df["[8/8]P"]) &        
                (df["close"] < df["close"].shift(1)) &
                short_market_conditions,               # Market filter added
                ["enter_short", "enter_tag"]
            ] = (1, "Confirmed_Max_Entry_MC")
            
            # Aggressive short entry - Früher Einstieg bei Overbought
            df.loc[
                (df["maxima_check"] == 0) &           
                (df["volume"] > 0) &
                (df["rsi"] > 70) &                    
                (df["close"] < df["close"].shift(1)) &
                short_market_conditions,               # Market filter added
                ["enter_short", "enter_tag"]
            ] = (1, "Aggressive_Max_Entry_MC")
            
            # Transitional short entry
            df.loc[
                (df["DI_catch"] == 1) &
                (df["maxima_check"] == 0) &
                (df["maxima_check"].shift(5) == 1) &
                (df["volume"] > 0) &
                (df["rsi"] > 68) &
                (df["close"] < df["close"].shift(1)) &
                short_market_conditions,               # Market filter added
                ["enter_short", "enter_tag"]
            ] = (1, "Transitional_Max_Entry_MC")
            
            # Rolling short entry - H2 Maxima Rejection
            df.loc[
                (df["maxh2"] > 0) &                   
                (df["rsi"] > 68) &
                (df["volume"] > 0) &
                (df["close"] < df["close"].shift(1)) &
                short_market_conditions,               # Market filter added
                ["enter_short", "enter_tag"]
            ] = (1, "Rolling_MaxH2_Entry_MC")

            # === TREND SHORT ENTRIES ===
            df.loc[
                (df["close"] < df["ema50"]) &         
                (df["rsi"] > 40) & (df["rsi"] < 60) & 
                (df["DI_values"] < 0) &               
                (df["close"] < df["[4/8]P"]) &        
                (df["volume"] > df["volume"].rolling(10).mean()) &
                short_market_conditions,               # Market filter added
                ["enter_short", "enter_tag"]
            ] = (1, "Trend_Short_MC")

            # === MEAN REVERSION SHORT ===
            df.loc[
                (df["rsi"] > 75) &                    
                (df["rsi"] < df["rsi"].shift(1)) &    
                (df["close"] > df["[6/8]P"]) &        
                (df["maxima"] == 1) &                 
                (df["volume"] > df["volume"].rolling(5).mean() * 1.2) &
                short_market_conditions,               # Market filter added
                ["enter_short", "enter_tag"]
            ] = (1, "Reversal_Short_MC")

        # === SPECIAL MARKET CONDITION ENTRIES ===
        if self.fear_greed_enabled.value:
            # Contrarian entries during extreme fear
            df.loc[
                (df['market_score'] < 0.2) &  # Extreme fear
                (df['rsi'] < 25) &
                (df['close'] > df['close'].shift(1)) &
                (df['volume'] > df['avg_volume']),
                ['enter_long', 'enter_tag']
            ] = (1, "Extreme_Fear_Reversal")
            
            # Take profits during extreme greed
            if self.can_short:
                df.loc[
                    (df['market_score'] > 0.8) &  # Extreme greed
                    (df['rsi'] > 75) &
                    (df['close'] < df['close'].shift(1)) &
                    (df['volume'] > df['avg_volume']),
                    ['enter_short', 'enter_tag']
                ] = (1, "Extreme_Greed_Reversal")

        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Enhanced exit strategy with market-aware conditions
        Keeps ROI as primary exit (100% win rate) but adds smart emergency exits
        """
        
        # === 1. MARKET CRASH PROTECTION ===
        # Exit all longs if market is crashing
        df.loc[
            # Market conditions
            (df['market_breadth'] < 0.2) &  # 80%+ of market is bearish
            (df['btc_trend'] == -1) &        # BTC in downtrend
            (df['market_volatility'] > 0.04) &  # High volatility
            # Position conditions
            (df['close'] < df['open']) &     # Current candle is red
            (df['volume'] > df['avg_volume'] * 2),  # Panic volume
            ["exit_long", "exit_tag"]
        ] = (1, "Market_Crash_Protection")
        
        # === 2. CORRELATION BREAKDOWN EXIT ===
        # Exit if correlation suddenly changes (pair decoupling)
        correlation_change = df['btc_correlation'].diff().abs()
        df.loc[
            (correlation_change > 0.4) &     # Sudden correlation change
            (df['rsi'] > 70) &               # While overbought
            (df['close'] < df['close'].shift(1)),  # Price turning down
            ["exit_long", "exit_tag"]
        ] = (1, "Correlation_Breakdown")
        
        # === 3. REGIME CHANGE EXIT ===
        # Exit positions when market regime changes unfavorably
        df.loc[
            (df['market_regime'] == 'high_volatility') &
            (df['market_regime'].shift(1) != 'high_volatility') &  # Just changed
            (df['rsi'] > 60),  # Not oversold
            ["exit_long", "exit_tag"]
        ] = (1, "Regime_Change_Exit")
        
        # === 4. SMART PARTIAL EXIT ENHANCEMENT ===
        # More aggressive partial exits during extreme greed
        df.loc[
            (df['market_score'] > 0.85) &    # Extreme market greed
            (df['rsi'] > 80) &               # Overbought
            (df['close'] > df['[7/8]P']),    # Near resistance
            ["exit_long", "exit_tag"]
        ] = (1, "Greed_Partial_Exit")
        
        # === 5. DIVERGENCE EXIT ===
        # Exit when price makes new high but market breadth doesn't
        price_high = df['close'].rolling(20).max()
        breadth_high = df['market_breadth'].rolling(20).max()
        
        df.loc[
            (df['close'] >= price_high) &     # New price high
            (df['market_breadth'] < breadth_high * 0.9) &  # But breadth is weak
            (df['rsi'] > 65),                 # Overbought
            ["exit_long", "exit_tag"]
        ] = (1, "Breadth_Divergence_Exit")
        
        # === SHORT EXITS (if enabled) ===
        if self.can_short:
            # Market recovery protection for shorts
            df.loc[
                (df['market_breadth'] > 0.8) &   # 80%+ of market bullish
                (df['btc_trend'] == 1) &         # BTC uptrend
                (df['close'] > df['open']) &     # Green candle
                (df['volume'] > df['avg_volume'] * 2),  # Strong volume
                ["exit_short", "exit_tag"]
            ] = (1, "Market_Recovery_Protection")
            
            # Exit shorts on correlation breakdown
            df.loc[
                (correlation_change > 0.4) &      # Sudden correlation change
                (df['rsi'] < 30) &                # While oversold
                (df['close'] > df['close'].shift(1)),  # Price turning up
                ["exit_short", "exit_tag"]
            ] = (1, "Correlation_Breakdown_Short")
            
            # Exit shorts during extreme fear (contrarian)
            df.loc[
                (df['market_score'] < 0.15) &    # Extreme fear
                (df['rsi'] < 20) &               # Extremely oversold
                (df['close'] < df['[1/8]P']),    # Near support
                ["exit_short", "exit_tag"]
            ] = (1, "Fear_Partial_Exit")
        
        return df

def calculate_rolling_vvrp_optimized(dataframe: pd.DataFrame, lookback_period: int, 
                                   num_bins: int, max_bars: int) -> pd.DataFrame:
    """
    OPTIMIERTE VVRP - nur alle 10 Candles berechnen
    FIXED: Pandas 2.0+ kompatibel
    """
    dataframe['Average_Price'] = (dataframe['high'] + dataframe['low']) / 2
    dataframe['price_range'] = dataframe['high'] - dataframe['low']
    
    # Vereinfachte Buy/Sell Volume Berechnung
    dataframe['buy_volume'] = np.where(
        dataframe['close'] > dataframe['open'],
        dataframe['volume'] * 0.6,  # Vereinfacht
        dataframe['volume'] * 0.4
    )
    dataframe['sell_volume'] = dataframe['volume'] - dataframe['buy_volume']

    bin_labels = range(1, num_bins + 1)
    for col in bin_labels:
        dataframe[f'VVRP_Buy_Bars_{col}'] = 0
        dataframe[f'VVRP_Sell_Bars_{col}'] = 0
        dataframe[f'VVRP_Mid_Price_{col}'] = np.nan

    # Nur alle 10 Candles berechnen
    calculation_step = 10
    
    for i in range(lookback_period - 1, len(dataframe), calculation_step):
        window = dataframe.iloc[max(0, i - lookback_period + 1):i + 1].copy()
        hi, lo = window['high'].max(), window['low'].min()
        width = hi - lo
        if width == 0: 
            continue
            
        bin_width = width / num_bins
        window['Price_Bins'] = pd.cut(window['Average_Price'], bins=num_bins, labels=bin_labels, include_lowest=True)
        grouped = window.groupby('Price_Bins', observed=True).agg({'buy_volume': 'sum', 'sell_volume': 'sum'}).reindex(bin_labels, fill_value=0)
        max_volume = max(grouped['buy_volume'].max(), grouped['sell_volume'].max(), 1)
        
        for j in bin_labels:
            dataframe.at[i, f'VVRP_Buy_Bars_{j}'] = int(np.round((grouped.at[j, 'buy_volume'] / max_volume) * max_bars))
            dataframe.at[i, f'VVRP_Sell_Bars_{j}'] = int(np.round((grouped.at[j, 'sell_volume'] / max_volume) * max_bars))
            dataframe.at[i, f'VVRP_Mid_Price_{j}'] = lo + bin_width * (j - 0.5)

    # FIXED: Forward-fill die Werte zwischen Berechnungen (moderne syntax)
    vvrp_columns = [col for col in dataframe.columns if col.startswith('VVRP_')]
    for col in vvrp_columns:
        dataframe[col] = dataframe[col].ffill()  # FIXED

    dataframe.drop(columns=['price_range', 'buy_volume', 'sell_volume'], errors='ignore', inplace=True)
    return dataframe