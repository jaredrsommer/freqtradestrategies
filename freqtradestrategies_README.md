# freqtradestrategies Python File Overview

This README summarizes the Python strategy files in [`RoboticAutomations/freqtradestrategies`](https://github.com/RoboticAutomations/freqtradestrategies). The repository describes itself as a collection of Freqtrade strategies and experiments. Use all strategies at your own risk and validate them with backtesting and dry-run trading before using real funds.

> Note: Several files are duplicates or numbered iterations of the same idea. Descriptions are intentionally short and focus on the apparent purpose of each strategy file.

## Python files

| File | Short description |
|---|---|
| [`2Candle (2).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/2Candle%20%282%29.py) | Two Candle Theory variant that classifies each candle by close location and prior-range breakout, then enters on bullish breakouts/support bounces and exits on opposite patterns. |
| [`2Candle (3).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/2Candle%20%283%29.py) | Two Candle Theory variant; likely an iteration of the candle-position/breakout strategy with adjusted entry, exit, or risk rules. |
| [`2Candle (4).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/2Candle%20%284%29.py) | Later Two Candle Theory experiment that tests candle close-position patterns against short-term support and resistance. |
| [`2Candle (5).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/2Candle%20%285%29.py) | Additional Two Candle Theory revision for comparing current candle behavior to the previous candle and trading continuation/reversal setups. |
| [`2Candle.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/2Candle.py) | Base Two Candle strategy using candle pattern classification, support/resistance, and simple long/short signals. |
| [`A9AV.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/A9AV.py) | Volume-average strategy that compares current volume to a 9-period volume SMA and uses recent price direction to generate buy/sell signals. |
| [`AlexBTK_CT.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/AlexBTK_CT.py) | Enhanced Alex Battle Tank Killer variant with Murrey Math levels, Heikin-Ashi extrema, ATR-based dynamic stoploss, DCA, leverage logic, and market correlation filters. |
| [`AlexBattleTankKiller.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/AlexBattleTankKiller.py) | Original Battle Tank Killer-style strategy focused on extrema, Murrey Math levels, DCA/position adjustment, and risk controls. |
| [`AlexBattleTankKillerV3.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/AlexBattleTankKillerV3.py) | Version 3 of the Battle Tank Killer strategy, adding or tuning Murrey Math/extrema logic and optimized risk parameters. |
| [`AlexBattleTankKillerV4H.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/AlexBattleTankKillerV4H.py) | Version 4H Battle Tank Killer variant, likely tuned for higher-timeframe confirmation, Murrey Math levels, extrema signals, and DCA behavior. |
| [`Astro.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/Astro.py) | Astrology-themed strategy experiment; likely combines market indicators with astro/cycle-style timing features for entries and exits. |
| [`AstroQAV4.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/AstroQAV4.py) | Version 4 quality/analysis iteration of the Astro strategy with additional filters or refined signal confirmation. |
| [`AwesomeMacd.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/AwesomeMacd.py) | Strategy combining Awesome Oscillator-style momentum with MACD trend/momentum confirmation. |
| [`BB_RPB_TSL.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/BB_RPB_TSL.py) | Bollinger Band + RPB-style dip-buying strategy with trailing stop-loss logic and multiple oversold/rebound filters. |
| [`BB_RPB_TSL_SMA_Tranz_1.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/BB_RPB_TSL_SMA_Tranz_1.py) | BB/RPB/TSL variant that adds SMA trend filtering or transition logic to the Bollinger/dip-buying framework. |
| [`CTIBS.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/CTIBS.py) | CTIBS strategy file; likely a custom technical-indicator blend with hyperoptable parameters and matching JSON configuration. |
| [`ElliotWave.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/ElliotWave.py) | Elliott Wave-inspired strategy that attempts to detect wave/cycle structure and trade trend continuation or reversals. |
| [`FVGChannel.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/FVGChannel.py) | Fair Value Gap channel strategy that looks for price inefficiencies/gaps and channel-based continuation or reversal setups. |
| [`ForexRobootSuperScalper (2).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/ForexRobootSuperScalper%20%282%29.py) | Super-scalper strategy adapted from a forex robot concept, intended for frequent short-timeframe entries with tight risk management. |
| [`GKD_Baseline.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/GKD_Baseline.py) | GKD baseline strategy using a primary moving-average or trend baseline as the core directional filter. |
| [`GKD_BaselineAllMAs (1).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/GKD_BaselineAllMAs%20%281%29.py) | GKD baseline variant that tests multiple moving-average types as interchangeable trend baselines. |
| [`GKD_C.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/GKD_C.py) | GKD custom strategy variant, likely focused on a compact indicator set and optimized conditions. |
| [`GKD_CT.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/GKD_CT.py) | GKD cycle/trend variant with configuration support, combining trend and timing filters. |
| [`GKD_FisherTransform (2).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/GKD_FisherTransform%20%282%29.py) | Fisher Transform strategy variant that uses normalized price turning points to identify reversals. |
| [`GKD_FisherTransform.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/GKD_FisherTransform.py) | Base GKD Fisher Transform strategy for detecting overbought/oversold reversals via Fisher-transformed price action. |
| [`GKD_FisherTransformMTF.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/GKD_FisherTransformMTF.py) | Multi-timeframe Fisher Transform strategy using higher-timeframe confirmation for lower-timeframe entries. |
| [`GKD_HurstExponent.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/GKD_HurstExponent.py) | Strategy using the Hurst exponent to distinguish trending, mean-reverting, or random market regimes. |
| [`GKD_PFE.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/GKD_PFE.py) | Strategy using Polarized Fractal Efficiency to measure trend efficiency and filter entries. |
| [`GPR.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/GPR.py) | Gaussian-process/regression-style strategy experiment, likely using predictive smoothing or statistical regression features. |
| [`GPTREV.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/GPTREV.py) | GPT-generated or GPT-assisted reversal strategy focused on detecting reversal setups with technical filters. |
| [`HEW.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/HEW.py) | Hurst/Elliott Wave-style strategy experiment that likely combines cycle analysis with wave-based entries. |
| [`HSI (2).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/HSI%20%282%29.py) | HSI indicator strategy variant; likely uses a custom strength/heat/sine-style indicator for trade signals. |
| [`HSI.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/HSI.py) | Base HSI strategy using a named custom indicator set for entries, exits, or trend filtering. |
| [`HilbertSineWave.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/HilbertSineWave.py) | Hilbert Transform sine-wave strategy for identifying cycle phase changes and potential turning points. |
| [`HurstCycle3 (2).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/HurstCycle3%20%282%29.py) | Hurst Cycle 3 variant using cycle/phase analysis to identify potential market turns. |
| [`HurstCycle3 (3).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/HurstCycle3%20%283%29.py) | Additional Hurst Cycle 3 revision with tuned thresholds or entry/exit conditions. |
| [`HurstCycle3.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/HurstCycle3.py) | Base Hurst Cycle 3 strategy using cyclic market structure for timing entries. |
| [`HurstCycle7 (2).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/HurstCycle7%20%282%29.py) | Hurst Cycle 7 variant with adjusted cycle windows or optimized parameters. |
| [`HurstCycle7.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/HurstCycle7.py) | Base Hurst Cycle 7 strategy for cycle-based entries and exits. |
| [`HurstCycleV4.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/HurstCycleV4.py) | Fourth Hurst Cycle strategy revision with updated cycle logic and risk filters. |
| [`HurstCycleV5 (2).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/HurstCycleV5%20%282%29.py) | Hurst Cycle V5 variant, likely an optimization pass over cycle detection and signal confirmation. |
| [`HurstCycleV5.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/HurstCycleV5.py) | Fifth Hurst Cycle strategy revision using cycle-phase signals and technical filters. |
| [`HurstCycleV5RSI (2).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/HurstCycleV5RSI%20%282%29.py) | Hurst Cycle V5 variant with RSI confirmation for overbought/oversold filtering. |
| [`HurstCycleV5RSI.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/HurstCycleV5RSI.py) | Hurst Cycle V5 strategy that adds RSI-based momentum or reversal confirmation. |
| [`HurstCycleV6.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/HurstCycleV6.py) | Sixth Hurst Cycle strategy revision with further tuning of cycle and signal filters. |
| [`IchiVwapAdx (2).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/IchiVwapAdx%20%282%29.py) | Ichimoku + VWAP + ADX strategy variant combining trend, average-price, and trend-strength filters. |
| [`IchiVwapAdx (3).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/IchiVwapAdx%20%283%29.py) | Additional Ichimoku/VWAP/ADX revision with modified thresholds or signal rules. |
| [`Ichimoku_v12 (copy).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/Ichimoku_v12%20%28copy%29.py) | Copy of Ichimoku v12 strategy using Ichimoku cloud components for trend confirmation and entries. |
| [`Ichimoku_v12.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/Ichimoku_v12.py) | Version 12 Ichimoku strategy built around cloud trend, baseline/conversion signals, and exit logic. |
| [`ImpulseV1.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/ImpulseV1.py) | Impulse strategy version 1, likely using momentum expansion and trend confirmation to catch strong moves. |
| [`KMM.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/KMM.py) | KMM strategy file; likely a custom indicator/momentum model with optimized buy/sell parameters. |
| [`KitchenSink (2).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/KitchenSink%20%282%29.py) | Kitchen Sink variant that combines many indicators/filters into one broad experimental strategy. |
| [`KitchenSink.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/KitchenSink.py) | Large multi-indicator strategy intended to test a wide set of technical signals together. |
| [`LorentzianClassification (2).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/LorentzianClassification%20%282%29.py) | Lorentzian Classification strategy variant inspired by nearest-neighbor/ML-style market classification. |
| [`MKR (3).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/MKR%20%283%29.py) | MKR strategy revision; likely a custom momentum/reversal strategy with tuned parameters. |
| [`MSO.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/MSO.py) | MSO strategy, likely based on a market strength or oscillator signal for trend/reversal entries. |
| [`NeuroV1.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/NeuroV1.py) | Neural-network or neuro-inspired strategy version 1 using predictive or classification-style signals. |
| [`NoTankAi_17.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/NoTankAi_17.py) | NoTank AI strategy version 17, an AI/indicator hybrid likely derived from the TankAi family with tuned filters. |
| [`NoTankAi_19_1.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/NoTankAi_19_1.py) | NoTank AI strategy version 19.1, a later tuned iteration with revised entry/exit and risk settings. |
| [`OmaGann (2).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/OmaGann%20%282%29.py) | OMA + Gann variant combining optimized moving average concepts with Gann-style levels or timing. |
| [`OmaGann.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/OmaGann.py) | Base OMA/Gann strategy using moving-average smoothing and Gann-inspired support/resistance or trend logic. |
| [`PnF (copy).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/PnF%20%28copy%29.py) | Point-and-Figure strategy copy that interprets price movement through PnF-style trend/reversal signals. |
| [`TD.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/TD.py) | Tom DeMark-style strategy using TD sequential/countdown or exhaustion concepts. |
| [`TGMA.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/TGMA.py) | Trend/Gann/Moving-Average-style strategy centered on moving-average trend filtering. |
| [`TRIWAVE.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/TRIWAVE.py) | Tri-wave strategy using multiple wave/cycle components to identify trend or reversal opportunities. |
| [`TSPredict.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/TSPredict.py) | Time-series prediction strategy that uses forecasting-style features to anticipate price direction. |
| [`TankAi.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/TankAi.py) | Tank AI base strategy, likely combining technical indicators with AI-inspired signal scoring and risk controls. |
| [`TankAiRevival.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/TankAiRevival.py) | Revived Tank AI strategy variant with updated logic, parameters, or Freqtrade compatibility. |
| [`TrendFollowingStrategy_4 (copy).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/TrendFollowingStrategy_4%20%28copy%29.py) | Copy of a trend-following strategy version 4 using directional filters and continuation entries. |
| [`TwoCandle.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/TwoCandle.py) | Two-candle pattern strategy that compares consecutive candles to identify continuation or reversal entries. |
| [`TwoCandleTheory (2).py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/TwoCandleTheory%20%282%29.py) | Two Candle Theory variant implementing candle-location and prior-candle relationship rules. |
| [`TwoCandleTheory.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/TwoCandleTheory.py) | Base Two Candle Theory implementation for trading candle close-position patterns and breakouts. |
| [`WTAI.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/WTAI.py) | WT AI strategy, likely combining WaveTrend-style oscillator signals with AI/tuned filters. |
| [`WTHO.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/WTHO.py) | WaveTrend/Heikin-Oscillator-style strategy using oscillator turns and trend filters. |
| [`WTRSIAI.py`](https://github.com/RoboticAutomations/freqtradestrategies/blob/main/WTRSIAI.py) | WaveTrend + RSI + AI-style strategy combining oscillator, momentum, and tuned confirmation filters. |
