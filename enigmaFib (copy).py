import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# Download the data
data = yf.download('TSLA', start='2020-01-01', end='2022-02-26')

# Define the function to calculate the Fibonacci levels
def calculate_fibonacci_levels(close, high, low, swing_lookback, fib_level_1, fib_level_2, fib_level_3, fib_level_4):
    swing_high = high.rolling(window=swing_lookback).max()
    swing_low = low.rolling(window=swing_lookback).min()
    is_uptrend = close > close.rolling(window=swing_lookback).mean()
    fib_dynamic_levels = np.zeros((len(close), 4))
    fib_dynamic_levels[:, :] = np.nan
    for i in range(swing_lookback, len(close)):
        if is_uptrend.iloc[i]:
            fib_dynamic_levels[i, 0] = swing_low.iloc[i] + (swing_high.iloc[i] - swing_low.iloc[i]) * fib_level_1
            fib_dynamic_levels[i, 1] = swing_low.iloc[i] + (swing_high.iloc[i] - swing_low.iloc[i]) * fib_level_2
            fib_dynamic_levels[i, 2] = swing_low.iloc[i] + (swing_high.iloc[i] - swing_low.iloc[i]) * fib_level_3
            fib_dynamic_levels[i, 3] = swing_low.iloc[i] + (swing_high.iloc[i] - swing_low.iloc[i]) * fib_level_4
        else:
            fib_dynamic_levels[i, 0] = swing_high.iloc[i] - (swing_high.iloc[i] - swing_low.iloc[i]) * fib_level_1
            fib_dynamic_levels[i, 1] = swing_high.iloc[i] - (swing_high.iloc[i] - swing_low.iloc[i]) * fib_level_2
            fib_dynamic_levels[i, 2] = swing_high.iloc[i] - (swing_high.iloc[i] - swing_low.iloc[i]) * fib_level_3
            fib_dynamic_levels[i, 3] = swing_high.iloc[i] - (swing_high.iloc[i] - swing_low.iloc[i]) * fib_level_4
    return fib_dynamic_levels

# Define the parameters
swing_lookback = 6
fib_level_1 = 0.5
fib_level_2 = 0.618
fib_level_3 = 0.72
fib_level_4 = 0.99

# Calculate the Fibonacci levels
fib_dynamic_levels = calculate_fibonacci_levels(data['Close'], data['High'], data['Low'], swing_lookback, fib_level_1, fib_level_2, fib_level_3, fib_level_4)

# Plot the data
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(data.index, data['Close'], label='Close')
ax.plot(data.index, fib_dynamic_levels[:, 0], color='orange', alpha=0.5, label='Fib Level 1')
ax.plot(data.index, fib_dynamic_levels[:, 1], color='blue', alpha=0.5, label='Fib Level 2')
ax.plot(data.index, fib_dynamic_levels[:, 2], color='green', alpha=0.5, label='Fib Level 3')
ax.plot(data.index, fib_dynamic_levels[:, 3], color='red', alpha=0.5, label='Fib Level 4')
ax.set_title('Price Action and Fibonacci Levels')
ax.legend(loc='upper left')
plt.show()