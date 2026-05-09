
#     plot_config = {
#       "main_plot": {
#         "smoothed_close": {
#           "color": "#1f77b4",
#           "type": "line"
#         },
#         "ew_upward_wave5": {
#           "color": "#00FF00",
#           "type": "scatter",
#           "pointStyle": "circle",
#           "pointRadius": 6
#         },
#         "ew_downward_wave5": {
#           "color": "#FF0000",
#           "type": "scatter",
#           "pointStyle": "circle",
#           "pointRadius": 6
#         },
#         "ew_practical12_upward": {
#           "color": "#32CD32",
#           "type": "scatter",
#           "pointStyle": "triangle",
#           "pointRadius": 5
#         },
#         "ew_practical12_downward": {
#           "color": "#FF69B4",
#           "type": "scatter",
#           "pointStyle": "triangle",
#           "pointRadius": 5
#         }
#       },
#       "subplots": {
#         "WaveSignals": {
#           "ew_traditional_upward": {
#             "color": "#00FF00",
#             "type": "scatter",
#             "pointStyle": "square",
#             "pointRadius": 4
#           },
#           "ew_traditional_downward": {
#             "color": "#FF0000",
#             "type": "scatter",
#             "pointStyle": "square",
#             "pointRadius": 4
#           },
#           "ew_alternative_upward": {
#             "color": "#32CD32",
#             "type": "scatter",
#             "pointStyle": "diamond",
#             "pointRadius": 4
#           },
#           "ew_alternative_downward": {
#             "color": "#FF69B4",
#             "type": "scatter",
#             "pointStyle": "diamond",
#             "pointRadius": 4
#           },
#           "ew_zigzag_upward": {
#             "color": "#00FF00",
#             "type": "scatter",
#             "pointStyle": "cross",
#             "pointRadius": 4
#           },
#           "ew_zigzag_downward": {
#             "color": "#FF0000",
#             "type": "scatter",
#             "pointStyle": "cross",
#             "pointRadius": 4
#           },
#           "ew_upward_wave3": {
#             "color": "#32CD32",
#             "type": "scatter",
#             "pointStyle": "circle",
#             "pointRadius": 4
#           },
#           "ew_downward_wave3": {
#             "color": "#FF69B4",
#             "type": "scatter",
#             "pointStyle": "circle",
#             "pointRadius": 4
#           }
#         },
#         "WaveStrength": {
#           "ew_upward_strength": {
#             "color": "#00FF00",
#             "type": "line"
#           },
#           "ew_downward_strength": {
#             "color": "#FF0000",
#             "type": "line"
#           }
#         },
#         "ConfluenceAndValidation": {
#           "ew_confluence_score": {
#             "color": "#32CD32",
#             "type": "line"
#           },
#           "ew_wave_count": {
#             "color": "#FF69B4",
#             "type": "line"
#           },
#           "ew_fibonacci_valid": {
#             "color": "#1f77b4",
#             "type": "scatter",
#             "pointStyle": "cross",
#             "pointRadius": 4
#           }
#         }
#       }
#     }


# #     # Plot Config 1 :focuses on visualizing the wave1234 points for upward and downward Elliott Waves, 
# #     # grouped to show the sequential progression of waves.
# #     {
# #       "main_plot": {
# #         "smoothed_close": {
# #           "color": "#1f77b4",
# #           "type": "line"
# #         },
# #         "upward_wave1234": {
# #           "color": "#00FF00",
# #           "type": "scatter"
# #         },
# #         "downward_wave1234": {
# #           "color": "#FF0000",
# #           "type": "scatter"
# #         }
# #       },
# #       "subplots": {
# #         "WaveCounts": {
# #           "ew_wave_count": {
# #             "color": "#32CD32",
# #             "type": "line"
# #           }
# #         }
# #       }
# #     }


# # Plot Configuration 2: Entry and Exit Signals
# # This plot visualizes entry and exit signals alongside technical indicators, grouping signals to show their relationship to wave completion.
# # ```json
# # {
# #   "main_plot": {
# #     "smoothed_close": {
# #       "color": "#1f77b4",
# #       "type": "line"
# #     },
# #     "bb_lowerband": {
# #     "color": "#FF69B4",
# #     "type": "line"
# #     },
# #     "bb_upperband": {
# #     "color": "#FF69B4",
# #     "type": "line"
# #     }
# #     "enter_long": {
# #       "color": "#00FF00",
# #       "type": "scatter"
# #     },
# #     "exit_long": {
# #       "color": "#FF0000",
# #       "type": "scatter"
# #     }
# #   },
# #   "subplots": {
# #     "RSI": {
# #       "rsi": {
# #         "color": "#32CD32",
# #         "type": "line"
# #       }
# #     },

# #   }
# # }
# # ```

# # Plot Configuration 3: Wave Strength and Confluence
# # This plot groups wave strength and confluence scores to evaluate signal reliability and wave progression.
# # ```json
# # {
# #   "main_plot": {
# #     "smoothed_close": {
# #       "color": "#1f77b4",
# #       "type": "line"
# #     }
# #   },
# #   "subplots": {
# #     "Confluence": {
# #       "ew_confluence_score": {
# #         "color": "#32CD32",
# #         "type": "line"
# #       }
# #     },
# #     "WaveCount": {
# #       "ew_wave_count": {
# #         "color": "#FF69B4",
# #         "type": "line"
# #       }
# #     },
# #     "EW Strength": {
# #         "ew_upward_strength": {
# #           "color": "#00FF00",
# #           "type": "line"
# #         },
# #         "ew_downward_strength": {
# #           "color": "#FF0000",
# #           "type": "line"
# #         }
# #     }
# #   }
# # }
# # ```



#     plot_config = {
#         "main_plot": {
#             "smoothed_close": {
#                 "color": "#1f77b4",
#                 "type": "line"
#             },
#             "ew_upward_wave5": {
#                 "color": "#00FF00",
#                 "type": "scatter",
#                 "pointStyle": "circle",
#                 "pointRadius": 6
#             },
#             "ew_downward_wave5": {
#                 "color": "#FF0000",
#                 "type": "scatter",
#                 "pointStyle": "circle",
#                 "pointRadius": 6
#             },
#             "ew_practical12_upward": {
#                 "color": "#32CD32",
#                 "type": "scatter",
#                 "pointStyle": "triangle",
#                 "pointRadius": 5
#             },
#             "ew_practical12_downward": {
#                 "color": "#FF69B4",
#                 "type": "scatter",
#                 "pointStyle": "triangle",
#                 "pointRadius": 5
#             }
#         },
#         "subplots": {
#             "WaveSignals": {
#                 "ew_traditional_upward": {
#                     "color": "#00FF00",
#                     "type": "scatter",
#                     "pointStyle": "square",
#                     "pointRadius": 4
#                 },
#                 "ew_traditional_downward": {
#                     "color": "#FF0000",
#                     "type": "scatter",
#                     "pointStyle": "square",
#                     "pointRadius": 4
#                 },
#                 "ew_alternative_upward": {
#                     "color": "#32CD32",
#                     "type": "scatter",
#                     "pointStyle": "diamond",
#                     "pointRadius": 4
#                 },
#                 "ew_alternative_downward": {
#                     "color": "#FF69B4",
#                     "type": "scatter",
#                     "pointStyle": "diamond",
#                     "pointRadius": 4
#                 },
#                 "ew_zigzag_upward": {
#                     "color": "#00FF00",
#                     "type": "scatter",
#                     "pointStyle": "cross",
#                     "pointRadius": 4
#                 },
#                 "ew_zigzag_downward": {
#                     "color": "#FF0000",
#                     "type": "scatter",
#                     "pointStyle": "cross",
#                     "pointRadius": 4
#                 },
#                 "ew_upward_wave3": {
#                     "color": "#32CD32",
#                     "type": "scatter",
#                     "pointStyle": "circle",
#                     "pointRadius": 4
#                 },
#                 "ew_downward_wave3": {
#                     "color": "#FF69B4",
#                     "type": "scatter",
#                     "pointStyle": "circle",
#                     "pointRadius": 4
#                 }
#             },
#             "WaveStrength": {
#                 "ew_upward_strength": {
#                     "color": "#00FF00",
#                     "type": "line"
#                 },
#                 "ew_downward_strength": {
#                     "color": "#FF0000",
#                     "type": "line"
#                 }
#             },
#             "ConfluenceAndValidation": {
#                 "ew_confluence_score": {
#                     "color": "#32CD32",
#                     "type": "line"
#                 },
#                 "ew_wave_count": {
#                     "color": "#FF69B4",
#                     "type": "line"
#                 },
#                 "ew_fibonacci_valid": {
#                     "color": "#1f77b4",
#                     "type": "scatter",
#                     "pointStyle": "cross",
#                     "pointRadius": 4
#                 }
#             }
#         }
#     }


# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union
from functools import reduce
import talib.abstract as ta
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IStrategy, IntParameter)
from technical import qtpylib
import copy


# Utility Functions
def diff(data):
    output_diff = []
    for i in range(1, len(data)):
        output_diff.append(data[i - 1] - data[i])
    return output_diff


def otherThan(data, otherthan=0):
    output_otherthan = []
    for i in range(len(data)):
        if data[i] != otherthan:
            output_otherthan.append(True)
        else:
            output_otherthan.append(False)
    return output_otherthan


def trimming(data, determineArray):
    if len(data) != len(determineArray):
        raise Exception('array/list size not equal')
    filtered_data = []
    for i in range(len(data)):
        if determineArray[i]:
            filtered_data.append(data[i])
    return filtered_data


# Fibonacci Validation Functions (thesis-inspired tolerance: ±10%)
def wave2_fibonacci_check(wave2_end, wave1_start, wave1_end):
    wave2_endabs = abs(wave2_end)
    wave1_startabs = abs(wave1_start)
    wave1_endabs = abs(wave1_end)
    fibonacci_ratio = [0.236, 0.382, 0.5, 0.618]
    for ratio in fibonacci_ratio:
        target = wave1_endabs - (wave1_endabs - wave1_startabs) * ratio
        if wave2_endabs <= target * 1.1 and wave2_endabs >= target * 0.9:
            return True
    return False


def wave3_fibonacci_check(wave3_end, wave2_start, wave2_end):
    wave3_endabs = abs(wave3_end)
    wave2_startabs = abs(wave2_start)
    wave2_endabs = abs(wave2_end)
    fibonacci_ratio = [1.236, 1.618, 2.0, 2.618]
    for ratio in fibonacci_ratio:
        target = wave2_endabs - (wave2_endabs - wave2_startabs) * ratio
        if wave3_endabs <= target * 1.1 and wave3_endabs >= target * 0.9:
            return True
    return False


def wave4_fibonacci_check(wave4_end, wave3_start, wave3_end):
    wave4_endabs = abs(wave4_end)
    wave3_startabs = abs(wave3_start)
    wave3_endabs = abs(wave3_end)
    fibonacci_ratio = [0.236, 0.382, 0.5, 0.618]
    for ratio in fibonacci_ratio:
        target = wave3_endabs - (wave3_endabs - wave3_startabs) * ratio
        if wave4_endabs <= target * 1.1 and wave4_endabs >= target * 0.9:
            return True
    return False


def wave5_fibonacci_check(wave5_end, wave1_start, wave1_end, wave3_start, wave3_end, wave4_end):
    wave5_endabs = abs(wave5_end)
    wave1_startabs = abs(wave1_start)
    wave1_endabs = abs(wave1_end)
    wave3_startabs = abs(wave3_start)
    wave3_endabs = abs(wave3_end)
    wave4_endabs = abs(wave4_end)
    wave5_y = abs(wave5_endabs - wave4_endabs)
    wave1_y = abs(wave1_endabs - wave1_startabs)
    wave4_y = abs(wave4_endabs - wave3_endabs)
    wave1plus3_y = wave1_y + abs(wave3_endabs - wave3_startabs)
    if wave5_y >= wave4_y * 0.9 and wave5_y <= wave4_y * 2:
        return True
    if wave5_y >= wave1_y * 0.9 and wave5_y <= wave1_y * 1.1:
        return True
    fibonacci_ratio = [0.382, 0.618, 0.764]
    for ratio in fibonacci_ratio:
        if wave5_y <= wave1plus3_y * ratio * 1.1 and wave5_y >= wave1plus3_y * ratio * 0.9:
            return True
    return False


# Wave Length Validation (thesis-inspired: filter noise)
def wave_length_validation(wave_indices: list, min_length: int = 5, max_length: int = 50) -> bool:
    if len(wave_indices) < 2:
        return False
    duration = wave_indices[-1] - wave_indices[0]
    return min_length <= duration <= max_length


# Elliott Wave Detection Functions
def Traditional_ElliottWave_label_upward(data, start_idx: int = 0):
    v = data
    j = range(start_idx, start_idx + len(data))
    x = []
    z = []
    for i in range(1, len(v) - 1):
        if (v[i] <= v[i + 1] and v[i - 1] >= v[i]) or (v[i] >= v[i + 1] and v[i - 1] <= v[i]):
            x.append(v[i])
            z.append(j[i])
            diff4x = diff(x)
            diff4x.insert(0, 1)
            diff4z = diff(z)
            diff4z.insert(0, 1)
            x = trimming(x, otherThan(diff4x, otherthan=0))
            z = trimming(z, otherThan(diff4z, otherthan=0))
    listofCandidateWave = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[i] < x[j]:
                wave = {'x': [x[i], x[j]], 'z': [z[i], z[j]], 'searchIndex': j}
                listofCandidateWave.append(wave)
    # print(f"Traditional upward candidate wave: {len(listofCandidateWave)}")
    listofCandidateWave12 = []
    for i in range(len(listofCandidateWave)):
        startSearchIndex = listofCandidateWave[i]['searchIndex']
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] < listofCandidateWave[i]['x'][1] and x[j] > listofCandidateWave[i]['x'][0] and
                wave2_fibonacci_check(x[j], listofCandidateWave[i]['x'][0], listofCandidateWave[i]['x'][1])):
                currWave = copy.deepcopy(listofCandidateWave[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                listofCandidateWave12.append(currWave)
    # print(f"Traditional upward candidate wave12: {len(listofCandidateWave12)}")
    listofCandidateWave123 = []
    for i in range(len(listofCandidateWave12)):
        startSearchIndex = listofCandidateWave12[i]['searchIndex']
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] > listofCandidateWave12[i]['x'][2] and
                (z[j] - listofCandidateWave12[i]['z'][2]) >= (listofCandidateWave12[i]['z'][1] - listofCandidateWave12[i]['z'][0]) and
                (z[j] - listofCandidateWave12[i]['z'][2]) >= (listofCandidateWave12[i]['z'][2] - listofCandidateWave12[i]['z'][1]) and
                wave3_fibonacci_check(x[j], listofCandidateWave12[i]['x'][1], listofCandidateWave12[i]['x'][2])):
                currWave = copy.deepcopy(listofCandidateWave12[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                listofCandidateWave123.append(currWave)
    # print(f"Traditional upward candidate wave123: {len(listofCandidateWave123)}")
    listofCandidateWave1234 = []
    for i in range(len(listofCandidateWave123)):
        startSearchIndex = listofCandidateWave123[i]['searchIndex']
        timeInterval = (listofCandidateWave123[i]['z'][1] - listofCandidateWave123[i]['z'][0]) * 0.618
        wave3length = (listofCandidateWave123[i]['z'][3] - listofCandidateWave123[i]['z'][2]) * 1.5
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] < listofCandidateWave123[i]['x'][3] and
                (z[j] - listofCandidateWave123[i]['z'][3]) <= timeInterval and
                x[j] > listofCandidateWave123[i]['x'][1] and
                (z[j] - listofCandidateWave123[i]['z'][3]) <= wave3length and
                wave4_fibonacci_check(x[j], listofCandidateWave123[i]['x'][2], listofCandidateWave123[i]['x'][3])):
                currWave = copy.deepcopy(listofCandidateWave123[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                if wave_length_validation(currWave['z']):
                    listofCandidateWave1234.append(currWave)
    # print(f"Traditional upward candidate wave1234: {len(listofCandidateWave1234)}")
    return listofCandidateWave1234


def Traditional_ElliottWave_label_downward(data, start_idx: int = 0):
    v = data
    j = range(start_idx, start_idx + len(data))
    x = []
    z = []
    for i in range(1, len(v) - 1):
        if (v[i] <= v[i + 1] and v[i - 1] >= v[i]) or (v[i] >= v[i + 1] and v[i - 1] <= v[i]):
            x.append(v[i])
            z.append(j[i])
            diff4x = diff(x)
            diff4x.insert(0, 1)
            diff4z = diff(z)
            diff4z.insert(0, 1)
            x = trimming(x, otherThan(diff4x, otherthan=0))
            z = trimming(z, otherThan(diff4z, otherthan=0))
    listofCandidateWave = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[i] > x[j]:
                wave = {'x': [x[i], x[j]], 'z': [z[i], z[j]], 'searchIndex': j}
                listofCandidateWave.append(wave)
    # print(f"Traditional downward candidate wave: {len(listofCandidateWave)}")
    listofCandidateWave12 = []
    for i in range(len(listofCandidateWave)):
        startSearchIndex = listofCandidateWave[i]['searchIndex']
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] > listofCandidateWave[i]['x'][1] and x[j] < listofCandidateWave[i]['x'][0] and
                wave2_fibonacci_check(x[j], listofCandidateWave[i]['x'][0], listofCandidateWave[i]['x'][1])):
                currWave = copy.deepcopy(listofCandidateWave[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                listofCandidateWave12.append(currWave)
    # print(f"Traditional downward candidate wave12: {len(listofCandidateWave12)}")
    listofCandidateWave123 = []
    for i in range(len(listofCandidateWave12)):
        startSearchIndex = listofCandidateWave12[i]['searchIndex']
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] < listofCandidateWave12[i]['x'][2] and
                (z[j] - listofCandidateWave12[i]['z'][2]) >= (listofCandidateWave12[i]['z'][1] - listofCandidateWave12[i]['z'][0]) and
                (z[j] - listofCandidateWave12[i]['z'][2]) >= (listofCandidateWave12[i]['z'][2] - listofCandidateWave12[i]['z'][1]) and
                wave3_fibonacci_check(x[j], listofCandidateWave12[i]['x'][1], listofCandidateWave12[i]['x'][2])):
                currWave = copy.deepcopy(listofCandidateWave12[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                listofCandidateWave123.append(currWave)
    # print(f"Traditional downward candidate wave123: {len(listofCandidateWave123)}")
    listofCandidateWave1234 = []
    for i in range(len(listofCandidateWave123)):
        startSearchIndex = listofCandidateWave123[i]['searchIndex']
        timeInterval = (listofCandidateWave123[i]['z'][1] - listofCandidateWave123[i]['z'][0]) * 0.618
        wave3length = (listofCandidateWave123[i]['z'][3] - listofCandidateWave123[i]['z'][2]) * 1.5
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] > listofCandidateWave123[i]['x'][3] and
                (z[j] - listofCandidateWave123[i]['z'][3]) <= timeInterval and
                x[j] < listofCandidateWave123[i]['x'][1] and
                (z[j] - listofCandidateWave123[i]['z'][3]) <= wave3length and
                wave4_fibonacci_check(x[j], listofCandidateWave123[i]['x'][2], listofCandidateWave123[i]['x'][3])):
                currWave = copy.deepcopy(listofCandidateWave123[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                if wave_length_validation(currWave['z']):
                    listofCandidateWave1234.append(currWave)
    # print(f"Traditional downward candidate wave1234: {len(listofCandidateWave1234)}")
    return listofCandidateWave1234


def Alternative_ElliottWave_label_upward(data, start_idx: int = 0):
    v = data
    j = range(start_idx, start_idx + len(data))
    x = []
    z = []
    for i in range(1, len(v) - 1):
        if (v[i] <= v[i + 1] and v[i - 1] >= v[i]) or (v[i] >= v[i + 1] and v[i - 1] <= v[i]):
            x.append(v[i])
            z.append(j[i])
            diff4x = diff(x)
            diff4x.insert(0, 1)
            diff4z = diff(z)
            diff4z.insert(0, 1)
            x = trimming(x, otherThan(diff4x, otherthan=0))
            z = trimming(z, otherThan(diff4z, otherthan=0))
    listofCandidateWave = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[i] < x[j]:
                wave = {'x': [x[i], x[j]], 'z': [z[i], z[j]], 'searchIndex': j}
                listofCandidateWave.append(wave)
    # print(f"Alternative upward candidate wave: {len(listofCandidateWave)}")
    listofCandidateWave12 = []
    for i in range(len(listofCandidateWave)):
        startSearchIndex = listofCandidateWave[i]['searchIndex']
        timeInterval = (listofCandidateWave[i]['z'][1] - listofCandidateWave[i]['z'][0]) * 0.618
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] < listofCandidateWave[i]['x'][1] and
                (z[j] - listofCandidateWave[i]['z'][1]) <= timeInterval and
                x[j] > listofCandidateWave[i]['x'][0] and
                wave2_fibonacci_check(x[j], listofCandidateWave[i]['x'][0], listofCandidateWave[i]['x'][1])):
                currWave = copy.deepcopy(listofCandidateWave[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                listofCandidateWave12.append(currWave)
    # print(f"Alternative upward candidate wave12: {len(listofCandidateWave12)}")
    listofCandidateWave123 = []
    for i in range(len(listofCandidateWave12)):
        startSearchIndex = listofCandidateWave12[i]['searchIndex']
        timeInterval = (listofCandidateWave12[i]['z'][1] - listofCandidateWave12[i]['z'][0]) * 2.0
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] > listofCandidateWave12[i]['x'][2] and
                (z[j] - listofCandidateWave12[i]['z'][2]) <= timeInterval and
                (z[j] - listofCandidateWave12[i]['z'][2]) >= (listofCandidateWave12[i]['z'][1] - listofCandidateWave12[i]['z'][0]) and
                (z[j] - listofCandidateWave12[i]['z'][2]) >= (listofCandidateWave12[i]['z'][2] - listofCandidateWave12[i]['z'][1]) and
                wave3_fibonacci_check(x[j], listofCandidateWave12[i]['x'][1], listofCandidateWave12[i]['x'][2])):
                currWave = copy.deepcopy(listofCandidateWave12[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                listofCandidateWave123.append(currWave)
    # print(f"Alternative upward candidate wave123: {len(listofCandidateWave123)}")
    listofCandidateWave1234 = []
    for i in range(len(listofCandidateWave123)):
        startSearchIndex = listofCandidateWave123[i]['searchIndex']
        timeInterval = (listofCandidateWave123[i]['z'][1] - listofCandidateWave123[i]['z'][0]) * 0.618
        wave3length = (listofCandidateWave123[i]['z'][3] - listofCandidateWave123[i]['z'][2]) * 1.5
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] < listofCandidateWave123[i]['x'][3] and
                (z[j] - listofCandidateWave123[i]['z'][3]) <= timeInterval and
                x[j] > listofCandidateWave123[i]['x'][1] and
                (z[j] - listofCandidateWave123[i]['z'][3]) <= wave3length and
                wave4_fibonacci_check(x[j], listofCandidateWave123[i]['x'][2], listofCandidateWave123[i]['x'][3])):
                currWave = copy.deepcopy(listofCandidateWave123[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                if wave_length_validation(currWave['z']):
                    listofCandidateWave1234.append(currWave)
    # print(f"Alternative upward candidate wave1234: {len(listofCandidateWave1234)}")
    return listofCandidateWave1234


def Alternative_ElliottWave_label_downward(data, start_idx: int = 0):
    v = data
    j = range(start_idx, start_idx + len(data))
    x = []
    z = []
    for i in range(1, len(v) - 1):
        if (v[i] <= v[i + 1] and v[i - 1] >= v[i]) or (v[i] >= v[i + 1] and v[i - 1] <= v[i]):
            x.append(v[i])
            z.append(j[i])
            diff4x = diff(x)
            diff4x.insert(0, 1)
            diff4z = diff(z)
            diff4z.insert(0, 1)
            x = trimming(x, otherThan(diff4x, otherthan=0))
            z = trimming(z, otherThan(diff4z, otherthan=0))
    listofCandidateWave = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[i] > x[j]:
                wave = {'x': [x[i], x[j]], 'z': [z[i], z[j]], 'searchIndex': j}
                listofCandidateWave.append(wave)
    # print(f"Alternative downward candidate wave: {len(listofCandidateWave)}")
    listofCandidateWave12 = []
    for i in range(len(listofCandidateWave)):
        startSearchIndex = listofCandidateWave[i]['searchIndex']
        timeInterval = (listofCandidateWave[i]['z'][1] - listofCandidateWave[i]['z'][0]) * 0.618
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] > listofCandidateWave[i]['x'][1] and
                (z[j] - listofCandidateWave[i]['z'][1]) <= timeInterval and
                x[j] < listofCandidateWave[i]['x'][0] and
                wave2_fibonacci_check(x[j], listofCandidateWave[i]['x'][0], listofCandidateWave[i]['x'][1])):
                currWave = copy.deepcopy(listofCandidateWave[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                listofCandidateWave12.append(currWave)
    # print(f"Alternative downward candidate wave12: {len(listofCandidateWave12)}")
    listofCandidateWave123 = []
    for i in range(len(listofCandidateWave12)):
        startSearchIndex = listofCandidateWave12[i]['searchIndex']
        timeInterval = (listofCandidateWave12[i]['z'][1] - listofCandidateWave12[i]['z'][0]) * 2.0
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] < listofCandidateWave12[i]['x'][2] and
                (z[j] - listofCandidateWave12[i]['z'][2]) <= timeInterval and
                (z[j] - listofCandidateWave12[i]['z'][2]) >= (listofCandidateWave12[i]['z'][1] - listofCandidateWave12[i]['z'][0]) and
                (z[j] - listofCandidateWave12[i]['z'][2]) >= (listofCandidateWave12[i]['z'][2] - listofCandidateWave12[i]['z'][1]) and
                wave3_fibonacci_check(x[j], listofCandidateWave12[i]['x'][1], listofCandidateWave12[i]['x'][2])):
                currWave = copy.deepcopy(listofCandidateWave12[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                listofCandidateWave123.append(currWave)
    # print(f"Alternative downward candidate wave123: {len(listofCandidateWave123)}")
    listofCandidateWave1234 = []
    for i in range(len(listofCandidateWave123)):
        startSearchIndex = listofCandidateWave123[i]['searchIndex']
        timeInterval = (listofCandidateWave123[i]['z'][1] - listofCandidateWave123[i]['z'][0]) * 0.618
        wave3length = (listofCandidateWave123[i]['z'][3] - listofCandidateWave123[i]['z'][2]) * 1.5
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] > listofCandidateWave123[i]['x'][3] and
                (z[j] - listofCandidateWave123[i]['z'][3]) <= timeInterval and
                x[j] < listofCandidateWave123[i]['x'][1] and
                (z[j] - listofCandidateWave123[i]['z'][3]) <= wave3length and
                wave4_fibonacci_check(x[j], listofCandidateWave123[i]['x'][2], listofCandidateWave123[i]['x'][3])):
                currWave = copy.deepcopy(listofCandidateWave123[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                if wave_length_validation(currWave['z']):
                    listofCandidateWave1234.append(currWave)
    # print(f"Alternative downward candidate wave1234: {len(listofCandidateWave1234)}")
    return listofCandidateWave1234


def Practical_ElliottWave3_label_upward(data, start_idx: int = 0):
    v = data
    j = range(start_idx, start_idx + len(data))
    x = []
    z = []
    for i in range(1, len(v) - 1):
        if (v[i] <= v[i + 1] and v[i - 1] >= v[i]) or (v[i] >= v[i + 1] and v[i - 1] <= v[i]):
            x.append(v[i])
            z.append(j[i])
            diff4x = diff(x)
            diff4x.insert(0, 1)
            diff4z = diff(z)
            diff4z.insert(0, 1)
            x = trimming(x, otherThan(diff4x, otherthan=0))
            z = trimming(z, otherThan(diff4z, otherthan=0))
    listofCandidateWave = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[i] < x[j]:
                wave = {'x': [x[i], x[j]], 'z': [z[i], z[j]], 'searchIndex': j}
                listofCandidateWave.append(wave)
    # print(f"Practical upward wave3 candidate wave: {len(listofCandidateWave)}")
    listofCandidateWave12 = []
    for i in range(len(listofCandidateWave)):
        startSearchIndex = listofCandidateWave[i]['searchIndex']
        timeInterval = (listofCandidateWave[i]['z'][1] - listofCandidateWave[i]['z'][0]) * 0.4011
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] < listofCandidateWave[i]['x'][1] and
                (z[j] - listofCandidateWave[i]['z'][1]) <= timeInterval and
                x[j] > listofCandidateWave[i]['x'][0] and
                wave2_fibonacci_check(x[j], listofCandidateWave[i]['x'][0], listofCandidateWave[i]['x'][1])):
                currWave = copy.deepcopy(listofCandidateWave[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                listofCandidateWave12.append(currWave)
    # print(f"Practical upward wave3 candidate wave12: {len(listofCandidateWave12)}")
    listofCandidateWave123 = []
    for i in range(len(listofCandidateWave12)):
        startSearchIndex = listofCandidateWave12[i]['searchIndex']
        timeInterval = (listofCandidateWave12[i]['z'][1] - listofCandidateWave12[i]['z'][0]) * 1.618
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] > listofCandidateWave12[i]['x'][2] and
                (z[j] - listofCandidateWave12[i]['z'][2]) <= timeInterval and
                wave3_fibonacci_check(x[j], listofCandidateWave12[i]['x'][1], listofCandidateWave12[i]['x'][2])):
                currWave = copy.deepcopy(listofCandidateWave12[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                listofCandidateWave123.append(currWave)
    # print(f"Practical upward wave3 candidate wave123: {len(listofCandidateWave123)}")
    return listofCandidateWave123


def Practical_ElliottWave3_label_downward(data, start_idx: int = 0):
    v = data
    j = range(start_idx, start_idx + len(data))
    x = []
    z = []
    for i in range(1, len(v) - 1):
        if (v[i] <= v[i + 1] and v[i - 1] >= v[i]) or (v[i] >= v[i + 1] and v[i - 1] <= v[i]):
            x.append(v[i])
            z.append(j[i])
            diff4x = diff(x)
            diff4x.insert(0, 1)
            diff4z = diff(z)
            diff4z.insert(0, 1)
            x = trimming(x, otherThan(diff4x, otherthan=0))
            z = trimming(z, otherThan(diff4z, otherthan=0))
    listofCandidateWave = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[i] > x[j]:
                wave = {'x': [x[i], x[j]], 'z': [z[i], z[j]], 'searchIndex': j}
                listofCandidateWave.append(wave)
    # print(f"Practical downward wave3 candidate wave: {len(listofCandidateWave)}")
    listofCandidateWave12 = []
    for i in range(len(listofCandidateWave)):
        startSearchIndex = listofCandidateWave[i]['searchIndex']
        timeInterval = (listofCandidateWave[i]['z'][1] - listofCandidateWave[i]['z'][0]) * 0.4011
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] > listofCandidateWave[i]['x'][1] and
                (z[j] - listofCandidateWave[i]['z'][1]) <= timeInterval and
                x[j] < listofCandidateWave[i]['x'][0] and
                wave2_fibonacci_check(x[j], listofCandidateWave[i]['x'][0], listofCandidateWave[i]['x'][1])):
                currWave = copy.deepcopy(listofCandidateWave[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                listofCandidateWave12.append(currWave)
    # print(f"Practical downward wave3 candidate wave12: {len(listofCandidateWave12)}")
    listofCandidateWave123 = []
    for i in range(len(listofCandidateWave12)):
        startSearchIndex = listofCandidateWave12[i]['searchIndex']
        timeInterval = (listofCandidateWave12[i]['z'][1] - listofCandidateWave12[i]['z'][0]) * 1.618
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] < listofCandidateWave12[i]['x'][2] and
                (z[j] - listofCandidateWave12[i]['z'][2]) <= timeInterval and
                wave3_fibonacci_check(x[j], listofCandidateWave12[i]['x'][1], listofCandidateWave12[i]['x'][2])):
                currWave = copy.deepcopy(listofCandidateWave12[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                listofCandidateWave123.append(currWave)
    # print(f"Practical downward wave3 candidate wave123: {len(listofCandidateWave123)}")
    return listofCandidateWave123


def Practical_ElliottWave5_label_upward(data, start_idx: int = 0):
    v = data
    j = range(start_idx, start_idx + len(data))
    x = []
    z = []
    for i in range(1, len(v) - 1):
        if (v[i] <= v[i + 1] and v[i - 1] >= v[i]) or (v[i] >= v[i + 1] and v[i - 1] <= v[i]):
            x.append(v[i])
            z.append(j[i])
            diff4x = diff(x)
            diff4x.insert(0, 1)
            diff4z = diff(z)
            diff4z.insert(0, 1)
            x = trimming(x, otherThan(diff4x, otherthan=0))
            z = trimming(z, otherThan(diff4z, otherthan=0))
    listofCandidateWave = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[i] < x[j]:
                wave = {'x': [x[i], x[j]], 'z': [z[i], z[j]], 'searchIndex': j}
                listofCandidateWave.append(wave)
    # print(f"Practical upward wave5 candidate wave: {len(listofCandidateWave)}")
    listofCandidateWave12 = []
    for i in range(len(listofCandidateWave)):
        startSearchIndex = listofCandidateWave[i]['searchIndex']
        timeInterval = (listofCandidateWave[i]['z'][1] - listofCandidateWave[i]['z'][0]) * 0.4011
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] < listofCandidateWave[i]['x'][1] and
                (z[j] - listofCandidateWave[i]['z'][1]) <= timeInterval and
                x[j] > listofCandidateWave[i]['x'][0] and
                wave2_fibonacci_check(x[j], listofCandidateWave[i]['x'][0], listofCandidateWave[i]['x'][1])):
                currWave = copy.deepcopy(listofCandidateWave[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                listofCandidateWave12.append(currWave)
    # print(f"Practical upward wave5 candidate wave12: {len(listofCandidateWave12)}")
    listofCandidateWave123 = []
    for i in range(len(listofCandidateWave12)):
        startSearchIndex = listofCandidateWave12[i]['searchIndex']
        timeInterval = (listofCandidateWave12[i]['z'][1] - listofCandidateWave12[i]['z'][0]) * 1.618
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] > listofCandidateWave12[i]['x'][2] and
                (z[j] - listofCandidateWave12[i]['z'][2]) <= timeInterval and
                wave3_fibonacci_check(x[j], listofCandidateWave12[i]['x'][1], listofCandidateWave12[i]['x'][2])):
                currWave = copy.deepcopy(listofCandidateWave12[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                listofCandidateWave123.append(currWave)
    # print(f"Practical upward wave5 candidate wave123: {len(listofCandidateWave123)}")
    listofCandidateWave1234 = []
    for i in range(len(listofCandidateWave123)):
        startSearchIndex = listofCandidateWave123[i]['searchIndex']
        timeInterval = (listofCandidateWave123[i]['z'][1] - listofCandidateWave123[i]['z'][0]) * 0.618
        wave3length = (listofCandidateWave123[i]['z'][3] - listofCandidateWave123[i]['z'][2]) * 1.5
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] < listofCandidateWave123[i]['x'][3] and
                (z[j] - listofCandidateWave123[i]['z'][3]) <= timeInterval and
                x[j] > listofCandidateWave123[i]['x'][1] and
                (z[j] - listofCandidateWave123[i]['z'][3]) <= wave3length and
                wave4_fibonacci_check(x[j], listofCandidateWave123[i]['x'][2], listofCandidateWave123[i]['x'][3])):
                currWave = copy.deepcopy(listofCandidateWave123[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                if wave_length_validation(currWave['z']):
                    listofCandidateWave1234.append(currWave)
    # print(f"Practical upward wave5 candidate wave1234: {len(listofCandidateWave1234)}")
    return listofCandidateWave1234


def Practical_ElliottWave5_label_downward(data, start_idx: int = 0):
    v = data
    j = range(start_idx, start_idx + len(data))
    x = []
    z = []
    for i in range(1, len(v) - 1):
        if (v[i] <= v[i + 1] and v[i - 1] >= v[i]) or (v[i] >= v[i + 1] and v[i - 1] <= v[i]):
            x.append(v[i])
            z.append(j[i])
            diff4x = diff(x)
            diff4x.insert(0, 1)
            diff4z = diff(z)
            diff4z.insert(0, 1)
            x = trimming(x, otherThan(diff4x, otherthan=0))
            z = trimming(z, otherThan(diff4z, otherthan=0))
    listofCandidateWave = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[i] > x[j]:
                wave = {'x': [x[i], x[j]], 'z': [z[i], z[j]], 'searchIndex': j}
                listofCandidateWave.append(wave)
    # print(f"Practical downward wave5 candidate wave: {len(listofCandidateWave)}")
    listofCandidateWave12 = []
    for i in range(len(listofCandidateWave)):
        startSearchIndex = listofCandidateWave[i]['searchIndex']
        timeInterval = (listofCandidateWave[i]['z'][1] - listofCandidateWave[i]['z'][0]) * 0.4011
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] > listofCandidateWave[i]['x'][1] and
                (z[j] - listofCandidateWave[i]['z'][1]) <= timeInterval and
                x[j] < listofCandidateWave[i]['x'][0] and
                wave2_fibonacci_check(x[j], listofCandidateWave[i]['x'][0], listofCandidateWave[i]['x'][1])):
                currWave = copy.deepcopy(listofCandidateWave[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                listofCandidateWave12.append(currWave)
    # print(f"Practical downward wave5 candidate wave12: {len(listofCandidateWave12)}")
    listofCandidateWave123 = []
    for i in range(len(listofCandidateWave12)):
        startSearchIndex = listofCandidateWave12[i]['searchIndex']
        timeInterval = (listofCandidateWave12[i]['z'][1] - listofCandidateWave12[i]['z'][0]) * 1.618
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] < listofCandidateWave12[i]['x'][2] and
                (z[j] - listofCandidateWave12[i]['z'][2]) <= timeInterval and
                wave3_fibonacci_check(x[j], listofCandidateWave12[i]['x'][1], listofCandidateWave12[i]['x'][2])):
                currWave = copy.deepcopy(listofCandidateWave12[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                listofCandidateWave123.append(currWave)
    # print(f"Practical downward wave5 candidate wave123: {len(listofCandidateWave123)}")
    listofCandidateWave1234 = []
    for i in range(len(listofCandidateWave123)):
        startSearchIndex = listofCandidateWave123[i]['searchIndex']
        timeInterval = (listofCandidateWave123[i]['z'][1] - listofCandidateWave123[i]['z'][0]) * 0.618
        wave3length = (listofCandidateWave123[i]['z'][3] - listofCandidateWave123[i]['z'][2]) * 1.5
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] > listofCandidateWave123[i]['x'][3] and
                (z[j] - listofCandidateWave123[i]['z'][3]) <= timeInterval and
                x[j] < listofCandidateWave123[i]['x'][1] and
                (z[j] - listofCandidateWave123[i]['z'][3]) <= wave3length and
                wave4_fibonacci_check(x[j], listofCandidateWave123[i]['x'][2], listofCandidateWave123[i]['x'][3])):
                currWave = copy.deepcopy(listofCandidateWave123[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                if wave_length_validation(currWave['z']):
                    listofCandidateWave1234.append(currWave)
    # print(f"Practical downward wave5 candidate wave1234: {len(listofCandidateWave1234)}")
    return listofCandidateWave1234


def Practical_ElliottWave12_label_upward(data, start_idx: int = 0):
    v = data
    j = range(start_idx, start_idx + len(data))
    x = []
    z = []
    for i in range(1, len(v) - 1):
        if (v[i] <= v[i + 1] and v[i - 1] >= v[i]) or (v[i] >= v[i + 1] and v[i - 1] <= v[i]):
            x.append(v[i])
            z.append(j[i])
            diff4x = diff(x)
            diff4x.insert(0, 1)
            diff4z = diff(z)
            diff4z.insert(0, 1)
            x = trimming(x, otherThan(diff4x, otherthan=0))
            z = trimming(z, otherThan(diff4z, otherthan=0))
    listofCandidateWave = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[i] < x[j]:
                wave = {'x': [x[i], x[j]], 'z': [z[i], z[j]], 'searchIndex': j}
                listofCandidateWave.append(wave)
    # print(f"Practical upward wave12 candidate wave: {len(listofCandidateWave)}")
    listofCandidateWave12 = []
    for i in range(len(listofCandidateWave)):
        startSearchIndex = listofCandidateWave[i]['searchIndex']
        timeInterval = (listofCandidateWave[i]['z'][1] - listofCandidateWave[i]['z'][0]) * 0.4011
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] < listofCandidateWave[i]['x'][1] and
                (z[j] - listofCandidateWave[i]['z'][1]) <= timeInterval and
                x[j] > listofCandidateWave[i]['x'][0] and
                wave2_fibonacci_check(x[j], listofCandidateWave[i]['x'][0], listofCandidateWave[i]['x'][1])):
                currWave = copy.deepcopy(listofCandidateWave[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                listofCandidateWave12.append(currWave)
    # print(f"Practical upward wave12 candidate wave12: {len(listofCandidateWave12)}")
    return listofCandidateWave12


def Practical_ElliottWave12_label_downward(data, start_idx: int = 0):
    v = data
    j = range(start_idx, start_idx + len(data))
    x = []
    z = []
    for i in range(1, len(v) - 1):
        if (v[i] <= v[i + 1] and v[i - 1] >= v[i]) or (v[i] >= v[i + 1] and v[i - 1] <= v[i]):
            x.append(v[i])
            z.append(j[i])
            diff4x = diff(x)
            diff4x.insert(0, 1)
            diff4z = diff(z)
            diff4z.insert(0, 1)
            x = trimming(x, otherThan(diff4x, otherthan=0))
            z = trimming(z, otherThan(diff4z, otherthan=0))
    listofCandidateWave = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[i] > x[j]:
                wave = {'x': [x[i], x[j]], 'z': [z[i], z[j]], 'searchIndex': j}
                listofCandidateWave.append(wave)
    # print(f"Practical downward wave12 candidate wave: {len(listofCandidateWave)}")
    listofCandidateWave12 = []
    for i in range(len(listofCandidateWave)):
        startSearchIndex = listofCandidateWave[i]['searchIndex']
        timeInterval = (listofCandidateWave[i]['z'][1] - listofCandidateWave[i]['z'][0]) * 0.4011
        for j in range(startSearchIndex + 1, len(x)):
            if (x[j] > listofCandidateWave[i]['x'][1] and
                (z[j] - listofCandidateWave[i]['z'][1]) <= timeInterval and
                x[j] < listofCandidateWave[i]['x'][0] and
                wave2_fibonacci_check(x[j], listofCandidateWave[i]['x'][0], listofCandidateWave[i]['x'][1])):
                currWave = copy.deepcopy(listofCandidateWave[i])
                currWave['x'].append(x[j])
                currWave['z'].append(z[j])
                currWave['searchIndex'] = j
                listofCandidateWave12.append(currWave)
    # print(f"Practical downward wave12 candidate wave12: {len(listofCandidateWave12)}")
    return listofCandidateWave12


# ZigZag-based Wave Detection (thesis-inspired, robust swing points)
def ZigZag_ElliottWave_label_upward(dataframe: DataFrame, start_idx: int = 0, lookback: int = 100, min_change: float = 0.00) -> list:
    prices = dataframe['smoothed_close'].iloc[start_idx:start_idx + lookback].values
    if len(prices) < lookback:
        # print(f"ZigZag upward: Insufficient data, len(prices)={len(prices)}, lookback={lookback}")
        return []
    indices = np.arange(start_idx, start_idx + len(prices))
    zigzag = []
    direction = 0
    last_pivot = prices[0]
    pivot_indices = [0]
    for i in range(1, len(prices)):
        price_change = (prices[i] - last_pivot) / last_pivot
        if direction == 0:
            if price_change > min_change:
                direction = 1
                zigzag.append({'x': last_pivot, 'z': start_idx + pivot_indices[-1]})
                last_pivot = prices[i]
                pivot_indices.append(i)
            elif price_change < -min_change:
                direction = -1
                zigzag.append({'x': last_pivot, 'z': start_idx + pivot_indices[-1]})
                last_pivot = prices[i]
                pivot_indices.append(i)
        elif direction == 1 and price_change < -min_change:
            direction = -1
            zigzag.append({'x': last_pivot, 'z': start_idx + pivot_indices[-1]})
            last_pivot = prices[i]
            pivot_indices.append(i)
        elif direction == -1 and price_change > min_change:
            direction = 1
            zigzag.append({'x': last_pivot, 'z': start_idx + pivot_indices[-1]})
            last_pivot = prices[i]
            pivot_indices.append(i)
    zigzag.append({'x': prices[-1], 'z': start_idx + len(prices) - 1})
    listofCandidateWave = []
    for i in range(len(zigzag) - 3):
        prices = [zigzag[i]['x'], zigzag[i+1]['x'], zigzag[i+2]['x'], zigzag[i+3]['x']]
        indices = [zigzag[i]['z'], zigzag[i+1]['z'], zigzag[i+2]['z'], zigzag[i+3]['z']]
        if (prices[1] > prices[0] and
            prices[2] < prices[1] and prices[2] > prices[0] and
            prices[3] > prices[2] and
            wave2_fibonacci_check(prices[2], prices[0], prices[1]) and
            wave3_fibonacci_check(prices[3], prices[1], prices[2]) and
            wave4_fibonacci_check(prices[3], prices[2], prices[3])):
            wave = {'x': prices, 'z': indices, 'searchIndex': indices[-1]}
            if wave_length_validation(wave['z']):
                listofCandidateWave.append(wave)
    # print(f"ZigZag upward candidate wave1234: {len(listofCandidateWave)}")
    return listofCandidateWave


def ZigZag_ElliottWave_label_downward(dataframe: DataFrame, start_idx: int = 0, lookback: int = 100, min_change: float = 0.0) -> list:
    prices = dataframe['smoothed_close'].iloc[start_idx:start_idx + lookback].values
    if len(prices) < lookback:
        # print(f"ZigZag downward: Insufficient data, len(prices)={len(prices)}, lookback={lookback}")
        return []
    indices = np.arange(start_idx, start_idx + len(prices))
    zigzag = []
    direction = 0
    last_pivot = prices[0]
    pivot_indices = [0]
    for i in range(1, len(prices)):
        price_change = (prices[i] - last_pivot) / last_pivot
        if direction == 0:
            if price_change > min_change:
                direction = 1
                zigzag.append({'x': last_pivot, 'z': start_idx + pivot_indices[-1]})
                last_pivot = prices[i]
                pivot_indices.append(i)
            elif price_change < -min_change:
                direction = -1
                zigzag.append({'x': last_pivot, 'z': start_idx + pivot_indices[-1]})
                last_pivot = prices[i]
                pivot_indices.append(i)
        elif direction == 1 and price_change < -min_change:
            direction = -1
            zigzag.append({'x': last_pivot, 'z': start_idx + pivot_indices[-1]})
            last_pivot = prices[i]
            pivot_indices.append(i)
        elif direction == -1 and price_change > min_change:
            direction = 1
            zigzag.append({'x': last_pivot, 'z': start_idx + pivot_indices[-1]})
            last_pivot = prices[i]
            pivot_indices.append(i)
    zigzag.append({'x': prices[-1], 'z': start_idx + len(prices) - 1})
    listofCandidateWave = []
    for i in range(len(zigzag) - 3):
        prices = [zigzag[i]['x'], zigzag[i+1]['x'], zigzag[i+2]['x'], zigzag[i+3]['x']]
        indices = [zigzag[i]['z'], zigzag[i+1]['z'], zigzag[i+2]['z'], zigzag[i+3]['z']]
        if (prices[1] < prices[0] and
            prices[2] > prices[1] and prices[2] < prices[0] and
            prices[3] < prices[2] and
            wave2_fibonacci_check(prices[2], prices[0], prices[1]) and
            wave3_fibonacci_check(prices[3], prices[1], prices[2]) and
            wave4_fibonacci_check(prices[3], prices[2], prices[3])):
            wave = {'x': prices, 'z': indices, 'searchIndex': indices[-1]}
            if wave_length_validation(wave['z']):
                listofCandidateWave.append(wave)
    # print(f"ZigZag downward candidate wave1234: {len(listofCandidateWave)}")
    return listofCandidateWave


# Wave Confluence Score
def wave_confluence_score(traditional_waves: list, alternative_waves: list, zigzag_waves: list, practical_waves: list, practical12_waves: list) -> float:
    wave_counts = [len(waves) for waves in [traditional_waves, alternative_waves, zigzag_waves, practical_waves, practical12_waves]]
    if not any(wave_counts):
        return 0.0
    return sum(1 for count in wave_counts if count > 0) / 5.0


class ElliotWave(IStrategy):
    INTERFACE_VERSION = 3

    minimal_roi = {
        "0": 0.05,
        "40": 0.02,
        "100": 0.01,
        "240": 0
    }

    stoploss = -0.04
    timeframe = '4h'
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 200

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    lookback_period = IntParameter(50, 200, default=100, space="buy")
    min_wave_confidence = DecimalParameter(0.1, 0.8, default=0.3, space="buy")
    use_traditional_waves = BooleanParameter(default=True, space="buy")
    use_alternative_waves = BooleanParameter(default=True, space="buy")
    use_zigzag_waves = BooleanParameter(default=True, space="buy")
    use_practical12_waves = BooleanParameter(default=True, space="buy")
    use_wave3_signals = BooleanParameter(default=True, space="buy")
    use_wave5_signals = BooleanParameter(default=True, space="buy")
    fibonacci_strict = BooleanParameter(default=False, space="buy")
    validate_wave2_fib = BooleanParameter(default=True, space="buy")
    validate_wave3_fib = BooleanParameter(default=True, space="buy")
    validate_wave4_fib = BooleanParameter(default=True, space="buy")
    validate_wave5_fib = BooleanParameter(default=False, space="buy")
    exit_on_wave5_complete = BooleanParameter(default=True, space="sell")
    exit_on_opposite_wave = BooleanParameter(default=True, space="sell")
    sma_period = IntParameter(3, 10, default=5, space="buy")
    divisor = IntParameter(2, 8, default=4, space="buy")
    min_chng = DecimalParameter(0.01, 0.05, default=0.01, decimals=3, space="buy")

    plot_config = {
        "main_plot": {
            "smoothed_close": {
                "color": "#1f77b4",
                "type": "line"
            },
            "ew_upward_wave5": {
                "color": "#00FF00",
                "type": "scatter",
                "pointStyle": "circle",
                "pointRadius": 6
            },
            "ew_downward_wave5": {
                "color": "#FF0000",
                "type": "scatter",
                "pointStyle": "circle",
                "pointRadius": 6
            },
            "ew_practical12_upward": {
                "color": "#32CD32",
                "type": "scatter",
                "pointStyle": "triangle",
                "pointRadius": 5
            },
            "ew_practical12_downward": {
                "color": "#FF69B4",
                "type": "scatter",
                "pointStyle": "triangle",
                "pointRadius": 5
            }
        },
        "subplots": {
            "WaveSignals": {
                "ew_traditional_upward": {
                    "color": "#00FF00",
                    "type": "scatter",
                    "pointStyle": "square",
                    "pointRadius": 4
                },
                "ew_traditional_downward": {
                    "color": "#FF0000",
                    "type": "scatter",
                    "pointStyle": "square",
                    "pointRadius": 4
                },
                "ew_alternative_upward": {
                    "color": "#32CD32",
                    "type": "scatter",
                    "pointStyle": "diamond",
                    "pointRadius": 4
                },
                "ew_alternative_downward": {
                    "color": "#FF69B4",
                    "type": "scatter",
                    "pointStyle": "diamond",
                    "pointRadius": 4
                },
                "ew_zigzag_upward": {
                    "color": "#00FF00",
                    "type": "scatter",
                    "pointStyle": "cross",
                    "pointRadius": 4
                },
                "ew_zigzag_downward": {
                    "color": "#FF0000",
                    "type": "scatter",
                    "pointStyle": "cross",
                    "pointRadius": 4
                },
                "ew_upward_wave3": {
                    "color": "#32CD32",
                    "type": "scatter",
                    "pointStyle": "circle",
                    "pointRadius": 4
                },
                "ew_downward_wave3": {
                    "color": "#FF69B4",
                    "type": "scatter",
                    "pointStyle": "circle",
                    "pointRadius": 4
                }
            },
            "WaveStrength": {
                "ew_upward_strength": {
                    "color": "#00FF00",
                    "type": "line"
                },
                "ew_downward_strength": {
                    "color": "#FF0000",
                    "type": "line"
                }
            },
            "ConfluenceAndValidation": {
                "ew_confluence_score": {
                    "color": "#32CD32",
                    "type": "line"
                },
                "ew_wave_count": {
                    "color": "#FF69B4",
                    "type": "line"
                },
                "ew_fibonacci_valid": {
                    "color": "#1f77b4",
                    "type": "scatter",
                    "pointStyle": "cross",
                    "pointRadius": 4
                }
            }
        }
    }

    def informative_pairs(self):
        return []

    def preprocess_price_data(self, price_data: list) -> list:
        try:
            # print(f"Input price_data length: {len(price_data)}")
            price_diff = diff(price_data)
            # print(f"price_diff length: {len(price_diff)}")
            non_zero_mask = otherThan(price_diff, 0.0)
            non_zero_mask = non_zero_mask + [True]
            # print(f"non_zero_mask length: {len(non_zero_mask)}")
            filtered_data = trimming(price_data, non_zero_mask)
            # print(f"filtered_data length: {len(filtered_data)}")
            return filtered_data if filtered_data else price_data
        except Exception as e:
            print(f"Price data preprocessing error: {e}")
            return price_data

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['sma_20'] = ta.SMA(dataframe['close'], timeperiod=20)
        dataframe['sma_50'] = ta.SMA(dataframe['close'], timeperiod=50)
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['smoothed_close'] = ta.SMA(dataframe['close'], timeperiod=self.sma_period.value)
        dataframe = self.calculate_elliott_waves(dataframe)
        return dataframe

    def calculate_elliott_waves(self, dataframe: DataFrame) -> DataFrame:
        dataframe['ew_traditional_upward'] = 0
        dataframe['ew_traditional_downward'] = 0
        dataframe['ew_alternative_upward'] = 0
        dataframe['ew_alternative_downward'] = 0
        dataframe['ew_zigzag_upward'] = 0
        dataframe['ew_zigzag_downward'] = 0
        dataframe['ew_practical12_upward'] = 0
        dataframe['ew_practical12_downward'] = 0
        dataframe['ew_upward_wave3'] = 0
        dataframe['ew_upward_wave5'] = 0
        dataframe['ew_downward_wave3'] = 0
        dataframe['ew_downward_wave5'] = 0
        dataframe['ew_upward_strength'] = 0.0
        dataframe['ew_downward_strength'] = 0.0
        dataframe['ew_fibonacci_valid'] = 0
        dataframe['ew_wave_count'] = 0
        dataframe['ew_confluence_score'] = 0.0

        if len(dataframe) < self.startup_candle_count:
            print(f"DataFrame too short: {len(dataframe)} < {self.startup_candle_count}")
            return dataframe

        # Process the entire DataFrame in sliding windows
        all_waves = {
            'traditional_upward': [],
            'traditional_downward': [],
            'alternative_upward': [],
            'alternative_downward': [],
            'zigzag_upward': [],
            'zigzag_downward': [],
            'practical12_upward': [],
            'practical12_downward': [],
            'upward_waves_3': [],
            'upward_waves_5': [],
            'downward_waves_3': [],
            'downward_waves_5': []
        }

        try:
            # Slide through the DataFrame with lookback_period windows
            for start_idx in range(0, len(dataframe) - self.lookback_period.value + 1, self.lookback_period.value // self.divisor.value):
                end_idx = start_idx + self.lookback_period.value
                price_data = dataframe['smoothed_close'].iloc[start_idx:end_idx].values
                if len(price_data) < self.lookback_period.value:
                    continue
                processed_prices = self.preprocess_price_data(price_data.tolist())

                all_waves['traditional_upward'].extend(Traditional_ElliottWave_label_upward(processed_prices, start_idx))
                all_waves['traditional_downward'].extend(Traditional_ElliottWave_label_downward(processed_prices, start_idx))
                all_waves['alternative_upward'].extend(Alternative_ElliottWave_label_upward(processed_prices, start_idx))
                all_waves['alternative_downward'].extend(Alternative_ElliottWave_label_downward(processed_prices, start_idx))
                all_waves['zigzag_upward'].extend(ZigZag_ElliottWave_label_upward(dataframe, start_idx, self.lookback_period.value, self.min_chng.value))
                all_waves['zigzag_downward'].extend(ZigZag_ElliottWave_label_downward(dataframe, start_idx, self.lookback_period.value, self.min_chng.value))
                all_waves['practical12_upward'].extend(Practical_ElliottWave12_label_upward(processed_prices, start_idx))
                all_waves['practical12_downward'].extend(Practical_ElliottWave12_label_downward(processed_prices, start_idx))
                all_waves['upward_waves_3'].extend(Practical_ElliottWave3_label_upward(processed_prices, start_idx))
                all_waves['upward_waves_5'].extend(Practical_ElliottWave5_label_upward(processed_prices, start_idx))
                all_waves['downward_waves_3'].extend(Practical_ElliottWave3_label_downward(processed_prices, start_idx))
                all_waves['downward_waves_5'].extend(Practical_ElliottWave5_label_downward(processed_prices, start_idx))

            # print(f"Traditional upward wave1234: {len(all_waves['traditional_upward'])}")
            # print(f"Traditional downward wave1234: {len(all_waves['traditional_downward'])}")
            # print(f"Alternative upward wave1234: {len(all_waves['alternative_upward'])}")
            # print(f"Alternative downward wave1234: {len(all_waves['alternative_downward'])}")
            # print(f"ZigZag upward wave1234: {len(all_waves['zigzag_upward'])}")
            # print(f"ZigZag downward wave1234: {len(all_waves['zigzag_downward'])}")
            # print(f"Practical upward wave12: {len(all_waves['practical12_upward'])}")
            # print(f"Practical downward wave12: {len(all_waves['practical12_downward'])}")
            # print(f"Upward Wave 3: {len(all_waves['upward_waves_3'])}")
            # print(f"Upward Wave 5 (1234): {len(all_waves['upward_waves_5'])}")
            # print(f"Downward Wave 3: {len(all_waves['downward_waves_3'])}")
            # print(f"Downward Wave 5 (1234): {len(all_waves['downward_waves_5'])}")

            # Assign waves to DataFrame
            for wave in all_waves['traditional_upward']:
                if self.validate_fibonacci_wave(wave):
                    wave_end_idx = wave['z'][-1]
                    if 0 <= wave_end_idx < len(dataframe):
                        # print(f"Assigning traditional upward at index {wave_end_idx}: {wave['x']}")
                        dataframe.loc[dataframe.index[wave_end_idx], 'ew_traditional_upward'] = 1
                        dataframe.loc[dataframe.index[wave_end_idx], 'ew_fibonacci_valid'] = 1

            for wave in all_waves['traditional_downward']:
                if self.validate_fibonacci_wave(wave):
                    wave_end_idx = wave['z'][-1]
                    if 0 <= wave_end_idx < len(dataframe):
                        # print(f"Assigning traditional downward at index {wave_end_idx}: {wave['x']}")
                        dataframe.loc[dataframe.index[wave_end_idx], 'ew_traditional_downward'] = 1
                        dataframe.loc[dataframe.index[wave_end_idx], 'ew_fibonacci_valid'] = 1

            for wave in all_waves['alternative_upward']:
                if self.validate_fibonacci_wave(wave):
                    wave_end_idx = wave['z'][-1]
                    if 0 <= wave_end_idx < len(dataframe):
                        # print(f"Assigning alternative upward at index {wave_end_idx}: {wave['x']}")
                        dataframe.loc[dataframe.index[wave_end_idx], 'ew_alternative_upward'] = 1
                        dataframe.loc[dataframe.index[wave_end_idx], 'ew_fibonacci_valid'] = 1

            for wave in all_waves['alternative_downward']:
                if self.validate_fibonacci_wave(wave):
                    wave_end_idx = wave['z'][-1]
                    if 0 <= wave_end_idx < len(dataframe):
                        # print(f"Assigning alternative downward at index {wave_end_idx}: {wave['x']}")
                        dataframe.loc[dataframe.index[wave_end_idx], 'ew_alternative_downward'] = 1
                        dataframe.loc[dataframe.index[wave_end_idx], 'ew_fibonacci_valid'] = 1

            for wave in all_waves['zigzag_upward']:
                if self.validate_fibonacci_wave(wave):
                    wave_end_idx = wave['z'][-1]
                    if 0 <= wave_end_idx < len(dataframe):
                        # print(f"Assigning zigzag upward at index {wave_end_idx}: {wave['x']}")
                        dataframe.loc[dataframe.index[wave_end_idx], 'ew_zigzag_upward'] = 1
                        dataframe.loc[dataframe.index[wave_end_idx], 'ew_fibonacci_valid'] = 1

            for wave in all_waves['zigzag_downward']:
                if self.validate_fibonacci_wave(wave):
                    wave_end_idx = wave['z'][-1]
                    if 0 <= wave_end_idx < len(dataframe):
                        # print(f"Assigning zigzag downward at index {wave_end_idx}: {wave['x']}")
                        dataframe.loc[dataframe.index[wave_end_idx], 'ew_zigzag_downward'] = 1
                        dataframe.loc[dataframe.index[wave_end_idx], 'ew_fibonacci_valid'] = 1

            for wave in all_waves['practical12_upward']:
                if self.validate_fibonacci_wave({'x': wave['x'][:3], 'z': wave['z'][:3]}):
                    wave_end_idx = wave['z'][-1]
                    if 0 <= wave_end_idx < len(dataframe):
                        # print(f"Assigning practical12 upward at index {wave_end_idx}: {wave['x']}")
                        dataframe.loc[dataframe.index[wave_end_idx], 'ew_practical12_upward'] = 1
                        dataframe.loc[dataframe.index[wave_end_idx], 'ew_fibonacci_valid'] = 1

            for wave in all_waves['practical12_downward']:
                if self.validate_fibonacci_wave({'x': wave['x'][:3], 'z': wave['z'][:3]}):
                    wave_end_idx = wave['z'][-1]
                    if 0 <= wave_end_idx < len(dataframe):
                        # print(f"Assigning practical12 downward at index {wave_end_idx}: {wave['x']}")
                        dataframe.loc[dataframe.index[wave_end_idx], 'ew_practical12_downward'] = 1
                        dataframe.loc[dataframe.index[wave_end_idx], 'ew_fibonacci_valid'] = 1

            for wave in all_waves['upward_waves_3']:
                wave_end_idx = wave['z'][-1]
                if 0 <= wave_end_idx < len(dataframe):
                    strength = self.calculate_wave_strength(wave)
                    # print(f"Assigning upward wave3 at index {wave_end_idx}: {wave['x']}, strength: {strength}")
                    dataframe.loc[dataframe.index[wave_end_idx], 'ew_upward_wave3'] = 1
                    dataframe.loc[dataframe.index[wave_end_idx], 'ew_upward_strength'] = strength

            for wave in all_waves['upward_waves_5']:
                if len(wave['x']) >= 4 and self.validate_fibonacci_wave(wave):
                    wave_end_idx = wave['z'][-1]
                    if 0 <= wave_end_idx < len(dataframe):
                        strength = self.calculate_wave_strength(wave)
                        # print(f"Assigning upward wave5 at index {wave_end_idx}: {wave['x']}, strength: {strength}")
                        dataframe.loc[dataframe.index[wave_end_idx], 'ew_upward_wave5'] = 1
                        dataframe.loc[dataframe.index[wave_end_idx], 'ew_upward_strength'] = strength

            for wave in all_waves['downward_waves_3']:
                wave_end_idx = wave['z'][-1]
                if 0 <= wave_end_idx < len(dataframe):
                    strength = self.calculate_wave_strength(wave)
                    # print(f"Assigning downward wave3 at index {wave_end_idx}: {wave['x']}, strength: {strength}")
                    dataframe.loc[dataframe.index[wave_end_idx], 'ew_downward_wave3'] = 1
                    dataframe.loc[dataframe.index[wave_end_idx], 'ew_downward_strength'] = strength

            for wave in all_waves['downward_waves_5']:
                if len(wave['x']) >= 4 and self.validate_fibonacci_wave(wave):
                    wave_end_idx = wave['z'][-1]
                    if 0 <= wave_end_idx < len(dataframe):
                        strength = self.calculate_wave_strength(wave)
                        # print(f"Assigning downward wave5 at index {wave_end_idx}: {wave['x']}, strength: {strength}")
                        dataframe.loc[dataframe.index[wave_end_idx], 'ew_downward_wave5'] = 1
                        dataframe.loc[dataframe.index[wave_end_idx], 'ew_downward_strength'] = strength

            dataframe['ew_wave_count'] = (
                dataframe['ew_traditional_upward'] +
                dataframe['ew_alternative_upward'] +
                dataframe['ew_zigzag_upward'] +
                dataframe['ew_practical12_upward'] +
                dataframe['ew_upward_wave3'] +
                dataframe['ew_upward_wave5']
            )
            # Update confluence score for each window
            for idx in range(len(dataframe)):
                start_idx = max(0, idx - self.lookback_period.value + 1)
                end_idx = idx + 1
                window_waves = {
                    'traditional_upward': [w for w in all_waves['traditional_upward'] if start_idx <= w['z'][-1] < end_idx],
                    'alternative_upward': [w for w in all_waves['alternative_upward'] if start_idx <= w['z'][-1] < end_idx],
                    'zigzag_upward': [w for w in all_waves['zigzag_upward'] if start_idx <= w['z'][-1] < end_idx],
                    'upward_waves_5': [w for w in all_waves['upward_waves_5'] if start_idx <= w['z'][-1] < end_idx],
                    'practical12_upward': [w for w in all_waves['practical12_upward'] if start_idx <= w['z'][-1] < end_idx]
                }
                dataframe.loc[dataframe.index[idx], 'ew_confluence_score'] = wave_confluence_score(
                    window_waves['traditional_upward'],
                    window_waves['alternative_upward'],
                    window_waves['zigzag_upward'],
                    window_waves['upward_waves_5'],
                    window_waves['practical12_upward']
                )

            # # Debugging: Print DataFrame slices to verify signal distribution
            # print("DataFrame head (50 rows) for debugging:")
            # print(dataframe[['ew_upward_wave5', 'ew_downward_wave5', 'ew_upward_strength', 'ew_downward_strength', 'ew_wave_count', 'ew_confluence_score']].head(50))
            # print("DataFrame tail (50 rows) for debugging:")
            # print(dataframe[['ew_upward_wave5', 'ew_downward_wave5', 'ew_upward_strength', 'ew_downward_strength', 'ew_wave_count', 'ew_confluence_score']].tail(50))
            # # Summary of signal distribution
            # print("Signal distribution summary:")
            # print(f"Total bars: {len(dataframe)}")
            # print(f"ew_upward_wave5 non-zero: {len(dataframe[dataframe['ew_upward_wave5'] > 0])}")
            # print(f"ew_downward_wave5 non-zero: {len(dataframe[dataframe['ew_downward_wave5'] > 0])}")
            # print(f"ew_upward_strength non-zero: {len(dataframe[dataframe['ew_upward_strength'] > 0])}")
            # print(f"ew_downward_strength non-zero: {len(dataframe[dataframe['ew_downward_strength'] > 0])}")
            # print(f"ew_wave_count non-zero: {len(dataframe[dataframe['ew_wave_count'] > 0])}")
            # print(f"ew_confluence_score non-zero: {len(dataframe[dataframe['ew_confluence_score'] > 0])}")

        except Exception as e:
            print(f"Elliott Wave calculation error: {e}")

        return dataframe

    def validate_fibonacci_wave(self, wave: dict) -> bool:
        if not self.fibonacci_strict.value:
            return True
        if 'x' not in wave or 'z' not in wave:
            return False
        prices = wave['x']
        if len(prices) < 2:
            return False
        try:
            wave1_start = prices[0]
            wave1_end = prices[1]
            wave2_start = prices[1]
            wave2_end = prices[2] if len(prices) > 2 else prices[1]
            wave3_start = prices[2] if len(prices) > 2 else prices[1]
            wave3_end = prices[3] if len(prices) > 3 else prices[1]
            wave4_start = prices[3] if len(prices) > 3 else prices[1]
            wave4_end = prices[4] if len(prices) > 4 else prices[1]

            validation_results = []
            if self.validate_wave2_fib.value and len(prices) >= 3:
                wave2_valid = wave2_fibonacci_check(wave2_end, wave1_start, wave1_end)
                validation_results.append(wave2_valid)
            if self.validate_wave3_fib.value and len(prices) >= 4:
                wave3_valid = wave3_fibonacci_check(wave3_end, wave2_start, wave2_end)
                validation_results.append(wave3_valid)
            if self.validate_wave4_fib.value and len(prices) >= 4:
                wave4_valid = wave4_fibonacci_check(wave4_end, wave3_start, wave3_end)
                validation_results.append(wave4_valid)
            if self.validate_wave5_fib.value and len(prices) > 4:
                wave5_valid = wave5_fibonacci_check(
                    wave4_end, wave1_start, wave1_end, wave3_start, wave3_end, wave4_end
                )
                validation_results.append(wave5_valid)
            return all(validation_results) if validation_results else True
        except Exception as e:
            print(f"Fibonacci validation error: {e}")
            return False

    def get_most_recent_wave(self, waves: list, data_length: int) -> Optional[dict]:
        if not waves:
            return None
        best_wave = None
        best_end_distance = float('inf')
        for wave in waves:
            if 'z' in wave and len(wave['z']) > 0:
                wave_end = wave['z'][-1]
                end_distance = data_length - wave_end
                if end_distance < best_end_distance and end_distance >= 0:
                    best_end_distance = end_distance
                    best_wave = wave
        return best_wave

    def calculate_wave_strength(self, wave: dict) -> float:
        if 'x' not in wave or 'z' not in wave:
            return 0.0
        prices = wave['x']
        positions = wave['z']
        if len(prices) < 3 or len(positions) < 3:
            return 0.0
        price_range = max(prices) - min(prices)
        time_span = positions[-1] - positions[0]
        if time_span > 0:
            strength = min(price_range / time_span, 1.0)
        else:
            strength = 0.0
        return strength

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Initialize enter_tag column
        dataframe['enter_tag'] = ''
        dataframe['enter_long'] = 0

        basic_conditions = [
            dataframe['volume'] > 0,
            dataframe['close'] > dataframe['sma_20'],
            dataframe['rsi'] < 70,
            dataframe['rsi'] > 30,
            dataframe['close'] > dataframe['bb_lowerband'],
            dataframe['close'].shift(1) <= dataframe['close'],
        ]
        if self.fibonacci_strict.value:
            basic_conditions.append(dataframe['ew_fibonacci_valid'] >= 1)

        wave_conditions = []
        wave_tags = []
        if self.use_traditional_waves.value:
            wave_conditions.append(dataframe['ew_traditional_upward'] == 1)
            wave_tags.append('traditional_wave')
        if self.use_alternative_waves.value:
            wave_conditions.append(dataframe['ew_alternative_upward'] == 1)
            wave_tags.append('alternative_wave')
        if self.use_zigzag_waves.value:
            wave_conditions.append(dataframe['ew_zigzag_upward'] == 1)
            wave_tags.append('zigzag_wave')
        if self.use_practical12_waves.value:
            wave_conditions.append(dataframe['ew_practical12_upward'] == 1)
            wave_tags.append('practical12_wave')
        if self.use_wave3_signals.value:
            wave_conditions.append(
                (dataframe['ew_upward_wave3'] == 1) &
                (dataframe['ew_upward_strength'] >= self.min_wave_confidence.value)
            )
            wave_tags.append('wave3_signal')
        if self.use_wave5_signals.value:
            wave_conditions.append(
                (dataframe['ew_upward_wave5'] == 1) &
                (dataframe['ew_upward_strength'] >= self.min_wave_confidence.value)
            )
            wave_tags.append('wave5_signal')
        confluence_condition = (dataframe['ew_wave_count'] >= 2) | (dataframe['ew_confluence_score'] >= 0.5)
        wave_conditions.append(confluence_condition)
        wave_tags.append('confluence')

        # Track condition triggers for debugging
        condition_counts = {tag: 0 for tag in wave_tags}

        # Apply combined conditions as in original logic
        if wave_conditions:
            elliott_condition = wave_conditions[0]
            for condition in wave_conditions[1:]:
                elliott_condition = elliott_condition | condition
            conditions = basic_conditions + [elliott_condition]
            combined_condition = reduce(lambda x, y: x & y, conditions)
            dataframe.loc[combined_condition, 'enter_long'] = 1

            # Assign tags for each condition
            for condition, tag in zip(wave_conditions, wave_tags):
                tag_conditions = basic_conditions + [condition]
                tag_condition = reduce(lambda x, y: x & y, tag_conditions)
                mask = tag_condition & (dataframe['enter_long'] == 1)
                dataframe.loc[mask, 'enter_tag'] = dataframe.loc[mask, 'enter_tag'] + tag + ';'
                condition_counts[tag] += len(dataframe[mask])

        # Log condition triggers
        # print("Entry condition triggers:")
        # for tag, count in condition_counts.items():
            # print(f"{tag}: {count} entries")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Initialize exit_tag column
        dataframe['exit_tag'] = ''
        dataframe['exit_long'] = 0

        basic_conditions = [dataframe['volume'] > 0]
        wave_exit_conditions = []
        wave_exit_tags = []
        if self.exit_on_wave5_complete.value:
            wave_exit_conditions.extend([
                (dataframe['ew_upward_wave5'] == 1) & (dataframe['ew_upward_wave5'].shift(1) == 0),
                (dataframe['ew_traditional_upward'] == 1) & (dataframe['ew_traditional_upward'].shift(1) == 0),
                (dataframe['ew_alternative_upward'] == 1) & (dataframe['ew_alternative_upward'].shift(1) == 0),
                (dataframe['ew_zigzag_upward'] == 1) & (dataframe['ew_zigzag_upward'].shift(1) == 0)
            ])
            wave_exit_tags.extend(['wave5_complete', 'traditional_complete', 'alternative_complete', 'zigzag_complete'])
        if self.exit_on_opposite_wave.value:
            wave_exit_conditions.extend([
                dataframe['ew_traditional_downward'] == 1,
                dataframe['ew_alternative_downward'] == 1,
                dataframe['ew_zigzag_downward'] == 1,
                (dataframe['ew_downward_wave3'] == 1) & (dataframe['ew_downward_strength'] >= self.min_wave_confidence.value)
            ])
            wave_exit_tags.extend(['opposite_traditional', 'opposite_alternative', 'opposite_zigzag', 'opposite_wave3'])
        tech_conditions = [
            dataframe['rsi'] > 75,
            dataframe['close'] < dataframe['sma_20'],
            dataframe['close'] > dataframe['bb_upperband'],
        ]
        tech_tags = ['rsi_overbought', 'below_sma20', 'above_bb_upper']

        # Track condition triggers for debugging
        condition_counts = {tag: 0 for tag in wave_exit_tags + tech_tags}

        # Apply combined conditions as in original logic
        exit_conditions = basic_conditions.copy()
        if wave_exit_conditions:
            wave_exit = wave_exit_conditions[0]
            for condition in wave_exit_conditions[1:]:
                wave_exit = wave_exit | condition
            exit_conditions.append(wave_exit)
        if tech_conditions:
            tech_exit = tech_conditions[0]
            for condition in tech_conditions[1:]:
                tech_exit = tech_exit | condition
            exit_conditions.append(tech_exit)
        if len(exit_conditions) > 1:
            combined_condition = reduce(lambda x, y: x & y, exit_conditions)
            dataframe.loc[combined_condition, 'exit_long'] = 1

            # Assign tags for each condition
            for condition, tag in zip(wave_exit_conditions + tech_conditions, wave_exit_tags + tech_tags):
                tag_conditions = basic_conditions + [condition]
                tag_condition = reduce(lambda x, y: x & y, tag_conditions)
                mask = tag_condition & (dataframe['exit_long'] == 1)
                dataframe.loc[mask, 'exit_tag'] = dataframe.loc[mask, 'exit_tag'] + tag + ';'
                condition_counts[tag] += len(dataframe[mask])

        # Log condition triggers
        # print("Exit condition triggers:")
        # for tag, count in condition_counts.items():
        #     print(f"{tag}: {count} exits")

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        return 1.0

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        if 'atr' in last_candle:
            atr_value = last_candle['atr']
            if atr_value > 0:
                atr_stop = (atr_value * 2) / current_rate
                return -min(atr_stop, 0.05)
        return self.stoploss

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                           side: str, **kwargs) -> bool:
        return True

    def confirm_trade_exit(self, pair: str, trade: 'Trade', order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        return True