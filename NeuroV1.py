# Freqtrade strategy integrating components from https://github.com/neurotrader888/TechnicalAnalysisAutomation
from freqtrade.strategy import IStrategy
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Union, List
from dataclasses import dataclass
from collections import deque
import scipy.stats
import scipy.signal

@dataclass
class XABCD:
    XA_AB: Union[float, List, None]
    AB_BC: Union[float, List, None]
    BC_CD: Union[float, List, None]
    XA_AD: Union[float, List, None]
    name: str

@dataclass
class XABCDFound:
    X: int
    A: int
    B: int
    C: int
    D: int
    error: float
    name: str
    bull: bool

@dataclass
class FlagPattern:
    base_x: int
    base_y: float
    tip_x: int = -1
    tip_y: float = -1.
    conf_x: int = -1
    conf_y: float = -1.
    pennant: bool = False
    flag_width: int = -1
    flag_height: float = -1.
    pole_width: int = -1
    pole_height: float = -1.
    support_intercept: float = -1.
    support_slope: float = -1.
    resist_intercept: float = -1.
    resist_slope: float = -1.

@dataclass
class HSPattern:
    inverted: bool
    l_shoulder: int = -1
    r_shoulder: int = -1
    l_armpit: int = -1
    r_armpit: int = -1
    head: int = -1
    l_shoulder_p: float = -1
    r_shoulder_p: float = -1
    l_armpit_p: float = -1
    r_armpit_p: float = -1
    head_p: float = -1
    start_i: int = -1
    break_i: int = -1
    break_p: float = -1
    neck_start: float = -1
    neck_end: float = -1
    neck_slope: float = -1
    head_width: float = -1
    head_height: float = -1
    pattern_r2: float = -1

class NeuroV1(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '1h'
    minimal_roi = {}
    stoploss = -0.10
    trailing_stop = False
    process_only_new_candles = True
    startup_candle_count: int = 365  # For support/resistance and trendlines

    # Strategy parameters
    sigma = 0.02  # Directional change
    harmonic_err_thresh = 0.2  # Harmonic patterns
    flag_order = 10  # Flags/pennants
    hs_order = 6  # Head and shoulders
    sr_lookback = 365  # Support/resistance
    atr_mult = 3.0
    prom_thresh = 0.25
    trendline_lookback = 30

    # Harmonic pattern definitions
    GARTLEY = XABCD(0.618, [0.382, 0.886], [1.13, 1.618], 0.786, "Gartley")
    BAT = XABCD([0.382, 0.50], [0.382, 0.886], [1.618, 2.618], 0.886, "Bat")
    BUTTERFLY = XABCD(0.786, [0.382, 0.886], [1.618, 2.24], [1.27, 1.41], "Butterfly")
    CRAB = XABCD([0.382, 0.618], [0.382, 0.886], [2.618, 3.618], 1.618, "Crab")
    DEEP_CRAB = XABCD(0.886, [0.382, 0.886], [2.0, 3.618], 1.618, "Deep Crab")
    CYPHER = XABCD([0.382, 0.618], [1.13, 1.41], [1.27, 2.00], 0.786, "Cypher")
    SHARK = XABCD(None, [1.13, 1.618], [1.618, 2.24], [0.886, 1.13], "Shark")
    ALL_PATTERNS = [GARTLEY, BAT, BUTTERFLY, CRAB, DEEP_CRAB, CYPHER, SHARK]

    def directional_change(self, close: np.array, high: np.array, low: np.array, sigma: float):
        up_zig = True
        tmp_max = high[0]
        tmp_min = low[0]
        tmp_max_i = 0
        tmp_min_i = 0
        tops = []
        bottoms = []
        for i in range(len(close)):
            if up_zig:
                if high[i] > tmp_max:
                    tmp_max = high[i]
                    tmp_max_i = i
                elif close[i] < tmp_max - tmp_max * sigma:
                    top = [i, tmp_max_i, tmp_max]
                    tops.append(top)
                    up_zig = False
                    tmp_min = low[i]
                    tmp_min_i = i
            else:
                if low[i] < tmp_min:
                    tmp_min = low[i]
                    tmp_min_i = i
                elif close[i] > tmp_min + tmp_min * sigma:
                    bottom = [i, tmp_min_i, tmp_min]
                    bottoms.append(bottom)
                    up_zig = True
                    tmp_max = high[i]
                    tmp_max_i = i
        return tops, bottoms

    def get_extremes(self, ohlc: pd.DataFrame, sigma: float):
        tops, bottoms = self.directional_change(ohlc['close'].to_numpy(), ohlc['high'].to_numpy(), ohlc['low'].to_numpy(), sigma)
        tops = pd.DataFrame(tops, columns=['conf_i', 'ext_i', 'ext_p'])
        bottoms = pd.DataFrame(bottoms, columns=['conf_i', 'ext_i', 'ext_p'])
        tops['type'] = 1
        bottoms['type'] = -1
        extremes = pd.concat([tops, bottoms]).set_index('conf_i').sort_index()
        return extremes

    def find_pips(self, data: np.array, n_pips: int, dist_measure: int):
        pips_x = [0, len(data) - 1]
        pips_y = [data[0], data[-1]]
        for curr_point in range(2, n_pips):
            md = 0.0
            md_i = -1
            insert_index = -1
            for k in range(0, curr_point - 1):
                left_adj = k
                right_adj = k + 1
                time_diff = pips_x[right_adj] - pips_x[left_adj]
                price_diff = pips_y[right_adj] - pips_y[left_adj]
                slope = price_diff / time_diff
                intercept = pips_y[left_adj] - pips_x[left_adj] * slope
                for i in range(pips_x[left_adj] + 1, pips_x[right_adj]):
                    d = abs((slope * i + intercept) - data[i]) / (slope ** 2 + 1) ** 0.5
                    if d > md:
                        md = d
                        md_i = i
                        insert_index = right_adj
                pips_x.insert(insert_index, md_i)
                pips_y.insert(insert_index, data[md_i])
        return pips_x, pips_y

    def rw_top(self, data: np.array, curr_index: int, order: int) -> bool:
        if curr_index < order * 2 + 1:
            return False
        top = True
        k = curr_index - order
        v = data[k]
        for i in range(1, order + 1):
            if data[k + i] > v or data[k - i] > v:
                top = False
                break
        return top

    def rw_bottom(self, data: np.array, curr_index: int, order: int) -> bool:
        if curr_index < order * 2 + 1:
            return False
        bottom = True
        k = curr_index - order
        v = data[k]
        for i in range(1, order + 1):
            if data[k + i] < v or data[k - i] < v:
                bottom = False
                break
        return bottom

    def check_bull_pattern_pips(self, pending: FlagPattern, data: np.array, i: int, order: int):
        data_slice = data[pending.base_x: i + 1]
        max_i = data_slice.argmax() + pending.base_x
        pole_width = max_i - pending.base_x
        if i - max_i < max(5, order * 0.5):
            return False
        flag_width = i - max_i
        if flag_width > pole_width * 0.5:
            return False
        pole_height = data[max_i] - pending.base_y
        flag_height = data[max_i] - data[max_i:i+1].min()
        if flag_height > pole_height * 0.5:
            return False
        pips_x, pips_y = self.find_pips(data[max_i:i+1], 5, 3)
        if not (pips_y[2] > pips_y[1] and pips_y[2] > pips_y[3]):
            return False
        resist_rise = pips_y[2] - pips_y[0]
        resist_run = pips_x[2] - pips_x[0]
        resist_slope = resist_rise / resist_run
        resist_intercept = pips_y[0]
        support_rise = pips_y[3] - pips_y[1]
        support_run = pips_x[3] - pips_x[1]
        support_slope = support_rise / support_run
        support_intercept = pips_y[1] + (pips_x[0] - pips_x[1]) * support_slope
        if resist_slope != support_slope:
            intersection = (support_intercept - resist_intercept) / (resist_slope - support_slope)
        else:
            intersection = -flag_width * 100
        if intersection <= pips_x[4] and intersection >= 0:
            return False
        if intersection < 0 and intersection > -1.0 * flag_width:
            return False
        resist_endpoint = pips_y[0] + resist_slope * pips_x[4]
        if pips_y[4] < resist_endpoint:
            return False
        pending.pennant = support_slope > 0
        pending.tip_x = max_i
        pending.tip_y = data[max_i]
        pending.conf_x = i
        pending.conf_y = data[i]
        pending.flag_width = flag_width
        pending.flag_height = flag_height
        pending.pole_width = pole_width
        pending.pole_height = pole_height
        pending.support_slope = support_slope
        pending.support_intercept = support_intercept
        pending.resist_slope = resist_slope
        pending.resist_intercept = resist_intercept
        return True

    def check_bear_pattern_pips(self, pending: FlagPattern, data: np.array, i: int, order: int):
        data_slice = data[pending.base_x: i + 1]
        min_i = data_slice.argmin() + pending.base_x
        if i - min_i < max(5, order * 0.5):
            return False
        pole_width = min_i - pending.base_x
        flag_width = i - min_i
        if flag_width > pole_width * 0.5:
            return False
        pole_height = pending.base_y - data[min_i]
        flag_height = data[min_i:i+1].max() - data[min_i]
        if flag_height > pole_height * 0.5:
            return False
        pips_x, pips_y = self.find_pips(data[min_i:i+1], 5, 3)
        if not (pips_y[2] < pips_y[1] and pips_y[2] < pips_y[3]):
            return False
        support_rise = pips_y[2] - pips_y[0]
        support_run = pips_x[2] - pips_x[0]
        support_slope = support_rise / support_run
        support_intercept = pips_y[0]
        resist_rise = pips_y[3] - pips_y[1]
        resist_run = pips_x[3] - pips_x[1]
        resist_slope = resist_rise / resist_run
        resist_intercept = pips_y[1] + (pips_x[0] - pips_x[1]) * resist_slope
        if resist_slope != support_slope:
            intersection = (support_intercept - resist_intercept) / (resist_slope - support_slope)
        else:
            intersection = -flag_width * 100
        if intersection <= pips_x[4] and intersection >= 0:
            return False
        if intersection < 0 and intersection > -flag_width:
            return False
        support_endpoint = pips_y[0] + support_slope * pips_x[4]
        if pips_y[4] > support_endpoint:
            return False
        pending.pennant = resist_slope < 0
        pending.tip_x = min_i
        pending.tip_y = data[min_i]
        pending.conf_x = i
        pending.conf_y = data[i]
        pending.flag_width = flag_width
        pending.flag_height = flag_height
        pending.pole_width = pole_width
        pending.pole_height = pole_height
        pending.support_slope = support_slope
        pending.support_intercept = support_intercept
        pending.resist_slope = resist_slope
        pending.resist_intercept = resist_intercept
        return True

    def find_flags_pennants_pips(self, data: np.array, order: int):
        pending_bull = None
        pending_bear = None
        bull_flags = []
        bear_flags = []
        bull_pennants = []
        bear_pennants = []
        for i in range(len(data)):
            if self.rw_top(data, i, order):
                pending_bear = FlagPattern(i - order, data[i - order])
            if self.rw_bottom(data, i, order):
                pending_bull = FlagPattern(i - order, data[i - order])
            if pending_bear is not None:
                if self.check_bear_pattern_pips(pending_bear, data, i, order):
                    if pending_bear.pennant:
                        bear_pennants.append(pending_bear)
                    else:
                        bear_flags.append(pending_bear)
                    pending_bear = None
            if pending_bull is not None:
                if self.check_bull_pattern_pips(pending_bull, data, i, order):
                    if pending_bull.pennant:
                        bull_pennants.append(pending_bull)
                    else:
                        bull_flags.append(pending_bull)
                    pending_bull = None
        return bull_flags, bear_flags, bull_pennants, bear_pennants

    def check_hs_pattern(self, extrema_indices: List[int], data: np.array, i: int) -> HSPattern:
        l_shoulder, l_armpit, head, r_armpit = extrema_indices
        if i - r_armpit < 2:
            return None
        r_shoulder = r_armpit + data[r_armpit + 1: i].argmax() + 1
        if data[head] <= max(data[l_shoulder], data[r_shoulder]):
            return None
        r_midpoint = 0.5 * (data[r_shoulder] + data[r_armpit])
        l_midpoint = 0.5 * (data[l_shoulder] + data[l_armpit])
        if data[l_shoulder] < r_midpoint or data[r_shoulder] < l_midpoint:
            return None
        r_to_h_time = r_shoulder - head
        l_to_h_time = head - l_shoulder
        if r_to_h_time > 2.5 * l_to_h_time or l_to_h_time > 2.5 * r_to_h_time:
            return None
        neck_run = r_armpit - l_armpit
        neck_rise = data[r_armpit] - data[l_armpit]
        neck_slope = neck_rise / neck_run
        neck_val = data[l_armpit] + (i - l_armpit) * neck_slope
        if data[i] > neck_val:
            return None
        head_width = r_armpit - l_armpit
        pat_start = -1
        neck_start = -1
        for j in range(1, head_width):
            neck = data[l_armpit] + (l_shoulder - l_armpit - j) * neck_slope
            if l_shoulder - j < 0:
                return None
            if data[l_shoulder - j] < neck:
                pat_start = l_shoulder - j
                neck_start = neck
                break
        if pat_start == -1:
            return None
        pat = HSPattern(inverted=False)
        pat.l_shoulder, pat.r_shoulder, pat.l_armpit, pat.r_armpit, pat.head = l_shoulder, r_shoulder, l_armpit, r_armpit, head
        pat.l_shoulder_p, pat.r_shoulder_p, pat.l_armpit_p, pat.r_armpit_p, pat.head_p = data[l_shoulder], data[r_shoulder], data[l_armpit], data[r_armpit], data[head]
        pat.start_i, pat.break_i, pat.break_p = pat_start, i, data[i]
        pat.neck_start, pat.neck_end = neck_start, neck_val
        pat.neck_slope = neck_slope
        pat.head_width = head_width
        pat.head_height = data[head] - (data[l_armpit] + (head - l_armpit) * neck_slope)
        line0_slope = (pat.l_shoulder_p - pat.neck_start) / (pat.l_shoulder - pat.start_i)
        line0 = pat.neck_start + np.arange(pat.l_shoulder - pat.start_i) * line0_slope
        line1_slope = (pat.l_armpit_p - pat.l_shoulder_p) / (pat.l_armpit - pat.l_shoulder)
        line1 = pat.l_shoulder_p + np.arange(pat.l_armpit - pat.l_shoulder) * line1_slope
        line2_slope = (pat.head_p - pat.l_armpit_p) / (pat.head - l_armpit)
        line2 = pat.l_armpit_p + np.arange(pat.head - pat.l_armpit) * line2_slope
        line3_slope = (pat.r_armpit_p - pat.head_p) / (pat.r_armpit - pat.head)
        line3 = pat.head_p + np.arange(pat.r_armpit - pat.head) * line3_slope
        line4_slope = (pat.r_shoulder_p - pat.r_armpit_p) / (pat.r_shoulder - pat.r_armpit)
        line4 = pat.r_armpit_p + np.arange(pat.r_shoulder - pat.r_armpit) * line4_slope
        line5_slope = (pat.break_p - pat.r_shoulder_p) / (pat.break_i - pat.r_shoulder)
        line5 = pat.r_shoulder_p + np.arange(pat.break_i - pat.r_shoulder) * line5_slope
        raw_data = data[pat.start_i:pat.break_i]
        hs_model = np.concatenate([line0, line1, line2, line3, line4, line5])
        mean = np.mean(raw_data)
        ss_res = np.sum((raw_data - hs_model) ** 2.0)
        ss_tot = np.sum((raw_data - mean) ** 2.0)
        pat.pattern_r2 = 1.0 - ss_res / ss_tot
        return pat

    def check_ihs_pattern(self, extrema_indices: List[int], data: np.array, i: int) -> HSPattern:
        l_shoulder, l_armpit, head, r_armpit = extrema_indices
        if i - r_armpit < 2:
            return None
        r_shoulder = r_armpit + data[r_armpit + 1: i].argmin() + 1
        if data[head] >= min(data[l_shoulder], data[r_shoulder]):
            return None
        r_midpoint = 0.5 * (data[r_shoulder] + data[r_armpit])
        l_midpoint = 0.5 * (data[l_shoulder] + data[l_armpit])
        if data[l_shoulder] > r_midpoint or data[r_shoulder] > l_midpoint:
            return None
        r_to_h_time = r_shoulder - head
        l_to_h_time = head - l_shoulder
        if r_to_h_time > 2.5 * l_to_h_time or l_to_h_time > 2.5 * r_to_h_time:
            return None
        neck_run = r_armpit - l_armpit
        neck_rise = data[r_armpit] - data[l_armpit]
        neck_slope = neck_rise / neck_run
        neck_val = data[l_armpit] + (i - l_armpit) * neck_slope
        if data[i] < neck_val:
            return None
        head_width = r_armpit - l_armpit
        pat_start = -1
        neck_start = -1
        for j in range(1, head_width):
            neck = data[l_armpit] + (l_shoulder - l_armpit - j) * neck_slope
            if l_shoulder - j < 0:
                return None
            if data[l_shoulder - j] > neck:
                pat_start = l_shoulder - j
                neck_start = neck
                break
        if pat_start == -1:
            return None
        pat = HSPattern(inverted=True)
        pat.l_shoulder, pat.r_shoulder, pat.l_armpit, pat.r_armpit, pat.head = l_shoulder, r_shoulder, l_armpit, r_armpit, head
        pat.l_shoulder_p, pat.r_shoulder_p, pat.l_armpit_p, pat.r_armpit_p, pat.head_p = data[l_shoulder], data[r_shoulder], data[l_armpit], data[r_armpit], data[head]
        pat.start_i, pat.break_i, pat.break_p = pat_start, i, data[i]
        pat.neck_start, pat.neck_end = neck_start, neck_val
        pat.neck_slope = neck_slope
        pat.head_width = head_width
        pat.head_height = (data[l_armpit] + (head - l_armpit) * neck_slope) - data[head]
        line0_slope = (pat.l_shoulder_p - pat.neck_start) / (pat.l_shoulder - pat.start_i)
        line0 = pat.neck_start + np.arange(pat.l_shoulder - pat.start_i) * line0_slope
        line1_slope = (pat.l_armpit_p - pat.l_shoulder_p) / (pat.l_armpit - pat.l_shoulder)
        line1 = pat.l_shoulder_p + np.arange(pat.l_armpit - pat.l_shoulder) * line1_slope
        line2_slope = (pat.head_p - pat.l_armpit_p) / (pat.head - l_armpit)
        line2 = pat.l_armpit_p + np.arange(pat.head - pat.l_armpit) * line2_slope
        line3_slope = (pat.r_armpit_p - pat.head_p) / (pat.r_armpit - pat.head)
        line3 = pat.head_p + np.arange(pat.r_armpit - pat.head) * line3_slope
        line4_slope = (pat.r_shoulder_p - pat.r_armpit_p) / (pat.r_shoulder - pat.r_armpit)
        line4 = pat.r_armpit_p + np.arange(pat.r_shoulder - pat.r_armpit) * line4_slope
        line5_slope = (pat.break_p - pat.r_shoulder_p) / (pat.break_i - pat.r_shoulder)
        line5 = pat.r_shoulder_p + np.arange(pat.break_i - pat.r_shoulder) * line5_slope
        raw_data = data[pat.start_i:pat.break_i]
        hs_model = np.concatenate([line0, line1, line2, line3, line4, line5])
        mean = np.mean(raw_data)
        ss_res = np.sum((raw_data - hs_model) ** 2.0)
        ss_tot = np.sum((raw_data - mean) ** 2.0)
        pat.pattern_r2 = 1.0 - ss_res / ss_tot
        return pat

    def find_hs_patterns(self, data: np.array, order: int):
        last_is_top = False
        recent_extrema = deque(maxlen=5)
        recent_types = deque(maxlen=5)
        hs_lock = False
        ihs_lock = False
        ihs_patterns = []
        hs_patterns = []
        for i in range(len(data)):
            if self.rw_top(data, i, order):
                recent_extrema.append(i - order)
                recent_types.append(1)
                ihs_lock = False
                last_is_top = True
            if self.rw_bottom(data, i, order):
                recent_extrema.append(i - order)
                recent_types.append(-1)
                hs_lock = False
                last_is_top = False
            if len(recent_extrema) < 5:
                continue
            hs_alternating = True
            ihs_alternating = True
            if last_is_top:
                for j in range(2, 5):
                    if recent_types[j] == recent_types[j - 1]:
                        ihs_alternating = False
                for j in range(1, 4):
                    if recent_types[j] == recent_types[j - 1]:
                        hs_alternating = False
                ihs_extrema = list(recent_extrema)[1:5]
                hs_extrema = list(recent_extrema)[0:4]
            else:
                for j in range(2, 5):
                    if recent_types[j] == recent_types[j - 1]:
                        hs_alternating = False
                for j in range(1, 4):
                    if recent_types[j] == recent_types[j - 1]:
                        ihs_alternating = False
                ihs_extrema = list(recent_extrema)[0:4]
                hs_extrema = list(recent_extrema)[1:5]
            ihs_pat = None if ihs_lock or not ihs_alternating else self.check_ihs_pattern(ihs_extrema, data, i)
            hs_pat = None if hs_lock or not hs_alternating else self.check_hs_pattern(hs_extrema, data, i)
            if hs_pat is not None:
                hs_lock = True
                hs_patterns.append(hs_pat)
            if ihs_pat is not None:
                ihs_lock = True
                ihs_patterns.append(ihs_pat)
        return hs_patterns, ihs_patterns

    def get_error(self, actual_ratio: float, pattern_ratio: Union[float, List, None]):
        if pattern_ratio is None:
            return 0.0
        log_actual = np.log(actual_ratio)
        if isinstance(pattern_ratio, list):
            log_pat0 = np.log(pattern_ratio[0])
            log_pat1 = np.log(pattern_ratio[1])
            if log_pat0 <= log_actual <= log_pat1:
                return 0.0
            err = min(abs(log_actual - log_pat0), abs(log_actual - log_pat1)) * 2.0
            return err
        err = abs(log_actual - np.log(pattern_ratio))
        return err

    def find_xabcd(self, ohlc: pd.DataFrame, extremes: pd.DataFrame):
        extremes['seg_height'] = (extremes['ext_p'] - extremes['ext_p'].shift(1)).abs()
        extremes['retrace_ratio'] = extremes['seg_height'] / extremes['seg_height'].shift(1)
        output = {pat.name: {'bull_signal': np.zeros(len(ohlc)), 'bull_patterns': [], 'bear_signal': np.zeros(len(ohlc)), 'bear_patterns': []} for pat in self.ALL_PATTERNS}
        first_conf = extremes.index[0]
        extreme_i = 0
        entry_taken = 0
        pattern_used = None
        for i in range(first_conf, len(ohlc)):
            if extreme_i + 1 < len(extremes) and extremes.index[extreme_i + 1] == i:
                entry_taken = 0
                extreme_i += 1
            if entry_taken != 0:
                if entry_taken == 1:
                    output[pattern_used]['bull_signal'][i] = 1
                else:
                    output[pattern_used]['bear_signal'][i] = -1
                continue
            if extreme_i + 1 >= len(extremes) or extreme_i < 3:
                continue
            ext_type = extremes.iloc[extreme_i]['type']
            last_conf_i = extremes.index[extreme_i]
            if ext_type > 0.0:
                D_price = ohlc.iloc[i]['low']
                if ohlc.iloc[last_conf_i:i]['low'].min() < D_price:
                    continue
            else:
                D_price = ohlc.iloc[i]['high']
                if ohlc.iloc[last_conf_i:i]['high'].max() > D_price:
                    continue
            dc_retrace = abs(D_price - extremes.iloc[extreme_i]['ext_p']) / extremes.iloc[extreme_i]['seg_height']
            xa_ad_retrace = abs(D_price - extremes.iloc[extreme_i - 2]['ext_p']) / extremes.iloc[extreme_i - 2]['seg_height']
            best_err = 1e30
            best_pat = None
            for pat in self.ALL_PATTERNS:
                err = self.get_error(extremes.iloc[extreme_i]['retrace_ratio'], pat.AB_BC)
                err += self.get_error(extremes.iloc[extreme_i - 1]['retrace_ratio'], pat.XA_AB)
                err += self.get_error(dc_retrace, pat.BC_CD)
                err += self.get_error(xa_ad_retrace, pat.XA_AD)
                if err < best_err:
                    best_err = err
                    best_pat = pat.name
            if best_err <= self.harmonic_err_thresh:
                pattern_data = XABCDFound(
                    int(extremes.iloc[extreme_i - 3]['ext_i']),
                    int(extremes.iloc[extreme_i - 2]['ext_i']),
                    int(extremes.iloc[extreme_i - 1]['ext_i']),
                    int(extremes.iloc[extreme_i]['ext_i']),
                    i, best_err, best_pat, True
                )
                pattern_used = best_pat
                if ext_type > 0.0:
                    entry_taken = 1
                    pattern_data.name = "Bull" + pattern_data.name
                    pattern_data.bull = True
                    output[pattern_used]['bull_signal'][i] = 1
                    output[pattern_used]['bull_patterns'].append(pattern_data)
                else:
                    entry_taken = -1
                    pattern_data.name = "Bear" + pattern_data.name
                    pattern_data.bull = False
                    output[pattern_used]['bear_signal'][i] = -1
                    output[pattern_used]['bear_patterns'].append(pattern_data)
        return output

    def check_trend_line(self, support: bool, pivot: int, slope: float, y: np.array):
        intercept = -slope * pivot + y[pivot]
        line_vals = slope * np.arange(len(y)) + intercept
        diffs = line_vals - y
        if support and diffs.max() > 1e-5:
            return -1.0
        elif not support and diffs.min() < -1e-5:
            return -1.0
        err = (diffs ** 2.0).sum()
        return err

    def optimize_slope(self, support: bool, pivot: int, init_slope: float, y: np.array):
        slope_unit = (y.max() - y.min()) / len(y)
        opt_step = 1.0
        min_step = 0.0001
        curr_step = opt_step
        best_slope = init_slope
        best_err = self.check_trend_line(support, pivot, init_slope, y)
        if best_err < 0:
            return (np.nan, np.nan)
        get_derivative = True
        derivative = None
        while curr_step > min_step:
            if get_derivative:
                slope_change = best_slope + slope_unit * min_step
                test_err = self.check_trend_line(support, pivot, slope_change, y)
                derivative = test_err - best_err if test_err >= 0 else best_err + 1e-6
                if test_err < 0:
                    slope_change = best_slope - slope_unit * min_step
                    test_err = self.check_trend_line(support, pivot, slope_change, y)
                    derivative = best_err - test_err
                get_derivative = False
            if derivative > 0.0:
                test_slope = best_slope - slope_unit * curr_step
            else:
                test_slope = best_slope + slope_unit * curr_step
            test_err = self.check_trend_line(support, pivot, test_slope, y)
            if test_err < 0 or test_err >= best_err:
                curr_step *= 0.5
            else:
                best_err = test_err
                best_slope = test_slope
                get_derivative = True
        return (best_slope, -best_slope * pivot + y[pivot])

    def fit_trendlines_high_low(self, high: np.array, low: np.array, close: np.array):
        x = np.arange(len(close))
        coefs = np.polyfit(x, close, 1)
        line_points = coefs[0] * x + coefs[1]
        upper_pivot = (high - line_points).argmax()
        lower_pivot = (low - line_points).argmin()
        support_coefs = self.optimize_slope(True, lower_pivot, coefs[0], low)
        resist_coefs = self.optimize_slope(False, upper_pivot, coefs[0], high)
        return (support_coefs, resist_coefs)

    def trendline_breakout(self, high: np.array, low: np.array, close: np.array, lookback: int):
        s_tl = np.full(len(close), np.nan)
        r_tl = np.full(len(close), np.nan)
        sig = np.zeros(len(close))
        for i in range(lookback, len(close)):
            window_high = high[i - lookback: i]
            window_low = low[i - lookback: i]
            window_close = close[i - lookback: i]
            s_coefs, r_coefs = self.fit_trendlines_high_low(window_high, window_low, window_close)
            s_val = s_coefs[1] + (lookback - 1) * s_coefs[0] if not np.isnan(s_coefs[0]) else np.nan
            r_val = r_coefs[1] + (lookback - 1) * r_coefs[0] if not np.isnan(r_coefs[0]) else np.nan
            s_tl[i] = s_val
            r_tl[i] = r_val
            if np.isnan(s_val) or np.isnan(r_val):
                sig[i] = sig[i - 1]
                continue
            if close[i] > r_val:
                sig[i] = 1
            elif close[i] < s_val:
                sig[i] = -1
            else:
                sig[i] = sig[i - 1]
        return s_tl, r_tl, sig

    def find_levels(self, price: np.array, atr: float):
        first_w = 0.1
        last_w = 1.0
        w_step = (last_w - first_w) / len(price)
        weights = first_w + np.arange(len(price)) * w_step
        weights[weights < 0] = 0.0
        kernal = scipy.stats.gaussian_kde(price, bw_method=atr * self.atr_mult, weights=weights)
        min_v = np.min(price)
        max_v = np.max(price)
        step = (max_v - min_v) / 200
        price_range = np.arange(min_v, max_v, step)
        pdf = kernal(price_range)
        pdf_max = np.max(pdf)
        prom_min = pdf_max * self.prom_thresh
        peaks, _ = scipy.signal.find_peaks(pdf, prominence=prom_min)
        levels = [np.exp(price_range[peak]) for peak in peaks]
        return levels

    def support_resistance_levels(self, data: pd.DataFrame):
        atr = ta.atr(np.log(data['high']), np.log(data['low']), np.log(data['close']), self.sr_lookback)
        all_levels = [None] * len(data)
        for i in range(self.sr_lookback, len(data)):
            i_start = i - self.sr_lookback
            vals = np.log(data.iloc[i_start + 1: i + 1]['close'].to_numpy())
            levels = self.find_levels(vals, atr.iloc[i])
            all_levels[i] = levels
        return all_levels

    def sr_penetration_signal(self, data: pd.DataFrame, levels: list):
        signal = np.zeros(len(data))
        curr_sig = 0.0
        close_arr = data['close'].to_numpy()
        for i in range(1, len(data)):
            if levels[i] is None:
                continue
            last_c = close_arr[i - 1]
            curr_c = close_arr[i]
            for level in levels[i]:
                if curr_c > level and last_c <= level:
                    curr_sig = 1.0
                elif curr_c < level and last_c >= level:
                    curr_sig = -1.0
            signal[i] = curr_sig
        return signal

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['log_close'] = np.log(dataframe['close'])
        extremes = self.get_extremes(dataframe, self.sigma)
        dataframe['top'] = 0
        dataframe['bottom'] = 0
        for idx in extremes.index:
            if extremes.loc[idx, 'type'] == 1:
                dataframe.loc[dataframe.index[idx], 'top'] = 1
            else:
                dataframe.loc[dataframe.index[idx], 'bottom'] = 1

        harmonic_output = self.find_xabcd(dataframe, extremes)
        dataframe['harmonic_bull'] = 0
        dataframe['harmonic_bear'] = 0
        for pat in self.ALL_PATTERNS:
            dataframe['harmonic_bull'] += harmonic_output[pat.name]['bull_signal']
            dataframe['harmonic_bear'] += harmonic_output[pat.name]['bear_signal']

        bull_flags, bear_flags, bull_pennants, bear_pennants = self.find_flags_pennants_pips(dataframe['log_close'].to_numpy(), self.flag_order)
        dataframe['flag_bull'] = 0
        dataframe['flag_bear'] = 0
        for flag in bull_flags + bull_pennants:
            dataframe.loc[dataframe.index[flag.conf_x], 'flag_bull'] = 1
        for flag in bear_flags + bear_pennants:
            dataframe.loc[dataframe.index[flag.conf_x], 'flag_bear'] = 1

        hs_patterns, ihs_patterns = self.find_hs_patterns(dataframe['log_close'].to_numpy(), self.hs_order)
        dataframe['hs_bull'] = 0
        dataframe['hs_bear'] = 0
        for pat in ihs_patterns:
            dataframe.loc[dataframe.index[pat.break_i], 'hs_bull'] = 1
        for pat in hs_patterns:
            dataframe.loc[dataframe.index[pat.break_i], 'hs_bear'] = 1

        levels = self.support_resistance_levels(dataframe)
        dataframe['sr_signal'] = self.sr_penetration_signal(dataframe, levels)

        _, _, trend_signal = self.trendline_breakout(
            dataframe['high'].to_numpy(),
            dataframe['low'].to_numpy(),
            dataframe['close'].to_numpy(),
            self.trendline_lookback
        )
        dataframe['trend_signal'] = trend_signal

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (dataframe['harmonic_bull'] > 0) |
                (dataframe['flag_bull'] > 0) |
                (dataframe['hs_bull'] > 0)
            ) &
            (dataframe['bottom'] == 1) &
            (dataframe['sr_signal'] == 1) &
            (dataframe['trend_signal'] == 1) &
            (dataframe['volume'] > 0),
            'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['harmonic_bull'] > 0) |
                (dataframe['flag_bull'] > 0) |
                (dataframe['hs_bull'] > 0) |
                (dataframe['sr_signal'] == 1) |
                (dataframe['trend_signal'] == 1)
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['harmonic_bear'] > 0) |
                (dataframe['flag_bear'] > 0) |
                (dataframe['hs_bear'] > 0)
            ) &
            (dataframe['top'] == 1) &
            (dataframe['sr_signal'] == -1) &
            (dataframe['trend_signal'] == -1) &
            (dataframe['volume'] > 0),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (dataframe['harmonic_bear'] > 0) |
                (dataframe['flag_bear'] > 0) |
                (dataframe['hs_bear'] > 0) |
                (dataframe['sr_signal'] == -1) |
                (dataframe['trend_signal'] == -1)
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['harmonic_bull'] > 0) |
                (dataframe['flag_bull'] > 0) |
                (dataframe['hs_bull'] > 0) |
                (dataframe['sr_signal'] == 1) |
                (dataframe['trend_signal'] == 1)
            ),
            'exit_short'] = 1

        return dataframe