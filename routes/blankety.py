import os
import json
import logging
from routes import app
from flask import jsonify, request
import math
logger = logging.getLogger(__name__)

N = 1000
def is_monotone(data):
    # Filter out None values and keep only the non-missing entries
    non_missing = [x for x in data if x is not None]
    
    if len(non_missing) < 2:
        # If there are fewer than 2 non-missing values, we can't determine monotonicity
        return False
    
    increasing_count = 0
    decreasing_count = 0
    
    # Iterate through the non-missing entries to count increases and decreases
    for i in range(1, len(non_missing)):
        if non_missing[i] > non_missing[i - 1]:
            increasing_count += 1
        elif non_missing[i] < non_missing[i - 1]:
            decreasing_count += 1
    
    total_comparisons = increasing_count + decreasing_count
    
    # Check if at least 90% are increasing or decreasing
    if total_comparisons == 0:
        return False  # No comparisons made (all values were equal)
    
    increasing_ratio = increasing_count / total_comparisons
    decreasing_ratio = decreasing_count / total_comparisons
    
    return increasing_ratio >= 0.9 or decreasing_ratio >= 0.9

def int_sqrt(x):
    if abs(x) < 1e9:
        return math.sqrt(x)
    if x < 0:
        raise ValueError("Input must be a non-negative integer.")
    if x == 0 or x == 1:
        return x
    
    left, right = 0, x
    
    while left <= right:
        mid = (left + right) // 2
        mid_squared = mid * mid
        
        if mid_squared == x:
            return mid
        elif mid_squared < x:
            left = mid + 1
            closest_sqrt = mid
        else:
            right = mid - 1

    # After the loop, `closest_sqrt` is the integer closest to the square root
    return closest_sqrt + 1 if (closest_sqrt + 1) * (closest_sqrt + 1) - x < x - closest_sqrt * closest_sqrt else closest_sqrt

class Sol:
    def __init__(self, data):
        self.data = data

    def closest_up(self, series, idx):
        for j in range(idx + 1, N):
            if series[j] is not None:
                return series[j]
        return None
    def closest_down(self, series, idx):
        for j in range(idx - 1, -1, -1):
            if series[j] is not None:
                return series[j]
        return None
    def closest_to_idx(self, series, idx):
        if series[idx] is not None:
            return series[idx]
        for dist in range(1, N):
            j = idx + dist
            if j < N and series[j] is not None:
                return series[j]
            j = idx - dist
            if j >= 0 and series[j] is not None:
                return series[j]
        assert False
    

    def solve_one(self, series):
        assert len(series) == N
        first = self.closest_to_idx(series, 0)
        last = self.closest_to_idx(series, N - 1)
        mid = self.closest_to_idx(series, N // 2)
        arit_avg = (first + last) / 2
        geom_avg = int_sqrt(last * first)
        # range = last - first
        exp_flag = False
        if is_monotone(series) and abs(mid - geom_avg) < abs(mid - arit_avg):
            # assume exponential
            exp_flag = True
        for i, x in enumerate(series):
            if x is None:
                down = self.closest_down(series, i)
                up = self.closest_up(series, i)
                if down is None:
                    down = up
                if up is None:
                    up = down
                if exp_flag:
                    series[i] = int_sqrt(up * down)
                else:
                    series[i] = (up + down) / 2
        
        if len(series) != N:
            logger.error(f"len is {len(series)}")
        if None in series:
            logger.error(f"still Nones: {series.count(None)}")
        return series

        
    def solve(self):
        return [self.solve_one(series) for series in self.data]

TEST_CASE = 0
@app.route("/blankety", methods = ["POST"])
def blankety():
    global TEST_CASE
    data = request.get_json(silent=True) or {}
    TEST_CASE += 1
    logger.info(f"PROCESSING #{TEST_CASE}")
    # logger.info(data['series'])
    answer = Sol(data['series']).solve()
    logger.info(f"answer shape: {len(answer)}x{len(answer[0])}")
    return answer