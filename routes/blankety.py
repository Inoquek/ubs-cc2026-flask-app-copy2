import os
import json
import logging
from routes import app
from flask import jsonify, request
import math
logger = logging.getLogger(__name__)

N = 1000
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
        print(first * last)
        geom_avg = math.sqrt(first * last)
        # range = last - first
        exp_flag = False
        if abs(mid - geom_avg) < abs(mid - arit_avg):
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
                    series[i] = math.sqrt(up * down)
                else:
                    series[i] = (up + down) / 2
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


    return Sol(data['series']).solve()