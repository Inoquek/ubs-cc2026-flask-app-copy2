import os
import json
import logging
from routes import app
from flask import jsonify, request
import math
logger = logging.getLogger(__name__)

MIN = int(6e4)
class Sol:
    def __init__(self, news):
        self.news = news
    def solve(self):
        known_times = {} # time : (open + close)/2
        assert (list([d['id'] for d in self.news]) == list(range(1000)))
        exit_times = []
        close_prices = []
        for d in self.news:
            exit_times.append(d['observation_candles'][0]['timestamp'] + 30 * MIN)
            close_prices.append(d['observation_candles'][0]['close'])
            entries = d['previous_candles'] + d['observation_candles']
            assert (len(entries) == 6)
            for entry in entries:
                timestamp = entry['timestamp']
                price = (entry['open'] + entry['close']) / 2
                if timestamp not in known_times:
                    known_times[timestamp] = price
        
        THRES = 0 * MIN # at which point we stop considering a known price as relevant
        best_trades = []
        for id, exit_time in enumerate(exit_times):
            closest_dist_for = (1e9, 0) # dist and price
            closest_dist_back = (1e9, 0) # dist and price
            for time, price in known_times.items():
                if time <= exit_time and abs(time - exit_time) <= THRES:
                    closest_dist_back = min(closest_dist_back, (abs(time - exit_time), price))
                if time >= exit_time and abs(time - exit_time) <= THRES:
                    closest_dist_for = min(closest_dist_for, (abs(time - exit_time), price))
            
            if closest_dist_for[1] == 0 or closest_dist_back[1] == 0:
                closest_dist_for = min(closest_dist_for, closest_dist_back)
                closest_dist_back = closest_dist_for
            
            if closest_dist_for[1] == 0:
                continue
            
            closest_price = (closest_dist_for[1] + closest_dist_back[1]) / 2
            best_trades.append((
                abs(closest_price - close_prices[id]),
                1 if closest_price > close_prices[id] else -1,
                id
                ))
        
        best_trades.sort(reverse=True)
        logger.info(len(best_trades))
        assert len(best_trades) >= 50
        return [
            {"id": id, "decision": "LONG" if side == 1 else "SHORT"}
            for _, side, id in best_trades[:50]
        ]

# TEST_CASE = 0
@app.route("/trading-bot", methods = ["POST"])
def trading():
    # global TEST_CASE
    data = request.get_json(silent=True) or {}
    # TEST_CASE += 1
    logger.info(f"---PROCESSING TEST---")

    answer = Sol(data).solve()

    return jsonify(answer)