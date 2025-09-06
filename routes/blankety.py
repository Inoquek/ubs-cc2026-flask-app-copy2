# import os
# import json
# import logging
# from routes import app
# from flask import jsonify, request
# import numpy as np
# logger = logging.getLogger(__name__)

# @app.route("/princess-diaries", methods = ["POST"])
# def princess_diaries():
#     global TEST_CASE
#     data = request.get_json(silent=True) or {}
#     TEST_CASE += 1
#     logger.info(f"PROCESSING #{TEST_CASE}")

#     # extract top-level keys
#     tasks = data.get("tasks", [])
#     subway = data.get("subway", [])
#     starting_station = data.get("starting_station")

#     # Example: log what we received
#     logger.info("Received tasks:")
#     logger.info(len(tasks))
#     logger.info("Received subway connections:")
#     logger.info(len(subway))
#     logger.info("Starting station: %s", starting_station)
    
#     solution = Sol(tasks, subway, starting_station).solve()
#     logger.info("Solution: %s", solution)
    
#     out = {
#     "max_score": int(solution.get("max_score", 0)),
#     "min_fee": int(solution.get("min_fee", 0)),
#     "schedule": list(solution.get("schedule", [])),
#     }
#     return jsonify(out)