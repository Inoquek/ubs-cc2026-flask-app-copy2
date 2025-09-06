import os
import json
import logging
from routes import app
from flask import jsonify, request
import math
logger = logging.getLogger(__name__)

# class Sol:
#     def __init__(self, news):
#         self.news = news

# TEST_CASE = 0
@app.route("/trading-bot", methods = ["POST"])
def trading():
    # global TEST_CASE
    data = request.get_json(silent=True) or {}
    # TEST_CASE += 1
    logger.info(f"---PROCESSING TEST---")

    logger.info(data)

    return jsonify([])