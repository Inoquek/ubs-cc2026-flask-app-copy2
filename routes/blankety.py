import os
import json
import logging
from routes import app
from flask import jsonify, request
import numpy as np
logger = logging.getLogger(__name__)

TEST_CASE = 0
@app.route("/blankety", methods = ["POST"])
def blankety():
    global TEST_CASE
    data = request.get_json(silent=True) or {}
    TEST_CASE += 1
    logger.info(f"PROCESSING #{TEST_CASE}")

    logger.info(data)

    return []