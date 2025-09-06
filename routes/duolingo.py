from collections import defaultdict
import math
import json
import logging
from routes import app
from flask import jsonify, request
logger = logging.getLogger(__name__)

class Sol:
    def __init__(self):
        pass

# TEST_CASE = 0
@app.route("/duolingo-sort", methods = ["POST"])
def duolingo():
    # global TEST_CASE
    data = request.get_json(silent=True) or {}
    logger.info(f"---PROCESSING QUERY, PART {data['part']}---")
    logger.info(str(data))
    # TEST_CASE += 1

    return jsonify([])