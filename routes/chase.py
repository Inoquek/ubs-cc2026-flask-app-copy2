from collections import defaultdict
import math
import json
import logging
from routes import app
from flask import jsonify, request
logger = logging.getLogger(__name__)

# TEST_CASE = 0
@app.route("/chasetheflag", methods = ["POST"])
def chasetheflag():
    # global TEST_CASE
    response = {
  "challenge1": "UBS{6e5aa87f066512891b827e349d1e26cd}",
  "challenge2": "your_flag_2",
  "challenge3": "your_flag_3",
  "challenge4": "your_flag_4",
  "challenge5": "your_flag_5"
}
    logger.info(str(response))
    # TEST_CASE += 1

    return jsonify(response)