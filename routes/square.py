import logging
from flask import request, jsonify
from routes import app

logger = logging.getLogger(__name__)

@app.route('/square', methods=['POST'])
def evaluate():
    data = request.get_json(silent=True) or {}
    logger.info("data sent for evaluation %s", data)

    if "input" not in data:
        return jsonify(error="Missing 'input'"), 400

    try:
        input_value = float(data["input"])
    except (TypeError, ValueError):
        return jsonify(error="'input' must be numeric"), 400

    result = input_value * input_value
    logger.info("My result: %s", result)

    # ✅ Always return a JSON object
    return jsonify({"input": input_value, "result": result})
