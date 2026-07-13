from flask import jsonify


def success_response(message, data=None, code=200):
    payload = {
        "success": True,
        "message": message,
        "data": data if data is not None else {},
    }
    return jsonify(payload), code


def error_response(message, code=400, data=None):
    payload = {
        "success": False,
        "message": message,
        "data": data if data is not None else {},
    }
    return jsonify(payload), code
