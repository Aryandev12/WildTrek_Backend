from app.db import mongo_client # Import PyMongo
from flask import Blueprint, jsonify, request
routes = Blueprint("routes", __name__)

#Test Route
@routes.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Hello, World!"})
    