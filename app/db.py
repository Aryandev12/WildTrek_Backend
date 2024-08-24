# app/db.py
from flask_pymongo import PyMongo
def init_db(app):
    global mongo_client
    mongo_client = PyMongo(app)
    return mongo_client