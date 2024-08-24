from flask import Flask
from flask_pymongo import PyMongo
from dotenv import load_dotenv
import os
from flask_cors import CORS
from .db import init_db 
load_dotenv()
def create_app():
    app = Flask(__name__)

    CORS(app)
    app.config["SECRET_KEY"] = "db24c608640f5034b30b8e1e1eb5618ed0ffdbf5"
    app.config["MONGO_URI"] = os.getenv("MONGO_URI")
    # MongoDB database
    mongo_client=init_db(app)
    if mongo_client is None:
        print("Error: mongo_client is None")    
    # Check connection
    try:
        # Test the connection by running a simple query
        print(mongo_client.db)
        collections = mongo_client.db.list_collection_names()  # Correct method to get collection names
        print("Connected to MongoDB successfully!")
    except Exception as e:
        print("Failed to connect to MongoDB:", e)
    # Import and register blueprints
    from .routes import routes as main
    app.register_blueprint(main)
    
    return app