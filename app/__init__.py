from flask import Flask
from flask_pymongo import PyMongo
from dotenv import load_dotenv
import os
from flask_cors import CORS
from .db import init_db 
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from .mail_config import configure_mail
load_dotenv()
bcrypt = Bcrypt()
jwt = JWTManager()
def create_app():
    app = Flask(__name__)

    CORS(app)
    mail=configure_mail(app)
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
    bcrypt.init_app(app)
    #JWT Config
    app.config["JWT_SECRET_KEY"] = "hdiisjslkojijkmfsi"  #Random Secret key for jwt.
    jwt.init_app(app)

    from .routes import routes as main
    app.register_blueprint(main)
    
    return app