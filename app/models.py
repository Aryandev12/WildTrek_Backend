from datetime import datetime
from marshmallow import Schema, fields, post_load,validate
import uuid
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token
from . import bcrypt


SECRET_KEY = "Harsdodiojfhudnsjijgijfjvlv" 
#User Schema And Model .

class UserSchema(Schema):
    id = fields.Str(dump_only=True)
    username = fields.Str(required=True)
    email = fields.Email(required=True)
    password = fields.Str(required=True)  # Don't include password in response

    @post_load
    def create_user(self, data, **kwargs):
        user = User(**data)
        return user
class User(object):
    def __init__(self, username, email, password, _id=None, created_at=None):
        self.id = _id  or str(uuid.uuid4()) # Generate a unique ID for each use

        self.username = username
        self.email = email
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        self.created_at = datetime.utcnow() 
        self.alerts =[] #an empty list of alerts.

    
    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "password": self.password_hash,
            "created_at": self.created_at,
            "alerts": [alert.to_dict() for alert in self.alerts]
        }
class AlertSchema(Schema):
    id = fields.Str(dump_only=True)
    animal_name = fields.Str(required=True)
    location = fields.Str(required=True)
    username = fields.Str(required=True)
    user_email = fields.Email(required=True)
    priority = fields.Str(required=True, validate=validate.OneOf(["Low", "Medium", "High"]))
    address = fields.Str(required=True)
    user_id = fields.Str(required=True)  # Foreign key to reference User
    created_at = fields.DateTime(dump_only=True)

    @post_load
    def create_alert(self, data, **kwargs):
        return Alert(**data)

class Alert:
    def __init__(self, animal_name, location, username, user_email, priority, address, user_id, _id=None, created_at=None):
        self.id = _id or str(uuid.uuid4())  # Generate a unique ID for each alert
        self.animal_name = animal_name
        self.location = location
        self.username = username
        self.user_email = user_email
        self.priority = priority
        self.address = address
        self.user_id = user_id  # Reference to the user who sent the alert
        self.created_at = created_at or datetime.utcnow()

    def to_dict(self):
        return {
            "id": self.id,
            "animal_name": self.animal_name,
            "location": self.location,
            "username": self.username,
            "user_email": self.user_email,
            "priority": self.priority,
            "address": self.address,
            "user_id": self.user_id,
            "created_at": self.created_at
        }