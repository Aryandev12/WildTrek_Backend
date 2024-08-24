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
        self.id = _id  # Generate a unique ID for each use

        self.username = username
        self.email = email
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        self.created_at = datetime.utcnow() 

    
    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "password": self.password_hash,
            "created_at": self.created_at
        }