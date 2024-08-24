# app/mail_config.py
from flask_mail import Mail
from flask import Flask
from dotenv import load_dotenv
import os


def configure_mail(app):
    load_dotenv()
    app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
    app.config['MAIL_PORT'] = 587
    app.config['MAIL_USE_TLS'] = True
    app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
    app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')  # Use the App Password here
    app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME')

    mail = Mail(app)
    return mail
