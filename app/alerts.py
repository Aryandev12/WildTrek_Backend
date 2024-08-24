
# app/alerts.py
from .cloudinary_config import initialize_cloudinary
import cloudinary.uploader
from .mail_config import configure_mail
from flask_mail import Message
from flask import current_app


#Initialize Cloudinary Configuration.
initialize_cloudinary()

def send_alert(data, image_file):
    animal_name = data['animal_name']
    location = data['location']
    user_name = data['user_name']
    user_email = data['user_email']
    priority = data['priority']
    manual_address = data['manual_address']

    # Attempt to upload image to Cloudinary with error handling
    try:
        upload_result = cloudinary.uploader.upload(image_file)
        photo_url = upload_result.get('secure_url')
    except Exception as e:
        print(f"Error uploading image to Cloudinary: {e}")
        # Handle the error (e.g., log it, retry, etc.)
        photo_url = "Image upload failed. URL not available."

    subject = f"{'High' if priority == 'high' else 'Low'} Priority: Injured Animal Detected - {animal_name}"

    alert_message = f"""
    Subject: {subject}
    
    An alert has been triggered in the Wildlife App:
    - Animal: {animal_name}
    - Location: {location} ({manual_address})
    - Reported by: {user_name} ({user_email})
    - Priority: {priority.capitalize()}
    
    Photo: {photo_url}
    """

    # Print the alert message to the console instead of sending an email
    print(alert_message)

    # Send an email with the alert message
    # Send the alert message via email with error handling
    
    try:
        with current_app.app_context():
            mail = configure_mail(current_app)
            msg = Message(subject=subject,
                          recipients=["aryandev512@gmail.com"],  # Add authority's email here
                          body=alert_message)
            mail.send(msg)
            print("Alert email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")
        # Handle the error (e.g., log it, retry, etc.)


