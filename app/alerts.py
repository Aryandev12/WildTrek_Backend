
# app/alerts.py
from .cloudinary_config import initialize_cloudinary
import cloudinary.uploader
from .mail_config import configure_mail
from flask_mail import Message
from flask import current_app
from .models import Alert, AlertSchema
from app.db import mongo_client # Import PyMongo
from bson.objectid import ObjectId  # Import ObjectId to work with MongoDB _id fields



#FUNCTION TO FIND USER BY EMAIL AND USER NAME
def find_user_by_email_and_name(email, user_name):
    user = mongo_client.db.users.find_one({'email': email, 'username': user_name})
    if user:
        return user
    else:
        return None



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
    # Find the user by email and username
    user = find_user_by_email_and_name(user_email, user_name)
    if user:
        user_id = user['_id']  # Correctly access the _id from the user document
        alert_data = {
            "animal_name": animal_name,
            "location": location,
            "username": user_name,
            "user_email": user_email,
            "priority": priority,
            "address": manual_address,
            "user_id": str(user_id),  # Ensure _id is stored as a string if needed
            "photo_url": photo_url

        }

        # Try to insert the alert into MongoDB
        try:
            mongo_client.db.alerts.insert_one(alert_data)
            print("Alert data inserted into MongoDB successfully.")
        except Exception as e:
            print(f"Error inserting alert data into MongoDB: {e}")
        # Push the alert into the user's alerts array
        try:
            mongo_client.db.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$push": {"alerts": alert_data}}
            )
            print("Alert added to the user's alerts array.")
        except Exception as e:
            print(f"Error updating user's alerts array: {e}")
        
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

    else:
        return False
        
    

  


    

    

