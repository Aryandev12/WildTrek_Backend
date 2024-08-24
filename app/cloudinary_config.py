# app/cloudinary_config.py
import cloudinary
from dotenv import load_dotenv
import os

def initialize_cloudinary():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get Cloudinary credentials from environment variables
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
    api_key = os.getenv("CLOUDINARY_API_KEY")
    api_secret = os.getenv("CLOUDINARY_API_SECRET")
    
    # Print the values (Optional: For debugging purposes)
    print(f"Cloudinary Cloud Name: {cloud_name}")
    print(f"Cloudinary API Key: {api_key}")
    
    # Configure Cloudinary
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
    )
