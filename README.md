# WildTrek Backend

WildTrek is a wildlife exploration application that uses advanced Image and Audio Recognition to help users identify animals and birds, while also providing a platform to alert authorities about wildlife in need of assistance.

## 🚀 Features

- **Species Recognition API**: Deep learning-powered endpoints for identifying animals and birds through images and audio
- **One-Click Alert System**: Emergency notification system to report animals in distress
- **Vet Locator Service**: API endpoints to locate nearby veterinary clinics based on geolocation
- **Authentication System**: Secure user management and authentication
- **MongoDB Integration**: Efficient data storage and retrieval for user data and wildlife information

## 🛠️ Tech Stack

- **Framework**: Flask (Python)
- **Database**: MongoDB
- **ML Model**: PyTorch
- **API**: REST Architecture
- **Authentication**: JWT (JSON Web Tokens)

## 📋 Prerequisites

- Python 3.8+
- MongoDB 4.4+
- PyTorch 2.0+
- Required Python packages (see `requirements.txt`)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/harshrox/wildtrek-backend.git
cd wildtrek-backend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## ⚙️ Configuration

Create a `.env` file with the following variables:
```
MONGO_URI=your_mongodb_connection_string
JWT_SECRET_KEY=your_jwt_secret
MODEL_PATH=path_to_saved_model
```

## 🚀 Running the Server

1. Start the Flask server:
```bash
python run.py
```

2. The API will be available at `http://localhost:5000`

## 📚 API Documentation

### Test Route
- `GET /` - Basic test route that returns a "Hello, World!" message

### Authentication Endpoints
- `POST /signup` - Register new user
  - Required fields in JSON body: username, email, password
  - Returns: User details and success message

- `POST /login` - User login
  - Required fields in JSON body: email, password
  - Returns: Username, email, and access token

### Alert System Endpoints
- `POST /sendalert` - Create new alert
  - Requires form data with 'image' file and user details
  - Returns: Alert confirmation message

- `GET /api/alerts` - Get all alerts for authenticated user
  - Requires JWT authentication
  - Returns: List of user's alerts

### Species Recognition Endpoints
- `POST /classify-bird-audio` - Identify bird species from audio
  - Requires audio file in form data
  - Returns: Bird classification response

- `POST /classify-animal-audio` - Identify animal species from audio
  - Requires audio file in form data
  - Returns: Animal classification response

- `POST /classify-bird-image` - Identify bird species from image
  - Requires image file in form data
  - Returns: Bird classification response

- `POST /classify-animal-image` - Identify animal species from image
  - Requires image file in form data
  - Returns: Animal classification response

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 📦 Project Structure
```
wildtrek-backend/
├── app/
│   ├── models/          # Database models
│   ├── routes/          # API routes
│   ├── services/        # Business logic
│   └── utils/           # Utility functions
├── ml_models/           # ML model files
├── tests/               # Test files
├── config.py           # Configuration
└── run.py             # Application entry point
```

## 🤝 Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Harsh Anand (@harshrox)
- Ajay Tiwari (@AjaiTiwarii)
- Aryan Dev (@Aryandev12)

## 🙏 Acknowledgments

- Thanks to contributors and supporters
- Special thanks to the open-source community

---
Built with ❤️ for wildlife conservation | August 2024
