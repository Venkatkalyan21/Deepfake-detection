# DeepShield — Multi-Modal AI Deepfake Detection

DeepShield is a comprehensive platform for detecting deepfakes across multiple modalities (Video, Image, and Audio). It uses state-of-the-art AI models to analyze media for facial inconsistencies, pixel-level artifacts, and synthetic voice generation.

---

## 🧠 AI/ML Stack

**Python:**
*   Main programming language used for AI development.
*   Provides easy integration with ML libraries.

**PyTorch:**
*   Framework used to build, train, and test deep learning models.
*   Provides GPU support for faster computation.

**EfficientNetV2 (local details):**
*   Used for image classification.
*   Detects fake images by analyzing:
    *   Facial inconsistencies
    *   Texture and pixel-level artifacts

**MTCNN (Multi-task Cascaded Convolutional Network):**
*   Used for face detection.
*   Extracts faces from images/videos.

**Wav2Vec:**
*   Processes raw audio signals.
*   Extracts deep audio features.
*   Helps detect AI-generated or cloned voices.

---

## 💻 Full-stack Development

**React + Tailwind CSS:**
*   Builds the user interface.
*   Provides a responsive and clean design.

**Flask:**
*   Backend framework.
*   Responsible for API requests.
*   Handles model execution.
*   Manages communication between the frontend and AI.

**MongoDB:**
*   NoSQL database.
*   Stores: User uploads, Detection Results, and logs.

---

## 🚀 How to Run Locally

### 1. Backend (Flask API)
```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate
pip install -r requirements.txt

# Create a .env file with your MongoDB URI
echo "MONGO_URI=mongodb://localhost:27017/deepshield" > .env

flask run
```
*Runs on http://localhost:5000*

### 2. Frontend (React UI)
Open a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
*Runs on http://localhost:5173*
