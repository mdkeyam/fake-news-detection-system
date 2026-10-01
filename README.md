# 🤖 AI Fake News Detection System

A machine-learning-based web application that analyzes news text and classifies it as **Likely Fake** or **Likely Genuine**.

The project uses a trained machine-learning model and TF-IDF text vectorization to generate predictions through a FastAPI backend and a web-based frontend.

## 🚀 Features

- News text classification
- Machine-learning-based prediction
- TF-IDF text vectorization
- User-friendly web interface
- FastAPI REST API
- Prediction label and confidence information
- Local execution on a web browser

## 🛠️ Technologies Used

- Python
- FastAPI
- Uvicorn
- scikit-learn
- TF-IDF Vectorizer
- HTML
- CSS
- JavaScript

## 📂 Project Structure

```text
fake news detection system/
├── app/
│   ├── main.py
│   └── services/
│       └── prediction_service.py
├── frontend/
├── ml/
│   ├── data/
│   │   ├── Fake.csv
│   │   └── True.csv
│   └── models/
│       ├── fake_news_model_v2.pkl
│       └── tfidf_vectorizer_v2.pkl
└── README.md
```

## ⚙️ How It Works

1. The user enters news text in the frontend.
2. JavaScript sends the text to the FastAPI backend.
3. The backend processes the input using the trained TF-IDF vectorizer.
4. The machine-learning model predicts the news category.
5. The backend returns the prediction result.
6. The frontend displays the result to the user.

## 📊 Dataset

The model was trained using fake and genuine news datasets.

After removing duplicate records and invalid or very short entries, the prepared dataset contained **39,100 records**.

| Category | Records |
|---|---:|
| Fake News | 17,903 |
| Genuine News | 21,197 |
| Total | 39,100 |

## 💻 How to Run the Project

### 1. Open the project folder

Open PowerShell and run:

```powershell
cd "C:\Users\keyam\OneDrive\Desktop\fake news detection system"
```

### 2. Activate the virtual environment

If your virtual environment is named `venv`:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Start the backend

Run this command in the first terminal:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 9000
```

Backend API documentation:

http://127.0.0.1:9000/docs

### 4. Start the frontend

Open a second PowerShell terminal and run:

```powershell
cd "C:\Users\keyam\OneDrive\Desktop\fake news detection system"
python -m http.server 5500 --bind 127.0.0.1 --directory frontend
```

Open the application:

http://127.0.0.1:5500/

## 🔌 API Endpoint

```text
POST /api/v1/predict
```

The API receives news text and returns the model's prediction and related information.

## ⚠️ Disclaimer

This application is an educational machine-learning project. Its predictions are based on patterns learned from the training data and do not prove whether a news story is true or false.

The system does not independently verify news sources, evidence, images, or real-time events. Always check important claims using reliable sources.

## 🎯 Project Objective

To develop a web-based application that uses machine learning and natural language processing techniques to classify news text as Likely Fake or Likely Genuine.

## 👨‍💻 Developer

**Mahammad Keyamuddin**

GitHub: [@mdkeyam](https://github.com/mdkeyam)
