📰 Fake News Detection System using Machine Learning

📌 Overview

The rapid spread of misinformation on digital platforms has made fake news detection a critical challenge.
This project presents a Machine Learning–based Fake News Detection System that classifies news articles as REAL or FAKE using Natural Language Processing (NLP) techniques.

The system is implemented using Logistic Regression and Naive Bayes, and deployed as a Flask web application for interactive usage.
This project is developed as part of a B.Tech Computer Science academic project.

🎯 Objectives

To analyze news content using NLP techniques

To build an efficient ML model for fake news classification

To provide a simple and user-friendly web interface

To demonstrate practical application of Machine Learning in real-world problems

🧠 System Architecture
User Input
   ↓
Text Preprocessing (Cleaning, Stopwords, Stemming)
   ↓
TF-IDF Feature Extraction
   ↓
Machine Learning Model
   ↓
Prediction (REAL / FAKE)

⚙️ Technologies Used
| Category             | Tools                            |
| -------------------- | -------------------------------- |
| Programming Language | Python                           |
| Machine Learning     | Logistic Regression, Naive Bayes |
| NLP                  | TF-IDF, NLTK                     |
| Web Framework        | Flask                            |
| Frontend             | HTML, CSS, Bootstrap             |
| Data Handling        | Pandas, NumPy                    |
| Model Storage        | Joblib                           |
| Version Control      | GitHub                           |

✨ Key Features

Fake vs Real news classification

Efficient text preprocessing pipeline

Confidence score for predictions

Clean and responsive Flask UI

Prediction analytics visualization

Modular and well-structured codebase

📂 Project Structure
Fake-News-Detection/
│
├── app.py                     # Flask application
├── data_train.py              # Model training script
├── utils.py                   # Text preprocessing utilities
├── model.pkl                  # Trained ML model
├── tfidf_vectorizer.pkl       # TF-IDF vectorizer
├── requirements.txt           # Dependencies
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── templates/
│   ├── base.html
│   ├── index.html
│   └── result.html
│
└── README.md


🚀 How to Run the Project (Localhost)
1️⃣ Clone the Repository
git clone https://github.com/abhi6378/Fake-News-Detection.git
cd Fake-News-Detection

2️⃣ Create Virtual Environment
python -m venv venv

3️⃣ Activate Virtual Environment

Windows

venv\Scripts\activate


Mac / Linux

source venv/bin/activate

4️⃣ Install Dependencies
pip install -r requirements.txt

5️⃣ Train the Model
python data_train.py

6️⃣ Run Flask App
python app.py


Open browser and visit:

http://127.0.0.1:5000

📊 Dataset

Fake and Real news datasets in CSV format

Text fields include title and content

Labeled for supervised learning

🧪 Experimental Results

Achieved high accuracy using TF-IDF features

Logistic Regression performed better compared to Naive Bayes

The system successfully classifies unseen news articles

🚫 Deployment Status

This application is currently designed to run on localhost only for academic purposes.
GitHub is used for source code hosting and version control.
Cloud deployment can be done using platforms such as Render or PythonAnywhere as a future enhancement.

🔮 Future Enhancements

Integration of Deep Learning models (LSTM, BERT)

Support for multiple languages

Real-time news fetching using APIs

Online cloud deployment

User authentication and history tracking

👨‍🎓 Developed By

Your Name
B.Tech – Computer Science Engineering
Semester VI
Academic Year: 2024–25

📜 License

This project is developed for educational purposes only.