# data_train.py
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# This is often needed in the environment where the script runs
# nltk.download('stopwords') 

STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)


def load_and_combine_datasets(fake_path='Fake.csv', real_path='True.csv'):
    # Load FAKE news data and assign label
    try:
        df_fake = pd.read_csv(fake_path)
        # Assign 'FAKE' label to all rows
        df_fake['label'] = 'FAKE'
    except FileNotFoundError:
        print(f"Error: Fake news file not found at {fake_path}")
        return pd.DataFrame({'clean': [], 'label': []})
    
    # Load REAL news data and assign label
    try:
        df_real = pd.read_csv(real_path)
        # Assign 'REAL' label to all rows
        df_real['label'] = 'REAL'
    except FileNotFoundError:
        print(f"Error: Real news file not found at {real_path}")
        print("Model training requires both 'FAKE' and 'REAL' data.")
        return pd.DataFrame({'clean': [], 'label': []})

    # Combine both datasets
    df = pd.concat([df_fake, df_real], ignore_index=True)

    # Use the shared columns 'title' and 'text' to create the content
    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df['clean'] = df['content'].apply(clean_text)
    
    # Select final columns and drop any remaining NaNs
    df = df[['clean', 'label']].dropna()
    return df


def train(save_model_path='model.pkl', save_vec_path='tfidf_vectorizer.pkl'):
    # Call the new function
    df = load_and_combine_datasets() 
    
    if df.empty:
        print("Cannot train: Dataframe is empty after loading/combining.")
        return

    X = df['clean']
    # Map both classes: 'FAKE': 1 and 'REAL': 0
    y = df['label'].map({'FAKE':1, 'REAL':0})  # 1 -> fake

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vec = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X_train_tfidf = vec.fit_transform(X_train)
    X_test_tfidf = vec.transform(X_test)

    # Train two models and pick the better one (simple approach)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_tfidf, y_train)
    y_pred_lr = lr.predict(X_test_tfidf)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    y_pred_nb = nb.predict(X_test_tfidf)
    acc_nb = accuracy_score(y_test, y_pred_nb)

    print('Logistic Regression accuracy:', acc_lr)
    print('Naive Bayes accuracy:', acc_nb)

    # Choose best
    best_model = lr if acc_lr >= acc_nb else nb
    joblib.dump(best_model, save_model_path)
    joblib.dump(vec, save_vec_path)
    print('Saved model to', save_model_path)
    print('Saved vectorizer to', save_vec_path)

    # Detailed report
    best_pred = best_model.predict(X_test_tfidf)
    print('\nClassification Report:\n', classification_report(y_test, best_pred, target_names=['REAL','FAKE']))
    print('\nConfusion Matrix:\n', confusion_matrix(y_test, best_pred))


if __name__ == '__main__':
    train()