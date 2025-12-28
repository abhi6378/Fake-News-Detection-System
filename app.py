from flask import Flask, render_template, request, send_file
import joblib, json, os, io, csv
from datetime import datetime
from utils import clean_text_short, detect_topic
from wordcloud import WordCloud
import matplotlib.pyplot as plt

app = Flask(__name__)

MODEL_PATH = 'model.pkl'
VEC_PATH = 'tfidf_vectorizer.pkl'
HISTORY_FILE = 'history.json'

model = joblib.load(MODEL_PATH)
vec = joblib.load(VEC_PATH)

# Save prediction history
def save_history(text, label, confidence, topic):
    record = {
        "text": text[:200] + ("..." if len(text) > 200 else ""),
        "label": label,
        "confidence": confidence,
        "topic": topic,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    data = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []

    data.append(record)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(data, f, indent=4)


# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form.get('title', '')
    text = request.form.get('news_text', '')
    content = title + " " + text
    clean = clean_text_short(content)
    vect = vec.transform([clean])
    pred = model.predict(vect)[0]

    # Confidence
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(vect)[0]
        confidence = round(max(prob) * 100, 2)
    else:
        confidence = None

    # Label & topic
    label = 'FAKE' if pred == 1 else 'REAL'
    topic = detect_topic(content)

    # Save history
    save_history(content, label, confidence, topic)

    return render_template('result.html', label=label, confidence=confidence, text=content, topic=topic)

# Stats & wordcloud
@app.route('/stats')
def stats():
    fake, real = 0, 0
    wordcloud_exists = False
    topic_counts = {}

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            data = json.load(f)

        fake = sum(1 for d in data if d['label'] == 'FAKE')
        real = sum(1 for d in data if d['label'] == 'REAL')

        # Topics
        for d in data:
            topic_counts[d['topic']] = topic_counts.get(d['topic'], 0) + 1

        # WordCloud
        text_corpus = " ".join([d["text"] for d in data])
        wc = WordCloud(width=800, height=400, background_color='white').generate(text_corpus)
        wc.to_file("static/wordcloud.png")
        wordcloud_exists = True

    return render_template('stats.html', fake=fake, real=real, wordcloud_exists=wordcloud_exists, topic_counts=topic_counts)

# Export CSV
@app.route('/export_csv')
def export_csv():
    if not os.path.exists(HISTORY_FILE):
        return "No history to export.", 404

    with open(HISTORY_FILE, 'r') as f:
        data = json.load(f)

    csv_file = 'prediction_history.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp','text','label','confidence','topic'])
        writer.writeheader()
        writer.writerows(data)

    return send_file(csv_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
