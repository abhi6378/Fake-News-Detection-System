import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Predefined topics & keywords
TOPIC_KEYWORDS = {
    "Politics": ["government", "president", "election", "policy", "vote", "minister"],
    "Entertainment": ["movie", "song", "celebrity", "tv", "music", "film"],
    "Sports": ["football", "cricket", "soccer", "tennis", "match", "team"],
    "Health": ["health", "covid", "vaccine", "doctor", "hospital", "disease"],
    "Technology": ["tech", "AI", "computer", "software", "internet", "gadget"]
}

def clean_text_short(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)

def detect_topic(text):
    text_lower = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return topic
    return "Other"
