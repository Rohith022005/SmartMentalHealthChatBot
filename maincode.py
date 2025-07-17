import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import spacy
from trans import match_question
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Allow cross-origin requests

# Load the chatbot model and data
model = load_model('model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Load NLP tools
nlp = spacy.load("en_core_web_sm")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    results = [{"intent": classes[i], "probability": r} for i, r in enumerate(res) if r > 0.25]
    return sorted(results, key=lambda x: x['probability'], reverse=True)

def get_response(ints):
    if ints:
        tag = ints[0]['intent']
        return random.choice([i['responses'] for i in intents['intents'] if i['tag'] == tag][0])
    return "Sorry, I didn't understand that."

@app.route("/",methods = ["GET"])
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"response": "Please provide a message."})

    matched_question = match_question(user_message)
    bot_response = get_response(predict_class(matched_question))

    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)