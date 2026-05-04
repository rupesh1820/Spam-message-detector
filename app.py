import os
import re
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder='.')

MODEL_PATH = os.path.join('model', 'model.pkl')
VECTORIZER_PATH = os.path.join('model', 'vectorizer.pkl')

# ---------------- LOAD MODEL ----------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Training model...")
        from train_model import train_and_save_model
        train_and_save_model()

    model = pickle.load(open(MODEL_PATH, 'rb'))
    vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))

    return model, vectorizer

model, vectorizer = load_model()

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# ---------------- HOME ----------------
@app.route('/')
def home():
    accuracy = None
    accuracy_file = os.path.join('model', 'accuracy.txt')
    if os.path.exists(accuracy_file):
        with open(accuracy_file, 'r') as f:
            accuracy = f.read().strip()
    return render_template('index.html', accuracy=accuracy)

# ---------------- PREDICT ----------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        message = request.form.get('message', '').strip()

        if not message:
            return jsonify({'result': 'error', 'confidence': 0})

        cleaned = clean_text(message)
        vectorized = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized)[0]

        # SAFE probability
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vectorized)[0]
            spam_conf = round(float(probs[1]) * 100, 1)
            ham_conf  = round(float(probs[0]) * 100, 1)
        else:
            spam_conf = ham_conf = 0

        return jsonify({
            'result': 'spam' if prediction == 1 else 'ham',
            'confidence': spam_conf if prediction == 1 else ham_conf,
            'spam_probability': spam_conf,
            'ham_probability': ham_conf
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({'result': 'error', 'confidence': 0})

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)