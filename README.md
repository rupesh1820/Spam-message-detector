# 🛡️ SpamShield — AI Spam Message Detector

A beginner-friendly ML web project to detect spam messages using **Naive Bayes + TF-IDF**, built with Python (Flask) backend and a modern HTML/CSS/JS frontend.

---

## 📁 Folder Structure

```
spam_detector/
├── app.py               ← Flask backend (routes + prediction)
├── train_model.py       ← ML training script
├── requirements.txt     ← Python dependencies
├── Procfile             ← For Railway/Render deployment
├── spam.csv             ← (Optional) Real SMS Spam dataset
├── model/
│   ├── model.pkl        ← Trained Naive Bayes model (auto-generated)
│   ├── vectorizer.pkl   ← TF-IDF vectorizer (auto-generated)
│   └── accuracy.txt     ← Model accuracy score
└── templates/
    └── index.html       ← Frontend UI
```

---

## 🚀 Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — (Optional) Train on real dataset
Download `spam.csv` from [Kaggle SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and place it in the root folder.

### Step 3 — Train the model
```bash
python train_model.py
```

### Step 4 — Start the server
```bash
python app.py
```

### Step 5 — Open browser
Go to: **http://localhost:5000**

> **Note:** If you skip Step 3, the app auto-trains when you first run `app.py`.

---

## 🤖 How It Works

```
User types message
       ↓
Text cleaned (lowercase, remove URLs/numbers/symbols)
       ↓
TF-IDF Vectorizer converts text → numbers
       ↓
Naive Bayes model predicts: Spam (1) or Ham (0)
       ↓
Confidence probability returned to UI
```

---

## 📊 Model Details

| Component        | Details                        |
|------------------|-------------------------------|
| Algorithm        | Multinomial Naive Bayes        |
| Vectorizer       | TF-IDF (3000 features, bigrams)|
| Train/Test Split | 80% / 20%                     |
| Smoothing        | Alpha = 0.1                   |

---

## 🌐 Deploy on Render (Free)

1. Push project to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt && python train_model.py`
   - **Start Command:** `python app.py`
5. Deploy!

## 🚂 Deploy on Railway

1. Push to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Railway auto-detects Python + Procfile
4. Set environment variable: `PORT=5000`
5. Done!

---

## 🧪 Test Messages

| Message | Expected |
|---------|----------|
| "Congratulations! You've won £1000. Claim now!" | 🚫 SPAM |
| "Hey, running 10 mins late, be there soon" | ✅ HAM |
| "FREE entry to win FA Cup tickets. Text WIN now!" | 🚫 SPAM |
| "Can you pick up milk on the way home?" | ✅ HAM |

---

Made with ❤️ using Python, Flask, scikit-learn
