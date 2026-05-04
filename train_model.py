# ============================================================
# train_model.py — Spam Detection Model Training Script
# Yeh script dataset load karke model train karegi aur save karegi
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------------------------
# STEP 1: Dataset banana (agar spam.csv nahi hai toh)
# Real dataset ki jagah hum ek solid sample dataset banate hain
# -------------------------------------------------------

SPAM_MESSAGES = [
    "Congratulations! You've won a £1,000 Walmart gift card. Go to http://bit.ly/123 to claim now",
    "FREE entry in 2 a weekly competition to win FA Cup final tkts 21st May 2005.",
    "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!",
    "You have 1 new voicemail. Call 09090900040 now to hear it. Cost 25p/min",
    "Urgent! Your Mobile number has been awarded a £2,000 Bonus Caller Prize",
    "Free ringtone waiting to be collected. Simply text the password MIX to 85069",
    "Congratulations ur awarded 500 of cd vouchers or 125gift guaranteed & Free entry 2 100 wkly draw",
    "IMPORTANT - You could be entitled up to £3,160 in compensation from mis-sold PPI",
    "Win a £1000 cash prize or a prize worth £5000. Call 09061701461. Claim code KL341.",
    "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575",
    "Urgent! call 09066350750 from landline. Your complimentary 4* Ibiza Holiday or £10,000 cash await",
    "You are a winner U have been specially selected 2 receive £1000 cash or a 4* holiday",
    "Congratulations! Nokia 3650 video camera phone is yours. Reply or call 0871-4719-523",
    "FREE MESSAGE: We are trying to contact you. Last weekends draw shows that you have won a £800 prize GUARANTEED.",
    "CASH for your survey! Call 09064019788 to get the reward",
    "Get cash now by replying CASH to 80488",
    "You've been selected! FREE prize waiting - txt CLAIM to 83600",
    "Congratulations you have won a Lucky Draw. Call 09701213186 now",
    "URGENT! You have won £1 million. Call now 0800 742 5135",
    "100% guaranteed. Claim your prize worth £10000. Text WIN to 80123",
    "FREE entry for our competition to win an iPod. Text IPOD to 83600",
    "Your free gift is waiting! Visit our site now or lose your reward",
    "SPECIAL OFFER: Get Viagra online, no prescription needed. Click here",
    "Make $$$ from home! Work 3 hours a week from home. Click here to learn more",
    "You qualify for a $500 gift card! Claim immediately at our website",
    "Congratulations! You've been pre-approved for a loan of up to $50,000",
    "ALERT: Your bank account has been suspended. Click here to reactivate immediately",
    "Hot singles in your area want to meet you. Click here to see profiles",
    "Lose weight fast! Our new pill guarantees results in 7 days",
    "Your computer has a virus! Call our toll-free number immediately to fix it",
]

HAM_MESSAGES = [
    "Hey, are you free this weekend? Wanna catch a movie?",
    "Just wanted to let you know that I'll be home by 7pm tonight",
    "Can you pick up some milk on your way home?",
    "The meeting has been rescheduled to 3pm tomorrow",
    "Happy birthday! Hope you have a wonderful day",
    "Thanks for dinner last night, it was really great!",
    "I'm running about 10 minutes late, be there soon",
    "Did you see the game last night? Amazing ending!",
    "Can we reschedule our lunch to Thursday?",
    "Just finished the report, sending it over now",
    "Mom called, she wants you to call her back when free",
    "I'll be at the library studying until 9pm",
    "Don't forget we have a dentist appointment at 2pm on Tuesday",
    "The package was delivered. It's on the front porch",
    "Great presentation today! The client was really impressed",
    "Call me when you get this, it's important",
    "Going to the gym after work. Want to join?",
    "The kids had a great time at the birthday party",
    "Just landed. Will take a cab home, don't worry about pickup",
    "Dinner's ready, come downstairs when you're free",
    "I got the job! Starting next Monday. So excited!",
    "Can you send me the recipe for that pasta dish?",
    "Traffic is terrible today, might be a bit late",
    "How are you feeling? Hope you're getting better",
    "The Wi-Fi password is on the back of the router",
    "I'll be in the office by 9am, we can discuss it then",
    "Finished my exams! Results in 2 weeks, fingers crossed",
    "Can we talk later? Having a rough day",
    "Your Amazon package has been shipped and will arrive tomorrow",
    "Reminder: Team lunch is at 12:30 today at the usual place",
    "Let me know when you want to catch up, it's been a while",
    "The doctor said everything looks good, no worries",
    "I'm at the grocery store, do you need anything?",
    "Got stuck in a meeting but free now, call me back",
    "Your table is confirmed for Saturday at 7pm, see you then!",
]

def create_dataset():
    """
    Dataset create karta hai — spam aur ham messages ko combine karke
    Returns: pandas DataFrame with 'label' and 'message' columns
    """
    # Spam ko 1, ham (normal) ko 0 label dete hain
    spam_df = pd.DataFrame({'label': 1, 'message': SPAM_MESSAGES})
    ham_df  = pd.DataFrame({'label': 0, 'message': HAM_MESSAGES})
    
    # Dono ko combine karo aur shuffle karo
    df = pd.concat([spam_df, ham_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[✓] Dataset ready: {len(df)} messages ({spam_df.shape[0]} spam, {ham_df.shape[0]} ham)")
    return df


def load_real_dataset(filepath='spam.csv'):
    """
    Agar real spam.csv available hai toh woh load karo
    Format: 'label' column (spam/ham or 1/0) and 'message' column
    """
    try:
        df = pd.read_csv(filepath, encoding='latin-1')
        # Pehle 2 useful columns rakh, baaki drop kar
        if 'v1' in df.columns and 'v2' in df.columns:
            df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
        # Label ko numeric mein convert karo
        if df['label'].dtype == object:
            df['label'] = df['label'].map({'spam': 1, 'ham': 0})
        df = df.dropna()
        print(f"[✓] Real dataset loaded: {len(df)} messages")
        return df
    except FileNotFoundError:
        print("[!] spam.csv not found — using built-in sample dataset instead")
        return None


def clean_text(text):
    """
    Text ko clean karta hai — lowercase + basic cleaning
    Zyada complex preprocessing ki zarurat nahi for basic project
    """
    import re
    text = str(text).lower()                     # Lowercase karo
    text = re.sub(r'http\S+|www\S+', '', text)   # URLs hatao
    text = re.sub(r'\d+', '', text)              # Numbers hatao
    text = re.sub(r'[^\w\s]', '', text)          # Special chars hatao
    text = text.strip()
    return text


def train_and_save_model():
    """
    Main training function — model train karke pickle mein save karta hai
    """
    print("\n" + "="*55)
    print("       🤖 SPAM DETECTOR — MODEL TRAINING")
    print("="*55)

    # -------------------------------------------------------
    # STEP 1: Dataset load karo
    # -------------------------------------------------------
    df = load_real_dataset('spam.csv')
    if df is None:
        df = create_dataset()

    # -------------------------------------------------------
    # STEP 2: Text clean karo
    # -------------------------------------------------------
    df['clean_message'] = df['message'].apply(clean_text)
    print(f"[✓] Text cleaned successfully")

    # -------------------------------------------------------
    # STEP 3: TF-IDF vectorizer se text ko numbers mein badlo
    # TF-IDF — har word ka importance calculate karta hai
    # -------------------------------------------------------
    vectorizer = TfidfVectorizer(
        max_features=3000,   # Top 3000 words use karo
        ngram_range=(1, 2),  # Unigrams + bigrams (pairs of words)
        stop_words='english' # Common English words ignore karo
    )

    X = vectorizer.fit_transform(df['clean_message'])
    y = df['label']
    print(f"[✓] TF-IDF vectorization done — Feature matrix shape: {X.shape}")

    # -------------------------------------------------------
    # STEP 4: Train-test split karo (80% train, 20% test)
    # -------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[✓] Train set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")

    # -------------------------------------------------------
    # STEP 5: Naive Bayes model train karo
    # MultinomialNB text classification ke liye best hai
    # -------------------------------------------------------
    model = MultinomialNB(alpha=0.1)  # Alpha = smoothing parameter
    model.fit(X_train, y_train)
    print(f"[✓] Naive Bayes model trained!")

    # -------------------------------------------------------
    # STEP 6: Model evaluate karo
    # -------------------------------------------------------
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n📊 MODEL PERFORMANCE:")
    print(f"   Accuracy : {accuracy * 100:.2f}%")
    print(f"\n   Classification Report:")
    report = classification_report(y_test, y_pred, target_names=['Ham (Normal)', 'Spam'], zero_division=0)
    for line in report.split('\n'):
        print(f"   {line}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   Confusion Matrix:")
    print(f"   [[TN={cm[0][0]}  FP={cm[0][1]}]")
    print(f"    [FN={cm[1][0]}  TP={cm[1][1]}]]")

    # -------------------------------------------------------
    # STEP 7: Model aur vectorizer ko pickle mein save karo
    # -------------------------------------------------------
    os.makedirs('model', exist_ok=True)
    
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('model/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # Accuracy bhi save karo (UI mein dikhane ke liye)
    with open('model/accuracy.txt', 'w') as f:
        f.write(f"{accuracy * 100:.2f}")

    print(f"\n[✓] model.pkl saved → model/model.pkl")
    print(f"[✓] vectorizer.pkl saved → model/vectorizer.pkl")
    print(f"\n🚀 Training complete! Run: python app.py")
    print("="*55 + "\n")

    return accuracy


# -------------------------------------------------------
# Script directly run karo toh training start hogi
# -------------------------------------------------------
if __name__ == '__main__':
    train_and_save_model()
