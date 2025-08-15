    #!/usr/bin/env python
# coding: utf-8

# Import Library
import streamlit as st
st.set_page_config(
    page_title="AI Email Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
<style>
    .main-header {
    font-size: 4rem !important;
    font-weight: bold;
    text-align: center;
    margin: 2rem 0;
    padding: 1rem 0;
    background: linear-gradient(135deg, #e53e3e 0%, #dd6b20 50%, #d69e2e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    }
    
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .spam-prediction {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
    }
    
    .ham-prediction {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    # Add this to your existing CSS in st.markdown at the top
    .explanation-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    
    .spam-word {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        margin: 0.1rem;
        display: inline-block;
    }
    
    .safe-word {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        margin: 0.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

from keras.callbacks import EarlyStopping
from keras.layers import Embedding, LSTM, Dense, Dropout
import random
import os
from pathlib import Path
import gc
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
pd.set_option('display.max_rows', None) #show all rows
pd.set_option('display.max_columns', None) #show all columns
import numpy as np
import math
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import re
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Bidirectional, GlobalMaxPooling1D, Conv1D
from keras.models import Model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from plotly.subplots import make_subplots
import json
from collections import Counter
import lime
from lime.lime_text import LimeTextExplainer
import shap
import warnings
warnings.filterwarnings('ignore')
import io
import random
from sklearn.metrics import roc_curve, auc

@st.cache_resource
def load_data():
    path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")

    print("Path to dataset files:", path)
    
    file_paths = []
    for roots, dirs, filenames in os.walk(path):
        for filename in filenames:
            file_paths.append(os.path.join(roots, filename))
            
    files_to_remove = [
    'phishing_email.csv',  # Combined file
    ]

    filtered_path = []
    for file_path in file_paths:
        filename = os.path.basename(file_path) # Get just the filename
        if filename not in files_to_remove:
            filtered_path.append(file_path)
        
        return filtered_path


# Load Data
file_paths = load_data()

@st.cache_resource
def load_and_standardize_data(file_paths):
    phishing_dfs = []

    for file_path in file_paths:
        try:
            df_temp = pd.read_csv(file_path, low_memory=False)
            print(f"Original shape: {df_temp.shape}")
            print(f"Original columns: {list(df_temp.columns)}")

            # Handle different file structures
            if 'text_combined' in df_temp.columns:
                # For the 2-column combined file
                df_temp = df_temp[['text_combined', 'label']].copy()
                df_temp.columns = ['Message', 'Category']
            elif len(df_temp.columns) >= 2:
                # For files with multiple columns, combine text fields
                text_columns = []
                if 'subject' in df_temp.columns:
                    text_columns.append('subject')
                if 'body' in df_temp.columns:
                    text_columns.append('body')

                if text_columns:
                    # Combine subject and body
                    df_temp['Message'] = df_temp[text_columns].fillna('').apply(
                        lambda x: ' '.join(x.astype(str)), axis=1
                    )
                else:
                    # Fallback to first text column
                    df_temp['Message'] = df_temp.iloc[:, 0].astype(str)

                # Get the label column
                if 'label' in df_temp.columns:
                    df_temp['Category'] = df_temp['label']
                else:
                    df_temp['Category'] = df_temp.iloc[:, -1]  # Assume last column is label

                # Keep only the standardized columns
                df_temp = df_temp[['Message', 'Category']].copy()

            print(f"Standardized shape: {df_temp.shape}")
            phishing_dfs.append(df_temp)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return phishing_dfs

phishing_dfs = load_and_standardize_data(file_paths)
df = pd.concat(phishing_dfs, ignore_index=True)

@st.cache_resource
def clean_data(df):
    # Check for empty columns
    print(df.isnull().sum().sum())

    # Check for duplicates & drop them
    print(df.duplicated().sum())
    df = df.drop_duplicates()
    print(df.duplicated().sum())
    
    return df

# Clean Data
df = clean_data(df)

def advanced_text_preprocessing(text):
    """Advanced text preprocessing with feature extraction"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Extract features before cleaning
    features = {
        'url_count': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'currency_symbols': len(re.findall(r'[$£€¥₹]', text)),
        'numbers': len(re.findall(r'\d+', text)),
        'caps_ratio': sum(1 for c in text if c.isupper()) / (len(text) + 1),
        'special_chars': len(re.findall(r'[^\w\s]', text))
    }
    
    # Clean text
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' URL ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' NUMBER ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text, features


@st.cache_resource
def train_model(df):
    # Prepare data for training
    df = df.sample(n=30000, random_state=42).reset_index(drop=True)
    processed_data = []
    feature_data = []
    
    for text in df['Message']:
        clean_text, features = advanced_text_preprocessing(text)
        processed_data.append(clean_text)
        feature_data.append(features)
    
    df['Processed_Message'] = processed_data
    feature_df = pd.DataFrame(feature_data)
    
    X_text = df['Processed_Message']
    X_features = feature_df.values
    y = df['Category']
    
    # Train-Validation-Test split
    X_train, X_temp, X_feat_train, X_feat_temp, y_train, y_temp = train_test_split(
        X_text, X_features, y, train_size=0.6, random_state=42, stratify=y
    )
    X_val, X_test, X_feat_val, X_feat_test, y_val, y_test = train_test_split(
        X_temp, X_feat_temp, y_temp, train_size=0.5, random_state=42, stratify=y_temp
    )

    # Tokenization
    # Translate text to numbers for model to understand
    vocab_size = 20000  # Maximum number of words to keep
    max_length = 200    # Maximum sequence length
    oov_token = '<OOV>' # Out-of-vocabulary token (Handles unknown words that is not seen before)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token) # Create a Dictionary to map words to numbers
    tokenizer.fit_on_texts(X_train)  # Map words in training data to numbers 

    # Convert texts to sequences
    X_train_sequences = tokenizer.texts_to_sequences(X_train) # Convert the training data words to numbers
    X_test_sequences = tokenizer.texts_to_sequences(X_test) 
    X_val_sequences = tokenizer.texts_to_sequences(X_val)
    
    X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding='post')
    X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding='post')
    X_val_padded = pad_sequences(X_val_sequences, maxlen=max_length, padding='post')

    # Convert y_val to numpy array if not already
    y_val = np.array(y_val, dtype=np.int32)

    # Clean text Input branch
    text_input = tf.keras.Input(shape=(max_length,), name='text_input')
    embedding = Embedding(vocab_size, 128, input_length=max_length, mask_zero=True)(text_input)
    lstm = LSTM(64, dropout=0.6, recurrent_dropout=0.6)(embedding)
    
    # Feature Input branch
    feature_input = tf.keras.Input(shape=(7,), name='feature_input')  # 7 features
    feature_dense = Dense(32, activation='relu')(feature_input)
    feature_dropout = Dropout(0.3)(feature_dense)
    
    # Combine branches
    combined = tf.keras.layers.Concatenate()([lstm, feature_dropout])
    dense1 = Dense(64, activation='relu')(combined)
    dropout1 = Dropout(0.6)(dense1)
    output = Dense(1, activation='sigmoid')(dropout1)

    # Call Model
    model = Model(inputs=[text_input, feature_input], outputs=output)
    
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

    # Early Stopping - Keep track of val_loss and stops training if val_loss gets worse
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit([X_train_padded, X_feat_train], y_train, epochs=15, batch_size=64,validation_data=([X_val_padded, X_feat_val], y_val),callbacks=[early_stopping])

    y_pred_prob = model.predict([X_test_padded,X_feat_test])
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Overall Information of Model Prediction
    test_loss, test_accuracy = model.evaluate([X_test_padded, X_feat_test], y_test, verbose=0)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_prob)
    
    feature_names = list(feature_df.columns)
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'vocab_size': vocab_size,
        'max_length': max_length,
        'history': history,
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'X_feat_train': X_feat_train,
        'X_feat_test': X_feat_test,
        'y_val': y_val,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'f1_score': f1,
        'auc_score': auc_score,
        'feature_names': feature_names,
        'processed_df': df
    }

# Train the model
model_data = train_model(df)

# Unpack model data
model = model_data['model']
tokenizer = model_data['tokenizer']
vocab_size = model_data['vocab_size']
max_length = model_data['max_length']
history = model_data['history']
X_train = model_data['X_train']
X_val   = model_data['X_val']
X_test  = model_data['X_test']
X_feat_train = model_data['X_feat_train']
X_feat_test = model_data['X_feat_test']
y_val   = model_data['y_val']
y_test  = model_data['y_test']
y_pred = model_data['y_pred']
y_pred_prob = model_data['y_pred_prob']
y_pred_prob = model_data['y_pred_prob']
test_accuracy = model_data['test_accuracy']
test_precision = model_data['test_precision']
test_recall = model_data['test_recall']
f1_score_val = model_data['f1_score']
auc_score = model_data['auc_score']

@st.cache_resource
def report(y_true, y_pred):
    print("\nClassification Report:")
    class_report = classification_report(y_test, y_pred)
    
    return class_report

# Classification Report
class_report = report(y_test, y_pred)
print(class_report)

@st.cache_resource
def confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return cm

# Confusion Matrix
cm = confusion(y_test, y_pred)

def loss_validation(history):
    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss', 'val_loss']].plot()
    print("Minimum validation loss: {:0.4f}".format(history_df['val_loss'].min()))
    
    return history_df

# Loss Validation Graph
history_df = loss_validation(history)

@st.cache_resource
def roc_auc_curve(y_test, y_pred_prob):
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    return fpr, tpr, roc_thresholds, roc_auc

# ROC AUC Curve
fpr, tpr, roc_thresholds, roc_auc = roc_auc_curve(y_test, y_pred_prob)

# PREDICTION FUNCTION (using your trained model and tokenizer)
def predict_spam_message(message, threshold=0.5):
    """Enhanced prediction with feature extraction and confidence analysis"""
    try:
        if not message or message.strip() == "":
            return None
        
        start_time = time.time()
        
        # Advanced preprocessing
        processed_message, features = advanced_text_preprocessing(message)
        
        # Tokenization
        sequence = tokenizer.texts_to_sequences([processed_message])
        padded = pad_sequences(sequence, maxlen=max_length, padding='post')
        
        # Feature array
        feature_array = np.array([[
            features['url_count'],
            features['exclamation_count'], 
            features['question_count'],
            features['currency_symbols'],
            features['numbers'],
            features['caps_ratio'],
            features['special_chars']
        ]])
        
        # Make prediction
        prediction_prob = model.predict([padded, feature_array], verbose=0)[0][0]
        
        # Calculate results
        predicted_class = 1 if prediction_prob > threshold else 0
        confidence = prediction_prob if predicted_class == 1 else 1 - prediction_prob
        
        processing_time = time.time() - start_time
        
        # Risk analysis
        risk_factors = []
        if features['url_count'] > 2:
            risk_factors.append("Multiple URLs detected")
        if features['exclamation_count'] > 3:
            risk_factors.append("Excessive exclamation marks")
        if features['currency_symbols'] > 0:
            risk_factors.append("Currency symbols present")
        if features['caps_ratio'] > 0.3:
            risk_factors.append("High capital letter ratio")
        
        result = {
            'prediction': 'Spam' if predicted_class == 1 else 'Ham',
            'confidence': confidence,
            'probability': prediction_prob,
            'processing_time': processing_time,
            'word_count': len(processed_message.split()),
            'features': features,
            'risk_factors': risk_factors,
            'threat_level': 'High' if prediction_prob > 0.8 else 'Medium' if prediction_prob > 0.5 else 'Low'
        }
        
        return result
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def explain_prediction(message):
    """Generate user-friendly LIME explanation for prediction"""
    try:
        # Create LIME explainer with better configuration
        def predictor_fn(texts):
            processed_texts = []
            features_list = []
            for text in texts:
                processed_text, features = advanced_text_preprocessing(text)
                processed_texts.append(processed_text)
                features_list.append([
                    features['url_count'],
                    features['exclamation_count'],
                    features['question_count'],
                    features['currency_symbols'],
                    features['numbers'],
                    features['caps_ratio'],
                    features['special_chars']
                ])
            
            sequences = tokenizer.texts_to_sequences(processed_texts)
            padded = pad_sequences(sequences, maxlen=max_length, padding='post')
            features_array = np.array(features_list, dtype=np.float32)
            
            predictions = model.predict([padded, features_array], verbose=0)
            return np.column_stack([1-predictions, predictions])
        
        # Configure explainer for better word-level explanations
        explainer = LimeTextExplainer(
            class_names=['Safe Email', 'Spam Email'],
            split_expression=r'\W+',  # Split on non-word characters
            bow=False  # Keep word order for better context
        )
        
        explanation = explainer.explain_instance(
            message, 
            predictor_fn, 
            num_features=10,
            num_samples=500,  # Increased for better accuracy
            distance_metric='cosine'
        )
        return explanation
    except Exception as e:
        st.error(f"Explanation error: {str(e)}")
        return None

def create_user_friendly_explanation(explanation, prediction_result):
    """Convert LIME explanation to user-friendly format"""
    if not explanation:
        return None
    
    # Get explanation data
    exp_data = explanation.as_list()
    
    # Create interpretable results
    explanation_results = []
    for word, impact in exp_data:
        # Clean up the word/phrase
        clean_word = word.strip().lower()
        if len(clean_word) < 2:  # Skip very short words
            continue
            
        # Determine impact direction and strength
        if impact > 0:
            direction = "Increases SPAM likelihood"
            emoji = "🚨" if impact > 0.1 else "⚠️" if impact > 0.05 else "📍"
            color = "red" if impact > 0.1 else "orange" if impact > 0.05 else "blue"
        else:
            direction = "Indicates SAFE email"
            emoji = "✅" if impact < -0.1 else "☑️" if impact < -0.05 else "📝"
            color = "green" if impact < -0.1 else "lightgreen" if impact < -0.05 else "gray"
        
        # Create user-friendly explanation
        strength = "Strong" if abs(impact) > 0.1 else "Moderate" if abs(impact) > 0.05 else "Weak"
        
        explanation_results.append({
            'word': clean_word,
            'impact': impact,
            'direction': direction,
            'strength': strength,
            'emoji': emoji,
            'color': color,
            'readable_impact': f"{strength} indicator - {direction.lower()}"
        })
    
    return explanation_results

def get_random_test_email():
    """Get a random email for training game"""
    training_emails = {
        "spam": [
            "URGENT: Your account will be suspended! Click here immediately to verify: http://fake-bank.com",
            "Congratulations! You've WON $1,000,000!!! Claim your prize NOW by clicking here!",
            "FREE VIAGRA! No prescription needed! Order now with 90% discount!",
            "ATTENTION: Your PayPal account has been limited. Restore access immediately!",
            "Make $5000 per week working from home! No experience required! Click to start!",
            "FINAL NOTICE: Your subscription will expire today! Renew now to avoid charges!",
            "You have inherited $2,500,000 from a distant relative. Contact us to claim!",
            "URGENT SECURITY ALERT: Suspicious activity detected. Verify your identity now!"
        ],
        "safe": [
            "Meeting scheduled for tomorrow at 2 PM in conference room A. Please bring quarterly reports.",
            "Thank you for your presentation yesterday. I'd like to schedule a follow-up meeting.",
            "The research paper submission deadline has been extended to March 31st.",
            "Please review the attached project proposal and provide your feedback by Friday.",
            "Team lunch is scheduled for Thursday at 12:30 PM at the Italian restaurant downtown.",
            "The software update will be deployed this weekend. Expect brief downtime on Sunday.",
            "Your monthly newsletter: Latest updates from our development team and upcoming features.",
            "Conference call notes from today's meeting are attached. Please review and confirm."
        ]
    }
    
    # Randomly select spam or safe, then pick random email from that category
    category = random.choice(['spam', 'safe'])
    email = random.choice(training_emails[category])
    
    return {
        'content': email,
        'true_label': category,
        'category': category
    }

def game_explaination(email_data, user_choice, ai_prediction):
    """Generate explanation for the training game"""
    correct_label = email_data['true_label']
    email_content = email_data['content']
    
    explanations = {
        'spam': {
            'urgent_words': "Contains urgent language like 'URGENT', 'IMMEDIATELY', 'NOW'",
            'money_offers': "Promises unrealistic money or prizes",
            'suspicious_links': "Contains suspicious links or requests to click",
            'poor_grammar': "Has suspicious grammar or excessive punctuation",
            'impersonation': "Pretends to be from legitimate institutions",
            'pressure_tactics': "Uses pressure tactics to force quick action"
        },
        'safe': {
            'professional_tone': "Uses professional, courteous language",
            'specific_details': "Contains specific, verifiable information",
            'no_urgency': "No artificial urgency or pressure tactics",
            'legitimate_sender': "Appears to be from a legitimate, known source",
            'reasonable_request': "Makes reasonable, work-related requests",
            'proper_grammar': "Uses proper grammar and punctuation"
        }
    }
    
    # Analyze the email content for specific indicators
    email_lower = email_content.lower()
    found_indicators = []
    
    if correct_label == 'spam':
        if any(word in email_lower for word in ['urgent', 'immediately', 'now', 'quick']):
            found_indicators.append(explanations['spam']['urgent_words'])
        if any(word in email_lower for word in ['won', 'prize', 'money', '$', 'free']):
            found_indicators.append(explanations['spam']['money_offers'])
        if 'http' in email_lower or 'click' in email_lower:
            found_indicators.append(explanations['spam']['suspicious_links'])
        if email_content.count('!') > 2:
            found_indicators.append(explanations['spam']['poor_grammar'])
    else:
        if any(word in email_lower for word in ['meeting', 'schedule', 'please', 'thank']):
            found_indicators.append(explanations['safe']['professional_tone'])
        if any(word in email_lower for word in ['tomorrow', 'pm', 'conference', 'attached']):
            found_indicators.append(explanations['safe']['specific_details'])
    
    if not found_indicators:
        found_indicators = [f"General {correct_label} characteristics detected"]
    
    return found_indicators


# MAIN STREAMLIT APPLICATION
def main():
    # Header
    st.markdown('<h1 class="main-header">Have You Been Phished?</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Email Spam Detector</p>', unsafe_allow_html=True)
    
    # Enhanced sidebar with real-time information
    with st.sidebar:
        st.header("Model Information")
        st.metric("Training Samples", f"{len(X_train):,}")
        st.metric("Vocabulary Size", f"{vocab_size:,}")
        st.metric("Max Sequence Length", max_length)
        st.metric("Model Accuracy", f"{test_accuracy:.1%}")
        st.metric("Precision", f"{test_precision:.3f}")
        st.metric("Recall", f"{test_recall:.3f}")
        st.metric("F1 Score", f"{f1_score_val:.3f}")
        st.metric("AUC Score", f"{auc_score * 100:.3f}")
        
        st.header("⚙️ Detection Settings")
        threshold = st.slider("Spam Threshold", 0.1, 0.9, 0.5, 0.01)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Single Prediction", "Batch Testing", "Email Classification Game", "Model Stats"])
    
    # TAB 1: Single Prediction
    with tab1:
        st.header("Advanced Email Analysis")
        
        # Message input options
        input_method = st.radio("Input Method:", ["Manual Input", "Example Messages"])
        
        if input_method == "Example Messages":
            example_messages = {
                "Phishing Email": "URGENT: Your account will be suspended unless you verify your identity immediately. Click here: http://fake-bank.com/verify",
                "Nigerian Scam": "Congratulations! You have inherited $5,000,000 from a distant relative. Send $500 processing fee to claim your inheritance.",
                "Legitimate Email": "Meeting scheduled for tomorrow at 2 PM in conference room A. Please bring your quarterly reports.",
                "Marketing Spam": "LIMITED TIME OFFER! Get 90% OFF on premium products! Act now before it's too late! Click here to buy now!",
                "Professional Email": "Thank you for your presentation yesterday. I'd like to schedule a follow-up meeting to discuss the project details."
            }
            
            selected_example = st.selectbox("Choose an example:", list(example_messages.keys()))
            message = st.text_area(
                "Email content:",
                value=example_messages[selected_example],
                height=150
            )
        else:
            message = st.text_area(
                "Enter email content to analyze:",
                height=200,
                placeholder="Paste your email content here...\n\nExample:\n'Congratulations! You have won $1,000,000! Click here to claim your prize now!'"
            )
        
        # Toggle AI explanations
        show_explanations = st.checkbox("Show AI Explanation", value=False)
        
        if st.button("Analyze Email", key="single_predict"):
            if message.strip():
                with st.spinner("Analyzing email with advanced AI model..."):
                    result = predict_spam_message(message, threshold=threshold)
                
                if result:
                    # Enhanced prediction display
                    prediction_class = "spam-prediction" if result['prediction'] == 'Spam' else "ham-prediction"
                    threat_emoji = "🚨" if result['threat_level'] == 'High' else "⚠️" if result['threat_level'] == 'Medium' else "✅"
                    
                    st.markdown(f"""
                    <div class="prediction-box {prediction_class}">
                        <h2>{threat_emoji} Prediction: {result['prediction'].upper()}</h2>
                        <h3>Confidence: {result['confidence']:.1%}</h3>
                        <h4>Threat Level: {result['threat_level']}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Spam Probability", f"{result['probability']:.1%}")
                    with col2:
                        st.metric("Processing Time", f"{result['processing_time']*1000:.1f}ms")
                    with col3:
                        st.metric("Word Count", result['word_count'])
                    with col4:
                        st.metric("URLs Found", result['features']['url_count'])
                    with col5:
                        st.metric("Risk Factors", len(result['risk_factors']))
                    
                    # Risk factors analysis
                    if result['risk_factors']:
                        st.subheader("🚩 Risk Factors Detected:")
                        for factor in result['risk_factors']:
                            st.markdown(f"• {factor}")
                    
                    # Probability gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = result['probability'] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Spam Probability (%)"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': threshold * 100
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # AI Explanation
                    if show_explanations:
                        st.subheader("🧠 AI Model Explanation:")
                        st.info("**How to read this:** The AI analyzes individual words and phrases to make its decision. This shows which words influenced the classification and how.")
                        
                        with st.spinner("Generating AI explanation..."):
                            explanation = explain_prediction(message)
                            if explanation:
                                # Get user-friendly explanation
                                user_explanation = create_user_friendly_explanation(explanation, result)
                                
                                if user_explanation:
                                    # Create two columns for better layout
                                    exp_col1, exp_col2 = st.columns([2, 1])
                                    
                                    with exp_col1:
                                        st.markdown("### AI Decision Summary")
                                        
                                        # Show simple summary instead of complex graph
                                        spam_indicators = [item for item in user_explanation if item['impact'] > 0]
                                        safe_indicators = [item for item in user_explanation if item['impact'] < 0]
                                        
                                        if spam_indicators:
                                            st.markdown("**Words that suggest SPAM:**")
                                            for item in spam_indicators[:5]:
                                                impact_strength = "Strong" if abs(item['impact']) > 0.1 else "Moderate" if abs(item['impact']) > 0.05 else "Weak"
                                                st.markdown(f"• **'{item['word']}'** - {impact_strength} spam signal")
                                        
                                        if safe_indicators:
                                            st.markdown("**Words that suggest SAFE email:**")
                                            for item in safe_indicators[:5]:
                                                impact_strength = "Strong" if abs(item['impact']) > 0.1 else "Moderate" if abs(item['impact']) > 0.05 else "Weak"
                                                st.markdown(f"• **'{item['word']}'** - {impact_strength} safety signal")
                                    
                                    with exp_col2:
                                        st.markdown("### Quick Overview")
                                        
                                        total_spam_signals = len([item for item in user_explanation if item['impact'] > 0])
                                        total_safe_signals = len([item for item in user_explanation if item['impact'] < 0])
                                        strongest_signal = max(user_explanation, key=lambda x: abs(x['impact']))
                                        
                                        st.metric("Spam Signals Found", total_spam_signals)
                                        st.metric("Safe Signals Found", total_safe_signals)
                                        st.markdown(f"**Strongest Signal:** '{strongest_signal['word']}' ({strongest_signal['strength']})")
                                    
                                    # Detailed explanation table
                                    st.markdown("### Detailed Word Analysis")
                                    
                                    # Create readable table
                                    exp_df = pd.DataFrame([{
                                        'Word/Phrase': item['word'].title(),
                                        'Effect on Decision': item['direction'],
                                        'Signal Strength': item['strength'], 
                                        'Impact Score': f"{item['impact']:.3f}",
                                        'What This Means': item['readable_impact']
                                    } for item in user_explanation[:8]])
                                    
                                    st.dataframe(exp_df, use_container_width=True, hide_index=True)
                                    
                                    # Updated interpretation guide
                                    st.markdown("""
                                    **Understanding the Table:**
                                    
                                    - **Word/Phrase**: The specific word or phrase the AI focused on
                                    - **Effect on Decision**: Whether this word pushes toward SPAM or indicates a SAFE email
                                    - **Signal Strength**: How much influence this word has (Strong/Moderate/Weak)
                                    - **Impact Score**: Technical score (positive = more spam-like, negative = more safe)
                                    - **What This Means**: Plain English explanation of the word's influence
                                    
                                    **Key Points:**
                                    - Words with **higher impact scores** have more influence on the AI's decision
                                    - **Positive scores** mean the word makes the email look more like spam
                                    - **Negative scores** mean the word makes the email look more legitimate
                                    - The AI considers **all words together** to make the final decision
                                    """)
                                else:
                                    st.warning("Could not generate detailed explanation for this message.")
                    
            else:
                st.warning("Please enter an email message to analyze.")
    
    # TAB 2: Batch Testing
    with tab2:
        st.header("Comprehensive Batch Testing")
    
        # File upload options
        st.subheader("Upload Your Email Data")
        
        # Add .txt to file uploader
        uploaded_file = st.file_uploader(
            "Upload file with emails", 
            type=['txt'],
            help="TXT files should have one email per line."
        )
        
        # Reading TXT File
        if uploaded_file is not None:
            try:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if file_extension == 'txt':
                    # Handle .txt files
                    st.info("Processing TXT file - each line will be treated as one email")
                    
                    # Read the text file
                    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                    file_content = stringio.read()
                    
                    # Split by lines and clean up
                    email_lines = file_content.strip().split('\n')
                    email_lines = [line.strip() for line in email_lines if line.strip()]  # Remove empty lines
                    
                    # Create a list of emails
                    batch_emails = email_lines
                    st.success(f"Loaded {len(batch_emails)} emails from TXT file")
                    
                    # Show preview of first few emails
                    with st.expander("Preview First 5 Emails"):
                        for i, email in enumerate(batch_emails[:5]):
                            st.write(f"**Email {i+1}:** {email[:100]}{'...' if len(email) > 100 else ''}")
                
                # Analysis button
                if st.button("Analyze All Emails", key="analyze_uploaded"):
                    if len(batch_emails) > 1000:
                        st.warning(f"Large file detected ({len(batch_emails)} emails). This may take several minutes.")
                        proceed = st.checkbox("I understand this will take time. Proceed anyway.")
                        if not proceed:
                            st.stop()
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []
                    
                    # Process each email
                    start_time = time.time()
                    for i, email in enumerate(batch_emails):
                        # Update progress
                        progress = (i + 1) / len(batch_emails)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing email {i+1}/{len(batch_emails)} ({progress:.1%} complete)")
                        
                        # Analyze email
                        result = predict_spam_message(email, threshold=threshold)
                        
                        if result:
                            results.append({
                                'Email_ID': i + 1,
                                'Message_Preview': email[:80] + "..." if len(email) > 80 else email,
                                'Full_Message': email,  # Store full message for detailed view
                                'Prediction': result['prediction'],
                                'Confidence': f"{result['confidence']:.1%}",
                                'Spam_Probability': result['probability'],
                                'Threat_Level': result['threat_level'],
                                'Risk_Factors_Count': len(result['risk_factors']),
                                'Processing_Time_ms': f"{result['processing_time']*1000:.1f}",
                                'Word_Count': result['word_count'],
                                'URLs_Found': result['features']['url_count'],
                                'Exclamation_Marks': result['features']['exclamation_count']
                            })
                    
                    processing_time = time.time() - start_time
                    progress_bar.progress(1.0)
                    status_text.text(f"Analysis complete! Processed {len(results)} emails in {processing_time:.1f} seconds")
                    
                    # Store results in session state for later use
                    st.session_state.batch_results = results
                    st.session_state.batch_processing_time = processing_time
                                    
                    # Display result from session state
                    if 'batch_results' in st.session_state and st.session_state.batch_results:
                        results = st.session_state.batch_results
                        processing_time = st.session_state.batch_processing_time
                        
                        st.success(f"Analysis Complete! Processed {len(results)} emails in {processing_time:.1f} seconds")
                    
                        # Display comprehensive results
                        st.success(f"Analysis Complete! Processed {len(results)} emails in {processing_time:.1f} seconds")
                        
                        # Summary statistics
                        spam_count = len([r for r in results if r['Prediction'] == 'Spam'])
                        ham_count = len(results) - spam_count
                        high_threat = len([r for r in results if r['Threat_Level'] == 'High'])
                        avg_processing = sum([float(r['Processing_Time_ms']) for r in results]) / len(results)
                        
                        # Metrics display
                        st.subheader("Analysis Summary")
                        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
                        
                        with metric_col1:
                            st.metric("Total Processed", len(results))
                        with metric_col2:
                            st.metric("Spam Detected", spam_count, f"{(spam_count/len(results)*100):.1f}%")
                        with metric_col3:
                            st.metric("Safe Emails", ham_count, f"{(ham_count/len(results)*100):.1f}%")
                        with metric_col4:
                            st.metric("High Threat", high_threat)
                        with metric_col5:
                            st.metric("Avg Time", f"{avg_processing:.1f}ms")
                        
                        # Results table with better formatting
                        st.subheader("Detailed Results")
                        
                        # Create display dataframe (without full message for table view)
                        display_df = pd.DataFrame([{
                            'ID': r['Email_ID'],
                            'Preview': r['Message_Preview'],
                            'Classification': r['Prediction'],
                            'Confidence': r['Confidence'],
                            'Threat Level': r['Threat_Level'],
                            'Risk Factors': r['Risk_Factors_Count'],
                            'URLs': r['URLs_Found'],
                            'Word Count': r['Word_Count']
                        } for r in results])
                        
                        # Color-code the dataframe
                        def color_rows(row):
                            if row['Classification'] == 'Spam':
                                return ['background-color: #ffebee; color: #000000; font-weight: bold'] * len(row)
                            else:
                                return ['background-color: #e8f5e8; color: #000000; font-weight: bold'] * len(row)

                        
                        styled_df = display_df.style.apply(color_rows, axis=1)
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
                        
                        # Download enhanced results
                        full_results_df = pd.DataFrame(results)
                        csv = full_results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Detailed Results (CSV)",
                            data=csv,
                            file_name=f"email_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Advanced visualizations
                        st.subheader("Analysis Visualizations")
                        
                        viz_col1, viz_col2 = st.columns(2)
                        
                        with viz_col1:
                            # Pie chart
                            fig_pie = px.pie(
                                values=[spam_count, ham_count],
                                names=['Spam', 'Safe'],
                                title='Email Classification Distribution',
                                color_discrete_map={'Spam': '#ff6b6b', 'Safe': '#51cf66'}
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with viz_col2:
                            # Threat level distribution
                            threat_counts = {}
                            for result in results:
                                threat_level = result['Threat_Level']
                                threat_counts[threat_level] = threat_counts.get(threat_level, 0) + 1
                            
                            fig_threat = px.bar(
                                x=list(threat_counts.keys()),
                                y=list(threat_counts.values()),
                                title='Threat Level Distribution',
                                color=list(threat_counts.keys()),
                                color_discrete_map={'High': '#ff4444', 'Medium': '#ffaa44', 'Low': '#44ff44'}
                            )
                            st.plotly_chart(fig_threat, use_container_width=True)
                        
                    if 'batch_results' in st.session_state:
                        if st.button("Clear Results", key="clear_batch_results"):
                            # Clear all batch-related session state
                            if 'batch_results' in st.session_state:
                                del st.session_state.batch_results
                            if 'batch_processing_time' in st.session_state:
                                del st.session_state.batch_processing_time
                            if 'selected_email_id' in st.session_state:
                                del st.session_state.selected_email_id
                            st.rerun()
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Make sure your TXT file has one email per line")
        
        # Predefined test cases
        st.subheader("Quick Test with Sample Messages")
        
        test_categories = {
            "Phishing Emails": [
                "URGENT: Your account will be suspended unless you verify your identity immediately. Click here: http://fake-bank.com/verify",
                "Security Alert: Unusual activity detected on your account. Login here to secure: http://secure-login-fake.com",
                "Your PayPal account has been limited. Verify your information to restore access: http://paypal-verify-fake.com"
            ],
            "Marketing Spam": [
                "LIMITED TIME OFFER! Get 90% OFF on premium products! Act now before it's too late!",
                "FREE VIAGRA! NO PRESCRIPTION NEEDED! ORDER NOW! DISCREET SHIPPING!",
                "Make $5000 per week working from home! No experience required! Click here to start!"
            ],
            "Legitimate Emails": [
                "Meeting scheduled for tomorrow at 2 PM in conference room A. Please bring your quarterly reports.",
                "Thank you for your presentation yesterday. I'd like to schedule a follow-up meeting to discuss the project.",
                "Research paper submission deadline extended to March 31st. Please submit via the university portal."
            ]
        }
        
        selected_category = st.selectbox("Choose test category:", list(test_categories.keys()))
        
        if st.button("Test Sample Messages"):
            test_messages = test_categories[selected_category]
            results = []
            
            progress_bar = st.progress(0)
            for i, msg in enumerate(test_messages):
                result = predict_spam_message(msg, threshold=threshold)
                if result:
                    results.append({
                        'Message': msg[:80] + "..." if len(msg) > 80 else msg,
                        'Prediction': result['prediction'],
                        'Confidence': f"{result['confidence']:.1%}",
                        'Probability': f"{result['probability']:.3f}",
                        'Threat Level': result['threat_level'],
                        'Processing Time': f"{result['processing_time']*1000:.1f}ms"
                    })
                progress_bar.progress((i + 1) / len(test_messages))
            
            # Display results
            if results:
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                
                # Summary
                spam_count = len([r for r in results if r['Prediction'] == 'Spam'])
                ham_count = len(results) - spam_count
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Spam Detected", spam_count)
                with col2:
                    st.metric("Ham (Legitimate)", ham_count)
    
    # TAB 3: AI Training Challenge
    with tab3:
        st.header("AI Training Challenge")
        st.markdown("""
        **Test your spam detection skills against our AI!**
        
        This interactive game helps you:
        - Learn to identify spam patterns
        - Compare your accuracy with AI
        - Understand why emails are classified as spam or safe
        - Improve your cybersecurity awareness
        """)
        
        """Interactive spam detection training"""
        st.subheader("Spam Detection Training Game")
        st.markdown("**Challenge yourself!** Can you identify spam as well as our AI? Test your skills and learn!")
        
        # Initialize game state
        if 'game_score' not in st.session_state:
            st.session_state.game_score = 0
            st.session_state.game_round = 1
            st.session_state.ai_score = 0
            st.session_state.current_email = None
            st.session_state.game_started = False
        
        # Game stats display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Round", st.session_state.game_round)
        with col2:
            st.metric("Your Score", st.session_state.game_score)
        with col3:
            st.metric("AI Score", st.session_state.ai_score)
        
        # Start new round button
        if st.button("Get New Email to Classify") or not st.session_state.game_started:
            st.session_state.current_email = get_random_test_email()
            st.session_state.game_started = True
            st.session_state.user_answered = False
            st.rerun()
        
        # Display current email if available
        if st.session_state.current_email:
            st.markdown("### Classify this email:")
            
            # Display the email in a nice box
            st.code(st.session_state.current_email['content'], language=None)
            
            # User choice buttons
            st.markdown("### What do you think?")
            col1, col2 = st.columns(2)
            
            user_choice = None
            with col1:
                if st.button("SAFE Email", key=f"safe_{st.session_state.game_round}", use_container_width=True):
                    user_choice = "safe"
            with col2:
                if st.button("SPAM Email", key=f"spam_{st.session_state.game_round}", use_container_width=True):
                    user_choice = "spam"
            
            # Process the answer
            if user_choice and not getattr(st.session_state, 'user_answered', False):
                st.session_state.user_answered = True
                
                # Get AI prediction
                ai_prediction = predict_spam_message(st.session_state.current_email['content'])
                ai_choice = "spam" if ai_prediction['prediction'].lower() == "spam" else "safe"
                correct_answer = st.session_state.current_email['true_label']
                
                # Score both user and AI
                user_correct = (user_choice == correct_answer)
                ai_correct = (ai_choice == correct_answer)
                
                # Update scores
                if user_correct:
                    st.session_state.game_score += 1
                if ai_correct:
                    st.session_state.ai_score += 1
                
                # Display results
                st.markdown("---")
                st.markdown("### Results:")
                
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    if user_correct:
                        st.success(f"**You:** Correct!\nYou said: {user_choice.upper()}")
                    else:
                        st.error(f"**You:** Wrong\nYou said: {user_choice.upper()}")
                
                with result_col2:
                    if ai_correct:
                        st.success(f"**AI:** Correct!\nAI said: {ai_choice.upper()} ({ai_prediction['confidence']:.1%} confidence)")
                    else:
                        st.error(f"**AI:** Wrong\nAI said: {ai_choice.upper()} ({ai_prediction['confidence']:.1%} confidence)")
                
                with result_col3:
                    st.info(f"**Correct Answer:** {correct_answer.upper()}")
                
                # Show explanation
                explanations = game_explaination(st.session_state.current_email, user_choice, ai_prediction)
                
                st.markdown("### Why this email is " + correct_answer.upper() + ":")
                for explanation in explanations:
                    st.markdown(f"• {explanation}")
                
                # Show detailed AI analysis if it was spam
                if correct_answer == "spam":
                    st.markdown("### Detailed Threat Analysis:")
                    threat_factors = []
                    content = st.session_state.current_email['content'].lower()
                    
                    if any(word in content for word in ['urgent', 'immediate', 'now', 'quick']):
                        threat_factors.append("Urgency pressure tactics")
                    if any(word in content for word in ['click', 'http', 'link']):
                        threat_factors.append("Suspicious links or click requests")
                    if any(word in content for word in ['$', 'money', 'prize', 'won', 'free']):
                        threat_factors.append("Money/prize offers")
                    if st.session_state.current_email['content'].count('!') > 2:
                        threat_factors.append("Excessive exclamation marks")
                    
                    for factor in threat_factors:
                        st.markdown(f"• {factor}")
                
                # Prepare for next round
                st.session_state.game_round += 1
                
                # Show progress
                if st.session_state.game_round > 1:
                    user_accuracy = (st.session_state.game_score / (st.session_state.game_round - 1)) * 100
                    ai_accuracy = (st.session_state.ai_score / (st.session_state.game_round - 1)) * 100
                    
                    st.markdown("### Performance Comparison:")
                    
                    perf_col1, perf_col2 = st.columns(2)
                    with perf_col1:
                        st.metric("Your Accuracy", f"{user_accuracy:.1f}%")
                    with perf_col2:
                        st.metric("AI Accuracy", f"{ai_accuracy:.1f}%")
                    
                    # Motivational messages
                    if user_accuracy > ai_accuracy:
                        st.success("You're beating the AI! Great spam detection skills!")
                    elif user_accuracy == ai_accuracy:
                        st.info("You're tied with the AI! Impressive performance!")
                    else:
                        st.warning("The AI is ahead, but keep trying! You're learning valuable skills!")
            
            # Reset game option
            if st.session_state.game_round > 1:
                if st.button("🔄 Start New Game"):
                    for key in ['game_score', 'game_round', 'ai_score', 'current_email', 'game_started', 'user_answered']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

    # TAB 4: Model Statistics
    with tab4:
        st.header("📊 Model Performance & Statistics")
        
        # Executive Summary Cards
        st.subheader("🎯 Executive Summary")
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 1.5rem; border-radius: 15px; text-align: center;">
                <h3 style="margin: 0; color: white;">Model Accuracy</h3>
                <h1 style="margin: 0.5rem 0; color: white;">{:.1%}</h1>
                <p style="margin: 0; color: white;">Overall Performance</p>
            </div>
            """.format(test_accuracy), unsafe_allow_html=True)
        
        with summary_col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    color: white; padding: 1.5rem; border-radius: 15px; text-align: center;">
                <h3 style="margin: 0; color: white;">AUC Score</h3>
                <h1 style="margin: 0.5rem 0; color: white;">{:.3f}</h1>
                <p style="margin: 0; color: white;">ROC Performance</p>
            </div>
            """.format(auc_score), unsafe_allow_html=True)
        
        with summary_col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    color: white; padding: 1.5rem; border-radius: 15px; text-align: center;">
                <h3 style="margin: 0; color: white;">F1 Score</h3>
                <h1 style="margin: 0.5rem 0; color: white;">{:.3f}</h1>
                <p style="margin: 0; color: white;">Balanced Performance</p>
            </div>
            """.format(f1_score_val), unsafe_allow_html=True)
        
        with summary_col4:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #2d5a87 0%, #1e3c72 100%); 
                    color: white; padding: 1.5rem; border-radius: 15px; text-align: center;">
                <h3 style="margin: 0; color: white;">Dataset Size</h3>
                <h1 style="margin: 0.5rem 0; color: white;">{:,}</h1>
                <p style="margin: 0; color: white;">Training Samples</p>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed Performance Metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Model Configuration")
            config_data = {
                "Parameter": [
                    "Model Type", "Total Dataset Size", "Training Set", "Validation Set", 
                    "Test Set", "Vocabulary Size", "Embedding Dimension", "LSTM Units",
                    "Max Sequence Length", "Dropout Rate", "Learning Rate", "Early Stopping Patience"
                ],
                "Value": [
                    "Hybrid LSTM + Features", f"{len(df):,} emails", f"{len(X_train):,} emails",
                    f"{len(X_val):,} emails", f"{len(X_test):,} emails", f"{len(tokenizer.word_index):,} words",
                    "128", "64", str(max_length), "0.6", "0.0005", "3 epochs"
                ]
            }
            config_df = pd.DataFrame(config_data)
            st.dataframe(config_df, use_container_width=True, hide_index=True)
            
        with col2:
            st.subheader("🎯 Performance Metrics")
            metrics_data = {
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC Score"],
                "Score": [f"{test_accuracy:.3f}", f"{test_precision:.3f}", f"{test_recall:.3f}", 
                        f"{f1_score_val:.3f}", f"{auc_score:.3f}"],
                "Percentage": [f"{test_accuracy:.1%}", f"{test_precision:.1%}", f"{test_recall:.1%}",
                            f"{f1_score_val:.1%}", f"{auc_score:.1%}"]
            }
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Training History Analysis
        st.subheader("📊 Training History Analysis")
        
        # Process history data
        history_df = pd.DataFrame(history.history)
        min_val_loss = history_df['val_loss'].min()
        final_epoch = len(history_df)
        
        # Training metrics
        train_col1, train_col2, train_col3, train_col4 = st.columns(4)
        with train_col1:
            st.metric("Training Epochs", final_epoch)
        with train_col2:
            st.metric("Min Validation Loss", f"{min_val_loss:.4f}")
        with train_col3:
            st.metric("Final Training Loss", f"{history_df['loss'].iloc[-1]:.4f}")
        with train_col4:
            early_stopped = "Yes" if final_epoch < 15 else "No"
            st.metric("Early Stopping", early_stopped)
        
        # Training vs Validation Loss Plot
        fig_training = go.Figure()
        
        fig_training.add_trace(go.Scatter(
            x=list(range(len(history_df))),
            y=history_df['loss'],
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6)
        ))
        
        fig_training.add_trace(go.Scatter(
            x=list(range(len(history_df))),
            y=history_df['val_loss'],
            mode='lines+markers',
            name='Validation Loss',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=6)
        ))
        
        fig_training.update_layout(
            title='Training vs Validation Loss Over Epochs',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_training, use_container_width=True)
        
        st.markdown("---")
        
        # ROC Curve and Confusion Matrix
        st.subheader("🎯 Model Evaluation Visualizations")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.markdown("#### ROC Curve Analysis")
            
            # ROC Curve
            fig_roc = go.Figure()
            
            # Add ROC curve
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                line=dict(color='darkorange', width=3)
            ))
            
            # Add diagonal reference line
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='navy', width=2, dash='dash')
            ))
            
            fig_roc.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=400,
                template='plotly_white',
                showlegend=True
            )
            
            st.plotly_chart(fig_roc, use_container_width=True)
            
            # ROC Interpretation
            if roc_auc > 0.9:
                roc_interpretation = "🟢 Excellent"
            elif roc_auc > 0.8:
                roc_interpretation = "🟡 Good"
            elif roc_auc > 0.7:
                roc_interpretation = "🟠 Fair"
            else:
                roc_interpretation = "🔴 Poor"
            
            st.markdown(f"**AUC Interpretation:** {roc_interpretation} ({roc_auc:.3f})")
        
        with viz_col2:
            st.markdown("#### Confusion Matrix")
            
            # Create confusion matrix heatmap
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Ham', 'Predicted Spam'],
                y=['Actual Ham', 'Actual Spam'],
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
                hoverongaps=False
            ))
            
            fig_cm.update_layout(
                title='Confusion Matrix',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Confusion Matrix Insights
            tn, fp, fn, tp = cm.ravel()
            st.markdown(f"""
            **Matrix Breakdown:**
            - True Negatives (Ham correctly identified): {tn:,}
            - False Positives (Ham misclassified as Spam): {fp:,}
            - False Negatives (Spam misclassified as Ham): {fn:,}
            - True Positives (Spam correctly identified): {tp:,}
            """)
        
        st.markdown("---")
        
        # Detailed Classification Report
        st.subheader("📋 Detailed Classification Report")
        
        # Create a nicer classification report display
        from sklearn.metrics import classification_report
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        
        # Extract metrics for each class
        report_data = []
        for class_name, metrics in report_dict.items():
            if class_name in ['0', '1']:  # Only for actual classes
                class_label = 'Ham (Safe)' if class_name == '0' else 'Spam'
                report_data.append({
                    'Class': class_label,
                    'Precision': f"{metrics['precision']:.3f}",
                    'Recall': f"{metrics['recall']:.3f}",
                    'F1-Score': f"{metrics['f1-score']:.3f}",
                    'Support': f"{int(metrics['support']):,}"
                })
        
        # Add overall metrics
        report_data.append({
            'Class': 'Overall (Weighted)',
            'Precision': f"{report_dict['weighted avg']['precision']:.3f}",
            'Recall': f"{report_dict['weighted avg']['recall']:.3f}",
            'F1-Score': f"{report_dict['weighted avg']['f1-score']:.3f}",
            'Support': f"{len(y_test):,}"
        })
        
        report_df = pd.DataFrame(report_data)
        st.dataframe(report_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Performance Insights
        st.subheader("💡 Model Performance Insights")
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.markdown("#### ✅ **Strengths**")
            strengths = []
            if test_accuracy > 0.9:
                strengths.append("High overall accuracy (>90%)")
            if test_precision > 0.9:
                strengths.append("Excellent precision - low false positives")
            if test_recall > 0.9:
                strengths.append("Excellent recall - catches most spam")
            if auc_score > 0.9:
                strengths.append("Outstanding ROC AUC performance")
            if len(strengths) == 0:
                strengths.append("Model shows reasonable performance")
            
            for strength in strengths:
                st.markdown(f"• {strength}")
        
        with insight_col2:
            st.markdown("#### ⚠️ **Areas for Improvement**")
            improvements = []
            if test_accuracy < 0.85:
                improvements.append("Overall accuracy could be improved")
            if test_precision < 0.85:
                improvements.append("Reduce false positive rate")
            if test_recall < 0.85:
                improvements.append("Improve spam detection rate")
            if len(improvements) == 0:
                improvements.append("Model performance is excellent!")
            
            for improvement in improvements:
                st.markdown(f"• {improvement}")

# Run the Streamlit app
if __name__ == "__main__":
    main()

