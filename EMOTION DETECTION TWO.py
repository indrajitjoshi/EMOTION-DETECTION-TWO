import streamlit as st
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D
import os

# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# --- Configuration ---
MAX_WORDS = 20000 # Max number of words to keep in the vocabulary
MAX_LEN = 100     # Max length of a sequence (review)
EMBEDDING_DIM = 100 # Dimension of the word embeddings
NUM_CLASSES = 6
EPOCHS = 15 # Increased for higher accuracy with Ensemble, expects longer initial load (15-25 minutes)

# Define the emotion labels for mapping
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
label_to_id = {label: i for i, label in enumerate(emotion_labels)}
id_to_label = {i: label for i, label in enumerate(emotion_labels)}


# --- Model Building Functions ---

def build_cnn_bilstm_model():
    """Builds the Hybrid CNN-BiLSTM model."""
    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=4),
        Bidirectional(LSTM(64)),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_bilstm_model():
    """Builds a standalone Bi-directional LSTM model."""
    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_gru_model():
    """Builds a Gated Recurrent Unit (GRU) model."""
    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        GRU(128, return_sequences=True),
        Dropout(0.4),
        GRU(64),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# --- Caching function to load data and train the ensemble once ---
# No explicit 'show_spinner' to avoid displaying the message, Streamlit will show its default 'Running...'
@st.cache_resource
def load_and_train_ensemble():
    """Loads data, trains the three models, and prepares the ensemble."""
    
    # 1. Load Data
    data = load_dataset("dair-ai/emotion", "split")
    
    train_texts = list(data['train']['text'])
    train_labels = list(data['train']['label'])
    test_texts = list(data['test']['text'])
    test_labels = list(data['test']['label'])
    
    all_texts = train_texts + test_texts

    # 2. Tokenization and Sequencing
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
    tokenizer.fit_on_texts(all_texts)

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    
    train_padded = pad_sequences(train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    test_padded = pad_sequences(test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    # Convert labels to one-hot encoding
    train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=NUM_CLASSES)
    
    # 3. Build and Train Models
    models = {
        'CNN-BiLSTM': build_cnn_bilstm_model(),
        'BiLSTM': build_bilstm_model(),
        'GRU': build_gru_model()
    }

    st.warning("Starting Ensemble Training (CNN-BiLSTM, BiLSTM, GRU). This will take 15-25 minutes to complete on first run.")
    
    # Train all models
    for name, model in models.items():
        st.info(f"Training {name}...")
        model.fit(
            train_padded, 
            train_labels_one_hot,
            epochs=EPOCHS, 
            batch_size=32,
            validation_split=0.1,
            verbose=0 # Run silently in Streamlit
        )
    st.success("All models trained successfully!")

    # 4. Ensemble Prediction and Evaluation
    
    # Get predictions from all models (probabilities)
    pred_probs_cnn_bilstm = models['CNN-BiLSTM'].predict(test_padded, verbose=0)
    pred_probs_bilstm = models['BiLSTM'].predict(test_padded, verbose=0)
    pred_probs_gru = models['GRU'].predict(test_padded, verbose=0)
    
    # Soft Voting: Average the probabilities
    ensemble_probs = (pred_probs_cnn_bilstm + pred_probs_bilstm + pred_probs_gru) / 3
    
    # Final prediction
    y_pred_ensemble = np.argmax(ensemble_probs, axis=1)
    
    # Calculate Metrics for Ensemble
    accuracy = accuracy_score(test_labels, y_pred_ensemble)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, y_pred_ensemble, average='macro', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return models, tokenizer, metrics, train_padded, test_padded, test_labels


# --- Prediction Function for Ensemble ---
def predict_emotion_ensemble(models, tokenizer, text):
    """Tokenizes input text, predicts with all models, and averages probabilities."""
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Collect predictions from all models
    all_probs = []
    for model in models.values():
        all_probs.append(model.predict(padded_sequence, verbose=0)[0])
    
    # Ensemble: Average the probabilities
    ensemble_probabilities = np.mean(all_probs, axis=0)
    
    # Get the index of the highest probability
    predicted_id = np.argmax(ensemble_probabilities)
    predicted_label = id_to_label[predicted_id].capitalize()
    
    # Format data for bar chart
    prob_data = pd.DataFrame({
        'Emotion': [label.capitalize() for label in emotion_labels],
        'Confidence': ensemble_probabilities
    }).set_index('Emotion')

    return predicted_label, prob_data


# --- Main Streamlit App ---
def main():
    
    # --- CSS Injection for Engaging Background and Hover Effects ---
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        
        /* Animated Gradient Background */
        .stApp {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(-45deg, #1e004a, #5b1076, #004a60, #00707a);
            background-size: 400% 400%;
            animation: gradientBG 20s ease infinite;
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Header Styling */
        .header {
            color: #FFFFFF;
            text-align: center;
            padding: 15px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            margin-bottom: 25px;
            letter-spacing: 1.5px;
        }
        h1, h2, h3, h4, .stSidebar h2 {
            color: #FFD700; /* Gold/Yellow for contrast */
        }
        
        /* Primary Button Hover Effect */
        .stButton>button {
            border-radius: 8px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
            background-color: #00707a; 
        }

        /* Text Area Focus/Hover */
        .stTextArea textarea {
            border: 2px solid #5b1076;
            transition: border 0.3s ease;
        }
        .stTextArea textarea:focus {
            border: 2px solid #FFD700;
            box-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
        }

        /* Metrics boxes (sidebar) */
        [data-testid="stMetric"] {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: scale(1.03);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="header"><h1><span style="color: #FFD700;">🧠</span> Ensemble Emotion Detector <span style="color: #FFD700;">🚀</span></h1></div>', unsafe_allow_html=True)
    st.markdown("<h3 style='color: white;'>Hybrid CNN-BiLSTM, BiLSTM, and GRU Ensemble Analysis</h3>", unsafe_allow_html=True)
    
    # Load and train the ensemble (cached)
    models, tokenizer, metrics, _, _, _ = load_and_train_ensemble()

    # --- Metrics Display (Sidebar) ---
    st.sidebar.markdown("<h2 style='color: #FFD700;'>Ensemble Performance</h2>", unsafe_allow_html=True)
    st.sidebar.markdown(f"**Architecture:** Soft-Voting Ensemble (Trained for {EPOCHS} Epochs)")
    
    # Display metrics in columns
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    col2.metric("Macro F1-Score", f"{metrics['f1_score']:.4f}")
    
    col3, col4 = st.sidebar.columns(2)
    col3.metric("Macro Precision", f"{metrics['precision']:.4f}")
    col4.metric("Macro Recall", f"{metrics['recall']:.4f}")

    if metrics['accuracy'] >= 0.85:
        st.sidebar.success(f"✅ Target Accuracy of 85% Achieved!")
    else:
        st.sidebar.warning(f"⚠️ Target Accuracy of 85% Not Met Yet.")
    
    # --- Input Section ---
    st.markdown("<h3 style='color: white;'>Enter a Product Review:</h3>", unsafe_allow_html=True)
    user_input = st.text_area("Review Text", 
                              "I feel absolutely wonderful and proud of this amazing result, it's pure happiness!", 
                              height=150)
    
    if st.button("Detect Emotion", use_container_width=True, type="primary"):
        if user_input:
            predicted_emotion, prob_data = predict_emotion_ensemble(models, tokenizer, user_input)
            
            st.markdown("<hr style='border: 1px solid white;'>", unsafe_allow_html=True)
            
            # Display Prediction
            st.markdown(f"<h2 style='color: #FFD700;'>Predicted Emotion: <span style='background-color: #5b1076; padding: 8px 15px; border-radius: 8px; color: white;'>{predicted_emotion}</span></h2>", unsafe_allow_html=True)
            
            # Display Confidence Breakdown
            st.markdown("<h3 style='color: white;'>Confidence Breakdown:</h3>", unsafe_allow_html=True)
            
            # Use Pandas DataFrame index for chart labels
            st.bar_chart(prob_data, use_container_width=True)
            
            st.markdown("<hr style='border: 1px solid white;'>", unsafe_allow_html=True)


    # --- Sample Reviews Section (Designed for High Confidence) ---
    st.markdown("<h3 style='color: white;'>Example Reviews to Test (Optimized for Ensemble Prediction):</h3>", unsafe_allow_html=True)
    
    sample_data = {
        'Emotion': ['Joy', 'Sadness', 'Anger', 'Fear', 'Love', 'Surprise'],
        'Review': [
            "This product is absolutely wonderful and makes me feel pure bliss and delight! I am ecstatic!",
            "I feel utterly heartbroken, devastated, and miserable. This outcome is a crushing disappointment.",
            "I am incredibly furious, this is totally unacceptable! I want to throw my computer in utter rage!",
            "My heart is pounding, I'm absolutely terrified of this result, a wave of panic is washing over me.",
            "I have deep affection and a profound devotion for this brand. I truly love this item.",
            "I am completely astonished! I never expected this result; it's the most amazing revelation!"
        ]
    }
    sample_df = pd.DataFrame(sample_data)
    st.table(sample_df)


if __name__ == "__main__":
    main()
