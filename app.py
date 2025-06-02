import streamlit as st
import pickle
import string
import nltk
import ssl
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Create NLTK data directory if it doesn't exist
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Configure SSL for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
def download_nltk_data():
    try:
        # Download required NLTK data
        for package in ['punkt', 'stopwords']:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                nltk.download(package)
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {str(e)}")
        st.info("Please try running these commands in your terminal/command prompt:\n```\npython -m nltk.downloader punkt\npython -m nltk.downloader stopwords\n```")
        return False

# Attempt to download NLTK data
if not download_nltk_data():
    st.stop()

# Custom CSS for innovative modern theme
st.markdown("""
<style>
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        color: white;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .main-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        display: flex;
        flex-direction: row;
        gap: 20px;
        margin-top: 0 !important;
    }
    
    .left-panel {
        width: 200px;
        padding: 20px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .right-panel {
        flex: 1;
    }
    
    .header {
        text-align: center;
        margin-bottom: 30px;
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .title {
        font-size: 3em;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        animation: titleGlow 2s infinite alternate;
    }
    
    @keyframes titleGlow {
        from { text-shadow: 0 0 10px rgba(255,255,255,0.5); }
        to { text-shadow: 0 0 20px rgba(255,255,255,0.8); }
    }
    
    .subtitle {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.2em;
        animation: fadeIn 1s ease-in 0.5s both;
    }
    
    .input-box {
        background: rgba(255, 255, 255, 0.1);
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        animation: slideUp 1s ease-out 0.5s both;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .input-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        padding: 15px !important;
        font-size: 16px !important;
        min-height: 150px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #4ecdc4 !important;
        box-shadow: 0 0 15px rgba(78, 205, 196, 0.5) !important;
    }
    
    .stTextArea textarea::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
        font-style: italic !important;
    }
    
    .stButton button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 16px;
        width: 100%;
        transition: all 0.3s ease;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    .prediction-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: fadeIn 1s ease-in 1s both;
    }
    
    .model-switch {
        display: flex;
        flex-direction: column;
        gap: 15px;
        margin: 20px 0;
    }
    
    .switch-button {
        padding: 15px;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: white;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .switch-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: 0.5s;
    }
    
    .switch-button:hover::before {
        left: 100%;
    }
    
    .switch-button.active {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        transform: translateX(10px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .result-box {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: slideUp 1s ease-out 1.5s both;
    }
    
    .confidence-meter {
        height: 12px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        transition: width 0.5s ease;
    }
    
    .warning {
        color: #ff6b6b;
        animation: shake 0.5s ease-in-out;
    }
    
    .success {
        color: #4ecdc4;
        animation: bounce 0.5s ease-in-out;
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
    
    /* Hide Streamlit's default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Remove empty box and top spacing */
    .element-container:empty {
        display: none;
    }
    
    .stApp > div:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Improve radio button styling */
    .stRadio > div {
        flex-direction: column;
        gap: 10px;
    }
    
    .stRadio > div > div {
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Simple word tokenization using split()
    text = text.split()
    
    # Remove special characters and keep only alphanumeric words
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
    
    # Apply stemming
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# Load the models
try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    # Determine which classifier is being used
    if isinstance(model, RandomForestClassifier):
        classifier_name = "Random Forest"
        rf_model = model
        xgb_model = None
    elif isinstance(model, xgb.XGBClassifier):
        classifier_name = "XGBoost"
        xgb_model = model
        rf_model = None
    else:
        classifier_name = "Unknown"
        rf_model = None
        xgb_model = None
        
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.info("Please make sure you have run the spam_detection_fixed.py script first to train and save the models.")
    st.stop()

# UI Elements
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Left Panel with Model Switch
st.markdown('<div class="left-panel">', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; margin-bottom: 20px;">Select Classifier</h3>', unsafe_allow_html=True)
model_choice = st.radio(
    "Select Classifier:",
    ["Random Forest", "XGBoost"],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# Right Panel with Main Content
st.markdown('<div class="right-panel">', unsafe_allow_html=True)
st.markdown('<h1 class="title">Email Spam Detector</h1>', unsafe_allow_html=True)

# Input box
input_text = st.text_area(
    "Enter the content",
    placeholder="Enter your email content here...",
    height=200
)

if st.button('Analyze Email'):
    if not input_text:
        st.warning("Please enter some text to analyze")
    else:
        try:
            # Preprocess and vectorize
            transform_input = transform_text(input_text)
            vector_input = tfidf.transform([transform_input])
            
            # Get prediction based on selected model
            if model_choice == "Random Forest" and rf_model is not None:
                result = rf_model.predict(vector_input)[0]
                proba = rf_model.predict_proba(vector_input)[0]
            elif model_choice == "XGBoost" and xgb_model is not None:
                result = xgb_model.predict(vector_input)[0]
                proba = xgb_model.predict_proba(vector_input)[0]
            else:
                st.error("Selected model is not available")
                st.stop()
            
            # Display results
            st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
            
            # Prediction result
            if result == 1:
                st.markdown('<h2 class="warning">Spam Detected!</h2>', unsafe_allow_html=True)
            else:
                st.markdown('<h2 class="success">Not Spam</h2>', unsafe_allow_html=True)
            
            # Confidence meter
            confidence = proba[result] * 100
            st.markdown(f"""
            <div class="result-box">
                <div class="confidence-meter">
                    <div class="confidence-fill" style="width: {confidence}%"></div>
                </div>
                <p style="text-align: right;">{confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
