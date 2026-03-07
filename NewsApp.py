import pickle
import streamlit as st
import re 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model and vectorizer
vector = pickle.load(open('CountVectorizer_nws.pkl','rb'))
model = pickle.load(open('Model-nws.pkl','rb'))

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Modern Minimal UI Styling
st.markdown("""
    <style>
        body {
            background: #eef2f3;
        }
        .main {
            background: #ffffff;
            padding: 2rem;
            border-radius: 18px;
            box-shadow: 0px 6px 25px rgba(0,0,0,0.12);
        }
        h1 {
            text-align: center;
            font-weight: 900;
            margin-bottom: 0px;
            background: linear-gradient(90deg, #1a2980, #26d0ce);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtext {
            text-align: center;
            color: #555;
            margin-top: -8px;
            margin-bottom: 25px;
            font-size: 18px;
        }
        .stTextInput>div>div>input {
            border: 2px solid #1a2980;
            border-radius: 12px;
            padding: 10px;
        }
        .stButton>button {
            background: linear-gradient(90deg, #1a2980, #26d0ce);
            color: white;
            border: none;
            padding: 0.7em 2em;
            border-radius: 12px;
            font-size: 1.1em;
            font-weight: 600;
            transition: all 0.25s ease;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        }
        .result-box {
            margin-top: 30px;
            padding: 22px;
            border-radius: 15px;
            font-size: 1.25em;
            font-weight: 700;
            text-align: center;
        }
        .fake {
            background-color: #ffe6e6;
            color: #c62828;
            border: 2px solid #c62828;
        }
        .real {
            background-color: #e7ffe7;
            color: #1b5e20;
            border: 2px solid #1b5e20;
        }
        .stTextArea textarea {
    min-height: 120px;
    max-height: 300px;
    border-radius: 12px;
    border: 2px solid #1a2980;
    padding: 12px;
    font-size: 16px;
}

    </style>
""", unsafe_allow_html=True)


# Title
st.markdown("<h1>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Instantly check whether a News headline is <b>FAKE</b> or <b>REAL</b>.</p>", unsafe_allow_html=True)


# Input field
input_sms = st.text_area(
    "✉️ Enter News Headline:",
    height=120,
    placeholder="Type or paste your news here...",
)



# Preprocessing Functions
def cleanText(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def remove_stop_words(sen):
    y = []
    for i in sen.split():
        if i.lower() not in stopwords.words('english'):
            y.append(i)
    return ' '.join(y)

ps = PorterStemmer()

def stemming(text):
    final = []
    for i in text.split():
        final.append(ps.stem(i))
    return ' '.join(final)


# Processing
cleaned = stemming(remove_stop_words(cleanText(input_sms)))
vector_inp = vector.transform([cleaned])
result = model.predict(vector_inp)[0]


# Button + Result
if st.button("🔍 Analyze News"):
    if result == 0:
        st.markdown("<div class='result-box fake'>🚫 This News is Classified as <b>FAKE</b>.</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-box real'>✅ This News is Classified as <b>REAL</b>.</div>",
                    unsafe_allow_html=True)


# Footer
st.markdown("<hr style='margin-top:40px; opacity:0.4;'>", unsafe_allow_html=True)
