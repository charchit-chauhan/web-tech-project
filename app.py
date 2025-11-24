import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Language map: Add more if needed
lang_map = {
    "English": "en",
    "Hindi": "hi",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Chinese (Simplified)": "zh",
    "Japanese": "ja"
}

# Function to load model/tokenizer dynamically
@st.cache_resource
def load_model(src, tgt):
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Function to perform translation
def translate_text(text, tokenizer, model):
    inputs = tokenizer.encode(text, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated

# Streamlit UI
st.set_page_config(page_title="AI Language Translator", layout="centered")
st.title("🌐 AI-Powered Multi-Language Translator")
st.markdown("Translate text between multiple languages using AI.")

col1, col2 = st.columns(2)
with col1:
    src_lang = st.selectbox("Source Language", list(lang_map.keys()), index=0)
with col2:
    tgt_lang = st.selectbox("Target Language", list(lang_map.keys()), index=1)

text_input = st.text_area("Enter text to translate:", height=150)

if st.button("Translate"):
    if src_lang == tgt_lang:
        st.warning("Source and target languages must be different.")
    elif not text_input.strip():
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner("Translating..."):
            try:
                src_code = lang_map[src_lang]
                tgt_code = lang_map[tgt_lang]
                tokenizer, model = load_model(src_code, tgt_code)
                result = translate_text(text_input, tokenizer, model)
                st.success("Translation Complete:")
                st.text_area("Translated Text", result, height=150)
            except Exception as e:
                st.error(f"Error: {str(e)}")