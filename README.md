
# 🌐 AI Language Translator

A simple and powerful web app built with **Streamlit** and **Hugging Face Transformers** that allows users to translate text between multiple languages using pretrained MarianMT models.

## 🚀 Features

- Translate between languages like **English, Hindi, French, German, Spanish, Chinese**, and **Japanese**
- Fast and accurate translations using `Helsinki-NLP/opus-mt` models
- Simple UI built with Streamlit
- Dynamic model loading and smart caching for efficiency

---

## 🔤 Supported Languages

- English (`en`)
- Hindi (`hi`)
- French (`fr`)
- German (`de`)
- Spanish (`es`)
- Chinese (Simplified) (`zh`)
- Japanese (`ja`)

---

## 📦 Requirements

- Python 3.8+
- Streamlit
- Transformers
- Torch

Install using:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

- Clone the repo:
```bash
https://github.com/charchit-chauhan/web-tech-project.git
cd AI-Multi-Language-Translator
```

- Install dependencies:
```bash
pip install -r requirements.txt
```

- Launch the app:
```bash
streamlit run app.py
```

---
## ⚙️ How It Works
- Uses `MarianMT models` from **Hugging Face** for translation.
- Each language pair loads a specific model like Helsinki-NLP/opus-mt-en-fr.
- The app dynamically loads models and caches them for efficiency using `@st.cache_resource`.

---
## 📂 Project Structure

```bash
Ai-Multi-Language-Translator/
│
├── app.py               # Streamlit application
├── requirements.txt     # Required Python packages
└── README.md            # Project documentation
```

---
## 📄 License
MIT License. Free to use, modify, and distribute.

---
## 👨‍💻 Author
Charchit Chauhan
Astitva Tiwari
Akash Kumar


Made with ❤️ using OpenAI's GPT + HuggingFace + Streamlit
