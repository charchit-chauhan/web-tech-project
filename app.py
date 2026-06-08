import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import requests
import html
import time
import base64
import re
from datetime import datetime

# ════════════════════════════════════════════════
#  LANGUAGE MAP
# ════════════════════════════════════════════════
lang_map = {
    "English": "en", "Hindi": "hi", "French": "fr", "German": "de",
    "Spanish": "es", "Chinese (Simplified)": "zh", "Japanese": "ja",
    "Arabic": "ar", "Bengali": "bn", "Portuguese": "pt", "Russian": "ru",
    "Italian": "it", "Korean": "ko", "Turkish": "tr", "Dutch": "nl",
    "Polish": "pl", "Swedish": "sv", "Norwegian": "no", "Danish": "da",
    "Finnish": "fi", "Greek": "el", "Czech": "cs", "Romanian": "ro",
    "Hungarian": "hu", "Ukrainian": "uk", "Thai": "th", "Vietnamese": "vi",
    "Indonesian": "id", "Malay": "ms", "Persian": "fa", "Urdu": "ur",
    "Punjabi": "pa", "Gujarati": "gu", "Marathi": "mr", "Tamil": "ta",
    "Telugu": "te", "Kannada": "kn", "Malayalam": "ml", "Swahili": "sw",
    "Afrikaans": "af", "Hebrew": "iw", "Catalan": "ca", "Croatian": "hr",
    "Serbian": "sr", "Slovak": "sk", "Bulgarian": "bg", "Latvian": "lv",
    "Lithuanian": "lt", "Estonian": "et", "Slovenian": "sl", "Albanian": "sq",
    "Macedonian": "mk", "Tagalog (Filipino)": "tl", "Welsh": "cy",
    "Irish": "ga", "Latin": "la", "Esperanto": "eo", "Zulu": "zu",
    "Yoruba": "yo", "Somali": "so", "Nepali": "ne", "Sinhala": "si",
    "Khmer": "km", "Mongolian": "mn", "Kazakh": "kk", "Uzbek": "uz",
    "Armenian": "hy", "Georgian": "ka", "Haitian Creole": "ht",
    "Icelandic": "is", "Maltese": "mt",
}

LANG_CODES_REVERSE = {v: k for k, v in lang_map.items()}

MARIAN_SUPPORTED = {
    ("en","fr"),("en","de"),("en","es"),("en","zh"),("fr","en"),("de","en"),
    ("es","en"),("zh","en"),("en","hi"),("hi","en"),("en","ja"),("ja","en"),
    ("en","ru"),("ru","en"),("en","ar"),("ar","en"),("en","pt"),("pt","en"),
    ("en","it"),("it","en"),("en","nl"),("nl","en"),("en","pl"),("pl","en"),
    ("en","tr"),("tr","en"),("en","ko"),("ko","en"),
}

TTS_SUPPORTED_LANGS = {
    "en","hi","fr","de","es","zh","ja","ar","pt","ru","it","ko","tr","nl",
    "pl","sv","da","fi","el","cs","ro","hu","uk","th","vi","id","ms","fa",
    "bn","gu","mr","ta","te","kn","ml","sw","iw","ca","hr","sk","bg","lv",
    "lt","sl","af","tl","cy","no","sr","is","mk",
}

# ════════════════════════════════════════════════
#  DISAMBIGUATION DICTIONARY
#  Words known to be ambiguous with their senses
#  and context-hint keywords that disambiguate them
# ════════════════════════════════════════════════
AMBIGUOUS_WORDS = {
    "bat": {
        "senses": {
            "Animal (flying mammal)": {
                "hint": "The word 'bat' here refers to the flying mammal animal.",
                "keywords": ["flying","cave","nocturnal","mammal","wing","sonar","vampire","fruit bat"],
            },
            "Cricket/Baseball bat (sports equipment)": {
                "hint": "The word 'bat' here refers to a cricket or baseball bat (sports equipment).",
                "keywords": ["cricket","baseball","hit","swing","sport","play","wicket","run","innings","score"],
            },
            "Verb: to bat (hit/strike)": {
                "hint": "The word 'bat' here is used as a verb meaning to hit or strike.",
                "keywords": ["batting","batted","eyelid","blink"],
            },
        },
        "default_sense": "Animal (flying mammal)",
    },
    "bats": {
        "senses": {
            "Animals (flying mammals, plural)": {
                "hint": "The word 'bats' here refers to flying mammal animals (plural).",
                "keywords": ["flying","cave","nocturnal","mammal","wing","sonar","vampire","fruit","colony"],
            },
            "Cricket/Baseball bats (sports equipment, plural)": {
                "hint": "The word 'bats' here refers to cricket or baseball bats (sports equipment, plural).",
                "keywords": ["cricket","baseball","hit","swing","sport","play","willow","rubber"],
            },
            "Verb: bats (third-person singular)": {
                "hint": "The word 'bats' here is a verb (he/she bats).",
                "keywords": ["he bats","she bats","player bats","batting","average"],
            },
            "Adjective: crazy/insane (informal)": {
                "hint": "The word 'bats' here means crazy or insane (informal slang).",
                "keywords": ["crazy","mad","insane","nuts","mental"],
            },
        },
        "default_sense": "Animals (flying mammals, plural)",
    },
    "bank": {
        "senses": {
            "Financial institution": {
                "hint": "The word 'bank' here refers to a financial institution.",
                "keywords": ["money","account","loan","deposit","withdraw","interest","finance","savings"],
            },
            "River bank": {
                "hint": "The word 'bank' here refers to the bank/shore of a river or lake.",
                "keywords": ["river","lake","stream","shore","water","flood","fish","boat"],
            },
            "Verb: to bank (tilt/store)": {
                "hint": "The word 'bank' here is used as a verb.",
                "keywords": ["banked","banking","turn","aircraft","tilt","rely","count on"],
            },
        },
        "default_sense": "Financial institution",
    },
    "crane": {
        "senses": {
            "Bird (crane)": {
                "hint": "The word 'crane' here refers to a crane bird.",
                "keywords": ["bird","fly","migration","flock","nest","feather","beak","wetland"],
            },
            "Construction crane (machine)": {
                "hint": "The word 'crane' here refers to a construction crane machine.",
                "keywords": ["construction","lift","building","machine","operator","site","load","hoist"],
            },
            "Verb: to crane (stretch neck)": {
                "hint": "The word 'crane' here is a verb meaning to stretch one's neck.",
                "keywords": ["neck","head","look","peer","stretch","craning"],
            },
        },
        "default_sense": "Construction crane (machine)",
    },
    "spring": {
        "senses": {
            "Season (spring)": {
                "hint": "The word 'spring' here refers to the season.",
                "keywords": ["summer","winter","autumn","fall","season","flower","bloom","warm","april","march"],
            },
            "Coil spring (mechanical)": {
                "hint": "The word 'spring' here refers to a mechanical coil spring.",
                "keywords": ["coil","metal","bounce","compress","elastic","mattress","mechanical","steel"],
            },
            "Water spring": {
                "hint": "The word 'spring' here refers to a natural water spring.",
                "keywords": ["water","natural","source","well","mineral","hot spring","geyser"],
            },
            "Verb: to spring (jump)": {
                "hint": "The word 'spring' here is a verb meaning to jump or leap.",
                "keywords": ["jump","leap","spring up","sprung","sprang","pounce","surprise"],
            },
        },
        "default_sense": "Season (spring)",
    },
    "fly": {
        "senses": {
            "Insect (fly)": {
                "hint": "The word 'fly' here refers to the insect.",
                "keywords": ["insect","bug","swat","buzz","housefly","mosquito","pest","wings"],
            },
            "Verb: to fly (travel by air)": {
                "hint": "The word 'fly' here is a verb meaning to travel through the air.",
                "keywords": ["plane","airplane","aircraft","pilot","airport","flight","soar","bird","kite"],
            },
            "Trouser fly (zipper)": {
                "hint": "The word 'fly' here refers to the zipper flap on trousers.",
                "keywords": ["trouser","zipper","pants","button","clothing"],
            },
        },
        "default_sense": "Verb: to fly (travel by air)",
    },
    "light": {
        "senses": {
            "Illumination / light (noun)": {
                "hint": "The word 'light' here refers to illumination or brightness.",
                "keywords": ["sun","lamp","bright","dark","shine","glow","beam","torch","candle"],
            },
            "Light (adjective: not heavy)": {
                "hint": "The word 'light' here means not heavy (adjective).",
                "keywords": ["heavy","weight","carry","lift","feather","kg","gram","pound"],
            },
            "Light (adjective: pale color)": {
                "hint": "The word 'light' here refers to a pale/soft color.",
                "keywords": ["color","colour","shade","pale","blue","green","pink","tone"],
            },
            "Verb: to light (ignite)": {
                "hint": "The word 'light' here is a verb meaning to ignite.",
                "keywords": ["fire","candle","match","ignite","flame","burn","kindle"],
            },
        },
        "default_sense": "Illumination / light (noun)",
    },
    "match": {
        "senses": {
            "Sports match (game)": {
                "hint": "The word 'match' here refers to a sports game or competition.",
                "keywords": ["cricket","football","game","play","score","team","win","lose","tournament","final"],
            },
            "Fire match (matchstick)": {
                "hint": "The word 'match' here refers to a matchstick used to light fire.",
                "keywords": ["fire","light","flame","burn","candle","strike","ignite","box"],
            },
            "Verb: to match (correspond/pair)": {
                "hint": "The word 'match' here is a verb meaning to correspond or pair.",
                "keywords": ["pair","suit","fit","correspond","colour","color","pattern","identical"],
            },
        },
        "default_sense": "Verb: to match (correspond/pair)",
    },
    "pool": {
        "senses": {
            "Swimming pool": {
                "hint": "The word 'pool' here refers to a swimming pool.",
                "keywords": ["swim","swimming","water","dive","chlorine","lap","indoor","outdoor","olympic"],
            },
            "Pool (game/billiards)": {
                "hint": "The word 'pool' here refers to the billiards/pool game.",
                "keywords": ["billiard","cue","ball","table","snooker","eight ball","pocket","game"],
            },
            "Resource pool / carpool": {
                "hint": "The word 'pool' here refers to a shared resource or carpool.",
                "keywords": ["car","share","resource","fund","gene pool","talent","worker","team"],
            },
        },
        "default_sense": "Swimming pool",
    },
    "row": {
        "senses": {
            "A row (line/sequence)": {
                "hint": "The word 'row' here refers to a line or sequence of things.",
                "keywords": ["line","seat","column","table","seat","sequence","order","queue"],
            },
            "To row (a boat)": {
                "hint": "The word 'row' here is a verb meaning to propel a boat with oars.",
                "keywords": ["boat","oar","paddle","river","lake","canoe","rowing"],
            },
            "A row (argument/fight, British English)": {
                "hint": "The word 'row' here refers to a noisy argument or fight.",
                "keywords": ["argument","fight","quarrel","dispute","noise","loud","trouble"],
            },
        },
        "default_sense": "A row (line/sequence)",
    },
}

# Context keywords that strongly imply animal meaning for "bat/bats"
BAT_ANIMAL_CONTEXT = {"fly","flying","cave","nocturnal","wing","sonar","echolocation","colony","roost","mammal","insect","night"}
BAT_SPORT_CONTEXT  = {"cricket","baseball","hit","swing","wicket","innings","over","score","bowler","pitch","willow"}

# ════════════════════════════════════════════════
#  ORIGINAL FUNCTIONS (unchanged)
# ════════════════════════════════════════════════
@st.cache_resource
def load_model(src, tgt):
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_text(text, tokenizer, model):
    inputs = tokenizer.encode(text, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated

# ════════════════════════════════════════════════
#  CORE TRANSLATION FUNCTIONS
# ════════════════════════════════════════════════
def google_translate_fallback(text: str, src: str, tgt: str) -> str:
    url = "https://translate.googleapis.com/translate_a/single"
    params = {"client": "gtx", "sl": src, "tl": tgt, "dt": "t", "q": text}
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return html.unescape("".join(part[0] for part in data[0] if part[0]))

def detect_language(text: str) -> str:
    url = "https://translate.googleapis.com/translate_a/single"
    params = {"client": "gtx", "sl": "auto", "tl": "en", "dt": "t", "q": text}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()
        detected_code = data[2] if len(data) > 2 else None
        return LANG_CODES_REVERSE.get(detected_code, detected_code or "Unknown")
    except Exception:
        return "Unknown"

def smart_translate(text: str, src_code: str, tgt_code: str):
    """
    Routes to MarianMT or Google. 
    Always uses Google when src is 'auto' or pair not in MARIAN_SUPPORTED.
    Google has much better word-sense disambiguation for ambiguous words.
    """
    if src_code != "auto" and (src_code, tgt_code) in MARIAN_SUPPORTED:
        try:
            tokenizer, model = load_model(src_code, tgt_code)
            return translate_text(text, tokenizer, model), "MarianMT (Local AI)"
        except Exception:
            pass
    result = google_translate_fallback(text, src_code, tgt_code)
    return result, "Google Translate"

# ════════════════════════════════════════════════
#  DISAMBIGUATION ENGINE
# ════════════════════════════════════════════════
def find_ambiguous_words(text: str) -> list[str]:
    """Return list of ambiguous words found in the text (lowercase)."""
    words_in_text = set(re.findall(r'\b\w+\b', text.lower()))
    return [w for w in AMBIGUOUS_WORDS if w in words_in_text]

def auto_detect_sense(word: str, text: str) -> str | None:
    """
    Try to auto-detect which sense of an ambiguous word is meant
    using context keywords. Returns sense name or None if unclear.
    """
    entry = AMBIGUOUS_WORDS.get(word.lower())
    if not entry:
        return None
    text_lower = text.lower()
    text_words = set(re.findall(r'\b\w+\b', text_lower))
    best_sense = None
    best_score = 0
    for sense_name, sense_data in entry["senses"].items():
        score = sum(1 for kw in sense_data["keywords"] if kw in text_lower)
        if score > best_score:
            best_score = score
            best_sense = sense_name
    # Only auto-pick if at least 1 keyword matched
    return best_sense if best_score >= 1 else None

def build_disambiguated_text(original: str, word_sense_map: dict) -> str:
    """
    Prepend context hints for each disambiguated word so the translator
    understands the correct meaning. Hints are stripped after translation.
    """
    if not word_sense_map:
        return original
    hint_parts = []
    for word, (sense_name, hint) in word_sense_map.items():
        hint_parts.append(f"[Context: {hint}]")
    # Put hints before the text so translator uses them as context
    return " ".join(hint_parts) + " " + original

def strip_context_hints(text: str) -> str:
    """Remove [Context: ...] prefixes that might bleed into the output."""
    return re.sub(r'\[Context:[^\]]*\]\s*', '', text).strip()

# ════════════════════════════════════════════════
#  OTHER FEATURE FUNCTIONS
# ════════════════════════════════════════════════
def get_tts_audio_b64(text: str, lang_code: str) -> str | None:
    tts_lang = lang_code if lang_code in TTS_SUPPORTED_LANGS else "en"
    url = "https://translate.google.com/translate_tts"
    params = {"ie": "UTF-8", "q": text[:200], "tl": tts_lang, "client": "gtx"}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code == 200 and resp.content:
            return base64.b64encode(resp.content).decode()
    except Exception:
        pass
    return None

def show_tts_player(text: str, lang_code: str, label: str = "🔊 Listen"):
    b64 = get_tts_audio_b64(text, lang_code)
    if b64:
        audio_html = f"""
        <audio controls style="width:100%;margin-top:6px">
          <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>"""
        st.markdown(f"**{label}**")
        st.markdown(audio_html, unsafe_allow_html=True)
    else:
        st.caption("⚠️ TTS not available for this language.")

def get_romanization(text: str, src_code: str) -> str | None:
    url = "https://translate.googleapis.com/translate_a/single"
    params = {"client": "gtx", "sl": src_code, "tl": "en", "dt": ["t", "rm"], "q": text}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()
        roman_parts = []
        for segment in data[0]:
            if segment and len(segment) > 3 and segment[3]:
                roman_parts.append(segment[3])
        return " ".join(roman_parts) if roman_parts else None
    except Exception:
        return None

FORMALITY_PROMPTS = {
    "Formal":   "In a formal professional tone: ",
    "Casual":   "In a casual friendly tone: ",
    "Neutral":  "",
}

def apply_formality(text: str, level: str) -> str:
    prefix = FORMALITY_PROMPTS.get(level, "")
    return prefix + text if prefix else text

def ocr_image(image_bytes: bytes, mime: str = "image/png") -> str:
    b64_img = base64.b64encode(image_bytes).decode()
    payload = {
        "base64Image": f"data:{mime};base64,{b64_img}",
        "language": "eng",
        "isOverlayRequired": False,
        "OCREngine": 2,
    }
    try:
        resp = requests.post(
            "https://api.ocr.space/parse/image",
            data=payload,
            headers={"apikey": "helloworld"},
            timeout=20,
        )
        result = resp.json()
        parsed = result.get("ParsedResults", [])
        if parsed:
            return parsed[0].get("ParsedText", "").strip()
        return ""
    except Exception as e:
        return f"OCR error: {e}"

def translate_to_multiple(text: str, src_code: str, targets: list) -> dict:
    results = {}
    for lang_name in targets:
        tgt_code = lang_map[lang_name]
        try:
            t, engine = smart_translate(text, src_code, tgt_code)
            results[lang_name] = {"text": t, "engine": engine, "code": tgt_code}
        except Exception as e:
            results[lang_name] = {"text": f"Error: {e}", "engine": "—", "code": tgt_code}
        time.sleep(0.15)
    return results

# NEW: Back-translation quality check
def back_translate(text: str, tgt_code: str, src_code: str = "en") -> str:
    """Translate back to source language to verify translation quality."""
    try:
        result, _ = smart_translate(text, tgt_code, src_code)
        return result
    except Exception:
        return "Back-translation failed."

# NEW: Sentence-level quality score (simple Levenshtein similarity)
def similarity_score(a: str, b: str) -> int:
    a, b = a.lower().strip(), b.lower().strip()
    if not a or not b:
        return 0
    a_words = set(a.split())
    b_words = set(b.split())
    if not a_words:
        return 0
    overlap = len(a_words & b_words) / len(a_words)
    return int(overlap * 100)

# ════════════════════════════════════════════════
#  STREAMLIT UI
# ════════════════════════════════════════════════
st.set_page_config(page_title="AI Language Translator", layout="wide", page_icon="🌐")

# ── THEME ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,700;0,900;1,300;1,700&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');

:root {
  --bg: #080b12; --surface: #0f1420; --surface2: #161d2e; --surface3: #1e2740;
  --border: #1f2d4a; --border2: #2a3d60;
  --teal: #00c9a7; --teal-dim: #00896f; --teal-glow: rgba(0,201,167,0.15);
  --coral: #ff6b6b; --gold: #ffd166; --sky: #74c0fc;
  --text: #e2e8f8; --text2: #a0aec8; --muted: #4a5878;
  --success: #00c9a7; --warn: #ffd166; --danger: #ff6b6b;
  --r: 12px; --r-sm: 8px; --r-lg: 18px;
}

* { box-sizing: border-box; }
html, body, [class*="css"], [class*="st-"] {
  font-family: 'Outfit', sans-serif !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}
.main, section.main { background: var(--bg) !important; }
.block-container { padding: 0 2rem 4rem !important; max-width: 1200px !important; }

/* ── HERO ── */
.hero-wrap {
  background: linear-gradient(160deg, #0f1d35 0%, #080b12 55%);
  border-bottom: 1px solid var(--border);
  padding: 2.8rem 3rem 2.2rem;
  position: relative; overflow: hidden;
  margin: 0 -2rem 2.5rem; /* bleed to edges */
}
.hero-wrap::before {
  content: '';
  position: absolute; top: -60px; right: -60px;
  width: 320px; height: 320px;
  background: radial-gradient(circle, rgba(0,201,167,0.12) 0%, transparent 65%);
  border-radius: 50%; pointer-events: none;
}
.hero-wrap::after {
  content: '';
  position: absolute; bottom: -80px; left: 10%;
  width: 260px; height: 260px;
  background: radial-gradient(circle, rgba(116,192,252,0.07) 0%, transparent 65%);
  border-radius: 50%; pointer-events: none;
}
.hero-inner { position: relative; z-index: 1; }
.hero-eyebrow {
  font-family: 'DM Mono', monospace;
  font-size: 0.7rem; color: var(--teal);
  letter-spacing: 0.22em; text-transform: uppercase;
  margin-bottom: 0.6rem;
}
.hero-title {
  font-family: 'Fraunces', serif;
  font-size: 3.6rem; font-weight: 900;
  line-height: 0.95; letter-spacing: -0.03em;
  color: var(--text); margin-bottom: 0.7rem;
}
.hero-title em {
  font-style: italic; font-weight: 300;
  background: linear-gradient(120deg, var(--teal), var(--sky));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub {
  font-size: 0.9rem; font-weight: 300;
  color: var(--text2); letter-spacing: 0.01em;
  max-width: 520px; line-height: 1.6;
}
.hero-chips {
  display: flex; flex-wrap: wrap; gap: 0.45rem;
  margin-top: 1.4rem;
}
.hero-chip {
  background: var(--surface2);
  border: 1px solid var(--border2);
  border-radius: 30px;
  padding: 0.22rem 0.8rem;
  font-size: 0.7rem; font-weight: 500;
  color: var(--text2); letter-spacing: 0.04em;
  font-family: 'DM Mono', monospace;
}
.hero-chip.active { border-color: var(--teal); color: var(--teal); }

/* ── TRANSLATE CARD ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 1.6rem 1.8rem;
  margin-bottom: 1rem;
  position: relative;
}
.card-accent {
  border-top: 2px solid var(--teal);
}
.card-label {
  font-family: 'DM Mono', monospace;
  font-size: 0.65rem; font-weight: 500;
  letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 0.7rem;
}
.card-result {
  background: var(--surface2);
  border: 1px solid var(--border2);
  border-radius: var(--r);
  padding: 1.2rem 1.4rem;
  font-size: 1.05rem; font-weight: 300;
  line-height: 1.75; color: var(--text);
  min-height: 80px;
  border-left: 3px solid var(--teal);
}
.engine-pill {
  display: inline-flex; align-items: center; gap: 0.4rem;
  background: var(--surface3);
  border: 1px solid var(--border2);
  border-radius: 20px;
  padding: 0.18rem 0.7rem;
  font-family: 'DM Mono', monospace;
  font-size: 0.62rem; color: var(--teal);
  letter-spacing: 0.06em; margin-top: 0.6rem;
}
.engine-pill::before { content: '●'; font-size: 0.4rem; }

/* ── LANG SELECTOR PANEL ── */
.lang-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 1.4rem 1.6rem 1rem;
  margin-bottom: 1rem;
}
.lang-arrow {
  display: flex; align-items: center; justify-content: center;
  height: 100%;
  font-size: 1.3rem; color: var(--border2);
  padding-top: 1.5rem;
}

/* ── TABS ── */
div[data-testid="stTabs"] {
  border-bottom: 1px solid var(--border) !important;
  margin-bottom: 1.5rem !important;
}
div[data-testid="stTabs"] button {
  font-family: 'Outfit', sans-serif !important;
  font-size: 0.8rem !important; font-weight: 500 !important;
  letter-spacing: 0.05em !important; color: var(--muted) !important;
  border: none !important; border-bottom: 2px solid transparent !important;
  border-radius: 0 !important; padding: 0.6rem 1.2rem !important;
  background: transparent !important; transition: all 0.18s !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--teal) !important;
  border-bottom: 2px solid var(--teal) !important;
}
div[data-testid="stTabs"] button:hover { color: var(--text) !important; }

/* ── SELECTBOXES ── */
div[data-testid="stSelectbox"] > div > div {
  background: var(--surface2) !important; border: 1px solid var(--border2) !important;
  border-radius: var(--r-sm) !important; color: var(--text) !important;
  font-family: 'Outfit', sans-serif !important; font-size: 0.88rem !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 3px var(--teal-glow) !important;
}
div[data-testid="stSelectbox"] label {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.65rem !important; font-weight: 500 !important;
  letter-spacing: 0.14em !important; text-transform: uppercase !important;
  color: var(--muted) !important;
}

/* ── TEXT AREAS ── */
div[data-testid="stTextArea"] textarea {
  background: var(--surface2) !important; border: 1px solid var(--border2) !important;
  border-radius: var(--r-sm) !important; color: var(--text) !important;
  font-family: 'Outfit', sans-serif !important; font-size: 0.97rem !important;
  font-weight: 300 !important; line-height: 1.7 !important;
  caret-color: var(--teal) !important; transition: all 0.18s !important;
  resize: vertical !important;
}
div[data-testid="stTextArea"] textarea:focus {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 3px var(--teal-glow) !important; outline: none !important;
}
div[data-testid="stTextArea"] label {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.65rem !important; font-weight: 500 !important;
  letter-spacing: 0.14em !important; text-transform: uppercase !important;
  color: var(--muted) !important;
}

/* ── PRIMARY BUTTON ── */
div[data-testid="stButton"] > button {
  background: var(--teal) !important; color: #050810 !important;
  border: none !important; border-radius: var(--r-sm) !important;
  font-family: 'Outfit', sans-serif !important; font-weight: 600 !important;
  font-size: 0.88rem !important; letter-spacing: 0.04em !important;
  padding: 0.6rem 1.6rem !important;
  transition: all 0.16s !important;
  box-shadow: 0 0 20px rgba(0,201,167,0.3) !important;
}
div[data-testid="stButton"] > button:hover {
  background: #00e0ba !important;
  box-shadow: 0 0 30px rgba(0,201,167,0.45) !important;
  transform: translateY(-1px) !important;
}
div[data-testid="stButton"] > button:active { transform: translateY(0) !important; }

/* ── DOWNLOAD BUTTON ── */
div[data-testid="stDownloadButton"] > button {
  background: transparent !important; color: var(--text2) !important;
  border: 1px solid var(--border2) !important; border-radius: var(--r-sm) !important;
  font-family: 'Outfit', sans-serif !important; font-weight: 400 !important;
  font-size: 0.82rem !important; box-shadow: none !important;
  padding: 0.45rem 1rem !important;
}
div[data-testid="stDownloadButton"] > button:hover {
  border-color: var(--teal) !important; color: var(--teal) !important;
  transform: translateY(-1px) !important; opacity: 1 !important;
}

/* ── CHECKBOX ── */
div[data-testid="stCheckbox"] label { font-size: 0.85rem !important; color: var(--text2) !important; }
div[data-testid="stCheckbox"] span[data-testid="stCheckboxChecked"] {
  background: var(--teal) !important; border-color: var(--teal) !important;
}

/* ── SELECT SLIDER ── */
div[data-testid="stSelectSlider"] div[role="slider"] {
  background: var(--teal) !important; border-color: var(--teal) !important;
}
div[data-testid="stSelectSlider"] .st-bq { background: var(--teal) !important; }

/* ── RADIO ── */
div[data-testid="stRadio"] label { font-size: 0.83rem !important; color: var(--text2) !important; }

/* ── ALERTS ── */
div[data-testid="stAlert"] {
  border-radius: var(--r-sm) !important; font-size: 0.84rem !important;
  border: 1px solid var(--border) !important;
}

/* ── EXPANDERS ── */
div[data-testid="stExpander"] {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r) !important; margin-bottom: 0.5rem !important;
}
div[data-testid="stExpander"] summary {
  font-size: 0.84rem !important; font-weight: 500 !important;
  color: var(--text2) !important; padding: 0.75rem 1.1rem !important;
}
div[data-testid="stExpander"] summary:hover { color: var(--teal) !important; }

/* ── METRICS ── */
div[data-testid="stMetric"] {
  background: var(--surface2) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r) !important; padding: 1rem 1.2rem !important;
}
div[data-testid="stMetric"] label {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.62rem !important; letter-spacing: 0.12em !important;
  text-transform: uppercase !important; color: var(--muted) !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
  font-family: 'Fraunces', serif !important;
  font-size: 2rem !important; color: var(--teal) !important;
}

/* ── PROGRESS ── */
div[data-testid="stProgressBar"] > div > div {
  background: linear-gradient(90deg, var(--teal-dim), var(--teal)) !important;
  border-radius: 4px !important;
}
div[data-testid="stProgressBar"] > div {
  background: var(--surface2) !important; border-radius: 4px !important; height: 4px !important;
}

/* ── FILE UPLOADER ── */
div[data-testid="stFileUploader"] {
  background: var(--surface2) !important;
  border: 2px dashed var(--border2) !important;
  border-radius: var(--r) !important; padding: 1.2rem !important;
  transition: border-color 0.18s !important;
}
div[data-testid="stFileUploader"]:hover { border-color: var(--teal) !important; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── MULTISELECT ── */
span[data-baseweb="tag"] {
  background: rgba(0,201,167,0.12) !important;
  border: 1px solid rgba(0,201,167,0.28) !important; border-radius: 4px !important;
}
span[data-baseweb="tag"] span { color: var(--teal) !important; font-size: 0.78rem !important; }

/* ── CAPTION ── */
div[data-testid="stCaptionContainer"] p, small {
  color: var(--muted) !important; font-size: 0.74rem !important;
}

/* ── SPINNER ── */
div[data-testid="stSpinner"] div { border-top-color: var(--teal) !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--teal-dim); }

/* ── HR ── */
hr { border-color: var(--border) !important; opacity: 0.5 !important; }

/* ── H TAGS ── */
h1, h2, h3 { font-family: 'Fraunces', serif !important; color: var(--text) !important; }
h3 { font-size: 1.05rem !important; font-weight: 700 !important; margin-bottom: 0.75rem !important; }

/* ── DISAMBIGUATION PANEL ── */
.disambig-panel {
  background: rgba(0,201,167,0.04);
  border: 1px solid rgba(0,201,167,0.2);
  border-left: 3px solid var(--teal);
  border-radius: var(--r); padding: 1rem 1.2rem; margin: 0.8rem 0;
}
/* ── QUALITY SCORE BAR ── */
.score-bar-wrap { margin-top: 0.5rem; }
.score-bar-bg {
  background: var(--surface3); border-radius: 4px;
  height: 6px; overflow: hidden; margin: 0.4rem 0;
}
.score-bar-fill {
  height: 100%; border-radius: 4px;
  transition: width 0.6s cubic-bezier(.4,0,.2,1);
}

/* ── HISTORY CARD ── */
.hist-card {
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: var(--r-sm); padding: 0.75rem 0.9rem; margin-bottom: 0.5rem;
}
.hist-langs {
  font-family: 'DM Mono', monospace; font-size: 0.62rem;
  color: var(--teal); letter-spacing: 0.08em; margin-bottom: 0.3rem;
}
.hist-orig { font-size: 0.82rem; color: var(--text2); margin-bottom: 0.15rem; }
.hist-trans { font-size: 0.82rem; color: var(--muted); font-style: italic; }

/* ── SENSE RADIO STYLED ── */
div[data-testid="stRadio"] > div { gap: 0.3rem !important; }

/* ── STAT GRID ── */
.stat-num {
  font-family: 'Fraunces', serif; font-size: 2.2rem;
  color: var(--teal); line-height: 1;
}
.stat-lbl {
  font-family: 'DM Mono', monospace; font-size: 0.62rem;
  color: var(--muted); text-transform: uppercase; letter-spacing: 0.12em;
  margin-top: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-inner">
    <div class="hero-eyebrow">// AI-Powered Language Translation</div>
    <div class="hero-title">AI-Powered Language Translator</em></div>
    <div class="hero-sub">
      Translate across 100+ languages with context-aware disambiguation,
      text-to-speech, OCR, and smart quality checking.
    </div>
    <div class="hero-chips">
      <div class="hero-chip active">MarianMT Local</div>
      <div class="hero-chip active">Google Translate</div>
      <div class="hero-chip active">Word-Sense Engine</div>
      <div class="hero-chip">TTS Audio</div>
      <div class="hero-chip">OCR Vision</div>
      <div class="hero-chip">Back-Translation QA</div>
      <div class="hero-chip">Romanization</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Session state
for key, default in [
    ("history", []),
    ("favorites", []),
    ("last_result", ""),
    ("last_src_code", "en"),
    ("last_tgt_code", "hi"),
    ("last_input", ""),
    ("ocr_text", ""),
    ("word_sense_overrides", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 0.8rem; border-bottom:1px solid var(--border); margin-bottom:1rem;">
      <div style="font-family:'Fraunces',serif;font-size:1.4rem;font-weight:900;color:var(--text);">
        AI-Powered Language Translator<span style="color:var(--teal);font-style:italic;"></span>
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:var(--muted);letter-spacing:0.15em;text-transform:uppercase;margin-top:0.2rem;">
        Translation Studio
      </div>
    </div>
    """, unsafe_allow_html=True)

    sidebar_tab = st.radio("", ["🕘 History", "⭐ Favorites", "📊 Stats"], horizontal=True, label_visibility="collapsed")

    if sidebar_tab == "🕘 History":
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:var(--muted);letter-spacing:0.12em;text-transform:uppercase;margin:0.8rem 0 0.6rem;">Recent Translations</div>', unsafe_allow_html=True)
        if st.session_state.history:
            if st.button("Clear All", use_container_width=True):
                st.session_state.history = []
                st.rerun()
            for item in reversed(st.session_state.history[-8:]):
                disambig_mark = " ·✦" if item.get("disambiguated") else ""
                st.markdown(f"""
<div class="hist-card">
  <div class="hist-langs">{item['src']} → {item['tgt']} · {item['time']}{disambig_mark}</div>
  <div class="hist-orig">{item['original'][:55]}{'…' if len(item['original'])>55 else ''}</div>
  <div class="hist-trans">{item['translated'][:55]}{'…' if len(item['translated'])>55 else ''}</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:var(--muted);font-size:0.82rem;padding:1rem 0;">No translations yet.</div>', unsafe_allow_html=True)

    elif sidebar_tab == "⭐ Favorites":
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:var(--muted);letter-spacing:0.12em;text-transform:uppercase;margin:0.8rem 0 0.6rem;">Saved Favorites</div>', unsafe_allow_html=True)
        if st.session_state.favorites:
            if st.button("Clear All", use_container_width=True, key="clr_fav"):
                st.session_state.favorites = []
                st.rerun()
            for fav in reversed(st.session_state.favorites):
                st.markdown(f"""
<div class="hist-card">
  <div class="hist-langs">{fav['src']} → {fav['tgt']} · {fav.get('time','')}</div>
  <div class="hist-orig">{fav['original'][:55]}{'…' if len(fav['original'])>55 else ''}</div>
  <div class="hist-trans">{fav['translated'][:55]}{'…' if len(fav['translated'])>55 else ''}</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:var(--muted);font-size:0.82rem;padding:1rem 0;">Click ⭐ after translating to save.</div>', unsafe_allow_html=True)

    elif sidebar_tab == "📊 Stats":
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:var(--muted);letter-spacing:0.12em;text-transform:uppercase;margin:0.8rem 0 1rem;">Session Analytics</div>', unsafe_allow_html=True)
        total = len(st.session_state.history)
        favs  = len(st.session_state.favorites)
        disambig_count = sum(1 for h in st.session_state.history if h.get("disambiguated"))
        st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-bottom:0.6rem;">
  <div class="hist-card" style="text-align:center">
    <div class="stat-num">{total}</div>
    <div class="stat-lbl">Translated</div>
  </div>
  <div class="hist-card" style="text-align:center">
    <div class="stat-num">{favs}</div>
    <div class="stat-lbl">Favorites</div>
  </div>
</div>
<div class="hist-card" style="text-align:center;margin-bottom:0.6rem">
  <div class="stat-num">{disambig_count}</div>
  <div class="stat-lbl">Disambiguated</div>
</div>""", unsafe_allow_html=True)
        if st.session_state.history:
            pairs = [f"{h['src']}→{h['tgt']}" for h in st.session_state.history]
            top_pair = max(set(pairs), key=pairs.count)
            engines = [h.get("engine","?") for h in st.session_state.history]
            local_pct = round(engines.count("MarianMT (Local AI)") / len(engines) * 100)
            st.markdown(f"""
<div class="hist-card" style="margin-bottom:0.5rem">
  <div class="hist-langs">Top Pair</div>
  <div style="font-size:0.85rem;color:var(--text);margin-top:0.2rem">{top_pair}</div>
</div>
<div class="hist-card">
  <div class="hist-langs">Local AI Usage</div>
  <div style="font-size:0.85rem;color:var(--teal);margin-top:0.2rem">{local_pct}%</div>
  <div class="score-bar-bg"><div class="score-bar-fill" style="width:{local_pct}%;background:var(--teal)"></div></div>
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════
#  MAIN TABS
# ════════════════════════════════════════════════
tab_main, tab_compare, tab_ocr, tab_disambig = st.tabs([
    "✏️  Translate", "🌍  Compare", "📷  OCR Vision", "🔍  Word Sense Guide"
])

# ────────────────────────────────────────────────
#  TAB 1 — MAIN TRANSLATE
# ────────────────────────────────────────────────
with tab_main:

    # ── Language selector card ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Language Pair</div>', unsafe_allow_html=True)

    col1, col_arr, col2, col_swap = st.columns([5, 0.6, 5, 1.5])
    with col1:
        src_lang = st.selectbox("From", list(lang_map.keys()), index=0, label_visibility="visible")
    with col_arr:
        st.markdown('<div class="lang-arrow">→</div>', unsafe_allow_html=True)
    with col2:
        tgt_lang = st.selectbox("To", list(lang_map.keys()), index=1, label_visibility="visible")
    with col_swap:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⇄ Swap", use_container_width=True):
            st.session_state["_swap"] = (tgt_lang, src_lang)
            st.rerun()

    opt1, opt2, opt3 = st.columns(3)
    with opt1:
        auto_detect = st.checkbox("🔍 Auto-detect language", value=False)
    with opt2:
        batch_mode_here = st.checkbox("📦 Batch mode (line-by-line)", value=False)
    with opt3:
        formality = st.select_slider("Tone", options=["Casual", "Neutral", "Formal"], value="Neutral")

    st.markdown('</div>', unsafe_allow_html=True)  # close lang card

    # ── Input card ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Input Text</div>', unsafe_allow_html=True)
    text_input = st.text_area("", height=160, placeholder="Type or paste text to translate… supports any script, emoji, mixed content.", label_visibility="collapsed")

    if text_input:
        wc = len(text_input.split())
        cc = len(text_input)
        sents = max(1, text_input.count('.') + text_input.count('!') + text_input.count('?'))
        st.markdown(f"""
<div style="display:flex;gap:1.4rem;margin-top:0.2rem;">
  <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:var(--muted);">{cc} chars</span>
  <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:var(--muted);">{wc} words</span>
  <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:var(--muted);">{sents} sent.</span>
</div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close input card

    batch_mode = batch_mode_here

    # ── Disambiguation panel ──
    ambig_words = find_ambiguous_words(text_input) if text_input.strip() else []
    word_sense_map = {}

    if ambig_words:
        st.markdown('<div class="disambig-panel">', unsafe_allow_html=True)
        st.markdown(f"""
<div style="display:flex;align-items:flex-start;gap:0.7rem;margin-bottom:0.8rem;">
  <div style="font-size:1.2rem;margin-top:0.05rem;">⚠️</div>
  <div>
    <div style="font-weight:600;font-size:0.9rem;color:var(--warn);margin-bottom:0.15rem;">
      Ambiguous word{'' if len(ambig_words)==1 else 's'} detected
    </div>
    <div style="font-size:0.8rem;color:var(--text2);">
      Confirm what <strong style="color:var(--text)">{', '.join(f'&ldquo;{w}&rdquo;' for w in ambig_words)}</strong> means in your sentence for an accurate translation.
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        for word in ambig_words:
            entry = AMBIGUOUS_WORDS[word]
            sense_names = list(entry["senses"].keys())
            auto_sense = auto_detect_sense(word, text_input)
            default_idx = sense_names.index(auto_sense) if auto_sense in sense_names else \
                          sense_names.index(entry["default_sense"]) if entry["default_sense"] in sense_names else 0

            if auto_sense:
                st.markdown(f'<div style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:var(--teal);margin-bottom:0.3rem;">🤖 Auto-detected: {auto_sense}</div>', unsafe_allow_html=True)

            st.markdown(f'<div style="font-size:0.83rem;font-weight:600;color:var(--text);margin-bottom:0.3rem;">What does <code style="background:var(--surface3);padding:0.1rem 0.4rem;border-radius:4px;color:var(--teal)">{word}</code> mean here?</div>', unsafe_allow_html=True)
            chosen_sense = st.radio("", options=sense_names, index=default_idx, key=f"sense_{word}", horizontal=False, label_visibility="collapsed")
            hint = entry["senses"][chosen_sense]["hint"]
            word_sense_map[word] = (chosen_sense, hint)
            st.markdown(f'<div style="font-size:0.73rem;color:var(--muted);margin:0.2rem 0 0.6rem;font-style:italic;">→ {hint}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Translate button ──
    btn_col, _ = st.columns([2, 5])
    with btn_col:
        translate_clicked = st.button("🚀  Translate", use_container_width=True)

    if translate_clicked:
        src_code = lang_map[src_lang]
        tgt_code = lang_map[tgt_lang]

        if not auto_detect and src_lang == tgt_lang:
            st.warning("Source and target languages must be different.")
        elif not text_input.strip():
            st.warning("Please enter some text to translate.")
        else:
            with st.spinner("Translating..."):
                try:
                    detected_name = None
                    if auto_detect:
                        detected_name = detect_language(text_input)
                        st.info(f"🔍 Detected language: **{detected_name}**")
                        src_code = "auto"

                    # Apply formality
                    text_to_translate = apply_formality(text_input, formality)

                    # ── Apply disambiguation hints ──
                    if word_sense_map:
                        text_to_translate = build_disambiguated_text(text_to_translate, word_sense_map)
                        # Always use Google when disambiguation context is injected
                        # (Google handles context hints in the prompt much better than MarianMT)
                        result = google_translate_fallback(text_to_translate, src_code if src_code != "auto" else "en", tgt_code)
                        result = strip_context_hints(result)
                        engine_used = "Google Translate (disambiguated)"
                    elif batch_mode:
                        lines = [l for l in text_input.split("\n") if l.strip()]
                        translated_lines = []
                        progress = st.progress(0)
                        for i, line in enumerate(lines):
                            t, _ = smart_translate(apply_formality(line, formality), src_code, tgt_code)
                            translated_lines.append(t)
                            progress.progress((i + 1) / len(lines))
                            time.sleep(0.1)
                        result = "\n".join(translated_lines)
                        engine_used = "Batch / Smart Router"
                    else:
                        result, engine_used = smart_translate(text_to_translate, src_code, tgt_code)

                    st.session_state.last_result = result
                    st.session_state.last_src_code = src_code
                    st.session_state.last_tgt_code = tgt_code
                    st.session_state.last_input = text_input

                    st.session_state.history.append({
                        "src": src_lang if not auto_detect else (detected_name or "Auto"),
                        "tgt": tgt_lang,
                        "original": text_input,
                        "translated": result,
                        "time": datetime.now().strftime("%H:%M"),
                        "engine": engine_used,
                        "formality": formality,
                        "disambiguated": bool(word_sense_map),
                    })

                    # ── Result card ──
                    st.markdown('<div class="card card-accent" style="margin-top:1.2rem;">', unsafe_allow_html=True)
                    st.markdown(f'<div class="card-label">{tgt_lang} — Translation</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="card-result">{html.escape(result)}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="engine-pill">{engine_used} · {formality}</div>', unsafe_allow_html=True)

                    # Disambiguation badges
                    if word_sense_map:
                        badges = " ".join(
                            f'<span style="display:inline-block;background:rgba(0,201,167,0.1);border:1px solid rgba(0,201,167,0.3);border-radius:20px;padding:0.15rem 0.6rem;font-size:0.68rem;color:var(--teal);margin:0.4rem 0.2rem 0 0;">✦ {w}: {sn.split("(")[0].strip()}</span>'
                            for w, (sn, _) in word_sense_map.items()
                        )
                        st.markdown(f'<div style="margin-top:0.5rem;">{badges}</div>', unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                    # ── Action row: TTS + Romanization + Favorites + Download ──
                    st.markdown('<div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin:0.8rem 0 0.4rem;">', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    act1, act2, act3, act4, act5 = st.columns(5)
                    with act1:
                        hear_orig = st.checkbox("🔊 Original", key="tts_orig")
                    with act2:
                        hear_trans = st.checkbox("🔊 Translation", key="tts_trans", value=True)
                    with act3:
                        show_roman = st.checkbox("🔤 Romanize", key="roman_cb")
                    with act4:
                        if st.button("⭐ Save", use_container_width=True):
                            st.session_state.favorites.append({
                                "src": src_lang if not auto_detect else (detected_name or "Auto"),
                                "tgt": tgt_lang,
                                "original": text_input,
                                "translated": result,
                                "time": datetime.now().strftime("%H:%M"),
                            })
                            st.success("Saved to favorites!")
                    with act5:
                        both = f"[Original]\n{text_input}\n\n[{tgt_lang}]\n{result}"
                        st.download_button("⬇️ Export", data=both,
                                           file_name="translation.txt", mime="text/plain",
                                           use_container_width=True)

                    if hear_orig:
                        show_tts_player(text_input, src_code if src_code != "auto" else "en", "🔊 Original audio")
                    if hear_trans:
                        show_tts_player(result, tgt_code, "🔊 Translation audio")
                    if show_roman:
                        roman_tgt = get_romanization(result, tgt_code)
                        st.markdown(f"""
<div style="background:var(--surface2);border:1px solid var(--border2);border-radius:var(--r-sm);padding:0.8rem 1rem;margin-top:0.4rem;">
  <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:var(--muted);letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.4rem;">Romanization / Pronunciation</div>
  <div style="font-size:0.95rem;color:var(--text2);font-style:italic;">{roman_tgt or 'Not available for this script.'}</div>
</div>""", unsafe_allow_html=True)

                    # ── Back-translation quality check ──
                    with st.expander("🔁 Back-Translation Quality Check"):
                        st.markdown('<div style="font-size:0.78rem;color:var(--text2);margin-bottom:0.6rem;">Translates the result back to source to verify meaning was preserved.</div>', unsafe_allow_html=True)
                        back = back_translate(result, tgt_code, src_code if src_code != "auto" else "en")
                        st.markdown(f"""
<div style="background:var(--surface2);border:1px solid var(--border2);border-radius:var(--r-sm);padding:0.8rem 1rem;margin-bottom:0.6rem;">
  <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:var(--muted);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.3rem;">Back-translated</div>
  <div style="font-size:0.9rem;color:var(--text2);">{html.escape(back)}</div>
</div>""", unsafe_allow_html=True)
                        score = similarity_score(text_input, back)
                        bar_color = "#00c9a7" if score >= 60 else "#ffd166" if score >= 35 else "#ff6b6b"
                        label = "High confidence" if score >= 60 else "Moderate — review suggested" if score >= 35 else "Low — meaning may have shifted"
                        st.markdown(f"""
<div>
  <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
    <span style="font-size:0.78rem;color:var(--text2);">{label}</span>
    <span style="font-family:'DM Mono',monospace;font-size:0.78rem;color:{bar_color};font-weight:500;">{score}%</span>
  </div>
  <div class="score-bar-bg">
    <div class="score-bar-fill" style="width:{score}%;background:{bar_color};"></div>
  </div>
</div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # File upload
    st.markdown("---")
    with st.expander("📁 Translate a .txt file"):
        uploaded = st.file_uploader("Upload a plain text file", type=["txt"])
        if uploaded:
            file_text = uploaded.read().decode("utf-8", errors="ignore")
            st.text_area("File Preview", file_text[:400] + ("…" if len(file_text)>400 else ""), height=120, disabled=True)
            st.caption(f"{len(file_text)} chars · {len(file_text.split())} words")
            file_tgt = st.selectbox("Translate file to:", list(lang_map.keys()), key="file_tgt")
            if st.button("Translate File"):
                tgt_c = lang_map[file_tgt]
                words = file_text.split()
                chunk_size = 200
                chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
                prog = st.progress(0)
                translated_chunks = []
                with st.spinner("Translating file…"):
                    try:
                        for i, ch in enumerate(chunks):
                            t, _ = smart_translate(ch, "auto", tgt_c)
                            translated_chunks.append(t)
                            prog.progress((i+1)/len(chunks))
                            time.sleep(0.1)
                        file_result = " ".join(translated_chunks)
                        st.text_area("Translated File", file_result[:800], height=150, disabled=True)
                        st.download_button("⬇️ Download", data=file_result,
                                           file_name=f"translated_{tgt_c}.txt", mime="text/plain")
                    except Exception as e:
                        st.error(f"File translation error: {e}")

    with st.expander("🌍 All supported languages"):
        cols = st.columns(3)
        for i, (name, code) in enumerate(sorted(lang_map.items())):
            cols[i % 3].markdown(f"- **{name}** `{code}`")


# ────────────────────────────────────────────────
#  TAB 2 — COMPARE MULTIPLE LANGUAGES
# ────────────────────────────────────────────────
with tab_compare:
    st.markdown("""
<div style="margin-bottom:1.4rem;">
  <div style="font-family:'Fraunces',serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.3rem;">Side-by-Side Comparison</div>
  <div style="font-size:0.85rem;color:var(--text2);">Translate one phrase into multiple languages simultaneously and compare the results.</div>
</div>""", unsafe_allow_html=True)

    cmp_src_lang = st.selectbox("Source Language", list(lang_map.keys()), key="cmp_src")
    cmp_text = st.text_area("Text to compare:", height=100, key="cmp_text", placeholder="e.g. Hello, how are you?")
    all_langs = [l for l in lang_map.keys() if l != cmp_src_lang]
    default_targets = [d for d in ["Hindi","French","Spanish","Arabic","Japanese"] if d in all_langs]
    selected_targets = st.multiselect("Select target languages (up to 8):", options=all_langs,
                                       default=default_targets, max_selections=8)

    if st.button("🔀  Compare All", use_container_width=False):
        if not cmp_text.strip():
            st.warning("Please enter text to compare.")
        elif not selected_targets:
            st.warning("Select at least one target language.")
        else:
            src_c = lang_map[cmp_src_lang]
            with st.spinner(f"Translating into {len(selected_targets)} languages…"):
                results = translate_to_multiple(cmp_text, src_c, selected_targets)
            st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
            pairs = list(results.items())
            for row_start in range(0, len(pairs), 2):
                row_langs = pairs[row_start:row_start+2]
                cols = st.columns(len(row_langs))
                for col, (lang_name, data) in zip(cols, row_langs):
                    with col:
                        st.markdown(f"""
<div class="card" style="margin-bottom:0.8rem;">
  <div class="card-label">{lang_name} <span style="color:var(--teal);font-family:'DM Mono',monospace">{data['code']}</span></div>
  <div style="font-size:0.95rem;color:var(--text);line-height:1.65;min-height:50px;">{html.escape(data['text'])}</div>
  <div class="engine-pill" style="margin-top:0.6rem;">{data['engine']}</div>
</div>""", unsafe_allow_html=True)
                        if st.checkbox(f"🔊 Hear {lang_name}", key=f"tts_cmp_{lang_name}"):
                            show_tts_player(data["text"], data["code"])
            export_lines = [f"Original ({cmp_src_lang}):\n{cmp_text}\n"]
            for lang_name, data in results.items():
                export_lines.append(f"{lang_name}:\n{data['text']}\n")
            st.download_button("⬇️ Download All Comparisons", data="\n".join(export_lines),
                               file_name="comparison.txt", mime="text/plain")


# ────────────────────────────────────────────────
#  TAB 3 — OCR → TRANSLATE
# ────────────────────────────────────────────────
with tab_ocr:
    st.markdown("""
<div style="margin-bottom:1.4rem;">
  <div style="font-family:'Fraunces',serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.3rem;">OCR Vision</div>
  <div style="font-size:0.85rem;color:var(--text2);">Upload an image containing text — extract it with OCR then translate it instantly.</div>
</div>""", unsafe_allow_html=True)

    ocr_img = st.file_uploader("Upload image (JPG, PNG, BMP)", type=["jpg","jpeg","png","bmp"], key="ocr_uploader")
    if ocr_img:
        st.image(ocr_img, caption="Uploaded Image", use_column_width=True)
        img_bytes = ocr_img.read()
        mime_type = f"image/{ocr_img.type.split('/')[-1]}" if "/" in ocr_img.type else "image/png"
        if st.button("🔍 Extract Text (OCR)"):
            with st.spinner("Running OCR…"):
                extracted = ocr_image(img_bytes, mime_type)
            if extracted:
                st.session_state["ocr_text"] = extracted
                st.success("Text extracted!")
            else:
                st.warning("Could not extract text. Try a clearer image.")

    if st.session_state.get("ocr_text"):
        ocr_editable = st.text_area("Edit if needed:", value=st.session_state["ocr_text"], height=150, key="ocr_editable")
        ocr_tgt = st.selectbox("Translate to:", list(lang_map.keys()), key="ocr_tgt_lang")
        if st.button("🚀 Translate Extracted Text"):
            tgt_c = lang_map[ocr_tgt]
            with st.spinner("Translating…"):
                try:
                    ocr_result, ocr_engine = smart_translate(ocr_editable, "auto", tgt_c)
                    st.success("Translation Complete:")
                    st.text_area("Translated Text", ocr_result, height=150, key="ocr_result")
                    st.caption(f"⚙️ Engine: **{ocr_engine}**")
                    if st.checkbox("🔊 Hear translation", key="tts_ocr"):
                        show_tts_player(ocr_result, tgt_c)
                    if st.checkbox("🔤 Show romanization", key="roman_ocr"):
                        roman = get_romanization(ocr_result, tgt_c)
                        st.info(roman or "Romanization not available for this script.")
                    if st.button("⭐ Save to Favorites", key="fav_ocr"):
                        st.session_state.favorites.append({
                            "src": "OCR (Auto)", "tgt": ocr_tgt,
                            "original": ocr_editable, "translated": ocr_result,
                            "time": datetime.now().strftime("%H:%M"),
                        })
                        st.success("Saved!")
                    st.download_button("⬇️ Download Translation", data=ocr_result,
                                       file_name=f"ocr_translation_{tgt_c}.txt", mime="text/plain")
                    st.session_state.history.append({
                        "src": "OCR (Auto)", "tgt": ocr_tgt,
                        "original": ocr_editable, "translated": ocr_result,
                        "time": datetime.now().strftime("%H:%M"), "engine": ocr_engine,
                        "formality": "Neutral",
                    })
                except Exception as e:
                    st.error(f"Error: {e}")


# ────────────────────────────────────────────────
#  TAB 4 — WORD SENSE DISAMBIGUATION GUIDE
# ────────────────────────────────────────────────
with tab_disambig:
    st.markdown("""
<div style="margin-bottom:1.4rem;">
  <div style="font-family:'Fraunces',serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.3rem;">Word Sense Guide</div>
  <div style="font-size:0.85rem;color:var(--text2);">These ambiguous English words are automatically detected and disambiguated before translating.</div>
</div>""", unsafe_allow_html=True)

    for word, entry in sorted(AMBIGUOUS_WORDS.items()):
        with st.expander(f"**`{word}`** — {len(entry['senses'])} meanings"):
            for sense_name, sense_data in entry["senses"].items():
                st.markdown(f"**{sense_name}**")
                st.caption(f"Context hint used: _{sense_data['hint']}_")
                st.caption(f"Trigger keywords: `{', '.join(sense_data['keywords'][:6])}`")
                st.markdown("---")

    st.markdown("---")
    st.info(
        "**How disambiguation works:**\n\n"
        "1. When you type text, the app scans for known ambiguous words.\n"
        "2. Context keywords in your sentence are checked to auto-suggest the right meaning.\n"
        "3. You confirm (or override) the meaning before translating.\n"
        "4. A context hint is injected into the translation prompt so the AI picks the correct sense.\n"
        "5. Google Translate is always used for disambiguated text (better contextual NLP than MarianMT).\n\n"
        "**Example:** *'bats are flying'* → auto-detects 'flying' keyword → suggests Animal sense → "
        "translates to *'चमगादड़ उड़ रहे हैं'* ✅ instead of *'बल्ला उड़ रहे हैं'* ❌"
    )
