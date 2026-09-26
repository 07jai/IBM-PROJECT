# ============================================================
# AI-POWERED QR CODE PHISHING RISK PREDICTOR
# Streamlit Web App — Multilingual Voice Alert
# ============================================================

import streamlit as st
import cv2
import av
import numpy as np
import requests
import ipaddress
import re
import os
import io
import base64
import hashlib
import threading
import joblib

from urllib.parse import urlparse, parse_qs
from PIL import Image
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI QR Phishing Risk Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
/* ── Base & Font ── */
html, body, [class*="css"] {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.stApp { background-color: #0d1117; color: #c9d1d9; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Result cards ── */
.result-safe {
    background: #0d2818; border: 2px solid #2ea043;
    border-radius: 12px; padding: 18px 20px; margin: 8px 0;
}
.result-suspicious {
    background: #1f1800; border: 2px solid #d29922;
    border-radius: 12px; padding: 18px 20px; margin: 8px 0;
}
.result-dangerous {
    background: #1e0a0a; border: 2px solid #f85149;
    border-radius: 12px; padding: 18px 20px; margin: 8px 0;
}

/* ── Verdict labels ── */
.big-label-safe { font-size: 1.9rem; font-weight: 800; color: #3fb950; letter-spacing: 3px; }
.big-label-sus  { font-size: 1.9rem; font-weight: 800; color: #d29922; letter-spacing: 3px; }
.big-label-dan  { font-size: 1.9rem; font-weight: 800; color: #f85149; letter-spacing: 3px; }

/* ── Reason text ── */
.reason { margin: 4px 0; font-size: 0.92rem; line-height: 1.5; }

/* ── QR type chip ── */
.qr-badge {
    display: inline-block; background: #161b22;
    border: 1px solid #30363d; border-radius: 20px;
    padding: 3px 12px; font-size: 0.82rem;
    color: #58a6ff; margin-top: 6px; letter-spacing: 0.5px;
}

/* ── Collapse empty space below WebRTC camera widget ── */
.stWebRtcComponent, [data-testid="stWebRtcComponent"],
div[class*="webrtc"] { margin-bottom: 0 !important; padding-bottom: 0 !important; }
iframe[title*="streamlit_webrtc"], iframe[src*="webrtc"] {
    display: block; margin-bottom: 0 !important;
}
/* Remove extra blank vertical space Streamlit injects after iframes/custom components */
.element-container:has(iframe) { margin-bottom: 0 !important; padding-bottom: 0 !important; }
.element-container + .element-container:empty { display: none !important; }
div[data-testid="column"] > div > div > div:empty { display: none !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}

/* ── Metric boxes ── */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px 12px;
}
[data-testid="metric-container"] label {
    color: #8b949e; font-size: 0.78rem;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #c9d1d9; font-size: 1.4rem; font-weight: 700;
}

/* ── Buttons ── */
.stButton > button {
    background: #21262d; color: #c9d1d9;
    border: 1px solid #30363d; border-radius: 8px;
    font-size: 0.88rem; font-weight: 600; padding: 8px 18px;
    transition: background 0.2s, border-color 0.2s, color 0.2s;
}
.stButton > button:hover {
    background: #30363d; border-color: #58a6ff; color: #58a6ff;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 10px 10px 0 0;
    border: 1px solid #30363d;
    border-bottom: none;
    gap: 2px; padding: 4px 6px 0 6px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px 8px 0 0;
    color: #8b949e; font-size: 0.9rem;
    font-weight: 600; padding: 8px 20px; border: none;
}
.stTabs [aria-selected="true"] {
    background: #0d1117 !important;
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 0 10px 10px 10px;
    padding: 1.4rem 1.6rem;
}

/* ── Alerts ── */
.stSuccess > div, .stInfo > div, .stWarning > div, .stError > div {
    border-radius: 8px; font-size: 0.9rem;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
    font-size: 0.88rem !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LANGUAGE DATA
# ============================================================

LANG_CODES = {
    "English":"en","Tamil":"ta","Telugu":"te","Kannada":"kn","Malayalam":"ml",
    "Hindi":"hi","Bengali":"bn","Gujarati":"gu","Punjabi":"pa","Urdu":"ur",
    "Spanish":"es","French":"fr","German":"de","Italian":"it","Portuguese":"pt",
    "Russian":"ru","Turkish":"tr","Dutch":"nl","Japanese":"ja","Korean":"ko",
    "Chinese":"zh-CN","Arabic":"ar","Indonesian":"id",
}

INDIA_STATE_LANG = {
    "tamil nadu":"Tamil","kerala":"Malayalam","karnataka":"Kannada",
    "telangana":"Telugu","andhra pradesh":"Telugu","west bengal":"Bengali",
    "gujarat":"Gujarati","punjab":"Punjabi","maharashtra":"Hindi",
    "delhi":"Hindi","uttar pradesh":"Hindi","rajasthan":"Hindi",
    "madhya pradesh":"Hindi","bihar":"Hindi","haryana":"Hindi",
    "himachal pradesh":"Hindi","uttarakhand":"Hindi","jharkhand":"Hindi",
    "chhattisgarh":"Hindi","odisha":"Bengali","assam":"Bengali","goa":"English",
}

COUNTRY_LANG = {
    "spain":"Spanish","mexico":"Spanish","argentina":"Spanish","colombia":"Spanish",
    "chile":"Spanish","france":"French","belgium":"French","germany":"German",
    "austria":"German","switzerland":"German","italy":"Italian","japan":"Japanese",
    "south korea":"Korean","china":"Chinese","saudi arabia":"Arabic",
    "united arab emirates":"Arabic","egypt":"Arabic","brazil":"Portuguese",
    "portugal":"Portuguese","russia":"Russian","turkey":"Turkish",
    "netherlands":"Dutch","indonesia":"Indonesian","bangladesh":"Bengali","pakistan":"Urdu",
}

VOICE_MESSAGES = {
    "English":    {"SAFE":"This QR code is safe. You can proceed.",
                   "SUSPICIOUS":"Warning! This QR code is suspicious. Verify before proceeding.",
                   "DANGEROUS":"Danger! This QR code is a phishing threat. Do not proceed."},
    "Tamil":      {"SAFE":"இந்த QR குறியீடு பாதுகாப்பானது. நீங்கள் தொடரலாம்.",
                   "SUSPICIOUS":"எச்சரிக்கை! இந்த QR குறியீடு சந்தேகத்திற்குரியது. தொடர்வதற்கு முன் சரிபார்க்கவும்.",
                   "DANGEROUS":"ஆபத்து! இந்த QR குறியீடு ஃபிஷிங் அச்சுறுத்தல். தொடர வேண்டாம்."},
    "Malayalam":  {"SAFE":"ഈ QR കോഡ് സുരക്ഷിതമാണ്. നിങ്ങൾക്ക് തുടരാം.",
                   "SUSPICIOUS":"മുന്നറിയിപ്പ്! ഈ QR കോഡ് സംശയാസ്പദമാണ്. തുടരുന്നതിന് മുമ്പ് പരിശോധിക്കുക.",
                   "DANGEROUS":"അപകടം! ഈ QR കോഡ് ഫിഷിംഗ് ഭീഷണിയാണ്. തുടരരുത്."},
    "Kannada":    {"SAFE":"ಈ QR ಕೋಡ್ ಸುರಕ್ಷಿತವಾಗಿದೆ. ನೀವು ಮುಂದುವರಿಯಬಹುದು.",
                   "SUSPICIOUS":"ಎಚ್ಚರಿಕೆ! ಈ QR ಕೋಡ್ ಅನುಮಾನಾಸ್ಪದವಾಗಿದೆ. ಮುಂದುವರಿಯುವ ಮೊದಲು ಪರಿಶೀಲಿಸಿ.",
                   "DANGEROUS":"ಅಪಾಯ! ಈ QR ಕೋಡ್ ಫಿಶಿಂಗ್ ಬೆದರಿಕೆ. ಮುಂದುವರಿಯಬೇಡಿ."},
    "Telugu":     {"SAFE":"ఈ QR కోడ్ సురక్షితంగా ఉంది. మీరు కొనసాగవచ్చు.",
                   "SUSPICIOUS":"హెచ్చరిక! ఈ QR కోడ్ అనుమానాస్పదంగా ఉంది. కొనసాగించే ముందు తనిఖీ చేయండి.",
                   "DANGEROUS":"ప్రమాదం! ఈ QR కోడ్ ఫిషింగ్ ముప్పు. కొనసాగవద్దు."},
    "Hindi":      {"SAFE":"यह QR कोड सुरक्षित है। आप आगे बढ़ सकते हैं।",
                   "SUSPICIOUS":"चेतावनी! यह QR कोड संदिग्ध है। आगे बढ़ने से पहले सत्यापित करें।",
                   "DANGEROUS":"खतरा! यह QR कोड एक फिशिंग खतरा है। आगे न बढ़ें।"},
    "Bengali":    {"SAFE":"এই QR কোডটি নিরাপদ। আপনি এগিয়ে যেতে পারেন।",
                   "SUSPICIOUS":"সতর্কতা! এই QR কোডটি সন্দেহজনক। এগিয়ে যাওয়ার আগে যাচাই করুন।",
                   "DANGEROUS":"বিপদ! এই QR কোডটি একটি ফিশিং হুমকি। এগিয়ে যাবেন না।"},
    "Gujarati":   {"SAFE":"આ QR કોડ સુરક્ષિત છે. તમે આગળ વધી શકો છો.",
                   "SUSPICIOUS":"ચેતવણી! આ QR કોડ શંકાસ્પદ છે. આગળ વધતા પહેલા ચકાસો.",
                   "DANGEROUS":"ખતરો! આ QR કોડ ફિશિંગ ખતરો છે. આગળ ન વધો."},
    "Punjabi":    {"SAFE":"ਇਹ QR ਕੋਡ ਸੁਰੱਖਿਅਤ ਹੈ। ਤੁਸੀਂ ਅੱਗੇ ਵਧ ਸਕਦੇ ਹੋ।",
                   "SUSPICIOUS":"ਚੇਤਾਵਨੀ! ਇਹ QR ਕੋਡ ਸ਼ੱਕੀ ਹੈ। ਅੱਗੇ ਵਧਣ ਤੋਂ ਪਹਿਲਾਂ ਜਾਂਚ ਕਰੋ।",
                   "DANGEROUS":"ਖਤਰਾ! ਇਹ QR ਕੋਡ ਫਿਸ਼ਿੰਗ ਖਤਰਾ ਹੈ। ਅੱਗੇ ਨਾ ਵਧੋ।"},
    "Urdu":       {"SAFE":"یہ QR کوڈ محفوظ ہے۔ آپ آگے بڑھ سکتے ہیں۔",
                   "SUSPICIOUS":"خبردار! یہ QR کوڈ مشکوک ہے۔ آگے بڑھنے سے پہلے تصدیق کریں۔",
                   "DANGEROUS":"خطرہ! یہ QR کوڈ فشنگ خطرہ ہے۔ آگے نہ بڑھیں۔"},
    "Spanish":    {"SAFE":"Este código QR es seguro. Puede continuar.",
                   "SUSPICIOUS":"Advertencia! Este código QR es sospechoso. Verifique antes de continuar.",
                   "DANGEROUS":"Peligro! Este código QR es una amenaza de phishing. No continúe."},
    "French":     {"SAFE":"Ce code QR est sûr. Vous pouvez continuer.",
                   "SUSPICIOUS":"Attention! Ce code QR est suspect. Vérifiez avant de continuer.",
                   "DANGEROUS":"Danger! Ce code QR est une menace de phishing. Ne continuez pas."},
    "German":     {"SAFE":"Dieser QR-Code ist sicher. Sie können fortfahren.",
                   "SUSPICIOUS":"Warnung! Dieser QR-Code ist verdächtig. Bitte prüfen Sie ihn.",
                   "DANGEROUS":"Gefahr! Dieser QR-Code ist eine Phishing-Bedrohung. Fahren Sie nicht fort."},
    "Italian":    {"SAFE":"Questo codice QR è sicuro. Puoi procedere.",
                   "SUSPICIOUS":"Attenzione! Questo codice QR è sospetto. Verifica prima di procedere.",
                   "DANGEROUS":"Pericolo! Questo codice QR è una minaccia di phishing. Non procedere."},
    "Portuguese": {"SAFE":"Este código QR é seguro. Você pode continuar.",
                   "SUSPICIOUS":"Aviso! Este código QR é suspeito. Verifique antes de continuar.",
                   "DANGEROUS":"Perigo! Este código QR é uma ameaça de phishing. Não continue."},
    "Russian":    {"SAFE":"Этот QR-код безопасен. Вы можете продолжить.",
                   "SUSPICIOUS":"Предупреждение! Этот QR-код подозрителен. Проверьте перед продолжением.",
                   "DANGEROUS":"Опасность! Этот QR-код является угрозой фишинга. Не продолжайте."},
    "Turkish":    {"SAFE":"Bu QR kodu güvenli. Devam edebilirsiniz.",
                   "SUSPICIOUS":"Uyarı! Bu QR kodu şüpheli. Devam etmeden önce doğrulayın.",
                   "DANGEROUS":"Tehlike! Bu QR kodu bir kimlik avı tehdididir. Devam etmeyin."},
    "Dutch":      {"SAFE":"Deze QR-code is veilig. U kunt doorgaan.",
                   "SUSPICIOUS":"Waarschuwing! Deze QR-code is verdacht. Controleer voordat u doorgaat.",
                   "DANGEROUS":"Gevaar! Deze QR-code vormt een phishing-bedreiging. Ga niet verder."},
    "Japanese":   {"SAFE":"このQRコードは安全です。続行できます。",
                   "SUSPICIOUS":"警告! このQRコードは疑わしいです。続行する前に確認してください。",
                   "DANGEROUS":"危険! このQRコードはフィッシングの脅威です。続行しないでください。"},
    "Korean":     {"SAFE":"이 QR 코드는 안전합니다. 계속 진행할 수 있습니다.",
                   "SUSPICIOUS":"경고! 이 QR 코드는 의심스럽습니다. 계속하기 전에 확인하세요.",
                   "DANGEROUS":"위험! 이 QR 코드는 피싱 위협입니다. 진행하지 마세요."},
    "Chinese":    {"SAFE":"此二维码是安全的。您可以继续。",
                   "SUSPICIOUS":"警告！此二维码可疑。请在继续之前进行验证。",
                   "DANGEROUS":"危险！此二维码是网络钓鱼威胁。请勿继续。"},
    "Arabic":     {"SAFE":"رمز QR هذا آمن. يمكنك المتابعة.",
                   "SUSPICIOUS":"تحذير! رمز QR هذا مشبوه. يرجى التحقق قبل المتابعة.",
                   "DANGEROUS":"خطر! رمز QR هذا يمثل تهديد تصيد. لا تتابع."},
    "Indonesian": {"SAFE":"Kode QR ini aman. Anda dapat melanjutkan.",
                   "SUSPICIOUS":"Peringatan! Kode QR ini mencurigakan. Verifikasi sebelum melanjutkan.",
                   "DANGEROUS":"Bahaya! Kode QR ini merupakan ancaman phishing. Jangan lanjutkan."},
}


# ============================================================
# SESSION STATE
# ============================================================

for _k, _v in {
    "scan_result":        None,
    "last_qr_hash":       None,
    "scan_count":         0,
    "dangerous_count":    0,
    "selected_language":  "English",
    "location_name":      "Unknown",
    "location_city":      "",
    "location_state":     "",
    "location_country":   "",
    "location_detected":  False,
    "auto_voice_done":    False,
    "pending_qr":         None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ============================================================
# LOCATION DETECTION
# ============================================================

def detect_location_language():
    APIS = [
        ("http://ip-api.com/json/",        "city", "regionName", "country"),
        ("https://freeipapi.com/api/json", "cityName", "regionName", "countryName"),
        ("https://ipapi.co/json/",         "city", "region",     "country_name"),
    ]
    for url, ck, sk, ctk in APIS:
        try:
            r = requests.get(url, timeout=8)
            if r.status_code != 200:
                continue
            d       = r.json()
            city    = d.get(ck,  "") or ""
            state   = d.get(sk,  "") or ""
            country = d.get(ctk, "") or ""
            sl = state.strip().lower()
            cl = country.strip().lower()
            if not sl and not cl:
                continue
            for key, lang in INDIA_STATE_LANG.items():
                if key in sl:
                    return lang, city, state, country
            for key, lang in COUNTRY_LANG.items():
                if key in cl:
                    return lang, city, state, country
            if "india" in cl:
                return "Hindi", city, state, country
            if cl:
                return "English", city, state, country
        except Exception:
            continue
    return "English", "", "", ""


# ============================================================
# ML MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    for p in ["qr_phishing_model.pkl", os.path.join("model", "qr_phishing_model.pkl")]:
        if os.path.exists(p):
            try:
                m = joblib.load(p)
                m.predict(np.array([[1, 1, -1, 1]]))
                return m
            except Exception:
                pass
    return None

ML_MODEL = load_model()


def ml_predict(url):
    if ML_MODEL is None:
        return None, 0.0
    try:
        parsed   = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        try:    ipaddress.ip_address(hostname); having_ip = -1
        except: having_ip = 1
        l       = len(url)
        url_len = 1 if l < 54 else (0 if l <= 75 else -1)
        https   = -1 if url.lower().startswith("https") else 1
        psfx    = -1 if "-" in hostname else 1
        X       = np.array([[having_ip, url_len, https, psfx]])
        pred    = ML_MODEL.predict(X)[0]
        try:
            proba   = ML_MODEL.predict_proba(X)[0]
            classes = list(ML_MODEL.classes_)
            conf    = float(proba[classes.index(-1)]) if -1 in classes else (0.8 if pred == -1 else 0.2)
        except:
            conf = 0.8 if pred == -1 else 0.2
        return ("DANGEROUS" if pred == -1 else "SAFE"), conf
    except:
        return None, 0.0


# ============================================================
# QR TYPE
# ============================================================

def detect_qr_type(data):
    t = data.strip().lower()
    if t.startswith("upi://"):                          return "UPI Payment"
    if re.match(r"^https?://", t):                      return "URL / Web"
    if t.startswith("wifi:"):                           return "Wi-Fi"
    if t.startswith("begin:vcard"):                     return "Contact / vCard"
    if t.startswith("mailto:"):                         return "Email"
    if t.startswith("tel:"):                            return "Phone"
    if t.startswith("sms:") or t.startswith("smsto:"): return "SMS"
    if t.startswith("geo:"):                            return "Location / Geo"
    if "begin:vevent" in t:                             return "Calendar / Event"
    return "Plain Text"


# ============================================================
# ANALYSIS RULES
# ============================================================

PHISHING_KW = [
    "login","signin","verify","verification","account","update","secure","security",
    "bank","banking","paypal","amazon","google","microsoft","apple","netflix",
    "password","confirm","unlock","suspended","urgent","alert","warning","free",
    "winner","prize","lottery","claim","reward","kyc","aadhar","aadhaar","refund",
]
SUSPICIOUS_TLDS = {".tk",".ml",".ga",".cf",".gq",".xyz",".top",".work",
                   ".click",".link",".online",".zip",".mov"}
URL_SHORTENERS  = {"bit.ly","tinyurl.com","goo.gl","t.co","ow.ly","is.gd",
                   "buff.ly","tiny.cc","rb.gy","cutt.ly","shorturl.at"}


def analyze_url(url):
    safe=[]; sus=[]; dan=[]
    try:
        parsed   = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        lower    = url.lower()
    except:
        return {"classification":"DANGEROUS","safe":[],"suspicious":[],"dangerous":["Invalid URL."],
                "qr_type":"URL / Web","content":url,"ml_label":None,"ml_conf":0.0}

    ml_label, ml_conf = ml_predict(url)
    if ml_label == "DANGEROUS": dan.append(f"🤖 ML: phishing pattern ({ml_conf:.0%})")
    elif ml_label == "SAFE":    safe.append(f"🤖 ML: no phishing ({1-ml_conf:.0%})")

    try:    ipaddress.ip_address(hostname); dan.append("Raw IP address used instead of domain.")
    except: safe.append("Named domain detected.")

    if "@" in url:              dan.append("'@' symbol hides real destination.")
    if "xn--" in hostname:      dan.append("Punycode domain — possible impersonation.")
    if len(url) > 150:          dan.append(f"Extremely long URL ({len(url)} chars).")
    elif len(url) > 75:         sus.append(f"Long URL ({len(url)} chars).")
    if lower.startswith("https://"): safe.append("HTTPS encrypted.")
    else:                            sus.append("Not HTTPS — unencrypted.")
    if len(re.findall(r"%[0-9a-fA-F]{2}", url)) >= 5: dan.append("Excessive percent-encoding.")
    if hostname.count(".") > 4: dan.append("Too many subdomains.")
    if any(hostname.endswith(t) for t in SUSPICIOUS_TLDS): sus.append("Suspicious TLD.")
    if hostname in URL_SHORTENERS: sus.append("URL shortener hides destination.")
    else:                          safe.append("Not a URL shortener.")
    if "-" in hostname: sus.append("Hyphenated domain — look-alike risk.")
    hits = [k for k in PHISHING_KW if k in lower]
    if hits: sus.append("Phishing keywords: " + ", ".join(hits[:5]))
    else:    safe.append("No phishing keywords.")
    if "//" in (parsed.path or ""): sus.append("Double slash in path.")

    cls = "DANGEROUS" if dan else ("SUSPICIOUS" if sus else "SAFE")
    return {"classification":cls,"safe":safe,"suspicious":sus,"dangerous":dan,
            "qr_type":"URL / Web","content":url,"ml_label":ml_label,"ml_conf":ml_conf}


def analyze_upi(data):
    safe=[]; sus=[]; dan=[]
    try:
        params = parse_qs(urlparse(data).query)
        payee  = params.get("pa",[""])[0]
        name   = params.get("pn",[""])[0]
        amount = params.get("am",[""])[0]
    except: payee=name=amount=""
    lower = data.lower()

    if not payee:
        dan.append("UPI payee (pa) missing.")
    elif re.match(r"^[a-zA-Z0-9._-]{2,}@[a-zA-Z]{2,}$", payee):
        safe.append(f"Valid UPI ID: {payee}")
    else:
        sus.append(f"Unusual UPI ID: {payee}")

    if name:  safe.append(f"Payee: {name}")
    else:     sus.append("Payee name missing.")

    if amount:
        try:
            a = float(amount)
            if a == 0:    sus.append("Amount is zero.")
            elif a>50000: sus.append(f"High amount: ₹{a:,.2f}")
            else:         safe.append(f"Amount: ₹{a:,.2f}")
        except: sus.append("Invalid amount.")

    for w in ["verify-bank","account-verify","otp","pin","kyc","aadhaar","aadhar",
              "reward","prize","lottery","cashback","refund","rbi","government",
              "expire","block","urgent"]:
        if w in lower: dan.append(f"Dangerous keyword: '{w}'"); break
    for w in ["verify","login","update","secure","confirm"]:
        if w in lower: sus.append(f"Suspicious keyword: '{w}'"); break

    if not dan and not sus: safe.append("No suspicious UPI patterns.")
    cls = "DANGEROUS" if dan else ("SUSPICIOUS" if sus else "SAFE")
    return {"classification":cls,"safe":safe,"suspicious":sus,"dangerous":dan,
            "qr_type":"UPI Payment","content":data,"ml_label":None,"ml_conf":0.0}


def analyze_generic(data, qr_type):
    safe=[]; sus=[]
    hits = [k for k in PHISHING_KW if k in data.lower()]
    if hits: sus.append("Keywords: " + ", ".join(hits[:5]))
    else:    safe.append("No phishing keywords.")
    if len(data) > 500: sus.append(f"Large content ({len(data)} chars).")
    safe.append(f"QR type: {qr_type}")
    cls = "SUSPICIOUS" if sus else "SAFE"
    return {"classification":cls,"safe":safe,"suspicious":sus,"dangerous":[],
            "qr_type":qr_type,"content":data,"ml_label":None,"ml_conf":0.0}


def classify_qr(data):
    if not data: return None
    qt = detect_qr_type(data)
    if qt == "URL / Web":   return analyze_url(data)
    if qt == "UPI Payment": return analyze_upi(data)
    return analyze_generic(data, qt)


# ============================================================
# QR DECODER
# ============================================================

def _try_decode(det, img):
    try:
        d, pts, _ = det.detectAndDecode(img)
        if d and d.strip(): return d.strip(), pts
    except: pass
    try:
        ok, dec, pts, _ = det.detectAndDecodeMulti(img)
        if ok and dec:
            for item in dec:
                if item and item.strip(): return item.strip(), pts
    except: pass
    return None, None


def decode_qr_robust(image):
    if image is None: return None, None
    det  = cv2.QRCodeDetector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    variants = [image, gray]
    h, w = gray.shape[:2]
    if min(h, w) < 800:
        s = 800 / min(h, w)
        variants.append(cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC))
    try: variants.append(cv2.createCLAHE(2.0, (8, 8)).apply(gray))
    except: pass
    try:
        _, t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(t)
    except: pass
    try: variants.append(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5))
    except: pass
    try: variants.append(cv2.bitwise_not(gray))
    except: pass
    for ang in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
        try: variants.append(cv2.rotate(gray, ang))
        except: pass
    for v in variants:
        d, pts = _try_decode(det, v)
        if d: return d, pts
    return None, None


# ============================================================
# LIVE CAMERA PROCESSOR
# ============================================================

class LiveQRScanner(VideoProcessorBase):
    def __init__(self):
        self.detector      = cv2.QRCodeDetector()
        self.lock          = threading.Lock()
        self.latest_data   = None
        self.new_ready     = False
        self._frame_no     = 0
        self._last_decoded = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._frame_no += 1
        if self._frame_no % 2 == 0:
            data = pts = None
            try:
                d, p, _ = self.detector.detectAndDecode(img)
                if d and d.strip():
                    data, pts = d.strip(), p
            except: pass
            if not data:
                try:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    data, pts = _try_decode(self.detector, gray)
                except: pass
            if data:
                with self.lock:
                    if data != self._last_decoded:
                        self.latest_data   = data
                        self.new_ready     = True
                        self._last_decoded = data
                if pts is not None:
                    try:
                        p = np.asarray(pts, dtype=np.int32).reshape(-1, 2)
                        if len(p) >= 4:
                            cv2.polylines(img, [p], True, (0, 255, 0), 3)
                    except: pass
                cv2.putText(img, "QR DETECTED", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(img, "Scanning for QR...", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_latest(self):
        with self.lock:
            if self.new_ready:
                self.new_ready = False
                return self.latest_data
        return None


# ============================================================
# VOICE
# ============================================================

def generate_voice(cls, lang):
    msg  = VOICE_MESSAGES.get(lang, VOICE_MESSAGES["English"]).get(cls, "")
    code = LANG_CODES.get(lang, "en")
    try:
        buf = io.BytesIO()
        gTTS(text=msg, lang=code, slow=False).write_to_fp(buf)
        buf.seek(0)
        return buf.read(), msg
    except:
        return None, msg

def autoplay_audio(audio_bytes):
    if not audio_bytes: return
    enc = base64.b64encode(audio_bytes).decode()
    st.markdown(
        f'<audio autoplay><source src="data:audio/mp3;base64,{enc}" type="audio/mp3"></audio>',
        unsafe_allow_html=True,
    )

def display_result(result, language, key_suffix=""):
    cls     = result["classification"]
    qr_type = result["qr_type"]
    content = result["content"]
    css  = {"SAFE":"result-safe", "SUSPICIOUS":"result-suspicious", "DANGEROUS":"result-dangerous"}
    lbl  = {"SAFE":"big-label-safe", "SUSPICIOUS":"big-label-sus", "DANGEROUS":"big-label-dan"}
    icon = {"SAFE":"🟢", "SUSPICIOUS":"🟡", "DANGEROUS":"🔴"}
    _ukey = hashlib.md5((content + key_suffix).encode()).hexdigest()[:8]
    st.markdown(
        f'<div class="{css[cls]}">'
        f'<div class="{lbl[cls]}">{icon[cls]} {cls}</div>'
        f'<div class="qr-badge">📌 {qr_type}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    dan_list = result.get("dangerous",  [])
    sus_list = result.get("suspicious", [])
    saf_list = result.get("safe",       [])
    st.markdown("#### 🔍 Why this result?")
    if dan_list:
        st.markdown(
            "<div style='background:#2d0a0a;border-left:4px solid #f85149;"
            "border-radius:6px;padding:10px 14px;margin:6px 0;'>"
            "<div style='color:#f85149;font-weight:700;font-size:0.88rem;"
            "margin-bottom:6px;'>🔴 DANGEROUS REASONS</div>"
            + "".join(
                f"<div class='reason' style='color:#ffa198;margin:3px 0;'>✕ {r}</div>"
                for r in dan_list
            )
            + "</div>",
            unsafe_allow_html=True,
        )
    if sus_list:
        st.markdown(
            "<div style='background:#2d1f00;border-left:4px solid #d29922;"
            "border-radius:6px;padding:10px 14px;margin:6px 0;'>"
            "<div style='color:#d29922;font-weight:700;font-size:0.88rem;"
            "margin-bottom:6px;'>🟡 SUSPICIOUS REASONS</div>"
            + "".join(
                f"<div class='reason' style='color:#e3b341;margin:3px 0;'>⚠ {r}</div>"
                for r in sus_list
            )
            + "</div>",
            unsafe_allow_html=True,
        )
    if saf_list:
        st.markdown(
            "<div style='background:#0d2818;border-left:4px solid #2ea043;"
            "border-radius:6px;padding:10px 14px;margin:6px 0;'>"
            "<div style='color:#3fb950;font-weight:700;font-size:0.88rem;"
            "margin-bottom:6px;'>🟢 SAFE INDICATORS</div>"
            + "".join(
                f"<div class='reason' style='color:#7ee787;margin:3px 0;'>✓ {r}</div>"
                for r in saf_list
            )
            + "</div>",
            unsafe_allow_html=True,
        )
    if not dan_list and not sus_list and not saf_list:
        st.info("No detailed reasons available.")
    with st.expander("📋 Decoded QR Content", expanded=False):
        st.code(content[:600], language=None)
    st.markdown("#### 🔊 Voice Alert")
    voice_text = VOICE_MESSAGES.get(language, VOICE_MESSAGES["English"]).get(cls, "")
    st.info(f"💬 **{language}:** {voice_text}")
    if st.button("▶️ Play Voice Alert", key=f"voice_{_ukey}"):
        audio, _ = generate_voice(cls, language)
        if audio:
            st.audio(audio, format="audio/mp3")
        else:
            st.warning("Voice unavailable — check internet.")
    if cls == "DANGEROUS":
        st.error("🚫 DO NOT proceed! Phishing threat detected!")
    elif cls == "SUSPICIOUS":
        st.warning("⚠️ Verify before proceeding.")
    else:
        st.success("✅ QR code appears safe.")

def process_new_qr(data):
    qr_hash = hashlib.sha256(data.encode()).hexdigest()
    if qr_hash == st.session_state.last_qr_hash:
        return st.session_state.scan_result
    result = classify_qr(data)
    if result is None:
        return None
    st.session_state.last_qr_hash    = qr_hash
    st.session_state.scan_result     = result
    st.session_state.auto_voice_done = False
    st.session_state.scan_count     += 1
    if result["classification"] == "DANGEROUS":
        st.session_state.dangerous_count += 1
    return result

with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    if not st.session_state.location_detected:
        with st.spinner("📡 Detecting your location..."):
            dl, city, state, country = detect_location_language()
        loc_str = ", ".join(x for x in [city, state, country] if x) or "Unknown"
        st.session_state.selected_language = dl
        st.session_state.location_name     = loc_str
        st.session_state.location_city     = city
        st.session_state.location_state    = state
        st.session_state.location_country  = country
        st.session_state.location_detected = True
    st.markdown(
        f"<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;"
        f"padding:10px;margin-bottom:8px;'>"
        f"<div style='color:#58a6ff;font-size:0.85rem;font-weight:600;'>📍 Your Location</div>"
        f"<div style='color:#c9d1d9;font-size:0.9rem;margin-top:4px;'>{st.session_state.location_name}</div>"
        f"<div style='color:#3fb950;font-size:0.85rem;margin-top:4px;'>🌐 Auto Language: "
        f"<b>{st.session_state.selected_language}</b></div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    lang_list = list(LANG_CODES.keys())
    cur_idx   = lang_list.index(st.session_state.selected_language) \
                if st.session_state.selected_language in lang_list else 0
    selected_language = st.selectbox("🔤 Override Language", lang_list, index=cur_idx)
    st.session_state.selected_language = selected_language
    st.markdown("---")
    st.markdown("**🤖 ML Model**")
    if ML_MODEL:
        st.success("Loaded — RandomForest")
        st.caption("IP · URL length · HTTPS · prefix/suffix")
    else:
        st.error("Model not found")
        st.caption("Rule-based analysis is active.")
    st.markdown("---")
    st.markdown("**📊 Session Stats**")
    c1, c2 = st.columns(2)
    c1.metric("Scanned", st.session_state.scan_count)
    c2.metric("Threats",  st.session_state.dangerous_count)
    st.markdown("---")
    st.markdown("**🏷️ Legend**\n\n🟢 SAFE — No threat\n\n🟡 SUSPICIOUS — Verify\n\n🔴 DANGEROUS — Phishing")

st.markdown("""
<div style="text-align:center;padding:10px 0 4px 0;">
<span style="font-size:2rem;font-weight:700;color:#58a6ff;letter-spacing:2px;">
    🛡️ AI QR PHISHING RISK PREDICTOR
  </span>
<br>
<span style="color:#8b949e;font-size:0.95rem;">
    ML + Rule-Based &nbsp;•&nbsp; Auto Location Language &nbsp;•&nbsp; Multilingual Voice Alert
  </span>
</div>
<hr style="border-color:#21262d;margin:8px 0 12px 0;">
""", unsafe_allow_html=True)

_lang_flag = {
    "Tamil":"🇮🇳 Tamil Nadu","Telugu":"🇮🇳 Telugu","Kannada":"🇮🇳 Karnataka",
    "Malayalam":"🇮🇳 Kerala","Hindi":"🇮🇳 Hindi Belt","Bengali":"🇮🇳 Bengali",
    "Gujarati":"🇮🇳 Gujarat","Punjabi":"🇮🇳 Punjab","English":"🌍 English",
    "Spanish":"🇪🇸 Spanish","French":"🇫🇷 French","German":"🇩🇪 German",
    "Italian":"🇮🇹 Italian","Portuguese":"🇧🇷 Portuguese","Russian":"🇷🇺 Russian",
    "Turkish":"🇹🇷 Turkish","Dutch":"🇳🇱 Dutch","Japanese":"🇯🇵 Japanese",
    "Korean":"🇰🇷 Korean","Chinese":"🇨🇳 Chinese","Arabic":"🇸🇦 Arabic",
    "Indonesian":"🇮🇩 Indonesian","Urdu":"🇵🇰 Urdu",
}
_cur_lang = st.session_state.selected_language
_cur_loc  = st.session_state.location_name
_flag_txt = _lang_flag.get(_cur_lang, _cur_lang)

st.markdown(
    f"<div style='background:#0d2818;border:1px solid #2ea043;border-radius:10px;"
    f"padding:10px 18px;margin-bottom:14px;display:flex;align-items:center;"
    f"justify-content:space-between;flex-wrap:wrap;gap:8px;'>"
    f"<span style='color:#c9d1d9;font-size:0.92rem;'>📍 <b>Location:</b> {_cur_loc}</span>"
    f"<span style='color:#3fb950;font-size:0.95rem;font-weight:700;'>"
    f"🔊 Voice Language Auto-Set → {_flag_txt} &nbsp;|&nbsp; {_cur_lang}</span>"
    f"</div>",
    unsafe_allow_html=True,
)

tab_camera, tab_upload, tab_manual = st.tabs(["📷 Live Camera", "🖼️ Upload Image", "✏️ Manual Input"])

with tab_camera:
    if st.session_state.pending_qr:
        _pending = st.session_state.pending_qr
        st.session_state.pending_qr = None
        _result = process_new_qr(_pending)
        if _result and not st.session_state.auto_voice_done:
            _audio, _ = generate_voice(_result["classification"], selected_language)
            if _audio:
                autoplay_audio(_audio)
            st.session_state.auto_voice_done = True

    # ── Camera centred ──
    _, col_cam, _ = st.columns([0.5, 2, 0.5])
    with col_cam:
        st.markdown("#### 📷 Camera Scanner")
        st.info("▶️ Click **START** — hold a QR code in front of the camera. "
                "It will be detected and analyzed automatically.")
        ctx = webrtc_streamer(
            key="qr-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=LiveQRScanner,
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                "audio": False,
            },
            async_processing=True,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
        if ctx.video_processor:
            _new = ctx.video_processor.get_latest()
            if _new:
                st.session_state.pending_qr = _new
                st.rerun()

    # ── Security Analysis below camera ──
    st.markdown("#### 🛡️ Security Analysis")
    if st.session_state.scan_result:
        display_result(st.session_state.scan_result, selected_language, key_suffix="cam")
    else:
        st.markdown(
            "<div style='text-align:center;color:#484f58;padding:40px 10px;font-size:1rem;'>"
            "🔍 Waiting for QR code...<br>"
            "<span style='font-size:0.85rem;'>Start camera and show a QR code</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    @st.fragment(run_every=0.5)
    def _poll_camera():
        if ctx.video_processor:
            _d = ctx.video_processor.get_latest()
            if _d:
                st.session_state.pending_qr = _d
                st.rerun()
    _poll_camera()

with tab_upload:
    st.markdown("#### 🖼️ Upload a QR Code Image")
    uploaded = st.file_uploader(
        "PNG, JPG, JPEG, WEBP, BMP",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        label_visibility="collapsed",
    )
    if uploaded:
        try:
            img_pil = Image.open(uploaded).convert("RGB")
            st.image(img_pil, caption="Uploaded Image", use_container_width=True)
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            data, _ = decode_qr_robust(img_bgr)
            if data:
                st.success(f"✅ QR decoded: `{data[:80]}`")
                st.session_state.last_qr_hash = None
                result = process_new_qr(data)
                if result:
                    display_result(result, selected_language, key_suffix="upload")
            else:
                st.error("❌ No QR code found. Try a clearer image.")
        except Exception as err:
            st.error(f"Image error: {err}")

with tab_manual:
    st.markdown("#### ✏️ Paste QR Content Manually")
    manual_input = st.text_area(
        "Content",
        height=90,
        placeholder="https://example.com  or  upi://pay?pa=name@bank&pn=Name&am=100",
        label_visibility="collapsed",
    )
    if st.button("🔍 Analyze", key="btn_manual"):
        text = manual_input.strip()
        if text:
            st.session_state.last_qr_hash = None
            result = process_new_qr(text)
            if result:
                display_result(result, selected_language, key_suffix="manual")
        else:
            st.warning("Please enter content first.")

st.markdown("""
<div style="text-align:center;color:#484f58;margin-top:28px;
            padding-top:12px;border-top:1px solid #21262d;font-size:0.82rem;">
  🔐 AI-Powered QR Phishing Risk Predictor &nbsp;|&nbsp; IBM Bob 2.0 Hackathon
</div>
""", unsafe_allow_html=True)