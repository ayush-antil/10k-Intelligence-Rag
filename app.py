import sys

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import re
import html
import base64
import time
import threading
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted


# =========================================================
# ICON ASSETS
# =========================================================
ASSETS_DIR = Path(__file__).parent / "assets"

ICON_FILES = {
    "brand": "bot.png",
    "Amazon": "amazon.png",
    "NVIDIA": "nvidia.png",
    "Starbucks": "starbucks.png",
    "JPMorgan": "jpmorgan.png",
}


@lru_cache(maxsize=None)
def icon_uri(name: str) -> str:
    """Base64 data URI for an icon in assets/. Empty string if the file is missing."""
    filename = ICON_FILES.get(name, "")
    if not filename:
        return ""

    path = ASSETS_DIR / filename
    if not path.is_file():
        return ""

    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"


def icon_img(name: str, css_class: str = "", fallback: str = "") -> str:
    """<img> tag for an icon, falling back to an emoji when the file is missing."""
    uri = icon_uri(name)
    if not uri:
        return fallback

    cls = f' class="{css_class}"' if css_class else ""
    return f'<img src="{uri}"{cls} alt="{name}">'


BRAND_ICON_PATH = ASSETS_DIR / ICON_FILES["brand"]


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="10-K Intelligence",
    page_icon=str(BRAND_ICON_PATH) if BRAND_ICON_PATH.is_file() else "📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# ENVIRONMENT
# =========================================================
load_dotenv()

# Locally this comes from .env via load_dotenv() above.
# On Streamlit Cloud, .env doesn't exist -- the key lives in
# st.secrets instead (Settings -> Secrets in the app dashboard).
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = ""

if not api_key:
    st.error(
        "GEMINI_API_KEY was not found. Add it to your .env file locally, "
        "or to this app's Secrets if running on Streamlit Cloud."
    )
    st.stop()

genai.configure(api_key=api_key)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #07111f;
        --bg-soft: #0a1727;
        --panel: #0d1b2f;
        --panel-2: #12243c;
        --border: #1d3655;
        --border-light: #2b507a;
        --text: #f4f8ff;
        --muted: #9eb1ca;
        --blue: #4da3ff;
        --cyan: #39d4d8;
        --green: #42d39b;
    }

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 0%, rgba(29, 104, 165, 0.12), transparent 28%),
            radial-gradient(circle at 45% 25%, rgba(25, 69, 115, 0.08), transparent 30%),
            var(--bg);
        color: var(--text);
    }

    /* Hide Streamlit chrome. We deliberately do NOT hide [data-testid="stToolbar"]:
       in recent Streamlit versions it also holds the "open sidebar" button, and
       hiding it leaves visitors with no way to reopen a collapsed sidebar.
       (.streamlit/config.toml -> toolbarMode = "minimal" already trims the toolbar.) */
    #MainMenu, footer, [data-testid="stDecoration"],
    [data-testid="stDeployButton"], .stAppDeployButton {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Always keep the "reopen sidebar" button visible (name differs by Streamlit version) */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        opacity: 1 !important;
        color: #cfe3ff !important;
    }

    [data-testid="stExpandSidebarButton"] *,
    [data-testid="stSidebarCollapsedControl"] *,
    [data-testid="collapsedControl"] * {
        visibility: visible !important;
        color: #cfe3ff !important;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 7rem;
    }

    /* -----------------------------------------------------
       SIDEBAR
    ----------------------------------------------------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #081527 0%, #09182b 100%);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.4rem;
    }

    .sidebar-brand {
        padding: 0.4rem 0.5rem 1.4rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.2rem;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }

    .brand-icon {
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 13px;
        background: linear-gradient(135deg, rgba(62, 165, 255, .25), rgba(57, 212, 216, .12));
        font-size: 1.35rem;
        flex: 0 0 42px;
    }

    .brand-icon img {
        width: 24px;
        height: 24px;
        object-fit: contain;
        /* white-ish icon on the dark panel */
        filter: brightness(0) invert(1) opacity(0.92);
    }

    .brand-title {
        color: #f4f8ff;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.03em;
    }

    .brand-subtitle {
        color: #9eb1ca;
        font-size: 0.78rem;
        line-height: 1.5;
        margin-top: 0.35rem;
        padding-left: 0.15rem;
    }

    .section-label {
        color: #7f97b7;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        margin: 1.2rem 0 0.6rem 0.2rem;
    }

    .side-info {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        color: #c6d4e5;
        font-size: 0.88rem;
        padding: 0.45rem 0.25rem;
    }

    .side-info span {
        color: #78b7ff;
        width: 22px;
        text-align: center;
    }

    .sidebar-footer {
        border-top: 1px solid var(--border);
        margin-top: 1.6rem;
        padding: 1.2rem 0.25rem 0;
        color: #9eb1ca;
        font-size: 0.75rem;
        line-height: 1.6;
    }

    .sidebar-footer b {
        color: #d7e7fb;
    }

    [data-testid="stSidebar"] .stCheckbox {
        padding: 0.25rem 0;
    }

    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stCheckbox label span,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #d6e1ef !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: var(--border);
        margin: 1rem 0;
    }

    /* -----------------------------------------------------
       TOP STATUS
    ----------------------------------------------------- */
    .topbar {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        margin-bottom: 1rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: #cce5db;
        background: rgba(18, 68, 61, 0.35);
        border: 1px solid rgba(66, 211, 155, 0.35);
        padding: 0.48rem 0.85rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--green);
        box-shadow: 0 0 12px rgba(66, 211, 155, 0.8);
    }

    /* -----------------------------------------------------
       HERO
    ----------------------------------------------------- */
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.3rem;
        margin-bottom: 1.25rem;
        border-radius: 18px;
        border: 1px solid var(--border);
        background:
            linear-gradient(115deg, rgba(16, 36, 60, 0.96), rgba(11, 30, 52, 0.92)),
            var(--panel);
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 420px;
        height: 420px;
        right: -130px;
        top: -260px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(53, 159, 255, 0.18), transparent 68%);
        pointer-events: none;
    }

    .hero-icon {
        position: relative;
        z-index: 1;
        font-size: 2.2rem;
        margin-bottom: 0.35rem;
    }

    .hero-icon img {
        width: 46px;
        height: 46px;
        object-fit: contain;
        filter: brightness(0) invert(1) opacity(0.92);
    }

    .hero h1 {
        position: relative;
        z-index: 1;
        color: #f4f8ff;
        margin: 0;
        font-size: clamp(2rem, 4vw, 3.25rem);
        letter-spacing: -0.045em;
        font-weight: 800;
    }

    .hero h1 span {
        background: linear-gradient(90deg, #f2f7ff 15%, #69adff 90%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        position: relative;
        z-index: 1;
        color: #b4c4d8;
        font-size: 1rem;
        line-height: 1.7;
        max-width: 760px;
        margin: 0.65rem 0 0;
    }

    /* -----------------------------------------------------
       FEATURE CARDS
    ----------------------------------------------------- */
    .feature-card {
        min-height: 145px;
        background: linear-gradient(145deg, rgba(15, 32, 54, 0.98), rgba(10, 25, 44, 0.98));
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 1.15rem;
        margin-bottom: 0.65rem;
        transition: 0.2s ease;
    }

    .feature-card:hover {
        border-color: var(--border-light);
        transform: translateY(-2px);
    }

    .feature-icon {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        margin-bottom: 0.75rem;
        background: rgba(77, 163, 255, 0.12);
        border: 1px solid rgba(77, 163, 255, 0.18);
    }

    .feature-title {
        color: #eaf2ff;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }

    .feature-text {
        color: #9eb1ca;
        font-size: 0.78rem;
        line-height: 1.5;
    }

    /* -----------------------------------------------------
       SUGGESTED QUESTIONS
    ----------------------------------------------------- */
    .suggestions {
        background: linear-gradient(145deg, rgba(11, 27, 47, 0.98), rgba(8, 22, 40, 0.98));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.3rem;
        margin: 0.9rem 0 1.2rem;
    }

    .suggestions-title {
        color: #eaf2ff;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .stButton > button {
        width: 100%;
        min-height: 58px;
        text-align: left;
        white-space: normal;
        color: #d8e4f3 !important;
        background: linear-gradient(145deg, #10233b, #0d1e34) !important;
        border: 1px solid #23456b !important;
        border-radius: 12px !important;
        padding: 0.7rem 0.9rem !important;
        font-size: 0.8rem !important;
        transition: all 0.18s ease;
    }

    .stButton > button:hover {
        border-color: #4384c9 !important;
        background: #132945 !important;
        color: #ffffff !important;
        transform: translateY(-1px);
    }

    /* -----------------------------------------------------
       CHAT SECTION
    ----------------------------------------------------- */
    .chat-header img {
        width: 21px;
        height: 21px;
        object-fit: contain;
        filter: brightness(0) invert(1) opacity(0.9);
    }

    .welcome-chat h3 img {
        width: 18px;
        height: 18px;
        object-fit: contain;
        vertical-align: -3px;
        margin-right: 7px;
        filter: brightness(0) invert(1) opacity(0.9);
    }

    .chat-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 1.4rem 0 0.8rem;
        padding: 0.9rem 1rem;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: rgba(11, 27, 47, 0.85);
        color: #eaf2ff;
        font-weight: 700;
    }

    .welcome-chat {
        max-width: 650px;
        margin: 1rem auto 1.2rem;
        padding: 1.2rem 1.3rem;
        background: linear-gradient(135deg, rgba(20, 46, 76, 0.92), rgba(15, 33, 57, 0.92));
        border: 1px solid rgba(54, 103, 151, 0.35);
        border-radius: 16px;
    }

    .welcome-chat h3 {
        color: #edf5ff;
        margin: 0 0 0.45rem;
        font-size: 1rem;
    }

    .welcome-chat p {
        color: #b9c8da;
        margin: 0;
        line-height: 1.7;
        font-size: 0.85rem;
    }

    [data-testid="stChatMessage"] {
        background: linear-gradient(145deg, rgba(12, 29, 49, 0.85), rgba(9, 22, 39, 0.85)) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        padding: 0.85rem 1rem !important;
        margin-bottom: 0.8rem !important;
    }

    [data-testid="stChatMessageContent"] {
        color: #eaf2ff !important;
        line-height: 1.7;
    }

    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li,
    [data-testid="stChatMessageContent"] span {
        color: #dce7f5 !important;
    }

    /* Source badges */
    .source-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.7rem 0 0.25rem;
    }

    .source-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        color: #9ac7ff;
        background: rgba(34, 78, 121, 0.22);
        border: 1px solid rgba(74, 145, 215, 0.38);
        border-radius: 999px;
        padding: 0.28rem 0.62rem;
        font-size: 0.7rem;
        font-weight: 600;
    }

    /* -----------------------------------------------------
       CHAT MESSAGE ROWS (avatar + bubble + timestamp)
    ----------------------------------------------------- */
    .msg-row {
        display: flex;
        align-items: flex-start;
        gap: 0.85rem;
        margin-bottom: 1.1rem;
    }

    .msg-avatar {
        width: 40px;
        height: 40px;
        min-width: 40px;
        border-radius: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.28);
    }

    .msg-avatar.user {
        background: linear-gradient(135deg, #4f9df5, #2b5da6);
    }

    .msg-avatar.assistant {
        background: linear-gradient(135deg, #eef4fc, #cfe0f7);
    }

    .msg-avatar svg {
        display: block;
    }

    .msg-avatar img {
        display: block;
        width: 23px;
        height: 23px;
        object-fit: contain;
    }

    .msg-avatar.user svg {
        color: #ffffff;
    }

    .msg-avatar.assistant svg {
        color: #12243c;
    }

    .msg-bubble-wrap {
        flex: 1;
        min-width: 0;
    }

    .msg-bubble {
        background: linear-gradient(145deg, rgba(15, 32, 54, 0.92), rgba(10, 25, 44, 0.92));
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.95rem 1.15rem;
        color: #dce7f5;
        font-size: 0.92rem;
        line-height: 1.7;
    }

    .msg-bubble p {
        margin: 0 0 0.55rem;
    }

    .msg-bubble p:last-child {
        margin-bottom: 0;
    }

    .msg-bubble ul {
        margin: 0.2rem 0 0.6rem 1.15rem;
        padding: 0;
    }

    .msg-bubble li {
        margin-bottom: 0.35rem;
    }

    .msg-bubble b {
        color: #f4f8ff;
    }

    .msg-time {
        text-align: right;
        color: #7f97b7;
        font-size: 0.72rem;
        margin-top: 0.4rem;
        padding-right: 0.2rem;
    }

    /* -----------------------------------------------------
       SIMPLE CHAT INPUT
    ----------------------------------------------------- */

    /* Remove the bright default Streamlit bottom strip */
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottom"],
    .stBottom {
        background: var(--bg) !important;
        border-top: none !important;
        box-shadow: none !important;
        padding: 0.5rem 1.25rem 0.7rem !important;
    }

    [data-testid="stBottomBlockContainer"] *,
    [data-testid="stBottom"] * {
        border-top: none !important;
    }

    /* Single, simple rounded box for the whole input */
    [data-testid="stChatInput"] {
        max-width: 1100px !important;
        margin: 0 auto !important;
        background: var(--panel-2) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 14px !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Kill every nested wrapper's own border/background so there is
       only ONE visible box, not a box inside a box */
    [data-testid="stChatInput"] *,
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="base-input"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Text area */
    [data-testid="stChatInput"] textarea {
        min-height: 34px !important;
        height: 34px !important;
        padding: 0.5rem 3.4rem 0.5rem 0.9rem !important;
        color: #edf5ff !important;
        caret-color: #ffffff !important;
        font-size: 0.9rem !important;
        line-height: 1.3 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #8fa6bd !important;
        opacity: 1 !important;
    }

    /* Send button */
    [data-testid="stChatInput"] button {
        width: 32px !important;
        height: 32px !important;
        min-width: 32px !important;
        background: #2d74c8 !important;
        color: #ffffff !important;
        border-radius: 9px !important;
    }

    [data-testid="stChatInput"] button:hover {
        background: #3a84dd !important;
    }

    /* Keep the final answer safely above the fixed input */
    .block-container {
        padding-bottom: 7rem !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    @media (max-width: 900px) {
        .block-container {
            padding: 1rem 1rem 2rem;
        }

        .hero {
            padding: 1.5rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# COMPANY LOGOS ON THE SIDEBAR CHECKBOXES
# =========================================================
CHECKBOX_ICON_KEYS = {
    "amazon_filter": "Amazon",
    "nvidia_filter": "NVIDIA",
    "starbucks_filter": "Starbucks",
    "jpmorgan_filter": "JPMorgan",
}

_checkbox_rules = []

for _widget_key, _icon_name in CHECKBOX_ICON_KEYS.items():
    _uri = icon_uri(_icon_name)
    if not _uri:
        continue

    _checkbox_rules.append(
        f'''
        .st-key-{_widget_key} [data-testid="stMarkdownContainer"] p::before {{
            content: "";
            display: inline-block;
            width: 19px;
            height: 19px;
            margin-right: 9px;
            vertical-align: -4px;
            background-image: url("{_uri}");
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
        }}
        '''
    )

if _checkbox_rules:
    st.markdown(
        "<style>" + "".join(_checkbox_rules) + "</style>",
        unsafe_allow_html=True,
    )


# =========================================================
# CACHED RAG RESOURCES
# =========================================================
CHROMA_PATH = str(Path(__file__).parent / "chroma_db")


@st.cache_resource
def load_resources():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_or_create_collection("tenk_filings")
    llm = genai.GenerativeModel("gemini-3.5-flash-lite")
    return embedder, collection, llm


embedder, collection, llm = load_resources()


# =========================================================
# SESSION STATE
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

MAX_QUESTIONS_PER_SESSION = 10

# Gemini's free tier is shared by EVERY visitor, so besides the per-session
# cap above we throttle across all sessions. Stay under the API's per-minute limit.
GLOBAL_MAX_CALLS = 12
GLOBAL_WINDOW_SECONDS = 60


@st.cache_resource
def _rate_limiter():
    return {"lock": threading.Lock(), "calls": []}


def global_rate_ok() -> bool:
    """True if a Gemini call is allowed right now (and reserves a slot for it)."""
    limiter = _rate_limiter()
    now = time.time()
    with limiter["lock"]:
        limiter["calls"] = [
            t for t in limiter["calls"] if now - t < GLOBAL_WINDOW_SECONDS
        ]
        if len(limiter["calls"]) >= GLOBAL_MAX_CALLS:
            return False
        limiter["calls"].append(now)
        return True


# =========================================================
# COMPANY FILTER HELPERS
# =========================================================
COMPANY_SOURCE_KEYWORDS = {
    "Amazon": ["amazon"],
    "NVIDIA": ["nvidia", "nvda"],
    "Starbucks": ["starbucks"],
    "JPMorgan": ["jpmorgan", "jp morgan", "jpm", "chase"],
}


def source_matches_selected_company(source, selected_companies):
    """
    Returns True when a ChromaDB source name belongs to one
    of the currently selected companies.
    """
    source_text = str(source).lower()

    for company in selected_companies:
        keywords = COMPANY_SOURCE_KEYWORDS.get(company, [])
        if any(keyword in source_text for keyword in keywords):
            return True

    return False


# =========================================================
# AUTO-SCROLL HELPER
# =========================================================
def scroll_chat_to_latest():
    """
    Smoothly scrolls the conversation to the newest answer while
    Streamlit keeps the chat input fixed at the bottom of the viewport.
    """
    components.html(
        """
        <script>
        function moveToLatestAnswer() {
            const doc = window.parent.document;
            const latest = doc.getElementById("latest-answer");

            if (latest) {
                latest.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }
        }

        // Wait for Streamlit to finish painting the answer.
        setTimeout(moveToLatestAnswer, 350);
        </script>
        """,
        height=0,
        width=0,
    )


def scroll_page_to_top():
    """
    On the very first load, make sure the page opens at the top (hero header
    visible). Streamlit focuses the bottom chat input on load, which can leave
    the page scrolled down, so we reset the scroll a few times while it settles.
    """
    components.html(
        """
        <script>
        function toTop() {
            const doc = window.parent.document;
            const targets = [
                doc.querySelector('[data-testid="stMain"]'),
                doc.querySelector('section.main'),
                doc.querySelector('[data-testid="stAppViewContainer"]'),
                doc.scrollingElement,
                doc.documentElement,
                doc.body,
            ];
            targets.forEach(function (el) { if (el) { el.scrollTop = 0; } });
            window.parent.scrollTo(0, 0);
        }
        [0, 150, 500, 1200].forEach(function (t) { setTimeout(toTop, t); });
        </script>
        """,
        height=0,
        width=0,
    )


# =========================================================
# RAG FUNCTIONS
# =========================================================
USER_AVATAR_SVG = """
<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
<circle cx="12" cy="8" r="4"/>
<path d="M4 20c0-4.4 3.6-7 8-7s8 2.6 8 7"/>
</svg>
"""

ASSISTANT_AVATAR_SVG = """
<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
<rect x="4" y="8" width="16" height="12" rx="3"/>
<rect x="9" y="2" width="2" height="4"/>
<circle cx="9" cy="14" r="1.6" fill="#eef4fc"/>
<circle cx="15" cy="14" r="1.6" fill="#eef4fc"/>
</svg>
"""

# bot.png when available; the SVG above is the fallback.
ASSISTANT_AVATAR = icon_img("brand", fallback=ASSISTANT_AVATAR_SVG)


def current_time_label():
    """Returns a clock label like '10:24 AM' without a leading zero."""
    return datetime.now().strftime("%I:%M %p").lstrip("0")


def markdown_lite_to_html(text):
    """
    Converts a small, safe subset of Markdown (bold text and bullet
    lists) coming back from the LLM into HTML so it can be embedded
    directly inside a custom chat bubble.
    """
    escaped = html.escape(text or "")
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)

    html_parts = []
    in_list = False

    for raw_line in escaped.split("\n"):
        line = raw_line.strip()

        if line.startswith("- ") or line.startswith("* "):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            html_parts.append(f"<li>{line[2:].strip()}</li>")
            continue

        if in_list:
            html_parts.append("</ul>")
            in_list = False

        if line:
            html_parts.append(f"<p>{line}</p>")

    if in_list:
        html_parts.append("</ul>")

    return "".join(html_parts)


def sources_html(sources):
    if not sources:
        return ""

    tags = "".join(
        f'<span class="source-tag">📄 {source}</span>'
        for source in sources
    )

    return f'<div class="source-wrap">{tags}</div>'


def render_chat_row(role, content, sources=None, timestamp=None, anchor_id=None):
    """
    Renders one chat turn as a custom row: a small square avatar on the
    left, and a bubble on the right containing the message, any source
    citations, and a timestamp aligned to the bottom-right.
    """
    is_user = role == "user"
    avatar_class = "user" if is_user else "assistant"
    avatar_svg = USER_AVATAR_SVG if is_user else ASSISTANT_AVATAR

    body_html = markdown_lite_to_html(content)
    src_html = sources_html(sources) if not is_user else ""
    time_html = f'<div class="msg-time">{timestamp}</div>' if timestamp else ""
    anchor_html = f'<div id="{anchor_id}"></div>' if anchor_id else ""

    st.markdown(
        f"""
        {anchor_html}
        <div class="msg-row {avatar_class}">
            <div class="msg-avatar {avatar_class}">{avatar_svg}</div>
            <div class="msg-bubble-wrap">
                <div class="msg-bubble">{body_html}{src_html}</div>
                {time_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Exact `source` metadata values written by 01_build_rag.py (PDF file name
# without the extension, e.g. data/amazon_10k.pdf -> "amazon_10k").
COMPANY_SOURCES = {
    "Amazon": "amazon_10k",
    "NVIDIA": "nvidia_10k",
    "Starbucks": "starbucks_10k",
    "JPMorgan": "jpmorgan_10k",
}


def retrieve(query, selected_companies, top_k=8):
    """
    Retrieves the most relevant chunks from each selected company's filing.

    Each company is queried separately, with Chroma filtering by `source`,
    so only a handful of chunks are ever loaded (not the whole database).
    Results are interleaved so every selected company gets fair space in
    the context -- with one global top_k, a single company could fill every
    slot for a question like "which company is strongest?".
    """
    if not selected_companies:
        return []

    query_embedding = embedder.encode([query]).tolist()
    per_company = max(2, top_k // len(selected_companies))

    buckets = []
    for company in selected_companies:
        source_name = COMPANY_SOURCES.get(company)
        if not source_name:
            continue

        res = collection.query(
            query_embeddings=query_embedding,
            n_results=per_company,
            where={"source": source_name},
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        buckets.append(list(zip(docs, metas)))

    results = []
    for i in range(per_company):
        for matches in buckets:
            if i < len(matches):
                results.append(matches[i])

    return results


def answer_question(query, selected_companies, top_k=8):
    retrieved = retrieve(
        query=query,
        selected_companies=selected_companies,
        top_k=top_k,
    )

    if not retrieved:
        selected_text = ", ".join(selected_companies)

        return (
            "I couldn't find relevant indexed documents for the currently "
            f"selected companies: **{selected_text}**.\n\n"
            "Try selecting another company or check whether the `source` "
            "metadata in your ChromaDB collection contains recognizable "
            "company names.",
            [],
        )

    context = "\n\n---\n\n".join(
        f"[Source: {meta.get('source', 'Unknown source')}]\n{doc}"
        for doc, meta in retrieved
    )

    prompt = f"""
You are a professional financial research assistant.

Answer the user's question using ONLY the context below.

Rules:
- Do not invent or guess information.
- If the context does not contain the answer, clearly say so.
- Give a direct, well-structured answer.
- Use bullet points when helpful.
- Keep financial explanations clear and professional.
- Mention source companies or filings when relevant.

Selected companies:
{", ".join(selected_companies)}

Context:
{context}

Question:
{query}
"""

    response = llm.generate_content(prompt, request_options={"timeout": 60})

    sources = sorted(
        {
            meta.get("source", "Unknown source")
            for _, meta in retrieved
        }
    )

    return response.text, sources


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="brand-row">
                <div class="brand-icon">{icon_img("brand", fallback="📊")}</div>
                <div class="brand-title">10-K Intelligence</div>
            </div>
            <div class="brand-subtitle">
                AI-powered financial research<br>
                across company filings
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">COMPANIES</div>',
        unsafe_allow_html=True,
    )

    amazon = st.checkbox("Amazon", value=True, key="amazon_filter")
    nvidia = st.checkbox("NVIDIA", value=True, key="nvidia_filter")
    starbucks = st.checkbox("Starbucks", value=True, key="starbucks_filter")
    jpmorgan = st.checkbox("JPMorgan", value=True, key="jpmorgan_filter")

    selected_companies = []

    if amazon:
        selected_companies.append("Amazon")
    if nvidia:
        selected_companies.append("NVIDIA")
    if starbucks:
        selected_companies.append("Starbucks")
    if jpmorgan:
        selected_companies.append("JPMorgan")

    if not selected_companies:
        st.warning("Select at least one company before asking a question.")

    remaining = MAX_QUESTIONS_PER_SESSION - st.session_state.question_count
    st.caption(f"{max(remaining, 0)} of {MAX_QUESTIONS_PER_SESSION} questions left this session")

    st.divider()

    st.markdown(
        '<div class="section-label">KNOWLEDGE BASE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="side-info"><span>▣</span>4 Companies</div>
        <div class="side-info"><span>▤</span>10-K Filings Indexed</div>
        <div class="side-info"><span>✦</span>RAG Powered</div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="section-label">FEATURES</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="side-info"><span>↔</span>Company Comparison</div>
        <div class="side-info"><span>↗</span>Financial Analysis</div>
        <div class="side-info"><span>⚠</span>Risk Insights</div>
        <div class="side-info"><span>◉</span>Key Metrics</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-footer">
            <b>✦ Powered by Gemini + RAG</b><br>
            Better insights. Smarter decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# KNOWLEDGE BASE CHECK
# =========================================================
if collection.count() == 0:
    st.warning(
        "No documents are loaded yet. Run your indexing script first "
        "to build the ChromaDB knowledge base."
    )
    st.stop()


# =========================================================
# MAIN UI
# =========================================================
st.markdown(
    """
    <div class="topbar">
        <div class="status-pill">
            <span class="status-dot"></span>
            RAG Active
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-icon">{icon_img("brand", fallback="📊")}</div>
        <h1><span>10-K Intelligence</span></h1>
        <p>
            Ask questions about company filings and get accurate,
            AI-powered insights from Amazon, NVIDIA, Starbucks,
            and JPMorgan annual reports.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# FEATURE CARDS
# =========================================================
feature_data = [
    ("📈", "Financial Performance", "Analyze revenue, profit margins, and growth trends."),
    ("🛡️", "Risk Analysis", "Identify key risks and challenges mentioned in filings."),
    ("↔️", "Compare Companies", "Compare business strategies and financial performance."),
    ("📄", "Source Citations", "Every answer is grounded in your 10-K documents."),
]

feature_cols = st.columns(4)

for col, (icon, title, description) in zip(feature_cols, feature_data):
    with col:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-text">{description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# SUGGESTED QUESTIONS
# =========================================================
if not st.session_state.messages:
    st.markdown(
        """
        <div class="suggestions">
            <div class="suggestions-title">✦ Try asking these questions</div>
        """,
        unsafe_allow_html=True,
    )

    questions = [
        "Compare Amazon and NVIDIA's revenue growth.",
        "Summarize JPMorgan's financial performance and key highlights.",
        "Compare the business strategies of Amazon and Starbucks.",
        "What are the major risks for Starbucks?",
        "Which company has the strongest financial position?",
        "What upcoming risks are mentioned in NVIDIA's 10-K?",
    ]

    rows = [questions[:3], questions[3:]]

    for row_index, row in enumerate(rows):
        cols = st.columns(3)

        for question_index, (col, question) in enumerate(zip(cols, row)):
            with col:
                if st.button(
                    question,
                    key=f"suggestion_{row_index}_{question_index}",
                ):
                    st.session_state.pending_query = question

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# CHAT AREA
# =========================================================
st.markdown(
    f"""
    <div class="chat-header">
        {icon_img("brand", fallback="🤖")} Financial Research Assistant
    </div>
    """,
    unsafe_allow_html=True,
)


if not st.session_state.messages:
    st.markdown(
        f"""
        <div class="welcome-chat">
            <h3>{icon_img("brand", fallback="🤖")}Hello! Welcome to 10-K Intelligence</h3>
            <p>
                Ask me anything about the company filings. I can compare
                businesses, summarize financial performance, analyze risks,
                and retrieve information directly from the indexed 10-K reports.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Display chat history
for msg in st.session_state.messages:
    render_chat_row(
        role=msg["role"],
        content=msg["content"],
        sources=msg.get("sources"),
        timestamp=msg.get("time"),
    )


# =========================================================
# FIXED BOTTOM CHAT INPUT
# =========================================================
# Native st.chat_input stays fixed at the bottom of the screen.
# All user and assistant messages render above it.
query = st.chat_input(
    "Ask anything about the 10-K filings...",
    key="main_chat_input",
)

# Suggested questions can also submit a query.
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None


# =========================================================
# PROCESS QUERY SAFELY
# =========================================================
if query:
    if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
        st.warning(
            f"You've reached the limit of {MAX_QUESTIONS_PER_SESSION} questions "
            "for this session. Please refresh the page to start a new session."
        )

    elif not selected_companies:
        st.warning(
            "Please select at least one company in the sidebar before asking a question."
        )

    elif not query.strip():
        st.info("Please enter a question.")

    elif not global_rate_ok():
        st.warning(
            "The assistant is busy right now (shared free-tier limit). "
            "Please try again in about a minute -- this didn't count "
            "against your question limit."
        )

    else:
        st.session_state.question_count += 1
        user_time = current_time_label()

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
                "sources": None,
                "time": user_time,
            }
        )

        render_chat_row(role="user", content=query, timestamp=user_time)

        with st.spinner(
            "Searching selected filings and analyzing financial information..."
        ):
            try:
                answer, sources = answer_question(
                    query=query,
                    selected_companies=selected_companies,
                )

                assistant_time = current_time_label()

                render_chat_row(
                    role="assistant",
                    content=answer,
                    sources=sources,
                    timestamp=assistant_time,
                    anchor_id="latest-answer",
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "time": assistant_time,
                    }
                )

                scroll_chat_to_latest()

            except ResourceExhausted:
                # Gemini's free tier caps requests per minute (15) and per
                # day (500). This fires when either is hit -- it's not a
                # bug, just too many questions in too short a window.
                error_message = (
                    "This assistant is getting a lot of questions right now "
                    "and has hit its free-tier rate limit.\n\n"
                    "Please wait a minute and try again."
                )

                assistant_time = current_time_label()

                render_chat_row(
                    role="assistant",
                    content=error_message,
                    timestamp=assistant_time,
                    anchor_id="latest-answer",
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                        "time": assistant_time,
                    }
                )

            except Exception as error:
                # Full details go to the server log (Manage app -> Logs),
                # not to the visitor.
                print(f"[10-K Intelligence] {type(error).__name__}: {error}", flush=True)
                error_message = (
                    "Sorry, something went wrong while generating the answer.\n\n"
                    "Please try again, or rephrase your question."
                )

                assistant_time = current_time_label()

                render_chat_row(
                    role="assistant",
                    content=error_message,
                    timestamp=assistant_time,
                    anchor_id="latest-answer",
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                        "time": assistant_time,
                    }
                )

                scroll_chat_to_latest()


# =========================================================
# FIRST LOAD: open at the top of the page (once per session)
# =========================================================
if "initial_scroll_done" not in st.session_state:
    st.session_state.initial_scroll_done = True
    scroll_page_to_top()