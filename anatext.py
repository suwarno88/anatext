import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from openai import OpenAI
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import json
from collections import Counter
from itertools import combinations
from datetime import datetime

APP_VERSION = "2.5.0-enhanced"

# ============================================================
#                  KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="AnaText - AI Text Analysis",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
#              DEFINISI STOPWORDS & CONSTANTS
# ============================================================
default_stopwords_id = [
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk', 'pada', 'adalah', 'sebagai',
    'dalam', 'tidak', 'akan', 'juga', 'atau', 'ada', 'mereka', 'sudah', 'saya', 'kita', 'kami', 'kalian',
    'dia', 'ia', 'anda', 'bisa', 'hanya', 'lebih', 'karena', 'tetapi', 'tapi', 'namun', 'jika', 'maka',
    'oleh', 'saat', 'agar', 'seperti', 'bahwa', 'telah', 'dapat', 'menjadi', 'tersebut', 'sangat', 'sehingga',
    'secara', 'antara', 'sebuah', 'suatu', 'begitu', 'lagi', 'masih', 'banyak', 'semua', 'setiap', 'serta',
    'hal', 'bila', 'pun', 'lalu', 'kemudian', 'yakni', 'yaitu', 'apabila', 'ketika', 'baik', 'paling',
    'demi', 'hingga', 'sampai', 'tanpa', 'belum', 'harus', 'sedang', 'maupun', 'selain', 'melalui',
    'sendiri', 'beberapa', 'apa', 'siapa', 'mana', 'kapan', 'bagaimana', 'mengapa', 'kenapa'
]

default_stopwords_en = [
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with',
    'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her',
    'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out',
    'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just',
    'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
    'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use',
    'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
    'give', 'day', 'most', 'us', 'is', 'are', 'was', 'were', 'been', 'being', 'has', 'had', 'having',
    'does', 'did', 'doing', 'am'
]

SENTIMENT_COLORS = {
    'Positif': '#28a745',
    'Negatif': '#dc3545',
    'Netral': '#ffc107',
    'Error': '#6c757d'
}

TOPIC_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
    '#F7DC6F', '#BB8FCE', '#F1948A', '#82E0AA', '#85C1E9'
]

# ============================================================
#                   STATE MANAGEMENT
# ============================================================
if 'stop_words' not in st.session_state:
    st.session_state.stop_words = list(set(default_stopwords_id + default_stopwords_en))
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'topic_details' not in st.session_state:
    st.session_state.topic_details = []
if 'ner_results' not in st.session_state:
    st.session_state.ner_results = None
if 'summary_cache' not in st.session_state:
    st.session_state.summary_cache = None
if 'entity_network' not in st.session_state:
    st.session_state.entity_network = None
if 'tab_insights' not in st.session_state:
    st.session_state.tab_insights = {}

# ============================================================
#                  CSS & THEME ENGINE
# ============================================================
def inject_custom_css(mode):
    if mode == 'Dark':
        bg_color = "#0e1117"
        sidebar_bg = "#1a1d24"
        text_color = "#e8e8e8"
        text_secondary = "#9ca3af"
        input_bg = "#2d3139"
        border_col = "#3d4150"
        btn_txt = "#ffffff"
        card_bg = "#1a1d24"
        card_border = "#2d3139"
        accent = "#4facfe"
        metric_bg = "#1e2128"
        hover_bg = "#262b36"
        tab_bg = "#1a1d24"
        success_bg = "#0d3320"
        info_bg = "#0d2847"
        warning_bg = "#3d2e00"
    else:
        bg_color = "#f8f9fb"
        sidebar_bg = "#ffffff"
        text_color = "#1a1a2e"
        text_secondary = "#64748b"
        input_bg = "#ffffff"
        border_col = "#e2e8f0"
        btn_txt = "#ffffff"
        card_bg = "#ffffff"
        card_border = "#e2e8f0"
        accent = "#3b82f6"
        metric_bg = "#f1f5f9"
        hover_bg = "#f8fafc"
        tab_bg = "#ffffff"
        success_bg = "#ecfdf5"
        info_bg = "#eff6ff"
        warning_bg = "#fffbeb"

    st.markdown(f"""
    <style>
        /* ---- BASE ---- */
        .stApp {{ background-color: {bg_color}; color: {text_color}; }}
        [data-testid="stSidebar"] {{ background-color: {sidebar_bg}; border-right: 1px solid {border_col}; }}
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {{ color: {text_color} !important; }}
        p, h1, h2, h3, h4, h5, h6, li, span, div, label {{ color: {text_color}; }}
        .stMarkdown {{ color: {text_color} !important; }}

        /* ---- INPUTS ---- */
        .stTextInput > div > div > input {{ color: {text_color}; background-color: {input_bg}; border: 1px solid {border_col}; border-radius: 8px; }}
        .stTextArea textarea {{ color: {text_color}; background-color: {input_bg}; border: 1px solid {border_col}; border-radius: 8px; }}
        div[data-baseweb="select"] > div {{ background-color: {input_bg}; color: {text_color}; border-radius: 8px; }}

        /* ---- UPLOAD AREA ---- */
        [data-testid='stFileUploader'] {{
            background-color: {card_bg};
            border: 2px dashed {accent};
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
        }}
        [data-testid='stFileUploader']:hover {{ border-color: #00f2fe; }}

        /* ---- BUTTONS ---- */
        .stButton button {{
            font-weight: 600;
            color: {btn_txt} !important;
            border-radius: 8px;
            transition: all 0.2s ease;
            border: none;
        }}
        .stButton button:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px rgba(79,172,254,0.3); }}

        /* ---- METRICS CARDS ---- */
        [data-testid="stMetric"] {{
            background: {metric_bg};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        [data-testid="stMetric"] label {{ color: {text_secondary} !important; font-size: 0.85rem; font-weight: 500; }}
        [data-testid="stMetric"] [data-testid="stMetricValue"] {{ color: {text_color} !important; font-weight: 700; }}

        /* ---- TABS ---- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background: {metric_bg};
            border-radius: 10px;
            padding: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 500;
            color: {text_secondary};
        }}
        .stTabs [aria-selected="true"] {{
            background: {accent} !important;
            color: white !important;
            font-weight: 600;
        }}

        /* ---- TABLES ---- */
        [data-testid="stDataFrame"] {{ border: 1px solid {border_col}; border-radius: 8px; overflow: hidden; }}

        /* ---- DIVIDER ---- */
        hr {{ border-color: {border_col}; opacity: 0.5; }}

        /* ---- CUSTOM CARDS ---- */
        .custom-card {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }}

        .stat-card {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            transition: all 0.2s ease;
        }}
        .stat-card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,0.08); transform: translateY(-2px); }}
        .stat-card .stat-value {{ font-size: 2rem; font-weight: 800; line-height: 1.2; }}
        .stat-card .stat-label {{ font-size: 0.85rem; color: {text_secondary}; margin-top: 4px; font-weight: 500; }}

        .ner-card {{
            background: {card_bg};
            border-left: 4px solid;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 8px;
        }}

        /* ---- FOOTER ---- */
        .footer-text {{
            text-align: center; font-size: 12px; color: {text_secondary};
            margin-top: 50px; border-top: 1px solid {border_col}; padding-top: 10px;
        }}

        /* ---- EXPANDER ---- */
        .streamlit-expanderHeader {{ background-color: {card_bg}; border-radius: 8px; }}

        /* ---- SCROLLBAR ---- */
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: {bg_color}; }}
        ::-webkit-scrollbar-thumb {{ background: {border_col}; border-radius: 3px; }}
    </style>
    """, unsafe_allow_html=True)

    return "plotly_dark" if mode == 'Dark' else "plotly_white"


# ============================================================
#                   HEADER COMPONENT
# ============================================================
def render_elegant_header(mode):
    title_color = "#ffffff" if mode == 'Dark' else "#1a1a2e"
    subtitle_color = "#9ca3af" if mode == 'Dark' else "#64748b"
    badge_bg = "rgba(79,172,254,0.15)" if mode == 'Dark' else "rgba(59,130,246,0.1)"
    badge_text = "#4facfe" if mode == 'Dark' else "#3b82f6"

    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 36px; padding-top: 16px;">
        <div style="
            display: inline-flex; align-items: center; justify-content: center;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            width: 72px; height: 72px; border-radius: 18px;
            box-shadow: 0 8px 24px rgba(79,172,254,0.35);
            margin-bottom: 12px;
        "><span style="font-size: 36px;">💡</span></div>
        <h1 style="
            color: {title_color}; font-family: 'Inter','Helvetica Neue',sans-serif;
            font-weight: 800; font-size: 2.6rem; margin: 0; letter-spacing: -1px;
        ">AnaText</h1>
        <p style="color: {subtitle_color}; font-size: 1.05rem; margin-top: 6px; font-weight: 400; letter-spacing: 0.3px;">
            Platform Analisis Teks Berbasis AI
        </p>
        <div style="margin-top: 10px;">
            <span style="background: {badge_bg}; color: {badge_text}; padding: 4px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 600;">
                ✨ Sentiment · Clustering · NER · N-Gram · Network · Entity Network
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
#              CORE ANALYSIS FUNCTIONS
# ============================================================

def clean_text(text, remove_sw, use_lemma, case_folding, stopwords_list, stemmer):
    """Membersihkan teks: lowering, punctuation removal, stopwords, stemming."""
    if not isinstance(text, str):
        return ""
    if case_folding:
        text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)          # hapus URL
    text = re.sub(r'@\w+', '', text)                       # hapus mention
    text = re.sub(r'#\w+', '', text)                       # hapus hashtag (simbol)
    text = re.sub(r'\d+', '', text)                         # hapus angka
    text = re.sub(r'[^\w\s]', '', text)                     # hapus punctuation
    text = re.sub(r'\s+', ' ', text).strip()                # hapus spasi berlebih
    tokens = text.split()
    if remove_sw:
        tokens = [w for w in tokens if w not in stopwords_list]
    if use_lemma and stemmer:
        tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)


# 1. Analisis Sentimen (AI) — dengan batching untuk efisiensi
def get_sentiment_ai(client, model, text_list, batch_size=5):
    """Menganalisis sentimen secara batch untuk mengurangi jumlah API call."""
    results = []
    progress_bar = st.progress(0)
    total = len(text_list)

    for i in range(0, total, batch_size):
        batch = text_list[i:i + batch_size]
        batch_texts = []
        for idx, text in enumerate(batch):
            if not text.strip():
                batch_texts.append(f"{idx + 1}. [kosong]")
            else:
                truncated = text[:500]
                batch_texts.append(f"{idx + 1}. {truncated}")

        numbered_texts = "\n".join(batch_texts)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": (
                        "Anda adalah analis sentimen. Klasifikasikan sentimen setiap teks: "
                        "Positif, Negatif, atau Netral. "
                        "Jawab HANYA dengan format JSON array, contoh: [\"Positif\",\"Negatif\",\"Netral\"]. "
                        "Tanpa penjelasan apapun."
                    )},
                    {"role": "user", "content": f"Klasifikasikan sentimen teks berikut:\n{numbered_texts}"}
                ],
                temperature=0,
                max_tokens=200
            )
            raw = response.choices[0].message.content.strip()
            # Coba parse JSON
            try:
                batch_results = json.loads(raw)
                # Normalisasi
                normalized = []
                for s in batch_results:
                    s_lower = s.lower().strip()
                    if "positif" in s_lower or "positive" in s_lower:
                        normalized.append("Positif")
                    elif "negatif" in s_lower or "negative" in s_lower:
                        normalized.append("Negatif")
                    else:
                        normalized.append("Netral")
                # Pastikan jumlah sesuai batch
                while len(normalized) < len(batch):
                    normalized.append("Netral")
                results.extend(normalized[:len(batch)])
            except json.JSONDecodeError:
                # Fallback: parse per baris
                for text in batch:
                    if not text.strip():
                        results.append("Netral")
                    else:
                        try:
                            resp = client.chat.completions.create(
                                model=model,
                                messages=[
                                    {"role": "system", "content": "Klasifikasikan sentimen: Positif, Negatif, atau Netral. Jawab 1 kata saja."},
                                    {"role": "user", "content": text}
                                ],
                                temperature=0, max_tokens=10
                            )
                            s = resp.choices[0].message.content.strip().replace(".", "")
                            if "positif" in s.lower():
                                results.append("Positif")
                            elif "negatif" in s.lower():
                                results.append("Negatif")
                            else:
                                results.append("Netral")
                        except Exception:
                            results.append("Error")
        except Exception:
            results.extend(["Error"] * len(batch))

        progress_bar.progress(min((i + len(batch)) / total, 1.0))

    progress_bar.empty()
    return results[:total]


# 2. Penamaan Topik (AI)
def get_topic_name_ai(client, model, keywords):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Berikan nama topik singkat (2-4 kata) dari keyword ini. Jawab hanya nama topik, tanpa tanda kutip."},
                {"role": "user", "content": f"Keywords: {', '.join(keywords)}"}
            ],
            temperature=0.3, max_tokens=30
        )
        return response.choices[0].message.content.strip().replace('"', '').replace("'", "")
    except Exception:
        return "Topik Umum"


# 3. N-Gram Analysis
def get_ngrams(text_series, n=2, top_k=10):
    """Menghasilkan frekuensi N-Gram tertinggi."""
    try:
        vec = CountVectorizer(ngram_range=(n, n)).fit(text_series)
        bag_of_words = vec.transform(text_series)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:top_k]
    except ValueError:
        return []


# 4. NER Analysis (AI)
def get_ner_ai(client, model, text_full):
    """Mengekstrak Named Entities menggunakan GPT-4.1."""
    try:
        text_sample = text_full[:15000]
        prompt = f"""
        Anda adalah Ahli Bahasa Indonesia dan Inggris. Tugas Anda adalah mengekstrak Entitas Penting (Named Entity Recognition) dari teks berikut.

        Kategorikan ke dalam 3 jenis:
        1. Person (Nama Tokoh/Orang)
        2. Organization (Nama Instansi/Lembaga/Perusahaan)
        3. Location (Nama Tempat/Kota/Negara/Lokasi)

        Output HARUS dalam format JSON murni tanpa markdown, dengan struktur:
        {{
            "Person": ["nama1", "nama2"],
            "Organization": ["org1", "org2"],
            "Location": ["loc1", "loc2"]
        }}

        Pastikan entitas unik (tidak duplikat) dan relevan.

        TEKS:
        {text_sample}
        """
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful NLP assistant that outputs only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"Person": [], "Organization": [], "Location": [], "Error": str(e)}


# 5. Comprehensive AI Summary (ENHANCED)
def generate_ai_summary(client, model, df, sc, topic_details, ner_results, text_type, language):
    """Menghasilkan Executive Summary yang komprehensif dan terstruktur."""
    topics_str = "\n".join([f"  - Topik '{t['Topik']}': kata kunci [{t['Keywords']}]" for t in topic_details])

    ner = ner_results or {}
    persons = ner.get('Person', [])[:8]
    orgs = ner.get('Organization', [])[:8]
    locs = ner.get('Location', [])[:8]

    # Hitung cross-tab sentimen per topik
    cross_data = []
    for topic in df['Topik'].unique():
        subset = df[df['Topik'] == topic]
        svc = subset['Sentimen'].value_counts().to_dict()
        cross_data.append(f"  - {topic}: Positif={svc.get('Positif', 0)}, Negatif={svc.get('Negatif', 0)}, Netral={svc.get('Netral', 0)}")
    cross_str = "\n".join(cross_data)

    # Hitung kata dominan
    all_words = " ".join(df['Teks_Clean']).split()
    top_words = Counter(all_words).most_common(15)
    top_words_str = ", ".join([f"{w} ({c}x)" for w, c in top_words])

    # Rasio sentimen
    total = len(df)
    pos_pct = round(sc.get('Positif', 0) / total * 100, 1) if total else 0
    neg_pct = round(sc.get('Negatif', 0) / total * 100, 1) if total else 0
    net_pct = round(sc.get('Netral', 0) / total * 100, 1) if total else 0

    # Bigram teratas
    bigrams = get_ngrams(df['Teks_Clean'], n=2, top_k=5)
    bigram_str = ", ".join([f"'{b[0]}' ({b[1]}x)" for b in bigrams]) if bigrams else "tidak cukup data"

    prompt = f"""Anda adalah **Senior Data Analyst** berpengalaman di bidang Text Mining & NLP.
Tulis sebuah **Laporan Analisis Eksekutif** yang komprehensif dan profesional berdasarkan data di bawah ini.

═══════════════ DATA INPUT ═══════════════

📊 METADATA
- Tipe Teks: {text_type}
- Bahasa: {language}
- Total Dokumen Dianalisis: {total}
- Tanggal Analisis: {datetime.now().strftime('%d %B %Y')}

🎭 DISTRIBUSI SENTIMEN
- Positif: {sc.get('Positif', 0)} dokumen ({pos_pct}%)
- Negatif: {sc.get('Negatif', 0)} dokumen ({neg_pct}%)
- Netral: {sc.get('Netral', 0)} dokumen ({net_pct}%)

📂 TOPIK & KATA KUNCI (dari K-Means Clustering + TF-IDF)
{topics_str}

📊 SENTIMEN PER TOPIK (Cross-Tabulation)
{cross_str}

🔠 KATA DOMINAN (Frekuensi Tertinggi)
{top_words_str}

🔗 FRASA DOMINAN (Bigram)
{bigram_str}

👤 ENTITAS TERDETEKSI (NER)
- Person: {', '.join(persons) if persons else 'tidak terdeteksi'}
- Organization: {', '.join(orgs) if orgs else 'tidak terdeteksi'}
- Location: {', '.join(locs) if locs else 'tidak terdeteksi'}

═══════════════ INSTRUKSI OUTPUT ═══════════════

Tulis laporan analisis dengan struktur berikut. Gunakan bahasa Indonesia yang profesional, data-driven, dan insightful.

ATURAN PENTING: Setiap bagian (section) di bawah ini WAJIB ditulis sepanjang **200 hingga 250 kata**. Jangan kurang dari 200 kata dan jangan lebih dari 250 kata per bagian. Tulis secara mendalam, elaboratif, dan kaya analisis untuk memenuhi target jumlah kata tersebut.

## 📋 Ringkasan Eksekutif
(200-250 kata) Tulis paragraf pembuka komprehensif yang merangkum seluruh temuan analisis. Sebutkan jumlah dokumen, distribusi sentimen dominan beserta persentasenya, topik-topik utama yang teridentifikasi, entitas kunci, dan pola linguistik yang menonjol. Berikan gambaran menyeluruh sehingga pembaca langsung memahami lanskap data tanpa harus membaca seluruh laporan.

## 🎭 Analisis Sentimen
(200-250 kata) Jelaskan distribusi sentimen secara detail dan dominansinya. Analisis MENGAPA sentimen tertentu dominan berdasarkan konteks topik dan kata kunci. Hubungkan sentimen dengan tipe teks ({text_type}). Berikan interpretasi proporsi sentimen (apakah sehat/mengkhawatirkan?). Bandingkan rasio positif-negatif dan diskusikan implikasinya. Jelaskan potensi faktor penyebab sentimen netral jika signifikan.

## 📂 Analisis Tematik (Topik)
(200-250 kata) Jelaskan setiap topik yang teridentifikasi beserta makna dan konteksnya berdasarkan kata kunci yang membentuknya. Identifikasi topik paling krusial/dominan dari segi jumlah dokumen. Jelaskan hubungan dan keterkaitan antar topik jika ada. Diskusikan apakah topik-topik tersebut saling mendukung atau bertentangan, dan apa implikasinya terhadap narasi keseluruhan data.

## 🔀 Sentimen × Topik (Cross-Analysis)
(200-250 kata) Analisis topik mana yang paling positif vs paling negatif berdasarkan data cross-tabulation. Identifikasi adakah topik kontroversial yang sentimennya terpolarisasi (campuran kuat positif dan negatif). Tentukan topik mana yang perlu perhatian dan prioritas penanganan. Berikan penjelasan mendalam tentang mengapa topik tertentu mendapat respons sentimen yang berbeda.

## 👤 Analisis Entitas (NER)
(200-250 kata) Jelaskan siapa saja tokoh/orang kunci yang terdeteksi dan apa relevansinya dalam konteks data. Analisis organisasi/lembaga yang muncul dan perannya. Diskusikan lokasi yang terdeteksi dan hubungannya dengan narasi data. Kaitkan entitas-entitas ini dengan topik dan sentimen yang ditemukan. Jelaskan implikasi kehadiran entitas-entitas tersebut.

## 🔗 Pola Linguistik
(200-250 kata) Interpretasikan kata-kata dominan dan frasa (bigram) yang paling sering muncul. Jelaskan apa yang bisa disimpulkan dari pola penggunaan kata dan frasa tersebut. Identifikasi pola bahasa apa yang muncul secara konsisten. Diskusikan apakah ada kosakata spesifik yang menunjukkan kecenderungan tertentu dalam data. Hubungkan temuan linguistik dengan topik dan sentimen.

## 📌 Kesimpulan & Rekomendasi Strategis
(200-250 kata) Tuliskan 3-5 poin kesimpulan kunci yang merangkum seluruh temuan. Berikan 3-5 rekomendasi aksi konkret dan spesifik berdasarkan temuan data. Tentukan prioritas tindakan dari yang paling mendesak. Sertakan justifikasi berbasis data untuk setiap rekomendasi yang diberikan.

Gunakan **bold** untuk poin penting. Pastikan setiap klaim didukung oleh data kuantitatif yang tersedia.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "Anda adalah konsultan data analytics senior. "
                    "Tulis laporan yang mendalam, terstruktur, dan actionable. "
                    "Selalu kaitkan temuan dengan data kuantitatif yang diberikan. "
                    "WAJIB: Setiap bagian/section harus ditulis sepanjang 200-250 kata. "
                    "Tidak boleh kurang dari 200 kata per bagian. Elaborasi secara mendalam."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=6000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Gagal menghasilkan ringkasan: {str(e)}"


# 5b. Tab-Specific AI Insight Generator
def generate_tab_insight(client, model, tab_key, context_data):
    """Menghasilkan ringkasan AI spesifik untuk setiap tab analisis."""

    tab_prompts = {
        "sentimen": {
            "title": "Analisis Sentimen",
            "instruction": """Analisis distribusi sentimen secara mendalam. Jelaskan:
1. Sentimen mana yang dominan dan mengapa berdasarkan konteks data
2. Proporsi dan rasio antar sentimen (positif vs negatif vs netral)
3. Apakah distribusi sentimen ini sehat atau mengkhawatirkan
4. Faktor-faktor yang mungkin mempengaruhi dominasi sentimen tertentu
5. Implikasi dan rekomendasi berdasarkan pola sentimen yang ditemukan"""
        },
        "topik": {
            "title": "Analisis Tematik (Topik)",
            "instruction": """Analisis topik-topik yang teridentifikasi secara mendalam. Jelaskan:
1. Makna dan konteks setiap topik berdasarkan kata kunci pembentuknya
2. Topik mana yang paling dominan/krusial dari segi jumlah dokumen
3. Hubungan dan keterkaitan antar topik
4. Apakah topik-topik saling mendukung atau bertentangan
5. Implikasi tematik terhadap narasi keseluruhan data"""
        },
        "cross": {
            "title": "Cross-Analysis: Sentimen × Topik",
            "instruction": """Analisis hubungan silang antara sentimen dan topik. Jelaskan:
1. Topik mana yang paling positif vs paling negatif
2. Adakah topik kontroversial yang sentimennya terpolarisasi
3. Topik mana yang perlu perhatian dan prioritas penanganan
4. Mengapa topik tertentu mendapat respons sentimen yang berbeda
5. Pola dan anomali dalam distribusi sentimen per topik"""
        },
        "katakunci": {
            "title": "Analisis Kata Kunci",
            "instruction": """Analisis kata kunci dan pola frekuensi kata. Jelaskan:
1. Kata-kata dengan skor TF-IDF tertinggi dan signifikansinya
2. Perbedaan antara kata frekuensi tinggi vs kata TF-IDF tinggi
3. Apa yang bisa disimpulkan dari pola penggunaan kata
4. Kosakata spesifik yang menunjukkan kecenderungan tertentu
5. Hubungan kata kunci dominan dengan topik dan sentimen"""
        },
        "ngram": {
            "title": "Analisis N-Gram (Frasa)",
            "instruction": """Analisis frasa (N-Gram) yang sering muncul. Jelaskan:
1. Makna dan konteks dari bigram dan trigram teratas
2. Pola frasa apa yang muncul secara konsisten
3. Apakah frasa-frasa tersebut menunjukkan narasi atau opini tertentu
4. Perbandingan antara bigram, trigram, dan fourgram
5. Hubungan frasa dominan dengan topik dan sentimen yang ditemukan"""
        },
        "ner": {
            "title": "Analisis Entitas (NER)",
            "instruction": """Analisis entitas yang terdeteksi. Jelaskan:
1. Siapa saja tokoh/orang kunci dan relevansinya dalam konteks data
2. Organisasi/lembaga yang muncul dan perannya
3. Lokasi yang terdeteksi dan hubungannya dengan narasi data
4. Keterkaitan antar entitas (person-organization-location)
5. Implikasi kehadiran entitas-entitas tersebut terhadap temuan analisis"""
        },
        "network": {
            "title": "Text Network Analysis",
            "instruction": """Analisis jaringan teks (hubungan topik-kata kunci). Jelaskan:
1. Struktur jaringan: topik mana yang paling terhubung dengan kata kunci beragam
2. Apakah ada kata kunci yang menjembatani beberapa topik sekaligus
3. Kepadatan dan pola konektivitas jaringan
4. Topik mana yang paling terisolasi vs paling terintegrasi
5. Implikasi struktur jaringan terhadap koherensi tematik data"""
        },
        "entity_network": {
            "title": "Jaringan Ko-kemunculan Entitas",
            "instruction": """Analisis jaringan ko-kemunculan entitas. Jelaskan:
1. Entitas mana yang paling sentral (paling banyak terhubung)
2. Pasangan entitas yang paling sering muncul bersama dan maknanya
3. Pola hubungan antar tipe entitas (Person-Organization-Location)
4. Kepadatan jaringan dan apa artinya bagi data
5. Insight strategis dari pola ko-kemunculan entitas"""
        },
        "doc_keywords": {
            "title": "Analisis Kata Kunci per Dokumen",
            "instruction": """Analisis pola kata kunci di level dokumen individual. Jelaskan:
1. Apakah kata kunci antar dokumen bervariasi atau cenderung seragam
2. Perbandingan pola TF-IDF vs frekuensi mentah (Raw Count)
3. Dokumen mana yang memiliki kata kunci paling unik/berbeda
4. Hubungan antara kata kunci dokumen dengan topik klasternya
5. Insight mengenai keragaman konten antar dokumen"""
        },
        "linguistik": {
            "title": "Pola Linguistik",
            "instruction": """Analisis pola linguistik keseluruhan data. Jelaskan:
1. Interpretasi kata-kata dominan dan frasa paling sering muncul
2. Pola bahasa apa yang muncul secara konsisten
3. Kosakata spesifik yang menunjukkan kecenderungan tertentu
4. Gaya bahasa dan register yang digunakan dalam corpus
5. Hubungan temuan linguistik dengan topik dan sentimen"""
        },
        "kesimpulan": {
            "title": "Kesimpulan & Rekomendasi Strategis",
            "instruction": """Tuliskan kesimpulan komprehensif dan rekomendasi strategis. Jelaskan:
1. 3-5 poin kesimpulan kunci yang merangkum seluruh temuan
2. 3-5 rekomendasi aksi konkret dan spesifik berdasarkan data
3. Prioritas tindakan dari yang paling mendesak
4. Justifikasi berbasis data untuk setiap rekomendasi
5. Potensi risiko jika temuan-temuan ini tidak ditindaklanjuti"""
        }
    }

    tab_info = tab_prompts.get(tab_key, {})
    title = tab_info.get("title", "Analisis")
    instruction = tab_info.get("instruction", "Berikan analisis mendalam.")

    prompt = f"""Anda adalah **Senior Data Analyst** berpengalaman di bidang Text Mining & NLP.
Berikan ringkasan dan analisis mendalam untuk bagian **{title}** berdasarkan data berikut.

═══════════════ DATA KONTEKS ═══════════════
{context_data}

═══════════════ INSTRUKSI ═══════════════
{instruction}

ATURAN:
- Tulis dalam bahasa Indonesia yang profesional dan data-driven
- Panjang: 200-300 kata, padat dan elaboratif
- Gunakan **bold** untuk poin penting
- Setiap klaim HARUS didukung oleh data kuantitatif yang tersedia
- Berikan insight yang actionable, bukan sekadar deskripsi
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "Anda adalah konsultan data analytics senior. "
                    "Tulis analisis yang mendalam, terstruktur, dan actionable. "
                    "Selalu kaitkan temuan dengan data kuantitatif yang diberikan."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Gagal menghasilkan ringkasan: {str(e)}"


def render_tab_insight_ui(client, model, tab_key, context_data, label=""):
    """Render UI tombol dan hasil ringkasan AI di setiap tab."""
    st.markdown("---")
    insight_label = label or tab_key.replace("_", " ").title()

    if tab_key in st.session_state.tab_insights:
        st.markdown(f"#### 🤖 Ringkasan AI: {insight_label}")
        st.markdown(st.session_state.tab_insights[tab_key])
        if st.button(f"🔄 Regenerasi Ringkasan", key=f"regen_{tab_key}"):
            del st.session_state.tab_insights[tab_key]
            st.rerun()
    else:
        if st.button(f"✨ Generate Ringkasan AI: {insight_label}", key=f"gen_{tab_key}", type="primary"):
            with st.spinner(f"🤖 AI sedang menganalisis {insight_label}..."):
                result = generate_tab_insight(client, model, tab_key, context_data)
                st.session_state.tab_insights[tab_key] = result
                st.rerun()


# 6. Network Graph (Enhanced with Plotly for interactivity)
def generate_text_network_plotly(topic_details, theme_mode):
    """Membuat network graph interaktif menggunakan Plotly."""
    G = nx.Graph()

    for idx, detail in enumerate(topic_details):
        topic_name = detail['Topik']
        keywords = detail['Keywords'].split(', ')
        color = TOPIC_COLORS[idx % len(TOPIC_COLORS)]
        G.add_node(topic_name, size=30, color=color, ntype='topic')
        for kw in keywords[:8]:
            if not G.has_node(kw):
                G.add_node(kw, size=12, color=color, ntype='keyword')
            G.add_edge(topic_name, kw)

    pos = nx.spring_layout(G, k=0.6, seed=42, iterations=50)

    # Edges
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(150,150,150,0.4)'),
        hoverinfo='none', mode='lines'
    )

    # Nodes
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_color.append(G.nodes[node]['color'])
        node_size.append(G.nodes[node]['size'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        textfont=dict(size=9, color='white' if theme_mode == 'Dark' else 'black'),
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color='rgba(255,255,255,0.3)'))
    )

    bg_color = '#0e1117' if theme_mode == 'Dark' else '#f8f9fb'
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        paper_bgcolor=bg_color,
                        plot_bgcolor=bg_color,
                        margin=dict(b=20, l=20, r=20, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    ))
    return fig


# 7. Matplotlib Network (fallback, preserved from original)
def generate_text_network(topic_details, theme_mode):
    G = nx.Graph()
    labels = {}
    for idx, detail in enumerate(topic_details):
        topic_name = detail['Topik']
        keywords = detail['Keywords'].split(', ')
        cluster_color = TOPIC_COLORS[idx % len(TOPIC_COLORS)]
        G.add_node(topic_name, size=2000, color=cluster_color, type='topic')
        labels[topic_name] = topic_name
        for kw in keywords:
            if not G.has_node(kw):
                G.add_node(kw, size=500, color=cluster_color, type='keyword')
                labels[kw] = kw
            G.add_edge(topic_name, kw)

    fig_bg = '#0e1117' if theme_mode == 'Dark' else '#ffffff'
    plt.figure(figsize=(12, 8), facecolor=fig_bg)
    pos = nx.spring_layout(G, k=0.5, seed=42)
    final_colors = [G.nodes[n]['color'] for n in G.nodes()]
    node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=final_colors, alpha=0.9, node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray')
    font_color = 'white' if theme_mode == 'Dark' else 'black'
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color=font_color, font_weight='bold')
    plt.axis('off')
    return plt


# 8. Topic Cluster Scatter (PCA)
def generate_cluster_scatter(tfidf_matrix, labels, topic_names, template):
    """Visualisasi 2D cluster menggunakan PCA."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(tfidf_matrix.toarray())
    df_scatter = pd.DataFrame({
        'PC1': coords[:, 0],
        'PC2': coords[:, 1],
        'Topik': [topic_names.get(l, f"Topik {l}") for l in labels]
    })
    fig = px.scatter(
        df_scatter, x='PC1', y='PC2', color='Topik',
        color_discrete_sequence=TOPIC_COLORS,
        template=template,
        title="Distribusi Dokumen per Cluster (PCA 2D)"
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(height=450)
    return fig


# 9. Entity Co-occurrence Network
def build_entity_cooccurrence(df_texts, ner_results, client, model):
    """Membangun jaringan ko-kemunculan entitas berdasarkan kemunculan bersama dalam dokumen."""
    if not ner_results:
        return None, None

    all_entities = []
    entity_type_map = {}
    for etype in ['Person', 'Organization', 'Location']:
        for ent in ner_results.get(etype, []):
            ent_clean = ent.strip()
            if ent_clean:
                all_entities.append(ent_clean)
                entity_type_map[ent_clean] = etype

    if len(all_entities) < 2:
        return None, None

    # Hitung ko-kemunculan: dua entitas muncul dalam dokumen yang sama
    entity_freq = Counter()
    cooccurrence = Counter()

    for text in df_texts:
        text_lower = text.lower() if isinstance(text, str) else ""
        present = [e for e in all_entities if e.lower() in text_lower]
        for e in present:
            entity_freq[e] += 1
        for pair in combinations(sorted(set(present)), 2):
            cooccurrence[pair] += 1

    # Bangun graph
    G = nx.Graph()

    type_colors = {
        'Person': '#4ECDC4',
        'Organization': '#45B7D1',
        'Location': '#FFA07A'
    }

    for ent in all_entities:
        if entity_freq.get(ent, 0) > 0:
            G.add_node(ent,
                       etype=entity_type_map.get(ent, 'Unknown'),
                       freq=entity_freq.get(ent, 1),
                       color=type_colors.get(entity_type_map.get(ent), '#cccccc'))

    for (e1, e2), weight in cooccurrence.items():
        if weight > 0:
            G.add_edge(e1, e2, weight=weight)

    return G, entity_type_map


def render_entity_network_plotly(G, entity_type_map, theme_mode):
    """Render jaringan ko-kemunculan entitas menggunakan Plotly."""
    if G is None or len(G.nodes()) == 0:
        return None

    pos = nx.spring_layout(G, k=1.2, seed=42, iterations=80)

    type_colors = {
        'Person': '#4ECDC4',
        'Organization': '#45B7D1',
        'Location': '#FFA07A'
    }

    # Edges
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        w = edge[2].get('weight', 1)
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=max(0.5, min(w * 1.5, 8)), color='rgba(150,150,150,0.5)'),
            hoverinfo='text',
            hovertext=f"{edge[0]} ↔ {edge[1]}: {w}x ko-kemunculan",
            mode='lines',
            showlegend=False
        ))

    # Nodes per type for legend
    fig = go.Figure()
    for trace in edge_traces:
        fig.add_trace(trace)

    for etype, color in type_colors.items():
        nodes_of_type = [n for n in G.nodes() if G.nodes[n].get('etype') == etype]
        if not nodes_of_type:
            continue
        nx_list = [pos[n][0] for n in nodes_of_type]
        ny_list = [pos[n][1] for n in nodes_of_type]
        sizes = [max(15, min(G.nodes[n].get('freq', 1) * 8, 50)) for n in nodes_of_type]
        hover = [f"<b>{n}</b><br>Tipe: {etype}<br>Frekuensi: {G.nodes[n].get('freq', 1)}<br>Koneksi: {G.degree(n)}" for n in nodes_of_type]

        fig.add_trace(go.Scatter(
            x=nx_list, y=ny_list,
            mode='markers+text',
            name=f"🔹 {etype} ({len(nodes_of_type)})",
            text=nodes_of_type,
            textposition="top center",
            textfont=dict(size=9, color='white' if theme_mode == 'Dark' else '#1a1a2e'),
            hoverinfo='text',
            hovertext=hover,
            marker=dict(size=sizes, color=color, line=dict(width=1.5, color='rgba(255,255,255,0.6)'),
                        opacity=0.9)
        ))

    bg_color = '#0e1117' if theme_mode == 'Dark' else '#f8f9fb'
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5,
                    font=dict(size=11)),
        hovermode='closest',
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        margin=dict(b=20, l=20, r=20, t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    return fig


def render_entity_network_matplotlib(G, entity_type_map, theme_mode):
    """Render jaringan ko-kemunculan entitas menggunakan Matplotlib."""
    if G is None or len(G.nodes()) == 0:
        return None

    type_colors = {
        'Person': '#4ECDC4',
        'Organization': '#45B7D1',
        'Location': '#FFA07A'
    }

    fig_bg = '#0e1117' if theme_mode == 'Dark' else '#ffffff'
    fig, ax = plt.subplots(figsize=(14, 9), facecolor=fig_bg)
    ax.set_facecolor(fig_bg)

    pos = nx.spring_layout(G, k=1.2, seed=42, iterations=80)

    # Draw edges with varying width
    edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    widths = [max(0.5, (w / max_w) * 4) for w in edge_weights]
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.4, edge_color='gray', ax=ax)

    # Draw nodes per type
    for etype, color in type_colors.items():
        nodes = [n for n in G.nodes() if G.nodes[n].get('etype') == etype]
        if not nodes:
            continue
        sizes = [max(300, min(G.nodes[n].get('freq', 1) * 150, 2000)) for n in nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color,
                               node_size=sizes, alpha=0.9, ax=ax, label=f"{etype} ({len(nodes)})")

    font_color = 'white' if theme_mode == 'Dark' else 'black'
    nx.draw_networkx_labels(G, pos, font_size=8, font_color=font_color, font_weight='bold', ax=ax)

    legend_color = 'white' if theme_mode == 'Dark' else 'black'
    leg = ax.legend(loc='upper left', fontsize=9, framealpha=0.7)
    for text in leg.get_texts():
        text.set_color(legend_color)

    ax.axis('off')
    plt.tight_layout()
    return fig


# ============================================================
#              STOPWORDS MANAGER (MODAL)
# ============================================================
def show_stopwords_manager():
    col1, col2 = st.columns([3, 1])
    with col1:
        new_word = st.text_input("Tambah kata:", label_visibility="collapsed", placeholder="Ketik kata...")
    with col2:
        if st.button("Tambah"):
            if new_word and new_word.lower() not in st.session_state.stop_words:
                st.session_state.stop_words.append(new_word.lower())
                st.rerun()
    current = st.multiselect("Daftar Stop Words:", st.session_state.stop_words, default=st.session_state.stop_words)
    if len(current) != len(st.session_state.stop_words):
        st.session_state.stop_words = current
        st.rerun()


if hasattr(st, "dialog"):
    @st.dialog("Kelola Stop Words")
    def open_stopwords_modal():
        show_stopwords_manager()
        st.button("Tutup", on_click=st.rerun)
else:
    def open_stopwords_modal():
        pass


# ============================================================
#              INFORMASI APLIKASI (DIALOG)
# ============================================================
def show_app_info():
    """Menampilkan informasi lengkap tentang AnaText."""
    st.markdown("""
### 💡 Tentang AnaText

**AnaText** adalah platform analisis teks berbasis kecerdasan buatan (AI) yang dikembangkan untuk membantu peneliti, analis data, dan praktisi dalam mengekstrak insight mendalam dari data teks secara otomatis. Platform ini mengintegrasikan pendekatan Natural Language Processing (NLP) tradisional dengan kemampuan Large Language Model (LLM) modern.

---

### 📋 Fitur Utama

| No | Fitur | Deskripsi |
|----|-------|-----------|
| 1 | **Analisis Sentimen** | Klasifikasi sentimen otomatis (Positif/Negatif/Netral) menggunakan AI |
| 2 | **Topic Modeling (Clustering)** | Pengelompokan dokumen ke dalam topik-topik menggunakan K-Means |
| 3 | **Word Cloud per Topik** | Visualisasi kata dominan untuk setiap klaster topik |
| 4 | **Cross-Analysis Sentimen × Topik** | Analisis silang distribusi sentimen di setiap topik |
| 5 | **Analisis Kata Kunci** | Identifikasi kata kunci berdasarkan skor TF-IDF dan frekuensi |
| 6 | **Analisis N-Gram** | Deteksi frasa berulang (Bigram, Trigram, Fourgram) |
| 7 | **Named Entity Recognition (NER)** | Ekstraksi entitas (Person, Organization, Location) berbasis AI |
| 8 | **Text Network Analysis** | Visualisasi jaringan hubungan topik dan kata kunci |
| 9 | **Jaringan Ko-kemunculan Entitas** | Jaringan entitas yang muncul bersama dalam dokumen |
| 10 | **Kata Kunci per Dokumen** | Top 10 kata kunci (TF-IDF & Raw Count) untuk setiap dokumen |
| 11 | **Ringkasan AI per Tab** | AI-generated insight untuk setiap tab analisis |
| 12 | **Laporan Eksekutif Komprehensif** | Laporan analisis terstruktur oleh AI |

---

### ⚙️ Algoritma & Teknologi

| Komponen | Teknologi/Algoritma |
|----------|---------------------|
| Preprocessing Teks | Tokenisasi, Case Folding, Stop Words Removal, Stemming (Sastrawi) |
| Representasi Fitur | TF-IDF (Term Frequency–Inverse Document Frequency) |
| Clustering/Topik | K-Means Clustering |
| Reduksi Dimensi | PCA (Principal Component Analysis) |
| Sentimen & NER | OpenAI GPT-4.1 / GPT-4.1-mini |
| Ringkasan AI | OpenAI GPT-4.1 (generatif) |
| Analisis Frasa | CountVectorizer (N-Gram) |
| Network Analysis | NetworkX (Graph Theory) |
| Visualisasi | Plotly, Matplotlib, WordCloud |
| Framework Web | Streamlit |
| Bahasa yang Didukung | Indonesia (dengan Sastrawi) dan Inggris |

---

### 📖 Format Sitasi

Jika Anda menggunakan AnaText dalam penelitian atau publikasi, silakan gunakan format sitasi berikut:

**APA Style (7th Edition):**

> Suwarno. (2026). *AnaText: Platform analisis teks berbasis kecerdasan buatan* (Versi 2.5.0) [Perangkat lunak]. https://anatext.streamlit.app/

**IEEE Style:**

> Suwarno, "AnaText: Platform Analisis Teks Berbasis Kecerdasan Buatan," versi 2.5.0, Jan. 2026. [Online]. Available: https://anatext.streamlit.app/

---

📌 **Pengembang:** Suwarno
📅 **Rilis Pertama:** Januari 2026
🌐 **Akses:** [https://anatext.streamlit.app/](https://anatext.streamlit.app/)
""")


if hasattr(st, "dialog"):
    @st.dialog("ℹ️ Informasi AnaText", width="large")
    def open_info_modal():
        show_app_info()
        st.button("Tutup", on_click=st.rerun)
else:
    def open_info_modal():
        pass


# ============================================================
#                      SIDEBAR
# ============================================================
with st.sidebar:
    st.title("⚙️ Pengaturan")
    theme_mode = st.radio("Tema Tampilan", ["Light", "Dark"], horizontal=True)
    plotly_template = inject_custom_css(theme_mode)

    st.divider()
    st.subheader("🌐 Bahasa & Teks")
    language = st.selectbox("Bahasa Teks", ["Indonesia", "Inggris"])
    text_type = st.selectbox("Tipe Teks", ["Umum", "Ulasan Produk", "Berita/Artikel", "Media Sosial", "Akademik"])

    st.divider()
    st.subheader("🤖 Model AI")
    num_clusters_input = st.slider("Jumlah Topik (Klaster)", 2, 10, 5)

    st.divider()
    st.subheader("🔧 Preprocessing")
    check_sw = st.checkbox("Hapus Stop Words", value=True, help="Menghapus kata umum yang minim makna.")
    check_lemma = st.checkbox("Aktifkan Lemmatization", value=True, help="Mengubah kata berimbuhan menjadi kata dasar.")
    check_lower = st.checkbox("Case Folding (lowercase)", value=True, help="Mengubah semua teks ke huruf kecil.")
    check_url = st.checkbox("Hapus URL & Mention", value=True, help="Membersihkan URL, mention (@), dan hashtag.")

    if hasattr(st, "dialog"):
        if st.button("📝 Kelola Stop Words", use_container_width=True):
            open_stopwords_modal()
    else:
        with st.expander("📝 Kelola Stop Words"):
            show_stopwords_manager()

    st.markdown("---")

    # Tombol Informasi Aplikasi
    if hasattr(st, "dialog"):
        if st.button("ℹ️ Informasi AnaText", use_container_width=True):
            open_info_modal()
    else:
        with st.expander("ℹ️ Informasi AnaText"):
            show_app_info()

    st.markdown("---")
    st.markdown(
        f'<div class="footer-text">Developed by <b>Suwarno</b><br>Powered by <b>GPT-4.1</b> & <b>GPT-4.1-mini</b><br>v{APP_VERSION}</div>',
        unsafe_allow_html=True
    )


# ============================================================
#                     MAIN UI
# ============================================================
render_elegant_header(theme_mode)

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = ""
client = OpenAI(api_key=api_key) if api_key else None
MODEL_FAST = "gpt-4.1-mini"   # Sentimen, Topik Naming, NER (cepat & hemat)
MODEL_SMART = "gpt-4.1"       # Summary Komprehensif (mendalam & detail)

# --- INPUT AREA ---
container_input = st.container()
with container_input:
    tab_upload, tab_text = st.tabs(["📂 Unggah Dokumen", "✍️ Teks Langsung"])
    input_text_list = []

    with tab_upload:
        st.info("📁 Format yang didukung: **.csv**, **.xlsx**, **.txt** — Maks. 10 MB")
        uploaded_file = st.file_uploader("Upload File", type=['csv', 'xlsx', 'txt'], label_visibility="collapsed")
        if uploaded_file:
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("⚠️ File terlalu besar (>10MB). Silakan kurangi ukuran file.")
            else:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        try:
                            df_u = pd.read_csv(uploaded_file, encoding='utf-8')
                        except UnicodeDecodeError:
                            uploaded_file.seek(0)
                            df_u = pd.read_csv(uploaded_file, encoding='latin-1')
                    elif uploaded_file.name.endswith('.xlsx'):
                        df_u = pd.read_excel(uploaded_file)
                    else:
                        try:
                            txt = uploaded_file.read().decode('utf-8')
                        except UnicodeDecodeError:
                            uploaded_file.seek(0)
                            txt = uploaded_file.read().decode('latin-1')
                        df_u = pd.DataFrame(txt.splitlines(), columns=['Teks'])

                    cols = [c for c in df_u.columns if df_u[c].dtype == 'object']
                    if cols:
                        t_col = st.selectbox("Pilih Kolom Teks:", cols)
                        input_text_list = df_u[t_col].dropna().astype(str).tolist()
                        st.success(f"✅ Berhasil dimuat: **{len(input_text_list)}** baris data dari kolom `{t_col}`")
                    else:
                        st.error("Tidak ada kolom bertipe teks.")
                except Exception as e:
                    st.error(f"Gagal membaca file: {str(e)}")

    with tab_text:
        dt = st.text_area("Tempel teks di sini (pisahkan dengan baris baru)...", height=150)
        if dt:
            input_text_list = [t for t in dt.split('\n') if t.strip()]
            if input_text_list:
                st.success(f"✅ **{len(input_text_list)}** baris teks terdeteksi.")


# ============================================================
#                     PROCESSING
# ============================================================
col_btn, _ = st.columns([1, 3])
with col_btn:
    run_analysis = st.button("🚀 Lakukan Analisis", type="primary", use_container_width=True)
if run_analysis:
    if not input_text_list:
        st.warning("⚠️ Data kosong. Silakan unggah file atau masukkan teks terlebih dahulu.")
    elif not client:
        st.error("🔑 API Key OpenAI belum dikonfigurasi. Tambahkan di Streamlit Secrets.")
    else:
        with st.spinner("🔄 Sedang memproses analisis lengkap..."):
            df = pd.DataFrame(input_text_list, columns=['Teks_Asli'])

            # --- Preprocessing ---
            status_text = st.empty()
            status_text.info("⏳ Tahap 1/4: Preprocessing teks...")

            factory = StemmerFactory()
            stemmer = factory.create_stemmer() if (language == "Indonesia" and check_lemma) else None

            clean_res = []
            pb = st.progress(0)
            for i, t in enumerate(df['Teks_Asli']):
                clean_res.append(clean_text(t, check_sw, check_lemma, check_lower, st.session_state.stop_words, stemmer))
                if i % 10 == 0 or i == len(df) - 1:
                    pb.progress((i + 1) / len(df))
            pb.empty()

            df['Teks_Clean'] = clean_res
            df = df[df['Teks_Clean'].str.strip() != ""].reset_index(drop=True)

            if len(df) < 2:
                st.error("Data terlalu sedikit setelah preprocessing. Minimal 2 dokumen diperlukan.")
                st.stop()

            # --- Clustering ---
            status_text.info("⏳ Tahap 2/4: Clustering topik dengan TF-IDF + K-Means...")
            k = min(num_clusters_input, len(df))
            if k < 2:
                k = 2

            vec = TfidfVectorizer(max_features=2000)
            tfidf = vec.fit_transform(df['Teks_Clean'])
            feats = vec.get_feature_names_out()

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(tfidf)
            df['Cluster_ID'] = kmeans.labels_

            topic_map = {}
            topic_data_list = []

            for i in range(k):
                center = kmeans.cluster_centers_[i]
                top_idx = center.argsort()[-10:][::-1]
                top_w = [feats[x] for x in top_idx]
                label = get_topic_name_ai(client, MODEL_FAST, top_w[:5]) if top_w else f"Topik {i + 1}"
                topic_map[i] = label
                topic_data_list.append({
                    'Nomor': i + 1,
                    'Topik': label,
                    'Keywords': ", ".join(top_w),
                    'Jumlah_Dokumen': int((df['Cluster_ID'] == i).sum())
                })

            df['Topik'] = df['Cluster_ID'].map(topic_map)

            # --- Sentiment AI ---
            status_text.info("⏳ Tahap 3/4: Analisis sentimen dengan AI...")
            df['Sentimen'] = get_sentiment_ai(client, MODEL_FAST, df['Teks_Asli'].tolist())

            # --- NER Analysis ---
            status_text.info("⏳ Tahap 4/4: Deteksi entitas (NER)...")
            full_text_sample = " ".join(df['Teks_Asli'].tolist())
            ner_result = get_ner_ai(client, MODEL_FAST, full_text_sample)

            status_text.empty()

            # --- Save to state ---
            st.session_state.data = df
            st.session_state.topic_details = topic_data_list
            st.session_state.ner_results = ner_result
            st.session_state.vectorizer = vec
            st.session_state.tfidf_matrix = tfidf
            st.session_state.kmeans = kmeans
            st.session_state.topic_map = topic_map
            st.session_state.analysis_done = True
            st.session_state.summary_cache = None
            st.session_state.tab_insights = {}
            st.rerun()


# ============================================================
#                     DASHBOARD
# ============================================================
if st.session_state.analysis_done and st.session_state.data is not None:
    df = st.session_state.data
    st.write("---")

    # --- HEADER METRICS ---
    sc = df['Sentimen'].value_counts().to_dict()
    total_docs = len(df)
    dominant_sentiment = max(sc, key=sc.get) if sc else "-"
    num_topics = df['Topik'].nunique()
    avg_words = round(df['Teks_Clean'].str.split().str.len().mean(), 1)

    # Custom styled metrics
    card_bg = "#1a1d24" if theme_mode == 'Dark' else "#ffffff"
    border_col = "#2d3139" if theme_mode == 'Dark' else "#e2e8f0"
    text_col = "#e8e8e8" if theme_mode == 'Dark' else "#1a1a2e"
    sub_col = "#9ca3af" if theme_mode == 'Dark' else "#64748b"

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="stat-card" style="border-top: 3px solid #4facfe;">
            <div class="stat-value" style="color: #4facfe;">📄 {total_docs}</div>
            <div class="stat-label">Total Dokumen</div></div>""", unsafe_allow_html=True)
    with m2:
        sent_color = SENTIMENT_COLORS.get(dominant_sentiment, '#6c757d')
        st.markdown(f"""<div class="stat-card" style="border-top: 3px solid {sent_color};">
            <div class="stat-value" style="color: {sent_color};">🎭 {dominant_sentiment}</div>
            <div class="stat-label">Sentimen Dominan</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="stat-card" style="border-top: 3px solid #BB8FCE;">
            <div class="stat-value" style="color: #BB8FCE;">📂 {num_topics}</div>
            <div class="stat-label">Topik Terdeteksi</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="stat-card" style="border-top: 3px solid #82E0AA;">
            <div class="stat-value" style="color: #82E0AA;">📏 {avg_words}</div>
            <div class="stat-label">Rata-rata Kata/Dok</div></div>""", unsafe_allow_html=True)

    st.write("")

    # --- TABS ---
    t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 = st.tabs([
        "📝 Ringkasan AI",
        "🎭 Sentimen",
        "📂 Topik",
        "🔀 Sentimen × Topik",
        "🔠 Kata Kunci",
        "🔗 N-Gram",
        "👤 Entitas (NER)",
        "🌐 Network",
        "🕸️ Jaringan Entitas",
        "📊 Kata Kunci/Dok"
    ])

    # ========================
    # TAB 1: RINGKASAN AI
    # ========================
    with t1:
        st.markdown("### 📝 Laporan Analisis Eksekutif")
        st.caption("AI akan menganalisis seluruh temuan dan menghasilkan laporan terstruktur yang komprehensif.")

        if st.session_state.summary_cache:
            st.markdown(st.session_state.summary_cache)
            if st.button("🔄 Regenerasi Ringkasan"):
                st.session_state.summary_cache = None
                st.rerun()
        else:
            if st.button("✨ Generate Laporan Komprehensif", type="primary"):
                with st.spinner("🤖 AI sedang menyusun laporan eksekutif..."):
                    summary = generate_ai_summary(
                        client, MODEL_SMART, df,
                        sc, st.session_state.topic_details,
                        st.session_state.ner_results, text_type, language
                    )
                    st.session_state.summary_cache = summary
                    st.rerun()

    # ========================
    # TAB 2: SENTIMEN
    # ========================
    with t2:
        st.markdown("### 🎭 Analisis Sentimen")
        c1, c2 = st.columns([1, 2])
        with c1:
            fig_pie = px.pie(
                values=list(sc.values()), names=list(sc.keys()),
                hole=0.45, color=list(sc.keys()),
                color_discrete_map=SENTIMENT_COLORS,
                template=plotly_template
            )
            fig_pie.update_traces(textinfo='percent+label', textfont_size=13)
            fig_pie.update_layout(
                showlegend=False, height=350,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Sentiment counts bar
            df_sc = pd.DataFrame({'Sentimen': sc.keys(), 'Jumlah': sc.values()})
            fig_bar_s = px.bar(
                df_sc, x='Sentimen', y='Jumlah', color='Sentimen',
                color_discrete_map=SENTIMENT_COLORS, template=plotly_template,
                text='Jumlah'
            )
            fig_bar_s.update_traces(textposition='outside')
            fig_bar_s.update_layout(showlegend=False, height=250, margin=dict(t=10, b=10))
            st.plotly_chart(fig_bar_s, use_container_width=True)

        with c2:
            f_s = st.multiselect("Filter Sentimen:", df['Sentimen'].unique(), default=list(df['Sentimen'].unique()))

            def color_sentiment(v):
                if v == 'Positif':
                    return 'background-color: #28a745; color: white'
                if v == 'Negatif':
                    return 'background-color: #dc3545; color: white'
                if v == 'Netral':
                    return 'background-color: #ffc107; color: black'
                return ''

            filtered = df[df['Sentimen'].isin(f_s)][['Teks_Asli', 'Topik', 'Sentimen']]
            st.dataframe(
                filtered.style.map(color_sentiment, subset=['Sentimen']),
                use_container_width=True, height=500
            )

        # Ringkasan AI - Sentimen
        total = len(df)
        pos_pct = round(sc.get('Positif', 0) / total * 100, 1) if total else 0
        neg_pct = round(sc.get('Negatif', 0) / total * 100, 1) if total else 0
        net_pct = round(sc.get('Netral', 0) / total * 100, 1) if total else 0
        ctx_sentimen = f"""Total Dokumen: {total}
Distribusi Sentimen:
- Positif: {sc.get('Positif', 0)} dokumen ({pos_pct}%)
- Negatif: {sc.get('Negatif', 0)} dokumen ({neg_pct}%)
- Netral: {sc.get('Netral', 0)} dokumen ({net_pct}%)
Sentimen Dominan: {max(sc, key=sc.get) if sc else '-'}
Tipe Teks: {text_type} | Bahasa: {language}"""
        render_tab_insight_ui(client, MODEL_SMART, "sentimen", ctx_sentimen, "Analisis Sentimen")

    # ========================
    # TAB 3: TOPIK
    # ========================
    with t3:
        st.markdown("### 📂 Analisis Topik (Clustering)")

        tc = df['Topik'].value_counts().reset_index()
        tc.columns = ['Topik', 'Jumlah']
        fig_bar_t = px.bar(
            tc, x='Jumlah', y='Topik', orientation='h',
            color='Jumlah', color_continuous_scale='Blues',
            template=plotly_template, text='Jumlah'
        )
        fig_bar_t.update_traces(textposition='outside')
        fig_bar_t.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
        st.plotly_chart(fig_bar_t, use_container_width=True)

        # PCA Scatter
        if hasattr(st.session_state, 'tfidf_matrix') and hasattr(st.session_state, 'topic_map'):
            fig_scatter = generate_cluster_scatter(
                st.session_state.tfidf_matrix,
                st.session_state.kmeans.labels_,
                st.session_state.topic_map,
                plotly_template
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("#### 📋 Detail Kata Kunci per Topik")
        df_topics = pd.DataFrame(st.session_state.topic_details)
        st.dataframe(df_topics, hide_index=True, use_container_width=True, column_config={
            "Nomor": st.column_config.NumberColumn("No.", width="small"),
            "Topik": st.column_config.TextColumn("Nama Topik", width="medium"),
            "Keywords": st.column_config.TextColumn("Kata Kunci (Top 10)", width="large"),
            "Jumlah_Dokumen": st.column_config.NumberColumn("Jumlah Dok.", width="small")
        })

        # Word Cloud per Topik
        st.markdown("---")
        st.markdown("#### ☁️ Word Cloud per Topik")
        st.caption("Visualisasi kata-kata dominan untuk setiap topik/klaster yang teridentifikasi.")

        wc_bg = 'black' if theme_mode == 'Dark' else 'white'
        topic_names_list = df['Topik'].unique().tolist()
        num_topics_wc = len(topic_names_list)

        if num_topics_wc > 0:
            # Tentukan grid layout: maks 3 kolom per baris
            cols_per_row = min(3, num_topics_wc)
            rows_needed = (num_topics_wc + cols_per_row - 1) // cols_per_row

            topic_color_map = {}
            for idx, detail in enumerate(st.session_state.topic_details):
                topic_color_map[detail['Topik']] = TOPIC_COLORS[idx % len(TOPIC_COLORS)]

            for row_idx in range(rows_needed):
                cols_wc = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    topic_idx = row_idx * cols_per_row + col_idx
                    if topic_idx >= num_topics_wc:
                        break
                    t_name = topic_names_list[topic_idx]
                    t_color = topic_color_map.get(t_name, '#4facfe')
                    topic_texts = df[df['Topik'] == t_name]['Teks_Clean']
                    combined_text = " ".join(topic_texts.tolist())

                    with cols_wc[col_idx]:
                        if combined_text.strip():
                            try:
                                def make_color_func(base_color):
                                    """Membuat fungsi warna gradasi dari warna dasar topik."""
                                    r = int(base_color[1:3], 16)
                                    g = int(base_color[3:5], 16)
                                    b = int(base_color[5:7], 16)
                                    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                                        # Variasi kecerahan berdasarkan font_size
                                        factor = max(0.4, min(font_size / 80, 1.0))
                                        ri = int(r * factor + (255 - r) * (1 - factor) * 0.3)
                                        gi = int(g * factor + (255 - g) * (1 - factor) * 0.3)
                                        bi = int(b * factor + (255 - b) * (1 - factor) * 0.3)
                                        return f"rgb({min(ri,255)},{min(gi,255)},{min(bi,255)})"
                                    return color_func

                                wc_topic = WordCloud(
                                    width=400, height=280,
                                    background_color=wc_bg,
                                    color_func=make_color_func(t_color),
                                    max_words=50,
                                    contour_width=0,
                                    prefer_horizontal=0.7,
                                    min_font_size=8
                                ).generate(combined_text)

                                fig_wc_t, ax_wc_t = plt.subplots(figsize=(5, 3.5), facecolor=wc_bg)
                                ax_wc_t.imshow(wc_topic, interpolation='bilinear')
                                ax_wc_t.axis("off")
                                title_color = 'white' if theme_mode == 'Dark' else '#1a1a2e'
                                doc_count = len(topic_texts)
                                ax_wc_t.set_title(
                                    f"{t_name}\n({doc_count} dokumen)",
                                    fontsize=10, fontweight='bold', color=title_color,
                                    pad=8
                                )
                                plt.tight_layout()
                                st.pyplot(fig_wc_t)
                                plt.close(fig_wc_t)
                            except ValueError:
                                st.info(f"Data tidak cukup untuk **{t_name}**")
                        else:
                            st.info(f"Tidak ada teks untuk **{t_name}**")

        # Ringkasan AI - Topik
        topics_str = "\n".join([f"  - Topik '{t['Topik']}': kata kunci [{t['Keywords']}], {t['Jumlah_Dokumen']} dokumen" for t in st.session_state.topic_details])
        ctx_topik = f"""Total Dokumen: {len(df)} | Jumlah Topik: {df['Topik'].nunique()}
Topik & Kata Kunci:
{topics_str}"""
        render_tab_insight_ui(client, MODEL_SMART, "topik", ctx_topik, "Analisis Tematik (Topik)")

    # ========================
    # TAB 4: CROSS ANALYSIS (NEW)
    # ========================
    with t4:
        st.markdown("### 🔀 Cross-Analysis: Sentimen × Topik")
        st.caption("Melihat distribusi sentimen di dalam setiap topik untuk menemukan pola dan area bermasalah.")

        cross_tab = pd.crosstab(df['Topik'], df['Sentimen'])
        for col in ['Positif', 'Negatif', 'Netral']:
            if col not in cross_tab.columns:
                cross_tab[col] = 0
        cross_tab = cross_tab[['Positif', 'Netral', 'Negatif']]

        # Stacked bar
        fig_stack = go.Figure()
        for sent, color in SENTIMENT_COLORS.items():
            if sent in cross_tab.columns:
                fig_stack.add_trace(go.Bar(
                    name=sent, y=cross_tab.index, x=cross_tab[sent],
                    orientation='h', marker_color=color, text=cross_tab[sent],
                    textposition='inside'
                ))
        fig_stack.update_layout(
            barmode='stack', template=plotly_template,
            yaxis={'categoryorder': 'total ascending'},
            height=max(300, len(cross_tab) * 50 + 100),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig_stack, use_container_width=True)

        # Heatmap
        cross_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0).round(2) * 100
        fig_heat = px.imshow(
            cross_pct, text_auto='.0f',
            color_continuous_scale='RdYlGn',
            labels=dict(x="Sentimen", y="Topik", color="Persen (%)"),
            template=plotly_template,
            aspect='auto'
        )
        fig_heat.update_layout(height=max(300, len(cross_tab) * 45 + 100))
        st.plotly_chart(fig_heat, use_container_width=True)

        # Insight tabel
        st.markdown("#### 📊 Tabel Cross-Tabulation")
        cross_display = cross_tab.copy()
        cross_display['Total'] = cross_display.sum(axis=1)
        cross_display['% Positif'] = (cross_display['Positif'] / cross_display['Total'] * 100).round(1)
        cross_display['% Negatif'] = (cross_display['Negatif'] / cross_display['Total'] * 100).round(1)
        st.dataframe(cross_display, use_container_width=True)

        # Ringkasan AI - Cross Analysis
        cross_data_lines = []
        for topic in cross_display.index:
            r = cross_display.loc[topic]
            cross_data_lines.append(f"  - {topic}: Positif={r.get('Positif',0)}, Negatif={r.get('Negatif',0)}, Netral={r.get('Netral',0)}, Total={r.get('Total',0)}, %Positif={r.get('% Positif',0)}%, %Negatif={r.get('% Negatif',0)}%")
        ctx_cross = f"""Cross-Tabulation Sentimen × Topik:
{chr(10).join(cross_data_lines)}"""
        render_tab_insight_ui(client, MODEL_SMART, "cross", ctx_cross, "Sentimen × Topik")

    # ========================
    # TAB 5: KATA KUNCI
    # ========================
    with t5:
        st.markdown("### 🔠 Analisis Kata Kunci")
        txt_all = " ".join(df['Teks_Clean'])
        if txt_all.strip():
            wc_bg = 'black' if theme_mode == 'Dark' else 'white'
            wc = WordCloud(
                width=800, height=400, background_color=wc_bg,
                colormap='viridis', max_words=100,
                contour_width=1, contour_color=wc_bg
            ).generate(txt_all)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5), facecolor=wc_bg)
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc)
            plt.close(fig_wc)

        # Top TF-IDF
        sum_tfidf = st.session_state.tfidf_matrix.sum(axis=0)
        words = [(word, sum_tfidf[0, idx]) for word, idx in st.session_state.vectorizer.vocabulary_.items()]
        words = sorted(words, key=lambda x: x[1], reverse=True)[:15]
        df_k = pd.DataFrame(words, columns=["Kata", "Skor TF-IDF"])
        fig_k = px.bar(
            df_k, x="Skor TF-IDF", y="Kata", orientation='h',
            template=plotly_template, color="Skor TF-IDF",
            color_continuous_scale="Blues", text=df_k['Skor TF-IDF'].round(2)
        )
        fig_k.update_traces(textposition='outside')
        fig_k.update_layout(yaxis={'categoryorder': 'total ascending'}, height=450)
        st.plotly_chart(fig_k, use_container_width=True)

        # Word frequency (raw count)
        with st.expander("📊 Frekuensi Kata Mentah (Raw Count)"):
            all_words = txt_all.split()
            wf = Counter(all_words).most_common(20)
            df_wf = pd.DataFrame(wf, columns=["Kata", "Frekuensi"])
            fig_wf = px.bar(df_wf, x="Frekuensi", y="Kata", orientation='h', template=plotly_template)
            fig_wf.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
            st.plotly_chart(fig_wf, use_container_width=True)

        # Ringkasan AI - Kata Kunci
        top_tfidf = [(word, round(sum_tfidf[0, idx], 3)) for word, idx in st.session_state.vectorizer.vocabulary_.items()]
        top_tfidf = sorted(top_tfidf, key=lambda x: x[1], reverse=True)[:15]
        tfidf_str = ", ".join([f"{w} ({s})" for w, s in top_tfidf])
        all_w = txt_all.split() if txt_all.strip() else []
        raw_top = Counter(all_w).most_common(15)
        raw_str = ", ".join([f"{w} ({c}x)" for w, c in raw_top])
        ctx_kk = f"""Top 15 Kata Kunci (TF-IDF): {tfidf_str}
Top 15 Frekuensi Kata Mentah: {raw_str}
Total Kata Unik: {len(set(all_w))} | Total Kata: {len(all_w)}"""
        render_tab_insight_ui(client, MODEL_SMART, "katakunci", ctx_kk, "Analisis Kata Kunci")

    # ========================
    # TAB 6: N-GRAM
    # ========================
    with t6:
        st.markdown("### 🔗 Analisis Frasa (N-Gram)")
        st.caption("Mendeteksi frasa yang sering muncul bersamaan dalam teks.")

        ng1, ng2 = st.columns(2)
        with ng1:
            st.markdown("**Bigram (2 Kata)**")
            bigrams = get_ngrams(df['Teks_Clean'], n=2, top_k=10)
            if bigrams:
                df_bi = pd.DataFrame(bigrams, columns=["Frasa", "Frekuensi"])
                fig_bi = px.bar(
                    df_bi, x="Frekuensi", y="Frasa", orientation='h',
                    template=plotly_template, color="Frekuensi",
                    color_continuous_scale="Teal", text="Frekuensi"
                )
                fig_bi.update_traces(textposition='outside')
                fig_bi.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                st.plotly_chart(fig_bi, use_container_width=True)
            else:
                st.info("Data tidak cukup untuk analisis Bigram.")

        with ng2:
            st.markdown("**Trigram (3 Kata)**")
            trigrams = get_ngrams(df['Teks_Clean'], n=3, top_k=10)
            if trigrams:
                df_tri = pd.DataFrame(trigrams, columns=["Frasa", "Frekuensi"])
                fig_tri = px.bar(
                    df_tri, x="Frekuensi", y="Frasa", orientation='h',
                    template=plotly_template, color="Frekuensi",
                    color_continuous_scale="Purp", text="Frekuensi"
                )
                fig_tri.update_traces(textposition='outside')
                fig_tri.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                st.plotly_chart(fig_tri, use_container_width=True)
            else:
                st.info("Data tidak cukup untuk analisis Trigram.")

        # Fourgram (new)
        with st.expander("🔬 Lihat Fourgram (4 Kata)"):
            fourgrams = get_ngrams(df['Teks_Clean'], n=4, top_k=10)
            if fourgrams:
                df_fg = pd.DataFrame(fourgrams, columns=["Frasa", "Frekuensi"])
                fig_fg = px.bar(
                    df_fg, x="Frekuensi", y="Frasa", orientation='h',
                    template=plotly_template, color="Frekuensi", text="Frekuensi"
                )
                fig_fg.update_traces(textposition='outside')
                fig_fg.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                st.plotly_chart(fig_fg, use_container_width=True)
            else:
                st.info("Data tidak cukup untuk analisis Fourgram.")

        # Ringkasan AI - N-Gram
        bigrams_ctx = get_ngrams(df['Teks_Clean'], n=2, top_k=10)
        trigrams_ctx = get_ngrams(df['Teks_Clean'], n=3, top_k=10)
        fourgrams_ctx = get_ngrams(df['Teks_Clean'], n=4, top_k=5)
        bi_str = ", ".join([f"'{b[0]}' ({b[1]}x)" for b in bigrams_ctx]) if bigrams_ctx else "tidak cukup data"
        tri_str = ", ".join([f"'{t[0]}' ({t[1]}x)" for t in trigrams_ctx]) if trigrams_ctx else "tidak cukup data"
        four_str = ", ".join([f"'{f[0]}' ({f[1]}x)" for f in fourgrams_ctx]) if fourgrams_ctx else "tidak cukup data"
        ctx_ngram = f"""Bigram (2 kata) teratas: {bi_str}
Trigram (3 kata) teratas: {tri_str}
Fourgram (4 kata) teratas: {four_str}"""
        render_tab_insight_ui(client, MODEL_SMART, "ngram", ctx_ngram, "Analisis N-Gram")

    # ========================
    # TAB 7: NER
    # ========================
    with t7:
        st.markdown("### 👤 Named Entity Recognition (NER)")
        st.caption("Entitas penting yang dideteksi secara otomatis oleh AI dari keseluruhan dokumen.")

        if st.session_state.ner_results:
            res = st.session_state.ner_results

            if res.get("Error"):
                st.error(f"Terjadi error pada NER: {res['Error']}")

            col_p, col_o, col_l = st.columns(3)

            with col_p:
                persons = res.get("Person", [])
                st.markdown(f"""<div class="ner-card" style="border-left-color: #4ECDC4;">
                    <h4 style="margin: 0 0 8px 0;">🧑 Person ({len(persons)})</h4>
                </div>""", unsafe_allow_html=True)
                if persons:
                    for p in persons:
                        st.markdown(f"- **{p}**")
                else:
                    st.write("Tidak ditemukan")

            with col_o:
                orgs = res.get("Organization", [])
                st.markdown(f"""<div class="ner-card" style="border-left-color: #45B7D1;">
                    <h4 style="margin: 0 0 8px 0;">🏢 Organization ({len(orgs)})</h4>
                </div>""", unsafe_allow_html=True)
                if orgs:
                    for o in orgs:
                        st.markdown(f"- **{o}**")
                else:
                    st.write("Tidak ditemukan")

            with col_l:
                locs = res.get("Location", [])
                st.markdown(f"""<div class="ner-card" style="border-left-color: #FFA07A;">
                    <h4 style="margin: 0 0 8px 0;">📍 Location ({len(locs)})</h4>
                </div>""", unsafe_allow_html=True)
                if locs:
                    for loc in locs:
                        st.markdown(f"- **{loc}**")
                else:
                    st.write("Tidak ditemukan")

            # NER Summary Chart
            ner_counts = {
                'Person': len(persons),
                'Organization': len(orgs),
                'Location': len(locs)
            }
            if sum(ner_counts.values()) > 0:
                st.markdown("---")
                st.markdown("#### 📊 Distribusi Entitas")
                fig_ner = px.bar(
                    x=list(ner_counts.keys()), y=list(ner_counts.values()),
                    color=list(ner_counts.keys()),
                    color_discrete_sequence=['#4ECDC4', '#45B7D1', '#FFA07A'],
                    template=plotly_template,
                    labels={'x': 'Tipe Entitas', 'y': 'Jumlah'},
                    text=list(ner_counts.values())
                )
                fig_ner.update_traces(textposition='outside')
                fig_ner.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_ner, use_container_width=True)
        else:
            st.error("Gagal memuat hasil NER.")

        # Ringkasan AI - NER
        if st.session_state.ner_results:
            ner_r = st.session_state.ner_results
            persons_ctx = ner_r.get('Person', [])
            orgs_ctx = ner_r.get('Organization', [])
            locs_ctx = ner_r.get('Location', [])
            ctx_ner = f"""Entitas Terdeteksi:
- Person ({len(persons_ctx)}): {', '.join(persons_ctx[:15]) if persons_ctx else 'tidak terdeteksi'}
- Organization ({len(orgs_ctx)}): {', '.join(orgs_ctx[:15]) if orgs_ctx else 'tidak terdeteksi'}
- Location ({len(locs_ctx)}): {', '.join(locs_ctx[:15]) if locs_ctx else 'tidak terdeteksi'}
Total Entitas: {len(persons_ctx) + len(orgs_ctx) + len(locs_ctx)}"""
            render_tab_insight_ui(client, MODEL_SMART, "ner", ctx_ner, "Analisis Entitas (NER)")

    # ========================
    # TAB 8: NETWORK
    # ========================
    with t8:
        st.markdown("### 🌐 Text Network Analysis")
        st.caption("Visualisasi hubungan antara topik (node besar) dan kata kunci dominan (node kecil).")

        if st.session_state.topic_details:
            net_mode = st.radio(
                "Pilih mode visualisasi:",
                ["Interaktif (Plotly)", "Statis (Matplotlib)"],
                horizontal=True
            )
            if net_mode == "Interaktif (Plotly)":
                fig_net = generate_text_network_plotly(st.session_state.topic_details, theme_mode)
                st.plotly_chart(fig_net, use_container_width=True)
            else:
                fig_net = generate_text_network(st.session_state.topic_details, theme_mode)
                st.pyplot(fig_net)
                plt.close()
        else:
            st.warning("Data topik belum tersedia.")

        # Ringkasan AI - Network
        if st.session_state.topic_details:
            net_lines = []
            for td in st.session_state.topic_details:
                net_lines.append(f"  - {td['Topik']}: kata kunci [{td['Keywords']}]")
            ctx_net = f"""Topik dalam jaringan ({len(st.session_state.topic_details)} topik):
{chr(10).join(net_lines)}
Setiap topik terhubung dengan kata kunci dominannya dalam graph."""
            render_tab_insight_ui(client, MODEL_SMART, "network", ctx_net, "Text Network Analysis")

    # ========================
    # TAB 9: ENTITY NETWORK
    # ========================
    with t9:
        st.markdown("### 🕸️ Visualisasi Jaringan Ko-kemunculan Entitas")
        st.caption("Membangun jaringan berdasarkan entitas (Person, Organization, Location) yang muncul bersama dalam dokumen yang sama.")

        if st.session_state.ner_results:
            ner = st.session_state.ner_results
            total_ents = len(ner.get('Person', [])) + len(ner.get('Organization', [])) + len(ner.get('Location', []))

            if total_ents < 2:
                st.warning("Entitas yang terdeteksi kurang dari 2. Jaringan ko-kemunculan tidak dapat dibangun.")
            else:
                # Build network
                G, etype_map = build_entity_cooccurrence(
                    df['Teks_Asli'].tolist(), ner, client, MODEL_FAST
                )

                if G is not None and len(G.nodes()) > 0:
                    # Metrics
                    en1, en2, en3, en4 = st.columns(4)
                    with en1:
                        st.metric("Total Node (Entitas)", len(G.nodes()))
                    with en2:
                        st.metric("Total Edge (Koneksi)", len(G.edges()))
                    with en3:
                        density = nx.density(G)
                        st.metric("Kepadatan Jaringan", f"{density:.3f}")
                    with en4:
                        components = nx.number_connected_components(G)
                        st.metric("Komponen Terhubung", components)

                    st.markdown("---")

                    ent_net_mode = st.radio(
                        "Pilih mode visualisasi jaringan entitas:",
                        ["Interaktif (Plotly)", "Statis (Matplotlib)"],
                        horizontal=True,
                        key="ent_net_mode"
                    )

                    if ent_net_mode == "Interaktif (Plotly)":
                        fig_ent = render_entity_network_plotly(G, etype_map, theme_mode)
                        if fig_ent:
                            st.plotly_chart(fig_ent, use_container_width=True)
                    else:
                        fig_ent = render_entity_network_matplotlib(G, etype_map, theme_mode)
                        if fig_ent:
                            st.pyplot(fig_ent)
                            plt.close(fig_ent)

                    # Co-occurrence table
                    st.markdown("---")
                    st.markdown("#### 📋 Tabel Ko-kemunculan Entitas (Top Pasangan)")
                    edge_data = []
                    for u, v, d in sorted(G.edges(data=True), key=lambda x: x[2].get('weight', 0), reverse=True):
                        edge_data.append({
                            'Entitas 1': u,
                            'Tipe 1': etype_map.get(u, '-'),
                            'Entitas 2': v,
                            'Tipe 2': etype_map.get(v, '-'),
                            'Ko-kemunculan': d.get('weight', 0)
                        })
                    if edge_data:
                        df_edges = pd.DataFrame(edge_data[:20])
                        st.dataframe(df_edges, hide_index=True, use_container_width=True)
                    else:
                        st.info("Tidak ada pasangan entitas yang muncul bersama dalam dokumen yang sama.")

                    # Top entities by degree centrality
                    with st.expander("📊 Sentralitas Entitas (Degree Centrality)"):
                        centrality = nx.degree_centrality(G)
                        df_cent = pd.DataFrame([
                            {'Entitas': k, 'Tipe': etype_map.get(k, '-'), 'Degree Centrality': round(v, 4)}
                            for k, v in sorted(centrality.items(), key=lambda x: x[1], reverse=True)
                        ])
                        st.dataframe(df_cent, hide_index=True, use_container_width=True)

                        if len(df_cent) > 0:
                            fig_cent = px.bar(
                                df_cent.head(15), x='Degree Centrality', y='Entitas',
                                orientation='h', color='Tipe',
                                color_discrete_map={'Person': '#4ECDC4', 'Organization': '#45B7D1', 'Location': '#FFA07A'},
                                template=plotly_template, text=df_cent.head(15)['Degree Centrality'].apply(lambda x: f"{x:.3f}")
                            )
                            fig_cent.update_traces(textposition='outside')
                            fig_cent.update_layout(yaxis={'categoryorder': 'total ascending'}, height=450)
                            st.plotly_chart(fig_cent, use_container_width=True)
                else:
                    st.info("Tidak ada ko-kemunculan entitas yang terdeteksi dalam dokumen.")
        else:
            st.error("Hasil NER belum tersedia. Jalankan analisis terlebih dahulu.")

        # Ringkasan AI - Entity Network
        if st.session_state.ner_results:
            ner_en = st.session_state.ner_results
            G_ctx, emap_ctx = build_entity_cooccurrence(df['Teks_Asli'].tolist(), ner_en, client, MODEL_FAST)
            if G_ctx and len(G_ctx.nodes()) > 0:
                top_edges = sorted(G_ctx.edges(data=True), key=lambda x: x[2].get('weight', 0), reverse=True)[:10]
                edges_str = ", ".join([f"{u}↔{v} ({d.get('weight',0)}x)" for u, v, d in top_edges])
                cent_ctx = nx.degree_centrality(G_ctx)
                top_cent = sorted(cent_ctx.items(), key=lambda x: x[1], reverse=True)[:5]
                cent_str = ", ".join([f"{k} ({round(v,3)})" for k, v in top_cent])
                ctx_ent_net = f"""Jaringan Entitas:
- Total Node: {len(G_ctx.nodes())} | Total Edge: {len(G_ctx.edges())}
- Kepadatan: {nx.density(G_ctx):.3f} | Komponen: {nx.number_connected_components(G_ctx)}
- Top Pasangan Ko-kemunculan: {edges_str}
- Top Entitas Sentral (Degree): {cent_str}"""
                render_tab_insight_ui(client, MODEL_SMART, "entity_network", ctx_ent_net, "Jaringan Ko-kemunculan Entitas")

    # ========================
    # TAB 10: KATA KUNCI PER DOKUMEN
    # ========================
    with t10:
        st.markdown("### 📊 10 Kata Kunci Teratas per Dokumen")
        st.caption("Menampilkan 10 kata kunci dengan skor TF-IDF tertinggi beserta frekuensi kemunculan (Raw Count) di setiap dokumen.")

        if hasattr(st.session_state, 'tfidf_matrix') and hasattr(st.session_state, 'vectorizer'):
            tfidf_mat = st.session_state.tfidf_matrix
            feature_names = st.session_state.vectorizer.get_feature_names_out()
            total_docs = len(df)

            # Helper: hitung frekuensi kata dalam satu dokumen
            def get_word_freq_doc(doc_text):
                words = doc_text.split()
                return Counter(words)

            # --- Mode Pilihan ---
            doc_view_mode = st.radio(
                "Mode tampilan:",
                ["📄 Pilih Dokumen", "📑 Semua Dokumen (Grid)"],
                horizontal=True,
                key="doc_kw_mode"
            )

            if doc_view_mode == "📄 Pilih Dokumen":
                # Preview teks untuk membantu pemilihan
                doc_options = []
                for i in range(total_docs):
                    preview = df['Teks_Asli'].iloc[i]
                    preview_short = (preview[:80] + "...") if len(preview) > 80 else preview
                    doc_options.append(f"Dokumen {i + 1}: {preview_short}")

                selected_doc = st.selectbox(
                    "Pilih dokumen:", doc_options, key="doc_kw_select"
                )
                doc_idx = doc_options.index(selected_doc)

                # Ambil top 10 kata kunci untuk dokumen terpilih
                row = tfidf_mat[doc_idx].toarray().flatten()
                top_indices = row.argsort()[-10:][::-1]
                top_words = [(feature_names[i], round(row[i], 4)) for i in top_indices if row[i] > 0]

                if top_words:
                    # Info dokumen
                    st.markdown(f"**Dokumen #{doc_idx + 1}** — Topik: `{df['Topik'].iloc[doc_idx]}` | Sentimen: `{df['Sentimen'].iloc[doc_idx]}`")
                    with st.expander("📖 Lihat Teks Lengkap", expanded=False):
                        st.write(df['Teks_Asli'].iloc[doc_idx])

                    # Hitung frekuensi kata untuk dokumen ini
                    word_freq = get_word_freq_doc(df['Teks_Clean'].iloc[doc_idx])

                    # Bangun dataframe gabungan
                    combined_data = []
                    for kata, skor in top_words:
                        combined_data.append({
                            "Kata Kunci": kata,
                            "Skor TF-IDF": skor,
                            "Frekuensi": word_freq.get(kata, 0)
                        })
                    df_doc_kw = pd.DataFrame(combined_data)

                    # --- Dua chart berdampingan ---
                    chart_left, chart_right = st.columns(2)

                    with chart_left:
                        fig_tfidf = px.bar(
                            df_doc_kw, x="Skor TF-IDF", y="Kata Kunci", orientation='h',
                            template=plotly_template, color="Skor TF-IDF",
                            color_continuous_scale="Teal",
                            text=df_doc_kw['Skor TF-IDF'].apply(lambda x: f"{x:.4f}")
                        )
                        fig_tfidf.update_traces(textposition='outside')
                        fig_tfidf.update_layout(
                            yaxis={'categoryorder': 'total ascending'},
                            height=420,
                            title=dict(text="Skor TF-IDF", font=dict(size=14)),
                            coloraxis_showscale=False
                        )
                        st.plotly_chart(fig_tfidf, use_container_width=True)

                    with chart_right:
                        fig_freq = px.bar(
                            df_doc_kw, x="Frekuensi", y="Kata Kunci", orientation='h',
                            template=plotly_template, color="Frekuensi",
                            color_continuous_scale="Purp",
                            text="Frekuensi"
                        )
                        fig_freq.update_traces(textposition='outside')
                        fig_freq.update_layout(
                            yaxis={'categoryorder': 'total ascending'},
                            height=420,
                            title=dict(text="Frekuensi Kata (Raw Count)", font=dict(size=14)),
                            coloraxis_showscale=False
                        )
                        st.plotly_chart(fig_freq, use_container_width=True)

                    # --- Grouped Bar Chart gabungan ---
                    st.markdown("##### 🔀 Perbandingan TF-IDF vs Frekuensi")
                    # Normalisasi ke 0-1 agar bisa dibandingkan secara visual
                    df_norm = df_doc_kw.copy()
                    max_tfidf = df_norm['Skor TF-IDF'].max()
                    max_freq = df_norm['Frekuensi'].max()
                    df_norm['TF-IDF (Normalisasi)'] = df_norm['Skor TF-IDF'] / max_tfidf if max_tfidf > 0 else 0
                    df_norm['Frekuensi (Normalisasi)'] = df_norm['Frekuensi'] / max_freq if max_freq > 0 else 0

                    fig_combined = go.Figure()
                    fig_combined.add_trace(go.Bar(
                        y=df_norm['Kata Kunci'], x=df_norm['TF-IDF (Normalisasi)'],
                        name='TF-IDF', orientation='h',
                        marker_color='#4ECDC4',
                        text=df_norm['Skor TF-IDF'].apply(lambda x: f"{x:.4f}"),
                        textposition='outside'
                    ))
                    fig_combined.add_trace(go.Bar(
                        y=df_norm['Kata Kunci'], x=df_norm['Frekuensi (Normalisasi)'],
                        name='Frekuensi', orientation='h',
                        marker_color='#BB8FCE',
                        text=df_norm['Frekuensi'].astype(int).astype(str) + 'x',
                        textposition='outside'
                    ))
                    fig_combined.update_layout(
                        barmode='group',
                        template=plotly_template,
                        yaxis={'categoryorder': 'total ascending'},
                        height=450,
                        title=dict(text=f"TF-IDF vs Frekuensi — Dokumen #{doc_idx + 1}", font=dict(size=13)),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                        xaxis_title="Skor Normalisasi (0–1)"
                    )
                    st.plotly_chart(fig_combined, use_container_width=True)

                    # Tabel detail gabungan
                    st.markdown("##### 📋 Tabel Detail Kata Kunci")
                    df_table = df_doc_kw.copy()
                    df_table.index = range(1, len(df_table) + 1)
                    df_table.index.name = "Peringkat"
                    st.dataframe(df_table, use_container_width=True)
                else:
                    st.warning("Dokumen ini tidak memiliki kata kunci yang terdeteksi setelah preprocessing.")

            else:
                # --- Grid View: Semua Dokumen ---
                st.info(f"Menampilkan diagram batang kata kunci untuk **{total_docs} dokumen**. Scroll ke bawah untuk melihat semua.")

                # Batasi jika terlalu banyak dokumen
                max_grid_docs = 30
                if total_docs > max_grid_docs:
                    st.warning(f"Jumlah dokumen ({total_docs}) melebihi batas tampilan grid ({max_grid_docs}). Hanya {max_grid_docs} dokumen pertama yang ditampilkan. Gunakan mode '📄 Pilih Dokumen' untuk melihat dokumen lainnya.")
                    display_count = max_grid_docs
                else:
                    display_count = total_docs

                cols_per_row_doc = 2
                rows_doc = (display_count + cols_per_row_doc - 1) // cols_per_row_doc

                for row_i in range(rows_doc):
                    cols_doc = st.columns(cols_per_row_doc)
                    for col_i in range(cols_per_row_doc):
                        d_idx = row_i * cols_per_row_doc + col_i
                        if d_idx >= display_count:
                            break

                        with cols_doc[col_i]:
                            row_data = tfidf_mat[d_idx].toarray().flatten()
                            top_idx = row_data.argsort()[-10:][::-1]
                            top_kw = [(feature_names[j], round(row_data[j], 4)) for j in top_idx if row_data[j] > 0]

                            if top_kw:
                                # Hitung frekuensi kata dokumen ini
                                wf_doc = get_word_freq_doc(df['Teks_Clean'].iloc[d_idx])
                                df_kw_grid = pd.DataFrame(top_kw, columns=["Kata", "TF-IDF"])
                                df_kw_grid["Frekuensi"] = df_kw_grid["Kata"].apply(lambda k: wf_doc.get(k, 0))

                                sent = df['Sentimen'].iloc[d_idx]

                                # Grouped bar: TF-IDF + Frekuensi (normalisasi)
                                max_t = df_kw_grid['TF-IDF'].max()
                                max_f = df_kw_grid['Frekuensi'].max()
                                df_kw_grid['TF-IDF_norm'] = df_kw_grid['TF-IDF'] / max_t if max_t > 0 else 0
                                df_kw_grid['Freq_norm'] = df_kw_grid['Frekuensi'] / max_f if max_f > 0 else 0

                                fig_grid = go.Figure()
                                fig_grid.add_trace(go.Bar(
                                    y=df_kw_grid['Kata'], x=df_kw_grid['TF-IDF_norm'],
                                    name='TF-IDF', orientation='h',
                                    marker_color='#4ECDC4',
                                    text=df_kw_grid['TF-IDF'].apply(lambda x: f"{x:.3f}"),
                                    textposition='outside',
                                    showlegend=(d_idx == 0)
                                ))
                                fig_grid.add_trace(go.Bar(
                                    y=df_kw_grid['Kata'], x=df_kw_grid['Freq_norm'],
                                    name='Frekuensi', orientation='h',
                                    marker_color='#BB8FCE',
                                    text=df_kw_grid['Frekuensi'].astype(int).astype(str) + 'x',
                                    textposition='outside',
                                    showlegend=(d_idx == 0)
                                ))

                                preview_t = df['Teks_Asli'].iloc[d_idx]
                                preview_short_t = (preview_t[:50] + "...") if len(preview_t) > 50 else preview_t

                                fig_grid.update_layout(
                                    barmode='group',
                                    template=plotly_template,
                                    yaxis={'categoryorder': 'total ascending'},
                                    height=360,
                                    title=dict(
                                        text=f"Dok #{d_idx + 1} [{sent}]<br><sub>{preview_short_t}</sub>",
                                        font=dict(size=11)
                                    ),
                                    margin=dict(t=60, b=10, l=10, r=10),
                                    legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='center', x=0.5, font=dict(size=9)),
                                    xaxis_title=""
                                )
                                st.plotly_chart(fig_grid, use_container_width=True)
                            else:
                                st.info(f"Dok #{d_idx + 1}: Tidak ada kata kunci.")
        else:
            st.warning("Data TF-IDF belum tersedia. Jalankan analisis terlebih dahulu.")

        # Ringkasan AI - Kata Kunci per Dokumen
        if hasattr(st.session_state, 'tfidf_matrix') and hasattr(st.session_state, 'vectorizer'):
            tfidf_m = st.session_state.tfidf_matrix
            fn = st.session_state.vectorizer.get_feature_names_out()
            doc_kw_samples = []
            sample_count = min(5, len(df))
            for si in range(sample_count):
                r = tfidf_m[si].toarray().flatten()
                ti = r.argsort()[-5:][::-1]
                kws = [fn[j] for j in ti if r[j] > 0]
                wf_s = Counter(df['Teks_Clean'].iloc[si].split())
                kw_detail = ", ".join([f"{w}(tfidf={round(r[list(fn).index(w)],3)}, freq={wf_s.get(w,0)})" for w in kws[:5]])
                doc_kw_samples.append(f"  Dok#{si+1} [{df['Topik'].iloc[si]}|{df['Sentimen'].iloc[si]}]: {kw_detail}")
            # Hitung variasi
            all_top_kws = set()
            for si in range(len(df)):
                r = tfidf_m[si].toarray().flatten()
                ti = r.argsort()[-5:][::-1]
                for j in ti:
                    if r[j] > 0:
                        all_top_kws.add(fn[j])
            ctx_dkw = f"""Total Dokumen: {len(df)}
Kata Kunci Unik (top-5 per dok): {len(all_top_kws)}
Sampel Dokumen:
{chr(10).join(doc_kw_samples)}"""
            render_tab_insight_ui(client, MODEL_SMART, "doc_keywords", ctx_dkw, "Kata Kunci per Dokumen")

    # ========================
    # EXPORT SECTION
    # ========================
    st.divider()
    st.markdown("### 📥 Ekspor Data")

    exp1, exp2, exp3 = st.columns(3)

    with exp1:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Unduh CSV (Data Lengkap)",
            csv_data, "anatext_analysis.csv", "text/csv",
            use_container_width=True
        )

    with exp2:
        # Export topic summary
        df_topic_export = pd.DataFrame(st.session_state.topic_details)
        st.download_button(
            "📥 Unduh Topik & Keywords",
            df_topic_export.to_csv(index=False).encode('utf-8'),
            "anatext_topics.csv", "text/csv",
            use_container_width=True
        )

    with exp3:
        # Export summary report as text
        if st.session_state.summary_cache:
            st.download_button(
                "📥 Unduh Laporan AI (TXT)",
                st.session_state.summary_cache.encode('utf-8'),
                "anatext_report.txt", "text/plain",
                use_container_width=True
            )
        else:
            st.button("📥 Generate laporan dulu", disabled=True, use_container_width=True)
