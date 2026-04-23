import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Document Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Advanced Document Analyzer\nAnalyze documents with AI-powered insights and visualizations."
    }
)

import math
import os
import spacy
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud, STOPWORDS
from collections import Counter, defaultdict
from io import BytesIO
from docx import Document
from textblob import TextBlob
import seaborn as sns
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from gensim import corpora, models
import textstat
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import plotly.express as px
import plotly.graph_objects as go
from spacy import displacy
import requests
import re
import string
from bs4 import BeautifulSoup
import ftfy
import contractions
from unidecode import unidecode
from cleantext import clean
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import plotly.figure_factory as ff
from chatbot import get_text_completion, render_structured_output

# --- Custom CSS ---
st.markdown("""
<style>
    /* Page background */
    .main .block-container { padding-top: 1.5rem; }

    /* Header banner */
    .app-header {
        background: linear-gradient(135deg, #0d1b2e 0%, #1a3a5c 100%);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.8rem;
        text-align: center;
        color: #e2e8f0;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
        border: 1px solid #2d4a6e;
    }
    .app-header h1 { font-size: 2.1rem; margin: 0; letter-spacing: 0.3px; font-weight: 700; }
    .app-header p  { margin: 0.5rem 0 0; opacity: 0.75; font-size: 1rem; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }

    /* Section headings */
    h3 { color: #7db0f7; border-bottom: 1px solid #2d3748; padding-bottom: 0.35rem; margin-top: 1.2rem; }

    /* Sidebar refinements */
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
    [data-testid="stSidebar"] hr { border-color: #2d3748; }

    /* Tab active state */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid #4f8ef7;
        color: #7db0f7;
        font-weight: 700;
    }

    /* AI response box */
    .ai-box {
        background: #101828;
        border-left: 4px solid #4f8ef7;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.3rem;
        margin-top: 0.6rem;
        color: #e2e8f0;
        line-height: 1.65;
    }

    /* Summary card */
    .summary-card {
        background: #0d1f18;
        border-left: 4px solid #48bb78;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.3rem;
        margin: 0.5rem 0 1.2rem;
        color: #c6f6d5;
        line-height: 1.6;
    }
    .summary-card strong { color: #68d391; }

    /* Stat table */
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# --- spaCy model (cached so it loads once) ---
@st.cache_resource
def _load_nlp():
    return spacy.load('en_core_web_sm')

nlp = _load_nlp()

# Custom stopwords (keeping the existing ones)
extended_filler_words = {
    "uh", "uhm", "like", "you know", "i mean", "kind of", "sort of", "um", "yeah", 
    "well", "okay", "so", "right", "i guess", "you see", "basically", "actually", 
    "literally", "i don't know", "just", "gotcha", "hmm", "oh", "anyway", "alright"
}

extended_stopwords = {
    "the", "and", "but", "or", "if", "when", "how", "what", "where", "who", 
    "which", "why", "whose", "whom", "these", "those", "this", "that", "am", 
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", 
    "having", "do", "does", "did", "doing", "would", "should", "could", "might", 
    "must", "shall", "will", "can", "may", "a", "an", "the"
}

custom_stopwords = extended_filler_words.union(extended_stopwords)

# ---------------------------------------------------------------------------
# YPAR / CPIM Framework
# ---------------------------------------------------------------------------

INTERVIEWER_LABELS = {
    'l', 's', 'i', 'int', 'interviewer', 'researcher', 'r', 'moderator',
    'facilitator', 'q', 'interviewer1', 'interviewer2',
}

# Matches inline "Firstname Lastname HH:MM" or "Firstname HH:MM" from auto-transcription tools
_INLINE_SPEAKER_RE = re.compile(
    r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(\d{1,2}:\d{2})\b\s*'
)

YPAR_THEMES = {
    "Spatial Justice": [
        "map", "space", "place", "neighborhood", "location", "geographic",
        "area", "district", "zone", "redlining", "segregation", "geography",
        "city", "urban", "street", "block", "environment",
    ],
    "Agency & Voice": [
        "i think", "i feel", "i want", "i believe", "we can", "we should",
        "action", "change", "impact", "power", "voice", "speak", "decide",
        "choice", "advocate", "demand", "fight", "stand",
    ],
    "Systemic Inequity": [
        "unfair", "inequality", "inequity", "system", "structure", "barrier",
        "access", "privilege", "discriminat", "bias", "oppression", "justice",
        "racism", "poverty", "resource", "lack", "denied", "excluded",
    ],
    "Mathematical Thinking": [
        "data", "number", "statistic", "percent", "ratio", "measure",
        "calculate", "graph", "pattern", "trend", "count", "quantit",
        "math", "chart", "average", "rate", "frequency",
    ],
    "Community": [
        "community", "family", "neighbor", "together", "belong",
        "connect", "support", "relationship", "people", "group",
        "network", "local", "around us", "each other",
    ],
    "Educational Experience": [
        "school", "class", "teacher", "learn", "grade", "college",
        "university", "student", "education", "curriculum", "course",
        "classroom", "homework", "test", "gpa", "graduate",
    ],
    "Identity & Culture": [
        "identity", "culture", "background", "race", "ethnicity",
        "gender", "immigrant", "first-generation", "who i am", "heritage",
        "language", "tradition", "belief", "faith",
    ],
    "Participatory Action": [
        "research", "project", "interview", "action", "investigate",
        "analyze", "present", "share", "participate", "ypar",
        "finding", "recommendation", "survey", "question",
    ],
}


def _normalize_transcript(text: str) -> str:
    """
    Convert inline auto-transcript markers like 'Stephen Caviness 06:41 text...'
    into proper newline-separated speaker turns: 'Stephen Caviness: text...'
    This handles output from Otter.ai, Teams, Zoom, etc.
    """
    return _INLINE_SPEAKER_RE.sub(lambda m: f'\n{m.group(1)}: ', text)


def _is_interviewer(speaker: str) -> bool:
    """
    Return True if a speaker label looks like the interviewer.
    Rules (in order):
      1. Matches a known interviewer label (l, i, int, ...)
      2. Is a full name with a space — auto-transcription tools embed the
         interviewer's full name (e.g. 'Stephen Caviness') while students
         get single-letter codes ('A', 'D', 'S').
    """
    if speaker.lower() in INTERVIEWER_LABELS:
        return True
    # Full name = two or more capitalized words, no digits
    parts = speaker.strip().split()
    if len(parts) >= 2 and all(p.replace('.', '').isalpha() for p in parts):
        return True
    return False


def is_interview(text: str) -> bool:
    """Detect whether a document looks like an interview transcript."""
    normalized = _normalize_transcript(text)
    speaker_re = re.compile(r'^[A-Za-z][A-Za-z .]{0,20}:\s', re.MULTILINE)
    return len(speaker_re.findall(normalized)) >= 5


def parse_transcript(text: str) -> List[Dict]:
    """
    Parse a transcript into speaker turns, correctly handling:
    - Simple labels:   'L: ...' / 'D: ...' / 'Willow: ...'
    - Auto-transcript: inline 'Stephen Caviness 06:41 ...' markers
    Each turn gets {'speaker', 'text', 'is_student'}.
    """
    text = _normalize_transcript(text)

    lines = text.split('\n')
    turns: List[Dict] = []
    current_speaker: str = None
    current_lines: List[str] = []

    speaker_re = re.compile(r'^([A-Za-z][A-Za-z .]{0,25}):\s*(.*)')

    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = speaker_re.match(line)
        if m:
            if current_speaker is not None and current_lines:
                turns.append({'speaker': current_speaker,
                               'text': ' '.join(current_lines).strip()})
            current_speaker = m.group(1).strip()
            rest = m.group(2).strip()
            current_lines = [rest] if rest else []
        elif current_speaker is not None:
            current_lines.append(line)

    if current_speaker is not None and current_lines:
        turns.append({'speaker': current_speaker,
                       'text': ' '.join(current_lines).strip()})

    if not turns:
        return turns

    # Label each turn
    interviewer_speakers = {t['speaker'] for t in turns if _is_interviewer(t['speaker'])}

    # Last resort: if still nothing found, treat most-frequent speaker as interviewer
    if not interviewer_speakers:
        freq = Counter(t['speaker'] for t in turns)
        interviewer_speakers = {freq.most_common(1)[0][0]}

    for turn in turns:
        turn['is_student'] = turn['speaker'] not in interviewer_speakers

    return turns


def get_student_text(turns: List[Dict]) -> str:
    return ' '.join(t['text'] for t in turns if t.get('is_student'))


def get_interviewer_text(turns: List[Dict]) -> str:
    return ' '.join(t['text'] for t in turns if not t.get('is_student'))


def score_ypar_themes(text: str) -> Dict[str, int]:
    """Count keyword hits per YPAR/CPIM theme."""
    text_lower = text.lower()
    return {
        theme: sum(text_lower.count(kw) for kw in keywords)
        for theme, keywords in YPAR_THEMES.items()
    }


def create_wordcloud(text: str, custom_title: str = "Word Cloud", additional_stopwords: set = set(),
                    background_color: str = '#0f1117', colormap: str = 'Blues',
                    width: int = 900, height: int = 420):
    """Create and display a word cloud visualization."""
    all_stopwords = custom_stopwords.union(STOPWORDS).union(additional_stopwords)

    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        stopwords=all_stopwords,
        min_font_size=10,
        max_font_size=150,
        random_state=42,
    ).generate(text)

    fig = plt.figure(figsize=(11, 4.5), facecolor=background_color)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(custom_title, pad=12, fontsize=13, color='#e2e8f0')
    st.pyplot(fig)

def create_entity_network(text: str, custom_title: str = "Entity Relationship Network",
                        node_size: int = 20, edge_width: float = 0.5,
                        node_colorscale: str = "YlGnBu", edge_color: str = "#888888"):
    """Create an interactive entity relationship network visualization with customization options."""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    if not entities:
        st.write("No entities found in the text.")
        return
    
    # Create network data
    nodes = []
    edges = []
    seen_entities = set()
    
    # Calculate node positions in a circular layout
    num_entities = len(set(ent[0] for ent in entities))
    radius = 1
    angle_step = 2 * math.pi / num_entities
    
    for i, (entity, label) in enumerate(entities):
        if entity not in seen_entities:
            angle = i * angle_step
            nodes.append({
                'id': entity,
                'label': entity,
                'group': label,
                'size': node_size,
                'x': radius * math.cos(angle),
                'y': radius * math.sin(angle)
            })
            seen_entities.add(entity)
        
        if i > 0:
            edges.append({
                'from': entities[i-1][0],
                'to': entity,
                'width': edge_width
            })
    
    # Create Plotly figure
    edge_x = []
    edge_y = []
    for edge in edges:
        x0, y0 = next((n['x'], n['y']) for n in nodes if n['id'] == edge['from'])
        x1, y1 = next((n['x'], n['y']) for n in nodes if n['id'] == edge['to'])
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=edge_width, color=edge_color),
        hoverinfo='none',
        mode='lines')

    node_x = [node['x'] for node in nodes]
    node_y = [node['y'] for node in nodes]
    node_text = [f"{node['label']} ({node['group']})" for node in nodes]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale=node_colorscale,
            size=node_size,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=custom_title,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    st.plotly_chart(fig)

def create_topic_visualization(texts: List[str], num_topics: int = 5, custom_title: str = "Topic-Keyword Heatmap",
                             color_scale: str = "Viridis", keywords_per_topic: int = 5):
    """Create an interactive topic modeling visualization with customization options."""
    # Prepare documents
    texts = [text.split() for text in texts]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Train LDA model
    lda_model = models.LdaModel(
        corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=15,
        random_state=42
    )
    
    # Get keywords and probabilities for each topic
    topic_data = []
    keywords_all = []
    probabilities = []
    
    for topic_id in range(num_topics):
        keywords = lda_model.show_topic(topic_id, topn=keywords_per_topic)
        words, probs = zip(*keywords)
        topic_data.append({
            'topic': f'Topic {topic_id + 1}',
            'keywords': words,
            'probabilities': probs
        })
        keywords_all.extend(words)
        probabilities.extend(probs)
    
    # Create a DataFrame for the heatmap
    heatmap_data = []
    for topic in topic_data:
        row = []
        for keyword in keywords_all[:5]:  # Use first 5 unique keywords
            if keyword in topic['keywords']:
                idx = topic['keywords'].index(keyword)
                row.append(topic['probabilities'][idx])
            else:
                row.append(0)
        heatmap_data.append(row)
    
    # Convert to numpy array
    heatmap_array = np.array(heatmap_data)
    
    # Create heatmap
    fig = px.imshow(
        heatmap_array,
        labels=dict(x="Keywords", y="Topics"),
        x=keywords_all[:keywords_per_topic],
        y=[f"Topic {i+1}" for i in range(num_topics)],
        title=custom_title,
        color_continuous_scale=color_scale,
        aspect="auto"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Keywords",
        yaxis_title="Topics",
        xaxis={'side': 'bottom'}
    )
    st.plotly_chart(fig)

def create_temporal_analysis(text: str, custom_title: str = "Document Flow Analysis",
                           colors: Dict[str, str] = None, line_styles: Dict[str, str] = None,
                           x_label: str = "Sentence Number", y_label: str = "Metric Value"):
    """Create a temporal analysis visualization of document metrics with customization options."""
    doc = nlp(text)
    sentences = list(doc.sents)
    
    metrics = {
        'sentence_length': [],
        'entity_count': [],
        'sentiment_score': []
    }
    
    for sent in sentences:
        metrics['sentence_length'].append(len(sent))
        metrics['entity_count'].append(len(list(sent.ents)))
        metrics['sentiment_score'].append(TextBlob(sent.text).sentiment.polarity)
    
    # Default colors and line styles if not provided
    if colors is None:
        colors = {
            'sentence_length': 'blue',
            'entity_count': 'red',
            'sentiment_score': 'green'
        }
    
    if line_styles is None:
        line_styles = {
            'sentence_length': 'solid',
            'entity_count': 'solid',
            'sentiment_score': 'solid'
        }
    
    # Create time series plot
    fig = go.Figure()
    
    metric_names = {
        'sentence_length': 'Sentence Length',
        'entity_count': 'Entity Count',
        'sentiment_score': 'Sentiment Score'
    }
    
    for metric, values in metrics.items():
        fig.add_trace(go.Scatter(
            y=values,
            name=metric_names[metric],
            line=dict(
                color=colors[metric],
                dash=line_styles[metric]
            )
        ))
    
    fig.update_layout(
        title=custom_title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig)

def detect_language(text: str) -> str:
    """Detect the language of the text."""
    try:
        return detect(text)
    except:
        return "unknown"
    
def clean_html(text: str) -> str:
    """Remove HTML tags from text."""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def expand_contractions(text: str) -> str:
    """Expand contractions in text."""
    return contractions.fix(text)

def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters and fix encoding issues."""
    text = ftfy.fix_text(text)
    return unidecode(text)

def remove_special_characters(text: str) -> str:
    """Remove special characters while preserving essential punctuation."""
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def detect_noise_ratio(text: str) -> float:
    """Calculate the ratio of noisy characters to total characters."""
    total_chars = len(text)
    if total_chars == 0:
        return 0
    noise_chars = len(re.findall(r'[^\w\s.,!?-]', text))
    return noise_chars / total_chars

def detect_structure_issues(text: str) -> List[str]:
    """Detect potential structural issues in the text."""
    issues = []
    
    # Check for very long paragraphs
    paragraphs = text.split('\n\n')
    for i, para in enumerate(paragraphs):
        if len(para.split()) > 200:  # Arbitrary threshold
            issues.append(f"Very long paragraph detected at position {i+1}")
    
    # Check for inconsistent spacing
    if re.search(r'\n{3,}', text):
        issues.append("Inconsistent paragraph spacing detected")
    
    # Check for potential header/list formatting issues
    if re.search(r'^[A-Z][^.!?]*(?:\n|\Z)', text, re.MULTILINE):
        issues.append("Possible unformatted headers detected")
    
    return issues

def analyze_text_structure(text: str) -> Dict:
    """Analyze the structure of the text."""
    structure = {
        'avg_paragraph_length': 0,
        'num_paragraphs': 0,
        'num_sentences': 0,
        'avg_sentence_length': 0,
        'noise_ratio': detect_noise_ratio(text),
        'issues': detect_structure_issues(text)
    }
    
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    structure['num_paragraphs'] = len(paragraphs)
    
    if paragraphs:
        structure['avg_paragraph_length'] = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
    
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    structure['num_sentences'] = len(sentences)
    
    if sentences:
        structure['avg_sentence_length'] = sum(len(s.split()) for s in sentences) / len(sentences)
    
    return structure

def clean_text(text: str, aggressive: bool = False) -> str:
    """
    Clean text with multiple stages of processing.
    If aggressive=True, applies more aggressive cleaning suitable for analysis.
    """
    # Basic cleaning
    text = clean_html(text)
    text = normalize_unicode(text)
    text = expand_contractions(text)
    
    if aggressive:
        # More aggressive cleaning for analysis
        text = text.lower()
        text = remove_special_characters(text)
        doc = nlp(text)
        tokens = [
            token.lemma_ for token in doc 
            if token.text.lower() not in custom_stopwords 
            and not token.is_stop
            and len(token.text) > 1
            and not token.text.isdigit()
        ]
        return ' '.join(tokens)
    else:
        # Light cleaning for readability
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

def detect_outliers(texts: List[str]) -> List[int]:
    """Detect outlier documents based on their content."""
    if len(texts) < 2:
        return []
    
    # Convert texts to TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    
    # Use DBSCAN for outlier detection
    clustering = DBSCAN(eps=0.5, min_samples=2)
    clustering.fit(X.toarray())
    
    # Return indices of outliers (-1 in labels indicates outliers)
    return [i for i, label in enumerate(clustering.labels_) if label == -1]

def generate_summary(text: str, num_sentences: int = 5) -> str:
    """Extractive summary using LSA (sumy)."""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        sentences = summarizer(parser.document, num_sentences)
        return " ".join(str(s) for s in sentences)
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def analyze_document(file_name: str, file_bytes: bytes, file_ext: str) -> Dict:
    """Process and analyze a document with enhanced cleaning and structure analysis."""
    try:
        if file_ext in ('.xlsx', '.xls'):
            df = pd.read_excel(BytesIO(file_bytes))
            original_text = '\n\n'.join(
                f"{col}:\n" + '\n'.join(df[col].astype(str))
                for col in df.columns
            )
        else:
            doc = Document(BytesIO(file_bytes))
            original_text = '\n'.join(para.text for para in doc.paragraphs)
    except Exception as e:
        st.error(f"Error processing {file_name}: {e}")
        return None

    language = detect_language(original_text)
    structure_analysis = analyze_text_structure(original_text)
    cleaned_text_light = clean_text(original_text, aggressive=False)
    cleaned_text_aggressive = clean_text(original_text, aggressive=True)

    # Interview-aware processing
    interview = is_interview(original_text)
    turns = parse_transcript(original_text) if interview else []
    student_text = get_student_text(turns) if turns else cleaned_text_light
    interviewer_text = get_interviewer_text(turns) if turns else ""

    # Sentiment on student voice (or full text for non-interviews)
    sentiment, polarity, subjectivity = analyze_sentiment(student_text)
    stats = get_document_stats(cleaned_text_light)
    entities = extract_named_entities(student_text)
    summary = generate_summary(student_text)
    ypar_themes = score_ypar_themes(student_text)

    return {
        'original_text': original_text,
        'cleaned_text_light': cleaned_text_light,
        'cleaned_text_aggressive': cleaned_text_aggressive,
        'language': language,
        'structure_analysis': structure_analysis,
        'sentiment': sentiment,
        'polarity': polarity,
        'subjectivity': subjectivity,
        'stats': stats,
        'entities': entities,
        'summary': summary,
        'is_interview': interview,
        'turns': turns,
        'student_text': student_text,
        'interviewer_text': interviewer_text,
        'ypar_themes': ypar_themes,
    }

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    if polarity > 0:
        sentiment = 'Positive'
    elif polarity < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return sentiment, polarity, subjectivity

def get_document_stats(text):
    stats = {
        'Reading Time (minutes)': round(len(text.split()) / 200, 2),
        'Word Count': len(text.split()),
        'Character Count': len(text),
        'Sentence Count': len(text.split('.')),
        'Average Word Length': round(len(text) / len(text.split()), 2),
        'Flesch Reading Ease': textstat.flesch_reading_ease(text),
        'Flesch-Kincaid Grade': textstat.flesch_kincaid_grade(text),
        'Unique Words': len(set(text.split())),
        'Vocabulary Richness': round(len(set(text.split())) / len(text.split()) * 100, 2)
    }
    return stats

def extract_named_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char
        })
    return entities

# --- Header ---
st.markdown("""
<div class="app-header">
    <h1>📄 Advanced Document Analyzer</h1>
    <p>Upload research documents for AI-powered analysis, visualizations, and insights</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## Document Analyzer")
    st.markdown("---")
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["docx", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Supports Word (.docx) and Excel (.xlsx, .xls) files",
    )
    st.markdown("---")
    if st.button("🔄 Clear Cache & Reprocess", help="Run this after uploading new files or if results look stale"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown("### Analysis Options")

uploaded_files = uploaded_files or []

if uploaded_files:
    docs_data = {}
    all_cleaned_texts = []
    
    # Process documents (cached by filename + content hash)
    with st.spinner("Processing documents..."):
        for file in uploaded_files:
            file_bytes = file.read()
            ext = os.path.splitext(file.name)[1].lower()
            result = analyze_document(file.name, file_bytes, ext)
            if result:
                docs_data[file.name] = result
                all_cleaned_texts.append(result['cleaned_text_aggressive'])
    
    # Sidebar: show interview detection badge per document
    interview_docs = [n for n, d in docs_data.items() if d.get('is_interview')]
    if interview_docs:
        st.sidebar.markdown("**Interview transcripts detected:**")
        for name in interview_docs:
            st.sidebar.caption(f"🎙 {name}")
        st.sidebar.markdown("---")

    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type",
        [
            "Document Structure",
            "Content Analysis",
            "Youth Voice",
            "YPAR Theme Tracker",
            "Language Analysis",
            "Comparative Analysis",
            "Advanced Visualizations",
        ],
    )
    
    if analysis_type == "Document Structure":
        doc_choice = st.selectbox("Select Document", list(docs_data.keys()), key="struct_doc")
        doc_name = doc_choice
        doc_data = docs_data[doc_name]

        if doc_data.get('summary'):
            st.markdown(
                f'<div class="summary-card"><strong>Auto Summary</strong><br><br>{doc_data["summary"]}</div>',
                unsafe_allow_html=True,
            )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Structure Analysis")
            structure = doc_data['structure_analysis']

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=structure['avg_sentence_length'],
                title={'text': "Avg Sentence Length (words)"},
                gauge={
                    'axis': {'range': [0, 50]},
                    'bar': {'color': "#4f8ef7"},
                    'steps': [
                        {'range': [0, 15], 'color': "#1a2744"},
                        {'range': [15, 30], 'color': "#1e3a6e"},
                    ],
                },
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=260)
            st.plotly_chart(fig, use_container_width=True)

            m1, m2 = st.columns(2)
            m1.metric("Paragraphs", structure['num_paragraphs'])
            m2.metric("Sentences", structure['num_sentences'])
            m3, m4 = st.columns(2)
            m3.metric("Avg Para Length", f"{structure['avg_paragraph_length']:.1f} w")
            m4.metric("Noise Ratio", f"{structure['noise_ratio']:.2%}")

            if structure['issues']:
                st.markdown("#### Detected Issues")
                for issue in structure['issues']:
                    st.warning(issue)

        with col2:
            st.markdown("### Document Statistics")
            stats_df = pd.DataFrame.from_dict(doc_data['stats'], orient='index', columns=['Value'])

            categories = ['Reading Time', 'Word Count', 'Flesch Reading Ease', 'Vocabulary Richness']
            values = [
                doc_data['stats']['Reading Time (minutes)'],
                doc_data['stats']['Word Count'] / 1000,
                doc_data['stats']['Flesch Reading Ease'] / 100,
                doc_data['stats']['Vocabulary Richness'] / 100,
            ]

            fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself',
                                                  line_color='#4f8ef7', fillcolor='rgba(79,142,247,0.2)'))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1], color='#7db0f7'),
                           bgcolor='rgba(0,0,0,0)'),
                showlegend=False,
                title=dict(text="Metrics Radar", font=dict(color='#e2e8f0')),
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0',
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(stats_df, use_container_width=True)

        st.markdown("### Document Flow Analysis")
        create_temporal_analysis(doc_data['original_text'])

        st.markdown("### Text Versions")
        tabs = st.tabs(["Original", "Light Cleaning", "Aggressive Cleaning"])
        with tabs[0]:
            st.text_area("Original Text", doc_data['original_text'], height=220, key="orig_text")
        with tabs[1]:
            st.text_area("Light Cleaning", doc_data['cleaned_text_light'], height=220, key="light_text")
        with tabs[2]:
            st.text_area("Aggressive Cleaning", doc_data['cleaned_text_aggressive'], height=220, key="agg_text")
            
    
        # Add AI Analysis section for document structure after displaying all documents
        st.markdown("### AI Analysis")
        st.write("Ask our AI about document structure and organization:")
        
        # Multi-document selection
        selected_docs = st.multiselect(
            "Select documents to analyze:",
            list(docs_data.keys())
        )
        
        question = st.text_input("Enter your question about document structure:")
        if question and selected_docs:
            with st.spinner("Getting AI analysis..."):
                # Create structure-focused context for all selected documents
                structure_context = {
                    "documents": {
                        doc_name: {
                            "document_info": {
                                "name": doc_name,
                                "word_count": docs_data[doc_name]['stats']['Word Count'],
                                "reading_grade": docs_data[doc_name]['stats']['Flesch-Kincaid Grade'],
                                "reading_ease": docs_data[doc_name]['stats']['Flesch Reading Ease']
                            },
                            "structure_analysis": docs_data[doc_name]['structure_analysis'],
                            "content": {
                                "original": docs_data[doc_name]['original_text'],
                                "cleaned": docs_data[doc_name]['cleaned_text_light']
                            },
                            "paragraph_metrics": {
                                "total_paragraphs": docs_data[doc_name]['structure_analysis']['num_paragraphs'],
                                "avg_paragraph_length": f"{docs_data[doc_name]['structure_analysis']['avg_paragraph_length']:.1f} words",
                                "total_sentences": docs_data[doc_name]['structure_analysis']['num_sentences'],
                                "avg_sentence_length": f"{docs_data[doc_name]['structure_analysis']['avg_sentence_length']:.1f} words"
                            }
                        }
                        for doc_name in selected_docs
                    },
                    "comparison_metrics": {
                        "total_documents": len(selected_docs),
                        "avg_paragraphs": sum(docs_data[doc]['structure_analysis']['num_paragraphs'] for doc in selected_docs) / len(selected_docs),
                        "avg_reading_grade": sum(docs_data[doc]['stats']['Flesch-Kincaid Grade'] for doc in selected_docs) / len(selected_docs)
                    }
                }
                
                # Enhance prompt for structure analysis
                if len(selected_docs) > 1:
                    structure_prompt = f"""Analyze the following documents: {', '.join(selected_docs)}

                    For each document, you can find its full content in documents[doc_name]['content']['original'].
                    
                    Focus on:
                    1. Document content and key themes (with specific quotes)
                    2. Document organization and flow
                    3. Paragraph and sentence structure patterns
                    4. Readability metrics and comparisons
                    5. Structural similarities and differences
                    6. Common issues or improvement opportunities
                    
                    Question: {question}
                    
                    IMPORTANT: Please reference and quote specific content from the documents to support your analysis. Use actual quotes from the text when discussing themes or patterns."""
                else:
                    structure_prompt = f"""Analyze the document: {selected_docs[0]}

                    The full document content is available in documents['{selected_docs[0]}']['content']['original'].
                    
                    Focus on:
                    1. Document content and key themes (with specific quotes)
                    2. Document organization and flow
                    3. Paragraph and sentence structure
                    4. Readability metrics
                    5. Any structural issues or improvements needed
                    
                    Question: {question}
                    
                    IMPORTANT: Please reference and quote specific content from the document to support your analysis. Use actual quotes from the text when discussing themes or patterns."""
                
                # Get AI response with structure context
                response = get_text_completion(
                    query=structure_prompt,
                    document_context=structure_context
                )
                
                if "error" not in response:
                    # Create a chat-like interface
                    with st.container():
                        ai_text = response.get("response", "No response generated")
                        st.markdown(f'<div class="ai-box"><strong>🤖 AI Assistant</strong><br><br>{ai_text}</div>', unsafe_allow_html=True)
                        
                        # Add analysis options
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Generate Structure Recommendations"):
                                with st.spinner("Analyzing document structure..."):
                                    recommendations_prompt = """
                                    Based on the document content and structural analysis, provide:
                                    1. Content-specific recommendations with example improvements
                                    2. Suggestions for enhancing readability with specific examples
                                    3. Best practices for paragraph and sentence structure
                                    4. Potential restructuring opportunities
                                    
                                    IMPORTANT: Reference specific parts of the text and show how they could be improved.
                                    """
                                    recommendations_response = get_text_completion(
                                        query=recommendations_prompt,
                                        document_context=structure_context
                                    )
                                    if "error" not in recommendations_response:
                                        st.markdown("**📋 Structure Recommendations:**")
                                        st.markdown(recommendations_response.get("response", ""))
                        
                        with col2:
                            if len(selected_docs) > 1 and st.button("Compare Document Structures"):
                                with st.spinner("Generating structural comparison..."):
                                    compare_prompt = """
                                    Provide a detailed comparison of the selected documents:
                                    1. Content themes and key differences (with quotes)
                                    2. Organization patterns and differences
                                    3. Readability and complexity variations
                                    4. Paragraph and sentence structure comparisons
                                    5. Common elements and unique features
                                    
                                    IMPORTANT: Support your analysis with specific quotes and examples from each document.
                                    """
                                    compare_response = get_text_completion(
                                        query=compare_prompt,
                                        document_context=structure_context
                                    )
                                    if "error" not in compare_response:
                                        st.markdown("**📊 Structural Comparison:**")
                                        st.markdown(compare_response.get("response", ""))
                else:
                    st.error(response["error"])
    
    elif analysis_type == "Content Analysis":
        doc_choice = st.selectbox("Select Document", list(docs_data.keys()))
        doc_data = docs_data[doc_choice]
        
        # Enhanced word cloud with customization
        st.markdown("### Word Cloud Analysis")
        st.write("""
        Word clouds provide an intuitive visualization of the most frequent words in your document. 
        Customize the visualization using the controls below.
        """)
        
        # Word cloud customization controls
        with st.expander("Word Cloud Customization"):
            wc_col1, wc_col2 = st.columns(2)
            with wc_col1:
                wc_title = st.text_input("Word Cloud Title", "Word Cloud")
                wc_bg_color = st.color_picker("Background Color", "#FFFFFF")
                wc_width = st.slider("Width", 400, 1200, 800)
            with wc_col2:
                wc_colormap = st.selectbox("Color Scheme", 
                    ["viridis", "plasma", "inferno", "magma", "cividis", "rainbow"])
                wc_height = st.slider("Height", 200, 800, 400)
                
            # Additional stopwords
            wc_stopwords = st.text_area("Additional Stopwords (one per line)", 
                placeholder="Enter words to exclude from the word cloud")
            additional_stopwords = set(word.strip() for word in wc_stopwords.split('\n') if word.strip())
            
        create_wordcloud(
            doc_data['cleaned_text_aggressive'],
            custom_title=wc_title,
            additional_stopwords=additional_stopwords,
            background_color=wc_bg_color,
            colormap=wc_colormap,
            width=wc_width,
            height=wc_height
        )
        
        # Entity network visualization with customization
        st.markdown("### Entity Relationship Network")
        st.write("""
        The entity relationship network shows how different named entities (people, organizations, locations) 
        are connected within your document. Customize the visualization using the controls below.
        """)
        
        # Entity network customization controls
        with st.expander("Entity Network Customization"):
            en_col1, en_col2 = st.columns(2)
            with en_col1:
                en_title = st.text_input("Network Title", "Entity Relationship Network")
                en_node_size = st.slider("Node Size", 10, 50, 20)
                en_edge_width = st.slider("Edge Width", 0.1, 2.0, 0.5, 0.1)
            with en_col2:
                en_colorscale = st.selectbox("Node Color Scheme", 
                    ["YlGnBu", "Viridis", "Plasma", "Inferno", "Magma", "RdBu"])
                en_edge_color = st.color_picker("Edge Color", "#888888")
        
        create_entity_network(
            doc_data['cleaned_text_light'],
            custom_title=en_title,
            node_size=en_node_size,
            edge_width=en_edge_width,
            node_colorscale=en_colorscale,
            edge_color=en_edge_color
        )
        
        # Named entity recognition with enhanced visualization
        st.markdown("### Named Entity Recognition")
        st.write("""
        The treemap visualization displays the distribution of named entities by category. 
        The size of each section represents the frequency of that entity, helping you understand 
        which entities are most prominent in your document.
        """)
        entity_counts = Counter(entity['text'].lower() for entity in doc_data['entities'])
        
        fig = px.treemap(
            names=list(entity_counts.keys()),
            parents=["Entities"] * len(entity_counts),
            values=list(entity_counts.values()),
            title="Entity Distribution"
        )
        st.plotly_chart(fig)
        
        # Topic modeling visualization with customization
        st.markdown("### Topic Analysis")
        st.write("""
        The topic-keyword heatmap shows the relationship between discovered topics and their most 
        relevant keywords. Customize the visualization using the controls below.
        """)
        
        # Topic visualization customization controls
        with st.expander("Topic Analysis Customization"):
            tp_col1, tp_col2 = st.columns(2)
            with tp_col1:
                tp_title = st.text_input("Topic Heatmap Title", "Topic-Keyword Heatmap")
                tp_num_topics = st.slider("Number of Topics", 2, 10, 5)
            with tp_col2:
                tp_colorscale = st.selectbox("Color Scheme", 
                    ["Viridis", "Plasma", "Inferno", "Magma", "RdBu", "YlOrRd"])
                tp_keywords = st.slider("Keywords per Topic", 3, 10, 5)
        
        create_topic_visualization(
            [doc_data['cleaned_text_aggressive']],
            num_topics=tp_num_topics,
            custom_title=tp_title,
            color_scale=tp_colorscale,
            keywords_per_topic=tp_keywords
        )
        
        # Add chatbot analysis with multi-document support
        st.markdown("### AI Analysis")
        st.write("Ask our AI to analyze documents:")
        
        # Multi-document selection
        selected_docs = st.multiselect(
            "Select documents to analyze:",
            list(docs_data.keys()),
            default=[doc_choice]
        )
        
        question = st.text_input("Enter your question about the document(s):")
        if question and selected_docs:
            with st.spinner("Getting AI analysis..."):
                # Create context for all selected documents
                multi_doc_context = {
                    "documents": {
                        doc_name: {
                            "document_info": {
                                "name": doc_name,
                                "word_count": docs_data[doc_name]['stats']['Word Count'],
                                "reading_grade": docs_data[doc_name]['stats']['Flesch-Kincaid Grade'],
                                "sentiment": docs_data[doc_name]['sentiment'],
                                "polarity": f"{docs_data[doc_name]['polarity']:.2f}",
                                "vocabulary_richness": f"{docs_data[doc_name]['stats']['Vocabulary Richness']}%"
                            },
                            "structure_analysis": docs_data[doc_name]['structure_analysis'],
                            "entities": docs_data[doc_name]['entities'],
                            "content": {
                                "original": docs_data[doc_name]['original_text'],
                                "cleaned": docs_data[doc_name]['cleaned_text_light']
                            }
                        }
                        for doc_name in selected_docs
                    },
                    "comparison_metrics": {
                        "total_documents": len(selected_docs),
                        "avg_word_count": sum(docs_data[doc]['stats']['Word Count'] for doc in selected_docs) / len(selected_docs),
                        "avg_reading_grade": sum(docs_data[doc]['stats']['Flesch-Kincaid Grade'] for doc in selected_docs) / len(selected_docs)
                    }
                }
                
                # Enhance prompt for multi-document analysis
                if len(selected_docs) > 1:
                    enhanced_question = f"""Analyze the following {len(selected_docs)} documents: {', '.join(selected_docs)}
                    
                    Each document's content is available in the context under documents[doc_name]['content'].
                    Consider relationships, patterns, and differences between the documents.
                    
                    Question: {question}
                    
                    Please reference specific content from the documents in your analysis."""
                else:
                    enhanced_question = f"""Analyze the document: {selected_docs[0]}
                    
                    The document's content is available in the context under documents['{selected_docs[0]}']['content'].
                    
                    Question: {question}
                    
                    Please reference specific content from the document in your analysis."""
                
                # Get AI response with multi-document context
                response = get_text_completion(
                    query=enhanced_question,
                    document_context=multi_doc_context
                )
                
                if "error" not in response:
                    # Create a chat-like interface
                    with st.container():
                        ai_text = response.get("response", "No response generated")
                        st.markdown(f'<div class="ai-box"><strong>🤖 AI Assistant</strong><br><br>{ai_text}</div>', unsafe_allow_html=True)
                        
                        # Add follow-up analysis options
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Generate Detailed Analysis"):
                                with st.spinner("Generating detailed analysis..."):
                                    detailed_prompt = f"""
                                    Based on the previous analysis of {len(selected_docs)} document(s), provide a detailed breakdown of:
                                    1. Key insights and patterns
                                    2. Notable metrics and their significance
                                    3. Recommendations for further analysis
                                    
                                    Previous question: {question}
                                    """
                                    detailed_response = get_text_completion(
                                        query=detailed_prompt,
                                        document_context=multi_doc_context
                                    )
                                    if "error" not in detailed_response:
                                        st.markdown("**🔍 Detailed Analysis:**")
                                        st.markdown(detailed_response.get("response", ""))
                        
                        with col2:
                            if len(selected_docs) > 1 and st.button("Compare Documents"):
                                with st.spinner("Generating comparison..."):
                                    compare_prompt = f"""
                                    Provide a detailed comparison of the selected documents:
                                    1. Key similarities and differences
                                    2. Unique characteristics of each document
                                    3. Overall patterns across the document set
                                    
                                    Documents: {', '.join(selected_docs)}
                                    """
                                    compare_response = get_text_completion(
                                        query=compare_prompt,
                                        document_context=multi_doc_context
                                    )
                                    if "error" not in compare_response:
                                        st.markdown("**📊 Document Comparison:**")
                                        st.markdown(compare_response.get("response", ""))
                else:
                    st.error(response["error"])
    
    elif analysis_type == "Youth Voice":
        interviews = {n: d for n, d in docs_data.items() if d.get('is_interview')}
        non_interviews = {n: d for n, d in docs_data.items() if not d.get('is_interview')}

        if non_interviews and not interviews:
            st.info("No interview transcripts detected in your uploaded documents. "
                    "Youth Voice analysis works best with interview files that use "
                    "speaker labels like  'L:', 'D:', or a name followed by a colon.")

        target_docs = interviews if interviews else docs_data
        doc_choice = st.selectbox("Select Interview", list(target_docs.keys()), key="yv_doc")
        doc_data = target_docs[doc_choice]
        turns_raw = doc_data.get('turns', [])

        if turns_raw:
            # ── Speaker override ────────────────────────────────────────────
            all_speakers = sorted({t['speaker'] for t in turns_raw})
            auto_interviewers = [s for s in all_speakers if _is_interviewer(s)]

            with st.expander("⚙️ Speaker Settings — adjust if labels are wrong", expanded=False):
                st.caption(
                    "Auto-detected based on speaker label format. "
                    "Single letters (L, S, I) and full names (Stephen Caviness) are assumed to be interviewers. "
                    "Override here if needed."
                )
                interviewer_override = st.multiselect(
                    "Treat these speakers as INTERVIEWERS:",
                    options=all_speakers,
                    default=auto_interviewers,
                    key=f"iv_{doc_choice}",
                )

            override_set = set(interviewer_override)
            # Re-label turns using the override (no cache needed — done in UI)
            turns = [{**t, 'is_student': t['speaker'] not in override_set} for t in turns_raw]

            # Recompute student text and theme scores from the override
            student_text_override = get_student_text(turns)
            theme_scores_override = score_ypar_themes(student_text_override)

            # Build interviewer stopwords to block from word cloud
            interviewer_name_words = set()
            for sp in override_set:
                for part in sp.split():
                    interviewer_name_words.add(part.lower())

            student_speakers = {t['speaker'] for t in turns if t['is_student']}
            interviewer_speakers = override_set
            student_turns = [t for t in turns if t['is_student']]
            interviewer_turns = [t for t in turns if not t['is_student']]

            # ── Turn breakdown ──────────────────────────────────────────────
            st.markdown("### Speaker Turn Breakdown")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Turns", len(turns))
            c2.metric("Student Turns", len(student_turns))
            c3.metric("Interviewer Turns", len(interviewer_turns))
            avg_student_words = (
                sum(len(t['text'].split()) for t in student_turns) / len(student_turns)
                if student_turns else 0
            )
            c4.metric("Avg Student Words/Turn", f"{avg_student_words:.0f}")

            fig_turns = go.Figure(go.Pie(
                labels=["Student", "Interviewer"],
                values=[len(student_turns), len(interviewer_turns)],
                hole=0.45,
                marker_colors=["#4f8ef7", "#48bb78"],
            ))
            fig_turns.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                showlegend=True, height=280,
                title=dict(text="Turn Distribution", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(fig_turns, use_container_width=True)

            # ── Student voice word cloud ────────────────────────────────────
            st.markdown("### Student Voice Word Cloud")
            if student_text_override.strip():
                create_wordcloud(
                    student_text_override,
                    custom_title="Student Voice — Most Frequent Words",
                    additional_stopwords=interviewer_name_words,
                )

            # ── Transcript viewer ───────────────────────────────────────────
            st.markdown("### Transcript")
            view_mode = st.radio(
                "Show turns from:", ["Student only", "Interviewer only", "Full transcript"],
                horizontal=True, key="yv_view",
            )
            filtered = (
                student_turns if view_mode == "Student only"
                else interviewer_turns if view_mode == "Interviewer only"
                else turns
            )
            for turn in filtered:
                color = "#4f8ef7" if turn['is_student'] else "#48bb78"
                st.markdown(
                    f'<div style="border-left:3px solid {color};padding:0.4rem 0.8rem;'
                    f'margin:0.3rem 0;border-radius:0 6px 6px 0;">'
                    f'<strong style="color:{color}">{turn["speaker"]}</strong>&nbsp;&nbsp;'
                    f'{turn["text"]}</div>',
                    unsafe_allow_html=True,
                )

            # ── YPAR theme scores for this student ──────────────────────────
            st.markdown("### YPAR Theme Presence (Student Voice)")
            theme_scores = doc_data.get('ypar_themes', {})
            if theme_scores_override:
                theme_df = pd.DataFrame(
                    list(theme_scores_override.items()), columns=["Theme", "Score"]
                ).sort_values("Score", ascending=True)
                fig_theme = px.bar(
                    theme_df, x="Score", y="Theme", orientation="h",
                    color="Score", color_continuous_scale="Blues",
                    title="Keyword Hits per YPAR Theme",
                )
                fig_theme.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                    showlegend=False, coloraxis_showscale=False,
                    xaxis_title="Keyword hits", yaxis_title="",
                )
                st.plotly_chart(fig_theme, use_container_width=True)

            # ── AI quote bank ───────────────────────────────────────────────
            st.markdown("### AI Quote Bank")
            st.caption("Click to extract the most meaningful student quotes, organized by YPAR theme.")
            if st.button("Extract Key Quotes", key="yv_quotes"):
                with st.spinner("Claude is reading the student's words..."):
                    quote_prompt = (
                        f"You are analyzing a YPAR (Youth Participatory Action Research) interview. "
                        f"Below is what the STUDENT said (interviewer turns removed).\n\n"
                        f"Student text:\n{student_text_override[:6000]}\n\n"
                        f"Task: Extract 2-3 of the most significant, direct quotes from the student "
                        f"for each of these CPIM/YPAR themes where relevant: "
                        f"Spatial Justice, Agency & Voice, Systemic Inequity, Mathematical Thinking, "
                        f"Community, Educational Experience, Identity & Culture, Participatory Action.\n\n"
                        f"Format each theme as a markdown header with bullet-point quotes. "
                        f"Only include themes where you found real evidence. Use the student's exact words."
                    )
                    qr = get_text_completion(query=quote_prompt)
                    if "error" not in qr:
                        st.markdown(
                            f'<div class="ai-box">{qr.get("response","")}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error(qr["error"])

            # ── AI open question ────────────────────────────────────────────
            st.markdown("### Ask About This Student")
            question = st.text_input("Your question:", key="yv_q",
                                      placeholder="What does this student think about their school community?")
            if question:
                with st.spinner("Analyzing..."):
                    ctx = {
                        "student_speaker": ", ".join(student_speakers),
                        "student_text_sample": student_text_override[:5000],
                        "ypar_theme_scores": theme_scores_override,
                        "total_student_turns": len(student_turns),
                    }
                    resp = get_text_completion(
                        query=f"YPAR interview analysis. Student speaker(s): {', '.join(student_speakers)}.\n\nQuestion: {question}",
                        document_context=ctx,
                    )
                    if "error" not in resp:
                        st.markdown(
                            f'<div class="ai-box">{resp.get("response","")}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error(resp["error"])
        else:
            st.info("This document wasn't detected as an interview transcript. "
                    "Make sure speaker turns use a format like  'L: text' or 'Willow: text'.")

    elif analysis_type == "YPAR Theme Tracker":
        st.markdown("### YPAR / CPIM Theme Tracker")
        st.caption(
            "Heatmap of CPIM framework themes across all uploaded interviews — "
            "student voice only. Higher values = more keyword hits."
        )

        # Build theme matrix — prefer student text, fall back to full text
        theme_rows = []
        for name, data in docs_data.items():
            scores = data.get('ypar_themes') or score_ypar_themes(
                data.get('student_text') or data.get('cleaned_text_light', '')
            )
            row = {"Document": name.replace('.docx', '').replace('.doc', '')}
            row.update(scores)
            theme_rows.append(row)

        theme_matrix_df = pd.DataFrame(theme_rows).set_index("Document")

        # Heatmap
        fig_heat = px.imshow(
            theme_matrix_df,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto",
            title="YPAR Theme Keyword Frequency (Student Voice)",
            labels=dict(x="Theme", y="Interview", color="Hits"),
        )
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
            xaxis_tickangle=-35,
            coloraxis_colorbar=dict(title="Hits", tickfont=dict(color="#e2e8f0")),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Dominant theme per student
        st.markdown("### Dominant Theme per Interview")
        dominant = []
        for name, data in docs_data.items():
            scores = data.get('ypar_themes', {})
            if scores:
                top = max(scores, key=scores.get)
                dominant.append({
                    "Interview": name.replace('.docx', ''),
                    "Dominant Theme": top,
                    "Score": scores[top],
                })
        if dominant:
            dom_df = pd.DataFrame(dominant)
            fig_dom = px.bar(
                dom_df, x="Interview", y="Score", color="Dominant Theme",
                title="Strongest YPAR Theme per Student",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig_dom.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                xaxis_tickangle=-20, legend_title_text="Theme",
            )
            st.plotly_chart(fig_dom, use_container_width=True)

        # Cross-interview theme totals
        st.markdown("### Theme Totals Across All Interviews")
        totals = theme_matrix_df.sum().sort_values(ascending=False).reset_index()
        totals.columns = ["Theme", "Total Hits"]
        fig_tot = px.bar(
            totals, x="Total Hits", y="Theme", orientation="h",
            color="Total Hits", color_continuous_scale="Blues",
            title="Most Present Themes Across Corpus",
        )
        fig_tot.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
            showlegend=False, coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_tot, use_container_width=True)

        # Dataframe download
        with st.expander("Raw theme scores table"):
            st.dataframe(theme_matrix_df, use_container_width=True)

        # AI cross-interview synthesis
        st.markdown("### AI Cross-Interview Synthesis")
        st.caption("Ask Claude to synthesize patterns across the full interview corpus.")
        synth_q = st.text_input(
            "Question for Claude:", key="ypar_synth_q",
            placeholder="What patterns emerge across students around spatial justice?",
        )
        if synth_q:
            with st.spinner("Synthesizing across all interviews..."):
                corpus_lines = []
                for name, data in docs_data.items():
                    scores = data.get('ypar_themes', {})
                    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    corpus_lines.append(
                        f"- {name.replace('.docx', '')}: "
                        + ", ".join(f"{t} ({s} hits)" for t, s in top3)
                    )

                ctx = {
                    "corpus_theme_summary": "\n".join(corpus_lines),
                    "theme_matrix": theme_matrix_df.to_string(),
                    "num_interviews": len(docs_data),
                }
                resp = get_text_completion(
                    query=(
                        f"You are analyzing a YPAR research corpus of {len(docs_data)} student interviews "
                        f"through the CPIM (Computational-Participatory Intersection Model) framework.\n\n"
                        f"Question: {synth_q}"
                    ),
                    document_context=ctx,
                )
                if "error" not in resp:
                    st.markdown(
                        f'<div class="ai-box">{resp.get("response","")}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(resp["error"])

    elif analysis_type == "Language Analysis":
        st.markdown("### Language Detection Results")
        st.write("""
        The pie chart shows the distribution of detected languages across your documents. 
        This is particularly useful when analyzing multilingual document collections or 
        verifying the language consistency of your corpus.
        """)
        lang_data = [{"Document": name, "Language": data['language']} 
                    for name, data in docs_data.items()]
        lang_df = pd.DataFrame(lang_data)
        
        fig = px.pie(lang_df, names='Language', title="Language Distribution")
        st.plotly_chart(fig)
        
        st.markdown("### Sentiment Analysis")
        st.write("""
        The sentiment analysis visualizations show both the polarity (positive/negative) and 
        subjectivity of your documents. The scatter plot helps identify emotional patterns, 
        while the bar chart compares sentiment metrics across documents.
        """)
        sentiment_data = []
        for doc_name, doc_data in docs_data.items():
            sentiment_data.append({
                'Document': doc_name,
                'Sentiment': doc_data['sentiment'],
                'Polarity': doc_data['polarity'],
                'Subjectivity': doc_data['subjectivity']
            })
        
        sentiment_df = pd.DataFrame(sentiment_data)
        
        # Enhanced sentiment visualization
        fig = px.scatter(
            sentiment_df,
            x='Polarity',
            y='Subjectivity',
            color='Sentiment',
            size=[1] * len(sentiment_df),
            text='Document',
            title="Document Sentiment Analysis",
            labels={'Polarity': 'Sentiment Polarity (-1 to 1)',
                   'Subjectivity': 'Subjectivity (0 to 1)'},
            color_discrete_map={
                'Positive': 'green',
                'Negative': 'red',
                'Neutral': 'gray'
            }
        )
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig)
        
        # Sentiment distribution
        fig = px.bar(
            sentiment_df,
            x='Document',
            y=['Polarity', 'Subjectivity'],
            title="Sentiment Metrics by Document",
            barmode='group'
        )
        st.plotly_chart(fig)
        
# Add chatbot analysis with context
        st.markdown("### AI Analysis")
        st.write("Ask our AI about sentiment patterns in your documents:")
        
        question = st.text_input("Enter your question about sentiment analysis:")
        if question:
            with st.spinner("Getting AI analysis..."):
                # Create sentiment context
                sentiment_context = {
                    "sentiment_analysis": {
                        doc: {
                            "sentiment": row["Sentiment"],
                            "polarity": f"{row['Polarity']:.2f}",
                            "subjectivity": f"{row['Subjectivity']:.2f}"
                        }
                        for doc, row in sentiment_df.iterrows()
                    },
                    "overall_stats": {
                        "avg_polarity": f"{sentiment_df['Polarity'].mean():.2f}",
                        "avg_subjectivity": f"{sentiment_df['Subjectivity'].mean():.2f}",
                        "document_count": len(sentiment_df)
                    }
                }
                
                # Get AI response with context
                response = get_text_completion(
                    query=question,
                    document_context=sentiment_context
                )
                
                if "error" not in response:
                    # Create a chat-like interface
                    with st.container():
                        ai_text = response.get("response", "No response generated")
                        st.markdown(f'<div class="ai-box"><strong>🤖 AI Assistant</strong><br><br>{ai_text}</div>', unsafe_allow_html=True)
                        
                        # Add comparative analysis option
                        if st.button("Generate Comparative Analysis"):
                            with st.spinner("Generating comparison..."):
                                compare_prompt = """
                                Provide a detailed comparison of sentiment patterns across documents:
                                1. Identify significant differences
                                2. Highlight common patterns
                                3. Suggest potential factors influencing sentiment variations
                                """
                                compare_response = get_text_completion(
                                    query=compare_prompt,
                                    document_context=sentiment_context
                                )
                                if "error" not in compare_response:
                                    st.markdown("**📊 Comparative Analysis:**")
                                    st.markdown(compare_response.get("response", ""))
                else:
                    st.error(response["error"])
    
    elif analysis_type == "Comparative Analysis":
        if len(docs_data) > 1:
            st.markdown("### Document Comparison")
            st.write("""
            The parallel coordinates plot shows how documents compare across multiple dimensions simultaneously.
            Each vertical axis represents a different metric, and lines connecting these axes represent individual
            documents. This helps identify patterns and relationships between different document characteristics.
            """)
            
            # Create comparison dataframe
            comparison_data = []
            for doc_name, doc_data in docs_data.items():
                comparison_data.append({
                    'Document': doc_name,
                    'Word Count': doc_data['stats']['Word Count'],
                    'Noise Ratio': doc_data['structure_analysis']['noise_ratio'],
                    'Avg Paragraph Length': doc_data['structure_analysis']['avg_paragraph_length'],
                    'Reading Grade': doc_data['stats']['Flesch-Kincaid Grade']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Parallel coordinates plot
            fig = px.parallel_coordinates(
                comparison_df,
                dimensions=['Word Count', 'Noise Ratio', 'Avg Paragraph Length', 'Reading Grade'],
                title="Multi-dimensional Document Comparison"
            )
            st.plotly_chart(fig)
            
            st.dataframe(comparison_df)
            
            # Enhanced metric comparison
            st.write("""
            The bar chart provides a detailed comparison of specific metrics across documents.
            This visualization makes it easy to spot differences and trends in individual measures.
            """)
            metrics = ['Word Count', 'Noise Ratio', 'Avg Paragraph Length', 'Reading Grade']
            selected_metric = st.selectbox("Select Metric for Comparison", metrics)
            
            fig = px.bar(
                comparison_df,
                x='Document',
                y=selected_metric,
                title=f"Document Comparison - {selected_metric}",
                color='Document',
                text=selected_metric
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig)
            
            # Document similarity matrix with enhanced visualization
            st.markdown("### Document Similarity Analysis")
            st.write("""
            The similarity matrix shows how similar documents are to each other based on their content.
            Darker colors indicate higher similarity. This helps identify groups of related documents
            and outliers in your collection.
            """)
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(all_cleaned_texts)
            similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
            
            similarity_df = pd.DataFrame(
                similarity_matrix,
                index=list(docs_data.keys()),
                columns=list(docs_data.keys())
            )
            
            fig = px.imshow(
                similarity_df,
                title="Document Similarity Matrix",
                labels=dict(x="Document", y="Document", color="Similarity"),
                color_continuous_scale="RdYlBu",
                aspect="auto"
            )
            fig.update_traces(text=similarity_df.values, texttemplate="%{z:.2f}")
            st.plotly_chart(fig)
            
            # Hierarchical clustering of documents
            if len(docs_data) > 2:
                st.markdown("### Document Clustering")
                st.write("""
                The dendrogram shows hierarchical relationships between documents based on their content similarity.
                Documents that cluster together early (lower in the tree) are more similar to each other.
                This helps identify natural groupings in your document collection.
                """)
                linkage_matrix = ff.create_dendrogram(similarity_matrix)
                st.plotly_chart(linkage_matrix)
            
            # Add chatbot analysis
            st.markdown("### AI Analysis")
            st.write("Ask our AI to compare specific aspects of your documents:")
            
            question = st.text_input("Enter your question about document comparison:")
            if question:
                with st.spinner("Getting AI analysis..."):
                    # Format comparison data in a readable way
                    comparison_summary = "Document Comparison:\n"
                    for _, row in comparison_df.iterrows():
                        comparison_summary += f"\n{row['Document']}:\n"
                        comparison_summary += f"- Word Count: {row['Word Count']}\n"
                        comparison_summary += f"- Reading Grade: {row['Reading Grade']:.1f}\n"
                        comparison_summary += f"- Noise Ratio: {row['Noise Ratio']:.2%}\n"
                    
                    # Add similarity information
                    comparison_summary += "\nDocument Similarities:\n"
                    for idx1, doc1 in enumerate(similarity_df.index):
                        for idx2, doc2 in enumerate(similarity_df.columns):
                            if idx1 < idx2:  # Only show each pair once
                                similarity = similarity_df.iloc[idx1, idx2]
                                comparison_summary += f"{doc1} <-> {doc2}: {similarity:.2%} similar\n"
                    
                    # Combine the question with formatted comparison data
                    prompt = f"Question: {question}\n\n{comparison_summary}"
                    response = get_text_completion(prompt)
                    if "error" not in response:
                        ai_text = response.get("response", "No response generated")
                        st.markdown(f'<div class="ai-box"><strong>🤖 AI Assistant</strong><br><br>{ai_text}</div>', unsafe_allow_html=True)
                    else:
                        st.error(response["error"])
        else:
            st.write("Upload multiple documents for comparison")
    
    elif analysis_type == "Advanced Visualizations":
        doc_choice = st.selectbox("Select Document", list(docs_data.keys()))
        doc_data = docs_data[doc_choice]
        
        st.markdown("### Document Flow Analysis")
        st.write("""
        The document flow analysis shows how various metrics change throughout your document.
        Track sentence length, entity density, and sentiment shifts to understand the document's
        rhythm and structure. This helps identify patterns in writing style and content delivery.
        """)
        create_temporal_analysis(doc_data['cleaned_text_light'])
        
        st.markdown("### Topic Evolution")
        st.write("""
        The topic evolution visualization reveals how different themes emerge and develop in your text.
        Each topic is represented by its most relevant keywords, helping you understand the main
        subjects and their relationships within your document.
        """)
        create_topic_visualization([doc_data['cleaned_text_aggressive']], num_topics=3)
        
        st.markdown("### Entity Network Analysis")
        st.write("""
        The entity network provides an interactive view of how named entities are connected.
        Explore relationships between people, organizations, and locations mentioned in your text.
        This helps uncover complex relationships and key players in your document.
        """)
        create_entity_network(doc_data['cleaned_text_light'])
        
        st.markdown("### Word Distribution")
        st.write("""
        The word cloud offers a visually striking representation of word frequencies.
        More frequent words appear larger, giving you an immediate sense of the most
        important terms and themes in your document.
        """)
        create_wordcloud(doc_data['cleaned_text_aggressive'])
        
        # Add chatbot analysis with context
        st.markdown("### AI Analysis")
        st.write("Ask our AI about patterns and insights from these visualizations:")
        
        question = st.text_input("Enter your question about the visualizations:")
        if question:
            with st.spinner("Getting AI analysis..."):
                # Create visualization context
                viz_context = {
                    "document_info": {
                        "name": doc_choice,
                        "word_count": doc_data['stats']['Word Count'],
                        "unique_words": doc_data['stats']['Unique Words'],
                        "vocabulary_richness": f"{doc_data['stats']['Vocabulary Richness']}%"
                    },
                    "structure_metrics": {
                        "avg_sentence_length": f"{np.mean(doc_data['structure_analysis']['avg_sentence_length']):.1f}",
                        "total_sentences": doc_data['structure_analysis']['num_sentences'],
                        "num_paragraphs": doc_data['structure_analysis']['num_paragraphs']
                    },
                    "entity_analysis": {
                        "total_entities": len(doc_data['entities']),
                        "entity_types": list(set(entity['label'] for entity in doc_data['entities']))
                    },
                    "topic_analysis": {
                        "num_topics": 3,
                        "content_sample": doc_data['cleaned_text_aggressive'][:500] + "... [truncated]"
                    }
                }
                
                # Get AI response with context
                response = get_text_completion(
                    query=question,
                    document_context=viz_context
                )
                
                if "error" not in response:
                    # Create a chat-like interface
                    with st.container():
                        ai_text = response.get("response", "No response generated")
                        st.markdown(f'<div class="ai-box"><strong>🤖 AI Assistant</strong><br><br>{ai_text}</div>', unsafe_allow_html=True)
                        
                        # Add visualization insights option
                        if st.button("Generate Visual Insights"):
                            with st.spinner("Analyzing visualizations..."):
                                insights_prompt = """
                                Provide detailed insights about the visualizations:
                                1. Key patterns and trends
                                2. Notable correlations
                                3. Suggestions for additional visualizations
                                """
                                insights_response = get_text_completion(
                                    query=insights_prompt,
                                    document_context=viz_context
                                )
                                if "error" not in insights_response:
                                    st.markdown("**📈 Visual Insights:**")
                                    st.markdown(insights_response.get("response", ""))
                else:
                    st.error(response["error"])

else:
    st.info("Upload one or more documents using the sidebar to get started.")
    st.markdown("""
    **Supported formats:** Word (`.docx`) · Excel (`.xlsx`, `.xls`)

    **What you get:**
    - Auto-generated extractive summary
    - Structure & readability metrics
    - Sentiment analysis and language detection
    - Named entity recognition and relationship network
    - Topic modeling heatmap
    - Document flow (sentence length / entity density / sentiment over time)
    - Cross-document similarity and clustering
    - AI-powered Q&A on any document or set of documents
    """)
