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

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

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

def create_wordcloud(text: str, custom_title: str = "Word Cloud", additional_stopwords: set = set(), 
                    background_color: str = 'white', colormap: str = 'viridis', 
                    width: int = 800, height: int = 400):
    """Create and display an enhanced word cloud visualization with customization options."""
    # Combine default stopwords with user-provided ones
    all_stopwords = custom_stopwords.union(STOPWORDS).union(additional_stopwords)
    
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        stopwords=all_stopwords,
        min_font_size=10,
        max_font_size=150,
        random_state=42
    ).generate(text)
    
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(custom_title, pad=20, fontsize=14)
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
    import math
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

def analyze_document(file) -> Dict:
    """Process and analyze a document with enhanced cleaning and structure analysis."""
    try:
        if file.name.endswith(('.xlsx', '.xls')):
            # Read Excel file
            df = pd.read_excel(BytesIO(file.read()))
            # Convert DataFrame to text
            original_text = '\n\n'.join([
                f"{col}:\n{'\n'.join(df[col].astype(str))}"
                for col in df.columns
            ])
        else:
            # Process DOCX file
            doc = Document(BytesIO(file.read()))
            original_text = '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error processing {file.name}: {e}")
        return None

    # Detect language
    language = detect_language(original_text)
    
    # Analyze structure
    structure_analysis = analyze_text_structure(original_text)
    
    # Clean text (both aggressive and light versions)
    cleaned_text_light = clean_text(original_text, aggressive=False)
    cleaned_text_aggressive = clean_text(original_text, aggressive=True)

    # Perform sentiment analysis
    sentiment, polarity, subjectivity = analyze_sentiment(cleaned_text_aggressive)
    
    # Get document statistics
    stats = get_document_stats(cleaned_text_light)
    
    # Extract named entities
    entities = extract_named_entities(cleaned_text_light)
    
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
        'entities': entities
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

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: darkblue;'>Advanced Document Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload and analyze documents with advanced cleaning and structure analysis</p>", unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload Documents (.docx, .xlsx, .xls)", type=["docx", "xlsx", "xls"], accept_multiple_files=True)

if uploaded_files:
    docs_data = {}
    all_cleaned_texts = []
    
    # Process documents
    with st.spinner("Processing documents..."):
        for file in uploaded_files:
            result = analyze_document(file)
            if result:
                docs_data[file.name] = result
                all_cleaned_texts.append(result['cleaned_text_aggressive'])
    
    # Detect outliers
    outliers = detect_outliers(all_cleaned_texts)
    if outliers:
        st.warning("⚠️ Potential outlier documents detected:")
        for idx in outliers:
            st.write(f"- {list(docs_data.keys())[idx]}")
    
    # Navigation
    st.sidebar.markdown("### Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Document Structure", "Content Analysis", "Language Analysis", "Comparative Analysis", "Advanced Visualizations"]
    )
    
    if analysis_type == "Document Structure":
        for doc_name, doc_data in docs_data.items():
            st.markdown(f"## {doc_name}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Structure Analysis")
                structure = doc_data['structure_analysis']
                
                # Create gauge chart for structure metrics
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = structure['avg_sentence_length'],
                    title = {'text': "Average Sentence Length"},
                    gauge = {'axis': {'range': [0, 50]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 20], 'color': "lightgray"},
                                {'range': [20, 35], 'color': "gray"}]}))
                st.plotly_chart(fig)
                
                # Structure metrics
                metrics = {
                    "Paragraphs": structure['num_paragraphs'],
                    "Avg Paragraph Length": f"{structure['avg_paragraph_length']:.1f}",
                    "Sentences": structure['num_sentences'],
                    "Noise Ratio": f"{structure['noise_ratio']:.2%}"
                }
                
                for metric, value in metrics.items():
                    st.metric(metric, value)
                
                if structure['issues']:
                    st.markdown("#### Structure Issues")
                    for issue in structure['issues']:
                        st.warning(issue)
            
            with col2:
                st.markdown("### Document Statistics")
                stats_df = pd.DataFrame.from_dict(doc_data['stats'], orient='index', columns=['Value'])
                
                # Create radar chart for key metrics
                categories = ['Reading Time', 'Word Count', 'Flesch Reading Ease', 'Vocabulary Richness']
                values = [
                    doc_data['stats']['Reading Time (minutes)'],
                    doc_data['stats']['Word Count'] / 1000,  # Normalize
                    doc_data['stats']['Flesch Reading Ease'] / 100,  # Normalize
                    doc_data['stats']['Vocabulary Richness'] / 100  # Already normalized
                ]
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=False,
                    title="Document Metrics Overview"
                )
                st.plotly_chart(fig)
                
                st.dataframe(stats_df)
            
            # Text versions with enhanced display
            st.markdown("### Text Versions")
            tabs = st.tabs(["Original", "Light Cleaning", "Aggressive Cleaning"])
            with tabs[0]:
                st.text_area("Original Text", doc_data['original_text'], height=200)
                
                # Temporal analysis customization
                with st.expander("Document Flow Customization"):
                    ta_col1, ta_col2 = st.columns(2)
                    with ta_col1:
                        ta_title = st.text_input("Flow Analysis Title", "Document Flow Analysis")
                        ta_x_label = st.text_input("X-Axis Label", "Sentence Number")
                    with ta_col2:
                        ta_y_label = st.text_input("Y-Axis Label", "Metric Value")
                        
                    # Line colors and styles
                    ta_colors = {
                        'sentence_length': st.color_picker("Sentence Length Color", "#0000FF"),
                        'entity_count': st.color_picker("Entity Count Color", "#FF0000"),
                        'sentiment_score': st.color_picker("Sentiment Score Color", "#00FF00")
                    }
                    
                    ta_styles = {
                        'sentence_length': st.selectbox("Sentence Length Style", 
                            ["solid", "dash", "dot", "dashdot"], key="sl_style"),
                        'entity_count': st.selectbox("Entity Count Style",
                            ["solid", "dash", "dot", "dashdot"], key="ec_style"),
                        'sentiment_score': st.selectbox("Sentiment Score Style",
                            ["solid", "dash", "dot", "dashdot"], key="ss_style")
                    }
                
                create_temporal_analysis(
                    doc_data['original_text'],
                    custom_title=ta_title,
                    colors=ta_colors,
                    line_styles=ta_styles,
                    x_label=ta_x_label,
                    y_label=ta_y_label
                )
            with tabs[1]:
                st.text_area("Light Cleaning", doc_data['cleaned_text_light'], height=200)
            with tabs[2]:
                st.text_area("Aggressive Cleaning", doc_data['cleaned_text_aggressive'], height=200)
            
    
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
                        st.markdown("**🤖 AI Assistant:**")
                        st.markdown(response.get("response", "No response generated"))
                        
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
                        st.markdown("**🤖 AI Assistant:**")
                        st.markdown(response.get("response", "No response generated"))
                        
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
                        st.markdown("**🤖 AI Assistant:**")
                        st.markdown(response.get("response", "No response generated"))
                        
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
                        st.write(response.get("response", "No response generated"))
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
                        st.markdown("**🤖 AI Assistant:**")
                        st.markdown(response.get("response", "No response generated"))
                        
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
    st.write("Please upload documents to analyze.")
