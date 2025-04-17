# Advanced Document Analyzer

A powerful Streamlit application for analyzing documents with AI-powered insights and visualizations. Supports both Word documents (.docx) and Excel files (.xlsx, .xls).

## Features

- **Multi-format Support**
  - Word documents (.docx)
  - Excel files (.xlsx, .xls)

- **Document Analysis**
  - Structure analysis
  - Content analysis
  - Language detection
  - Sentiment analysis
  - Named entity recognition
  - Topic modeling

- **Interactive Visualizations**
  - Customizable Word Clouds
  - Entity Relationship Networks
  - Topic-Keyword Heatmaps
  - Document Flow Analysis
  - Sentiment Analysis Plots
  - Document Comparison Tools

- **AI-Powered Insights**
  - Natural language queries
  - Content recommendations
  - Structure analysis
  - Comparative analysis

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Text_Analyzer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Launch the application
2. Upload one or more documents (.docx, .xlsx, .xls)
3. Choose an analysis type from the sidebar:
   - Document Structure
   - Content Analysis
   - Language Analysis
   - Comparative Analysis
   - Advanced Visualizations
4. Use the customization options to adjust visualizations:
   - Modify colors, sizes, and styles
   - Add/remove stopwords
   - Adjust topic numbers and keywords
   - Customize chart titles and labels

## Visualization Customization

### Word Cloud
- Title customization
- Background color selection
- Color scheme options
- Size adjustments
- Custom stopwords

### Entity Network
- Title customization
- Node size and colors
- Edge width and colors
- Layout options

### Topic Analysis
- Number of topics
- Keywords per topic
- Color scheme
- Title customization

### Document Flow
- Title and axis labels
- Line colors and styles
- Metric customization

## Requirements

- Python 3.8+
- Streamlit
- spaCy
- pandas
- numpy
- plotly
- textblob
- gensim
- And other dependencies listed in requirements.txt

## License

MIT License
