import requests
import streamlit as st
import json
from typing import Dict, Any, Optional
import spacy
from spacy import displacy

# Load spaCy model for entity extraction
nlp = spacy.load('en_core_web_sm')

class PromptTemplate:
    """Manages prompt templates for different extraction tasks"""
    
    TEMPLATES = {
        "entity_extraction": """
        Extract key information from the following text and return it in JSON format.
        Focus on identifying:
        - Main topics
        - Key entities (people, organizations, locations)
        - Important dates or numbers
        - Core concepts or ideas
        
        Text: {text}
        
        Return the information in the following JSON schema:
        {
            "main_topics": ["topic1", "topic2", ...],
            "entities": {
                "people": ["person1", "person2", ...],
                "organizations": ["org1", "org2", ...],
                "locations": ["location1", "location2", ...]
            },
            "dates_and_numbers": ["date/number1", "date/number2", ...],
            "core_concepts": ["concept1", "concept2", ...]
        }
        """,
        
        "summary_extraction": """
        Analyze the following text and provide a structured summary.
        Focus on:
        - Key points
        - Main arguments
        - Supporting evidence
        - Conclusions
        
        Text: {text}
        
        Return the information in the following JSON schema:
        {
            "key_points": ["point1", "point2", ...],
            "main_arguments": ["argument1", "argument2", ...],
            "supporting_evidence": ["evidence1", "evidence2", ...],
            "conclusions": ["conclusion1", "conclusion2", ...]
        }
        """
    }
    
    @staticmethod
    def get_prompt(template_name: str, **kwargs) -> str:
        """Get a formatted prompt template"""
        if template_name not in PromptTemplate.TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")
        return PromptTemplate.TEMPLATES[template_name].format(**kwargs)

class OutputParser:
    """Parses and validates LLM outputs"""
    
    @staticmethod
    def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response and handle potential errors"""
        try:
            # Find JSON content between curly braces
            json_content = response[response.find('{'):response.rfind('}')+1]
            return json.loads(json_content)
        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"Failed to parse response: {str(e)}")
            return None

class DocumentAnalyzer:
    """Analyzes documents to extract relevant information for LLM input"""
    
    @staticmethod
    def extract_entities(text: str) -> Dict[str, list]:
        """Extract named entities using spaCy"""
        doc = nlp(text)
        entities = {
            "people": [],
            "organizations": [],
            "locations": []
        }
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["people"].append(ent.text)
            elif ent.label_ in ["ORG", "NORP"]:
                entities["organizations"].append(ent.text)
            elif ent.label_ in ["GPE", "LOC"]:
                entities["locations"].append(ent.text)
                
        return entities

def get_text_completion(query: str, template_name: str = None, document_context: Dict = None) -> Dict[str, Any]:
    """Get structured completion from OpenRouter API using Claude-2"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    # If template is provided, format the prompt
    if template_name:
        prompt = PromptTemplate.get_prompt(template_name, text=query)
    else:
        prompt = query

    # Add document context if provided
    context_str = ""
    if document_context:
        context_str = "\nDocument Context:\n"
        for key, value in document_context.items():
            if isinstance(value, dict):
                context_str += f"\n{key}:\n"
                for subkey, subvalue in value.items():
                    context_str += f"- {subkey}: {subvalue}\n"
            else:
                context_str += f"- {key}: {value}\n"
    
    # Format the prompt to emphasize available content
    formatted_prompt = f"""ANALYZING DOCUMENT DATA:
{'-' * 40}
{prompt}
{context_str}
{'-' * 40}

Based on the above document data and context, please provide your analysis. Remember to reference specific details from the content, metrics, and context provided."""
    
    # Truncate if needed
    max_chars = 4000
    if len(formatted_prompt) > max_chars:
        formatted_prompt = formatted_prompt[:max_chars] + "... [truncated]"
        
    data = {
        "model": "anthropic/claude-2",
        "messages": [
            {
                "role": "system",
                "content": """You are an AI assistant integrated into a document analysis application. You have direct access to the documents through the provided context. Your role is to analyze this data and provide insights.

When responding:
1. Always reference specific content from the documents
2. For each document, you can find its content in:
   - Original text: documents[doc_name]['content']['original']
   - Cleaned text: documents[doc_name]['content']['cleaned']
3. Use the document metrics and analysis data to support your insights
4. When comparing documents, highlight key differences and similarities with specific examples
5. Quote relevant passages from the documents to support your analysis

Remember: The full document content is available in the context under the 'content' key for each document. Always use this content to provide specific, detailed analysis."""
            },
            {
                "role": "user",
                "content": formatted_prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 1,
        "stream": False
    }
    
    headers = {
        'Authorization': f'Bearer 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',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://github.com/gabrieldavila/text-analyzer',
        'X-Title': 'Text Analyzer'
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        
        # Extract the completion from OpenRouter response
        completion = response_data['choices'][0]['message']['content']
        
        # Parse structured output if template was used
        if template_name:
            parsed_response = OutputParser.parse_json_response(completion)
            if parsed_response:
                return parsed_response
            return {"error": "Failed to parse structured response"}
        
        return {"response": completion}
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}

def render_structured_output(data: Dict[str, Any], container=st):
    """Render structured output in a user-friendly format"""
    if "error" in data:
        container.error(data["error"])
        return
        
    # Create expandable sections for different types of information
    if "main_topics" in data:
        with container.expander("Main Topics", expanded=True):
            for topic in data["main_topics"]:
                container.write(f"• {topic}")
                
    if "entities" in data:
        with container.expander("Named Entities", expanded=True):
            cols = container.columns(3)
            with cols[0]:
                st.subheader("People")
                for person in data["entities"]["people"]:
                    st.write(f"• {person}")
            with cols[1]:
                st.subheader("Organizations")
                for org in data["entities"]["organizations"]:
                    st.write(f"• {org}")
            with cols[2]:
                st.subheader("Locations")
                for loc in data["entities"]["locations"]:
                    st.write(f"• {loc}")
                    
    if "dates_and_numbers" in data:
        with container.expander("Dates and Numbers"):
            for item in data["dates_and_numbers"]:
                container.write(f"• {item}")
                
    if "core_concepts" in data:
        with container.expander("Core Concepts", expanded=True):
            for concept in data["core_concepts"]:
                container.write(f"• {concept}")
                
    # Summary-specific sections
    if "key_points" in data:
        with container.expander("Key Points", expanded=True):
            for point in data["key_points"]:
                container.write(f"• {point}")
                
    if "main_arguments" in data:
        with container.expander("Main Arguments"):
            for arg in data["main_arguments"]:
                container.write(f"• {arg}")
                
    if "supporting_evidence" in data:
        with container.expander("Supporting Evidence"):
            for evidence in data["supporting_evidence"]:
                container.write(f"• {evidence}")
                
    if "conclusions" in data:
        with container.expander("Conclusions", expanded=True):
            for conclusion in data["conclusions"]:
                container.write(f"• {conclusion}")

# Streamlit UI
st.title("Smart Document Analyzer")
st.write("Upload text to extract structured information using AI")

# Analysis type selector
analysis_type = st.radio(
    "Select Analysis Type",
    ["Entity Extraction", "Summary Analysis"],
    help="Choose the type of analysis to perform on your text"
)

# Text input
text = st.text_area("Enter your text", height=200)

if text:
    with st.spinner("Analyzing..."):
        # Pre-process with spaCy for initial entity extraction
        doc_entities = DocumentAnalyzer.extract_entities(text)
        
        # Get template based on analysis type
        template_name = "entity_extraction" if analysis_type == "Entity Extraction" else "summary_extraction"
        
        # Get structured response from LLM
        response = get_text_completion(text, template_name)
        
        # Render the structured output
        render_structured_output(response)
        
        # Show raw entities from spaCy
        with st.expander("Raw Entity Analysis"):
            st.json(doc_entities)
