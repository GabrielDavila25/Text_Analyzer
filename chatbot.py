import os
import json
import anthropic
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()  # reads .env from the project root

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are an expert document analysis assistant embedded in a research tool. "
    "You have direct access to document content and analysis metrics provided in the user message. "
    "Provide specific, evidence-based insights. Quote directly from documents when relevant. "
    "Be concise and structured. Use markdown formatting for clarity."
)


def _get_client() -> anthropic.Anthropic:
    # Check .env / environment first, then Streamlit Cloud secrets
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            pass
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "Locally: add it to your .env file. "
            "On Streamlit Cloud: add it under App Settings → Secrets."
        )
    return anthropic.Anthropic(api_key=api_key)


def _build_context_string(document_context: Dict) -> str:
    lines = ["\n\n--- Document Context ---"]
    for key, value in document_context.items():
        if isinstance(value, dict):
            lines.append(f"\n{key}:")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):
                    lines.append(f"  {subkey}:")
                    for k, v in subvalue.items():
                        lines.append(f"    - {k}: {str(v)[:800]}")
                else:
                    lines.append(f"  - {subkey}: {str(subvalue)[:800]}")
        else:
            lines.append(f"- {key}: {str(value)[:800]}")
    return "\n".join(lines)


def get_text_completion(
    query: str,
    template_name: str = None,
    document_context: Dict = None,
) -> Dict[str, Any]:
    """Get a completion from Claude via the Anthropic API."""
    try:
        client = _get_client()
    except ValueError as e:
        return {"error": str(e)}

    prompt = query
    if document_context:
        prompt = query + _build_context_string(document_context)

    max_chars = 14000
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars] + "\n... [truncated for length]"

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": prompt}],
        )
        return {"response": response.content[0].text}
    except anthropic.AuthenticationError:
        return {"error": "Invalid API key. Check your ANTHROPIC_API_KEY."}
    except anthropic.RateLimitError:
        return {"error": "Rate limit hit. Wait a moment and try again."}
    except anthropic.APIError as e:
        return {"error": f"API error: {str(e)}"}


def render_structured_output(data: Dict[str, Any], container=None) -> None:
    """Render structured LLM output in a user-friendly format."""
    import streamlit as st

    if container is None:
        container = st

    if "error" in data:
        container.error(data["error"])
        return

    if "main_topics" in data:
        with container.expander("Main Topics", expanded=True):
            for topic in data["main_topics"]:
                container.write(f"• {topic}")

    if "entities" in data:
        with container.expander("Named Entities", expanded=True):
            cols = container.columns(3)
            with cols[0]:
                st.subheader("People")
                for person in data["entities"].get("people", []):
                    st.write(f"• {person}")
            with cols[1]:
                st.subheader("Organizations")
                for org in data["entities"].get("organizations", []):
                    st.write(f"• {org}")
            with cols[2]:
                st.subheader("Locations")
                for loc in data["entities"].get("locations", []):
                    st.write(f"• {loc}")

    if "core_concepts" in data:
        with container.expander("Core Concepts", expanded=True):
            for concept in data["core_concepts"]:
                container.write(f"• {concept}")

    if "key_points" in data:
        with container.expander("Key Points", expanded=True):
            for point in data["key_points"]:
                container.write(f"• {point}")

    if "main_arguments" in data:
        with container.expander("Main Arguments"):
            for arg in data["main_arguments"]:
                container.write(f"• {arg}")

    if "conclusions" in data:
        with container.expander("Conclusions", expanded=True):
            for conclusion in data["conclusions"]:
                container.write(f"• {conclusion}")
