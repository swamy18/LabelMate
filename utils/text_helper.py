#Text labeling via models from Gemini/OpenAI via APIs.

import os
import json
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import streamlit as st

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# API Keys
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LLM provider
provider = None
llm = None

if GEMINI_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        llm = genai.GenerativeModel("gemini-1.5-flash")
        provider = "gemini"
    except ImportError:
        st.warning("Google Generative AI not installed. Run: pip install google-generativeai")

if not provider and OPENAI_KEY:
    try:
        import openai
        openai.api_key = OPENAI_KEY
        provider = "openai"
    except ImportError:
        st.warning("OpenAI not installed. Run: pip install openai")

if not provider:
    st.error("⚠️ No LLM provider configured. Add GOOGLE_API_KEY or OPENAI_API_KEY to .env file")

# System prompts for different labeling tasks
SENTIMENT_PROMPT = """
You are a sentiment analysis expert. Analyze the given text and classify it into exactly ONE category.
Return ONLY one word from: Positive, Negative, or Neutral.
Do not include explanations or additional text.
"""

TOPIC_PROMPT = """
You are a topic classification expert. Analyze the given text and assign it to the most relevant category.
Return ONLY one word from: Technology, Business, Politics, Sports, Entertainment, Health, Education, or Other.
Do not include explanations or additional text.
"""

def suggest_text_label(text: str, task: str = "sentiment") -> str:
    """
    Get AI-suggested label for text.
    
    Args:
        text: Input text to label
        task: Either 'sentiment' or 'topic'
    
    Returns:
        Suggested label as string
    """
    if not provider:
        return "Error: No API key"
    
    prompt = SENTIMENT_PROMPT if task == "sentiment" else TOPIC_PROMPT
    
    try:
        if provider == "gemini":
            response = llm.generate_content([prompt, f"Text: {text}"])
            return response.text.strip()
        
        elif provider == "openai":
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}
                ],
                max_tokens=10,
                temperature=0
            )
            return response["choices"][0]["message"]["content"].strip()
    
    except Exception as e:
        print(f"LLM Error: {e}")
        return "Error"

def load_text_csv(uploaded_file) -> pd.DataFrame:
    """
    Load and validate CSV file for text labeling.
    Adds required columns if missing.
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validate required column
        if "text" not in df.columns:
            raise ValueError("❌ CSV must contain a 'text' column")
        
        # Add labeling columns if missing
        if "label" not in df.columns:
            df["label"] = ""
        if "ai_suggested" not in df.columns:
            df["ai_suggested"] = ""
        if "human_changed" not in df.columns:
            df["human_changed"] = False
        if "confidence" not in df.columns:
            df["confidence"] = 0.0
        
        return df
    
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

def save_text_csv(df: pd.DataFrame, filename: str = "labeled_text.csv") -> str:
    """Save labeled DataFrame to exports folder."""
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)
    
    out_path = exports_dir / filename
    df.to_csv(out_path, index=False)
    return str(out_path)

def get_labeling_stats(df: pd.DataFrame) -> Dict:
    """Calculate labeling progress and statistics."""
    total = len(df)
    labeled = (df["label"] != "").sum()
    ai_generated = (df["ai_suggested"] != "").sum()
    human_changed = df["human_changed"].sum()
    
    return {
        "total": total,
        "labeled": labeled,
        "progress": labeled / max(total, 1) * 100,
        "ai_generated": ai_generated,
        "human_changed": human_changed,
        "accuracy": (ai_generated - human_changed) / max(ai_generated, 1) * 100
    }
