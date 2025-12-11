import os
import re
import google.generativeai as genai
from typing import List

# Import "dumb" DB methods directly from database module
from src.storage.database import (
    get_all_category_names, 
    get_stored_category, 
    insert_master_category, 
    upsert_item_mapping
)

# Setup Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
LIGHT_MODEL_NAME = "gemini-2.5-flash"

# Patterns to remove quantities / weights / case counts / standalone numbers
_QUANTITY_RE = re.compile(r'''
    (?ix)
    (?:\b\d+(?:[.,]\d+)?\s*(?:kg|g|gram|grams|lb|lbs|pounds|oz|ounce|ounces|cs|case|ct|pack|pcs|pc|l|ml)\b)
    |(?:\b\d+\s*(?:x|×)\s*\d+(?:[.,]\d+)?\s*(?:kg|lb|oz|ml|l)?\b)
    |(?:\b\d+(?:[.,]\d+)?\b)
''', re.VERBOSE)

# Keep + & ' in product names
_PUNCT_RE = re.compile(r"[^\w\s\+\&']")

def clean_description(raw_description: str) -> str:
    """
    Clean noisy SKU/description lines and return ONLY the product name.
    """
    if not raw_description:
        return ""

    # Use first line only (common vendor pattern)
    s = raw_description.splitlines()[0].strip().lower()

    # Normalize separators
    s = re.sub(r'[_\-\–\/\*]', ' ', s)

    # Remove quantities / sizes / numbers
    s = _QUANTITY_RE.sub(' ', s)

    # Remove unwanted punctuation
    s = _PUNCT_RE.sub(' ', s)

    # Collapse whitespace
    tokens = [t for t in re.split(r'\s+', s) if t]

    # Leave tokens as-is (don't remove 'bag', 'of', etc., because you wanted “bag of ecliptic”)
    return ' '.join(tokens).strip()


def build_categorization_prompt(description: str, existing_categories: List[str]) -> str:
    """
    Constructs a strict prompt for the LLM to categorize the item.
    """
    categories_str = ", ".join(existing_categories)
    return (
        "You are a precise categorization assistant for restaurant invoices.\n"
        f"Existing Categories: [{categories_str}]\n"
        f"Item Description: '{description}'\n\n"
        f"Task: Assign the item to one of the Existing Categories. "
        f"If it absolutely does not fit any, create a new, short, generic category name (e.g., 'Dairy', 'Produce', 'Kitchen Supplies').\n"
        f"Rules: Return ONLY the category name. No explanations. No extra text."
    )

def predict_category_with_llm(description: str, existing_categories: List[str]) -> str:
    """
    Interacts with the Gemini API to predict a category.
    Does NOT interact with the database.
    """
    # 1. Build Prompt
    prompt = build_categorization_prompt(description, existing_categories)

    try:
        # 2. Call Gemini API
        model = genai.GenerativeModel(LIGHT_MODEL_NAME)
        response = model.generate_content(prompt)
        
        # 3. Clean and Return Response
        predicted_category = response.text.strip().replace("```", "").replace("**", "")
        return predicted_category

    except Exception as e:
        print(f"[ERROR] LLM Prediction failed: {e}")
        return "Uncategorized"

def save_category_result(description: str, predicted_category: str, existing_categories: List[str]) -> None:
    """
    Only if category is unseen save to database, but delegates 
    the actual saving to storage/database.py methods.
    """
    if not predicted_category or predicted_category == "Uncategorized":
        return

    # LOGIC: Check if this is actually a new category
    existing_lower = {c.lower() for c in existing_categories}
    
    if predicted_category.lower() not in existing_lower:
        print(f"[INFO] New master category detected: {predicted_category}")
        # ACTION: Call DB method
        insert_master_category(predicted_category)

    # ACTION: Call DB method
    upsert_item_mapping(description, predicted_category)

def get_line_item_category(description: str) -> str:
    if not description:
        return "Uncategorized"

    cleaned_description = clean_description(description)
    # 1. Fetch from DB
    stored_category = get_stored_category(cleaned_description)
    if stored_category:
        return stored_category

    # 2. Fetch Context from DB
    print(f"[INFO] Categorizing via LLM: '{cleaned_description}'")
    existing_categories = get_all_category_names()

    # 3. Predict via LLM
    predicted_category = predict_category_with_llm(cleaned_description, existing_categories)

    # 4. Save via DB Delegate
    save_category_result(cleaned_description, predicted_category, existing_categories)

    return predicted_category
