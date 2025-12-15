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
LIGHT_MODEL_NAME = "gemini-2.5-flash-lite"

# Units, packaging, and size indicators
_UNITS = r"(?:lb|lbs|oz|fl\s*oz|ml|l|g|kg|ea|ct|pcs?|pack|case|cs|bn|bag)"
_NUM = r"\d+(?:[.,]\d+)?"

# Matches size / quantity / packaging blocks
_SIZE_BLOCK_RE = re.compile(
    rf"""
    (?ix)
    \b(
        {_NUM}\s*(?:x|×)\s*{_NUM}\s*(?:{_UNITS})?   # 4 x 5 lb
        |
        {_NUM}\s*(?:-\s*{_NUM})?\s*(?:{_UNITS})    # 50 lb, 12-16 oz
        |
        \#\d+                                      # #10
        |
        \b{_UNITS}\b
    )\b
    """,
    re.VERBOSE,
)

# Words that indicate non-product financial/system lines
_SPECIAL_LINES = {
    "tax",
    "sales tax",
    "fuel surcharge",
}

# Remove parenthetical codes like ( SC4 )
_PAREN_RE = re.compile(r"\([^)]*\)")

# Keep letters, spaces, +, &
_CLEAN_RE = re.compile(r"[^\w\s\+\&]")

def clean_description(raw_description: str) -> str:
    """
    Extracts the core product name from noisy invoice descriptions.
    """
    if not raw_description:
        return ""

    # Normalize and take first non-empty line
    lines = [l.strip() for l in raw_description.splitlines() if l.strip()]
    s = lines[0].lower()

    # Remove parenthetical codes early
    s = _PAREN_RE.sub(" ", s)

    # Normalize separators
    s = s.replace(",", " ")
    s = re.sub(r"[_\-/]", " ", s)

    # Fast path for known system lines
    s_clean = s.strip()
    if s_clean in _SPECIAL_LINES:
        return s_clean

    # Truncate at first size / quantity block
    match = _SIZE_BLOCK_RE.search(s)
    if match:
        s = s[:match.start()]

    # Remove residual numbers
    s = re.sub(r"\b\d+\b", " ", s)

    # Remove unwanted punctuation
    s = _CLEAN_RE.sub(" ", s)

    # Token cleanup
    tokens = [t for t in s.split() if t]

    # Produce logic: keep only first semantic noun
    # Example: "onion italian red" -> "onion"
    if tokens and tokens[0] in {
        "onion", "tomato", "cucumber", "herbs"
    }:
        return tokens[0]

    return " ".join(tokens)



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
        print(f"[INFO] Found decryption-category pair in DB: Description: {description}, Category: {stored_category}")
        return stored_category

    # 2. Fetch Context from DB
    # print(f"[INFO] Categorizing via LLM: '{cleaned_description}'")
    existing_categories = get_all_category_names()

    # 3. Predict via LLM
    predicted_category = predict_category_with_llm(cleaned_description, existing_categories)

    # 4. Save via DB Delegate
    save_category_result(cleaned_description, predicted_category, existing_categories)
    
    print(f"[INFO] Used LLM to get decryption-category pair: Description: {description}, Category: {predicted_category}")

    return predicted_category
