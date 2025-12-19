import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any
import mimetypes
import re
import pandas as pd

import google.generativeai as genai

from src.processing.categorization import get_menu_item_category

# NOTE: This module provides three responsibilities:
# 1) Build the strict prompt that we will later send to a Visual LLM
# 2) Provide a Visual LLM caller abstraction (placeholder)
# 3) Validate the returned JSON and return a cleaned list / DataFrame


def build_menu_prompt() -> str:
    """
    Return the strict prompt text for menu extraction (no API call here).

    This prompt is designed for visual menu understanding with deterministic
    group-title inheritance for item naming.
    """
    return (
        "You are a data extraction system processing a restaurant menu.\n"
        "The input is a single image or PDF of a restaurant menu.\n\n"

        "Your task is to extract ONLY sellable menu items and their prices.\n"
        "Each size, variant, or portion with a distinct price MUST be treated as a separate item.\n\n"

        "OUTPUT FORMAT (MANDATORY):\n"
        "You MUST return valid JSON only.\n"
        "No markdown, no comments, no explanations, no extra text.\n"
        "The JSON MUST match this structure exactly:\n"
        "{\n"
        "  \"menu_items\": [\n"
        "    {\n"
        "      \"item_name\": \"\",\n"
        "      \"price\": \"\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"

        "FIELD DEFINITIONS:\n"
        "- item_name: A complete, deterministic menu item name.\n"
        "- price: The numeric price for that specific item.\n\n"

        "GROUP TITLE HANDLING (CRITICAL AND STRICT):\n"
        "- Menu group titles or section headers are NOT menu items by themselves.\n"
        "- Group titles provide CONTEXT and may be used ONLY as a prefix.\n"
        "- When used, the item_name MUST follow this exact format:\n"
        "  \"<Group Title> - <Item Label>\"\n"
        "- Do NOT reorder, rephrase, pluralize, or grammatically adjust either part.\n\n"

        "YOU MUST USE THE GROUP TITLE IF:\n"
        "- The item label alone does NOT uniquely describe what is being sold.\n"
        "- The item could represent multiple business meanings without context.\n"
        "  Examples: Plain, Tomato, Cheese, Extra, Add On, 12 Pack, Small, Large.\n\n"

        "YOU MUST NOT USE THE GROUP TITLE IF:\n"
        "- The item label alone clearly represents a sellable product.\n\n"

        "CANONICAL NAMING EXAMPLES (FOLLOW EXACTLY):\n"
        "- Group: \"Bagels\", Item: \"Plain\", Price: 2.75 → \"Bagels - Plain\"\n"
        "- Group: \"Utensils\", Item: \"12 Pack\", Price: 12.00 → \"Utensils - 12 Pack\"\n"
        "- Group: \"Extras\", Item: \"Plain\", Price: 0.50 → \"Extras - Plain\"\n"
        "- Group: \"Add Ons\", Item: \"Tomato\", Price: 6.00 → \"Add Ons - Tomato\"\n"
        "- Group: \"Toppings\", Item: \"Cheese\", Price: 1.25 → \"Toppings - Cheese\"\n"
        "- Group: \"Beverages\", Item: \"Coke\", Price: 1.99 → \"Beverages - Coke\"\n\n"

        "STRICTLY FORBIDDEN:\n"
        "- Returning \"Tomato\" alone when it is under Add Ons, Extras, or Toppings.\n"
        "- Converting \"Bagels - Plain\" into \"Plain Bagel\" or similar rewrites.\n"
        "- Using the group title as a standalone item.\n\n"

        "STRICT EXTRACTION RULES:\n"
        "- If an item does NOT have a clearly visible price, DO NOT include it.\n"
        "- Do NOT guess, infer, estimate, or calculate prices.\n"
        "- Do NOT merge different sizes or variants into one item.\n"
        "- If the same item_name appears with different prices, keep them as separate entries.\n"
        "- If the same item_name AND the same price appear multiple times, return only one entry.\n\n"

        "IGNORE COMPLETELY:\n"
        "- Descriptions or ingredients\n"
        "- Combo explanations\n"
        "- Notes, disclaimers, footnotes\n"
        "- Taxes, service charges, delivery fees\n"
        "- Discounts, coupons, or promotional text\n\n"

        "IMPORTANT CONSTRAINTS:\n"
        "- Do NOT normalize spelling.\n"
        "- Do NOT creatively rewrite names.\n"
        "- Do NOT output nulls, empty objects, or placeholders.\n"
        "- If no valid items with prices are found, return:\n"
        "{ \"menu_items\": [] }\n\n"

        "FINAL CHECK BEFORE RESPONDING:\n"
        "- Response must be valid JSON.\n"
        "- Top-level key must be exactly \"menu_items\".\n"
        "- Every entry must contain both item_name and price.\n"
        "- No extra keys are allowed.\n"
    )


def _parse_json_str(json_text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(json_text)
        return parsed
    except json.JSONDecodeError as e:
        # Attempt to salvage common issues (strip code fences etc.) and re-raise if still invalid
        cleaned = json_text.strip().replace('```', '')
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON returned from Visual LLM. Error: {e}\nRaw output:\n{json_text}")


def validate_and_dedup(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = parsed.get("menu_items") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        raise ValueError("Invalid menu JSON: missing 'menu_items' list")

    validated = []
    seen = set()

    for it in items:
        if not isinstance(it, dict):
            raise ValueError(f"Invalid item in menu_items: expected object, got {type(it)}")

        name = it.get("item_name")
        price = it.get("price")

        if not name:
            raise ValueError(f"Menu item has empty 'item_name': {it}")

        # Strict validation: price must be present and numeric. Do NOT silently continue.
        try:
            price_val = float(price)
        except Exception as e:
            raise ValueError(f"Invalid or missing price for item '{name}': {price}. Error: {e}")

        key = (name.strip(), price_val)
        if key in seen:
            continue
        seen.add(key)

        validated.append({"menu_item": name.strip(), "price": price_val})

    return validated

MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")


def call_visual_llm(file_path: Path, max_retries: int = 2) -> str:
    """
    Call Gemini Vision (multimodal) to extract menu JSON from file_path.
    """


    # Real call: check API key
    gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise EnvironmentError("Missing GEMINI_API_KEY. Please set GEMINI_API_KEY environment variable.")

    genai.configure(api_key=gemini_api_key)

    # Upload file
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type is None:
        # Default to image/jpeg for unknowns
        mime_type = "image/jpeg"

    file_obj = genai.upload_file(str(file_path), mime_type=mime_type)

    # Build strict prompt for menu extraction (JSON only)
    prompt = build_menu_prompt() + "\nUse the provided file to extract the menu items exactly as JSON."

    # Call the model (multimodal prompt: [text_prompt, file_obj])
    for attempt in range(max_retries + 1):
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content([prompt, file_obj])
        raw_text = response.text.strip() if response and response.text else ""

        # Clean code fences if present
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()

        # Quick validation: must start with '{' or '['
        if cleaned and (cleaned.lstrip().startswith("{") or cleaned.lstrip().startswith("[")):
            return cleaned

        # else, retry or raise
        if attempt < max_retries:
            time.sleep(1 + attempt)
            continue
        else:
            raise ValueError(f"Visual LLM did not return valid JSON for file {file_path}. Raw output:\n{raw_text}")


def extract_menu(restaurant_id: str, file_path: Path) -> pd.DataFrame:
    """Extract menu from `file_path` for `restaurant_id`.

    Returns a DataFrame with columns: restaurant_id, menu_item, price, category
    and STRICT validation (raises on malformed items).
    """
    raw = call_visual_llm(file_path)

    # Parse & validate JSON
    parsed = _parse_json_str(raw)
    validated = validate_and_dedup(parsed)

    rows = []
    for rec in validated:
        menu_item = rec["menu_item"]
        price = rec["price"]

        # Determine category via menu-specific categorization helper (may call LLM if unseen)
        category = get_menu_item_category(menu_item)

        rows.append({
            "restaurant_id": str(restaurant_id),
            "menu_item": menu_item,
            "price": float(price),
            "category": category,
        })

    df = pd.DataFrame(rows, columns=["restaurant_id", "menu_item", "price", "category"])
    return df
