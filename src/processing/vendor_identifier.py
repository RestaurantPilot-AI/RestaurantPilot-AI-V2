import os
import re
import time
import json
import mimetypes
from typing import Optional, Dict, Any, Tuple, List, Union
from dotenv import load_dotenv

# For real LLM integration
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from src.storage import (
    get_vendor_by_email,
    get_vendor_by_website,
    get_vendor_by_address,
    get_vendor_by_phone,
    get_vendor_by_name,
    get_vendor_regex_patterns,
    create_vendor,
    save_vendor_regex_template,
    get_vendor_name_by_id,
)

# ----------------------------
# Config / Feature flags
# ----------------------------
load_dotenv()


MODEL_NAME = "gemini-2.5-flash"

# ----------------------------
# Setup
# ----------------------------
def _setup_environment() -> None:
    """Sets Gemini API key in environment and initializes Gemini client."""

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GEMINI_API_KEY. Please set it as an environment variable.")

    # Configure Gemini client
    genai.configure(api_key=api_key)

# ----------------------------
# Helper: normalization / small utils
# ----------------------------
def _normalize_name(name: str) -> str:
    """Lowercase, strip punctuation & extra whitespace for robust matching."""
    if not name:
        return ""
    return re.sub(r'[^a-z0-9]+', '', name.lower().strip())

# ----------------------------
# Basic DB operations (stubs)
# ----------------------------
def save_regex_for_vendor(new_vendor_id: str, new_regexes: Dict[str, Any]) -> None:
    """Save regex extraction template for new vendor to DB."""
    save_vendor_regex_template(new_vendor_id,new_regexes)


def get_regex_for_vendor(vendor_id: str) -> Optional[List[str]]:
    """
    Fetches regex patterns for a specific vendor ID from the DB.
    """
    if not vendor_id:
        return None

    # Call the new DB method which returns List[str]
    return get_vendor_regex_patterns(vendor_id)


def save_vendor_details(vendor_master_data: Dict[str, Any]) -> None:
    """
    Persist vendor_master_data (dict). Expected keys:
      - vendor_name (str)
      - vendor_email_id (optional)
      - vendor_phone_number (optional)
      - vendor_physical_address (optional)
      - vendor_website (optional)
    Implement idempotency / upsert behavior as appropriate.
    """
    # validate minimally
    if not vendor_master_data.get("vendor_name"):
        raise ValueError("vendor_name required to save vendor.")
    
    return create_vendor(vendor_master_data)


# ----------------------------
# Vendor signal extraction
# ----------------------------
def extract_vendor_signals(text: str) -> Dict[str, Optional[str]]:
    """
    Extract signals to identify vendor:
    - vendor_email_id (best-effort)
    - vendor_phone_number
    - website
    - vendor_name 
    - vendor_physical_address (very heuristic)
    """
    # 1. Initialize result dictionary strictly matching docstring
    signals: Dict[str, Optional[str]] = {
        "vendor_email_id": None,
        "vendor_phone_number": None,
        "website": None,
        "vendor_name": None,
        "vendor_physical_address": None
    }

    if not text:
        return signals

    # Pre-process: Split lines and keep a clean version
    lines = [ln.strip() for ln in re.split(r'\r\n|\r|\n', text) if ln.strip()]

    # --- 1. Vendor Email (High Confidence) ---
    # Regex looks for standard email format
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    # Find all matches
    emails = re.findall(email_pattern, text)
    if emails:
        # Heuristic: Filter out emails that look like "support@" or "info@" if specific names exist, 
        # but for a vendor signal, "billing@" or "info@" is actually good. 
        # We take the first one found in the header usually, or just the first valid one.
        signals["vendor_email_id"] = emails[0]

    # --- 2. Vendor Phone Number (Label Priority + Format) ---
    # Strategy: Look for lines with "Phone/Tel" explicitly first. 
    # If not found, look for international format patterns.
    
    phone_label_pattern = r'(?:Phone|Tel|Mobile|Cell|Ph|T)[:\.\-\s]+([+\d\(\)\-\s]{7,20})'
    phone_strict_pattern = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    
    # Attempt 1: Look for labelled phone numbers (Best Match)
    for line in lines[:30]: # Usually in header
        m_label = re.search(phone_label_pattern, line, re.IGNORECASE)
        if m_label:
            # Clean up the match (remove unrelated text caught by greedy regex)
            raw_num = m_label.group(1).strip()
            if sum(c.isdigit() for c in raw_num) >= 7: # Valid check
                signals["vendor_phone_number"] = raw_num
                break
    
    # Attempt 2: Strict Regex scan if no label found
    if not signals["vendor_phone_number"]:
        # Find all strict matches in the first half of text
        matches = re.findall(phone_strict_pattern, text[:2000])
        if matches:
            # specific filter to avoid dates (e.g. 2023-12-05)
            # Valid phones usually don't start with 202x unless international
            valid_phones = [p for p in matches if not re.match(r'20[2-3]\d', p)]
            if valid_phones:
                signals["vendor_phone_number"] = valid_phones[0]

    # --- 3. Website (New Logic) ---
    # Strategy: Find URL patterns, excluding the email domain we just found.
    url_pattern = r'(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'
    
    potential_urls = []
    for line in lines[:40]: # Websites usually in header or footer
        # Skip lines that look like emails
        if "@" in line: 
            continue
            
        found_urls = re.findall(url_pattern, line)
        for url in found_urls:
            # Clean validation
            clean_url = url.lower()
            if any(ext in clean_url for ext in ['.com', '.net', '.org', '.io', '.co', '.us', '.eu', '.de']):
                potential_urls.append(url)

    if potential_urls:
        signals["website"] = potential_urls[0]

    # --- 4. Vendor Name (Hierarchical Logic) ---
    # Strategy: Explicit Header -> Legal Suffix -> UpperCase Header
    
    # A. Explicit Label (e.g., "Vendor: Acme Corp")
    if not signals["vendor_name"]:
        header_text = " ".join(lines[:15])
        m_vendor = re.search(r'(?:Vendor|Supplier|Billed by|Sold by|Payable to)[:\s\-]+([A-Za-z0-9&\.\-\,\s]{3,50})', header_text, flags=re.IGNORECASE)
        if m_vendor:
            clean_name = m_vendor.group(1).split('  ')[0].strip() # Stop at double space
            signals["vendor_name"] = clean_name

    # B. Legal Entity Suffix Search (The most robust method for native text)
    if not signals["vendor_name"]:
        # Exhaustive list of global company suffixes
        legal_suffixes = r'\b(Inc|LLC|Ltd|GmbH|BV|B\.V\.|Co\.|Company|Corp|Corporation|S\.A\.|S\.L\.|AG|Pty|Pvt|Private|Plc)\b'
        
        for i, line in enumerate(lines[:20]): # Vendor is usually at the top
            # Skip lines that look like "Bill To" or "Ship To"
            if re.search(r'(Bill|Ship|Sold)\s+To:', line, re.IGNORECASE):
                continue
                
            if re.search(legal_suffixes, line, re.IGNORECASE):
                # Split by noise chars to isolate name
                cand = re.split(r',|\||\-|\—|:|Tel|Ph', line)[0].strip()
                # Heuristic: Name shouldn't be too long or numeric
                if 2 < len(cand) < 60 and not cand.replace(" ", "").isdigit():
                    signals["vendor_name"] = cand
                    break

    # C. Fallback: First Significant Uppercase Line (Risky, used as last resort)
    if not signals["vendor_name"]:
        for line in lines[:10]:
            # Clean non-alphanumeric to check for "all caps" title style
            clean_line = re.sub(r'[^A-Z0-9 ]', '', line)
            if 2 < len(line) < 50 and clean_line == line and not line.isdigit():
                # Avoid common invoice keywords behaving like names
                if "INVOICE" not in line and "ORDER" not in line:
                    signals["vendor_name"] = line
                    break

    # --- 5. Vendor Physical Address (Keywords + ZipCode Context) ---
    # Strategy: Look for Street/Road keywords OR Zip Code patterns strictly
    
    address_keywords = r'\b(Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Blvd|Boulevard|Way|Plaza|Square|Sq|P\.O\.\s*Box|Suite|Floor|Unit)\b'
    # Matches US (5+4), UK (Alpha), EU (4-5 digits) zip codes
    zip_regex = r'\b([A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}|[A-Z]{2}-\d{4}|\d{5}(?:-\d{4})?|\d{4}\s?[A-Z]{2})\b'
    
    addr_candidates = []
    
    # We scan the first 40 lines (header addresses) and last 10 lines (footer addresses)
    search_lines = lines[:40] + lines[-10:]
    
    for i, line in enumerate(search_lines):
        # 1. Check for street keywords
        if re.search(address_keywords, line, re.IGNORECASE):
            # Capture this line and potentially the next line (City/State/Zip)
            addr_block = line
            if i + 1 < len(search_lines):
                next_line = search_lines[i+1]
                # If next line looks like a city/zip line (has digits or limited length)
                if re.search(zip_regex, next_line) or len(next_line) < 50:
                    addr_block += ", " + next_line
            addr_candidates.append(addr_block)
            
        # 2. Check for Zip Code lines if we haven't found keywords
        elif re.search(zip_regex, line):
            # If we find a zip, looking at the previous line often gives the street
            prev_line = search_lines[i-1] if i > 0 else ""
            if len(prev_line) < 60:
                addr_candidates.append(f"{prev_line}, {line}")

    if addr_candidates:
        # Pick the longest/most complete looking one, or the first one
        signals["vendor_physical_address"] = addr_candidates[0]

    return signals


# ----------------------------
# LLM helpers (isolated)
# ----------------------------
def _safe_extract_json_from_llm(text: str) -> Optional[Dict[str, Any]]:
    """
    Given an LLM textual response, attempt to extract the first top-level JSON object and parse it.
    Returns dict on success, None on failure.
    """
    if not text or not isinstance(text, str):
        return None
    
    # Try greedy match from first { to last }
    try:
        first = text.index('{')
        last = text.rindex('}')
        candidate = text[first:last + 1]
    except ValueError:
        return None

    try:
        parsed = json.loads(candidate)
        return parsed
    except Exception:
        return None

def call_llm_api(prompt: str, file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Calls Gemini API with multimodal support (file_path) by using genai.upload_file.
    
    The uploaded file is deleted after the attempt to generate content to conserve resources.

    Args:
        prompt (str): The textual prompt/instructions for the LLM.
        file_path (Optional[str]): Path to the file (image/PDF) to include 
                                    for visual multimodal input.
                                    
    Returns parsed JSON dictionary.
    
    Raises:
        RuntimeError: If API retries fail.
        ValueError: If Gemini output is not valid JSON.
        FileNotFoundError: If the provided file_path does not exist.
    """
    _setup_environment()
    model = genai.GenerativeModel(MODEL_NAME)

    max_retries = 3
    base_wait = 30  # Start waiting 30 seconds
    
    file_obj = None # Initialize file object outside the loop
    
    try:
        # 1. Prepare Content (Prompt + File Part)
        content_parts: List[Union[str, Any]] = [prompt] # Start with the prompt

        if file_path:
            if not os.path.exists(file_path):
                 raise FileNotFoundError(f"Multimodal file not found at: {file_path}")
            
            # Use mimetypes to guess the file type for robust uploading
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type is None:
                # Fallback for unknown file types
                mime_type = "image/jpeg" 
            
            try:
                # Use the built-in SDK upload feature for files (required for PDFs/large files)
                print(f"[INFO] Uploading file: {file_path} with MIME type: {mime_type}")
                file_obj = genai.upload_file(file_path, mime_type=mime_type)
                
                # Add the uploaded file object (Part) to the beginning of the content list
                content_parts.insert(0, file_obj)
                
            except Exception as e:
                # If file upload fails, raise error, but ensure cleanup occurs
                raise RuntimeError(f"Failed to upload file to Gemini API: {e}")
            
        # 2. API Call with Retry Logic
        for attempt in range(max_retries):
            try:
                # --- Call Gemini multimodal model ---
                response = model.generate_content(content_parts)
                
                # --- Clean & parse JSON ---
                raw_text = response.text.strip() if response and response.text else ""
                
                # Clean Markdown fences (```json or ```) and parse
                cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
                
                try:
                    corrected_json = json.loads(cleaned)
                    return corrected_json
                except json.JSONDecodeError as e:
                     raise ValueError(
                         f"Gemini did not return valid JSON. Error: {e}\nRaw output:\n{raw_text}"
                     )
                
            except ResourceExhausted as e:
                # This catches the 429 Quota Exceeded error
                wait_time = base_wait * (attempt + 1)
                print(f"\n[WARN] Gemini Quota Exceeded. Waiting {wait_time}s before retry ({attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            
            except ValueError as e:
                # Re-raise JSON errors immediately 
                raise e 
    
            except Exception as e:
                # Other errors (auth, network) should crash immediately
                print(f"[ERROR] Gemini API Failed: {e}")
                raise e

        # If we run out of retries
        raise RuntimeError("Gemini API Quota Exceeded after multiple retries. Please check your billing/limits.")

    finally:
        # --- CRITICAL: Delete the uploaded file object ---
        if file_obj:
            try:
                genai.delete_file(name=file_obj.name)
                print(f"[INFO] Deleted uploaded file: {file_obj.name}")
            except Exception as e:
                print(f"[WARN] Failed to delete uploaded file {file_obj.name}: {e}")

def make_phase1_prompt() -> str:
    """
    Construct phase-1 prompt for the visual model. 
    It instructs the LLM to extract data solely from the provided image/PDF content.
    """
    # Define the mandatory fields at the top of the prompt for emphasis
    mandatory_fields = [
        "invoice_number", 
        "invoice_date", 
        "invoice_total_amount", 
        "vendor_name", 
        "description (in line_items)", 
        "line_total (in line_items)"
    ]

    return f"""
You are a strict data extraction engine. Your task is to extract structured data solely from the **VISUAL INVOICE IMAGE** provided. You must use the layout, font size, and visual position to determine the correct fields.

Rules:
- Output ONLY the JSON object. No explanations or code fences (e.g., ```json).
- Field names MUST match the required JSON structure exactly.
- If a value is genuinely missing, return 'null'.
- Only refer null if a field is missing, check thoroughly using the visual evidence.
- For all currency/numeric fields, ensure the value is a string formatted as "0.00".

==========================
HIGH PRIORITY MANDATORY FIELDS
==========================
The following fields CANNOT be null. If they are not explicitly labeled, you must use **visual contextual cues** (e.g., location, size, format, bolding) to find them:
- {', '.join(mandatory_fields)}

VENDOR NAME STRATEGY:
The 'vendor_name' is the name of the company that issued the invoice. It is the single most critical field. You MUST identify the single largest or most prominent business name/logo text located in the header/top section of the invoice and use that value.

Required JSON structure:

{{
  "invoice_details": {{
    "invoice_number": "",
    "invoice_date": "YYYY-MM-DD", 
    "invoice_total_amount": "0.00",
    "order_date": "YYYY-MM-DD" 
  }},
  "line_items": [
    {{
      "description": "",
      "quantity": "0.00",
      "unit": "", 
      "unit_price": "0.00",
      "line_total": "0.00"
    }}
  ],
  "vendor_master_data": {{
    "vendor_name": "",
    "vendor_email_id": "",
    "vendor_phone_number": "",
    "vendor_physical_address": "",
    "vendor_website": ""
  }}
}}

Context Definitions:
- "unit": The unit of measure (e.g., "lb", "case", "oz", "each").
- "order_date": The date the order was placed (distinct from invoice date), if available.
""".strip()

def make_phase2_prompt(raw_text: str, verified_json: Dict[str, Any]) -> str:
    """
    Construct a single-shot prompt that asks the model to return strict,
    reusable regex patterns for this invoice layout.

    The model will receive:
      - RAW INVOICE TEXT (the full invoice as plain text)
      - VERIFIED JSON (the ground-truth extraction for that invoice)

    The model must return only a JSON object (no explanations, no markdown).
    """
    
    # Schema to match 'vendor_regex_templates' requirements
    schema = {
        "invoice_level": {
            "invoice_number": "",
            "invoice_date": "",
            "invoice_total_amount": "",
            "order_date": "" 
        },
        "line_item_level": {
            "line_item_block_start": "",
            "line_item_block_end": "",
            "description": "",
            "quantity": "",
            "unit": "",       
            "unit_price": "",
            "line_total": ""
        }
    }

    return f"""
You are given two inputs below: 1) RAW INVOICE TEXT and 2) VERIFIED JSON that contains the correct extracted values for that invoice.

Task: Produce reusable, strict regex patterns for this invoice layout so that future invoices with the same layout can be parsed without calling an LLM.

OUTPUT RULES:
- Return ONLY one JSON object and nothing else (no prose, no markdown, no code fences).
- The JSON MUST match this exact structure (keys and nesting).
- Do NOT include delimiters or flags (no /.../, no (?i), etc.).
- Regexes must be as STRICT as reasonable: use surrounding labels, punctuation, and layout.

CRITICAL REGEX RULES (STRICT ENFORCEMENT):
1. **EXACTLY ONE CAPTURING GROUP**: Every regex that extracts a value MUST contain exactly one capturing group `(...)` that isolates the target data.
2. **USE NON-CAPTURING GROUPS**: If you need to group tokens for logic (e.g., matching "CS" or "EA"), you MUST use non-capturing groups `(?:...)`.
   - BAD: `(CS|EA)` -> This captures the unit, creating a second group.
   - GOOD: `(?:CS|EA)` -> This matches but does not capture.
3. **MANDATORY FIELDS**: 'invoice_number', 'invoice_date', and 'invoice_total_amount' MUST HAVE A REGEX. Do not return empty strings for these.

FIELD GUIDANCE:
[Invoice Level]
- invoice_number, invoice_date, invoice_total_amount: Regex should capture the exact value shown in VERIFIED JSON.
- order_date: Capture the date the order was placed. Distinct from Invoice Date.

[Line Item Level]
- line_item_block_start: Regex that matches the first line or header (e.g., "Description   Qty   Price"). (No capture group needed here).
- line_item_block_end: Regex that matches the line AFTER the last item (e.g., "Subtotal" or footer text). (No capture group needed here).
- description, quantity, unit_price, line_total: Regexes that extract fields from a SINGLE line item row.
- unit: Regex that captures the unit (e.g., "CS", "EA") if present on the line.

REQUIRED JSON STRUCTURE:
{json.dumps(schema, indent=2)}

RAW INVOICE TEXT:
------------------
{raw_text}

VERIFIED JSON (correct extraction for this invoice):
---------------------------------------------------
{json.dumps(verified_json, indent=2)}
"""



class Phase1ParseError(ValueError):
    """Raised when LLM output cannot be parsed as JSON."""


class Phase1ValidationError(ValueError):
    """Raised when parsed JSON is missing required fields or has wrong types."""


def _coerce_none_values(obj: Any) -> Any:
    """
    Recursively convert the string "None" (and literal "null" if present as string)
    to Python None, and keep other values as-is.
    """
    if isinstance(obj, str) and obj.strip() in {"None", "null", ""}:
        return None
    if isinstance(obj, dict):
        return {k: _coerce_none_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_none_values(v) for v in obj]
    return obj


def llm_phase1_extract(file_path: str) -> Dict[str, Any]:
    """
    Phase 1: ask the MULTIMODAL LLM to extract key fields as JSON, 
    using the visual content of the file.

    Args:
        file_path (str): Path to the invoice file (image or PDF).

    Returns parsed JSON dict on success.

    Raises:
        Phase1ParseError: when LLM output is not valid JSON / not extractable.
        Phase1ValidationError: when JSON is missing required keys or has wrong types.
    """
    print("\n[INFO] Starting Phase 1: Extract Vendor Details and Patterns (Multimodal)...")
    
    # 1. Generate the visual-focused prompt (takes no arguments now)
    prompt = make_phase1_prompt()
    
    # 2. Call the LLM API, passing the prompt and the file_path for multimodal processing
    # The OCR text step is now bypassed entirely.
    parsed = call_llm_api(prompt, file_path=file_path)

    # 1) Try to extract JSON (Validation)
    # Note: We skip _safe_extract_json_from_llm because 'parsed' is already a dict
    if not isinstance(parsed, dict):
        # FIX: Convert to string first to avoid 'KeyError: slice' crash
        snippet = str(parsed or "")[:1000]
        raise Phase1ParseError(f"Phase1 parse failure: LLM output not JSON-dict. Raw output snippet: {snippet!s}")

    # normalize "None" strings -> None
    parsed = _coerce_none_values(parsed)

    # --- REQUIRED TOP-LEVEL KEYS ---
    required_root_keys = {"invoice_details", "line_items", "vendor_master_data"}
    missing_root = sorted(list(required_root_keys - set(parsed.keys())))
    if missing_root:
        raise Phase1ValidationError(f"Missing top-level keys: {missing_root}")

    # --- invoice_details (all required) ---
    invoice = parsed.get("invoice_details")
    if not isinstance(invoice, dict):
        raise Phase1ValidationError("invoice_details must be an object/dict.")
    
    # FIX: Removed "vendor_name" from here. It is handled in vendor_master_data.
    required_invoice_keys = {"invoice_number", "invoice_date", "invoice_total_amount"}
    
    missing_invoice = sorted(list(required_invoice_keys - set(invoice.keys())))
    if missing_invoice:
        raise Phase1ValidationError(f"invoice_details missing keys: {missing_invoice}")

    # --- vendor_master_data ---
    vendor = parsed.get("vendor_master_data")
    if not isinstance(vendor, dict):
        raise Phase1ValidationError("vendor_master_data must be an object/dict.")
    # Only vendor_name is required; other contact fields are optional
    required_vendor_keys = {"vendor_name"}
    missing_vendor = sorted(list(required_vendor_keys - set(vendor.keys())))
    if missing_vendor:
        raise Phase1ValidationError(f"vendor_master_data missing keys: {missing_vendor}")

    optional_vendor_keys = [
        "vendor_email_id",
        "vendor_phone_number",
        "vendor_physical_address",
        "vendor_website",
    ]
    # Ensure optional vendor keys exist (set to None if absent)
    for k in optional_vendor_keys:
        if k not in vendor:
            vendor[k] = None

    # --- line_items ---
    items = parsed.get("line_items")
    if not isinstance(items, list):
        raise Phase1ValidationError("line_items must be a list.")
    # Require description and line_total; quantity and unit_price are optional
    required_item_keys = {"description", "line_total"}
    optional_item_keys = {"quantity", "unit_price"}
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise Phase1ValidationError(f"line_items[{idx}] must be an object/dict.")
        missing_item_keys = sorted(list(required_item_keys - set(item.keys())))
        if missing_item_keys:
            raise Phase1ValidationError(f"line_items[{idx}] missing required keys: {missing_item_keys}")
        # Ensure optional item keys exist (set to None if absent)
        for k in optional_item_keys:
            if k not in item:
                item[k] = None

    # Re-assign normalized/filled structures back into parsed (in case we mutated copies)
    parsed["invoice_details"] = invoice
    parsed["vendor_master_data"] = vendor
    parsed["line_items"] = items

    return parsed

class Phase2ParseError(Exception):
    pass

class Phase2ValidationError(Exception):
    pass

def _count_capture_groups(pattern: str) -> int:
    """
    Count non-non-capturing, non-escaped capturing groups in a regex string.
    This counts occurrences of '(' that are not preceded by a backslash and
    are not followed by '?:' (non-capturing group).
    """
    if not isinstance(pattern, str):
        return 0
    # find '(' that are not escaped and not followed by '?:'
    return len(re.findall(r'(?<!\\)\((?!\?:)', pattern))

def llm_phase2_generate_regex(text: str, phase1_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 2: ask LLM to generate strict regex templates (invoice_level & line_item_level).
    Returns parsed template dict on success.

    Raises:
        Phase2ParseError: when LLM output is not valid JSON / not extractable.
        Phase2ValidationError: when JSON is missing keys or regexes fail basic checks.
    """
    print("\n[INFO] Starting Phase 2: Generating Regex Patterns...")
    prompt = make_phase2_prompt(text, phase1_json)
    
    # FIX: call_llm_api now returns the parsed JSON dict directly
    parsed = call_llm_api(prompt)

    # DEBUG: Print the raw output to see what the LLM actually gave us
    # print(f"[DEBUG] Phase 2 Raw LLM Output:\n{json.dumps(parsed, indent=2)}")

    # 1) Try to extract JSON (Validation)
    if not isinstance(parsed, dict):
        snippet = str(parsed or "")[:1000]
        raise Phase2ParseError(f"Phase2 parse failure: LLM output not JSON-dict. Raw output snippet: {snippet!s}")

    # --- REQUIRED TOP-LEVEL KEYS ---
    required_root_keys = {"invoice_level", "line_item_level"}
    missing_root = sorted(list(required_root_keys - set(parsed.keys())))
    if missing_root:
        raise Phase2ValidationError(f"Missing top-level keys: {missing_root}")

    # =========================================================
    # 1. INVOICE LEVEL VALIDATION
    # =========================================================
    invoice = parsed.get("invoice_level")
    if not isinstance(invoice, dict):
        raise Phase2ValidationError("invoice_level must be an object/dict.")

    # All these keys MUST be present in the JSON keys
    required_invoice_keys = {"invoice_number", "invoice_date", "invoice_total_amount", "order_date"}
    missing_invoice = sorted(list(required_invoice_keys - set(invoice.keys())))
    if missing_invoice:
        raise Phase2ValidationError(f"invoice_level missing keys: {missing_invoice}")

    # Which fields are strictly required to have a non-empty regex?
    strict_invoice_fields = {"invoice_number", "invoice_date", "invoice_total_amount"}

    empty_invoice_keys = []
    bad_group_keys = []

    for k in required_invoice_keys:
        v = invoice.get(k)
        if not isinstance(v, str):
            empty_invoice_keys.append(f"invoice_level.{k} (not a string)")
            continue

        # If it's a strict field, it cannot be empty.
        if k in strict_invoice_fields and v.strip() == "":
            print(f"[WARN] Field '{k}' is empty in LLM output! in Phase 1")
            empty_invoice_keys.append(f"invoice_level.{k}")
        
        # If we have a regex (non-empty), validate capture groups
        if v.strip() != "":
            groups = _count_capture_groups(v)
            # FIX: RELAXED VALIDATION. Allow > 1 group.
            if groups < 1:
                bad_group_keys.append(f"invoice_level.{k} (capture groups={groups})")
            elif groups > 1:
                print(f"[WARN] invoice_level.{k} has {groups} capture groups. System will default to group(1).")

    if empty_invoice_keys:
        raise Phase2ValidationError(f"Empty/Invalid strictly required invoice_level regex values: {empty_invoice_keys}")
    if bad_group_keys:
        raise Phase2ValidationError(f"invoice_level regex capture-group errors: {bad_group_keys}")

    # =========================================================
    # 2. LINE ITEM LEVEL VALIDATION
    # =========================================================
    line_item = parsed.get("line_item_level")
    if not isinstance(line_item, dict):
        raise Phase2ValidationError("line_item_level must be an object/dict.")

    # All these keys MUST be present in the JSON keys
    required_line_item_keys = {
        "line_item_block_start",
        "line_item_block_end",
        "description",
        "quantity",
        "unit",
        "unit_price",
        "line_total",
    }
    missing_line_item = sorted(list(required_line_item_keys - set(line_item.keys())))
    if missing_line_item:
        raise Phase2ValidationError(f"line_item_level missing keys: {missing_line_item}")

    # Which fields are strictly required to have a non-empty regex?
    strict_line_fields = {
        "line_item_block_start", 
        "line_item_block_end", 
        "description", 
        "quantity", 
        "unit_price", 
        "line_total"
    }
    
    # These fields represent 'markers' (headers/footers) so they don't NEED a capture group.
    marker_fields = {"line_item_block_start", "line_item_block_end"}

    empty_line_item_keys = []
    bad_line_item_group_keys = []

    for k in required_line_item_keys:
        v = line_item.get(k)
        if not isinstance(v, str):
            empty_line_item_keys.append(f"line_item_level.{k} (not a string)")
            continue

        # If it's a strict field, it cannot be empty.
        if k in strict_line_fields and v.strip() == "":
            print(f"[WARN] Field '{k}' is empty in LLM output!")
            empty_line_item_keys.append(f"line_item_level.{k}")

        # If we have a regex (non-empty), validate capture groups
        # FIX: Skip check for marker fields
        if v.strip() != "" and k not in marker_fields:
            groups = _count_capture_groups(v)
            # FIX: RELAXED VALIDATION. Allow > 1 group.
            if groups < 1:
                bad_line_item_group_keys.append(f"line_item_level.{k} (capture groups={groups})")
            elif groups > 1:
                 print(f"[WARN] line_item_level.{k} has {groups} capture groups. System will default to group(1).")

    if empty_line_item_keys:
        raise Phase2ValidationError(f"Empty/Invalid strictly required line_item_level regex values: {empty_line_item_keys}")
    if bad_line_item_group_keys:
        raise Phase2ValidationError(f"line_item_level regex capture-group errors: {bad_line_item_group_keys}")

    # If we reach here, parsed appears valid per rules
    print("[SUCCESS] Phase 2 Validation Passed.")
    return parsed

def parse_llm_json(output: str) -> dict:
    """
    Extract and parse JSON from an LLM string output.
    Returns a Python dict.
    Raises ValueError if valid JSON cannot be extracted.
    """

    # 1. Try direct JSON parsing (ideal case)
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown-style fences
    fenced = re.search(r"```(?:json)?(.*?)```", output, re.S)
    if fenced:
        cleaned = fenced.group(1).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # 3. Extract the first {...} block
    bracket_match = re.search(r"\{[\s\S]*\}", output)
    if bracket_match:
        possible_json = bracket_match.group(0)
        try:
            return json.loads(possible_json)
        except json.JSONDecodeError:
            pass
    
    # 4. Try to extract JSON between specific markers
    json_markers = [
        (r"```json\s*(.*?)\s*```", re.DOTALL),
        (r"```\s*(.*?)\s*```", re.DOTALL),
        (r"JSON:\s*(\{.*?\})", re.DOTALL),
        (r"Output:\s*(\{.*?\})", re.DOTALL)
    ]
    
    for pattern, flags in json_markers:
        match = re.search(pattern, output, flags)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue
    
    # 5. Last attempt: clean common issues and try again
    cleaned_output = output.strip()
    # Remove trailing commas before closing braces/brackets
    cleaned_output = re.sub(r',(\s*[}\]])', r'\1', cleaned_output)
    # Remove comments
    cleaned_output = re.sub(r'//.*?$', '', cleaned_output, flags=re.MULTILINE)
    cleaned_output = re.sub(r'/\*.*?\*/', '', cleaned_output, flags=re.DOTALL)
    
    try:
        return json.loads(cleaned_output)
    except json.JSONDecodeError:
        pass

    # 6. If everything fails → explicit error with sample output
    sample = output[:200] + "..." if len(output) > 200 else output
    print(f"[ERROR] Could not parse LLM JSON. Sample output: {sample}")
    raise ValueError(f"Could not extract valid JSON from LLM output. First 200 chars: {sample}")

# ----------------------------
# Public API functions
# ----------------------------
def search_vendor_by_signals(signals: Dict[str, Optional[str]]) -> Optional[Tuple[str, str]]:
    """
    Attempt to find a vendor_id using signals in order of confidence:
    1) website (Exact/Domain match)
    2) vendor_email_id (Exact match)
    3) vendor_phone_number (Digits match)
    4) vendor_physical_address (Normalized match)
    5) vendor_name (Normalized match)

    Returns (vendor_id, matched_by) or None if not found.
    matched_by is one of "website", "email", "phone", "address", "name"
    """
    # 1. Unpack signals using the specific keys you provided
    website = signals.get("website")
    email = signals.get("vendor_email_id")
    phone = signals.get("vendor_phone_number")
    address = signals.get("vendor_physical_address")
    name = signals.get("vendor_name")

    # 2. Search by Website (Highest Confidence)
    if website:
        vid = find_vendor_by_website(website)
        if vid:
            return vid, "website"

    # 3. Search by Email
    if email:
        vid = find_vendor_by_email(email)
        if vid:
            return vid, "email"

    # 4. Search by Phone
    if phone:
        vid = find_vendor_by_phone(phone)
        if vid:
            return vid, "phone"

    # 5. Search by Physical Address (New)
    if address:
        vid = find_vendor_by_address(address)
        if vid:
            return vid, "address"

    # 6. Search by Name (Lowest Confidence / Fallback)
    if name:
        normalized = _normalize_name(name)
        if len(normalized) >= 3:
            vid = find_vendor_by_name(normalized)
            if vid:
                return vid, "name"

    return None

# --- Helpers ---
def find_vendor_by_address(extracted_addr: str) -> Optional[str]:
    """
    Finds vendor by exact address match. Returns vendor_id string only.
    """
    if not extracted_addr:
        return None
        
    # Call DB method (returns ObjectId or None)
    vendor_id_object = get_vendor_by_address(extracted_addr.strip())
    
    # Convert ObjectId -> string
    return str(vendor_id_object) if vendor_id_object else None

def find_vendor_by_website(extracted_url: str) -> Optional[str]:
    """
    Finds vendor by exact website match. Returns vendor_id string only.
    """
    if not extracted_url: 
        return None
        
    vendor_id_object = get_vendor_by_website(extracted_url.strip())
    
    return str(vendor_id_object) if vendor_id_object else None

def find_vendor_by_email(extracted_email: str) -> Optional[str]:
    """
    Finds vendor by exact email match. Returns vendor_id string only.
    """
    if not extracted_email:
        return None
        
    vendor_id_object = get_vendor_by_email(extracted_email.strip())
    
    return str(vendor_id_object) if vendor_id_object else None

def find_vendor_by_phone(extracted_phone: str) -> Optional[str]:
    """
    Finds vendor by exact phone match. Returns vendor_id string only.
    """
    # Strip non-digits to ensure clean match against DB
    target = re.sub(r'\D', '', str(extracted_phone))
    
    if len(target) < 7: 
        return None

    vendor_id_object = get_vendor_by_phone(target)
    
    return str(vendor_id_object) if vendor_id_object else None

def find_vendor_by_name(target: str) -> Optional[str]:
    """
    Finds vendor by exact name match. Returns vendor_id string only.
    """
    if not target:
        return None
        
    vendor_id_object = get_vendor_by_name(target)
    
    return str(vendor_id_object) if vendor_id_object else None


def find_vendor_name_by_id(id: str) -> Optional[str]:
    """Finds Vendor name based on id

    Args:
        id (str): vendor_id

    Returns:
        Optional[str]: vendor_name
    """
    if not id:
        return None
    
    vendor_name = get_vendor_name_by_id(id)
    
    return vendor_name if vendor_name else None

def apply_regex_extraction(text: str, regex_patterns: Union[List[str], Dict[str, Any]]) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """
    Applies the strict 0-10 positional regex array to the raw invoice text.
    
    Index Mapping:
    0: invoice_number        (Invoice Level)
    1: invoice_date          (Invoice Level)
    2: invoice_total_amount  (Invoice Level)
    3: order_date            (Invoice Level)
    4: line_item_block_start (Start Marker)
    5: line_item_block_end   (End Marker)
    6: quantity              (Line Item Level)
    7: description           (Line Item Level)
    8: unit                  (Line Item Level)
    9: unit_price            (Line Item Level)
    10: line_total           (Line Item Level)
    """

    # ---------------------------------------------------
    # 0. ADAPTER: Convert Dict to List if needed
    # ---------------------------------------------------
    if isinstance(regex_patterns, dict):
        # We need to flatten the dictionary to match the strict index mapping
        inv = regex_patterns.get("invoice_level", {})
        li = regex_patterns.get("line_item_level", {})
        
        regex_patterns = [
            inv.get("invoice_number", ""),       # 0
            inv.get("invoice_date", ""),         # 1
            inv.get("invoice_total_amount", ""), # 2
            inv.get("order_date", ""),           # 3
            li.get("line_item_block_start", ""), # 4
            li.get("line_item_block_end", ""),   # 5
            li.get("quantity", ""),              # 6
            li.get("description", ""),           # 7
            li.get("unit", ""),                  # 8
            li.get("unit_price", ""),            # 9
            li.get("line_total", "")             # 10
        ]

    # ---------------------------------------------------
    # 1. UNPACK PATTERNS (Indices 0-10)
    # ---------------------------------------------------
    # We unpack them into named variables for absolute clarity
    p_inv_num      = regex_patterns[0]
    p_inv_date     = regex_patterns[1]
    p_inv_total    = regex_patterns[2]
    p_order_date   = regex_patterns[3]
    
    p_block_start  = regex_patterns[4]
    p_block_end    = regex_patterns[5]
    
    p_li_qty       = regex_patterns[6]
    p_li_desc      = regex_patterns[7]
    p_li_unit      = regex_patterns[8]
    p_li_price     = regex_patterns[9]
    p_li_total     = regex_patterns[10]

    # ---------------------------------------------------
    # 2. INVOICE LEVEL EXTRACTION (Indices 0, 1, 2, 3)
    # ---------------------------------------------------
    inv_data = {
        "invoice_number": None,
        "invoice_date": None,
        "invoice_total_amount": None,
        "order_date": None
    }

    # Helper to apply regex safely
    def extract_val(pattern, text):
        if pattern and pattern.strip():
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return None

    inv_data["invoice_number"]       = extract_val(p_inv_num, text)
    inv_data["invoice_date"]         = extract_val(p_inv_date, text)
    inv_data["invoice_total_amount"] = extract_val(p_inv_total, text)
    inv_data["order_date"]           = extract_val(p_order_date, text)

    # ---------------------------------------------------
    # 3. DEFINE SEARCH BLOCK (Indices 4, 5)
    # ---------------------------------------------------
    start_index = 0
    end_index = len(text)

    # Index 4: Block Start
    if p_block_start and p_block_start.strip():
        s_match = re.search(p_block_start, text)
        if s_match:
            start_index = s_match.end() # Start reading AFTER the header

    # Index 5: Block End
    if p_block_end and p_block_end.strip():
        # Search only in the remaining text
        e_match = re.search(p_block_end, text[start_index:])
        if e_match:
            end_index = start_index + e_match.start() # Stop reading BEFORE the footer

    # Slice the text to isolate the table
    block_text = text[start_index:end_index]
    lines = block_text.split('\n')

    # ---------------------------------------------------
    # 4. LINE ITEM EXTRACTION (Indices 6, 7, 8, 9, 10)
    # ---------------------------------------------------
    line_items = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # We will try to match ALL line item regexes (6-10).
        # If a regex pattern exists (is not empty) but fails to match, 
        # then this line is likely NOT a valid line item (or is garbage text).
        
        current_item = {}
        is_valid_line = True

        # Mapping for Line Items
        # (Pattern, Key Name)
        li_map = [
            (p_li_qty,   "quantity"),    # Index 6
            (p_li_desc,  "description"), # Index 7
            (p_li_unit,  "unit"),        # Index 8
            (p_li_price, "unit_price"),  # Index 9
            (p_li_total, "line_total")   # Index 10
        ]

        for pattern, key in li_map:
            if pattern and pattern.strip():
                match = re.search(pattern, line)
                if match:
                    current_item[key] = match.group(1).strip()
                else:
                    # STRICT MODE: If a pattern was provided but didn't match, 
                    # we assume this line is not a valid line item.
                    is_valid_line = False
                    break
            else:
                # If pattern is empty string "", we just set the value to None
                current_item[key] = None

        if is_valid_line:
            # Only add if we actually extracted something useful (e.g. at least a description)
            if current_item.get("description") or current_item.get("line_total"):
                current_item["raw_line"] = line # Helpful for debugging
                line_items.append(current_item)

    return inv_data, line_items

def identify_vendor_and_get_regex(text: str, file_path: str) -> Dict[str, Any]:
    """
    Main orchestration function used by pipeline.

    Flow:
      1) extract signals
      2) search DB by signal priority
      3) if vendor found and regex exists -> return it
      4) if vendor found but no regex -> if USE_LLM call phases to generate & save; else return vendor with regex=None created=False
      5) if vendor not found -> if USE_LLM call phases to generate vendor+regex and save; else create vendor with signals and return created=True with regex=None

    Returns:
      {
        "vendor_id": str,
        "vendor_name": str,
        "regex": List[str] | None,
        "created": bool,
        "matched_by": Optional[str]  # "email"|"phone"|"website"|"name"|None
      }
    """
    # 1. Extract signals using the new corrected method
    signals = extract_vendor_signals(text)

    # 2. Search for existing vendor
    search_result = search_vendor_by_signals(signals)
    if search_result:
        vendor_id, matched_by = search_result
    else:
        vendor_id = None
        matched_by = None

    # --- CASE A: Vendor Found in DB ---
    if vendor_id:
        vendor_name = find_vendor_name_by_id(vendor_id)
        # A1. Existing Regex
        regex_list = get_regex_for_vendor(vendor_id)
        print("fetched existing vendor regex")
        
        if regex_list:
            return {
                "vendor_id": vendor_id,
                "vendor_name": vendor_name,
                "regex": regex_list,
                "created": False,
                "matched_by": matched_by
            }
        
        # A2. Vendor exists, but NO Regex found -> Raise error
        else:
            raise ValueError("Vendor found — But no regex for that vendor found in DB")

    # --- CASE B: Vendor NOT Found (Create New) ---
    else:
        # Phase 1: Extract clean master data
        phase1 = llm_phase1_extract(text)
        if phase1:
                # Merge new LLM data with existing signals
                vendor = phase1["vendor_master_data"]
                
                # Phase 2: Generate Regex
                new_regexes = llm_phase2_generate_regex(text, phase1)
                
                # Create Vendor
                new_vendor_id = save_vendor_details(vendor)
                vendor_name = find_vendor_name_by_id(vendor_id)
                
                if new_regexes:
                    save_regex_for_vendor(new_vendor_id, new_regexes)
                    print("Generated new vendor regex")
                    return {
                        "vendor_id": new_vendor_id,
                        "vendor_name": vendor_name,
                        "regex": new_regexes,
                        "created": True,
                        "matched_by": None
                    }
                else:
                    # LLM Phase 2 failed
                    raise ValueError("Vendor not found — LLM Phase 2 failed")
        else:
            raise ValueError("Vendor not found — But llm phase-1 output failed")
