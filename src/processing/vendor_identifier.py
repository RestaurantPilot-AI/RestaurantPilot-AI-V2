import os
import re
import time
import json
import traceback
import mimetypes
from typing import Optional, Dict, Any, Tuple, List, Union
from dotenv import load_dotenv

# For real LLM integration
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
# import google.api_core.exceptions

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
    
    print(f"[INFO] Vendor info to save [vendor_identifier.py]: {vendor_master_data}")
    return create_vendor(vendor_master_data)

def normalize_item_block(block: str) -> str:
    lines = block.splitlines()

    new_lines = []
    removed = False

    for line in lines:
        if not removed and line.strip():
            removed = True
            continue
        new_lines.append(line)

    return "\n".join(new_lines)

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
    
    phone_label_pattern = re.compile(
        r'(?:Phone|Tel|Mobile|Cell|Ph|T)[:\.\-\s]+([+\d\(\)\-\s]{7,20})',
        re.IGNORECASE
    )

    phone_strict_pattern = re.compile(
        r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    )

    def normalize_phone(p: str) -> str:
        return re.sub(r'\D', '', p)

    BLACKLISTED_PHONES = {
        "3525028078"
    }

    # Attempt 1: Look for labelled phone numbers (Best Match)
    for line in lines[:50]:  # Usually in header
        m_label = phone_label_pattern.search(line)
        if m_label:
            raw_num = m_label.group(1).strip()
            if sum(c.isdigit() for c in raw_num) >= 7:
                if normalize_phone(raw_num) not in BLACKLISTED_PHONES:
                    signals["vendor_phone_number"] = raw_num
                    break

# Attempt 2: Strict Regex scan if no label found
    if not signals.get("vendor_phone_number"):
        matches = phone_strict_pattern.findall(text[:2000])
        if matches:
            valid_phones = []
            for p in matches:
                # Calculate normalized version once for checks below
                norm = normalize_phone(p)

                # --- NEW CONDITION 1: Spacing Check ---
                # Should not contain more than 2 continuous spaces (rejects 3 or more)
                if '   ' in p:
                    continue

                # --- NEW CONDITION 2: Digit Length Check ---
                # Valid if: 10 digits OR (11 digits AND starts with '1')
                if len(norm) == 10:
                    pass # Valid
                elif len(norm) == 11 and norm.startswith('1'):
                    pass # Valid
                else:
                    continue # Skip if length is weird (e.g. 7, 12) or 11 digits without country code 1

                # 1. Filter out date-like false positives
                if re.match(r'20[2-3]\d', p):
                    continue
                
                # 2. Check Blacklist
                if norm in BLACKLISTED_PHONES:
                    continue

                # 3. Strict Bracket Rule
                if '(' in p:
                    prefix = p.split('(')[0].strip()
                    if prefix not in ['', '1', '+1']:
                        continue

                valid_phones.append(p)

            if valid_phones:
                signals["vendor_phone_number"] = valid_phones[0]


    # --- 3. Website ---
    # Strategy: Find URL patterns, excluding the email domain we just found.
    url_pattern = r'(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'
    
    potential_urls = []
    for line in lines: # Websites usually in header or footer
        # Skip lines that look like emails
        # if "@" in line: 
        #     continue
            
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
        legal_suffixes = r'\b(Inc|LLC|Ltd|GmbH|BV|B\.V\.|Co\.|Company|Corp|Corporation|S\.A\.|S\.L\.|AG|Pty|Pvt|Private|Plc|Service)\b'
        
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
def call_visual_llm_api(prompt: str, file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Calls Gemini API with optional multimodal file upload and automatic retry on Rate Limit (429) errors.
    If `file_path` is provided, the file will be uploaded and passed to the model alongside the prompt.
    Returns parsed JSON dictionary.
    """
    _setup_environment()
    model = genai.GenerativeModel(MODEL_NAME)

    # Prepare optional file upload for multimodal inputs
    file_obj = None
    if file_path:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found or invalid: {file_path}")
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            # Fallback based on extension
            if file_path.lower().endswith(".pdf"):
                mime_type = "application/pdf"
            else:
                mime_type = "image/jpeg"
        try:
            file_obj = genai.upload_file(file_path, mime_type=mime_type)
        except Exception as e:
            print(f"[ERROR] Failed to upload file to Gemini: {e}")
            raise

    max_retries = 3
    base_wait = 30  # Start waiting 30 seconds

    for attempt in range(max_retries):
        try:
            # --- Call Gemini model ---
            if file_obj:
                response = model.generate_content([prompt, file_obj])
            else:
                response = model.generate_content(prompt)

            raw_text = response.text.strip() if response and getattr(response, "text", None) else ""
            # Use existing robust parser
            return parse_llm_json(raw_text)

        except ResourceExhausted:
            # This catches the 429 Quota Exceeded error
            wait_time = base_wait * (attempt + 1)
            print(f"\n[WARN] Gemini Quota Exceeded. Waiting {wait_time}s before retry ({attempt + 1}/{max_retries})...")
            time.sleep(wait_time)

        except ValueError as e:
            # JSON parsing / content errors from the model
            print(f"[ERROR] Gemini returned invalid JSON: {e}")
            raise

        except Exception as e:
            # Other errors (auth, network) should crash immediately
            print(f"[ERROR] Gemini API Failed: {e}")
            raise

    # If we run out of retries
    raise RuntimeError("Gemini API Quota Exceeded after multiple retries. Please check your billing/limits.")

def call_llm_api(prompt: str) -> Dict[str, Any]:
    """
    Calls Gemini API text llm and automatic retry on Rate Limit (429) errors.
    Passes the prompt to text based llm.
    Returns parsed JSON dictionary.
    """
    _setup_environment()
    model = genai.GenerativeModel(MODEL_NAME)

    max_retries = 3
    base_wait = 30  # Start waiting 30 seconds

    for attempt in range(max_retries):
        try:

            response = model.generate_content(prompt)

            raw_text = response.text.strip() if response and getattr(response, "text", None) else ""
            # Use existing robust parser
            return parse_llm_json(raw_text)

        except ResourceExhausted:
            # This catches the 429 Quota Exceeded error
            wait_time = base_wait * (attempt + 1)
            print(f"\n[WARN] Gemini Quota Exceeded. Waiting {wait_time}s before retry ({attempt + 1}/{max_retries})...")
            time.sleep(wait_time)

        except ValueError as e:
            # JSON parsing / content errors from the model
            print(f"[ERROR] Gemini returned invalid JSON: {e}")
            raise

        except Exception as e:
            # Other errors (auth, network) should crash immediately
            print(f"[ERROR] Gemini API Failed: {e}")
            raise

    # If we run out of retries
    raise RuntimeError("Gemini API Quota Exceeded after multiple retries. Please check your billing/limits.")

def make_phase1_prompt(text: str) -> str:
    """Construct phase-1 prompt (extract structured JSON matching DB Schema)."""
    return f"""
Extract structured data from the following invoice text.

Rules:
- Use ONLY the information explicitly present.
- If a value is missing or not found, return null (not the string "None").
- Output ONLY the JSON object. No explanations.
- Field names MUST match exactly.

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
- "order_date": The date the order was placed (distinct from invoice date if not specified use date), if available.
- "vendor_name": The official name of the vendor/supplier.

-------------------------
RAW INVOICE TEXT BELOW:
-------------------------
{text}
""".strip()

def make_phase2_prompt(raw_text: str, verified_json: Dict[str, Any]) -> str:
    """
    Construct a single-shot prompt that asks the model to return strict,
    reusable, POSITION-LOCKED regex patterns for this invoice layout.
    """

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
            "line_item_split": "",
            "quantity": "",
            "description": "",
            "unit": "",
            "unit_price": "",
            "line_total": ""
        }
    }

    return f"""
You are given two inputs:
1) RAW INVOICE TEXT (plain text extracted from a PDF; line order is preserved but visual alignment is NOT)
2) VERIFIED JSON (the correct extracted values for this invoice)

Your task is NOT to extract values.
Your task is to GENERATE REUSABLE, STRICT REGEX TEMPLATES that will extract the same fields
from future invoices with the SAME LAYOUT but DIFFERENT VALUES.

The layout is assumed to be stable. Values change. Positions do not.

========================
ABSOLUTE OUTPUT RULES
========================
- Return ONLY one JSON object and nothing else.
- JSON keys and nesting MUST match the required schema exactly.
- Do NOT include regex delimiters or flags (no /.../, no (?i), etc.).
- Do NOT explain anything. No comments. No markdown.

========================
CRITICAL REGEX RULES (HARD)
========================
1. EXACTLY ONE CAPTURING GROUP:
   - Every regex that extracts a value MUST contain exactly ONE capturing group (...).
   - All other grouping MUST be non-capturing (?:...).

2. POSITIONAL, NOT SEMANTIC:
   - Regexes MUST rely on layout position, labels, and fixed offsets.
   - Do NOT "search" for values using .*?, [\\s\\S]*?, or loose patterns.
   - If a value appears N lines below its label, the regex MUST encode those exact \\n jumps.

3. NO GLOBAL MATCHING:
   - A value regex MUST FAIL if applied to the wrong section of the invoice.
   - Invoice-level regexes must rely on their labels.
   - Line-item regexes must ONLY work inside a single line-item string.

4. MANDATORY FIELDS:
   - invoice_number
   - invoice_date
   - invoice_total_amount
   These MUST NOT be empty.

========================
LABEL PRIORITY RULES
========================
- Prefer "Invoice Number" over "Order Number" if both exist.
- Prefer "Invoice Date" over "Order Date".
- Only fall back if the preferred label is completely absent from RAW TEXT.

========================
INVOICE-LEVEL RULES
========================
- Labels and values may NOT be on the same line.
- If the VERIFIED JSON value appears X lines below its label, encode that offset explicitly.
- Do NOT assume visual alignment implies same-line text.

========================
LINE ITEM RULES (VERY IMPORTANT)
========================
Line item parsing MUST be MECHANICAL and POSITIONAL.

1. line_item_block_start:
   - Match the header or first marker that indicates line items begin.
   - No capturing group.

2. line_item_block_end:
   - Match the first line immediately AFTER the final line item
     (e.g., Subtotal, Tax, Total, footer).
   - No capturing group.

3. line_item_split:
   - Regex used to split the line-item block into INDIVIDUAL LINE ITEM STRINGS.
   - Each split result represents ONE item only.
   - No capturing group.

4. Field extraction (description, quantity, unit, unit_price, line_total):
   - These regexes MUST assume they are applied to ONE line-item string only.
   - They MUST rely on FIXED POSITION / ORDER within that string.
   - Do NOT infer meaning from words.
   - Do NOT search the full invoice.

========================
INTERNAL REASONING REQUIREMENT (DO NOT OUTPUT)
========================
Before writing each regex, you MUST determine:
1. The anchor label or structural marker
2. The exact number of \\n jumps to the value (if any)
3. The minimal strict pattern that matches ONLY the VERIFIED value

========================
REQUIRED JSON STRUCTURE
========================
{json.dumps(schema, indent=2)}

========================
RAW INVOICE TEXT
========================
{raw_text}

========================
VERIFIED JSON (GROUND TRUTH)
========================
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


def llm_phase1_extract(text: str, file_path: str) -> Dict[str, Any]:
    """
    Phase 1: ask LLM to extract key fields as JSON.

    Returns parsed JSON dict on success.

    Raises:
        Phase1ParseError: when LLM output is not valid JSON / not extractable.
        Phase1ValidationError: when JSON is missing required keys or has wrong types.
    """
    print("\n[INFO] Starting Phase 1: Extract Vendor Details and Patterns...")
    
    prompt = make_phase1_prompt(text)
    
    # call_visual_llm_api that is multimodal ll,
    parsed = call_visual_llm_api(prompt, file_path)

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

def llm_phase2_generate_regex(text: str, phase1_json: Dict[str, Any], file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Phase 2: ask LLM to generate strict regex templates (invoice_level & line_item_level).
    Returns parsed template dict on success.

    Raises:
        Phase2ParseError: when LLM output is not valid JSON / not extractable.
        Phase2ValidationError: when JSON is missing keys or regexes fail basic checks.
    """
    print("\n[INFO] Starting Phase 2: Generating Regex Patterns...")
    prompt = make_phase2_prompt(text, phase1_json)
    
    # call_llm_api passes the prompt to a text based llm
    parsed = call_llm_api(prompt)

    # DEBUG: Print the raw output to see what the LLM actually gave us
    # print(f"[DEBUG] Phase 2 Raw LLM Output:\n{json.dumps(parsed, indent=2)}")

    # =========================================================
    # 0. ROOT VALIDATION
    # =========================================================
    if not isinstance(parsed, dict):
        snippet = str(parsed or "")[:1000]
        raise Phase2ParseError(
            f"Phase2 parse failure: LLM output not JSON-dict. Raw output snippet: {snippet!s}"
        )

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

    required_invoice_keys = {
        "invoice_number",
        "invoice_date",
        "invoice_total_amount",
        "order_date",
    }
    missing_invoice = sorted(list(required_invoice_keys - set(invoice.keys())))
    if missing_invoice:
        raise Phase2ValidationError(f"invoice_level missing keys: {missing_invoice}")

    strict_invoice_fields = {
        "invoice_number",
        "invoice_date",
        "invoice_total_amount",
    }

    empty_invoice_keys = []
    bad_group_keys = []

    for k in required_invoice_keys:
        v = invoice.get(k)
        if not isinstance(v, str):
            empty_invoice_keys.append(f"invoice_level.{k} (not a string)")
            continue

        if k in strict_invoice_fields and v.strip() == "":
            print(f"[WARN] Field '{k}' is empty in LLM output!")
            empty_invoice_keys.append(f"invoice_level.{k}")

        if v.strip() != "":
            groups = _count_capture_groups(v)
            if groups < 1:
                bad_group_keys.append(f"invoice_level.{k} (capture groups={groups})")
            elif groups > 1:
                print(
                    f"[WARN] invoice_level.{k} has {groups} capture groups. "
                    "System will default to group(1)."
                )

    if empty_invoice_keys:
        raise Phase2ValidationError(
            f"Empty/Invalid strictly required invoice_level regex values: {empty_invoice_keys}"
        )
    if bad_group_keys:
        raise Phase2ValidationError(
            f"invoice_level regex capture-group errors: {bad_group_keys}"
        )

    # =========================================================
    # 2. LINE ITEM LEVEL VALIDATION
    # =========================================================
    line_item = parsed.get("line_item_level")
    if not isinstance(line_item, dict):
        raise Phase2ValidationError("line_item_level must be an object/dict.")

    required_line_item_keys = {
        "line_item_block_start",
        "line_item_block_end",
        "line_item_split",
        "description",
        "quantity",
        "unit",
        "unit_price",
        "line_total",
    }
    missing_line_item = sorted(list(required_line_item_keys - set(line_item.keys())))
    if missing_line_item:
        raise Phase2ValidationError(f"line_item_level missing keys: {missing_line_item}")

    strict_line_fields = {
        "line_item_block_start",
        "line_item_block_end",
        "description",
        "quantity",
        "unit_price",
        "line_total",
    }


    # Fields that must NEVER be capture-group validated
    non_capture_fields = {
        "line_item_block_start",
        "line_item_block_end",
        "line_item_split",
    }

    empty_line_item_keys = []
    bad_line_item_group_keys = []

    for k in required_line_item_keys:
        v = line_item.get(k)
        if not isinstance(v, str):
            empty_line_item_keys.append(f"line_item_level.{k} (not a string)")
            continue

        if k in strict_line_fields and v.strip() == "":
            print(f"[WARN] Field '{k}' is empty in LLM output!")
            empty_line_item_keys.append(f"line_item_level.{k}")

        if v.strip() != "" and k not in non_capture_fields:
            groups = _count_capture_groups(v)
            if groups < 1:
                bad_line_item_group_keys.append(
                    f"line_item_level.{k} (capture groups={groups})"
                )
            elif groups > 1:
                print(
                    f"[WARN] line_item_level.{k} has {groups} capture groups. "
                    "System will default to group(1)."
                )

    if empty_line_item_keys:
        raise Phase2ValidationError(
            f"Empty/Invalid strictly required line_item_level regex values: {empty_line_item_keys}"
        )
    if bad_line_item_group_keys:
        raise Phase2ValidationError(
            f"line_item_level regex capture-group errors: {bad_line_item_group_keys}"
        )

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
        vid = find_vendor_by_name(name)
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
    vendor_id_object = get_vendor_by_phone(extracted_phone)
    
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

def apply_regex_extraction(
    text: str,
    regex_patterns: Union[List[str], Dict[str, Any]]
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """
    Applies the strict positional regex array to the raw invoice text.

    Index Mapping:
    0: invoice_number        (Invoice Level)
    1: invoice_date          (Invoice Level)
    2: invoice_total_amount  (Invoice Level)
    3: order_date            (Invoice Level)
    4: line_item_block_start (Start Marker)
    5: line_item_block_end   (End Marker)
    6: line_item_split       (Split block into individual items)
    7: quantity              (Line Item Level)
    8: description           (Line Item Level)
    9: unit                  (Line Item Level)
    10: unit_price           (Line Item Level)
    11: line_total           (Line Item Level)
    """

    # ---------------------------------------------------
    # 0. ADAPTER: Convert Dict to List if needed
    # ---------------------------------------------------
    if isinstance(regex_patterns, dict):
        inv = regex_patterns.get("invoice_level", {})
        li  = regex_patterns.get("line_item_level", {})

        regex_patterns = [
            inv.get("invoice_number", ""),
            inv.get("invoice_date", ""),
            inv.get("invoice_total_amount", ""),
            inv.get("order_date", ""),
            li.get("line_item_block_start", ""),
            li.get("line_item_block_end", ""),
            li.get("line_item_split", ""),
            li.get("quantity", ""),
            li.get("description", ""),
            li.get("unit", ""),
            li.get("unit_price", ""),
            li.get("line_total", ""),
        ]

    (
        p_inv_num,
        p_inv_date,
        p_inv_total,
        p_order_date,
        p_block_start,
        p_block_end,
        p_li_split,
        p_li_qty,
        p_li_desc,
        p_li_unit,
        p_li_price,
        p_li_total,
    ) = regex_patterns

    # ---------------------------------------------------
    # 1. INVOICE LEVEL EXTRACTION
    # ---------------------------------------------------
    def extract_val(pattern, txt):
        if not pattern or not txt:
            return None

        m = re.search(pattern, txt)
        if not m:
            return None

        val = m.group(1)
        if not val:
            return None

        return val.strip()

    inv_data = {
        "invoice_number": extract_val(p_inv_num, text),
        "invoice_date": extract_val(p_inv_date, text),
        "invoice_total_amount": extract_val(p_inv_total, text),
        "order_date": extract_val(p_order_date, text),
    }

    # ---------------------------------------------------
    # 2. ISOLATE LINE ITEM BLOCK
    # ---------------------------------------------------
    start_index = 0
    end_index = len(text)

    if p_block_start and p_block_start.strip():
        m = re.search(p_block_start, text, re.MULTILINE)
        if m:
            start_index = m.end()

    if p_block_end and p_block_end.strip():
        m = re.search(p_block_end, text[start_index:], re.MULTILINE)
        if m:
            end_index = start_index + m.start()

    block_text = text[start_index:end_index]
    
    print(f"Block text: {block_text}")

    # ---------------------------------------------------
    # 3. MODE DETECTION
    # ---------------------------------------------------

    is_block_mode = p_li_split is not None
    
    line_items: List[Dict[str, Any]] = []


    # ---------------------------------------------------
    # 4A. BLOCK-WISE PARSING (multi-line items)
    # ---------------------------------------------------
    if is_block_mode and p_li_split:
        # Split block into item chunks using index 6
        try:
            chunks = re.split(p_li_split, block_text)

            for idx, chunk in enumerate(chunks):
                try:
                    chunk = chunk.strip()
                    if not chunk:
                        continue

                    print(f"Chunk[{idx}]: {chunk}\n")

                    description = extract_val(p_li_desc, chunk)
                    if description:
                        description = re.sub(r"\s*\n\s*", " ", description).strip()

                    item = {
                        "quantity": extract_val(p_li_qty, chunk),
                        "description": description,
                        "unit": extract_val(p_li_unit, chunk),
                        "unit_price": extract_val(p_li_price, chunk),
                        "line_total": extract_val(p_li_total, chunk),
                    }

                    if item["description"] and (item["line_total"] or item["unit_price"]):
                        line_items.append(item)

                except Exception as e:
                    print(
                        f"[ERROR] Failed processing chunk index={idx}\n"
                        f"Chunk preview: {chunk[:200]}\n"
                        f"Exception: {e}"
                    )
                    traceback.print_exc()
                    raise

        except Exception:
            print("[ERROR] Line-item split stage failed")
            traceback.print_exc()
            raise

    # ---------------------------------------------------
    # 4B. LINE-WISE PARSING (single-line items)
    # ---------------------------------------------------
    else:
        for line in block_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            item = {
                "quantity": extract_val(p_li_qty, line),
                "description": extract_val(p_li_desc, line),
                "unit": extract_val(p_li_unit, line),
                "unit_price": extract_val(p_li_price, line),
                "line_total": extract_val(p_li_total, line),
            }

            if item["description"] or item["line_total"]:
                line_items.append(item)

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
    # The above code is a Python script that contains a commented-out line `# print(signals)`. This
    # line is not executed when the script runs because it is commented out. It appears that the
    # script is intended to print the variable `signals`, but it is currently disabled.
    print(signals)

    # 2. Search for existing vendor
    search_result = search_vendor_by_signals(signals)
    if search_result:
        vendor_id, matched_by = search_result
    else:
        vendor_id = None
        matched_by = None

    # print(f"Matched by: ", matched_by)
    # --- CASE A: Vendor Found in DB ---
    if vendor_id:
        print(f"[INFO] Vendor ID: {vendor_id}")
        vendor_name = find_vendor_name_by_id(vendor_id)
        # A1. Existing Regex
        # print(vendor_name)
        regex_list = get_regex_for_vendor(vendor_id)
        
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
        phase1 = llm_phase1_extract(text,file_path)
        if phase1:
                # Get new Vendor data from llm Phase 1 output
                vendor = phase1["vendor_master_data"]
                
                # Extract the specific pieces of data
                invoice_data = phase1.get("invoice_details")
                line_items = phase1.get("line_items")
                
                # Create a filtered context for Phase 2 
                # This ensures "vendor_master_data" is NOT included
                phase2_input = {
                    "invoice_details": invoice_data,
                    "line_items": line_items
                }
                
                # Phase 2: Generate Regex
                new_regexes = llm_phase2_generate_regex(text, phase2_input, file_path)
                
                # Create Vendor
                new_vendor_id = save_vendor_details(vendor)
                vendor_name = find_vendor_name_by_id(vendor_id)
                print(f"[DEBUG] vendor_identifier.py regex:\n{json.dumps(new_regexes, indent=2)}")

                
                if new_regexes:
                    save_regex_for_vendor(new_vendor_id, new_regexes)
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
