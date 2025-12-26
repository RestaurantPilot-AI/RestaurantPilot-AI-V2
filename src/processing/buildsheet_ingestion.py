import os
import re
from typing import List, Dict, Any
import pandas as pd

from src.storage.database import (
    get_buildsheet_regex_patterns,
    save_buildsheet_regex_template,
)

from src.processing.vendor_identifier import call_llm_api

# Separator tokens used to convert DataFrame -> text. These are explicit and documented
# so downstream regexes and LLM prompts can rely on them.
CELL_SEP = "<CS>"
ROW_SEP = "<RS>"


def _df_to_text(extracted_df: pd.DataFrame) -> str:
    """Convert a DataFrame into a normalized text representation.

    - Each cell is separated by <CS>
    - Each row is separated by <RS>.
    - Header row is included as the first row

    Returns a single string used as input for regex/LLM extraction.
    """
    if extracted_df is None or extracted_df.empty:
        return ""

    # Ensure all columns are stringified and preserve order
    df = extracted_df.fillna("").astype(str)

    rows: List[str] = []

    # Header
    header_cells = [str(c) for c in df.columns]
    rows.append(CELL_SEP.join(header_cells))

    for _, r in df.iterrows():
        cells = [str(r[c]) for c in df.columns]
        rows.append(CELL_SEP.join(cells))

    return ROW_SEP.join(rows)


def _apply_regex_patterns_to_text(extracted_text: str, regex_patterns: List[str]) -> pd.DataFrame:
    """
    Apply positional regex list to the extracted_text and return a DataFrame
    with columns: name, yield_quantity, ingredients (list of dicts).

    This implementation is BLOCK-BASED:
    - item_start / item_end define menu item boundaries
    - ingredient_start / ingredient_end define ingredient blocks
    - All regexes are Python re-compatible string literals
    """

    # Defensive: ensure list of length 10
    pats = list(regex_patterns) if regex_patterns else []
    if len(pats) < 10:
        pats += [""] * (10 - len(pats))

    (
        p_item_start,
        p_item_end,
        p_menu_item_name,
        p_ingredient_start,
        p_ingredient_end,
        p_ingredient_name,
        p_ingredient_unit,
        p_ingredient_quantity,
        p_yield_quantity,
        p_row_exclusion,
    ) = pats[:10]

    text = extracted_text

    # -------------------------
    # 1. Split into ITEM BLOCKS
    # -------------------------
    item_blocks = []

    if p_item_start and p_item_end:
        start_re = re.compile(p_item_start, flags=re.IGNORECASE)
        end_re = re.compile(p_item_end, flags=re.IGNORECASE)

        starts = [m.start() for m in start_re.finditer(text)]

        for s in starts:
            end_match = end_re.search(text, pos=s + 1)
            if not end_match:
                continue

            block = text[s:end_match.start()]
            item_blocks.append(block)


    else:
        # Fallback: treat each row as its own item
        print("\n[WARN] No Blocks Found\n")

    results = []

    for block in item_blocks:
        print(f"\nBlock: \n{block}")


        # Optional exclusion
        if p_row_exclusion and re.search(p_row_exclusion, block, flags=re.IGNORECASE):
            continue

        # -------------------------
        # 2. MENU ITEM NAME
        # -------------------------
        name = None
        if p_menu_item_name:
            m = re.search(p_menu_item_name, block, flags=re.IGNORECASE)
            if m:
                name = m.group(1) if m.lastindex else m.group(0)
                name = name.strip()

        if not name:
            continue

        # -------------------------
        # 3. YIELD QUANTITY
        # -------------------------
        y = None
        if p_yield_quantity:
            m = re.search(p_yield_quantity, block, flags=re.IGNORECASE)
            if m:
                try:
                    y = float(m.group(1))
                except Exception:
                    try:
                        y = float(m.group(0))
                    except Exception:
                        y = None

        # -------------------------
        # 4. INGREDIENT BLOCK
        # -------------------------
        ingredient_block = None
        if p_ingredient_start and p_ingredient_end:
            m = re.search(
                p_ingredient_start + r"(.*?)" + p_ingredient_end,
                block,
                flags=re.DOTALL | re.IGNORECASE
            )
            if m:
                ingredient_block = m.group(1)
        else:
            ingredient_block = block

        ingredients = []

        if ingredient_block and p_ingredient_name:
            for m in re.finditer(p_ingredient_name, ingredient_block, flags=re.IGNORECASE):
                material = m.group(1) if m.lastindex else m.group(0)
                material = material.strip()

                unit = None
                qty = None

                if p_ingredient_unit:
                    mu = re.search(p_ingredient_unit, m.string[m.end():], flags=re.IGNORECASE)
                    if mu:
                        unit = mu.group(1) if mu.lastindex else mu.group(0)

                if p_ingredient_quantity:
                    mq = re.search(p_ingredient_quantity, m.string[m.end():], flags=re.IGNORECASE)
                    if mq:
                        try:
                            qty = float(mq.group(1))
                        except Exception:
                            try:
                                qty = float(mq.group(0))
                            except Exception:
                                qty = None

                ingredients.append({
                    "material_name": material,
                    "measure_unit": unit.strip() if unit else None,
                    "measure_quantity": qty,
                })
                

        results.append({
            "name": name,
            "yield_quantity": y,
            "ingredients": ingredients,
        })

    return pd.DataFrame(results, columns=["name", "yield_quantity", "ingredients"])



# -------------------------
# LLM Fallback Helpers
# -------------------------
# vendor_identifier import is performed lazily in `extract_buildsheet` to avoid
# import-time dependency on heavyweight LLM client modules during test runs.

def _make_buildsheet_phase1_prompt(extracted_text: str) -> str:
    return f"""
You are a deterministic data extraction assistant.

INPUT FORMAT:
- You are given normalized text extracted from a spreadsheet.
- Cells are separated by the literal token <CS>.
- Rows are separated by the literal token <RS>.
- Menu items may span MULTIPLE rows.

FIELD DEFINITIONS:
- name: Menu/buildsheet item name (include size or variant).
- yield_quantity: Total units or servings produced by the buildsheet.
- ingredients: Materials required to produce ONE yield batch.

INGREDIENT RULES:
- If multiple unit/quantity columns exist:
  - Choose the most descriptive unit (e.g., '2 oz' over '3 mm').
  - Choose the column that contains values for ALL ingredients.
- Do not mix quantities from different columns.

RULES:
- Use ONLY information present in the source.
- Do NOT invent ingredients or quantities.
- If something is missing or unclear, use null.
- The solution MUST include BOTH menu-item blocks AND standalone rows that can be interpreted as sides or extras.

OUTPUT FORMAT (JSON ONLY):
{{
  "buildsheet_items": [
    {{
      "name": "string",
      "yield_quantity": null | number,
      "ingredients": [
        {{
          "material_name": "string",
          "measure_unit": "string" | null,
          "measure_quantity": null | number
        }}
      ]
    }}
  ]
}}

SOURCE:
{extracted_text}
""".strip()



def _make_buildsheet_phase2_prompt(extracted_text: str, phase1_output: Dict[str, Any]) -> str:
    return f"""
You are a deterministic REGEX GENERATOR for Python's built-in `re` module.

THIS IS CRITICAL:
- You MUST generate ONLY Python `re` compatible regex.
- Your regex will be EXECUTED directly in Python.
- Invalid regex will cause the system to FAIL.

ABSOLUTE RESTRICTIONS (DO NOT VIOLATE):
- DO NOT use lookahead or lookbehind assertions of any kind.
  (No (?=...), (?!...), (?<=...), (?<!...))
- DO NOT use inline flags such as (?i), (?s), (?m).
- DO NOT use backreferences (\\1, \\2, etc.).
- DO NOT use named capture groups.
- DO NOT assume PCRE or JavaScript regex features.

ALLOWED REGEX FEATURES ONLY:
- Literal characters
- Character classes like [A-Za-z0-9 ]
- Quantifiers: *, +, ?, {{m,n}}
- Capturing groups: ( ... )
- Anchors: ^ and $
- Alternation: |

EXECUTION MODEL:
- item_start and item_end regex are applied to the FULL SOURCE TEXT.
- All other regexes are applied INSIDE EACH ITEM BLOCK.
- Regexes are used with Python flags re.IGNORECASE and re.DOTALL externally.
- Therefore you MUST NOT include flags in the pattern itself.

IMPORTANT STRUCTURAL FACT:
The SOURCE TEXT contains TWO DIFFERENT FORMATS:
1) STANDARD MENU ITEMS:
   - Have a clear item header (menu item name)
   - Followed by multiple ingredient rows
   - Terminated by empty / separator rows
2) STANDALONE / SIDE ROWS:
   - Do NOT have a menu item header
   - Each row is a self-contained item
   - The ingredient name appears where an item name would normally appear

YOUR REGEX MUST BE ROBUST ENOUGH TO HANDLE BOTH FORMATS:
- item_start / item_end should match STANDARD MENU ITEMS where possible
- If item boundaries are not found, the fallback behavior treats EACH ROW as its own item
- ingredient_name, quantity, and unit regexes MUST work for BOTH formats
- Do NOT assume that ingredient rows always follow an item header

TASK:
Return a JSON ARRAY of EXACTLY 10 STRINGS.
Each string must be a VALID Python regex or an empty string.
Index position is SEMANTICALLY SIGNIFICANT and MUST be preserved.

INDEX MAP (FIXED — DO NOT CHANGE):
0: item_start  – regex that matches the START of one buildsheet item
1: item_end    – regex that matches the END of one buildsheet item
2: menu_item_name – regex that CAPTURES the item name in group (1)
3: ingredient_start – start of ingredient section (optional)
4: ingredient_end   – end of ingredient section (optional)
5: ingredient_name  – regex that CAPTURES ingredient name in group (1)
6: ingredient_unit  – regex that CAPTURES unit in group (1)
7: ingredient_quantity – regex that CAPTURES numeric quantity in group (1)
8: yield_quantity   – regex that CAPTURES numeric yield in group (1)
9: row_exclusion    – regex to EXCLUDE headers or noise rows

CAPTURE RULES:
- If a value must be extracted, it MUST be in CAPTURE GROUP (1).
- Do NOT rely on group(0) unless unavoidable.

EXPECTED STRUCTURED RESULT (GROUND TRUTH):
{phase1_output}

SOURCE TEXT:
{extracted_text}

OUTPUT FORMAT:
- Return ONLY a JSON array of 10 strings.
- No explanation. No markdown. No comments.
- Empty string "" is allowed where a pattern is not needed.

FINAL CHECK:
- The regex you generate MUST be able to produce the ground truth exactly.
- The solution MUST work for BOTH menu-item blocks AND standalone rows that can be interpreted as sides or extras.
""".strip()



def extract_buildsheet(restaurant_id: str, file_path: str) -> pd.DataFrame:
    """Main entry point to extract a normalized buildsheet for a restaurant.

    Steps:
      - Read CSV/Excel (first sheet for Excel)
      - Convert DataFrame -> normalized text (CELL_SEP/ROW_SEP)
      - Attempt regex extraction using stored template for restaurant
      - If no template: call LLM Phase1 (extract structured items) and Phase2 (generate regex), save template, then re-run
      - Enrich and return DataFrame in database-ready form
    """
    # 1. Read file
    ext = os.path.splitext(file_path)[1].lower()
    if ext in {".xls", ".xlsx"}:
        df = pd.read_excel(file_path, sheet_name=0)
    else:
        df = pd.read_csv(file_path)

    extracted_df = df.copy()

    # print(f"[INFO] extracted_df:\n {extracted_df}\n")
    # 2. Convert to text
    extracted_text = _df_to_text(extracted_df)
    # print(f"\n[INFO] extracted_text:\n {extracted_text}")

    # 3. Try regex path
    regex_list = get_buildsheet_regex_patterns(str(restaurant_id)) or []

    if regex_list:
        print("\n[INFO] Restaurant regex found. Extracting data using regex.")
        buildsheet_df = _apply_regex_patterns_to_text(extracted_text, regex_list)
    else:
        # LLM path
        print("[INFO] Restaurant regex not found. Generating regex using LLM.")

        print("[INFO] Calling Phase 1 prompt.")
        prompt1 = _make_buildsheet_phase1_prompt(extracted_text)
        print(f"\n[INFO] Phase 1 prompt: \n{prompt1}\n")
        phase1_output = call_llm_api(prompt1)
        print(f"\n[INFO] Phase 1 prompt output:\n {phase1_output}\n")
        print(f"\n[INFO] Phase 1 prompt output:\n {phase1_output}\n")

        # Expect parsed to be a dict containing 'buildsheet_items'
        if not isinstance(phase1_output, dict) or "buildsheet_items" not in phase1_output:
            raise ValueError("Phase1 LLM output missing 'buildsheet_items'")


        # Phase2: ask for regex
        print("[INFO] Calling Phase 2 prompt.")
        prompt2 = _make_buildsheet_phase2_prompt(extracted_text, phase1_output)
        print(f"\n[INFO] Phase 2 prompt: \n{prompt2}\n")
        phase2_output = call_llm_api(prompt2)
        print(f"\n[INFO] Phase 2 prompt output\n: {phase2_output}\n")

        # parsed2 should be either list or dict (if dict, try to extract ordered list)
        if isinstance(phase2_output, dict) and "regex_patterns" in phase2_output:
            regex_list = phase2_output["regex_patterns"]
        elif isinstance(phase2_output, list):
            regex_list = phase2_output
        else:
            raise ValueError("Phase2 LLM output invalid: expected list or {regex_patterns: [...]}")

        # Save template (ensure we pass list of strings)
        save_buildsheet_regex_template(str(restaurant_id), list(regex_list))

        # Re-run extraction using generated regex
        buildsheet_df = _apply_regex_patterns_to_text(extracted_text, regex_list)
    # 4. Enrich
    final_df = enrich_buildsheet_df(restaurant_id, buildsheet_df)
    return final_df


def enrich_buildsheet_df(restaurant_id: str, buildsheet_df: pd.DataFrame) -> pd.DataFrame:
    """Add DB-ready fields:
      - restaurant_id
      - estimated_price (null)
      - is_raw_material (False on ingredients)

    Returns updated DataFrame.
    """
    if buildsheet_df is None:
        return pd.DataFrame(columns=["restaurant_id", "name", "yield_quantity", "estimated_price", "ingredients"])

    df = buildsheet_df.copy()
    df["restaurant_id"] = str(restaurant_id)
    df["estimated_price"] = None

    # Ensure ingredients list objects have is_raw_material flag
    def normalize_ings(ings):
        out = []
        if not ings:
            return out
        for i in ings:
            item = {
                "material_name": i.get("material_name"),
                "measure_unit": i.get("measure_unit"),
                "measure_quantity": i.get("measure_quantity"),
                "is_raw_material": False,
            }
            out.append(item)
        return out

    df["ingredients"] = df["ingredients"].apply(normalize_ings)

    # Final column order
    df = df[["restaurant_id", "name", "yield_quantity", "estimated_price", "ingredients"]]
    return df