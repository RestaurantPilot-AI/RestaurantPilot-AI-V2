import os
import re
from typing import List, Dict, Any
import pandas as pd


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

    # 3. LLM-only extraction flow 
    print("[INFO] Extracting buildsheet using LLM (Phase 1).")
    prompt1 = _make_buildsheet_phase1_prompt(extracted_text)
    # print(f"\n[INFO] Phase 1 prompt: \n{prompt1}\n")
    phase1_output = call_llm_api(prompt1)
    # print(f"\n[INFO] Phase 1 prompt output:\n {phase1_output}\n")

    # Expect parsed to be a dict containing 'buildsheet_items'
    if not isinstance(phase1_output, dict) or "buildsheet_items" not in phase1_output:
        raise ValueError("Phase1 LLM output missing 'buildsheet_items'")

    buildsheet_items = phase1_output["buildsheet_items"]
    if not isinstance(buildsheet_items, list):
        raise ValueError("Phase1 LLM output 'buildsheet_items' must be a list")

    # Convert parsed items directly into a DataFrame for enrichment/saving
    buildsheet_df = pd.DataFrame(buildsheet_items)
    # Ensure required columns exist and are ordered correctly
    for _c in ["name", "yield_quantity", "ingredients"]:
        if _c not in buildsheet_df.columns:
            buildsheet_df[_c] = None
    buildsheet_df = buildsheet_df[["name", "yield_quantity", "ingredients"]]
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