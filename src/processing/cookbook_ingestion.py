import pandas as pd
from typing import List, Dict, Any

from src.extraction.invoice_extractor import process_text
from src.processing.vendor_identifier import call_llm_api


def _make_cookbook_prompt(extracted_text: str) -> str:
    """Builds a deterministic prompt for the LLM to extract cookbook items in JSON.

    The prompt requests a JSON-only response with a top-level key `cookbook_items` that
    is a list of cookbook item objects. Each cookbook item MUST include:
      - cookbook_item_name (string)
      - yield (string or null)
      - raw_materials (array of objects with material_name, measure_unit, measure_quantity)

    The prompt instructs the model to use only data present in the source and to return
    an empty list when no cookbook items can be extracted.
    """
    return f"""
You are a deterministic extraction assistant.

INPUT:
- You are given raw extracted text from a cookbook or recipe-like document.
- Use ONLY information present in the SOURCE. Do NOT invent ingredients, quantities, or yields.

OUTPUT FORMAT (JSON ONLY):
{{
  "cookbook_items": [
    {{
      "cookbook_item_name": "string",
      "yield": "string | null",
      "raw_materials": [
        {{
          "material_name": "string",
          "measure_unit": "string | null",
          "measure_quantity": null | number
        }}
      ]
    }}
  ]
}}

SOURCE:
{extracted_text}
""".strip()


def extract_cookbook(restaurant_id: str, file_path: str) -> pd.DataFrame:
    """Extract cookbook items from `file_path` and return an enriched DataFrame ready for DB save.

    Steps:
      - Extract raw text via `process_text`
      - Build and call the LLM prompt
      - Validate and convert LLM JSON output to a DataFrame
      - Enrich with `enrich_cookbook_df`
    """
    # 1. Extract text
    extracted_text = process_text(file_path)

    if not extracted_text:
        # Return an empty enriched DataFrame
        return enrich_cookbook_df(restaurant_id, pd.DataFrame())

    # 2. Build prompt & call LLM
    prompt = _make_cookbook_prompt(extracted_text)
    llm_out = call_llm_api(prompt)

    if not isinstance(llm_out, dict) or "cookbook_items" not in llm_out:
        raise ValueError("LLM output missing 'cookbook_items'")

    items = llm_out["cookbook_items"]
    if not isinstance(items, list):
        raise ValueError("LLM 'cookbook_items' must be a list")

    # 3. Convert to DataFrame using canonical column names
    df = pd.DataFrame(items)

    # Accept both 'cookbook_item_name' and 'name' by normalizing
    if "cookbook_item_name" in df.columns and "name" not in df.columns:
        df = df.rename(columns={"cookbook_item_name": "name"})

    for _c in ["name", "yield", "raw_materials"]:
        if _c not in df.columns:
            df[_c] = None

    df = df[["name", "yield", "raw_materials"]]

    # 4. Enrich & return
    final_df = enrich_cookbook_df(restaurant_id, df)
    return final_df


def enrich_cookbook_df(restaurant_id: str, cookbook_df: pd.DataFrame) -> pd.DataFrame:
    """Add DB-ready fields and normalize `raw_materials` objects.

    Final structure:
      - restaurant_id
      - name
      - yield
      - estimated_price (None)
      - raw_materials (list of dicts with material_name, measure_unit, measure_quantity, is_raw_material)
    """
    if cookbook_df is None or cookbook_df.empty:
        return pd.DataFrame(columns=["restaurant_id", "name", "yield", "estimated_price", "raw_materials"])

    df = cookbook_df.copy()
    df["restaurant_id"] = str(restaurant_id)
    df["estimated_price"] = None

    def normalize_rms(rms):
        out = []
        if not rms:
            return out
        # If the LLM returned a string, try to coerce via json.loads; if it is already a list, iterate
        if isinstance(rms, str):
            try:
                import json
                rms = json.loads(rms)
            except Exception:
                return out

        for r in rms:
            try:
                material_name = r.get("material_name") if isinstance(r, dict) else None
                measure_unit = r.get("measure_unit") if isinstance(r, dict) else None
                mq = r.get("measure_quantity") if isinstance(r, dict) else None
                # Try to coerce quantity to float when possible
                try:
                    if mq is None or mq == "":
                        mq_val = None
                    else:
                        mq_val = float(mq)
                except Exception:
                    mq_val = None

                out.append({
                    "material_name": material_name,
                    "measure_unit": measure_unit,
                    "measure_quantity": mq_val,
                    "is_raw_material": False,
                })
            except Exception:
                # ignore malformed entry
                continue
        return out

    df["raw_materials"] = df["raw_materials"].apply(normalize_rms)

    df = df[["restaurant_id", "name", "yield", "estimated_price", "raw_materials"]]
    return df