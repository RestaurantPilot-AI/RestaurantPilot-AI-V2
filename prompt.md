## Task: Buildsheet Ingestion Pipeline

Create the implementation for **`buildsheet_test.py`** and update the following files:

* `src/processing/buildsheet_ingestion.py`
* `src/storage/database.py`

---

## Part 1: Core Entry Method

### Method

`extract_buildsheet(restaurant_id, file_path)`

### Responsibilities

1. **Input Handling**

   * Accepts:

     * `restaurant_id`
     * `file_path` (CSV or Excel)
   * If **Excel**:

     * Read **only the first sheet**
   * If **CSV**:

     * Read the full file

2. **Data Extraction**

   * Load the file into a **pandas DataFrame** named:

     * `extracted_df`

3. **DataFrame → Text Conversion**

   * Convert `extracted_df` into a **normalized text format** called:

     * `extracted_text`
   * Use **explicit separators**:

     * One separator to mark **cell boundaries**
     * One separator to mark **row boundaries**
   * These separators must be documented and consistent, as they will be relied on by regex and LLMs.

---

## Part 2: Regex-Based Extraction (Primary Path)

### Regex Lookup

Using `restaurant_id`, check for an existing template in:

### Collection: `buildsheet_regex_templates`

**Schema**

```
restaurant_id: ObjectId
regex_patterns: Array[String] (strict positional indexing)
```

### Regex Index Mapping

| Index | Field Name          | Context                     |
| ----: | ------------------- | --------------------------- |
|     0 | item_start          | Start of a new menu item    |
|     1 | item_end            | End of a menu item          |
|     2 | menu_item_name      | Menu item name              |
|     3 | ingredient_start    | Ingredient block start      |
|     4 | ingredient_end      | Ingredient block end        |
|     5 | ingredient_name     | Ingredient name             |
|     6 | ingredient_unit     | Measurement unit            |
|     7 | ingredient_quantity | Measurement quantity        |
|     8 | yield_quantity      | Menu item yield             |
|     9 | row_exclusion       | Ignore headers / noise rows |

Empty strings **must preserve index order**.

---

### Regex Application

If a template exists:

1. Fetch the `regex_patterns`
2. Apply them to `extracted_text`
3. Extract structured data into a DataFrame named:

   * `buildsheet_df`

---

## Part 3: Expected `buildsheet_df` Structure (Phase 1 Output)

### Columns

| Column Name    | Type   | Required |
| -------------- | ------ | -------- |
| name           | String | Yes      |
| yield_quantity | Double | Yes      |
| ingredients    | Array  | Yes      |

### Ingredient Object Structure (Phase 1)

| Field Name       | Type   | Required |
| ---------------- | ------ | -------- |
| material_name    | String | Yes      |
| measure_unit     | String | Yes      |
| measure_quantity | Double | Yes      |

---

## Part 4: Enrichment to Database-Ready Format

Create a **second method**:

### Method

`enrich_buildsheet_df(restaurant_id, buildsheet_df)`

### Responsibilities

* Add the following fields:

  * `restaurant_id`
  * `estimated_price` (optional; may be null)
  * `is_raw_material` → **always False for now**
* Convert `material_name` to an ObjectId reference later (leave as-is for now)
* Return the **updated DataFrame**

---

### Final Database Model Target

Each row represents **one build sheet per menu item**.

| Field Name      | Type     | Required |
| --------------- | -------- | -------- |
| restaurant_id   | ObjectId | Yes      |
| name            | String   | Yes      |
| yield_quantity  | Double   | Yes      |
| estimated_price | Double   | No       |
| ingredients     | Array    | Yes      |

#### Ingredient Object (Final)

| Field Name       | Type     | Required    |
| ---------------- | -------- | ----------- |
| material_name    | ObjectId | Yes         |
| measure_unit     | String   | Yes         |
| measure_quantity | Double   | Yes         |
| is_raw_material  | Boolean  | Yes (False) |

Return this final DataFrame.

---

## Part 5: LLM Fallback Path (When Regex Is Missing)

If **no regex template exists** for the restaurant:

### Phase 1 – LLM Extraction

1. Create a method to:

   * Build a **fixed prompt**
   * Include:

     * Schema of the expected output (database-ready structure)
     * `extracted_text`
   * Clearly explain:

     * How the DataFrame was converted to text
     * What separators mean
2. Call **Gemini (text-only)**
3. Receive output as:

   * **Text-based DataFrame representation** (LLMs cannot return actual DataFrames)

---

### Phase 2 – Regex Generation

Create another method to build a **Phase 2 prompt** that includes:

* Regex schema (index-based)
* `extracted_text`
* Phase 1 LLM output
* Clear explanation:

  * Phase 1 output = expected structured result
  * `extracted_text` = raw source
  * Goal = generate regex that reproduces Phase 1 output from the text

The LLM must return:

* A regex array matching the `buildsheet_regex_templates` schema

---

### Phase 1 Prompt — Data Model (for buildsheet_df)

When building Phase 1 prompts, the LLM should be asked to return a top-level JSON object with key `buildsheet_items` containing an array of objects. The exact schema expected for each item is:

- name: string (required) — menu item name
- yield_quantity: number | null (required in output; may be null)
- ingredients: array of ingredient objects (required; can be empty)

Ingredient object schema:

- material_name: string (required)
- measure_unit: string | null
- measure_quantity: number | null

Example Phase 1 JSON output:

```
{
  "buildsheet_items": [
    {
      "name": "Pizza",
      "yield_quantity": 4,
      "ingredients": [
        {"material_name": "Cheese", "measure_unit": "g", "measure_quantity": 200},
        {"material_name": "Tomato", "measure_unit": "g", "measure_quantity": 100}
      ]
    }
  ]
}
```

The prompt must explicitly describe how the DataFrame was converted to text and include the separators tokens (e.g., `CELL_SEP = "<CELL_SEP>"` and `ROW_SEP = "<ROW_SEP>"`) with an example row to avoid ambiguity.

---

### Phase 2 Prompt — Regex Schema (index-based)

Phase 2 must instruct the LLM to return a JSON array of exactly 10 strings corresponding to the indices below. If an index is not needed, return an empty string at that position — do not change positions.

Index map (0..9):

- 0: item_start — regex to find the beginning of an item block
- 1: item_end — regex to find the end of an item block
- 2: menu_item_name — regex capturing the menu item name (use group(1) to capture)
- 3: ingredient_start — regex marking ingredient block start
- 4: ingredient_end — regex marking ingredient block end
- 5: ingredient_name — regex to find ingredient name(s)
- 6: ingredient_unit — regex capturing unit (group(1) if applicable)
- 7: ingredient_quantity — regex capturing quantity (group(1))
- 8: yield_quantity — regex capturing yield value (group(1))
- 9: row_exclusion — regex to ignore header/noise rows

Example Phase 2 return (JSON array):

```
["", "", "^(.*?)<CELL_SEP>", "", "", "", "", "", "", "^name"]
```

Notes:
- Explain that the extraction pipeline will use group(1) where capture groups are present.
- Ask the LLM to prefer anchored, simple regexes and to preserve empty positions explicitly.

---

### Persistence & Rejoin Flow

1. Save the generated regex to:

   * `buildsheet_regex_templates`
2. Immediately re-run extraction:

   * Treat it as if the regex always existed
3. Return the final enriched DataFrame

---

## Notes

* Reuse or extend existing CRUD methods or make new methods in:

  * `src/storage/database.py` Just make sure to have good documenting update in existing method if they are not good right now
* For LLM usage patterns and 2-phase prompting reference:

  * `src/processing/vendor_identifier.py`
  * `src/processing/menu_ingestion.py`

