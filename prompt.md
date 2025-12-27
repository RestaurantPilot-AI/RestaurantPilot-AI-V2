We have to make the cookbook_test.py work for that we need to make the def extract_cookbook(restaurant_id, file_path): method in src\processing\cookbook_ingestion.py

Now lets discuss how will we make this method.
For every file it should calldef process_text(file_path: str) -> str from from src.extraction, it returns extracted textnor pass that extracted text to an llm for that make an method for writting the promp thismethot will explain tehe task and takes in etractes text. to gereate teh protm and returns it.
The task is that it will get extracted text from a cook book it has to read it and has to then convert it info belwo json format:-


| Field Name      | Data Type | Required | Description                                                      |
| --------------- | --------- | -------- | ---------------------------------------------------------------- |
| cookbook_item_name            | String    | Yes      | Name of the ingredient being produced (e.g., "Cream Cheese").    |
| yield           | String    | Yes      | Yield output expressed as a string (e.g., "3 lbs", "2 batches"). |
| raw_materials   | Array     | Yes      | Array of raw material definition objects (see structure below).  |

---

### raw_materials (Array of JSON Objects)

Each element in the `raw_materials` array represents **one input** used to produce the cookbook item.
Inputs may reference either a raw material or another cookbook item, determined by a boolean flag.

#### Raw Material Object Structure

| Field Name       | Data Type | Required | Description                                                                   |
| ---------------- | --------- | -------- | ----------------------------------------------------------------------------- |
| material_name      | String  | Yes      | Name of material. |
| measure_unit     | String    | Yes      | Unit of measurement (e.g., oz, lb, cup).                                      |
| measure_quantity | Double    | Yes      | Quantity required for this input.                                             |

once this prompt has been made pass it tp llm (see src\processing\buildsheet_ingestion.py to know hot to do it use from src.processing.vendor_identifier import call_llm_api) get the outut convert itto df make amethod def enrich_cookbook_df(restaurant_id: str, buildsheet_df: pd.DataFrame) -> pd.DataFrame to improve and add columns so it looks like this and reutrn it

## 12. Collection: cookbook_items

**Description:** Stores cookbook definitions for intermediate or prepared ingredients produced by a restaurant. Each document represents one cookbook item that may be used as an ingredient in build sheets or other cookbooks. Cookbook items may be composed of raw materials or other cookbook items.

| Field Name      | Data Type | Required | Description                                                      |
| --------------- | --------- | -------- | ---------------------------------------------------------------- |
| restaurant_id   | ObjectId  | Yes      | Foreign key referencing restaurants._id.                         |
| name            | String    | Yes      | Name of the ingredient being produced (e.g., "Cream Cheese").    |
| yield           | String    | Yes      | Yield output expressed as a string (e.g., "3 lbs", "2 batches"). |
| estimated_price | Double    | No       | Estimated cost calculated from raw material or cookbook inputs.  |
| raw_materials   | Array     | Yes      | Array of raw material definition objects (see structure below).  |

---

### raw_materials (Array of JSON Objects)

Each element in the `raw_materials` array represents **one input** used to produce the cookbook item.
Inputs may reference either a raw material or another cookbook item, determined by a boolean flag.

#### Raw Material Object Structure

| Field Name       | Data Type | Required | Description                                                                   |
| ---------------- | --------- | -------- | ----------------------------------------------------------------------------- |
| material_name      | String  | Yes      | Name of material. |
| measure_unit     | String    | Yes      | Unit of measurement (e.g., oz, lb, cup).                                      |
| measure_quantity | Double    | Yes      | Quantity required for this input.                                             |
| is_raw_material  | Boolean   | Yes      | False for now  |

If you are not sure on how to do something look to src\processing\buildsheet_ingestion.py it deos a lot of same thng jst for build sheet instead of cookbook