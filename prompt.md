Update the following files to add the table:

## 12. Collection: cookbook_items

**Description:** Stores cookbook definitions for intermediate or prepared ingredients produced by a restaurant. Each document represents one cookbook item that may be used as an ingredient in build sheets or other cookbooks. Cookbook items may be composed of raw materials or other cookbook items.

| Field Name      | Data Type | Required | Description                                                      |
| --------------- | --------- | -------- | ---------------------------------------------------------------- |
| _id             | ObjectId  | Yes      | Primary key. System-generated.                                   |
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

The changes you need to make are:-
src\storage\db_init.py update this to add this file to add another table that is checked at start and makes it if absent: cookbook_items.csv
src\storage\database.py update this file to make crud operation for the new table cookbook_items