# Database Overview

## Database Name: invoice_processing_db

---

## 1. Collection: restaurants

**Description:** Stores physical location data for the business entities receiving the goods.

| Field Name      | Data Type | Required | Description                                                                        |
| --------------- | --------- | -------- | ---------------------------------------------------------------------------------- |
| _id             | ObjectId  | Yes      | Primary key.                                                                       |
| name            | String    | Yes      | Name of the restaurant.                                                            |
| location_name   | String    | Yes      | Restaurant location.                                                               |
| phone_number    | String    | No       | Restaurant phone number.                                                           |
| restaurant_type | String    | No       | Restaurant type.                                                                   |
| address         | String    | No       | Full physical address.                                                             |
| created_at      | Date      | Yes      | Timestamp of creation.                                                             |
| is_active       | Boolean   | Yes      | Default true. Soft-delete or disable a location without breaking historical links. |

---

## 2. Collection: vendors

**Description:** The single source of truth for vendor identities.

| Field Name    | Data Type | Required | Description                               |
| ------------- | --------- | -------- | ----------------------------------------- |
| _id           | ObjectId  | Yes      | Primary key.                              |
| name          | String    | Yes      | Normalized name (e.g., "Sysco") (Unique). |
| contact_email | String    | No       | Vendor contact email (Unique).            |
| phone_number  | String    | No       | Vendor phone number (Unique).             |
| address       | String    | No       | Vendor address (Unique).                  |
| website       | String    | No       | Vendor website (Unique).                  |

---

## 3. Collection: vendor_regex_templates

**Description:** Stores regex patterns used to identify vendors from raw OCR text. Linked to vendors.

| Field Name     | Data Type       | Required | Description                          |
| -------------- | --------------- | -------- | ------------------------------------ |
| vendor_id      | ObjectId        | Yes      | Foreign key referencing vendors._id. |
| regex_patterns | Array of String | Yes      | List of raw regex strings.           |

### Regex Patterns: Index Mapping

The `regex_patterns` array uses strict positional indexing.
Empty strings (`""`) indicate missing values but **preserve array order**.

| Index | Field Name                | Context                        |
| ----- | ------------------------- | ------------------------------ |
| 0     | `invoice_number`        | Invoice Level                  |
| 1     | `invoice_date`          | Invoice Level                  |
| 2     | `invoice_total_amount`  | Invoice Level                  |
| 3     | `order_date`            | Invoice Level                  |
| 4     | `line_item_block_start` | Line Item Level (Start Marker) |
| 5     | `line_item_block_end`   | Line Item Level (End Marker)   |
| 6     | `quantity`              | Line Item Level                |
| 7     | `description`           | Line Item Level                |
| 8     | `unit`                  | Line Item Level                |
| 9     | `unit_price`            | Line Item Level                |
| 10    | `line_total`            | Line Item Level                |

---

## 4. Collection: invoices

**Description:** Stores the data and high-level summary information for each unique invoice document.

| Field Name           | Data Type  | Required | Description                                      |
| -------------------- | ---------- | -------- | ------------------------------------------------ |
| _id                  | ObjectId   | Yes      | Primary key.                                     |
| filename             | String     | Yes      | Original filename (e.g., scan_001.pdf).          |
| restaurant_id        | ObjectId   | Yes      | Foreign key referencing restaurants._id.         |
| vendor_id            | ObjectId   | Yes      | Foreign key referencing vendors._id.             |
| invoice_number       | String     | Yes      | The invoice ID captured from the document.       |
| invoice_date         | Date       | Yes      | Invoice date (ISO 8601).                         |
| invoice_total_amount | Decimal128 | Yes      | Total cost. Decimal128 prevents rounding errors. |
| text_length          | Int32      | No       | Metadata for validation.                         |
| page_count           | Int32      | No       | Metadata for validation.                         |
| extraction_timestamp | Date       | Yes      | Timestamp when OCR extraction ran.               |
| order_date           | Date       | Yes      | Ordered date (ISO 8601).                         |

---

## 5. Collection: line_items

**Description:** Stores individual transactional items associated with a specific invoice.

| Field Name  | Data Type  | Required | Description                                        |
| ----------- | ---------- | -------- | -------------------------------------------------- |
| _id         | ObjectId   | Yes      | Primary key.                                       |
| invoice_id  | Int32      | Yes      | Foreign Key. Reference to invoices._id.            |
| vendor_name | String     | Yes      | Redundant but useful for fast filtering by vendor. |
| category    | String     | Yes      | Foreign Key. Reference to categories._id.          |
| quantity    | Double     | Yes      | Quantity purchased.                                |
| unit        | String     | Yes      | Unit of measurement (lb, oz, bottle, etc.)         |
| description | String     | Yes      | Item description.                                  |
| unit_price  | Decimal128 | Yes      | Price per unit.                                    |
| line_total  | Decimal128 | Yes      | Total cost for the line (quantity * unit price).   |
| line_number | Decimal128 | Yes      | Line number on invoice.                            |

---

## 6. Collection: item_lookup_map

**Description:** Dictionary that maps a description to a standardized Category ID.

| Field Name | Data Type | Required | Description                                                                     |
| ---------- | --------- | -------- | ------------------------------------------------------------------------------- |
| _id        | String    | Yes      | Primary Key. Normalized description. Acts as the unique description identifier. |
| category   | String    | Yes      | Category Name.                                                                  |

---

## 7. Collection: categories

**Description:** Master list for UI dropdowns and validation.

| Field Name | Data Type | Required | Description                                                                                      |
| ---------- | --------- | -------- | ------------------------------------------------------------------------------------------------ |
| _id        | String    | Yes      | Primary Key. Unique Category Name (e.g. "Dairy", "Fruits"). For human-readable and fast lookups. |

---

## 8. Collection: sales

**Description:** Daily Sales Tracking.

| Field Name     | Data Type | Required | Description                               |
|----------------|-----------|----------|-------------------------------------------|
| _id            | ObjectId  | Yes      | Primary key.                              |
| date           | Date      | Yes      | The date of the sales record.             |
| restaurant_id  | ObjectId  | Yes      | Reference to the restaurant.              |
| revenue        | Double    | Yes      | Total revenue amount.                     |
| covers         | Int       | Yes      | Number of covers served.                  |
| created_at     | Date      | Yes      | Timestamp of when the record was created. |
