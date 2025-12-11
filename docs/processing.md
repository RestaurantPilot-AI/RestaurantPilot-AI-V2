# Invoice Automation: Processing Module

This module handles vendor identification and prepares raw OCR text to be shaped into structured invoice data that can be parsed, validated, and stored.

## Module Components

The `src/processing` directory contains:

- **`build_dataframe.py`**: Stub for converting raw text into two pandas DataFrames (invoice-level and line items). 
- **`vendor_identifier.py`**: Implements vendor detection and regex management with in-memory storage.
  - `identifyVendor(text)`: Extracts a plausible vendor name using labels, business suffixes, headers, or email hints.
  - `searchVendorName(name)`: Looks up the normalized vendor name in the in-memory vendor store.
  - `identify_vendor_and_get_regex(text)`: Orchestrates vendor lookup or creation, returns vendor metadata plus a regex (generated if missing).
  - `generateRegexWithLLM(text)`: Placeholder that simulates an LLM call to produce a regex when no vendor match exists, then saves a new vendor.
  - `extractVendorDetails(text)` / `save_vendor_details(...)`: Pulls basic contact details and attaches them to the vendor record in memory.

## Current Behavior

1. Accept raw invoice text.
2. Detect a likely vendor name (`identifyVendor`).
3. If the vendor exists in memory, return its saved regex; otherwise, simulate an LLM call to generate a regex and create the vendor record.
4. Attach basic contact details to the in-memory vendor record.
