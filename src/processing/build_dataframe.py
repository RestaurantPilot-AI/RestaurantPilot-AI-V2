"""
Data frame builder for invoice processing.

This module converts extracted invoice data into pandas DataFrames
for storage and further processing.
"""

from typing import Tuple, Dict, Any, Optional, List
import pandas as pd
import re

from .vendor_identifier import identify_vendor_and_get_regex, apply_regex_extraction
from .categorization import get_line_item_category


class MultipleInvoiceNumberWarning(UserWarning):
    """Raised when multiple invoice numbers are detected in extraction."""

def get_structured_data_from_text(
    extracted_text: str, 
    filename: str, 
    text_length: int, 
    page_count: int, 
    extraction_timestamp: Any,
    restaurant_id: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Accept raw invoice text and metadata to return structured DataFrames.
    
    Args:
        extracted_text (str): The full text content of the invoice.
        filename (str): The name of the file (e.g., "invoice.pdf").
        text_length (int): The character count of the text.
        page_count (int): The number of pages in the document.
        extraction_timestamp (Any): Time when extraction occurred (str or datetime).
        restaurant_id (str): Identifier for the restaurant associated with this invoice.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (invoice_df, line_items_df)
    """

    if not extracted_text.strip():
        raise ValueError("Input text cannot be empty")

    try:
        vendor_context = identify_vendor_and_get_regex(extracted_text)

    except Exception as exc: 
        raise ValueError(f"Failed to identify or create vendor: {exc}")
    
    try:
        extracted_inv_data, extracted_li_data = apply_regex_extraction(
            text = extracted_text, 
            regex_patterns = vendor_context.get("regex")
        )
    except Exception as exc:  
        raise ValueError(f"Failed to apply regex: {exc}")
    
    # Build the Invoice DataFrame matching the 'invoices' collection schema
    inv_df = _build_invoice_record(
        extracted_inv_data=extracted_inv_data,
        vendor_context=vendor_context,
        filename=filename,
        restaurant_id=restaurant_id,
        text_length=text_length,
        page_count=page_count,
        extraction_timestamp=extraction_timestamp
    )

    # Build the Line Items DataFrame matching the 'line_items' collection schema
    # We retrieve the invoice number here to link items to the invoice record
    # (Note: The actual database linking via _id will happen in storage part)    
    li_df = _build_line_items_records(
        extracted_li_data=extracted_li_data,
        vendor_context=vendor_context,
    )

    return inv_df, li_df

def _build_invoice_record(
    extracted_inv_data: Dict[str, str],
    vendor_context: Dict[str, Any],
    filename: str,
    restaurant_id: str,
    text_length: int,
    page_count: int,
    extraction_timestamp: Any
) -> pd.DataFrame:
    """
    Build a single invoice record from normalized extraction payload and return it
    as a single-row pandas DataFrame.
    """
    
    # We extract the fields from the dictionary returned by apply_regex_extraction
    invoice_number = extracted_inv_data.get("invoice_number")
    invoice_date = extracted_inv_data.get("invoice_date")
    invoice_total_amount = extracted_inv_data.get("invoice_total_amount")
    order_date = extracted_inv_data.get("order_date")

    # Construct the record dictionary to match the 'invoices' collection Schema EXACTLY
    record = {
        "filename": filename,
        "restaurant_id": restaurant_id,
        "vendor_id": vendor_context.get("vendor_id"),
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "invoice_total_amount": invoice_total_amount,
        "text_length": text_length,
        "page_count": page_count,
        "extraction_timestamp": extraction_timestamp,
        "order_date": order_date
    }

    # Return as a pandas DataFrame
    return pd.DataFrame([record])


def _clean_quantity(val: Any) -> float:
    """
    Converts a string quantity to a float.
    Handles commas and basic formatting issues.
    """
    if val is None:
        return 0.0
    try:
        return float(str(val).replace(',', '').strip())
    except Exception:
        return 0.0


def _parse_currency_amount(amount_str: Optional[str]) -> Optional[float]:
    """
    Parse currency string to float.
    
    Examples:
        €1,234.56 -> 1234.56
        $1.234,56 -> 1234.56
        1,234.56 -> 1234.56
    
    Args:
        amount_str: String representation of currency amount
        
    Returns:
        Float value or None if parsing fails
    """
    if not amount_str:
        return None
    
    # Remove currency symbols and whitespace
    cleaned = re.sub(r'[€$£¥\s]', '', str(amount_str))
    
    # Handle both comma and period as decimal separator
    # If there are multiple commas or periods, assume the last one is decimal
    if ',' in cleaned and '.' in cleaned:
        # Determine which is decimal separator (the last one usually)
        if cleaned.rfind(',') > cleaned.rfind('.'):
            # Comma is decimal separator
            cleaned = cleaned.replace('.', '').replace(',', '.')
        else:
            # Period is decimal separator
            cleaned = cleaned.replace(',', '')
    elif ',' in cleaned:
        # Only comma - could be thousands or decimal
        # If there are 2 digits after the last comma, it's decimal
        parts = cleaned.rsplit(',', 1)
        if len(parts) == 2 and len(parts[1]) == 2:
            cleaned = cleaned.replace(',', '.')
        else:
            cleaned = cleaned.replace(',', '')
    
    try:
        return float(cleaned)
    except (ValueError, AttributeError):
        return None

def _determine_category(description: str) -> Optional[str]:
    """
    Routes the category determination logic to the external processing module.
    """
    # This calls the actual logic which will be implemented in src/processing.py
    return get_line_item_category(description)

def _build_line_items_records(
    extracted_li_data: List[Dict[str, Any]],
    vendor_context: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build line item records from normalized extraction payload.
    
    Parameters:
        extracted_li_data (List[Dict]): The list of raw line item dictionaries from regex.
        vendor_context (Dict): Vendor metadata.
        
    Returns:
        pd.DataFrame: DataFrame matching the 'line_items' collection schema.
    """
    # Vendor Name is redundant in line_items but required by the schema
    vendor_name = vendor_context.get("vendor_name", "Unknown")
    
    line_items = []
    
    for idx, item in enumerate(extracted_li_data):
        if not isinstance(item, dict):
            continue
        
        # Parse and clean fields
        description = item.get("description", "").strip()
        quantity = _clean_quantity(item.get("quantity"))
        unit_val = item.get("unit")
        unit_price = _parse_currency_amount(item.get("unit_price"))
        line_total = _parse_currency_amount(item.get("line_total"))
        
        # Only include items that have at least a description
        if description:
            # Call the new method to determine category
            category = _determine_category(description)

            line_item = {
                # --- Schema Fields Only ---
                "invoice_id": None, # Placeholder: To be updated with real ObjectId/Int32 later
                "vendor_name": vendor_name,
                "category": category,   # populated by the method
                "quantity": quantity if quantity is not None else 0.0,
                "unit": unit_val,
                "description": description,
                "unit_price": unit_price,
                "line_total": line_total,
                "line_number": idx + 1 # 1-based index
            }
            
            line_items.append(line_item)
    
    # If no items were found, return an empty DataFrame with the correct schema columns
    if not line_items:
        return pd.DataFrame(columns=[
            "invoice_id", "vendor_name", "category", "quantity", 
            "unit", "description", "unit_price", "line_total", "line_number"
        ])
    
    return pd.DataFrame(line_items)
