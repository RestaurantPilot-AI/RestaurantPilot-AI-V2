import os
import re
import datetime
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from bson import ObjectId
from bson.decimal128 import Decimal128
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup MongoDB Connection
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# ---------------------------------------------------------
# Collection Names
# ---------------------------------------------------------
COL_VENDORS = "vendors"
COL_VENDOR_REGEXES = "vendor_regex_templates"

# ---------------------------------------------------------
# Vendor Regex Methods
# ---------------------------------------------------------
def save_vendor_regex_template(new_vendor_id: str, new_regexes: Dict[str, Any]) -> None:
    """
    Parses a nested dictionary of regexes, flattens them into a list,
    and upserts them into the vendor_regex_templates collection.

    The list strictly follows the 0-10 index mapping defined in the schema:
        0: invoice_number        (Invoice Level)
        1: invoice_date          (Invoice Level)
        2: invoice_total_amount  (Invoice Level)
        3: order_date            (Invoice Level)
        4: line_item_block_start (Start Marker)
        5: line_item_block_end   (End Marker)
        6: quantity              (Line Item Level)
        7: description           (Line Item Level)
        8: unit                  (Line Item Level)
        9: unit_price            (Line Item Level)
        10: line_total           (Line Item Level)
    """
    if not new_vendor_id or not new_regexes:
        return

    try:
        vid_object = ObjectId(new_vendor_id)
    except Exception:
        print(f"Error: Invalid vendor_id format: {new_vendor_id}")
        return

    # Extract sub-dictionaries to make the mapping cleaner
    inv_data = new_regexes.get("invoice_level", {})
    li_data = new_regexes.get("line_item_level", {})

    # Construct the list explicitly to guarantee index positions 0 through 10
    flattened_patterns = [
        inv_data.get("invoice_number", ""),       # 0
        inv_data.get("invoice_date", ""),         # 1
        inv_data.get("invoice_total_amount", ""), # 2
        inv_data.get("order_date", ""),           # 3
        li_data.get("line_item_block_start", ""), # 4
        li_data.get("line_item_block_end", ""),   # 5
        li_data.get("quantity", ""),              # 6
        li_data.get("description", ""),           # 7
        li_data.get("unit", ""),                  # 8
        li_data.get("unit_price", ""),            # 9
        li_data.get("line_total", "")             # 10
    ]

    # Update or Insert (Upsert) into Database
    db[COL_VENDOR_REGEXES].update_one(
        {"vendor_id": vid_object},
        {
            "$set": {
                "vendor_id": vid_object,
                "regex_patterns": flattened_patterns
            }
        },
        upsert=True
    )

def get_vendor_regex_patterns(vendor_id: str) -> List[str]:
    """
    Fetches the list of regex patterns for a specific vendor_id.
    """
    if not vendor_id:
        return []

    try:
        vid_object = ObjectId(vendor_id)
    except Exception:
        print(f"Error: Invalid vendor_id format for lookup: {vendor_id}")
        return []

    doc = db[COL_VENDOR_REGEXES].find_one(
        {"vendor_id": vid_object},
        {"regex_patterns": 1, "_id": 0}
    )

    return doc["regex_patterns"] if doc and "regex_patterns" in doc else []

# ---------------------------------------------------------
# Vendor Collection Methods
# ---------------------------------------------------------
def create_vendor(vendor_data: Dict[str, Any]) -> Optional[str]:
    """
    Creates a new vendor document in the collection.
    
    Args:
        vendor_data: Dictionary. Keys are mapped to schema fields internally.
                     Expected keys: vendor_name, vendor_email_id, vendor_phone_number, etc.
    """
    # 1. Validation
    vendor_name = vendor_data.get("vendor_name")
    if not vendor_name:
        return None

    # 2. Map input to New Schema Field Names
    new_vendor = {
        "name": vendor_name,
        "contact_email": vendor_data.get("vendor_email_id"),
        "phone_number": vendor_data.get("vendor_phone_number"),
        "address": vendor_data.get("vendor_physical_address"),
        "website": vendor_data.get("vendor_website"),
    }
    
    # Remove keys with None values to allow sparse indexing/clean docs
    new_vendor = {k: v for k, v in new_vendor.items() if v is not None}
    
    try:
        # 3. Insert
        result = db[COL_VENDORS].insert_one(new_vendor)
        return str(result.inserted_id)
        
    except Exception as e:
        print(f"Error creating vendor: {e}")
        return None

def get_vendor_by_email(email: str) -> Optional[str]:
    if not email:
        return None
    regex = re.compile(rf"^{re.escape(email)}$", re.IGNORECASE)

    # Updated field: contact_email
    doc = db[COL_VENDORS].find_one(
        {"contact_email": regex},
        {"_id": 1}
    )
    return str(doc["_id"]) if doc else None

def get_vendor_by_website(website: str) -> Optional[str]:
    if not website:
        return None
    regex = re.compile(rf"^{re.escape(website)}$", re.IGNORECASE)

    # Field matches schema: website
    doc = db[COL_VENDORS].find_one(
        {"website": regex},
        {"_id": 1}
    )
    return str(doc["_id"]) if doc else None

def get_vendor_by_address(address: str) -> Optional[str]:
    if not address:
        return None
    regex = re.compile(rf"^{re.escape(address)}$", re.IGNORECASE)

    # Updated field: address
    doc = db[COL_VENDORS].find_one(
        {"address": regex},
        {"_id": 1}
    )
    return str(doc["_id"]) if doc else None

def get_vendor_by_phone(phone: str) -> Optional[str]:
    if not phone:
        return None
    regex = re.compile(rf"^{re.escape(phone)}$", re.IGNORECASE)

    # Updated field: phone_number
    doc = db[COL_VENDORS].find_one(
        {"phone_number": regex},
        {"_id": 1}
    )
    return str(doc["_id"]) if doc else None

def get_vendor_by_name(name: str) -> Optional[str]:
    if not name:
        return None
    regex = re.compile(rf"^{re.escape(name)}$", re.IGNORECASE)

    # Updated field: name
    doc = db[COL_VENDORS].find_one(
        {"name": regex},
        {"_id": 1}
    )
    return str(doc["_id"]) if doc else None

def get_vendor_name_by_id(vendor_id: str) -> Optional[str]:
    if not vendor_id:
        return None

    try:
        oid = ObjectId(vendor_id)
    except Exception:
        return None

    doc = db[COL_VENDORS].find_one(
        {"_id": oid},
        {"name": 1}
    )

    return doc.get("name") if doc else None

# ---------------------------------------------------------
# Invoice + Line Item Save Method 
# ---------------------------------------------------------
# ---------------------------------------------------------
# HELPER: Type Conversion
# ---------------------------------------------------------
def to_decimal128(val):
    """Helper to convert generic numbers/strings to MongoDB Decimal128."""
    if pd.isna(val) or val is None:
        return Decimal128("0.00")
    return Decimal128(str(val))

# ---------------------------------------------------------
# MAIN SAVE FUNCTION
# ---------------------------------------------------------
def save_inv_li_to_db(inv_df: pd.DataFrame, li_df: pd.DataFrame):
    """
    Saves the invoice and line items to MongoDB with transactional consistency.
    
    1. Inserts the Invoice record -> Gets the new _id.
    2. Updates the Line Items DataFrame with that new invoice_id.
    3. Bulk Inserts the Line Items.
    """
    if inv_df.empty:
        print("[WARN] No invoice data to save.")
        return

    # NOTE: We use the global 'db' object initialized at the top of the file.

    # 2. Prepare Invoice Record
    # Take the first row (assuming one invoice per DF)
    inv_data = inv_df.iloc[0].to_dict()

    try:
        # Convert pandas/native types to MongoDB BSON types
        invoice_doc = {
            "filename": inv_data.get("filename"),
            "restaurant_id": ObjectId(inv_data.get("restaurant_id")),
            "vendor_id": ObjectId(inv_data.get("vendor_id")),
            "invoice_number": str(inv_data.get("invoice_number")),
            "invoice_date": pd.to_datetime(inv_data.get("invoice_date")),
            "invoice_total_amount": to_decimal128(inv_data.get("invoice_total_amount")),
            "text_length": int(inv_data.get("text_length", 0)),
            "page_count": int(inv_data.get("page_count", 0)),
            "extraction_timestamp": pd.to_datetime(inv_data.get("extraction_timestamp")),
            "order_date": pd.to_datetime(inv_data.get("order_date"))
        }

        # 3. Insert Invoice
        print(f"[INFO] Inserting invoice: {invoice_doc['invoice_number']}...")
        result = db.invoices.insert_one(invoice_doc)
        new_invoice_id = result.inserted_id
        print(f"[SUCCESS] Invoice saved. ID: {new_invoice_id}")

        # 4. Prepare Line Items
        if not li_df.empty:
            # Convert DF to list of dicts for iteration
            li_records = li_df.to_dict("records")
            
            clean_line_items = []
            for item in li_records:
                # Map fields and enforce types
                clean_item = {
                    "invoice_id": new_invoice_id,  # LINKING HAPPENS HERE (ObjectId)
                    "vendor_name": str(item.get("vendor_name", "")),
                    "category": str(item.get("category") or "Uncategorized"),
                    "quantity": float(item.get("quantity", 0.0)),
                    "unit": str(item.get("unit") or ""),
                    "description": str(item.get("description", "")),
                    "unit_price": to_decimal128(item.get("unit_price")),
                    "line_total": to_decimal128(item.get("line_total")),
                    "line_number": to_decimal128(item.get("line_number"))
                }
                clean_line_items.append(clean_item)

            # 5. Bulk Insert Line Items
            if clean_line_items:
                db.line_items.insert_many(clean_line_items)
                print(f"[SUCCESS] Saved {len(clean_line_items)} line items.")
        else:
            print("[INFO] No line items found to save.")

    except Exception as e:
        print(f"[ERROR] Failed to save invoice/line_items: {e}")

# ---------------------------------------------------------
# Invoice Retrieval & Update Methods (CRUD Operations)
# ---------------------------------------------------------

def get_invoice_by_id(invoice_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve an invoice by its ID with its line items.
    
    Args:
        invoice_id: The invoice ObjectId as string
        
    Returns:
        Invoice document with line_items array or None
    """
    try:
        oid = ObjectId(invoice_id)
        invoice = db.invoices.find_one({"_id": oid})
        
        if invoice:
            # Fetch associated line items from separate collection
            line_items = list(db.line_items.find({"invoice_id": oid}))
            invoice["line_items"] = line_items
            
        return invoice
    except Exception as e:
        print(f"Error retrieving invoice: {e}")
        return None


def check_duplicate_invoice(vendor_id: str, invoice_number: str) -> Optional[Dict[str, Any]]:
    """
    Check if an invoice already exists in the database.
    
    Args:
        vendor_id: The vendor ObjectId as string
        invoice_number: The invoice number to check
        
    Returns:
        Existing invoice document or None
    """
    try:
        vid = ObjectId(vendor_id)
        return db.invoices.find_one({
            "vendor_id": vid,
            "invoice_number": invoice_number
        })
    except Exception as e:
        print(f"Error checking duplicate: {e}")
        return None


def update_invoice(invoice_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update invoice fields (header level only, not line items).
    
    Args:
        invoice_id: The invoice ObjectId as string
        update_data: Dictionary of fields to update
        
    Returns:
        dict: {"success": bool, "message": str}
    """
    try:
        import datetime
        from decimal import Decimal
        
        oid = ObjectId(invoice_id)
        
        # Process special fields
        if "invoice_date" in update_data and isinstance(update_data["invoice_date"], str):
            update_data["invoice_date"] = pd.to_datetime(update_data["invoice_date"])
        
        if "order_date" in update_data and isinstance(update_data["order_date"], str):
            update_data["order_date"] = pd.to_datetime(update_data["order_date"])
        
        if "invoice_total_amount" in update_data:
            val = update_data["invoice_total_amount"]
            if isinstance(val, str):
                val = float(val.replace(",", ""))
            elif isinstance(val, Decimal):
                val = float(val)
            update_data["invoice_total_amount"] = Decimal128(str(val))
        
        update_data["updated_at"] = datetime.datetime.now()
        
        result = db.invoices.update_one(
            {"_id": oid},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            return {"success": True, "message": "Invoice updated successfully"}
        else:
            return {"success": False, "message": "No changes made or invoice not found"}
            
    except Exception as e:
        return {"success": False, "message": f"Error updating invoice: {str(e)}"}


def update_line_item(line_item_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a specific line item in the line_items collection.
    
    Args:
        line_item_id: The line_item ObjectId as string
        update_data: Dictionary of fields to update
        
    Returns:
        dict: {"success": bool, "message": str}
    """
    try:
        import datetime
        from decimal import Decimal
        
        oid = ObjectId(line_item_id)
        
        # Process numeric fields
        if "unit_price" in update_data:
            val = update_data["unit_price"]
            if isinstance(val, str):
                val = float(val.replace(",", ""))
            elif isinstance(val, Decimal):
                val = float(val)
            update_data["unit_price"] = Decimal128(str(val))
        
        if "line_total" in update_data:
            val = update_data["line_total"]
            if isinstance(val, str):
                val = float(val.replace(",", ""))
            elif isinstance(val, Decimal):
                val = float(val)
            update_data["line_total"] = Decimal128(str(val))
        
        if "quantity" in update_data:
            if isinstance(update_data["quantity"], str):
                update_data["quantity"] = float(update_data["quantity"].replace(",", ""))
            update_data["quantity"] = float(update_data["quantity"])
        
        update_data["updated_at"] = datetime.datetime.now()
        
        result = db.line_items.update_one(
            {"_id": oid},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            return {"success": True, "message": "Line item updated successfully"}
        else:
            return {"success": False, "message": "No changes made or line item not found"}
            
    except Exception as e:
        return {"success": False, "message": f"Error updating line item: {str(e)}"}


def delete_line_item(line_item_id: str) -> Dict[str, Any]:
    """
    Delete a specific line item from the line_items collection.
    
    Args:
        line_item_id: The line_item ObjectId as string
        
    Returns:
        dict: {"success": bool, "message": str}
    """
    try:
        oid = ObjectId(line_item_id)
        
        result = db.line_items.delete_one({"_id": oid})
        
        if result.deleted_count > 0:
            return {"success": True, "message": "Line item deleted successfully"}
        else:
            return {"success": False, "message": "Line item not found"}
            
    except Exception as e:
        return {"success": False, "message": f"Error deleting line item: {str(e)}"}


def add_line_item(invoice_id: str, line_item_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a new line item to an invoice in the line_items collection.
    
    Args:
        invoice_id: The invoice ObjectId as string
        line_item_data: Dictionary containing line item fields
        
    Returns:
        dict: {"success": bool, "message": str, "line_item_id": str}
    """
    try:
        from decimal import Decimal
        
        oid = ObjectId(invoice_id)
        
        # Check if invoice exists
        invoice = db.invoices.find_one({"_id": oid})
        if not invoice:
            return {"success": False, "message": "Invoice not found"}
        
        # Get next line number
        max_line = db.line_items.find_one(
            {"invoice_id": oid},
            sort=[("line_number", -1)]
        )
        next_line_number = (max_line.get("line_number", 0) if max_line else 0) + 1
        
        # Process and validate line item data
        unit_price = line_item_data.get("unit_price", 0)
        if isinstance(unit_price, str):
            unit_price = float(unit_price.replace(",", ""))
        elif isinstance(unit_price, Decimal):
            unit_price = float(unit_price)
        
        line_total = line_item_data.get("line_total", 0)
        if isinstance(line_total, str):
            line_total = float(line_total.replace(",", ""))
        elif isinstance(line_total, Decimal):
            line_total = float(line_total)
        
        quantity = line_item_data.get("quantity", 0)
        if isinstance(quantity, str):
            quantity = float(quantity.replace(",", ""))
        
        new_line_item = {
            "invoice_id": oid,
            "vendor_name": line_item_data.get("vendor_name", invoice.get("vendor_name", "")),
            "category": line_item_data.get("category", "Uncategorized"),
            "description": str(line_item_data.get("description", "")),
            "quantity": float(quantity),
            "unit": str(line_item_data.get("unit", "")),
            "unit_price": Decimal128(str(unit_price)),
            "line_total": Decimal128(str(line_total)),
            "line_number": Decimal128(str(next_line_number)),
        }
        
        # Insert the new line item
        result = db.line_items.insert_one(new_line_item)
        
        return {
            "success": True,
            "message": "Line item added successfully",
            "line_item_id": str(result.inserted_id)
        }
            
    except Exception as e:
        return {"success": False, "message": f"Error adding line item: {str(e)}"}

def get_line_items_by_invoice(invoice_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve all line items for a specific invoice.
    
    Args:
        invoice_id: The invoice ObjectId as string
        
    Returns:
        List of line item documents
    """
    try:
        oid = ObjectId(invoice_id)
        return list(db.line_items.find({"invoice_id": oid}))
    except Exception as e:
        print(f"Error retrieving line items: {e}")
        return []

# ---------------------------------------------------------
# Category & Lookup Method
# ---------------------------------------------------------
def get_all_category_names() -> List[str]:
    """Fetches just the list of category names."""
    cursor = db.categories.find({}, {"_id": 1})
    return [doc["_id"] for doc in cursor]

def get_stored_category(description: str) -> Optional[str]:
    """Finds if we have already categorized this description."""
    doc = db.item_lookup_map.find_one({"_id": description})
    return doc.get("category") if doc else None

def insert_master_category(category_name: str) -> None:
    """Inserts a new category into the master list."""
    try:
        db.categories.insert_one({"_id": category_name})
    except Exception:
        pass # Ignore duplicates if they race-condition in

def upsert_item_mapping(description: str, category_name: str) -> None:
    """Links a description to a category forever."""
    db.item_lookup_map.update_one(
        {"_id": description},
        {"$set": {"category": category_name}},
        upsert=True
    )

# ---------------------------------------------------------
# Temporary Upload Session Methods (for session persistence)
# ---------------------------------------------------------

def save_temp_upload(session_id: str, upload_data: Dict[str, Any]) -> bool:
    """
    Save temporary upload data for session persistence.
    
    Args:
        session_id: Unique session identifier
        upload_data: Dictionary containing extracted invoice data
        
    Returns:
        bool: Success status
    """
    try:
        import datetime
        
        upload_data["session_id"] = session_id
        upload_data["created_at"] = datetime.datetime.now()
        upload_data["updated_at"] = datetime.datetime.now()
        
        db.temp_uploads.update_one(
            {"session_id": session_id},
            {"$set": upload_data},
            upsert=True
        )
        return True
    except Exception as e:
        print(f"Error saving temp upload: {e}")
        return False


def get_temp_upload(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve temporary upload data for a session.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Dictionary with upload data or None
    """
    try:
        return db.temp_uploads.find_one({"session_id": session_id})
    except Exception as e:
        print(f"Error retrieving temp upload: {e}")
        return None


def delete_temp_upload(session_id: str) -> bool:
    """
    Delete temporary upload data after successful save.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        bool: Success status
    """
    try:
        db.temp_uploads.delete_one({"session_id": session_id})
        return True
    except Exception as e:
        print(f"Error deleting temp upload: {e}")
        return False


def cleanup_old_temp_uploads(days: int = 7) -> int:
    """
    Clean up temporary uploads older than specified days.
    
    Args:
        days: Number of days to keep temp data
        
    Returns:
        Number of deleted documents
    """
    try:
        import datetime
        
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        result = db.temp_uploads.delete_many({"created_at": {"$lt": cutoff_date}})
        return result.deleted_count
    except Exception as e:
        print(f"Error cleaning temp uploads: {e}")
        return 0


# ============================================================================
# DASHBOARD QUERY FUNCTIONS
# ============================================================================

def decimal128_to_float(val):
    """Convert Decimal128 to float for pandas compatibility."""
    if isinstance(val, Decimal128):
        return float(val.to_decimal())
    return val


def get_all_restaurants() -> List[Dict[str, Any]]:
    """
    Get all active restaurants for filtering.
    
    Returns:
        List of restaurant documents with _id, name, location_name
    """
    return list(db.restaurants.find(
        {"is_active": True},
        {"_id": 1, "name": 1, "location_name": 1}
    ).sort("location_name", 1))


def get_all_vendors() -> List[Dict[str, Any]]:
    """
    Get all vendors for filtering.
    
    Returns:
        List of vendor documents with _id and name
    """
    return list(db.vendors.find({}, {"_id": 1, "name": 1}).sort("name", 1))


def get_invoice_line_items_joined(
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
    restaurant_ids: Optional[List[ObjectId]] = None,
    vendor_ids: Optional[List[ObjectId]] = None
) -> pd.DataFrame:
    """
    Get joined invoice and line item data with vendor and restaurant names.
    
    This is the primary query for dashboard analytics. Returns flattened data
    with one row per line item, including all invoice and vendor details.
    
    Args:
        start_date: Filter invoices from this date (inclusive)
        end_date: Filter invoices to this date (inclusive)
        restaurant_ids: Filter by restaurant IDs (None = all)
        vendor_ids: Filter by vendor IDs (None = all)
    
    Returns:
        DataFrame with columns: invoice_id, invoice_number, invoice_date, 
        location, vendor, category, item_name, quantity, unit, unit_price, line_total
    """
    # Build match filter
    match_filter = {}
    
    if start_date or end_date:
        match_filter["invoice_date"] = {}
        if start_date:
            match_filter["invoice_date"]["$gte"] = start_date
        if end_date:
            # Include the entire end date (until 23:59:59)
            end_of_day = end_date.replace(hour=23, minute=59, second=59)
            match_filter["invoice_date"]["$lte"] = end_of_day
    
    if restaurant_ids:
        match_filter["restaurant_id"] = {"$in": restaurant_ids}
    
    if vendor_ids:
        match_filter["vendor_id"] = {"$in": vendor_ids}
    
    # Aggregation pipeline
    pipeline = [
        {"$match": match_filter},
        
        # Join with line_items
        {
            "$lookup": {
                "from": "line_items",
                "localField": "_id",
                "foreignField": "invoice_id",
                "as": "line_items"
            }
        },
        
        # Join with vendors
        {
            "$lookup": {
                "from": "vendors",
                "localField": "vendor_id",
                "foreignField": "_id",
                "as": "vendor_info"
            }
        },
        
        # Join with restaurants
        {
            "$lookup": {
                "from": "restaurants",
                "localField": "restaurant_id",
                "foreignField": "_id",
                "as": "restaurant_info"
            }
        },
        
        # Unwind arrays
        {"$unwind": "$line_items"},
        {"$unwind": "$vendor_info"},
        {"$unwind": "$restaurant_info"},
        
        # Project final structure
        {
            "$project": {
                "invoice_id": {"$toString": "$_id"},
                "invoice_number": 1,
                "invoice_date": 1,
                "location": "$restaurant_info.location_name",
                "vendor": "$vendor_info.name",
                "category": "$line_items.category",
                "item_name": "$line_items.description",
                "quantity": "$line_items.quantity",
                "unit": "$line_items.unit",
                "unit_price": "$line_items.unit_price",
                "line_total": "$line_items.line_total"
            }
        },
        
        # Sort by date descending
        {"$sort": {"invoice_date": -1}}
    ]
    
    # Execute query
    results = list(db.invoices.aggregate(pipeline))
    
    # Convert to DataFrame
    if not results:
        return pd.DataFrame(columns=[
            "invoice_id", "invoice_number", "invoice_date", "location", 
            "vendor", "category", "item_name", "quantity", "unit", 
            "unit_price", "line_total"
        ])
    
    df = pd.DataFrame(results)
    
    # Convert Decimal128 to float
    df['unit_price'] = df['unit_price'].apply(decimal128_to_float)
    df['line_total'] = df['line_total'].apply(decimal128_to_float)
    
    # Ensure invoice_date is datetime
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    
    return df


def get_sales_data(
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
    restaurant_ids: Optional[List[ObjectId]] = None
) -> pd.DataFrame:
    """
    Get daily sales data (revenue and covers) for food cost calculations.
    
    Args:
        start_date: Filter from this date (inclusive)
        end_date: Filter to this date (inclusive)
        restaurant_ids: Filter by restaurant IDs (None = all)
    
    Returns:
        DataFrame with columns: date, location, revenue, covers
    """
    # Build match filter
    match_filter = {}
    
    if start_date or end_date:
        match_filter["date"] = {}
        if start_date:
            match_filter["date"]["$gte"] = start_date
        if end_date:
            end_of_day = end_date.replace(hour=23, minute=59, second=59)
            match_filter["date"]["$lte"] = end_of_day
    
    if restaurant_ids:
        match_filter["restaurant_id"] = {"$in": restaurant_ids}
    
    # Aggregation pipeline
    pipeline = [
        {"$match": match_filter},
        
        # Join with restaurants
        {
            "$lookup": {
                "from": "restaurants",
                "localField": "restaurant_id",
                "foreignField": "_id",
                "as": "restaurant_info"
            }
        },
        
        {"$unwind": "$restaurant_info"},
        
        # Project final structure
        {
            "$project": {
                "date": 1,
                "location": "$restaurant_info.location_name",
                "revenue": 1,
                "covers": 1
            }
        },
        
        # Sort by date
        {"$sort": {"date": 1}}
    ]
    
    # Execute query
    results = list(db.sales.aggregate(pipeline))
    
    # Convert to DataFrame
    if not results:
        return pd.DataFrame(columns=["date", "location", "revenue", "covers"])
    
    df = pd.DataFrame(results)
    df['date'] = pd.to_datetime(df['date'])
    
    return df


def get_spending_by_period(
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    restaurant_ids: Optional[List[ObjectId]] = None,
    group_by: str = "day"
) -> pd.DataFrame:
    """
    Get total spending aggregated by time period.
    
    Args:
        start_date: Start date for analysis
        end_date: End date for analysis
        restaurant_ids: Filter by restaurants (None = all)
        group_by: Aggregation period - "day", "week", or "month"
    
    Returns:
        DataFrame with columns: period, total_spend
    """
    # Define date grouping based on period
    if group_by == "day":
        date_format = "%Y-%m-%d"
        group_expr = {"$dateToString": {"format": date_format, "date": "$invoice_date"}}
    elif group_by == "week":
        group_expr = {
            "$dateToString": {
                "format": "%Y-W%V",
                "date": "$invoice_date"
            }
        }
    elif group_by == "month":
        date_format = "%Y-%m"
        group_expr = {"$dateToString": {"format": date_format, "date": "$invoice_date"}}
    else:
        raise ValueError(f"Invalid group_by: {group_by}")
    
    # Build match filter
    match_filter = {
        "invoice_date": {
            "$gte": start_date,
            "$lte": end_date.replace(hour=23, minute=59, second=59)
        }
    }
    
    if restaurant_ids:
        match_filter["restaurant_id"] = {"$in": restaurant_ids}
    
    # Aggregation pipeline
    pipeline = [
        {"$match": match_filter},
        
        {
            "$group": {
                "_id": group_expr,
                "total_spend": {"$sum": "$invoice_total_amount"}
            }
        },
        
        {"$sort": {"_id": 1}},
        
        {
            "$project": {
                "period": "$_id",
                "total_spend": 1,
                "_id": 0
            }
        }
    ]
    
    results = list(db.invoices.aggregate(pipeline))
    
    if not results:
        return pd.DataFrame(columns=["period", "total_spend"])
    
    df = pd.DataFrame(results)
    df['total_spend'] = df['total_spend'].apply(decimal128_to_float)
    
    return df


def get_category_breakdown(
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    restaurant_ids: Optional[List[ObjectId]] = None
) -> pd.DataFrame:
    """
    Get spending breakdown by category.
    
    Args:
        start_date: Start date for analysis
        end_date: End date for analysis
        restaurant_ids: Filter by restaurants (None = all)
    
    Returns:
        DataFrame with columns: category, total_spend, percentage
    """
    # Build match filter
    match_filter = {
        "invoice_date": {
            "$gte": start_date,
            "$lte": end_date.replace(hour=23, minute=59, second=59)
        }
    }
    
    if restaurant_ids:
        match_filter["restaurant_id"] = {"$in": restaurant_ids}
    
    # Aggregation pipeline
    pipeline = [
        {"$match": match_filter},
        
        # Join with line_items
        {
            "$lookup": {
                "from": "line_items",
                "localField": "_id",
                "foreignField": "invoice_id",
                "as": "line_items"
            }
        },
        
        {"$unwind": "$line_items"},
        
        # Group by category
        {
            "$group": {
                "_id": "$line_items.category",
                "total_spend": {"$sum": "$line_items.line_total"}
            }
        },
        
        {"$sort": {"total_spend": -1}},
        
        {
            "$project": {
                "category": "$_id",
                "total_spend": 1,
                "_id": 0
            }
        }
    ]
    
    results = list(db.invoices.aggregate(pipeline))
    
    if not results:
        return pd.DataFrame(columns=["category", "total_spend", "percentage"])
    
    df = pd.DataFrame(results)
    df['total_spend'] = df['total_spend'].apply(decimal128_to_float)
    
    # Calculate percentages
    total = df['total_spend'].sum()
    df['percentage'] = (df['total_spend'] / total * 100).round(2)
    
    return df


def get_vendor_spending(
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    restaurant_ids: Optional[List[ObjectId]] = None
) -> pd.DataFrame:
    """
    Get spending breakdown by vendor.
    
    Args:
        start_date: Start date for analysis
        end_date: End date for analysis
        restaurant_ids: Filter by restaurants (None = all)
    
    Returns:
        DataFrame with columns: vendor, total_spend, invoice_count
    """
    # Build match filter
    match_filter = {
        "invoice_date": {
            "$gte": start_date,
            "$lte": end_date.replace(hour=23, minute=59, second=59)
        }
    }
    
    if restaurant_ids:
        match_filter["restaurant_id"] = {"$in": restaurant_ids}
    
    # Aggregation pipeline
    pipeline = [
        {"$match": match_filter},
        
        # Join with vendors
        {
            "$lookup": {
                "from": "vendors",
                "localField": "vendor_id",
                "foreignField": "_id",
                "as": "vendor_info"
            }
        },
        
        {"$unwind": "$vendor_info"},
        
        # Group by vendor
        {
            "$group": {
                "_id": "$vendor_info.name",
                "total_spend": {"$sum": "$invoice_total_amount"},
                "invoice_count": {"$sum": 1}
            }
        },
        
        {"$sort": {"total_spend": -1}},
        
        {
            "$project": {
                "vendor": "$_id",
                "total_spend": 1,
                "invoice_count": 1,
                "_id": 0
            }
        }
    ]
    
    results = list(db.invoices.aggregate(pipeline))
    
    if not results:
        return pd.DataFrame(columns=["vendor", "total_spend", "invoice_count"])
    
    df = pd.DataFrame(results)
    df['total_spend'] = df['total_spend'].apply(decimal128_to_float)
    
    return df


def get_top_items_by_spend(
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    restaurant_ids: Optional[List[ObjectId]] = None,
    limit: int = 20
) -> pd.DataFrame:
    """
    Get top items by total spending.
    
    Args:
        start_date: Start date for analysis
        end_date: End date for analysis
        restaurant_ids: Filter by restaurants (None = all)
        limit: Number of top items to return
    
    Returns:
        DataFrame with columns: item_name, category, total_spend, avg_price
    """
    # Build match filter
    match_filter = {
        "invoice_date": {
            "$gte": start_date,
            "$lte": end_date.replace(hour=23, minute=59, second=59)
        }
    }
    
    if restaurant_ids:
        match_filter["restaurant_id"] = {"$in": restaurant_ids}
    
    # Aggregation pipeline
    pipeline = [
        {"$match": match_filter},
        
        # Join with line_items
        {
            "$lookup": {
                "from": "line_items",
                "localField": "_id",
                "foreignField": "invoice_id",
                "as": "line_items"
            }
        },
        
        {"$unwind": "$line_items"},
        
        # Group by item
        {
            "$group": {
                "_id": "$line_items.description",
                "category": {"$first": "$line_items.category"},
                "total_spend": {"$sum": "$line_items.line_total"},
                "avg_price": {"$avg": "$line_items.unit_price"}
            }
        },
        
        {"$sort": {"total_spend": -1}},
        {"$limit": limit},
        
        {
            "$project": {
                "item_name": "$_id",
                "category": 1,
                "total_spend": 1,
                "avg_price": 1,
                "_id": 0
            }
        }
    ]
    
    results = list(db.invoices.aggregate(pipeline))
    
    if not results:
        return pd.DataFrame(columns=["item_name", "category", "total_spend", "avg_price"])
    
    df = pd.DataFrame(results)
    df['total_spend'] = df['total_spend'].apply(decimal128_to_float)
    df['avg_price'] = df['avg_price'].apply(decimal128_to_float)
    
    return df


def get_price_variations(
    item_name: str,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None
) -> pd.DataFrame:
    """
    Track price variations for a specific item over time.
    
    Args:
        item_name: Name of the item to track
        start_date: Start date for analysis (optional)
        end_date: End date for analysis (optional)
    
    Returns:
        DataFrame with columns: date, vendor, unit_price, quantity
    """
    # Build match filter
    match_filter = {}
    
    if start_date or end_date:
        match_filter["invoice_date"] = {}
        if start_date:
            match_filter["invoice_date"]["$gte"] = start_date
        if end_date:
            match_filter["invoice_date"]["$lte"] = end_date.replace(hour=23, minute=59, second=59)
    
    # Aggregation pipeline
    pipeline = [
        {"$match": match_filter},
        
        # Join with line_items
        {
            "$lookup": {
                "from": "line_items",
                "localField": "_id",
                "foreignField": "invoice_id",
                "as": "line_items"
            }
        },
        
        {"$unwind": "$line_items"},
        
        # Match specific item
        {"$match": {"line_items.description": item_name}},
        
        # Join with vendors
        {
            "$lookup": {
                "from": "vendors",
                "localField": "vendor_id",
                "foreignField": "_id",
                "as": "vendor_info"
            }
        },
        
        {"$unwind": "$vendor_info"},
        
        {"$sort": {"invoice_date": 1}},
        
        {
            "$project": {
                "date": "$invoice_date",
                "vendor": "$vendor_info.name",
                "unit_price": "$line_items.unit_price",
                "quantity": "$line_items.quantity",
                "_id": 0
            }
        }
    ]
    
    results = list(db.invoices.aggregate(pipeline))
    
    if not results:
        return pd.DataFrame(columns=["date", "vendor", "unit_price", "quantity"])
    
    df = pd.DataFrame(results)
    df['unit_price'] = df['unit_price'].apply(decimal128_to_float)
    df['date'] = pd.to_datetime(df['date'])
    
    return df


def get_recent_invoices(
    limit: int = 10,
    restaurant_ids: Optional[List[ObjectId]] = None
) -> pd.DataFrame:
    """
    Get most recent invoices with summary information.
    
    Args:
        limit: Number of invoices to return
        restaurant_ids: Filter by restaurants (None = all)
    
    Returns:
        DataFrame with invoice summary information
    """
    match_filter = {}
    if restaurant_ids:
        match_filter["restaurant_id"] = {"$in": restaurant_ids}
    
    pipeline = [
        {"$match": match_filter},
        {"$sort": {"invoice_date": -1}},
        {"$limit": limit},
        
        # Join with vendors
        {
            "$lookup": {
                "from": "vendors",
                "localField": "vendor_id",
                "foreignField": "_id",
                "as": "vendor_info"
            }
        },
        
        # Join with restaurants
        {
            "$lookup": {
                "from": "restaurants",
                "localField": "restaurant_id",
                "foreignField": "_id",
                "as": "restaurant_info"
            }
        },
        
        {"$unwind": "$vendor_info"},
        {"$unwind": "$restaurant_info"},
        
        {
            "$project": {
                "invoice_number": 1,
                "invoice_date": 1,
                "vendor": "$vendor_info.name",
                "location": "$restaurant_info.location_name",
                "total_amount": "$invoice_total_amount",
                "_id": 0
            }
        }
    ]
    
    results = list(db.invoices.aggregate(pipeline))
    
    if not results:
        return pd.DataFrame(columns=[
            "invoice_number", "invoice_date", "vendor", "location", "total_amount"
        ])
    
    df = pd.DataFrame(results)
    df['total_amount'] = df['total_amount'].apply(decimal128_to_float)
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    
    return df