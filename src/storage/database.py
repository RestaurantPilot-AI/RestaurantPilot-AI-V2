import os
import re
import csv
import json
import uuid
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------------------------------
# CSV Database Setup
# ---------------------------------------------------------
# Navigate: src/storage/ -> src/ -> root/ -> data/database
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CSV_FOLDER = BASE_DIR / "data" / "database"
CSV_FOLDER.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Collection (File) Names
# ---------------------------------------------------------
COL_VENDORS = "vendors"
COL_VENDOR_REGEXES = "vendor_regex_templates"

# ---------------------------------------------------------
# 1. Internal CSV Helpers (Private)
# ---------------------------------------------------------
def _get_table_path(table_name: str) -> Path:
    return CSV_FOLDER / f"{table_name}.csv"

def _read_table(table_name: str) -> pd.DataFrame:
    """Reads CSV. If missing, returns empty DF."""
    path = _get_table_path(table_name)
    if not path.exists():
        return pd.DataFrame()
    try:
        # Read as string to preserve IDs
        return pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def _save_table(df: pd.DataFrame, table_name: str):
    """Saves DF to CSV."""
    path = _get_table_path(table_name)
    df.to_csv(path, index=False, encoding='utf-8')

def _generate_id() -> str:
    """Generates a unique string ID."""
    return str(uuid.uuid4())

def _matches_query(row: pd.Series, query: Dict) -> bool:
    """Simple query matcher for find/find_one."""
    if not query: return True
    for k, v in query.items():
        val = str(row.get(k, ""))
        
        if isinstance(v, dict):
            # Basic Mongo Operators
            if "$in" in v:
                # Convert list items to string for comparison
                check_list = [str(x) for x in v["$in"]]
                if val not in check_list: return False
            elif "$gte" in v:
                if not val >= str(v["$gte"]): return False
            elif "$lte" in v:
                if not val <= str(v["$lte"]): return False
            elif "$lt" in v:
                if not val < str(v["$lt"]): return False
            elif "$gt" in v:
                if not val > str(v["$gt"]): return False
        else:
            # Direct Equality
            if val != str(v): return False
    return True

# ---------------------------------------------------------
# 2. Mock Classes (The MongoDB Shim)
# ---------------------------------------------------------

class ObjectId:
    """Mock ObjectId for compatibility."""
    def __init__(self, oid=None):
        self._id = str(oid) if oid else _generate_id()
    def __str__(self): return self._id
    def __repr__(self): return self._id
    def __eq__(self, other): return str(self) == str(other)

class Decimal128:
    """Mock Decimal128 for compatibility."""
    def __init__(self, value):
        self.value = str(value)
    def to_decimal(self):
        try:
            return float(self.value)
        except:
            return 0.0
    def __str__(self): return self.value
    def __repr__(self): return f"Decimal128('{self.value}')"

class MockCursor:
    """Simulates a MongoDB Cursor."""
    def __init__(self, data: List[Dict]):
        self._data = data
    
    def sort(self, key_or_list, direction=None):
        if not self._data: return self
        # Handle list of tuples [(key, direction)] or single key
        key = key_or_list[0][0] if isinstance(key_or_list, list) else key_or_list
        reverse = (direction == -1) or (isinstance(key_or_list, list) and key_or_list[0][1] == -1)
        
        self._data.sort(key=lambda x: str(x.get(key, "")), reverse=reverse)
        return self

    def limit(self, n):
        self._data = self._data[:n]
        return self

    def __iter__(self):
        for item in self._data:
            yield item
            
    def __list__(self):
        return self._data

class MockInsertResult:
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id

class MockUpdateResult:
    def __init__(self, modified_count):
        self.modified_count = modified_count

class MockDeleteResult:
    def __init__(self, deleted_count):
        self.deleted_count = deleted_count

class MockCollection:
    """Simulates a MongoDB Collection using CSVs."""
    def __init__(self, name):
        self.name = name

    def find_one(self, query=None, projection=None):
        df = _read_table(self.name)
        if df.empty: return None
        
        # Iterate to find match
        for _, row in df.iterrows():
            if _matches_query(row, query):
                return row.dropna().to_dict()
        return None

    def find(self, query=None, projection=None):
        df = _read_table(self.name)
        results = []
        if not df.empty:
            for _, row in df.iterrows():
                if _matches_query(row, query):
                    results.append(row.dropna().to_dict())
        return MockCursor(results)

    def insert_one(self, document):
        # Sanitize types for CSV
        clean_doc = {}
        for k, v in document.items():
            if isinstance(v, (ObjectId, Decimal128)):
                clean_doc[k] = str(v)
            elif isinstance(v, datetime.datetime):
                clean_doc[k] = v.isoformat()
            else:
                clean_doc[k] = v
        
        if "_id" not in clean_doc:
            clean_doc["_id"] = _generate_id()
            
        df = _read_table(self.name)
        df = pd.concat([df, pd.DataFrame([clean_doc])], ignore_index=True)
        _save_table(df, self.name)
        return MockInsertResult(clean_doc["_id"])

    def insert_many(self, documents):
        clean_docs = []
        ids = []
        for doc in documents:
            clean = doc.copy()
            if "_id" not in clean:
                clean["_id"] = _generate_id()
            # Stringify custom types
            for k, v in clean.items():
                if isinstance(v, (ObjectId, Decimal128)):
                    clean[k] = str(v)
                elif isinstance(v, datetime.datetime):
                    clean[k] = v.isoformat()
            
            clean_docs.append(clean)
            ids.append(clean["_id"])

        df = _read_table(self.name)
        df = pd.concat([df, pd.DataFrame(clean_docs)], ignore_index=True)
        _save_table(df, self.name)
        # Return a simple object with inserted_ids
        return type('obj', (object,), {'inserted_ids': ids})

    def update_one(self, filter, update, upsert=False):
        df = _read_table(self.name)
        target_idx = -1
        
        # Find row index
        if not df.empty:
            for idx, row in df.iterrows():
                if _matches_query(row, filter):
                    target_idx = idx
                    break
        
        if target_idx == -1:
            if upsert:
                # Create new
                new_doc = filter.copy()
                if "$set" in update:
                    new_doc.update(update["$set"])
                if "_id" not in new_doc:
                    new_doc["_id"] = _generate_id()
                self.insert_one(new_doc)
                return MockUpdateResult(0)
            return MockUpdateResult(0)
            
        # Update existing
        if "$set" in update:
            for k, v in update["$set"].items():
                val = v
                if isinstance(v, (ObjectId, Decimal128)): val = str(v)
                elif isinstance(v, datetime.datetime): val = v.isoformat()
                df.at[target_idx, k] = val
        
        _save_table(df, self.name)
        return MockUpdateResult(1)

    def delete_one(self, filter):
        df = _read_table(self.name)
        if df.empty: return MockDeleteResult(0)
        
        target_idx = -1
        for idx, row in df.iterrows():
            if _matches_query(row, filter):
                target_idx = idx
                break
        
        if target_idx != -1:
            df = df.drop(target_idx)
            _save_table(df, self.name)
            return MockDeleteResult(1)
        return MockDeleteResult(0)
    
    def delete_many(self, filter):
        df = _read_table(self.name)
        if df.empty: return MockDeleteResult(0)
        
        # Identify indices to drop
        drop_indices = []
        for idx, row in df.iterrows():
            if _matches_query(row, filter):
                drop_indices.append(idx)
        
        if drop_indices:
            df = df.drop(drop_indices)
            _save_table(df, self.name)
        
        return MockDeleteResult(len(drop_indices))

    def aggregate(self, pipeline):
        # NOTE: Aggregation is strictly handled by specific functions below.
        return []

class MockDatabase:
    def __getitem__(self, name):
        return MockCollection(name)
    def __getattr__(self, name):
        return MockCollection(name)

class MockClient:
    def __getitem__(self, name):
        return MockDatabase()
    @property
    def admin(self):
         # Mock admin.command('ping')
         return type('obj', (object,), {'command': lambda x: True})

# ---------------------------------------------------------
# 3. GLOBAL EXPORTS (The Fix for ImportError)
# ---------------------------------------------------------
client = MockClient()
db = MockDatabase() 

# ---------------------------------------------------------
# 4. Specific Business Logic Methods (1:1 Signatures)
# ---------------------------------------------------------

def to_decimal128(val):
    """Helper to convert generic numbers/strings to Mock Decimal128."""
    if pd.isna(val) or val is None:
        return Decimal128("0.00")
    try:
        if isinstance(val, str): val = val.replace(',', '')
        return Decimal128(str(float(val)))
    except:
        return Decimal128("0.00")

def _get_float(val):
    """Internal helper to extract float from Decimal128 or string."""
    if isinstance(val, Decimal128): return val.to_decimal()
    try: return float(str(val).replace(',', ''))
    except: return 0.0

# --- Vendor Regex Methods ---
def save_vendor_regex_template(new_vendor_id: str, new_regexes: Dict[str, Any]) -> None:
    inv = new_regexes.get("invoice_level", {})
    li = new_regexes.get("line_item_level", {})
    
    flattened = [
        inv.get("invoice_number", ""), inv.get("invoice_date", ""),
        inv.get("invoice_total_amount", ""), inv.get("order_date", ""),
        li.get("line_item_block_start", ""), li.get("line_item_block_end", ""),
        li.get("quantity", ""), li.get("description", ""),
        li.get("unit", ""), li.get("unit_price", ""), li.get("line_total", "")
    ]
    
    db[COL_VENDOR_REGEXES].update_one(
        {"vendor_id": str(new_vendor_id)},
        {"$set": {"regex_patterns": json.dumps(flattened)}},
        upsert=True
    )

def get_vendor_regex_patterns(vendor_id: str) -> List[str]:
    doc = db[COL_VENDOR_REGEXES].find_one({"vendor_id": str(vendor_id)})
    if doc and "regex_patterns" in doc:
        try: return json.loads(doc["regex_patterns"])
        except: return []
    return []

# --- Vendor Methods ---
def create_vendor(vendor_data: Dict[str, Any]) -> Optional[str]:
    name = vendor_data.get("vendor_name")
    if not name: 
        return None
    
    # Check duplicate
    existing = db[COL_VENDORS].find_one({"name": name}) 
    if existing: 
        return None
    
    # --- CRITICAL FIX: Clean the Address Data ---
    raw_address = vendor_data.get("vendor_physical_address")
    
    # Replace all newline characters (\n, \r, \r\n) with a single space.
    # We use re.sub for robust replacement of different newline types.
    cleaned_address = re.sub(r'[\r\n]+', ' ', raw_address).strip() if raw_address else None
    
    new_vendor = {
        "name": name,
        "contact_email": vendor_data.get("vendor_email_id"),
        "phone_number": vendor_data.get("vendor_phone_number"),
        "address": cleaned_address,  # Use the cleaned value here
        "website": vendor_data.get("vendor_website"),
    }
    
    # Clean None/empty values
    new_vendor = {k: v for k, v in new_vendor.items() if v}
    
    res = db[COL_VENDORS].insert_one(new_vendor)
    return str(res.inserted_id)

def _find_vid(query):
    doc = db[COL_VENDORS].find_one(query)
    return str(doc["_id"]) if doc else None

def get_vendor_by_email(email: str) -> Optional[str]: return _find_vid({"contact_email": email})
def get_vendor_by_website(website: str) -> Optional[str]: return _find_vid({"website": website})
def get_vendor_by_address(address: str) -> Optional[str]: return _find_vid({"address": address})
def get_vendor_by_phone(phone: str) -> Optional[str]: return _find_vid({"phone_number": phone})
def get_vendor_by_name(name: str) -> Optional[str]: return _find_vid({"name": name})

def get_vendor_name_by_id(vendor_id: str) -> Optional[str]:
    doc = db[COL_VENDORS].find_one({"_id": str(vendor_id)})
    return doc.get("name") if doc else None

# --- Main Save Logic ---
def save_inv_li_to_db(inv_df: pd.DataFrame, li_df: pd.DataFrame):
    if inv_df.empty: return {"success": False, "message": "No data", "invoice_id": None}
    
    inv_data = inv_df.iloc[0].to_dict()
    
    # Convert dates
    idate = pd.to_datetime(inv_data.get("invoice_date"))
    odate = pd.to_datetime(inv_data.get("order_date"))
    edate = pd.to_datetime(inv_data.get("extraction_timestamp"))
    
    inv_doc = {
        "filename": inv_data.get("filename"),
        "restaurant_id": str(inv_data.get("restaurant_id")),
        "vendor_id": str(inv_data.get("vendor_id")),
        "invoice_number": str(inv_data.get("invoice_number")),
        "invoice_date": idate.isoformat() if pd.notna(idate) else None,
        "invoice_total_amount": to_decimal128(inv_data.get("invoice_total_amount")),
        "text_length": int(inv_data.get("text_length", 0)),
        "page_count": int(inv_data.get("page_count", 0)),
        "extraction_timestamp": edate.isoformat() if pd.notna(edate) else None,
        "order_date": odate.isoformat() if pd.notna(odate) else None
    }
    
    res_inv = db.invoices.insert_one(inv_doc)
    new_inv_id = str(res_inv.inserted_id)
    print(f"[INFO] Invoice Saved: {new_inv_id}")
    
    if not li_df.empty:
        li_records = li_df.to_dict("records")
        for item in li_records:
            item_doc = {
                "invoice_id": new_inv_id,
                "vendor_name": str(item.get("vendor_name", "")),
                "category": str(item.get("category") or "Uncategorized"),
                "quantity": float(item.get("quantity", 0)),
                "unit": str(item.get("unit", "")),
                "description": str(item.get("description", "")),
                "unit_price": to_decimal128(item.get("unit_price")),
                "line_total": to_decimal128(item.get("line_total")),
                "line_number": to_decimal128(item.get("line_number"))
            }
            db.line_items.insert_one(item_doc)
            
    return {"success": True, "message": "Saved", "invoice_id": new_inv_id}

# --- CRUD Wrappers ---
def get_invoice_by_id(invoice_id: str):
    inv = db.invoices.find_one({"_id": str(invoice_id)})
    if inv:
        items = list(db.line_items.find({"invoice_id": str(invoice_id)}))
        inv["line_items"] = items
    return inv

def check_duplicate_invoice(vendor_id: str, invoice_number: str):
    return db.invoices.find_one({"vendor_id": str(vendor_id), "invoice_number": str(invoice_number)})

def update_invoice(invoice_id: str, update_data: Dict):
    # Flatten dates/decimals for the Mock Update logic
    if "invoice_date" in update_data: 
        update_data["invoice_date"] = pd.to_datetime(update_data["invoice_date"]).isoformat()
    if "invoice_total_amount" in update_data:
        update_data["invoice_total_amount"] = to_decimal128(update_data["invoice_total_amount"])
    
    db.invoices.update_one({"_id": str(invoice_id)}, {"$set": update_data})
    return {"success": True}

def update_line_item(line_item_id: str, update_data: Dict):
    for f in ["unit_price", "line_total", "quantity"]:
        if f in update_data: update_data[f] = to_decimal128(update_data[f])
    db.line_items.update_one({"_id": str(line_item_id)}, {"$set": update_data})
    return {"success": True}

def delete_line_item(line_item_id: str):
    db.line_items.delete_one({"_id": str(line_item_id)})
    return {"success": True}

def add_line_item(invoice_id: str, line_item_data: Dict):
    inv = get_invoice_by_id(invoice_id)
    if not inv: return {"success": False}
    
    # Simple max line logic via reading table
    items = list(db.line_items.find({"invoice_id": str(invoice_id)}))
    max_l = 0.0
    for i in items:
        try: max_l = max(max_l, float(str(i.get("line_number", 0))))
        except: pass
        
    new_doc = {
        "invoice_id": str(invoice_id),
        "vendor_name": str(line_item_data.get("vendor_name", inv.get("vendor_name", ""))),
        "category": str(line_item_data.get("category", "Uncategorized")),
        "description": str(line_item_data.get("description", "")),
        "quantity": float(line_item_data.get("quantity", 0)),
        "unit": str(line_item_data.get("unit", "")),
        "unit_price": to_decimal128(line_item_data.get("unit_price", 0)),
        "line_total": to_decimal128(line_item_data.get("line_total", 0)),
        "line_number": to_decimal128(max_l + 1)
    }
    
    res = db.line_items.insert_one(new_doc)
    return {"success": True, "line_item_id": str(res.inserted_id)}

def get_line_items_by_invoice(invoice_id: str):
    return list(db.line_items.find({"invoice_id": str(invoice_id)}))

# --- Categories ---
def get_all_category_names() -> List[str]:
    cats = list(db.categories.find())
    return [c["_id"] for c in cats if "_id" in c]

def get_stored_category(description: str) -> Optional[str]:
    doc = db["item_lookup_map"].find_one({"_id": description})
    return doc.get("category") if doc else None

def insert_master_category(category_name: str):
    if not db.categories.find_one({"_id": category_name}):
        db.categories.insert_one({"_id": category_name})

def upsert_item_mapping(description: str, category_name: str):
    db["item_lookup_map"].update_one(
        {"_id": description}, 
        {"$set": {"category": category_name}}, 
        upsert=True
    )

# --- Temp Uploads ---
def save_temp_upload(session_id: str, upload_data: Dict):
    data_str = json.dumps(upload_data, default=str)
    db.temp_uploads.update_one(
        {"session_id": session_id},
        {"$set": {"data": data_str, "updated_at": datetime.datetime.now().isoformat()}},
        upsert=True
    )
    return True

def get_temp_upload(session_id: str):
    doc = db.temp_uploads.find_one({"session_id": session_id})
    if doc and "data" in doc:
        return json.loads(doc["data"])
    return None

def delete_temp_upload(session_id: str):
    db.temp_uploads.delete_one({"session_id": session_id})
    return True

def cleanup_old_temp_uploads(days=7):
    # Basic cleanup simulation
    return 0

# ---------------------------------------------------------
# Aggregation / Dashboard Logic (Rewritten in Pandas)
# ---------------------------------------------------------

def decimal128_to_float(val):
    return _get_float(val)

def get_all_restaurants() -> List[Dict[str, Any]]:
    return list(db.restaurants.find({"is_active": "True"}).sort("location_name"))

def get_all_vendors() -> List[Dict[str, Any]]:
    return list(db.vendors.find().sort("name"))

def get_invoice_line_items_joined(start_date=None, end_date=None, restaurant_ids=None, vendor_ids=None):
    inv_df = _read_table("invoices")
    li_df = _read_table("line_items")
    rest_df = _read_table("restaurants")
    vend_df = _read_table("vendors")
    
    empty = pd.DataFrame(columns=["invoice_id", "invoice_number", "invoice_date", "location", "vendor", "category", "item_name", "quantity", "unit", "unit_price", "line_total"])
    if inv_df.empty: return empty

    inv_df["invoice_date"] = pd.to_datetime(inv_df["invoice_date"])
    
    if start_date: inv_df = inv_df[inv_df["invoice_date"] >= start_date]
    if end_date: 
        e = end_date.replace(hour=23, minute=59, second=59)
        inv_df = inv_df[inv_df["invoice_date"] <= e]
    
    if restaurant_ids:
        rids = [str(i) for i in restaurant_ids]
        inv_df = inv_df[inv_df["restaurant_id"].isin(rids)]
    if vendor_ids:
        vids = [str(i) for i in vendor_ids]
        inv_df = inv_df[inv_df["vendor_id"].isin(vids)]
        
    if inv_df.empty: return empty
    
    # Merges
    merged = pd.merge(inv_df, li_df, left_on="_id", right_on="invoice_id", suffixes=('_inv', '_li'))
    merged = pd.merge(merged, rest_df, left_on="restaurant_id", right_on="_id", suffixes=('', '_rest'), how="left")
    merged = pd.merge(merged, vend_df, left_on="vendor_id", right_on="_id", suffixes=('', '_vend'), how="left")
    
    final = pd.DataFrame()
    final["invoice_id"] = merged["_id_inv"]
    final["invoice_number"] = merged["invoice_number"]
    final["invoice_date"] = merged["invoice_date"]
    final["location"] = merged["location_name"]
    final["vendor"] = merged["name_vend"] if "name_vend" in merged else merged.get("name", "")
    final["category"] = merged["category"]
    final["item_name"] = merged["description"]
    final["quantity"] = merged["quantity"].astype(float)
    final["unit"] = merged["unit"]
    final["unit_price"] = merged["unit_price"].apply(_get_float)
    final["line_total"] = merged["line_total"].apply(_get_float)
    
    return final.sort_values("invoice_date", ascending=False)

def get_sales_data(start_date=None, end_date=None, restaurant_ids=None):
    df = _read_table("sales")
    if df.empty: return pd.DataFrame(columns=["date", "location", "revenue", "covers"])
    
    df["date"] = pd.to_datetime(df["date"])
    if start_date: df = df[df["date"] >= start_date]
    if end_date: df = df[df["date"] <= end_date.replace(hour=23)]
    
    rest = _read_table("restaurants")
    m = pd.merge(df, rest, left_on="restaurant_id", right_on="_id", how="left")
    
    final = pd.DataFrame()
    final["date"] = m["date"]
    final["location"] = m["location_name"]
    final["revenue"] = m["revenue"].astype(float)
    final["covers"] = m["covers"].astype(int)
    return final.sort_values("date")

def get_spending_by_period(start_date, end_date, restaurant_ids=None, group_by="day"):
    inv = _read_table("invoices")
    if inv.empty: return pd.DataFrame(columns=["period", "total_spend"])
    inv["invoice_date"] = pd.to_datetime(inv["invoice_date"])
    inv = inv[(inv["invoice_date"] >= start_date) & (inv["invoice_date"] <= end_date.replace(hour=23))]
    
    if group_by == "day": fmt = "%Y-%m-%d"
    elif group_by == "month": fmt = "%Y-%m"
    else: fmt = "%Y-%m-%d"
    
    inv["period"] = inv["invoice_date"].dt.strftime(fmt)
    inv["amt"] = inv["invoice_total_amount"].apply(_get_float)
    return inv.groupby("period")["amt"].sum().reset_index(name="total_spend")

def get_category_breakdown(start_date, end_date, restaurant_ids=None):
    df = get_invoice_line_items_joined(start_date, end_date, restaurant_ids)
    if df.empty: return pd.DataFrame(columns=["category", "total_spend", "percentage"])
    
    g = df.groupby("category")["line_total"].sum().reset_index(name="total_spend")
    g["percentage"] = (g["total_spend"] / g["total_spend"].sum() * 100).round(2)
    return g.sort_values("total_spend", ascending=False)

def get_vendor_spending(start_date, end_date, restaurant_ids=None):
    inv = _read_table("invoices")
    if inv.empty: return pd.DataFrame(columns=["vendor", "total_spend", "invoice_count"])
    inv["invoice_date"] = pd.to_datetime(inv["invoice_date"])
    inv = inv[(inv["invoice_date"] >= start_date) & (inv["invoice_date"] <= end_date.replace(hour=23))]
    
    vend = _read_table("vendors")
    m = pd.merge(inv, vend, left_on="vendor_id", right_on="_id", how="left")
    m["amt"] = m["invoice_total_amount"].apply(_get_float)
    
    g = m.groupby("name").agg(total_spend=("amt", "sum"), invoice_count=("amt", "count")).reset_index()
    return g.rename(columns={"name": "vendor"}).sort_values("total_spend", ascending=False)

def get_top_items_by_spend(start_date, end_date, restaurant_ids=None, limit=20):
    df = get_invoice_line_items_joined(start_date, end_date, restaurant_ids)
    if df.empty: return pd.DataFrame(columns=["item_name", "category", "total_spend", "avg_price"])
    g = df.groupby("item_name").agg(category=("category", "first"), total_spend=("line_total", "sum"), avg_price=("unit_price", "mean")).reset_index()
    return g.sort_values("total_spend", ascending=False).head(limit)

def get_price_variations(item_name, start_date=None, end_date=None):
    df = get_invoice_line_items_joined(start_date, end_date)
    if df.empty: return pd.DataFrame(columns=["date", "vendor", "unit_price", "quantity"])
    df = df[df["item_name"] == item_name]
    return df[["invoice_date", "vendor", "unit_price", "quantity"]].rename(columns={"invoice_date": "date"}).sort_values("date")

def get_price_variations_overview(restaurant_id: str, start_date=None, end_date=None, vendor_ids=None, min_occurrences: int = 1):
    """
    Returns an overview DataFrame for all items in a given restaurant showing min/max unit price,
    absolute and percent change, vendors involved, and occurrence counts. Sorted by absolute change desc.
    """
    df = get_invoice_line_items_joined(start_date, end_date, restaurant_ids=[restaurant_id], vendor_ids=vendor_ids)
    if df.empty:
        return pd.DataFrame(columns=["item_name", "category", "min_price", "max_price", "abs_change", "pct_change", "vendor_min", "vendor_max", "occurrences", "vendors"])

    # Drop rows missing critical values
    df = df.dropna(subset=["item_name", "unit_price"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["item_name", "category", "min_price", "max_price", "abs_change", "pct_change", "vendor_min", "vendor_max", "occurrences", "vendors"])

    agg = df.groupby("item_name").agg(
        category=("category", "first"),
        min_price=("unit_price", "min"),
        max_price=("unit_price", "max"),
        mean_price=("unit_price", "mean"),
        occurrences=("invoice_id", "nunique"),
        vendor_count=("vendor", "nunique")
    ).reset_index()

    # vendor with min and max price
    try:
        idx_min = df.groupby("item_name")["unit_price"].idxmin()
        idx_max = df.groupby("item_name")["unit_price"].idxmax()
        vendor_min = df.loc[idx_min, ["item_name", "vendor"]].set_index("item_name")["vendor"]
        vendor_max = df.loc[idx_max, ["item_name", "vendor"]].set_index("item_name")["vendor"]
    except Exception:
        vendor_min = pd.Series()
        vendor_max = pd.Series()

    agg["vendor_min"] = agg["item_name"].map(lambda x: str(vendor_min.get(x, "")))
    agg["vendor_max"] = agg["item_name"].map(lambda x: str(vendor_max.get(x, "")))

    agg["abs_change"] = agg["max_price"] - agg["min_price"]
    # percent change relative to min_price where applicable
    def _pct(row):
        if row["min_price"] and row["min_price"] > 0:
            return (row["abs_change"] / row["min_price"]) * 100
        return float("nan")

    agg["pct_change"] = agg.apply(_pct, axis=1)

    # Compute signed change (latest - earliest) and percent relative to earliest
    # We need first and last prices by invoice_date
    df_sorted = df.sort_values("invoice_date")
    first_price = df_sorted.groupby("item_name")["unit_price"].first()
    last_price = df_sorted.groupby("item_name")["unit_price"].last()
    agg["first_price"] = agg["item_name"].map(lambda x: float(first_price.get(x, np.nan)))
    agg["last_price"] = agg["item_name"].map(lambda x: float(last_price.get(x, np.nan)))
    agg["signed_change"] = agg["last_price"] - agg["first_price"]
    def _signed_pct(row):
        if pd.notna(row["first_price"]) and row["first_price"] and row["first_price"] > 0:
            return (row["signed_change"] / row["first_price"]) * 100
        return float("nan")
    agg["signed_pct_change"] = agg.apply(_signed_pct, axis=1)

    # vendors list
    vendors_series = df.groupby("item_name")["vendor"].unique().apply(lambda x: ", ".join([str(i) for i in x]))
    agg["vendors"] = agg["item_name"].map(lambda x: vendors_series.get(x, ""))

    # Filter by occurrence threshold
    agg = agg[agg["occurrences"] >= min_occurrences]
    # Sorting
    agg = agg.sort_values("abs_change", ascending=False).reset_index(drop=True)
    # Clean columns - include signed change and first/last prices for callers
    return agg[["item_name", "category", "min_price", "max_price", "first_price", "last_price", "abs_change", "pct_change", "signed_change", "signed_pct_change", "vendor_min", "vendor_max", "vendors", "occurrences"]]

def get_item_price_timeseries(restaurant_id: str, item_name: str, start_date=None, end_date=None, vendor_ids=None):
    """Return price history rows for a specific item name at a restaurant."""
    df = get_invoice_line_items_joined(start_date, end_date, restaurant_ids=[restaurant_id], vendor_ids=vendor_ids)
    if df.empty: return pd.DataFrame(columns=["date", "invoice_id", "invoice_number", "vendor", "unit_price", "quantity", "category"])
    sub = df[df["item_name"] == item_name].copy()
    if sub.empty: return pd.DataFrame(columns=["date", "invoice_id", "invoice_number", "vendor", "unit_price", "quantity", "category"])
    sub = sub.sort_values("invoice_date")
    out = pd.DataFrame()
    out["date"] = sub["invoice_date"]
    out["invoice_id"] = sub["invoice_id"]
    out["invoice_number"] = sub.get("invoice_number", "")
    out["vendor"] = sub["vendor"]
    out["unit_price"] = sub["unit_price"]
    out["quantity"] = sub["quantity"]
    out["category"] = sub["category"]
    return out.reset_index(drop=True)

def get_descriptions_for_restaurant(restaurant_id: str, start_date=None, end_date=None):
    df = get_invoice_line_items_joined(start_date, end_date, restaurant_ids=[restaurant_id])
    if df.empty: return []
    return sorted(df["item_name"].dropna().unique().tolist())

def get_recent_invoices(limit=10, restaurant_ids=None):
    inv = _read_table("invoices")
    if inv.empty: return pd.DataFrame(columns=["invoice_number", "invoice_date", "vendor", "location", "total_amount"])
    inv["invoice_date"] = pd.to_datetime(inv["invoice_date"])
    
    rest = _read_table("restaurants")
    vend = _read_table("vendors")
    
    m = pd.merge(inv, vend, left_on="vendor_id", right_on="_id", suffixes=('', '_v'), how="left")
    m = pd.merge(m, rest, left_on="restaurant_id", right_on="_id", suffixes=('', '_r'), how="left")
    
    final = pd.DataFrame()
    final["invoice_number"] = m["invoice_number"]
    final["invoice_date"] = m["invoice_date"]
    final["vendor"] = m["name"]
    final["location"] = m["location_name"]
    final["total_amount"] = m["invoice_total_amount"].apply(_get_float)
    return final.sort_values("invoice_date", ascending=False).head(limit)