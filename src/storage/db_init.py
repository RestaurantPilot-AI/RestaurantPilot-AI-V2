import os
import datetime
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import CollectionInvalid

# 1. Load .env from the Root Directory
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Get variables
URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "invoice_processing_db")

def start_connection(create_dummy=False):
    """
    Connects to MongoDB.
    Returns: (db_object, restaurant_id)
    If create_dummy is True, it ensures a dummy restaurant exists and returns its ID.
    """
    try:
        client = MongoClient(URI)
        client.admin.command('ping') # Check connection
        
        existing_dbs = client.list_database_names()
        if DB_NAME in existing_dbs:
            print(f"[INFO] Database '{DB_NAME}' exists. Connected.")
        else:
            print(f"[INFO] Database '{DB_NAME}' created (virtual). Connected.")

        db = client[DB_NAME]
        restaurant_id = None

        if create_dummy:
            # Create dummy data that satisfies all schema constraints
            dummy_data = {
                "name": "Westman's Bagel & Coffee - Capitol Hill",
                "phone_number": "(206) 000-0000",
                "restaurant_type": "Bagel shop",
                "address": "1509 E Madison St, Seattle, WA 98122",
                "created_at": datetime.datetime.now(),
                "is_active": True
            }

            # Check if it already exists to prevent duplicates
            existing_rest = db.restaurants.find_one({"name": dummy_data["name"]})
            
            if existing_rest:
                restaurant_id = existing_rest["_id"]
                print(f"[INFO] Found existing dummy restaurant ID: {restaurant_id}")
            else:
                result = db.restaurants.insert_one(dummy_data)
                restaurant_id = result.inserted_id
                print(f"[INFO] Created new dummy restaurant ID: {restaurant_id}")
            
            return restaurant_id

        else:
            return None# return nothing if create_dummy=False 

    except Exception as e:
        print(f"[ERROR] Could not connect to MongoDB: {e}")
        return None

def create_validation_rules(db):
    """Creates collections with JSON Schema Validation based on the new schema."""
    
    # 1. RESTAURANTS
    restaurant_validator = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["name", "created_at", "is_active"],
            "properties": {
                "name": {"bsonType": "string"},
                "location_name": {"bsonType": "string"},
                "phone_number": {"bsonType": "string"},
                "restaurant_type": {"bsonType": "string"},
                "address": {"bsonType": "string"},
                "created_at": {"bsonType": "date"},
                "is_active": {"bsonType": "bool"}
            }
        }
    }

    # 2. VENDORS
    vendor_validator = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["name"],
            "properties": {
                "name": {"bsonType": "string"},
                "contact_email": {"bsonType": "string"},
                "phone_number": {"bsonType": "string"},
                "address": {"bsonType": "string"},
                "website": {"bsonType": "string"}
            }
        }
    }

    # 3. VENDOR REGEX TEMPLATES
    regex_validator = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["vendor_id", "regex_patterns"],
            "properties": {
                "vendor_id": {"bsonType": "objectId"},
                "regex_patterns": {
                    "bsonType": "array",
                    "items": {"bsonType": "string"} # Strict list of strings
                }
            }
        }
    }

    # 4. INVOICES (Updated Fields & Removed embedded line_items)
    invoice_validator = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [
                "filename", "restaurant_id", "vendor_id", "invoice_number", 
                "invoice_date", "invoice_total_amount", "extraction_timestamp", "order_date"
            ],
            "properties": {
                "filename": {"bsonType": "string"},
                "restaurant_id": {"bsonType": "objectId"},
                "vendor_id": {"bsonType": "objectId"},
                "invoice_number": {"bsonType": "string"},
                "invoice_date": {"bsonType": "date"},
                "invoice_total_amount": {"bsonType": "decimal"},
                "text_length": {"bsonType": "int"},
                "page_count": {"bsonType": "int"},
                "extraction_timestamp": {"bsonType": "date"},
                "order_date": {"bsonType": "date"}
            }
        }
    }

    # 5. LINE ITEMS (New Collection)
    # Note: invoice_id is set to 'int' to match your spec, though typically this links to invoices._id (ObjectId)
    line_item_validator = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [
                "invoice_id", "vendor_name", "category", "quantity", 
                "unit", "description", "unit_price", "line_total", "line_number"
            ],
            "properties": {
                "invoice_id": {"bsonType": "objectId"}, 
                "vendor_name": {"bsonType": "string"},
                "category": {"bsonType": "string"},
                "quantity": {"bsonType": "double"},
                "unit": {"bsonType": "string"},
                "description": {"bsonType": "string"},
                "unit_price": {"bsonType": "decimal"},
                "line_total": {"bsonType": "decimal"},
                "line_number": {"bsonType": "decimal"}
            }
        }
    }

    # 6. ITEM LOOKUP MAP (New Collection)
    # _id is the "Normalized description" (string)
    lookup_validator = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["category"],
            "properties": {
                "_id": {"bsonType": "string"}, 
                "category": {"bsonType": "string"}
            }
        }
    }

    # 7. CATEGORIES (New Collection)
    # _id is the "Category Name" (string)
    category_validator = {
        "$jsonSchema": {
            "bsonType": "object",
            "properties": {
                "_id": {"bsonType": "string"} 
            }
        }
    }

    # 8. TEMP_UPLOADS (Session Persistence)
    temp_upload_validator = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["session_id", "created_at"],
            "properties": {
                "session_id": {"bsonType": "string"},
                "created_at": {"bsonType": "date"},
                "updated_at": {"bsonType": "date"}
            }
        }
    }

    # 9. SALES (Daily Sales Tracking)
    sales_validator = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["date", "restaurant_id", "revenue", "covers", "created_at"],
            "properties": {
                "date": {"bsonType": "date"},
                "restaurant_id": {"bsonType": "objectId"},
                "revenue": {"bsonType": "double"},
                "covers": {"bsonType": "int"},
                "created_at": {"bsonType": "date"}
            }
        }
    }

    collections = {
        "restaurants": restaurant_validator,
        "vendors": vendor_validator,
        "vendor_regex_templates": regex_validator,
        "invoices": invoice_validator,
        "line_items": line_item_validator,
        "item_lookup_map": lookup_validator,
        "categories": category_validator,
        "temp_uploads": temp_upload_validator,
        "sales": sales_validator
    }

    for name, validator in collections.items():
        try:
            db.create_collection(name, validator=validator)
            print(f"[CREATED] Collection: {name}")
        except CollectionInvalid:
            try:
                db.command("collMod", name, validator=validator)
                print(f"[UPDATED] Validator: {name}")
            except Exception as e:
                print(f"[ERROR] Update failed for {name}: {e}")

def create_indexes(db):
    """Applies unique constraints and performance indexes."""
    print("[INFO] Checking Indexes...")
    
    # 1. Restaurants
    # No unique constraints specified other than _id, but we index commonly queried fields
    db.restaurants.create_index([("name", ASCENDING)])

    # 2. Vendors (Unique constraints as per description)
    db.vendors.create_index([("name", ASCENDING)], unique=True)
    # Sparse indexes allow multiple "null" values but enforce uniqueness if the value exists
    db.vendors.create_index([("contact_email", ASCENDING)], unique=True, sparse=True)
    db.vendors.create_index([("phone_number", ASCENDING)], unique=True, sparse=True)
    db.vendors.create_index([("address", ASCENDING)], unique=True, sparse=True)
    db.vendors.create_index([("website", ASCENDING)], unique=True, sparse=True)

    # 3. Vendor Regex Templates
    db.vendor_regex_templates.create_index([("vendor_id", ASCENDING)])

    # 4. Invoices
    # Compound unique index to prevent duplicate invoice uploads for the same vendor
    db.invoices.create_index([("vendor_id", ASCENDING), ("invoice_number", ASCENDING)], unique=True)

    # 8. Sales (Daily Sales Tracking)
    # Compound unique index to prevent duplicate sales entries per restaurant per day
    db.sales.create_index([("restaurant_id", ASCENDING), ("date", ASCENDING)], unique=True)
    # Compound index for dashboard date range queries
    db.sales.create_index([("restaurant_id", ASCENDING), ("date", DESCENDING)])

    # Sorting index for UI
    db.invoices.create_index([("restaurant_id", ASCENDING), ("invoice_date", DESCENDING)])

    # 5. Line Items
    db.line_items.create_index([("invoice_id", ASCENDING)])
    db.line_items.create_index([("category", ASCENDING)])

    # 6. Item Lookup Map
    # _id is already indexed by default, but we might want to query by category
    db.item_lookup_map.create_index([("category", ASCENDING)])

    # 7. Temp Uploads (Session Persistence)
    db.temp_uploads.create_index([("session_id", ASCENDING)], unique=True)
    # TTL index: auto-delete temp uploads after 7 days
    db.temp_uploads.create_index([("created_at", ASCENDING)], expireAfterSeconds=604800)

    print("[SUCCESS] Indexes verified.")

if __name__ == "__main__":
    # We unpack the tuple here since we updated the return signature
    result = start_connection()
    
    if result is not None:
        # When create_dummy=False, start_connection returns None, not a db object
        # Let's connect directly here
        try:
            from pymongo import MongoClient
            client = MongoClient(URI)
            db = client[DB_NAME]
            
            create_validation_rules(db)
            create_indexes(db)
            print("[FINISH] Database setup complete.")
        except Exception as e:
            print(f"[ERROR] Setup failed: {e}") 