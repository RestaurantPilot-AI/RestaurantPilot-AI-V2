import csv
import datetime
from pathlib import Path
from dotenv import load_dotenv

# 1. Load .env from the Root Directory
# Path: src/storage/db_init.py -> src/storage -> src -> root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
env_path = ROOT_DIR / '.env'
load_dotenv(dotenv_path=env_path)

# 2. Setup Database Path
# We explicitly point to 'data/database' to match your file tree
DB_PATH = ROOT_DIR / "data" / "database"

def start_connection(create_dummy=False):
    """
    Connects to the CSV 'Database' (Folder).
    Returns: restaurant_id (if create_dummy=True), else None.
    If create_dummy is True, it ensures a dummy restaurant exists in restaurants.csv.
    """
    try:
        # Check if "Database" (Folder) exists
        if DB_PATH.exists():
            print(f"[INFO] Database Folder '{DB_PATH}' exists. Connected.")
        else:
            DB_PATH.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Database Folder '{DB_PATH}' created. Connected.")

        # Ensure all expected table files exist and have headers
        try:
            create_validation_rules(DB_PATH)
            create_indexes(DB_PATH)
        except Exception as e:
            print(f"[WARNING] Validation/index creation failed during start: {e}")

        restaurant_id = None

        if create_dummy:
            # Define headers for restaurants to ensure file structure exists
            headers = ["_id", "name", "location_name", "phone_number", "restaurant_type", "address", "created_at", "is_active"]
            file_path = DB_PATH / "restaurants.csv"
            
            # Create file if it doesn't exist
            if not file_path.exists():
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)

            # Create dummy data dictionary
            dummy_data = {
                "name": "Westman's Bagel & Coffee - Capitol Hill",
                "phone_number": "(206) 000-0000",
                "restaurant_type": "Bagel shop",
                "address": "1509 E Madison St, Seattle, WA 98122",
                "created_at": datetime.datetime.now().isoformat(),
                "is_active": "True"
            }

            # Check if it already exists to prevent duplicates
            found = False
            existing_id = None
            
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["name"] == dummy_data["name"]:
                        found = True
                        existing_id = row["_id"]
                        break
            
            if found:
                restaurant_id = existing_id
                print(f"[INFO] Found existing dummy restaurant ID: {restaurant_id}")
            else:
                # Generate a new ID (UUID to mimic ObjectId)
                new_id = 1 # str(uuid.uuid4())
                row_to_write = [
                    new_id,
                    dummy_data["name"],
                    "Capitol Hill", # location_name default
                    dummy_data["phone_number"],
                    dummy_data["restaurant_type"],
                    dummy_data["address"],
                    dummy_data["created_at"],
                    dummy_data["is_active"]
                ]
                
                with open(file_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(row_to_write)
                
                restaurant_id = new_id
                print(f"[INFO] Created new dummy restaurant ID: {restaurant_id}")
            
            return restaurant_id

        else:
            return None # return nothing if create_dummy=False 

    except Exception as e:
        print(f"[ERROR] Could not connect to CSV Database: {e}")
        return None

def create_validation_rules(db):
    """
    Creates CSV files with correct Headers (Schema) if they don't exist.
    'db' argument is expected to be the Path object to the folder.
    """
    
    # Schema Definition (Matching the SQL columns exactly)
    schemas = {
        "restaurants.csv": [
            "_id", "name", "location_name", "phone_number", "restaurant_type", "address", "created_at", "is_active"
        ],
        "vendors.csv": [
            "_id", "name", "contact_email", "phone_number", "address", "website"
        ],
        "vendor_regex_templates.csv": [
            "vendor_id", "regex_patterns"
        ],
        "invoices.csv": [
            "_id", "filename", "restaurant_id", "vendor_id", "invoice_number", "invoice_date", 
            "invoice_total_amount", "text_length", "page_count", "extraction_timestamp", "order_date"
        ],
        "line_items.csv": [
            "_id", "invoice_id", "vendor_name", "category", "quantity", "unit", 
            "description", "unit_price", "line_total", "line_number"
        ],
        "item_lookup_map.csv": [
            "_id", "category"
        ],
        "categories.csv": [
            "_id"
        ],
        # Menu-related tables
        "menu_items.csv": [
            "_id", "restaurant_id", "menu_item", "price", "category"
        ],
        "menu_item_lookup_map.csv": [
            "_id", "category"
        ],
        "menu_categories.csv": [
            "_id"
        ],
        "temp_uploads.csv": [
            "session_id", "created_at", "updated_at", "data"
        ],
        "buildsheet_items.csv": [
            "restaurant_id", "name", "yield_quantity", "estimated_price", "ingredients"
        ],
        "buildsheet_regex_templates.csv": [
            "restaurant_id", "regex_patterns"
        ]
    }

    for filename, headers in schemas.items():
        file_path = db / filename
        try:
            if not file_path.exists():
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                print(f"[CREATED] Table (File): {filename}")
            else:
                # Optional: Validate existing headers
                with open(file_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    try:
                        existing_headers = next(reader)
                        if existing_headers == headers:
                            # print(f"[UPDATED] Validator (Headers OK): {filename}")
                            pass
                        else:
                            print(f"[WARNING] Header Mismatch in {filename}. Expected {headers}")
                    except StopIteration:
                        # File is empty, write headers
                        pass
        except Exception as e:
             print(f"[ERROR] Update failed for {filename}: {e}")

def create_indexes(db):
    """
    Simulates applying unique constraints and performance indexes.
    For CSVs, this is mostly a placeholder or a check for file integrity.
    """
    print("[INFO] Checking Indexes (File Integrity)...")
    
    # In a CSV system, "Indexes" don't exist physically.
    # We verify that we can access the files that would represent these collections.
    
    required_files = [
        "restaurants.csv", "vendors.csv", "vendor_regex_templates.csv",
        "invoices.csv", "line_items.csv", "item_lookup_map.csv", 
        "menu_items.csv", "menu_item_lookup_map.csv", "menu_categories.csv", 
        "buildsheet_items.csv", "buildsheet_regex_templates.csv",
        "temp_uploads.csv"
    ]
    
    missing_files = []
    for fname in required_files:
        if not (db / fname).exists():
            missing_files.append(fname)
            
    if not missing_files:
        print("[SUCCESS] Indexes verified (All required files present).")
    else:
        print(f"[FAIL] Missing files for indexing: {missing_files}")

if __name__ == "__main__":
    # We unpack the tuple here since we updated the return signature
    # (Keeping comments from original file for structure)
    result = start_connection(True)
    
    if result is not None:
        # When create_dummy=False, start_connection returns None.
        # Logic adapted to connect to "File DB" directly.
        try:
            # Mimic the "client = MongoClient(URI)" step by setting the DB path
            db = DB_PATH
            
            create_validation_rules(db)
            create_indexes(db)
            print("[FINISH] Database setup complete.")
        except Exception as e:
            print(f"[ERROR] Setup failed: {e}") 