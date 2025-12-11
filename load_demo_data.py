"""
Demo Data Loader Script
Loads sample CSV data into MongoDB for dashboard demonstration.

This script:
1. Creates restaurants (Main, Downtown, Waterfront)
2. Creates vendors (Sysco, US Foods, Local Farm Co., Fresh Dairy Ltd)
3. Creates product categories (Proteins, Produce, Dairy, Dry Goods, Beverages)
4. Loads invoices with line items from sample_dashboard_data.csv
5. Loads daily sales data from sample_sales_data.csv

Run this script once to populate the database with demo data.
"""

import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
from bson.decimal128 import Decimal128

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

# Load environment
load_dotenv()
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "invoice_processing_db")


def get_db():
    """Connect to MongoDB."""
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]


def clear_demo_collections(db):
    """Clear existing demo data (optional - comment out if you want to preserve data)."""
    print("\n[CLEARING] Removing existing demo data...")
    
    # Get demo restaurant IDs to cascade delete
    demo_locations = ["Main", "Downtown", "Waterfront"]
    demo_restaurants = list(db.restaurants.find({"location_name": {"$in": demo_locations}}))
    demo_restaurant_ids = [r["_id"] for r in demo_restaurants]
    
    # Get demo vendor IDs
    demo_vendor_names = ["Sysco", "US Foods", "Local Farm Co.", "Fresh Dairy Ltd"]
    demo_vendors = list(db.vendors.find({"name": {"$in": demo_vendor_names}}))
    demo_vendor_ids = [v["_id"] for v in demo_vendors]
    
    if demo_restaurant_ids:
        # Delete related sales
        sales_result = db.sales.delete_many({"restaurant_id": {"$in": demo_restaurant_ids}})
        print(f"  - Deleted {sales_result.deleted_count} sales records")
        
        # Delete related invoices and line items
        invoices = list(db.invoices.find({"restaurant_id": {"$in": demo_restaurant_ids}}))
        invoice_ids = [inv["_id"] for inv in invoices]
        
        if invoice_ids:
            line_items_result = db.line_items.delete_many({"invoice_id": {"$in": invoice_ids}})
            print(f"  - Deleted {line_items_result.deleted_count} line items")
            
            invoices_result = db.invoices.delete_many({"_id": {"$in": invoice_ids}})
            print(f"  - Deleted {invoices_result.deleted_count} invoices")
    
    # Delete demo vendors and regex templates
    if demo_vendor_ids:
        db.vendor_regex_templates.delete_many({"vendor_id": {"$in": demo_vendor_ids}})
        vendors_result = db.vendors.delete_many({"_id": {"$in": demo_vendor_ids}})
        print(f"  - Deleted {vendors_result.deleted_count} vendors")
    
    # Delete demo restaurants
    if demo_restaurant_ids:
        restaurants_result = db.restaurants.delete_many({"_id": {"$in": demo_restaurant_ids}})
        print(f"  - Deleted {restaurants_result.deleted_count} restaurants")
    
    print("[CLEARED] Demo data removed.\n")


def create_restaurants(db):
    """Create demo restaurant locations."""
    print("[CREATING] Restaurants...")
    
    restaurants_data = [
        {
            "name": "Westman's Bagel & Coffee - Main Street",
            "location_name": "Main",
            "phone_number": "(206) 555-0101",
            "restaurant_type": "Bagel Shop",
            "address": "1509 E Madison St, Seattle, WA 98122",
            "created_at": datetime.now(),
            "is_active": True
        },
        {
            "name": "Westman's Bagel & Coffee - Downtown",
            "location_name": "Downtown",
            "phone_number": "(206) 555-0102",
            "restaurant_type": "Bagel Shop",
            "address": "300 Pike St, Seattle, WA 98101",
            "created_at": datetime.now(),
            "is_active": True
        },
        {
            "name": "Westman's Bagel & Coffee - Waterfront",
            "location_name": "Waterfront",
            "phone_number": "(206) 555-0103",
            "restaurant_type": "Bagel Shop",
            "address": "1001 Alaskan Way, Seattle, WA 98104",
            "created_at": datetime.now(),
            "is_active": True
        }
    ]
    
    restaurant_map = {}
    for rest in restaurants_data:
        result = db.restaurants.insert_one(rest)
        restaurant_map[rest["location_name"]] = result.inserted_id
        print(f"  ✓ Created: {rest['location_name']} (ID: {result.inserted_id})")
    
    return restaurant_map


def create_vendors(db):
    """Create demo vendors."""
    print("\n[CREATING] Vendors...")
    
    vendors_data = [
        {
            "name": "Sysco",
            "contact_email": "orders@sysco.com",
            "phone_number": "(800) 555-7000",
            "address": "1234 Industrial Blvd, Seattle, WA 98134",
            "website": "www.sysco.com"
        },
        {
            "name": "US Foods",
            "contact_email": "sales@usfoods.com",
            "phone_number": "(800) 555-8000",
            "address": "5678 Commerce Dr, Seattle, WA 98108",
            "website": "www.usfoods.com"
        },
        {
            "name": "Local Farm Co.",
            "contact_email": "info@localfarmco.com",
            "phone_number": "(206) 555-9000",
            "address": "100 Farm Rd, Woodinville, WA 98072",
            "website": "www.localfarmco.com"
        },
        {
            "name": "Fresh Dairy Ltd",
            "contact_email": "orders@freshdairy.com",
            "phone_number": "(425) 555-6000",
            "address": "789 Dairy Ln, Bellevue, WA 98004",
            "website": "www.freshdairy.com"
        }
    ]
    
    vendor_map = {}
    for vendor in vendors_data:
        result = db.vendors.insert_one(vendor)
        vendor_map[vendor["name"]] = result.inserted_id
        print(f"  ✓ Created: {vendor['name']} (ID: {result.inserted_id})")
    
    return vendor_map


def create_categories(db):
    """Create product categories."""
    print("\n[CREATING] Categories...")
    
    categories = ["Proteins", "Produce", "Dairy", "Dry Goods", "Beverages"]
    
    for category in categories:
        db.categories.update_one(
            {"_id": category},
            {"$setOnInsert": {"_id": category}},
            upsert=True
        )
        print(f"  ✓ Created/Updated: {category}")
    
    return categories


def load_invoices_and_line_items(db, restaurant_map, vendor_map, csv_path):
    """Load invoices with line items from CSV."""
    print(f"\n[LOADING] Invoices from {csv_path}...")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    
    # Group by invoice
    invoices_created = 0
    line_items_created = 0
    
    for invoice_id, group in df.groupby('invoice_id'):
        # Get first row for invoice-level data
        first_row = group.iloc[0]
        
        # Map location and vendor to IDs
        location = first_row['location']
        vendor_name = first_row['vendor']
        
        if location not in restaurant_map:
            print(f"  ⚠ Warning: Unknown location '{location}' - skipping invoice {invoice_id}")
            continue
        
        if vendor_name not in vendor_map:
            print(f"  ⚠ Warning: Unknown vendor '{vendor_name}' - skipping invoice {invoice_id}")
            continue
        
        restaurant_id = restaurant_map[location]
        vendor_id = vendor_map[vendor_name]
        
        # Calculate invoice total
        invoice_total = group['line_total'].sum()
        
        # Create invoice document
        invoice_doc = {
            "filename": f"demo_invoice_{invoice_id}.pdf",
            "restaurant_id": restaurant_id,
            "vendor_id": vendor_id,
            "invoice_number": f"INV-{invoice_id}",
            "invoice_date": first_row['invoice_date'],
            "invoice_total_amount": Decimal128(str(round(invoice_total, 2))),
            "text_length": 1000,
            "page_count": 1,
            "extraction_timestamp": datetime.now(),
            "order_date": first_row['invoice_date']
        }
        
        # Insert invoice
        try:
            invoice_result = db.invoices.insert_one(invoice_doc)
            invoices_created += 1
            
            # Create line items
            line_items = []
            for idx, row in group.iterrows():
                line_item = {
                    "invoice_id": invoice_result.inserted_id,
                    "vendor_name": vendor_name,
                    "category": row['category'],
                    "quantity": float(row['quantity']),
                    "unit": "ea",  # Default unit
                    "description": row['item_name'],
                    "unit_price": Decimal128(str(round(row['unit_price'], 2))),
                    "line_total": Decimal128(str(round(row['line_total'], 2))),
                    "line_number": Decimal128(str(len(line_items) + 1))
                }
                line_items.append(line_item)
            
            # Insert line items
            if line_items:
                db.line_items.insert_many(line_items)
                line_items_created += len(line_items)
            
        except Exception as e:
            print(f"  ⚠ Error creating invoice {invoice_id}: {e}")
    
    print(f"  ✓ Created {invoices_created} invoices with {line_items_created} line items")
    return invoices_created, line_items_created


def load_sales_data(db, restaurant_map, csv_path):
    """Load daily sales data from CSV."""
    print(f"\n[LOADING] Sales data from {csv_path}...")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Create sales records
    sales_records = []
    for _, row in df.iterrows():
        location = row['location']
        
        if location not in restaurant_map:
            print(f"  ⚠ Warning: Unknown location '{location}' - skipping")
            continue
        
        sales_record = {
            "date": row['date'],
            "restaurant_id": restaurant_map[location],
            "revenue": float(row['revenue']),
            "covers": int(row['covers']),
            "created_at": datetime.now()
        }
        sales_records.append(sales_record)
    
    # Insert sales data
    if sales_records:
        try:
            db.sales.insert_many(sales_records)
            print(f"  ✓ Created {len(sales_records)} sales records")
            return len(sales_records)
        except Exception as e:
            print(f"  ⚠ Error loading sales data: {e}")
            return 0
    
    return 0


def main():
    """Main execution function."""
    print("=" * 60)
    print("DEMO DATA LOADER FOR INVOICE AUTOMATION")
    print("=" * 60)
    
    # Connect to database
    db = get_db()
    print(f"\n[CONNECTED] Database: {DB_NAME}")
    
    # Optional: Clear existing demo data (comment out if not needed)
    clear_demo_collections(db)
    
    # Create base data
    restaurant_map = create_restaurants(db)
    vendor_map = create_vendors(db)
    create_categories(db)
    
    # Define CSV paths
    base_path = Path(__file__).parent / "data"
    invoices_csv = base_path / "sample_dashboard_data.csv"
    sales_csv = base_path / "sample_sales_data.csv"
    
    # Load transactional data
    if invoices_csv.exists():
        load_invoices_and_line_items(db, restaurant_map, vendor_map, invoices_csv)
    else:
        print(f"\n⚠ Warning: Invoice CSV not found at {invoices_csv}")
    
    if sales_csv.exists():
        load_sales_data(db, restaurant_map, sales_csv)
    else:
        print(f"\n⚠ Warning: Sales CSV not found at {sales_csv}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)
    print(f"Restaurants: {db.restaurants.count_documents({})}")
    print(f"Vendors: {db.vendors.count_documents({})}")
    print(f"Categories: {db.categories.count_documents({})}")
    print(f"Invoices: {db.invoices.count_documents({})}")
    print(f"Line Items: {db.line_items.count_documents({})}")
    print(f"Sales Records: {db.sales.count_documents({})}")
    print("=" * 60)
    print("\n✅ DEMO DATA LOADING COMPLETE!\n")


if __name__ == "__main__":
    main()
