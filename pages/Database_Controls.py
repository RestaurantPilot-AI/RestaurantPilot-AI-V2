import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
from bson import ObjectId
from bson.decimal128 import Decimal128

from src.storage.database import (
    db,
    cleanup_old_temp_uploads,
    create_vendor,
    get_all_vendors,
    save_vendor_regex_template,
    get_vendor_regex_patterns,
    insert_master_category,
    get_all_category_names
)

st.set_page_config(page_title="Database Controls", page_icon="🔧", layout="wide")

st.title("🔧 Database Administration")
st.markdown("System maintenance, vendor management, category management, and bulk operations")

# Initialize session state
if "admin_tab" not in st.session_state:
    st.session_state.admin_tab = "Maintenance"

# Tab selection
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧹 Maintenance", 
    "👥 Vendor Management", 
    "🏷️ Category Management", 
    "📦 Bulk Operations",
    "📊 Database Browser"
])

# TAB 1: MAINTENANCE
with tab1:
    st.header("🧹 Database Maintenance")
    
    st.markdown("### 📊 Database Statistics")
    
    try:
        # Get collection counts for all 11 collections
        invoices_count = db.invoices.count_documents({})
        line_items_count = db.line_items.count_documents({})
        vendors_count = db.vendors.count_documents({})
        restaurants_count = db.restaurants.count_documents({})
        categories_count = db.categories.count_documents({})
        temp_uploads_count = db.temp_uploads.count_documents({})
        vendor_regex_count = db.vendor_regex_templates.count_documents({})
        item_lookup_count = db.item_lookup_map.count_documents({})
        menu_items_count = db.menu_items.count_documents({})
        menu_categories_count = db.menu_categories.count_documents({})
        menu_lookup_count = db.menu_item_lookup_map.count_documents({})
        
        # Row 1: Core invoice data
        col1, col2, col3 = st.columns(3)
        col1.metric("📄 Invoices", f"{invoices_count:,}")
        col2.metric("📦 Line Items", f"{line_items_count:,}")
        col3.metric("👥 Vendors", f"{vendors_count:,}")
        
        # Row 2: Restaurant and category data
        col4, col5, col6 = st.columns(3)
        col4.metric("🏢 Restaurants", f"{restaurants_count:,}")
        col5.metric("🏷️ Categories", f"{categories_count:,}")
        col6.metric("⏳ Temp Uploads", f"{temp_uploads_count:,}")
        
        # Row 3: Menu data
        col7, col8, col9 = st.columns(3)
        col7.metric("🍽️ Menu Items", f"{menu_items_count:,}")
        col8.metric("📂 Menu Categories", f"{menu_categories_count:,}")
        col9.metric("🔍 Menu Lookups", f"{menu_lookup_count:,}")
        
        # Row 4: Advanced/lookup collections
        col10, col11, col12 = st.columns(3)
        col10.metric("🔧 Vendor Regex", f"{vendor_regex_count:,}")
        col11.metric("🗂️ Item Lookups", f"{item_lookup_count:,}")
        col12.metric("💾 Total Collections", "11")
        
    except Exception as e:
        st.error(f"Error fetching statistics: {str(e)}")
    
    st.divider()
    
    # Cleanup operations
    st.markdown("### 🧹 Cleanup Operations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Clean Temporary Uploads")
        st.markdown("Remove old temporary upload sessions from the database")
        
        days_to_keep = st.number_input(
            "Keep uploads from last (days)",
            min_value=1,
            max_value=365,
            value=7,
            help="Delete temporary uploads older than this many days"
        )
        
        if st.button("🗑️ Clean Old Temp Uploads", type="primary", width='stretch'):
            try:
                with st.spinner("Cleaning temporary uploads..."):
                    deleted_count = cleanup_old_temp_uploads(days=days_to_keep)
                    st.success(f"✅ Cleaned {deleted_count} temporary upload(s) older than {days_to_keep} days")
            except Exception as e:
                st.error(f"Error during cleanup: {str(e)}")
    
    with col2:
        st.markdown("#### Database Info")
        st.markdown("View database connection and storage details")
        
        if st.button("📊 Show Database Info", width='stretch'):
            try:
                # Get database stats
                db_stats = db.command("dbStats")
                
                st.json({
                    "Database Name": db.name,
                    "Collections": db_stats.get("collections", "N/A"),
                    "Data Size": f"{db_stats.get('dataSize', 0) / 1024 / 1024:.2f} MB",
                    "Storage Size": f"{db_stats.get('storageSize', 0) / 1024 / 1024:.2f} MB",
                    "Indexes": db_stats.get("indexes", "N/A"),
                    "Index Size": f"{db_stats.get('indexSize', 0) / 1024 / 1024:.2f} MB"
                })
            except Exception as e:
                st.error(f"Error fetching database info: {str(e)}")

# TAB 2: VENDOR MANAGEMENT
with tab2:
    st.header("👥 Vendor Management")
    
    # List existing vendors
    st.markdown("### 📋 Existing Vendors")
    
    try:
        vendors = get_all_vendors()
        
        if vendors:
            # Create display dataframe
            vendor_data = []
            for vendor in vendors:
                # Get additional vendor details
                full_vendor = db.vendors.find_one({"_id": vendor["_id"]})
                if full_vendor:
                    vendor_data.append({
                        "Name": vendor["name"],
                        "Email": full_vendor.get("email", ""),
                        "Phone": full_vendor.get("phone", ""),
                        "Website": full_vendor.get("website", ""),
                        "Active": "✅" if full_vendor.get("is_active", True) else "❌",
                        "_id": str(vendor["_id"])
                    })
            
            vendor_df = pd.DataFrame(vendor_data)
            
            # Display vendors
            st.dataframe(
                vendor_df.drop(columns=["_id"]),
                width='stretch',
                hide_index=True
            )
            
            st.info(f"📊 Total Vendors: {len(vendors)}")
        else:
            st.info("No vendors found in the database.")
    
    except Exception as e:
        st.error(f"Error loading vendors: {str(e)}")
    
    st.divider()
    
    # Add new vendor
    st.markdown("### ➕ Add New Vendor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_vendor_name = st.text_input("Vendor Name *", placeholder="e.g., US Foods")
        new_vendor_email = st.text_input("Email", placeholder="contact@vendor.com")
        new_vendor_phone = st.text_input("Phone", placeholder="(555) 123-4567")
    
    with col2:
        new_vendor_website = st.text_input("Website", placeholder="https://vendor.com")
        new_vendor_address = st.text_area("Address", placeholder="123 Main St, City, State ZIP")
        new_vendor_active = st.checkbox("Active Vendor", value=True)
    
    if st.button("💾 Add Vendor", type="primary"):
        if not new_vendor_name:
            st.error("❌ Vendor name is required")
        else:
            try:
                vendor_data = {
                    "name": new_vendor_name,
                    "email": new_vendor_email if new_vendor_email else None,
                    "phone": new_vendor_phone if new_vendor_phone else None,
                    "website": new_vendor_website if new_vendor_website else None,
                    "address": new_vendor_address if new_vendor_address else None,
                    "is_active": new_vendor_active
                }
                
                vendor_id = create_vendor(vendor_data)
                
                if vendor_id:
                    st.success(f"✅ Vendor added successfully! ID: {vendor_id}")
                    st.rerun()
                else:
                    st.error("❌ Error adding vendor: Failed to create vendor")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    # Manage vendor regex templates
    st.markdown("### 🔧 Vendor Regex Templates")
    st.markdown("Configure extraction patterns for specific vendors (advanced)")
    
    if vendors:
        selected_vendor_name = st.selectbox(
            "Select Vendor",
            [""] + [v["name"] for v in vendors],
            help="Choose a vendor to view or edit regex templates"
        )
        
        if selected_vendor_name:
            selected_vendor = next((v for v in vendors if v["name"] == selected_vendor_name), None)
            
            if selected_vendor:
                vendor_id = str(selected_vendor["_id"])
                
                # Load existing patterns
                existing_patterns = get_vendor_regex_patterns(vendor_id)
                
                if existing_patterns:
                    st.info(f"✅ Regex templates exist for {selected_vendor_name}")
                    
                    with st.expander("View Existing Patterns"):
                        pattern_labels = [
                            "Invoice Number",
                            "Invoice Date",
                            "Invoice Total Amount",
                            "Order Date",
                            "Line Item Block Start",
                            "Line Item Block End",
                            "Quantity",
                            "Description",
                            "Unit",
                            "Unit Price",
                            "Line Total"
                        ]
                        
                        for idx, (label, pattern) in enumerate(zip(pattern_labels, existing_patterns)):
                            st.text_input(f"{idx}. {label}", value=pattern, disabled=True, key=f"existing_{idx}")
                else:
                    st.warning(f"⚠️ No regex templates found for {selected_vendor_name}")
                
                st.markdown("**Note:** Regex template management requires technical knowledge. Contact system administrator for pattern configuration.")

# TAB 3: CATEGORY MANAGEMENT
with tab3:
    st.header("🏷️ Category Management")
    
    # List existing categories
    st.markdown("### 📋 Existing Categories")
    
    try:
        # Get all categories from database
        categories = list(db.categories.find({}))
        
        if categories:
            # Create display dataframe
            cat_data = []
            for cat in categories:
                cat_data.append({
                    "Name": cat.get("name", ""),
                    "Type": cat.get("type", ""),
                    "_id": str(cat["_id"])
                })
            
            cat_df = pd.DataFrame(cat_data)
            
            # Check if all types are empty
            has_types = cat_df["Type"].notna().any() and (cat_df["Type"] != "").any()
            
            if has_types:
                # Group by type
                for cat_type in cat_df["Type"].unique():
                    if cat_type:  # Skip empty types
                        with st.expander(f"📁 {cat_type}", expanded=True):
                            type_cats = cat_df[cat_df["Type"] == cat_type].drop(columns=["_id", "Type"])
                            st.dataframe(
                                type_cats,
                                width='stretch',
                                hide_index=True
                            )
                
                # Show uncategorized if any
                uncategorized = cat_df[(cat_df["Type"].isna()) | (cat_df["Type"] == "")]
                if not uncategorized.empty:
                    with st.expander("📁 Uncategorized", expanded=True):
                        st.dataframe(
                            uncategorized.drop(columns=["_id", "Type"]),
                            width='stretch',
                            hide_index=True
                        )
            else:
                # Display all categories in a single table if no types
                st.dataframe(
                    cat_df.drop(columns=["_id"]),
                    width='stretch',
                    hide_index=True
                )
            
            st.info(f"📊 Total Categories: {len(categories)}")
        else:
            st.info("No categories found in the database.")
    
    except Exception as e:
        st.error(f"Error loading categories: {str(e)}")
    
    st.divider()
    
    # Add new category
    st.markdown("### ➕ Add New Category")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_cat_name = st.text_input("Category Name *", placeholder="e.g., Dairy Products")
    
    with col2:
        new_cat_type = st.selectbox(
            "Category Type *",
            ["", "Food", "Beverage", "Supplies", "Equipment", "Service", "Other"],
            help="Select the type of category"
        )
    
    if st.button("💾 Add Category", type="primary"):
        if not new_cat_name or not new_cat_type:
            st.error("❌ Category name and type are required")
        else:
            try:
                # Insert category into database
                cat_doc = {
                    "name": new_cat_name,
                    "type": new_cat_type
                }
                
                result = db.categories.insert_one(cat_doc)
                
                if result.inserted_id:
                    st.success(f"✅ Category added successfully! ID: {result.inserted_id}")
                    # Also add to master category list
                    insert_master_category(new_cat_name)
                    st.rerun()
                else:
                    st.error("❌ Error adding category")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    # Category mapping info
    st.markdown("### 🔗 Category Mapping")
    st.markdown("Line items are automatically categorized during invoice processing using LLM-based categorization.")
    
    try:
        # Show sample of categorized items
        sample_items = list(db.line_items.find({"category": {"$exists": True, "$ne": None}}).limit(10))
        
        if sample_items:
            st.markdown("**Sample Categorized Items:**")
            sample_data = []
            for item in sample_items:
                sample_data.append({
                    "Description": item.get("description", "")[:50],
                    "Category": item.get("category", "Uncategorized")
                })
            
            st.dataframe(pd.DataFrame(sample_data), width='stretch', hide_index=True)
        else:
            st.info("No categorized items found. Categories are assigned during invoice processing.")
    
    except Exception as e:
        st.error(f"Error loading sample: {str(e)}")

# TAB 4: BULK OPERATIONS
with tab4:
    st.header("📦 Bulk Operations")
    
    st.markdown("### 📥 Export Data")
    st.markdown("Export database collections to CSV format")
    
    # Export invoices
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Export Invoices")
        
        export_date_range = st.selectbox(
            "Date Range",
            ["Last 30 Days", "Last 90 Days", "This Year", "All Time"],
            key="export_date"
        )
        
        if st.button("📥 Export Invoices to CSV", width='stretch'):
            try:
                with st.spinner("Preparing export..."):
                    # Build query based on date range
                    query = {}
                    if export_date_range == "Last 30 Days":
                        cutoff = datetime.now() - timedelta(days=30)
                        query["invoice_date"] = {"$gte": cutoff}
                    elif export_date_range == "Last 90 Days":
                        cutoff = datetime.now() - timedelta(days=90)
                        query["invoice_date"] = {"$gte": cutoff}
                    elif export_date_range == "This Year":
                        cutoff = datetime(datetime.now().year, 1, 1)
                        query["invoice_date"] = {"$gte": cutoff}
                    
                    # Fetch invoices
                    invoices = list(db.invoices.find(query))
                    
                    if not invoices:
                        st.warning("No invoices found for the selected date range.")
                    else:
                        # Prepare data
                        export_data = []
                        for inv in invoices:
                            # Get vendor name
                            vendor = db.vendors.find_one({"_id": inv.get("vendor_id")})
                            vendor_name = vendor["name"] if vendor else "Unknown"
                            
                            # Convert Decimal128 to float
                            total = inv.get("invoice_total_amount", 0)
                            if isinstance(total, Decimal128):
                                total = float(total.to_decimal())
                            
                            export_data.append({
                                "Invoice ID": str(inv["_id"]),
                                "Invoice Number": inv.get("invoice_number", ""),
                                "Date": inv.get("invoice_date", "").strftime("%Y-%m-%d") if isinstance(inv.get("invoice_date"), datetime) else "",
                                "Vendor": vendor_name,
                                "Total Amount": total,
                                "Order Number": inv.get("order_number", "")
                            })
                        
                        export_df = pd.DataFrame(export_data)
                        csv = export_df.to_csv(index=False)
                        
                        st.download_button(
                            label=f"💾 Download {len(invoices)} Invoices",
                            data=csv,
                            file_name=f"invoices_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            width='stretch'
                        )
                        
                        st.success(f"✅ Ready to download {len(invoices)} invoices")
            
            except Exception as e:
                st.error(f"Error exporting invoices: {str(e)}")
    
    with col2:
        st.markdown("#### Export Line Items")
        
        export_li_date_range = st.selectbox(
            "Date Range",
            ["Last 30 Days", "Last 90 Days", "This Year", "All Time"],
            key="export_li_date"
        )
        
        if st.button("📥 Export Line Items to CSV", width='stretch'):
            try:
                with st.spinner("Preparing export..."):
                    # Build query for invoices based on date range
                    invoice_query = {}
                    if export_li_date_range == "Last 30 Days":
                        cutoff = datetime.now() - timedelta(days=30)
                        invoice_query["invoice_date"] = {"$gte": cutoff}
                    elif export_li_date_range == "Last 90 Days":
                        cutoff = datetime.now() - timedelta(days=90)
                        invoice_query["invoice_date"] = {"$gte": cutoff}
                    elif export_li_date_range == "This Year":
                        cutoff = datetime(datetime.now().year, 1, 1)
                        invoice_query["invoice_date"] = {"$gte": cutoff}
                    
                    # Get invoice IDs
                    invoices = list(db.invoices.find(invoice_query, {"_id": 1, "invoice_number": 1, "vendor_id": 1}))
                    invoice_ids = [inv["_id"] for inv in invoices]
                    
                    # Create lookup dict for invoice numbers
                    inv_lookup = {inv["_id"]: inv for inv in invoices}
                    
                    if not invoice_ids:
                        st.warning("No invoices found for the selected date range.")
                    else:
                        # Fetch line items
                        line_items = list(db.line_items.find({"invoice_id": {"$in": invoice_ids}}))
                        
                        if not line_items:
                            st.warning("No line items found.")
                        else:
                            # Prepare data
                            export_data = []
                            for li in line_items:
                                invoice = inv_lookup.get(li.get("invoice_id"))
                                
                                # Get vendor name
                                vendor_name = "Unknown"
                                if invoice:
                                    vendor = db.vendors.find_one({"_id": invoice.get("vendor_id")})
                                    vendor_name = vendor["name"] if vendor else "Unknown"
                                
                                # Convert Decimal128 fields
                                quantity = li.get("quantity", 0)
                                if isinstance(quantity, Decimal128):
                                    quantity = float(quantity.to_decimal())
                                
                                unit_price = li.get("unit_price", 0)
                                if isinstance(unit_price, Decimal128):
                                    unit_price = float(unit_price.to_decimal())
                                
                                line_total = li.get("line_total", 0)
                                if isinstance(line_total, Decimal128):
                                    line_total = float(line_total.to_decimal())
                                
                                export_data.append({
                                    "Invoice Number": invoice.get("invoice_number", "") if invoice else "",
                                    "Vendor": vendor_name,
                                    "Line Number": li.get("line_number", ""),
                                    "Description": li.get("description", ""),
                                    "Quantity": quantity,
                                    "Unit": li.get("unit", ""),
                                    "Unit Price": unit_price,
                                    "Line Total": line_total,
                                    "Category": li.get("category", "")
                                })
                            
                            export_df = pd.DataFrame(export_data)
                            csv = export_df.to_csv(index=False)
                            
                            st.download_button(
                                label=f"💾 Download {len(line_items)} Line Items",
                                data=csv,
                                file_name=f"line_items_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                width='stretch'
                            )
                            
                            st.success(f"✅ Ready to download {len(line_items)} line items")
            
            except Exception as e:
                st.error(f"Error exporting line items: {str(e)}")
    
    st.divider()
    
    # Bulk delete operations with safety features
    st.markdown("### 🗑️ Bulk Delete Operations")
    st.error("⚠️ **DANGER ZONE:** These operations permanently delete ALL data from the selected collections. This action **CANNOT BE UNDONE**.")
    st.markdown("**Use only in development/testing environments. Always backup your data first!**")
    
    # Initialize session state for delete operations
    if "delete_cooldown" not in st.session_state:
        st.session_state.delete_cooldown = {}
    
    st.divider()
    
    # Helper function to check cooldown
    def can_delete(collection_name):
        import time
        if collection_name in st.session_state.delete_cooldown:
            elapsed = time.time() - st.session_state.delete_cooldown[collection_name]
            if elapsed < 3:  # 3 second cooldown
                return False
        return True
    
    # Helper function to perform delete with logging
    def delete_all_from_collection(collection_name, display_name):
        import time
        import logging
        
        logger = logging.getLogger(__name__)
        
        try:
            # Get count before delete
            before_count = db[collection_name].count_documents({})
            
            # Perform delete
            result = db[collection_name].delete_many({})
            deleted_count = result.deleted_count
            
            # Get count after delete
            after_count = db[collection_name].count_documents({})
            
            # Log the action
            logger.info(f"BULK DELETE: Deleted {deleted_count} documents from {collection_name} collection at {datetime.now()}")
            
            # Update cooldown
            st.session_state.delete_cooldown[collection_name] = time.time()
            
            return {
                "success": True,
                "before": before_count,
                "deleted": deleted_count,
                "after": after_count
            }
        except Exception as e:
            logger.error(f"Error deleting from {collection_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # Delete Invoices
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📄 Delete All Invoices")
        try:
            invoice_count = db.invoices.count_documents({})
            st.info(f"Current count: **{invoice_count:,}** invoices")
        except:
            invoice_count = 0
            st.warning("Unable to fetch count")
        
        confirm_invoices = st.checkbox(
            "I understand this will permanently delete ALL invoices",
            key="confirm_delete_invoices"
        )
        
        delete_invoices_disabled = not confirm_invoices or not can_delete("invoices")
        
        if st.button(
            "🗑️ DELETE ALL INVOICES",
            type="secondary",
            disabled=delete_invoices_disabled,
            key="btn_delete_invoices",
            use_container_width=True
        ):
            with st.spinner("Deleting all invoices..."):
                result = delete_all_from_collection("invoices", "Invoices")
                
                if result["success"]:
                    st.success(f"✅ Deleted {result['deleted']:,} invoices. Collection now has {result['after']} documents.")
                    st.session_state.confirm_delete_invoices = False
                    st.rerun()
                else:
                    st.error(f"❌ Error: {result['error']}")
    
    # Delete Line Items
    with col2:
        st.markdown("#### 📦 Delete All Line Items")
        try:
            line_items_count = db.line_items.count_documents({})
            st.info(f"Current count: **{line_items_count:,}** line items")
        except:
            line_items_count = 0
            st.warning("Unable to fetch count")
        
        confirm_line_items = st.checkbox(
            "I understand this will permanently delete ALL line items",
            key="confirm_delete_line_items"
        )
        
        delete_line_items_disabled = not confirm_line_items or not can_delete("line_items")
        
        if st.button(
            "🗑️ DELETE ALL LINE ITEMS",
            type="secondary",
            disabled=delete_line_items_disabled,
            key="btn_delete_line_items",
            use_container_width=True
        ):
            with st.spinner("Deleting all line items..."):
                result = delete_all_from_collection("line_items", "Line Items")
                
                if result["success"]:
                    st.success(f"✅ Deleted {result['deleted']:,} line items. Collection now has {result['after']} documents.")
                    st.session_state.confirm_delete_line_items = False
                    st.rerun()
                else:
                    st.error(f"❌ Error: {result['error']}")
    
    # Delete Menu Items
    with col3:
        st.markdown("#### 🍽️ Delete All Menu Items")
        try:
            menu_items_count = db.menu_items.count_documents({})
            st.info(f"Current count: **{menu_items_count:,}** menu items")
        except:
            menu_items_count = 0
            st.warning("Unable to fetch count")
        
        confirm_menu_items = st.checkbox(
            "I understand this will permanently delete ALL menu items",
            key="confirm_delete_menu_items"
        )
        
        delete_menu_items_disabled = not confirm_menu_items or not can_delete("menu_items")
        
        if st.button(
            "🗑️ DELETE ALL MENU ITEMS",
            type="secondary",
            disabled=delete_menu_items_disabled,
            key="btn_delete_menu_items",
            use_container_width=True
        ):
            with st.spinner("Deleting all menu items..."):
                result = delete_all_from_collection("menu_items", "Menu Items")
                
                if result["success"]:
                    st.success(f"✅ Deleted {result['deleted']:,} menu items. Collection now has {result['after']} documents.")
                    st.session_state.confirm_delete_menu_items = False
                    st.rerun()
                else:
                    st.error(f"❌ Error: {result['error']}")

# TAB 5: DATABASE BROWSER
with tab5:
    st.header("📊 Database Browser")
    st.markdown("**Read-only view** of all database collections. Browse and inspect data without making changes.")
    
    # Initialize session state for browser
    if "db_browser_page" not in st.session_state:
        st.session_state.db_browser_page = 0
    
    # Collection selector
    all_collections = [
        "invoices",
        "line_items",
        "vendors",
        "restaurants",
        "categories",
        "temp_uploads",
        "vendor_regex_templates",
        "item_lookup_map",
        "menu_items",
        "menu_categories",
        "menu_item_lookup_map"
    ]
    
    # Collection metadata for better display
    collection_info = {
        "invoices": {"icon": "📄", "name": "Invoices"},
        "line_items": {"icon": "📦", "name": "Line Items"},
        "vendors": {"icon": "👥", "name": "Vendors"},
        "restaurants": {"icon": "🏢", "name": "Restaurants"},
        "categories": {"icon": "🏷️", "name": "Categories"},
        "temp_uploads": {"icon": "⏳", "name": "Temporary Uploads"},
        "vendor_regex_templates": {"icon": "🔧", "name": "Vendor Regex Templates"},
        "item_lookup_map": {"icon": "🗂️", "name": "Item Lookup Map"},
        "menu_items": {"icon": "🍽️", "name": "Menu Items"},
        "menu_categories": {"icon": "📂", "name": "Menu Categories"},
        "menu_item_lookup_map": {"icon": "🔍", "name": "Menu Item Lookup Map"}
    }
    
    # Create collection dropdown options with icons
    collection_options = [f"{collection_info[col]['icon']} {collection_info[col]['name']}" for col in all_collections]
    
    selected_display = st.selectbox(
        "Select Collection to Browse",
        collection_options,
        key="db_browser_collection_select"
    )
    
    # Extract actual collection name from display string
    selected_collection = None
    for col in all_collections:
        if collection_info[col]['icon'] in selected_display and collection_info[col]['name'] in selected_display:
            selected_collection = col
            break
    
    if selected_collection:
        st.divider()
        
        try:
            # Get total count
            total_count = db[selected_collection].count_documents({})
            
            if total_count == 0:
                st.info(f"📭 The **{collection_info[selected_collection]['name']}** collection is empty.")
            else:
                # Pagination settings
                items_per_page = 20
                total_pages = (total_count + items_per_page - 1) // items_per_page
                
                # Display count and pagination controls
                col1, col2, col3 = st.columns([2, 1, 2])
                with col1:
                    st.metric("Total Records", f"{total_count:,}")
                with col2:
                    current_page = st.number_input(
                        "Page",
                        min_value=1,
                        max_value=max(1, total_pages),
                        value=1,
                        key=f"page_{selected_collection}"
                    )
                with col3:
                    st.metric("Total Pages", f"{total_pages:,}")
                
                # Calculate skip value
                skip = (current_page - 1) * items_per_page
                
                # Fetch documents
                documents = list(db[selected_collection].find({}).skip(skip).limit(items_per_page))
                
                if documents:
                    # Convert BSON types for display
                    display_data = []
                    for doc in documents:
                        display_row = {}
                        for key, value in doc.items():
                            # Handle different BSON types
                            if isinstance(value, ObjectId):
                                display_row[key] = str(value)
                            elif isinstance(value, Decimal128):
                                display_row[key] = float(value.to_decimal())
                            elif isinstance(value, datetime):
                                display_row[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                            elif isinstance(value, str) and len(value) > 100:
                                # Truncate long strings
                                display_row[key] = value[:100] + "..."
                            elif isinstance(value, (dict, list)):
                                # Convert complex types to string representation
                                display_row[key] = str(value)[:100] + ("..." if len(str(value)) > 100 else "")
                            else:
                                display_row[key] = value
                        display_data.append(display_row)
                    
                    # Create DataFrame
                    df = pd.DataFrame(display_data)
                    
                    # Display as table
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        height=600
                    )
                    
                    # Show record range
                    start_record = skip + 1
                    end_record = min(skip + items_per_page, total_count)
                    st.caption(f"Showing records {start_record:,} to {end_record:,} of {total_count:,}")
                else:
                    st.warning("No documents found in this range.")
                    
        except Exception as e:
            st.error(f"Error loading collection data: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())



# Footer
st.divider()
st.caption(f"🔧 Database Administration | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Help section
with st.expander("ℹ️ Help & Information"):
    st.markdown("""
    ### Database Administration Guide
    
    #### **Maintenance Tab**
    - **Database Statistics**: View document counts for all 11 collections
    - **Cleanup Operations**: Remove old temporary upload sessions
    - **Database Info**: View storage and index details
    
    #### **Vendor Management Tab**
    - **View Vendors**: Browse all vendors in the database
    - **Add Vendor**: Create new vendor records manually
    - **Regex Templates**: Configure advanced extraction patterns (admin only)
    
    #### **Category Management Tab**
    - **View Categories**: Browse categories organized by type
    - **Add Category**: Create new category types
    - **Category Mapping**: Items are auto-categorized during processing
    
    #### **Bulk Operations Tab**
    - **Export Invoices**: Download invoice data as CSV
    - **Export Line Items**: Download line item data as CSV
    - **Date Ranges**: Filter exports by time period
    - **Delete Operations**: Permanently remove all data from invoices, line_items, or menu_items
      - ⚠️ **WARNING**: Delete operations are irreversible!
      - Always backup data before deleting
      - Use only in development/testing environments
      - 3-second cooldown between operations
    
    #### **Database Browser Tab**
    - **Read-Only Viewer**: Browse all 11 database collections
    - **Pagination**: Navigate through large datasets (20 records per page)
    - **BSON Handling**: Automatically converts ObjectId, Decimal128, and datetime types
    - **No Edit Access**: View-only mode prevents accidental modifications
    
    #### **Best Practices**
    - Run cleanup operations weekly to maintain performance
    - **Always backup data before bulk delete operations**
    - Verify vendor information after manual entry
    - Use consistent category naming conventions
    - Use Database Browser for inspection only
    """)
