import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.storage.database import (
    db,
    get_vendor_name_by_id,
    get_invoice_by_id,
    update_invoice,
    update_line_item,
    add_line_item,
    delete_line_item
)
from bson import ObjectId

st.set_page_config(page_title="View Invoices", page_icon="👁️", layout="wide")

# Initialize session state
if "selected_invoice_id" not in st.session_state:
    st.session_state.selected_invoice_id = None

if "edit_invoice_mode" not in st.session_state:
    st.session_state.edit_invoice_mode = False


def fetch_invoices(filters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """
    Fetch invoices from database with optional filters.
    
    Args:
        filters: Dictionary of filter criteria
        
    Returns:
        List of invoice documents
    """
    query = filters if filters else {}
    
    try:
        invoices = list(db["invoices"].find(query).sort("date", -1))
        
        # Enrich with vendor names
        for invoice in invoices:
            vendor_id = invoice.get("vendor_id")
            if vendor_id:
                invoice["vendor_name"] = get_vendor_name_by_id(str(vendor_id)) or "Unknown"
            else:
                invoice["vendor_name"] = "Unknown"
        
        return invoices
    except Exception as e:
        st.error(f"Error fetching invoices: {e}")
        return []


def convert_invoice_to_df(invoice: Dict[str, Any]) -> pd.DataFrame:
    """Convert invoice document to DataFrame for display."""
    # Handle Decimal128 conversion
    total_amt = invoice.get('total_amount', 0)
    if hasattr(total_amt, 'to_decimal'):  # Decimal128
        total_amt = float(total_amt.to_decimal())
    else:
        total_amt = float(total_amt) if total_amt else 0
    
    return pd.DataFrame([{
        "Invoice ID": str(invoice["_id"]),
        "Invoice Number": invoice.get("invoice_number", ""),
        "Date": invoice.get("date", "").strftime("%Y-%m-%d") if isinstance(invoice.get("date"), datetime) else "",
        "Vendor": invoice.get("vendor_name", "Unknown"),
        "Total Amount": f"${total_amt:,.2f}",
        "Filename": invoice.get("filename", ""),
        "Line Items": len(invoice.get("line_items", []))
    }])


def convert_line_items_to_df(line_items: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert line items to DataFrame for display."""
    if not line_items:
        return pd.DataFrame(columns=["Description", "Quantity", "Unit", "Unit Price", "Line Total"])
    
    return pd.DataFrame([{
        "Description": item.get("description", ""),
        "Quantity": item.get("quantity", 0),
        "Unit": item.get("unit", ""),
        "Unit Price": float(item.get("unit_price", 0)),
        "Line Total": float(item.get("line_total", 0))
    } for item in line_items])


def render_filters():
    """Render filter sidebar."""
    st.sidebar.title("🔍 Filters")
    
    filters = {}
    
    # Date range filter
    st.sidebar.subheader("Date Range")
    date_option = st.sidebar.selectbox(
        "Select Period",
        ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days", "Custom Range"]
    )
    
    if date_option == "Last 7 Days":
        start_date = datetime.now() - timedelta(days=7)
        filters["date"] = {"$gte": start_date}
    elif date_option == "Last 30 Days":
        start_date = datetime.now() - timedelta(days=30)
        filters["date"] = {"$gte": start_date}
    elif date_option == "Last 90 Days":
        start_date = datetime.now() - timedelta(days=90)
        filters["date"] = {"$gte": start_date}
    elif date_option == "Custom Range":
        col1, col2 = st.sidebar.columns(2)
        start_date = col1.date_input("From", value=datetime.now() - timedelta(days=30))
        end_date = col2.date_input("To", value=datetime.now())
        
        if start_date and end_date:
            filters["date"] = {
                "$gte": datetime.combine(start_date, datetime.min.time()),
                "$lte": datetime.combine(end_date, datetime.max.time())
            }
    
    # Vendor filter
    st.sidebar.subheader("Vendor")
    vendors = list(db["vendors"].find({}, {"name": 1}))
    vendor_names = ["All Vendors"] + [v["name"] for v in vendors]
    selected_vendor = st.sidebar.selectbox("Select Vendor", vendor_names)
    
    if selected_vendor != "All Vendors":
        vendor_doc = db["vendors"].find_one({"name": selected_vendor})
        if vendor_doc:
            filters["vendor_id"] = vendor_doc["_id"]
    
    # Invoice number search
    st.sidebar.subheader("Invoice Number")
    invoice_search = st.sidebar.text_input("Search by Invoice #")
    if invoice_search:
        filters["invoice_number"] = {"$regex": invoice_search, "$options": "i"}
    
    # Amount range filter
    st.sidebar.subheader("Amount Range")
    use_amount_filter = st.sidebar.checkbox("Filter by Amount")
    if use_amount_filter:
        col1, col2 = st.sidebar.columns(2)
        min_amount = col1.number_input("Min $", min_value=0.0, value=0.0, step=10.0)
        max_amount = col2.number_input("Max $", min_value=0.0, value=10000.0, step=10.0)
        
        # Note: MongoDB Decimal128 comparison might need special handling
        # For now, we'll fetch all and filter in Python if needed
        filters["_amount_range"] = (min_amount, max_amount)
    
    return filters


def render_invoice_list(invoices: List[Dict[str, Any]]):
    """Render the list of invoices."""
    if not invoices:
        st.info("No invoices found matching the filters.")
        return
    
    st.markdown(f"### 📋 Found {len(invoices)} invoice(s)")
    
    # Create summary DataFrame
    invoices_df = pd.concat([convert_invoice_to_df(inv) for inv in invoices], ignore_index=True)
    
    # Display as interactive table
    selected_indices = st.dataframe(
        invoices_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    # Handle row selection
    if selected_indices and "selection" in selected_indices and "rows" in selected_indices["selection"]:
        selected_rows = selected_indices["selection"]["rows"]
        if selected_rows:
            selected_idx = selected_rows[0]
            selected_invoice_id = invoices_df.iloc[selected_idx]["Invoice ID"]
            st.session_state.selected_invoice_id = selected_invoice_id


def render_invoice_detail():
    """Render detailed view of selected invoice."""
    if not st.session_state.selected_invoice_id:
        st.info("👆 Select an invoice from the list above to view details")
        return
    
    invoice = get_invoice_by_id(st.session_state.selected_invoice_id)
    
    if not invoice:
        st.error("Invoice not found")
        st.session_state.selected_invoice_id = None
        return
    
    st.divider()
    st.markdown("## 📄 Invoice Details")
    
    # Header with edit toggle
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"### Invoice #{invoice.get('invoice_number', 'N/A')}")
    
    with col2:
        if st.button(
            "✏️ Edit" if not st.session_state.edit_invoice_mode else "👁️ View",
            use_container_width=True
        ):
            st.session_state.edit_invoice_mode = not st.session_state.edit_invoice_mode
            st.rerun()
    
    # Invoice metadata
    if st.session_state.edit_invoice_mode:
        # Edit mode
        st.markdown("#### Edit Invoice Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_invoice_number = st.text_input(
                "Invoice Number",
                value=invoice.get("invoice_number", ""),
                key="edit_inv_num"
            )
            
            new_date = st.date_input(
                "Invoice Date",
                value=invoice.get("date", datetime.now()),
                key="edit_inv_date"
            )
        
        with col2:
            new_total = st.number_input(
                "Total Amount",
                value=float(invoice.get("total_amount", 0)),
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key="edit_inv_total"
            )
            
            new_order_number = st.text_input(
                "Order Number (Optional)",
                value=invoice.get("order_number", ""),
                key="edit_order_num"
            )
        
        # Save changes button
        if st.button("💾 Save Invoice Changes", type="primary"):
            update_data = {
                "invoice_number": new_invoice_number,
                "date": new_date,
                "total_amount": new_total,
            }
            
            if new_order_number:
                update_data["order_number"] = new_order_number
            
            result = update_invoice(st.session_state.selected_invoice_id, update_data)
            
            if result["success"]:
                st.success("✅ Invoice updated successfully!")
                st.session_state.edit_invoice_mode = False
                st.rerun()
            else:
                st.error(f"❌ {result['message']}")
    else:
        # View mode
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Invoice Number", invoice.get("invoice_number", "N/A"))
        with col2:
            date_val = invoice.get("date", "")
            st.metric("Date", date_val.strftime("%Y-%m-%d") if isinstance(date_val, datetime) else "N/A")
        with col3:
            st.metric("Total Amount", f"${invoice.get('total_amount', 0):,.2f}")
        with col4:
            st.metric("Vendor", get_vendor_name_by_id(str(invoice.get("vendor_id", ""))) or "Unknown")
        
        # Additional info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.text(f"📁 Filename: {invoice.get('filename', 'N/A')}")
        with col2:
            st.text(f"📄 Pages: {invoice.get('page_count', 'N/A')}")
        with col3:
            extraction_time = invoice.get("extraction_timestamp", "")
            if isinstance(extraction_time, datetime):
                st.text(f"⏱️ Extracted: {extraction_time.strftime('%Y-%m-%d %H:%M')}")
    
    # Line items section
    st.markdown("#### 📦 Line Items")
    
    line_items = invoice.get("line_items", [])
    
    if line_items:
        if st.session_state.edit_invoice_mode:
            # Editable line items
            line_items_df = convert_line_items_to_df(line_items)
            
            edited_df = st.data_editor(
                line_items_df,
                num_rows="dynamic",
                use_container_width=True,
                key="edit_line_items",
                column_config={
                    "Description": st.column_config.TextColumn("Description", width="large"),
                    "Quantity": st.column_config.NumberColumn("Quantity", format="%.2f"),
                    "Unit": st.column_config.TextColumn("Unit"),
                    "Unit Price": st.column_config.NumberColumn("Unit Price", format="$%.2f"),
                    "Line Total": st.column_config.NumberColumn("Line Total", format="$%.2f")
                }
            )
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("💾 Save Line Items", type="primary"):
                    # Update each line item
                    success_count = 0
                    for idx, row in edited_df.iterrows():
                        update_data = {
                            "description": row["Description"],
                            "quantity": row["Quantity"],
                            "unit": row["Unit"],
                            "unit_price": row["Unit Price"],
                            "line_total": row["Line Total"]
                        }
                        
                        result = update_line_item(
                            st.session_state.selected_invoice_id,
                            idx,
                            update_data
                        )
                        
                        if result["success"]:
                            success_count += 1
                    
                    if success_count > 0:
                        st.success(f"✅ Updated {success_count} line item(s)")
                        st.rerun()
            
            with col2:
                if st.button("➕ Add New Line Item"):
                    new_item = {
                        "description": "New Item",
                        "quantity": 1.0,
                        "unit": "ea",
                        "unit_price": 0.0,
                        "line_total": 0.0
                    }
                    
                    result = add_line_item(st.session_state.selected_invoice_id, new_item)
                    
                    if result["success"]:
                        st.success("✅ Line item added")
                        st.rerun()
                    else:
                        st.error(f"❌ {result['message']}")
        else:
            # View mode
            line_items_df = convert_line_items_to_df(line_items)
            st.dataframe(
                line_items_df,
                use_container_width=True,
                hide_index=True
            )
        
        st.info(f"📦 Total: {len(line_items)} line item(s)")
    else:
        st.warning("No line items found")
        
        if st.session_state.edit_invoice_mode:
            if st.button("➕ Add First Line Item"):
                new_item = {
                    "description": "New Item",
                    "quantity": 1.0,
                    "unit": "ea",
                    "unit_price": 0.0,
                    "line_total": 0.0
                }
                
                result = add_line_item(st.session_state.selected_invoice_id, new_item)
                
                if result["success"]:
                    st.success("✅ Line item added")
                    st.rerun()
                else:
                    st.error(f"❌ {result['message']}")


def main():
    st.title("👁️ View Invoices")
    st.markdown("View and edit saved invoices from the database")
    
    # Render filters in sidebar
    filters = render_filters()
    
    # Remove special filter keys that need Python-side filtering
    amount_range = filters.pop("_amount_range", None)
    
    # Fetch invoices
    invoices = fetch_invoices(filters)
    
    # Apply amount range filter if specified
    if amount_range:
        min_amt, max_amt = amount_range
        invoices = [
            inv for inv in invoices
            if min_amt <= float(inv.get("total_amount", 0)) <= max_amt
        ]
    
    # Summary metrics
    if invoices:
        col1, col2, col3, col4 = st.columns(4)
        
        # Handle Decimal128 conversion
        total_amount = 0
        for inv in invoices:
            amt = inv.get("total_amount", 0)
            if hasattr(amt, 'to_decimal'):  # Decimal128
                total_amount += float(amt.to_decimal())
            else:
                total_amount += float(amt) if amt else 0
        
        total_items = sum(len(inv.get("line_items", [])) for inv in invoices)
        unique_vendors = len(set(inv.get("vendor_id") for inv in invoices if inv.get("vendor_id")))
        
        col1.metric("Total Invoices", len(invoices))
        col2.metric("Total Amount", f"${total_amount:,.2f}")
        col3.metric("Total Line Items", total_items)
        col4.metric("Unique Vendors", unique_vendors)
    
    st.divider()
    
    # Render invoice list
    render_invoice_list(invoices)
    
    # Render selected invoice detail
    render_invoice_detail()


if __name__ == "__main__":
    main()
