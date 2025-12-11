import streamlit as st
import pandas as pd
from datetime import datetime

from src import get_structured_data_from_text, save_inv_li_to_db

# Constants
TEMP_RESTAURANT_ID = 1

# Fixed demo invoice text
DEMO_INVOICE_TEXT = """US FOODS
1000 PARKWOOD CIRCLE SE
ATLANTA, GA 30339
888-777-6666

SOLD TO:                           INVOICE
Demo Restaurant                    Invoice #: INV-2024-001
123 Main Street                    Order #: ORD-2024-001  
Atlanta, GA 30301                  Date: 12/08/2024

ITEM#    DESCRIPTION              QTY    UNIT    PRICE      AMOUNT
----------------------------------------------------------------------
101234   Chicken Breast Fresh     25.0   LB      $3.99      $99.75
202345   Tomatoes Roma            15.0   LB      $2.49      $37.35
303456   Lettuce Iceberg          10.0   EA      $1.99      $19.90
404567   Ground Beef 80/20        30.0   LB      $4.99      $149.70
505678   Onions Yellow            12.0   LB      $1.29      $15.48

                                          SUBTOTAL:         $322.18
                                          TAX:              $25.77
                                          TOTAL:            $347.95

Thank you for your business!
"""

# Page configuration
st.title("🧪 Database Controls - CRUD Operations Demo")
st.markdown("Perform CRUD operations on invoice data using Steps 2 & 3 only")

# Initialize session state
if 'inv_df' not in st.session_state:
    st.session_state.inv_df = None
if 'li_df' not in st.session_state:
    st.session_state.li_df = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
if 'saved_invoice_id' not in st.session_state:
    st.session_state.saved_invoice_id = None

# Display fixed demo invoice
st.header("📄 Step 1: Demo Invoice (Fixed)")
with st.expander("View Demo Invoice Text", expanded=False):
    st.code(DEMO_INVOICE_TEXT, language="text")

st.info("💡 This demo uses a fixed invoice text. Step 1 (file extraction) is skipped.")

# Processing section
st.header("⚙️ Step 2: Structure Data")

col1, col2 = st.columns([1, 1])
with col1:
    process_button = st.button("🚀 Extract & Structure Data", type="primary", use_container_width=True)
with col2:
    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state.inv_df = None
        st.session_state.li_df = None
        st.session_state.error_message = None
        st.session_state.edit_mode = False
        st.session_state.saved_invoice_id = None
        st.rerun()

# Step 2: Structure data from text
if process_button:
    with st.spinner("🔄 Structuring data and identifying vendor..."):
        try:
            # Use fixed demo data
            inv_df, li_df = get_structured_data_from_text(
                DEMO_INVOICE_TEXT,
                "demo_invoice.txt",
                len(DEMO_INVOICE_TEXT),
                1,
                datetime.now().isoformat(),
                TEMP_RESTAURANT_ID
            )
            
            # Store in session state
            st.session_state.inv_df = inv_df
            st.session_state.li_df = li_df
            st.session_state.error_message = None
            
            st.success("✅ Data structured successfully!")
            st.balloons()
            st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error during data structuring: {str(e)}")
            st.session_state.error_message = f"Step 2 Error: {str(e)}"

# Display and edit structured data
if st.session_state.inv_df is not None:
    st.divider()
    st.header("✏️ CRUD Operations on Structured Data")
    
    # Toggle edit mode
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📋 Invoice Data")
    with col2:
        edit_toggle = st.checkbox("Enable Edit Mode", value=st.session_state.edit_mode, key="edit_toggle")
        st.session_state.edit_mode = edit_toggle
    
    # Invoice DataFrame editing
    if st.session_state.edit_mode:
        st.info("✏️ **UPDATE MODE** - Modify the invoice data below")
        
        # Make a copy for editing
        inv_df_display = st.session_state.inv_df.copy()
        
        edited_inv_df = st.data_editor(
            inv_df_display,
            use_container_width=True,
            num_rows="fixed",
            key="edit_inv_df",
            column_config={
                "invoice_number": st.column_config.TextColumn("Invoice #", required=True),
                "vendor_name": st.column_config.TextColumn("Vendor Name", required=True),
                "invoice_date": st.column_config.TextColumn("Invoice Date"),
                "invoice_total_amount": st.column_config.NumberColumn("Total Amount", format="$%.2f"),
                "order_number": st.column_config.TextColumn("Order #"),
            }
        )
        
        if st.button("💾 Update Invoice Data", type="primary"):
            st.session_state.inv_df = edited_inv_df
            st.success("✅ Invoice data updated in memory!")
            st.rerun()
    else:
        st.dataframe(st.session_state.inv_df, use_container_width=True)
    
    # Display key invoice metrics
    if not st.session_state.inv_df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            vendor = st.session_state.inv_df.get('vendor_name', pd.Series([None])).iloc[0]
            st.metric("Vendor", vendor if pd.notna(vendor) else "N/A")
        with col2:
            inv_date = st.session_state.inv_df.get('invoice_date', pd.Series([None])).iloc[0]
            st.metric("Invoice Date", inv_date if pd.notna(inv_date) else "N/A")
        with col3:
            total = st.session_state.inv_df.get('invoice_total_amount', pd.Series([0])).iloc[0]
            st.metric("Total Amount", f"${float(total):.2f}" if pd.notna(total) else "N/A")
    
    st.divider()
    
    # Line Items DataFrame editing
    st.subheader("📦 Line Items Data")
    
    if st.session_state.li_df is not None and not st.session_state.li_df.empty:
        if st.session_state.edit_mode:
            st.info("✏️ **UPDATE MODE** - Add, edit, or delete line items below")
            
            # Make a copy for editing
            li_df_display = st.session_state.li_df.copy()
            
            edited_li_df = st.data_editor(
                li_df_display,
                use_container_width=True,
                num_rows="dynamic",
                key="edit_li_df",
                column_config={
                    "description": st.column_config.TextColumn("Description", width="large", required=True),
                    "quantity": st.column_config.NumberColumn("Quantity", format="%.2f", required=True),
                    "unit": st.column_config.TextColumn("Unit", required=True),
                    "unit_price": st.column_config.NumberColumn("Unit Price", format="$%.2f", required=True),
                    "line_total": st.column_config.NumberColumn("Line Total", format="$%.2f", required=True)
                }
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Update Line Items", type="primary", use_container_width=True):
                    st.session_state.li_df = edited_li_df
                    st.success(f"✅ Line items updated! Total items: {len(edited_li_df)}")
                    st.rerun()
            with col2:
                if st.button("➕ Add Blank Row", use_container_width=True):
                    # Add a blank row with default values
                    new_row = pd.DataFrame([{
                        "description": "New Item",
                        "quantity": 1.0,
                        "unit": "EA",
                        "unit_price": 0.0,
                        "line_total": 0.0,
                        "line_number": len(st.session_state.li_df) + 1
                    }])
                    st.session_state.li_df = pd.concat([st.session_state.li_df, new_row], ignore_index=True)
                    st.success("➕ Blank row added!")
                    st.rerun()
        else:
            st.dataframe(st.session_state.li_df, use_container_width=True, hide_index=True)
        
        st.info(f"📦 Total line items: {len(st.session_state.li_df)}")
    else:
        st.warning("No line items available")
        
        if st.session_state.edit_mode:
            if st.button("➕ Add First Line Item"):
                # Create first line item
                new_item = pd.DataFrame([{
                    "description": "New Item",
                    "quantity": 1.0,
                    "unit": "EA",
                    "unit_price": 0.0,
                    "line_total": 0.0,
                    "line_number": 1
                }])
                st.session_state.li_df = new_item
                st.success("✅ First line item added!")
                st.rerun()
    
    # Step 3: Save to Database
    st.divider()
    st.header("💾 Step 3: Database Operations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 CREATE - Save to DB", type="primary", use_container_width=True):
            try:
                with st.spinner("💾 Saving to database..."):
                    result = save_inv_li_to_db(st.session_state.inv_df, st.session_state.li_df, TEMP_RESTAURANT_ID)
                    
                    if result and result.get('success'):
                        st.session_state.saved_invoice_id = result['invoice_id']
                        st.success(f"✅ {result.get('message', 'Data saved successfully!')}")
                        st.info(f"📄 Invoice ID: `{result['invoice_id']}`")
                    else:
                        st.error(f"❌ {result.get('message', 'Save failed')}")
                    st.balloons()
            except Exception as e:
                st.error(f"❌ Error saving to DB: {str(e)}")
    
    with col2:
        if st.button("📖 READ - View from DB", use_container_width=True):
            if st.session_state.saved_invoice_id:
                st.info(f"📖 READ: Invoice ID {st.session_state.saved_invoice_id}")
                st.write("Use the **View Invoices** page to read from database")
            else:
                st.warning("⚠️ No invoice saved yet. Save first to get an ID.")
    
    with col3:
        if st.button("🗑️ DELETE - Remove from DB", use_container_width=True):
            st.warning("🗑️ DELETE operations available in **View Invoices** page")

    # Show saved invoice ID if available
    if st.session_state.saved_invoice_id:
        st.success(f"✅ Last saved Invoice ID: `{st.session_state.saved_invoice_id}`")
        st.info("💡 Go to **View Invoices** page to view, edit, or delete this invoice")

# Display error message if any
if st.session_state.error_message:
    st.divider()
    st.error(f"⚠️ Last Error: {st.session_state.error_message}")

# Footer
st.divider()
st.caption(f"🧪 Database Controls Demo | Restaurant ID: {TEMP_RESTAURANT_ID} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Help section
with st.expander("ℹ️ How to use this demo", expanded=False):
    st.markdown("""
    ### CRUD Operations Demo
    
    This page demonstrates Create, Read, Update operations on invoice data:
    
    #### **Step 2: Structure Data**
    - Click **"Extract & Structure Data"** to process the fixed demo invoice
    - Uses LLM (Google Gemini) to identify vendor and extract structured data
    - Results stored in memory as DataFrames
    
    #### **CRUD Operations**
    1. **CREATE**: 
       - Click **"CREATE - Save to DB"** to insert the invoice into MongoDB
       - Returns an Invoice ID for reference
    
    2. **READ**:
       - Use the **View Invoices** page to fetch and display saved invoices
       - Filter by vendor, date, amount, etc.
    
    3. **UPDATE**:
       - Enable **"Edit Mode"** checkbox to modify invoice/line items
       - Click **"Update Invoice Data"** or **"Update Line Items"** to save changes
       - Add/remove line items dynamically
    
    4. **DELETE**:
       - Available in the **View Invoices** page
       - Select an invoice and delete it from the database
    
    #### **Demo Features**
    - ✏️ Inline editing with `st.data_editor()`
    - ➕ Add new line items dynamically
    - 🔄 Session state persistence (data survives page refreshes)
    - 💾 Real database integration with MongoDB
    """)
