import streamlit as st
import pandas as pd
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.extraction.invoice_extractor import process_invoice
from src.processing.build_dataframe import get_structured_data_from_text
from src.storage.database import (
    db,
    save_inv_li_to_db,
    save_temp_upload,
    get_temp_upload,
    delete_temp_upload,
    check_duplicate_invoice,
    update_invoice,
    update_line_item,
    add_line_item,
    delete_line_item,
    get_vendor_name_by_id
)

st.set_page_config(page_title="Upload Invoices", page_icon="📤", layout="wide")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "uploaded_files_data" not in st.session_state:
    st.session_state.uploaded_files_data = []

if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

if "save_complete" not in st.session_state:
    st.session_state.save_complete = False

if "current_step" not in st.session_state:
    st.session_state.current_step = "upload"  # upload, review, saved

if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = {}

# Load from database if session exists
if not st.session_state.uploaded_files_data:
    saved_session = get_temp_upload(st.session_state.session_id)
    if saved_session and "invoices" in saved_session:
        st.session_state.uploaded_files_data = saved_session["invoices"]
        st.session_state.processing_complete = True
        st.session_state.current_step = "review"


def save_session_to_db():
    """Save current session state to temporary database."""
    upload_data = {
        "invoices": st.session_state.uploaded_files_data,
        "processing_complete": st.session_state.processing_complete,
        "current_step": st.session_state.current_step
    }
    save_temp_upload(st.session_state.session_id, upload_data)


def process_single_file(uploaded_file, temp_dir: Path) -> Dict[str, Any]:
    """
    Process a single uploaded file and extract invoice data.
    
    Returns:
        Dictionary containing extraction results and status
    """
    result = {
        "filename": uploaded_file.name,
        "status": "processing",
        "message": "",
        "invoice_df": None,
        "line_items_df": None,
        "extracted_text": "",
        "vendor_id": None,
        "vendor_name": "",
        "is_duplicate": False,
        "duplicate_id": None,
        "extraction_failed": False
    }
    
    try:
        # Save uploaded file temporarily
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Step 1: Extract text using process_invoice
        extracted_text = process_invoice(str(file_path))
        
        if not extracted_text or not extracted_text.strip():
            result["status"] = "failed"
            result["message"] = "Text extraction failed"
            result["extraction_failed"] = True
            return result
        
        result["extracted_text"] = extracted_text
        
        # Step 2: Build structured dataframes
        # Get default restaurant_id from database
        restaurant = db["restaurants"].find_one({}, {"_id": 1})
        restaurant_id = str(restaurant["_id"]) if restaurant else "000000000000000000000000"
        
        inv_df, li_df = get_structured_data_from_text(
            extracted_text=extracted_text,
            filename=uploaded_file.name,
            text_length=len(extracted_text),
            page_count=1,  # Single file processing
            extraction_timestamp=datetime.now(),
            restaurant_id=restaurant_id
        )
        
        if inv_df is None or inv_df.empty:
            result["status"] = "partial"
            result["message"] = "Data extraction incomplete - manual review required"
            result["extraction_failed"] = True
            result["invoice_df"] = pd.DataFrame({
                "filename": [uploaded_file.name],
                "invoice_number": [""],
                "invoice_date": [datetime.now()],
                "invoice_total_amount": [0.0],
                "vendor_id": [""],
                "vendor_name": ["Unknown"],
                "text_length": [len(extracted_text)],
                "page_count": [1]
            })
            result["line_items_df"] = pd.DataFrame(columns=[
                "description", "quantity", "unit", "unit_price", "line_total"
            ])
            return result
        
        result["invoice_df"] = inv_df
        result["line_items_df"] = li_df if li_df is not None else pd.DataFrame()
        
        # Get vendor information
        if not inv_df.empty and "vendor_id" in inv_df.columns:
            vendor_id = inv_df.iloc[0]["vendor_id"]
            result["vendor_id"] = vendor_id
            result["vendor_name"] = get_vendor_name_by_id(str(vendor_id)) or "Unknown"
        
        # Check for duplicates
        if not inv_df.empty and "invoice_number" in inv_df.columns and result["vendor_id"]:
            invoice_number = inv_df.iloc[0]["invoice_number"]
            duplicate = check_duplicate_invoice(str(result["vendor_id"]), str(invoice_number))
            if duplicate:
                result["is_duplicate"] = True
                result["duplicate_id"] = str(duplicate["_id"])
                result["status"] = "duplicate"
                result["message"] = f"Duplicate found: Invoice #{invoice_number} already exists"
            else:
                result["status"] = "success"
                result["message"] = "Extraction successful"
        else:
            result["status"] = "success"
            result["message"] = "Extraction successful"
            
    except Exception as e:
        result["status"] = "failed"
        result["message"] = f"Error: {str(e)}"
        result["extraction_failed"] = True
    
    return result


def generate_demo_data():
    """Generate dummy invoice data for demonstration."""
    from bson import ObjectId
    
    # Get vendor IDs from database
    vendors = list(db["vendors"].find({}, {"_id": 1, "name": 1}).limit(3))
    if not vendors:
        # Create placeholder vendor IDs
        vendors = [
            {"_id": ObjectId(), "name": "Demo Vendor 1"},
            {"_id": ObjectId(), "name": "Demo Vendor 2"},
        ]
    
    demo_invoices = [
        {
            "filename": "demo_invoice_001.pdf",
            "status": "success",
            "message": "Extraction successful",
            "invoice_df": pd.DataFrame({
                "filename": ["demo_invoice_001.pdf"],
                "invoice_number": ["INV-2024-001"],
                "invoice_date": [datetime(2024, 12, 1)],
                "invoice_total_amount": [1245.80],
                "vendor_id": [str(vendors[0]["_id"])],
                "vendor_name": [vendors[0]["name"]],
                "text_length": [1523],
                "page_count": [2]
            }),
            "line_items_df": pd.DataFrame({
                "description": ["Fresh Organic Tomatoes", "Premium Lettuce Mix", "Yellow Onions"],
                "quantity": [25.0, 10.0, 50.0],
                "unit": ["lb", "case", "lb"],
                "unit_price": [3.49, 18.99, 0.89],
                "line_total": [87.25, 189.90, 44.50]
            }),
            "extracted_text": "INVOICE\n\nBill To: Demo Restaurant\nInvoice Number: INV-2024-001\nDate: 12/01/2024\n\nITEM DESCRIPTION    QTY    UNIT    PRICE    TOTAL\nFresh Organic Tomatoes    25    lb    $3.49    $87.25\nPremium Lettuce Mix    10    case    $18.99    $189.90\nYellow Onions    50    lb    $0.89    $44.50\n\nSubtotal: $321.65\nTax: $25.73\nTOTAL: $1,245.80",
            "vendor_id": str(vendors[0]["_id"]),
            "vendor_name": vendors[0]["name"],
            "is_duplicate": False,
            "duplicate_id": None,
            "extraction_failed": False
        },
        {
            "filename": "demo_invoice_002.pdf",
            "status": "success",
            "message": "Extraction successful",
            "invoice_df": pd.DataFrame({
                "filename": ["demo_invoice_002.pdf"],
                "invoice_number": ["INV-2024-002"],
                "invoice_date": [datetime(2024, 12, 3)],
                "invoice_total_amount": [875.45],
                "vendor_id": [str(vendors[1]["_id"]) if len(vendors) > 1 else str(vendors[0]["_id"])],
                "vendor_name": [vendors[1]["name"] if len(vendors) > 1 else vendors[0]["name"]],
                "text_length": [1342],
                "page_count": [1]
            }),
            "line_items_df": pd.DataFrame({
                "description": ["Prime Ribeye Steak", "Chicken Breast", "Pork Tenderloin"],
                "quantity": [15.0, 20.0, 10.0],
                "unit": ["lb", "lb", "lb"],
                "unit_price": [24.99, 6.99, 8.99],
                "line_total": [374.85, 139.80, 89.90]
            }),
            "extracted_text": "INVOICE\n\nInvoice #: INV-2024-002\nDate: 12/03/2024\nVendor: Quality Meats Co.\n\nPrime Ribeye Steak    15 lb    $24.99    $374.85\nChicken Breast    20 lb    $6.99    $139.80\nPork Tenderloin    10 lb    $8.99    $89.90\n\nTotal Due: $875.45",
            "vendor_id": str(vendors[1]["_id"]) if len(vendors) > 1 else str(vendors[0]["_id"]),
            "vendor_name": vendors[1]["name"] if len(vendors) > 1 else vendors[0]["name"],
            "is_duplicate": False,
            "duplicate_id": None,
            "extraction_failed": False
        },
        {
            "filename": "demo_invoice_003.pdf",
            "status": "partial",
            "message": "Data extraction incomplete - manual review required",
            "invoice_df": pd.DataFrame({
                "filename": ["demo_invoice_003.pdf"],
                "invoice_number": [""],
                "invoice_date": [datetime.now()],
                "invoice_total_amount": [0.0],
                "vendor_id": [str(vendors[2]["_id"]) if len(vendors) > 2 else str(vendors[0]["_id"])],
                "vendor_name": [vendors[2]["name"] if len(vendors) > 2 else vendors[0]["name"]],
                "text_length": [892],
                "page_count": [1]
            }),
            "line_items_df": pd.DataFrame({
                "description": ["Whole Milk", "Cheddar Cheese"],
                "quantity": [12.0, 8.0],
                "unit": ["gal", "lb"],
                "unit_price": [4.49, 7.99],
                "line_total": [53.88, 63.92]
            }),
            "extracted_text": "INVOICE - Dairy Delight\n\nWhole Milk (Gallon)    12 gal    @ $4.49    $53.88\nCheddar Cheese Block    8 lb    @ $7.99    $63.92\n\nPlease remit payment within 30 days.",
            "vendor_id": str(vendors[2]["_id"]) if len(vendors) > 2 else str(vendors[0]["_id"]),
            "vendor_name": vendors[2]["name"] if len(vendors) > 2 else vendors[0]["name"],
            "is_duplicate": False,
            "duplicate_id": None,
            "extraction_failed": True
        },
        {
            "filename": "demo_invoice_004_duplicate.pdf",
            "status": "duplicate",
            "message": "Duplicate found: Invoice #INV-2024-001 already exists",
            "invoice_df": pd.DataFrame({
                "filename": ["demo_invoice_004_duplicate.pdf"],
                "invoice_number": ["INV-2024-001"],
                "invoice_date": [datetime(2024, 12, 1)],
                "invoice_total_amount": [1245.80],
                "vendor_id": [str(vendors[0]["_id"])],
                "vendor_name": [vendors[0]["name"]],
                "text_length": [1523],
                "page_count": [2]
            }),
            "line_items_df": pd.DataFrame({
                "description": ["Fresh Organic Tomatoes", "Premium Lettuce Mix"],
                "quantity": [25.0, 10.0],
                "unit": ["lb", "case"],
                "unit_price": [3.49, 18.99],
                "line_total": [87.25, 189.90]
            }),
            "extracted_text": "INVOICE (DUPLICATE DEMO)\n\nThis is a duplicate invoice for demonstration purposes.",
            "vendor_id": str(vendors[0]["_id"]),
            "vendor_name": vendors[0]["name"],
            "is_duplicate": True,
            "duplicate_id": "demo_duplicate_id",
            "extraction_failed": False
        }
    ]
    
    return demo_invoices


def render_upload_section():
    """Render the file upload section."""
    st.title("📤 Upload Invoices")
    st.markdown("Upload up to 255 invoice files for batch processing")
    
    # Demo mode toggle
    col1, col2 = st.columns([3, 1])
    with col2:
        demo_mode = st.checkbox("📺 Demo Mode", help="Show sample extracted data for demonstration")
    
    if demo_mode:
        st.info("🎬 **Demo Mode Active** - Sample data will be generated instead of actual extraction.")
        
        # Show demo info
        with st.expander("ℹ️ About Demo Mode", expanded=True):
            st.markdown("""
            **Demo mode generates sample extracted data including:**
            - ✅ **Success** (Invoice 1): Fresh Foods Wholesale - Fully extracted with all data
            - ✅ **Success** (Invoice 2): Quality Meats Co. - Complete extraction
            - ⚠️ **Partial** (Invoice 3): Dairy Delight - Incomplete extraction requiring manual review
            - 🔄 **Duplicate** (Invoice 4+): Duplicate invoice detection
            
            Upload any files and demo data will cycle through these patterns.
            """)
    
    # File uploader (always show)
    uploaded_files = st.file_uploader(
        "Choose invoice files",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="file_uploader",
        help="Supported formats: PDF, PNG, JPG, JPEG (Max 255 files)"
    )
    
    if uploaded_files:
        num_files = len(uploaded_files)
        
        if num_files > 255:
            st.error(f"⚠️ You uploaded {num_files} files. Maximum allowed is 255. Please reduce the number of files.")
            return
        
        st.info(f"📁 {num_files} file(s) selected")
        
        # Process button
        if st.button("🚀 Process Invoices", type="primary", use_container_width=True):
            # Check if demo mode is active
            if demo_mode:
                with st.spinner("Generating demo data..."):
                    # Generate demo data based on number of files
                    demo_data = generate_demo_data()
                    
                    # Repeat demo patterns to match file count
                    processed_data = []
                    for idx, uploaded_file in enumerate(uploaded_files):
                        # Cycle through demo data patterns
                        demo_template = demo_data[idx % len(demo_data)].copy()
                        demo_template["filename"] = uploaded_file.name
                        demo_template["invoice_df"]["filename"] = [uploaded_file.name]
                        processed_data.append(demo_template)
                    
                    # Store in session state
                    st.session_state.uploaded_files_data = processed_data
                    st.session_state.processing_complete = True
                    st.session_state.current_step = "review"
                    save_session_to_db()
                    
                    st.success("✅ Demo data generated!")
                    st.rerun()
            else:
                with st.spinner("Processing invoices..."):
                    # Create temporary directory
                    temp_dir = Path("data/temp_uploads")
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    processed_data = []
                    
                    for idx, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {idx + 1}/{num_files}: {uploaded_file.name}")
                        
                        result = process_single_file(uploaded_file, temp_dir)
                        processed_data.append(result)
                        
                        progress_bar.progress((idx + 1) / num_files)
                    
                    # Store in session state
                    st.session_state.uploaded_files_data = processed_data
                    st.session_state.processing_complete = True
                    st.session_state.current_step = "review"
                    
                    # Save to database for persistence
                    save_session_to_db()
                    
                    # Clean up temp files
                    for file in temp_dir.glob("*"):
                        try:
                            file.unlink()
                        except:
                            pass
                    
                    status_text.text("✅ Processing complete!")
                    progress_bar.empty()
                    
                    st.rerun()


def render_invoice_editor(invoice_data: Dict[str, Any], idx: int):
    """Render an editable invoice card."""
    
    invoice_df = invoice_data.get("invoice_df")
    line_items_df = invoice_data.get("line_items_df")
    
    if invoice_df is None or invoice_df.empty:
        st.warning("No invoice data available")
        return
    
    # Status badge
    status = invoice_data.get("status", "unknown")
    status_colors = {
        "success": "🟢",
        "partial": "🟡",
        "duplicate": "🟠",
        "failed": "🔴"
    }
    
    status_icon = status_colors.get(status, "⚪")
    
    with st.expander(
        f"{status_icon} {invoice_data['filename']} - {invoice_data.get('message', '')}",
        expanded=(status in ["partial", "duplicate", "failed"])
    ):
        # Action buttons row
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            if invoice_data.get("is_duplicate"):
                action = st.radio(
                    "Duplicate Action",
                    ["Skip", "Rename & Save", "Overwrite"],
                    key=f"dup_action_{idx}",
                    horizontal=True
                )
                invoice_data["duplicate_action"] = action
        
        with col2:
            if invoice_data.get("extraction_failed"):
                st.warning("⚠️ Extraction incomplete")
        
        with col3:
            if st.button("🔄 Reset", key=f"reset_{idx}"):
                # Reset to original data from session
                st.rerun()
        
        with col4:
            edit_key = f"edit_{idx}"
            if edit_key not in st.session_state.edit_mode:
                st.session_state.edit_mode[edit_key] = False
            
            if st.button(
                "✏️ Edit Mode" if not st.session_state.edit_mode[edit_key] else "👁️ View Mode",
                key=f"edit_toggle_{idx}"
            ):
                st.session_state.edit_mode[edit_key] = not st.session_state.edit_mode[edit_key]
                st.rerun()
        
        # Invoice details section
        st.markdown("### 📄 Invoice Details")
        
        if st.session_state.edit_mode.get(f"edit_{idx}", False):
            # Editable mode
            col1, col2 = st.columns(2)
            
            with col1:
                invoice_df.loc[0, "invoice_number"] = st.text_input(
                    "Invoice Number",
                    value=str(invoice_df.iloc[0]["invoice_number"]),
                    key=f"inv_num_{idx}"
                )
                
                invoice_df.loc[0, "invoice_date"] = st.date_input(
                    "Invoice Date",
                    value=pd.to_datetime(invoice_df.iloc[0]["invoice_date"]),
                    key=f"inv_date_{idx}"
                )
            
            with col2:
                invoice_df.loc[0, "invoice_total_amount"] = st.number_input(
                    "Total Amount",
                    value=float(invoice_df.iloc[0]["invoice_total_amount"]),
                    min_value=0.0,
                    step=0.01,
                    format="%.2f",
                    key=f"inv_total_{idx}"
                )
                
                invoice_df.loc[0, "vendor_name"] = st.text_input(
                    "Vendor Name",
                    value=str(invoice_df.iloc[0].get("vendor_name", invoice_data.get("vendor_name", ""))),
                    key=f"vendor_{idx}"
                )
            
            # Update the invoice_data with edited values
            invoice_data["invoice_df"] = invoice_df
        else:
            # Display mode
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Invoice Number", invoice_df.iloc[0]["invoice_number"])
            with col2:
                st.metric("Date", pd.to_datetime(invoice_df.iloc[0]["invoice_date"]).strftime("%Y-%m-%d"))
            with col3:
                st.metric("Total Amount", f"${invoice_df.iloc[0]['invoice_total_amount']:,.2f}")
            with col4:
                st.metric("Vendor", invoice_data.get("vendor_name", "Unknown"))
        
        # Line items section
        st.markdown("### 📋 Line Items")
        
        if line_items_df is not None and not line_items_df.empty:
            if st.session_state.edit_mode.get(f"edit_{idx}", False):
                # Editable data editor
                edited_df = st.data_editor(
                    line_items_df,
                    num_rows="dynamic",
                    use_container_width=True,
                    key=f"line_items_{idx}",
                    column_config={
                        "description": st.column_config.TextColumn("Description", width="large"),
                        "quantity": st.column_config.NumberColumn("Quantity", format="%.2f"),
                        "unit": st.column_config.TextColumn("Unit"),
                        "unit_price": st.column_config.NumberColumn("Unit Price", format="$%.2f"),
                        "line_total": st.column_config.NumberColumn("Line Total", format="$%.2f")
                    }
                )
                invoice_data["line_items_df"] = edited_df
            else:
                # Display only
                st.dataframe(
                    line_items_df,
                    use_container_width=True,
                    hide_index=True
                )
            
            st.info(f"📦 {len(line_items_df)} line item(s)")
        else:
            st.warning("No line items found. You can add them in edit mode.")
            
            if st.session_state.edit_mode.get(f"edit_{idx}", False):
                if st.button("➕ Add Line Item", key=f"add_line_{idx}"):
                    new_row = pd.DataFrame({
                        "description": [""],
                        "quantity": [0.0],
                        "unit": [""],
                        "unit_price": [0.0],
                        "line_total": [0.0]
                    })
                    if invoice_data["line_items_df"] is None or invoice_data["line_items_df"].empty:
                        invoice_data["line_items_df"] = new_row
                    else:
                        invoice_data["line_items_df"] = pd.concat(
                            [invoice_data["line_items_df"], new_row],
                            ignore_index=True
                        )
                    st.rerun()
        
        # Extracted text (collapsible)
        if invoice_data.get("extracted_text"):
            with st.expander("📄 View Extracted Text"):
                st.text_area(
                    "Raw Text",
                    value=invoice_data["extracted_text"][:2000] + "..." if len(invoice_data["extracted_text"]) > 2000 else invoice_data["extracted_text"],
                    height=200,
                    disabled=True,
                    key=f"text_{idx}"
                )


def render_review_section():
    """Render the review and edit section."""
    st.title("📝 Review & Edit Invoices")
    
    if not st.session_state.uploaded_files_data:
        st.info("No invoices to review. Please upload files first.")
        if st.button("⬅️ Back to Upload"):
            st.session_state.current_step = "upload"
            st.rerun()
        return
    
    # Summary metrics
    total_invoices = len(st.session_state.uploaded_files_data)
    successful = sum(1 for inv in st.session_state.uploaded_files_data if inv["status"] == "success")
    duplicates = sum(1 for inv in st.session_state.uploaded_files_data if inv["is_duplicate"])
    failed = sum(1 for inv in st.session_state.uploaded_files_data if inv["status"] == "failed")
    partial = sum(1 for inv in st.session_state.uploaded_files_data if inv["status"] == "partial")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total", total_invoices)
    col2.metric("✅ Ready", successful)
    col3.metric("⚠️ Partial", partial)
    col4.metric("🔄 Duplicates", duplicates)
    col5.metric("❌ Failed", failed)
    
    st.divider()
    
    # Render each invoice editor
    for idx, invoice_data in enumerate(st.session_state.uploaded_files_data):
        render_invoice_editor(invoice_data, idx)
    
    st.divider()
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("⬅️ Back to Upload", use_container_width=True):
            st.session_state.current_step = "upload"
            st.session_state.uploaded_files_data = []
            st.session_state.processing_complete = False
            delete_temp_upload(st.session_state.session_id)
            st.rerun()
    
    with col2:
        if st.button("💾 Save Draft", use_container_width=True):
            save_session_to_db()
            st.success("✅ Draft saved successfully!")
    
    with col3:
        if st.button("✅ Save All to Database", type="primary", use_container_width=True):
            st.session_state.current_step = "saving"
            st.rerun()


def render_save_section():
    """Render the save to database section."""
    st.title("💾 Saving Invoices to Database")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    invoices_to_save = st.session_state.uploaded_files_data
    total = len(invoices_to_save)
    
    for idx, invoice_data in enumerate(invoices_to_save):
        status_text.text(f"Saving {idx + 1}/{total}: {invoice_data['filename']}")
        
        # Handle duplicates
        if invoice_data.get("is_duplicate"):
            action = invoice_data.get("duplicate_action", "Skip")
            if action == "Skip":
                results.append({
                    "filename": invoice_data["filename"],
                    "status": "skipped",
                    "message": "Skipped (duplicate)"
                })
                progress_bar.progress((idx + 1) / total)
                continue
            elif action == "Rename & Save":
                # Append timestamp to invoice number
                inv_df = invoice_data["invoice_df"]
                if not inv_df.empty:
                    original_num = inv_df.iloc[0]["invoice_number"]
                    inv_df.loc[0, "invoice_number"] = f"{original_num}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    invoice_data["invoice_df"] = inv_df
        
        # Skip failed extractions if user didn't edit them
        if invoice_data.get("status") == "failed":
            results.append({
                "filename": invoice_data["filename"],
                "status": "failed",
                "message": invoice_data.get("message", "Extraction failed")
            })
            progress_bar.progress((idx + 1) / total)
            continue
        
        # Save to database
        try:
            result = save_inv_li_to_db(
                invoice_data["invoice_df"],
                invoice_data["line_items_df"]
            )
            
            results.append({
                "filename": invoice_data["filename"],
                "status": "saved" if result["success"] else "error",
                "message": result["message"]
            })
        except Exception as e:
            results.append({
                "filename": invoice_data["filename"],
                "status": "error",
                "message": f"Error: {str(e)}"
            })
        
        progress_bar.progress((idx + 1) / total)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    st.success("🎉 Save operation complete!")
    
    # Results summary
    saved_count = sum(1 for r in results if r["status"] == "saved")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    error_count = sum(1 for r in results if r["status"] in ["error", "failed"])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Saved", saved_count)
    col2.metric("⏭️ Skipped", skipped_count)
    col3.metric("❌ Errors", error_count)
    
    # Detailed results table
    st.markdown("### 📊 Detailed Results")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Clean up session
    delete_temp_upload(st.session_state.session_id)
    st.session_state.save_complete = True
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📤 Upload More Invoices", use_container_width=True, type="primary"):
            # Reset session
            st.session_state.uploaded_files_data = []
            st.session_state.processing_complete = False
            st.session_state.save_complete = False
            st.session_state.current_step = "upload"
            st.session_state.edit_mode = {}
            st.rerun()
    
    with col2:
        if st.button("👁️ View Saved Invoices", use_container_width=True):
            st.switch_page("pages/View_Invoices.py")


# Main navigation logic
def main():
    current_step = st.session_state.current_step
    
    if current_step == "upload":
        render_upload_section()
    elif current_step == "review":
        render_review_section()
    elif current_step == "saving":
        render_save_section()
    else:
        # Default to upload
        st.session_state.current_step = "upload"
        render_upload_section()


if __name__ == "__main__":
    main()
