import streamlit as st
from pathlib import Path
import shutil
import pandas as pd
from src.storage.db_init import start_connection
from src.storage import database as db
from src.processing.buildsheet_ingestion import extract_buildsheet

# Directories (same conventions as other upload pages)
DATA_DIR = Path("data") / "my_files"
STAGING_DIR = Path("data") / "staging_area"
PROCESSED_DIR = Path("data") / "processed_area"
TEMP_UPLOADS_DIR = Path("data") / "temp_uploads"

st.title("Upload Buildsheet")

# Helper: clear all files and subdirectories inside a directory (safe)
def clear_directory(p: Path):
    try:
        if p.exists():
            for child in p.iterdir():
                try:
                    if child.is_file() or child.is_symlink():
                        child.unlink()
                    elif child.is_dir():
                        shutil.rmtree(child)
                except Exception:
                    continue
        else:
            p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.warning(f"Failed to clear {p}: {e}")

# Initial cleanup on page load to avoid duplicate re-processing
for _d in [DATA_DIR, STAGING_DIR, PROCESSED_DIR, TEMP_UPLOADS_DIR]:
    clear_directory(_d)

# Safe rerun helper
def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        try:
            from streamlit.runtime.scriptrunner.script_runner import RerunException
            raise RerunException()
        except Exception:
            st.stop()

# Allow users to upload files directly (CSV/XLSX)
uploaded = st.file_uploader("Upload buildsheet files (CSV / XLSX)", type=["csv", "xls", "xlsx"], accept_multiple_files=True)

if uploaded:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    staged = []
    for f in uploaded:
        dest = STAGING_DIR / f.name
        if dest.exists():
            dest = STAGING_DIR / f"{dest.stem}_dup{dest.suffix}"
        with open(dest, "wb") as out:
            out.write(f.getbuffer())
        staged.append(dest.name)

    if 'buildsheet_staged_files' not in st.session_state:
        st.session_state['buildsheet_staged_files'] = set()
    st.session_state['buildsheet_staged_files'].update(staged)


# Show staged files
st.markdown("### Staged files")
if 'buildsheet_staged_files' in st.session_state and st.session_state['buildsheet_staged_files']:
    for name in sorted(st.session_state['buildsheet_staged_files']):
        st.write(f"- {name}")
else:
    st.write("No staged files. Upload files to begin.")

# Extraction control — only enabled when there are staged files
staged_exists = 'buildsheet_staged_files' in st.session_state and bool(st.session_state['buildsheet_staged_files'])
if not staged_exists:
    st.info("Upload at least one file to enable extraction.")
else:
    if st.button("Start Extraction"):
        restaurant_id = start_connection(create_dummy=True)
        st.info(f"Using restaurant id: {restaurant_id}")

        all_dfs = []
        failed = []

        for name in list(st.session_state.get('buildsheet_staged_files', [])):
            src = STAGING_DIR / name
            if not src.exists():
                failed.append((name, "file missing"))
                st.warning(f"Missing staged file {name}, skipping")
                st.session_state['buildsheet_staged_files'].discard(name)
                continue

            # Process only CSV/XLS/XLSX
            if src.suffix.lower() not in {'.csv', '.xls', '.xlsx'}:
                st.warning(f"Skipping unsupported file type: {name}")
                dest = PROCESSED_DIR / name
                shutil.move(str(src), dest)
                st.session_state['buildsheet_staged_files'].discard(name)
                continue

            st.write(f"Processing {name}...")
            try:
                df = extract_buildsheet(restaurant_id, src)
                # Make sure schema matches (restaurant_id may already be present)
                if df is None or df.empty:
                    st.warning(f"No buildsheet items extracted from {name}")
                    dest = PROCESSED_DIR / name
                    shutil.move(str(src), dest)
                    st.session_state['buildsheet_staged_files'].discard(name)
                    continue

                df['source_file'] = name
                # Ensure required columns
                for col in ['restaurant_id', 'name', 'yield_quantity', 'estimated_price', 'ingredients']:
                    if col not in df.columns:
                        df[col] = None
                all_dfs.append(df)

                # move to processed area
                dest = PROCESSED_DIR / name
                shutil.move(str(src), dest)
                st.session_state['buildsheet_staged_files'].discard(name)
            except NotImplementedError as e:
                st.warning(f"Skipped {name}: {e}")
                failed.append((name, str(e)))
                dest = PROCESSED_DIR / name
                if src.exists():
                    shutil.move(str(src), dest)
                st.session_state['buildsheet_staged_files'].discard(name)
            except Exception as e:
                st.error(f"Failed to extract {name}: {e}")
                failed.append((name, str(e)))
                dest = PROCESSED_DIR / name
                if src.exists():
                    shutil.move(str(src), dest)
                st.session_state['buildsheet_staged_files'].discard(name)

        if not all_dfs:
            st.warning("No valid buildsheet data extracted from staged files.")
        else:
            combined = pd.concat(all_dfs, ignore_index=True)
            # normalize ingredients to list objects
            def norm_ings(x):
                if isinstance(x, str):
                    try:
                        return pd.read_json(x) if x.strip().startswith('[') else x
                    except Exception:
                        return []
                if x is None or (isinstance(x, float) and pd.isna(x)):
                    return []
                return x

            combined['ingredients'] = combined['ingredients'].apply(lambda x: x if isinstance(x, list) else (x if not isinstance(x, str) else []))

            st.session_state['buildsheet_extraction_df'] = combined
            st.session_state['buildsheet_extracted'] = True
            # store editable ingredients per row
            ing_map = {}
            for idx, row in combined.iterrows():
                ings = row.get('ingredients') or []
                if isinstance(ings, str):
                    # best-effort parse
                    try:
                        import json
                        parsed = json.loads(ings)
                        ings = parsed if isinstance(parsed, list) else []
                    except Exception:
                        ings = []
                ing_map[idx] = ings
            st.session_state['buildsheet_row_ingredients'] = ing_map
            st.success(f"Extracted {len(combined)} buildsheet items from uploaded files.")

# If extraction completed, show preview/editor and Save button
if st.session_state.get('buildsheet_extracted'):
    combined = st.session_state.get('buildsheet_extraction_df')

    # Duplicate detection
    restaurant_id = start_connection(create_dummy=True)
    existing_items = db.get_buildsheet_items(restaurant_id=restaurant_id)
    existing_names = {r['name'] for r in existing_items}

    seen = set()
    dup_flags = []
    dup_reasons = []
    for idx, row in combined.iterrows():
        name = row['name']
        reason = ''
        if name in existing_names:
            dup_flags.append(True)
            reason = 'name already exists in DB'
        elif name in seen:
            dup_flags.append(True)
            reason = 'duplicate in uploaded data'
        else:
            dup_flags.append(False)
            reason = ''
            seen.add(name)
        dup_reasons.append(reason)

    combined['duplicate'] = dup_flags
    combined['dup_reason'] = dup_reasons
    combined['overwrite'] = combined['duplicate'].astype(bool)

    st.markdown("### Preview extracted buildsheet items")
    editable_cols = ['name', 'yield_quantity', 'estimated_price', 'duplicate', 'overwrite', 'source_file', 'dup_reason']

    try:
        edited = st.experimental_data_editor(combined[editable_cols], num_rows='dynamic')
    except Exception:
        st.write("Your Streamlit version may not support data editor; showing read-only preview.")
        st.dataframe(combined[editable_cols])
        edited = combined[editable_cols]

    # Render per-row ingredients editor / material breakdown
    st.markdown("---")
    st.markdown("### Material breakdown (per item)")
    for idx, row in edited.reset_index().iterrows():
        orig_idx = row['index']
        st.markdown(f"**Item {orig_idx}: {row.get('name', '')}**")
        st.write(f"Duplicate: {row.get('duplicate')} — {row.get('dup_reason', '')}")
        # Get current ingredients from session mapping
        ing_list = st.session_state['buildsheet_row_ingredients'].get(orig_idx, [])
        # Convert to DataFrame for editing
        ing_df = pd.DataFrame(ing_list)
        # Ensure columns exist
        for c in ['material_name', 'measure_unit', 'measure_quantity']:
            if c not in ing_df.columns:
                ing_df[c] = None
        try:
            edited_ing = st.experimental_data_editor(ing_df[['material_name', 'measure_unit', 'measure_quantity']], key=f"ings_{orig_idx}", num_rows='dynamic')
        except Exception:
            st.write(ing_df[['material_name', 'measure_unit', 'measure_quantity']])
            edited_ing = ing_df[['material_name', 'measure_unit', 'measure_quantity']]

        # Normalize edited ingredients back to list of dicts
        new_ings = []
        for _, r in edited_ing.iterrows():
            if (pd.isna(r.get('material_name')) or not r.get('material_name')):
                continue
            new_ings.append({
                'material_name': r.get('material_name'),
                'measure_unit': r.get('measure_unit') if r.get('measure_unit') else None,
                'measure_quantity': float(r.get('measure_quantity')) if r.get('measure_quantity') not in (None, '', float('nan')) else None
            })
        st.session_state['buildsheet_row_ingredients'][orig_idx] = new_ings
        st.markdown("---")

    # Save operation
    if st.button("Save to Database"):
        to_insert_rows = []
        # Build list of rows to insert (respect overwrite choices)
        for _, r in edited.reset_index().iterrows():
            orig_idx = r['index']
            name = r.get('name')
            if not name or pd.isna(name):
                st.error("Item name cannot be empty")
                st.stop()
            try:
                yq = None
                if r.get('yield_quantity') not in (None, '', float('nan')):
                    yq = float(r.get('yield_quantity'))
            except Exception:
                st.error(f"Invalid yield_quantity for item {name}: {r.get('yield_quantity')}")
                st.stop()

            ep = r.get('estimated_price')
            try:
                ep_val = float(ep) if ep not in (None, '', float('nan')) else None
            except Exception:
                st.error(f"Invalid estimated_price for item {name}: {ep}")
                st.stop()

            dup = bool(r.get('duplicate'))
            overwrite = bool(r.get('overwrite'))

            # Assemble record to insert
            rec = {
                'restaurant_id': str(restaurant_id),
                'name': name,
                'yield_quantity': yq,
                'estimated_price': ep_val,
                'ingredients': st.session_state['buildsheet_row_ingredients'].get(orig_idx, [])
            }

            if dup:
                if overwrite:
                    # delete existing items with same name
                    matches = db.get_buildsheet_items(restaurant_id=restaurant_id)
                    for m in matches:
                        if m.get('name') == name and m.get('_id'):
                            db.delete_buildsheet_item(m.get('_id'))
                    to_insert_rows.append(rec)
                else:
                    # skip
                    continue
            else:
                to_insert_rows.append(rec)

        if to_insert_rows:
            insert_df = pd.DataFrame(to_insert_rows)
            res = db.save_buildsheet_db(insert_df)
            inserted = res.get('inserted', 0) if isinstance(res, dict) else 0
        else:
            inserted = 0

        st.success(f"Inserted/updated {inserted} records.")

        # Clean upload-related folders after successful save
        for _d in [DATA_DIR, STAGING_DIR, PROCESSED_DIR, TEMP_UPLOADS_DIR]:
            clear_directory(_d)

        # Reset extraction / staging state
        st.session_state.pop('buildsheet_extraction_df', None)
        st.session_state.pop('buildsheet_extracted', None)
        st.session_state.pop('buildsheet_staged_files', None)
        st.session_state.pop('buildsheet_row_ingredients', None)

        st.success("Insert complete. Upload folders cleared.")
        try:
            safe_rerun()
        except Exception:
            st.stop()
