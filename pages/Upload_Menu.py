import streamlit as st
from pathlib import Path
import shutil
import pandas as pd
from src.storage.db_init import start_connection
from src.storage import database as db
from src.processing.menu_ingestion import extract_menu

# Directories
DATA_DIR = Path("data") / "my_files"
STAGING_DIR = Path("data") / "staging_area"
PROCESSED_DIR = Path("data") / "processed_area"
TEMP_UPLOADS_DIR = Path("data") / "temp_uploads"

st.title("Upload Menus")

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
                    # ignore per-item failures
                    continue
        else:
            p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.warning(f"Failed to clear {p}: {e}")

# Clean upload-related folders on page start to avoid stale state
for _d in [DATA_DIR, STAGING_DIR, PROCESSED_DIR, TEMP_UPLOADS_DIR]:
    clear_directory(_d)

# Safe rerun helper to be compatible with different Streamlit versions
def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        try:
            from streamlit.runtime.scriptrunner.script_runner import RerunException
            raise RerunException()
        except Exception:
            # Fallback: stop the script (user can refresh)
            st.stop()

uploaded = st.file_uploader("Upload menu files (PDF / images)", type=["pdf", "png", "jpg", "jpeg", "tiff"], accept_multiple_files=True)

# If user uploaded files, save them to staging only (do NOT extract yet)
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

    # keep a set of staged filenames in session state so users can upload multiple times before extracting
    if 'menu_staged_files' not in st.session_state:
        st.session_state['menu_staged_files'] = set()
    st.session_state['menu_staged_files'].update(staged)

# Show staged files
st.markdown("### Staged files")
if 'menu_staged_files' in st.session_state and st.session_state['menu_staged_files']:
    for name in sorted(st.session_state['menu_staged_files']):
        st.write(f"- {name}")
else:
    st.write("No staged files. Upload files to begin.")

# Extraction control
if st.button("Start Extraction"):
    # Ensure DB and get temp restaurant
    restaurant_id = start_connection(create_dummy=True)
    st.info(f"Using restaurant id: {restaurant_id}")

    all_dfs = []
    failed = []

    for name in list(st.session_state.get('menu_staged_files', [])):
        src = STAGING_DIR / name
        if not src.exists():
            failed.append((name, "file missing"))
            st.warning(f"Missing staged file {name}, skipping")
            st.session_state['menu_staged_files'].discard(name)
            continue

        st.write(f"Processing {name}...")
        try:
            df = extract_menu(restaurant_id, src)
            df["source_file"] = name
            all_dfs.append(df)
            # move to processed area
            dest = PROCESSED_DIR / name
            shutil.move(str(src), dest)
            st.session_state['menu_staged_files'].discard(name)
        except NotImplementedError as e:
            st.warning(f"Skipped {name}: {e}")
            failed.append((name, str(e)))
            dest = PROCESSED_DIR / name
            if src.exists():
                shutil.move(str(src), dest)
            st.session_state['menu_staged_files'].discard(name)
        except Exception as e:
            st.error(f"Failed to extract {name}: {e}")
            failed.append((name, str(e)))
            dest = PROCESSED_DIR / name
            if src.exists():
                shutil.move(str(src), dest)
            st.session_state['menu_staged_files'].discard(name)

    if not all_dfs:
        st.warning("No valid menu data extracted from staged files.")
    else:
        combined = pd.concat(all_dfs, ignore_index=True)
        st.session_state['menu_extraction_df'] = combined
        st.session_state['menu_extracted'] = True
        st.success(f"Extracted {len(combined)} rows from uploaded files.")

# If extraction completed, show preview/editor and Save button
if st.session_state.get('menu_extracted'):
    combined = st.session_state.get('menu_extraction_df')

    # Duplicate detection
    restaurant_id = start_connection(create_dummy=True)
    existing = {r['menu_item'] for r in db.find_menu_items(restaurant_id=restaurant_id)}

    seen = set()
    dup_flags = []
    for idx, row in combined.iterrows():
        name = row['menu_item']
        if name in existing:
            dup_flags.append(True)
        elif name in seen:
            dup_flags.append(True)
        else:
            dup_flags.append(False)
            seen.add(name)

    combined['duplicate'] = dup_flags
    combined['overwrite'] = combined['duplicate'].astype(bool)

    st.markdown("### Preview extracted menu items")
    editable_cols = ['menu_item', 'price', 'category', 'duplicate', 'overwrite', 'source_file']
    try:
        edited = st.experimental_data_editor(combined[editable_cols], num_rows='dynamic')
    except Exception:
        st.write("Your Streamlit version may not support data editor; showing read-only preview.")
        st.dataframe(combined[editable_cols])
        edited = combined[editable_cols]

    if st.button("Save to Database"):
        # Validate edited rows
        to_insert = []
        for _, r in edited.iterrows():
            if not r['menu_item'] or pd.isna(r['menu_item']):
                st.error("menu_item cannot be empty")
                st.stop()
            try:
                price = float(r['price'])
            except Exception:
                st.error(f"Invalid price for item {r['menu_item']}: {r['price']}")
                st.stop()
            to_insert.append(r)

        inserted = 0
        for r in to_insert:
            rid = restaurant_id
            name = r['menu_item']
            price = float(r['price'])
            category = r.get('category') if r.get('category') else 'Uncategorized'
            dup = bool(r.get('duplicate'))
            overwrite = bool(r.get('overwrite'))

            if dup:
                if overwrite:
                    db.delete_menu_items(rid, name)
                    db.insert_menu_item(rid, name, price, category)
                    inserted += 1
                else:
                    continue
            else:
                matches = db.find_menu_items(restaurant_id=rid, menu_item=name)
                exact_found = False
                for m in matches:
                    try:
                        if float(str(m.get('price', 0))) == float(price):
                            exact_found = True
                            break
                    except Exception:
                        continue

                if not exact_found:
                    db.insert_menu_item(rid, name, price, category)
                    inserted += 1

        st.success(f"Inserted/updated {inserted} records.")

        # Clean all upload-related folders after successful save
        for _d in [DATA_DIR, STAGING_DIR, PROCESSED_DIR, TEMP_UPLOADS_DIR]:
            clear_directory(_d)

        # Reset extraction / staging state
        st.session_state.pop('menu_extraction_df', None)
        st.session_state.pop('menu_extracted', None)
        st.session_state.pop('menu_staged_files', None)

        st.success("Inserted/updated {0} records. Upload folders cleared.".format(inserted))

        # Refresh to initial upload view
        try:
            safe_rerun()
        except Exception:
            # If rerun fails, just stop so user can manually refresh
            st.stop()
