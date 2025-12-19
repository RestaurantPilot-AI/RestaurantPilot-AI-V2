import streamlit as st
import pandas as pd
from src.storage import database as db

st.title("View & Edit Menu")

# Load all menu items
records = db.find_menu_items()
if not records:
    st.info("No menu items in database.")
    st.stop()

df = pd.DataFrame(records)
# Ensure columns
for c in ["_id", "restaurant_id", "menu_item", "price", "category"]:
    if c not in df.columns:
        df[c] = ""

# Restaurant filter
rests = db.get_all_restaurants()
rest_options = {r["_id"]: r.get("name", r.get("location_name", r["_id"])) for r in rests}
rest_keys = ["All"] + [f"{k} | {v}" for k, v in rest_options.items()]
sel = st.selectbox("Restaurant", rest_keys)

filtered = df.copy()
if sel != "All":
    rid = sel.split(" | ")[0]
    filtered = filtered[filtered["restaurant_id"] == rid]

# Category filter
cats = db.get_all_menu_category_names()
if not cats:
    cats = ["Uncategorized"]
options = ["All"] + sorted(cats)
cat_sel = st.multiselect("Category", options=options, default=["All"])
if cat_sel and "All" not in cat_sel:
    filtered = filtered[filtered["category"].isin(cat_sel)]

# Price filters with an option to ignore the price range
col_min, col_max, col_ignore = st.columns([1, 1, 0.7])
with col_min:
    # sensible defaults from current filtered data
    # min_default = float(filtered["price"].min()) if not filtered.empty else 0.0
    min_default = 0.5
    min_price = st.number_input("Min price", value=min_default)
with col_max:
    # max_default = float(filtered["price"].max()) if not filtered.empty else 1000.0
    max_default = 125
    max_price = st.number_input("Max price", value=max_default)
with col_ignore:
    ignore_price = st.checkbox("Ignore price range", value=False, help="When checked, price filtering will be ignored")

if not ignore_price:
    filtered = filtered[(filtered["price"].astype(float) >= float(min_price)) & (filtered["price"].astype(float) <= float(max_price))]

st.markdown("### Editable menu table")
# allow editing of menu_item, price, category
display_cols = ["_id", "restaurant_id", "menu_item", "price", "category"]
try:
    edited = st.experimental_data_editor(filtered[display_cols], num_rows='dynamic')
    editor_supported = True
except Exception:
    # st.write("Your Streamlit version may not support data editor; showing selection-based editor.")
    st.dataframe(filtered[display_cols])
    editor_supported = False

    # Build selection options (label -> _id mapping)
    options = []
    id_map = {}
    for _, row in filtered.iterrows():
        label = f"{row['_id']} | {row['menu_item']} | ${row['price']}"
        options.append(label)
        id_map[label] = str(row['_id'])

    selected_labels = st.multiselect("Select rows to update", options=options)

    if selected_labels:
        # Category options
        cats = db.get_all_menu_category_names()
        if not cats:
            cats = ["Uncategorized"]
        cat_options = ["Uncategorized"] + sorted(cats)

        edits = []
        for label in selected_labels:
            _id = id_map[label]
            row = filtered[filtered["_id"] == _id].iloc[0]

            st.markdown(f"#### Edit {_id}")
            col1, col2, col3 = st.columns([3, 1, 2])
            with col1:
                new_name = st.text_input("Menu item", value=row["menu_item"], key=f"mitem_{_id}")
            with col2:
                try:
                    price_val = float(row["price"])
                except Exception:
                    price_val = 0.0
                new_price = st.number_input("Price", value=price_val, key=f"price_{_id}", format="%.2f")
            with col3:
                try:
                    idx = cat_options.index(row.get("category") if row.get("category") in cat_options else "Uncategorized")
                except Exception:
                    idx = 0
                new_cat = st.selectbox("Category", options=cat_options, index=idx, key=f"cat_{_id}")

            edits.append({"_id": _id, "menu_item": new_name, "price": new_price, "category": new_cat})


# Unified Preview & Apply (single button for both editor and fallback)
if st.button("Preview changes", key="preview_changes_unified"):
    diffs = []
    preview_rows = []

    if editor_supported:
        # Build index lookup from displayed filtered rows
        original = filtered.set_index("_id")
        for _, row in edited.iterrows():
            _id = str(row["_id"])
            if _id not in original.index:
                continue
            orig_row = original.loc[_id]
            changes = {}
            # Only allow changes to these three fields
            for field in ["menu_item", "price", "category"]:
                new_val = row[field]
                old_val = orig_row[field]
                # Normalize price comparison as float
                if field == "price":
                    try:
                        new_val_f = float(new_val)
                        old_val_f = float(old_val)
                    except Exception:
                        st.error(f"Invalid price value for {_id}: {new_val}")
                        new_val_f = new_val
                        old_val_f = old_val
                    if new_val_f != old_val_f:
                        changes[field] = {"old": old_val_f, "new": new_val_f}
                else:
                    if str(new_val).strip() != str(old_val).strip():
                        changes[field] = {"old": old_val, "new": new_val}

            if changes:
                diffs.append({"_id": _id, "menu_item": row["menu_item"], "changes": changes})
                for k, v in changes.items():
                    preview_rows.append({"_id": _id, "field": k, "old": str(v["old"]), "new": str(v["new"])})
    else:
        if edits:
            for e in edits:
                orig = filtered[filtered["_id"] == e["_id"]].iloc[0]
                changes = {}
                if str(e["menu_item"]).strip() != str(orig["menu_item"]).strip():
                    changes["menu_item"] = {"old": orig["menu_item"], "new": e["menu_item"]}
                try:
                    orig_price = float(orig["price"])
                except Exception:
                    orig_price = 0.0
                if float(e["price"]) != orig_price:
                    changes["price"] = {"old": orig_price, "new": float(e["price"])}
                if str(e["category"]).strip() != str(orig["category"]).strip():
                    changes["category"] = {"old": orig["category"], "new": e["category"]}

                if changes:
                    diffs.append({"_id": e["_id"], "changes": changes})
                    for k, v in changes.items():
                        preview_rows.append({"_id": e["_id"], "field": k, "old": str(v["old"]), "new": str(v["new"])})

    if not diffs:
        st.info("No changes detected.")
    else:
        st.markdown("### Pending updates")
        preview_df = pd.DataFrame(preview_rows)
        st.dataframe(preview_df)

        ids_to_choose = sorted({d["_id"] for d in diffs})
        selected_ids = st.multiselect("Select rows (by _id) to apply updates", options=ids_to_choose, default=ids_to_choose)

        if st.button("Confirm and apply selected updates", key="confirm_apply_unified"):
            applied = 0
            for d in diffs:
                if d["_id"] not in selected_ids:
                    continue
                update_payload = {}
                for field, change in d["changes"].items():
                    if field == "price":
                        update_payload[field] = str(float(change["new"]))
                    else:
                        update_payload[field] = str(change["new"]).strip() if change["new"] is not None else ""
                try:
                    res = db.update_menu_item_by_id(d["_id"], update_payload)
                    if res:
                        applied += 1
                except Exception as e:
                    st.error(f"Failed to update {d['_id']}: {e}")
            st.success(f"Applied {applied} updates.")
            try:
                st.experimental_rerun()
            except Exception:
                try:
                    from streamlit.runtime.scriptrunner.script_runner import RerunException
                    raise RerunException()
                except Exception:
                    st.stop()
